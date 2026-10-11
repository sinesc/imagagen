"""Phase 1 reference captioning: pure-CPU component tests.

Shape/determinism checks on a tiny vision tower, the caption token
stream assembly, the MRoPE position ids, the sha1 caption cache
round-trip, and the prompt application. (The full-size GPU path is
covered by dev/smoke_reference_caption.py.)
"""

import numpy as np
import torch

from imagagen.encoders.qwen3vl_vision import (
    IMAGE_PAD_ID,
    Qwen3VLVisionTower,
    process_qwen2vl_images,
    qwen2vl_mrope_position_ids,
)
from imagagen.reference import (
    _build_stream,
    _cache_load,
    _cache_save,
    apply_reference,
    image_sha1,
    parse_caption_sections,
    selected_caption_titles,
)

TINY_CFG = dict(
    hidden_size=32, intermediate_size=64, depth=2,
    deepstack_visual_indexes=[0, 1], num_heads=4, patch_size=16,
    temporal_patch_size=2, in_channels=3, spatial_merge_size=2,
    num_position_embeddings=2304, out_hidden_size=40,
)


def test_preprocess_grid_and_patch_shape():
    img = torch.rand(1, 64, 96, 3)
    patches, grid = process_qwen2vl_images(img)
    assert grid.tolist() == [[1, 4, 6]]  # 64/16 x 96/16
    # 24 patches x (3ch * 2 temporal * 16 * 16)
    assert patches.shape == (24, 3 * 2 * 16 * 16)


def test_preprocess_clamps_to_min_pixels():
    img = torch.rand(1, 32, 32, 3)  # 1024 px < min_pixels 3136
    _, grid = process_qwen2vl_images(img)
    assert grid[0, 1].item() * grid[0, 2].item() * 16 * 16 >= 3136


def test_vision_tower_shapes_and_determinism():
    torch.manual_seed(0)
    tower = Qwen3VLVisionTower(TINY_CFG).eval()
    img = torch.rand(1, 64, 96, 3)
    patches, grid = process_qwen2vl_images(img)
    with torch.no_grad():
        m1, ds1 = tower(patches, grid)
        m2, ds2 = tower(patches, grid)
    # 24 patches -> 6 merged rows (2x2 merge), out_hidden 40
    assert m1.shape == (6, 40)
    assert [d.shape for d in ds1] == [(6, 40), (6, 40)]
    assert torch.equal(m1, m2)  # deterministic
    assert torch.isfinite(m1).all()


def test_build_stream_expands_pad_and_finds_image_span():
    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            # any deterministic tokenization; the stream logic only needs
            # the text parts tokenized (vocab.json-based here)
            from imagagen.encoders.krea2 import _USER  # noqa: F401
            return {"input_ids": [[1, 2] if "user" in text else [3]]}

    ids, img_index = _build_stream(_Tok(), 5)
    n_pad = ids.count(IMAGE_PAD_ID)
    assert n_pad == 5
    # vision_start directly before the pad block, vision_end after it
    assert ids[img_index + 1] == IMAGE_PAD_ID
    assert ids[img_index + 1 + 5] == 151653  # vision_end
    # im_start / im_end markers present
    assert 151644 in ids and 151645 in ids


def test_mrope_position_ids_layout():
    grid = torch.tensor([[1, 4, 6]])
    index, size, seq = 3, 6 + 2, 30
    pos = qwen2vl_mrope_position_ids(index, size, grid, seq, "cpu")
    assert pos.shape == (3, seq)
    # text before the image runs sequentially
    assert pos[0, :index].tolist() == [0, 1, 2]
    # the T row is constant over the image span
    assert pos[0, index:index + size].tolist() == [index] * size
    # text after the image continues at start_next = len_max + index
    len_max = int(grid.max()) // 2
    assert pos[0, index + size] == len_max + index


def test_caption_cache_round_trip(tmp_path):
    path = tmp_path / "reference_captions.json"
    assert _cache_load(path) == {}
    img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    key = image_sha1(img)
    assert key == image_sha1(img.copy())
    assert key != image_sha1(img + 1)
    _cache_save(path, {key: "a cat"})
    assert _cache_load(path) == {key: "a cat"}


def test_apply_reference():
    out = apply_reference("a painting", "a red fox in snow")
    assert out == "a painting Referencing an image: a red fox in snow."
    # an empty caption leaves the prompt untouched
    assert apply_reference("a painting", "") == "a painting"


MOCK_STRUCTURED_CAPTION = (
    "This is a mock reference caption.\n\n"
    "**Subject:**\nA mock subject paragraph.\n\n"
    "In summary, a mock summary."
)


def test_parse_caption_sections_structure():
    secs = parse_caption_sections(MOCK_STRUCTURED_CAPTION)
    assert [t for t, _ in secs] == ["Overview", "Subject", "Summary"]
    assert secs[1][1] == "A mock subject paragraph."
    assert secs[2][1] == "In summary, a mock summary."


def test_parse_caption_sections_edge_cases():
    # no headings -> a single Overview section
    assert parse_caption_sections("just text") == [("Overview", "just text")]
    # inline heading body + multiple unlabeled trailing paragraphs
    cap = "intro\n\n**A:** body one\n\nmore a\n\n**B:**\nb b\n\ntail one\n\ntail two"
    secs = parse_caption_sections(cap)
    assert [t for t, _ in secs] == ["Overview", "A", "B", "Summary"]
    assert secs[1][1] == "body one\n\nmore a"
    assert secs[3][1] == "tail one\n\ntail two"
    # duplicate titles are deduped; empty sections dropped
    secs = parse_caption_sections("**A:**\nx\n\n**A:**\n\n**A:**\nz")
    assert secs == [("A", "x"), ("A (3)", "z")]


def test_apply_reference_section_filter():
    cap = MOCK_STRUCTURED_CAPTION
    # None = all sections, in caption order
    out = apply_reference("p", cap)
    assert out == ("p Referencing an image: This is a mock reference "
                   "caption.\n\nA mock subject paragraph.\n\n"
                   "In summary, a mock summary.")
    # subset: only the selected section text
    out = apply_reference("p", cap, ["Subject"])
    assert out == "p Referencing an image: A mock subject paragraph."
    # empty selection leaves the prompt untouched
    assert apply_reference("p", cap, []) == "p"
    # selected_caption_titles mirrors the applied selection
    assert selected_caption_titles(cap, None) == [
        "Overview", "Subject", "Summary"]
    assert selected_caption_titles(cap, ["Subject"]) == ["Subject"]
    assert selected_caption_titles(cap, []) == []
