"""Reference-image captioning: the universal VLM tier (phase 1).

The reference image is described with the Qwen3-VL-4B weights the Krea 2
text encoder already ships (vision tower + text tower, tied lm_head):
the caption is appended to the POSITIVE prompt of every family, so one
implementation serves SDXL, Flux, Z-Image and Krea 2 alike.

Determinism + caching: greedy (temperature-free) decoding with a fixed
template, and the caption is cached by sha1 of the image bytes at
``<root>/cache/reference_captions.json`` -- the same image always yields
the same caption, generations stay reproducible, and repeats of a
reference cost nothing. The caption is recorded in the PNG metadata.

VRAM: the captioner towers (~4.8 GB text + ~0.45 GB vision) are built
once on the CPU and cross to the GPU only for the caption; the calling
backend parks its transformer around the call (the encode-swap pattern,
see the backends' ``_encode_prompts``). A Krea 2 backend passes its
RESIDENT text tower + tokenizer, so only the vision tower is loaded
(lazily) and it stays GPU-resident afterwards.

Runs on the generation worker thread (the only thread that touches torch
models); a module lock serializes the shared towers across GPUs, and the
per-GPU registry lock keeps captioning from racing a generation.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading

import numpy as np
import torch

from .backends import PHASE_REFERENCE, PHASE_REFERENCE_DONE, PHASE_REFERENCE_LOADING, report_phase
from .encoders.caption_prompt import CAPTION_PARTS
from .encoders.krea2 import (
    _END_OF_TEXT,
    _IM_START,
    IMAGE_PAD_ID,
    load_krea2_encoder,
)
from .encoders.qwen3vl_vision import (
    MARKER_IDS,
    load_vision_tower,
    process_qwen2vl_images,
    qwen2vl_mrope_position_ids,
)

#: greedy decoding cap (captions are short; the cap bounds the worst case)
MAX_CAPTION_TOKENS = 1024

#: the prompt sentence a caption is folded into
_CAPTION_PREFIX = "Referencing an image: "

#: a ``**Title:**`` section heading (optional inline body text)
_SECTION_HEADING = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$", re.S)


def parse_caption_sections(caption: str) -> list[tuple[str, str]]:
    """``[(title, text)]`` in caption order. Captions are structured as
    an unlabeled preamble, ``**Title:**`` sections and a trailing
    "In summary, ..." paragraph: unlabeled paragraphs before the first
    heading become section ``Overview``, unlabeled ones after the last
    heading become ``Summary``; duplicate titles get `` (2)``-style
    suffixes; empty sections are dropped. The UI mirrors this parser
    (static/app.js parseCaptionSections) -- keep both in sync."""
    paras = [p.strip() for p in caption.split("\n\n") if p.strip()]
    sections: list[tuple[str | None, list[str]]] = []
    for p in paras:
        m = _SECTION_HEADING.match(p)
        if m:
            body = [m.group(2).strip()] if m.group(2).strip() else []
            sections.append((m.group(1).strip(), body))
        elif sections:
            sections[-1][1].append(p)
        else:
            sections.append((None, [p]))
    # trailing unlabeled paragraphs after the LAST heading are the
    # summary: captions use one paragraph per section, so everything
    # past the first paragraph of the last section is the trailing
    # "In summary, ..." block
    if sections and sections[-1][0] is not None \
            and len(sections[-1][1]) > 1:
        title, body = sections[-1]
        sections[-1] = (title, body[:1])
        sections.append((None, body[1:]))
    out: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for i, (title, body) in enumerate(sections):
        if title is None:
            title = "Overview" if i == 0 else "Summary"
        n = seen.get(title, 0) + 1
        seen[title] = n
        if n > 1:
            title = f"{title} ({n})"
        text = "\n\n".join(body).strip()
        if text:
            out.append((title, text))
    return out

_LOCK = threading.RLock()
_towers: _CaptionTowers | None = None


def reference_cache_path(config):
    """The caption cache file (same location convention as the other
    caches: ``<root>/cache/``)."""
    return config.root / "cache" / "reference_captions.json"


def _cache_load(path) -> dict:
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _cache_save(path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1) + "\n",
                   "utf-8")
    tmp.replace(path)


def image_sha1(image: np.ndarray) -> str:
    """Cache key of an HxWx3 uint8 image: sha1 of its raw bytes."""
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


class _CaptionTowers:
    """The captioner's modules, built once on the CPU.

    ``tower`` is a private text tower (with the final norm loaded, for
    the tied-lm_head logits); ``visual`` is the vision host (kept
    GPU-resident once loaded -- 0.45 GB is cheap to park).
    """

    def __init__(self, config):
        infra = config.infra_dir / "qwen_3vl_4b.safetensors"
        self.tower = load_krea2_encoder(str(infra), "cpu", load_norm=True)
        # the vision host (parameter names match the checkpoint's
        # model.visual.* keys; load_vision_tower fills them)
        self.visual = load_vision_tower(str(infra), "cpu")
        from transformers import Qwen2Tokenizer

        self.tok = Qwen2Tokenizer.from_pretrained(
            str(config.res_dir / "tokenizer_krea2"))


def _get_towers(config) -> _CaptionTowers:
    global _towers
    with _LOCK:
        if _towers is None:
            _towers = _CaptionTowers(config)
        return _towers


def _build_stream(tok, n_pad: int):
    """Assemble the caption token stream from :data:`CAPTION_PARTS`:
    marker ids inserted directly (the tokenizer dir has special=None --
    only im_start/im_end split from text), text parts BPE-tokenized
    verbatim, and the single pad marker expanded to ``n_pad`` image-pad
    tokens (the merged vision rows). Returns ``(ids, img_index)`` where
    img_index is the vision_start position (the image span is
    vision_start .. vision_end)."""
    ids: list[int] = []
    img_index = 0
    for kind, text in CAPTION_PARTS:
        if kind == "pad":
            ids.extend([IMAGE_PAD_ID] * n_pad)
        elif kind == "text":
            seg = tok(text, add_special_tokens=False)["input_ids"]
            if seg and isinstance(seg[0], list):
                seg = seg[0]
            ids.extend(seg)
        else:
            if kind == "vision_start":
                img_index = len(ids)
            ids.append(MARKER_IDS[kind])
    return ids, img_index


def _generate_caption(tower, tok, visual, image: np.ndarray,
                      max_tokens: int = MAX_CAPTION_TOKENS) -> str:
    """Greedy caption of one HxWx3 uint8 image. ``tower`` (text) and
    ``visual`` must be on the same CUDA device."""
    device = next(tower.parameters()).device
    arr = torch.from_numpy(
        np.ascontiguousarray(image).astype(np.float32) / 255.0)[None]
    patches, grid = process_qwen2vl_images(arr.to(device))
    # the vision forward MUST run under no_grad/inference_mode: at ~5k
    # patches the autograd graph across the 24 vision blocks costs ~10 GiB
    # (measured) and OOMs a 16 GB card that is otherwise empty.
    with torch.no_grad():
        merged, deepstack = visual.visual(patches.to(device), grid.to(device))
    n_pad = merged.shape[0]
    stream, img_index = _build_stream(tok, n_pad)
    pos = qwen2vl_mrope_position_ids(
        img_index, n_pad + 2, grid, len(stream), device)
    ids = list(stream)
    gen: list[int] = []
    from imagagen.encoders.krea2 import decode_weight_cache
    # the dequantized decode weight cache needs ~6.5 GB extra VRAM
    free_gib = torch.cuda.mem_get_info(device)[0] / 2**30
    with torch.inference_mode(), decode_weight_cache(
            tower, enabled=free_gib > 8.5):
        # ONE prefill over the full stream, then cached single-token
        # decodes (re-running the prefill per token costs minutes).
        logits, cache = tower.forward_caption(
            torch.tensor([ids], device=device), merged, deepstack, pos)
        cur_pos = int(pos[0, -1])
        for _ in range(max_tokens):
            nxt = int(logits[0].argmax())
            if nxt in (_END_OF_TEXT, _IM_START):
                break
            gen.append(nxt)
            cur_pos += 1
            logits, cache = tower.forward_caption(
                torch.tensor([[nxt]], device=device),
                position_ids=torch.tensor([[cur_pos]] * 3, device=device),
                kv_cache=cache)
    return tok.decode(gen, skip_special_tokens=True).strip()


def caption_reference(
    image: np.ndarray,
    device,
    config,
    park=None,
    restore=None,
    text_tower=None,
    tokenizer=None,
    progress=None,
) -> str:
    """Describe ``image`` (HxWx3 uint8) for reference conditioning.

    The caption is sha1-cached (deterministic greedy decoding), so a
    repeat reference is free. ``park`` / ``restore`` are the backend's
    transformer-swap callbacks (the towers cross to ``device`` for the
    caption; the transformer does not fit next to them on a 16 GB
    card). ``text_tower`` / ``tokenizer`` let a Krea 2 backend reuse its
    RESIDENT text tower (then only the vision tower moves, and it stays
    resident afterwards); without them the captioner's own towers are
    swapped in and out.
    """
    key = image_sha1(image)
    cache_path = reference_cache_path(config)
    cache = _cache_load(cache_path)
    if key in cache:
        return cache[key]
    report_phase(progress, PHASE_REFERENCE)
    with _LOCK:
        if _towers is None:
            # first caption of the process: the tower load takes ~20 s
            report_phase(progress, PHASE_REFERENCE_LOADING)
            report_phase(progress, PHASE_REFERENCE)
        st = _get_towers(config)
        if text_tower is not None:
            tower, tok = text_tower, tokenizer
            tower_was_cpu = next(tower.parameters()).device.type != "cuda"
            if tower_was_cpu:
                tower.to(device)
            if next(st.visual.parameters()).device.type != "cuda":
                st.visual.to(device)
            try:
                caption = _generate_caption(tower, tok, st.visual, image)
            finally:
                if tower_was_cpu:
                    tower.to("cpu")
                    torch.cuda.empty_cache()
        else:
            if park is not None:
                park()
            st.tower.to(device)
            st.visual.to(device)
            try:
                caption = _generate_caption(st.tower, st.tok, st.visual,
                                            image)
            finally:
                st.tower.to("cpu")
                st.visual.to("cpu")
                torch.cuda.empty_cache()
                if restore is not None:
                    restore()
    cache[key] = caption
    report_phase(progress, PHASE_REFERENCE_DONE)
    _cache_save(cache_path, cache)
    return caption


def apply_reference(prompt: str, caption: str,
                    sections: list[str] | None = None) -> str:
    """Fold the caption into the positive prompt (the negative prompt is
    untouched). Applied after styles/wildcards, right before encoding.
    ``sections`` selects which caption sections to apply (None = all;
    [] = none; see :func:`parse_caption_sections`)."""
    if not caption:
        return prompt
    parsed = parse_caption_sections(caption)
    text = "\n\n".join(x for t, x in parsed
                       if sections is None or t in sections)
    if not text:
        return prompt
    period = "" if text.endswith((".", "!", "?")) else "."
    return f"{prompt} {_CAPTION_PREFIX}{text}{period}"


def selected_caption_titles(caption: str,
                            sections: list[str] | None) -> list[str]:
    """The caption section titles that ``apply_reference`` folds in
    (caption order), for the PNG metadata record."""
    return [t for t, _ in parse_caption_sections(caption)
            if sections is None or t in sections]
