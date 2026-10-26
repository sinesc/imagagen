import io
import json

import numpy as np
from PIL import Image

from imagagen import metadata


def test_non_ascii_prompt_survives_png_write(tmp_path):
    # non-ASCII text makes Pillow write the chunk as iTXt; the reader must
    # handle that
    img = Image.new("RGB", (4, 4), "red")
    params = metadata.build_parameters(
        prompt="eine Eule \u00fcber dem Fluss", negative_prompt="", steps=10,
        sampler="Euler a", scheduler="Karras", cfg=7.0, seed=1, width=4,
        height=4, model_name="m", model_hash="h",
    )
    img, info = metadata.attach_metadata(img, params)
    img.save(tmp_path / "x.png", format="PNG", pnginfo=info)
    out = metadata.read_metadata((tmp_path / "x.png").read_bytes())
    assert out["prompt"] == "eine Eule \u00fcber dem Fluss"


def test_roundtrip_full():
    params = metadata.build_parameters(
        prompt="a red fox",
        negative_prompt="blurry",
        steps=25,
        sampler="Euler a",
        scheduler="Karras",
        cfg=6.5,
        seed=1234,
        width=1024,
        height=768,
        model_name="sd_xl_base_1.0",
        model_hash="ab12cd34ef56ab78",
        loras=[("sdxl/dmd2", 0.8)],
    )
    parsed = metadata.parse_parameters(params)
    assert parsed["format"] == "imagagen"
    assert parsed["prompt"] == "a red fox"
    assert parsed["negative_prompt"] == "blurry"
    assert parsed["steps"] == 25
    assert parsed["sampler"] == "Euler a"
    assert parsed["scheduler"] == "Karras"
    assert parsed["cfg"] == 6.5
    assert parsed["seed"] == 1234
    assert (parsed["width"], parsed["height"]) == (1024, 768)
    assert parsed["model"] == "sd_xl_base_1.0"
    assert parsed["model_hash"] == "ab12cd34ef56ab78"
    assert parsed["loras"] == ["sdxl/dmd2@0.8"]


def test_model_family_roundtrip():
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=12, sampler="Euler a",
        scheduler="Karras", cfg=1.5, seed=1, width=64, height=64,
        model_name="some/dir/model", model_hash="h",
        model_config="My Config", model_family="sdxl",
    )
    parsed = metadata.parse_parameters(params)
    assert parsed["model_config"] == "My Config"
    assert parsed["model_family"] == "sdxl"
    # absent when the family is not given (older callers)
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=12, sampler="Euler a",
        scheduler="Karras", cfg=1.5, seed=1, width=64, height=64,
        model_name="m", model_hash="h",
    )
    assert "model_family" not in metadata.parse_parameters(params)


def test_denoise_roundtrip():
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=12, sampler="Euler a",
        scheduler="Polyexponential", cfg=1.5, seed=1, width=1280,
        height=1440, model_name="m", model_hash="h", denoise=0.55,
    )
    assert metadata.parse_parameters(params)["denoise"] == 0.55
    # absent for generate-mode images (no denoise passed)
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=12, sampler="Euler a",
        scheduler="Polyexponential", cfg=1.5, seed=1, width=64,
        height=64, model_name="m", model_hash="h",
    )
    assert "denoise" not in metadata.parse_parameters(params)


def test_distilled_roundtrip():
    # both states are recorded (like the SAG flags); omitted for non-Flux
    # (distilled=None), where the line must not appear at all
    for value in (True, False):
        params = metadata.build_parameters(
            prompt="p", negative_prompt="", steps=20, sampler="Euler",
            scheduler="Flow", cfg=3.5, seed=1, width=1024, height=1024,
            model_name="flux", model_hash="h", distilled=value,
        )
        assert metadata.parse_parameters(params)["Distilled"] == str(value)
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=20, sampler="Euler a",
        scheduler="Karras", cfg=7.0, seed=1, width=512, height=512,
        model_name="sdxl", model_hash="h",
    )
    assert "Distilled" not in metadata.parse_parameters(params)


def test_parameters_is_json_document():
    params = metadata.build_parameters(
        prompt="hello", negative_prompt="", steps=10, sampler="Euler a",
        scheduler="Karras", cfg=7.0, seed=1, width=64, height=64,
        model_name="m", model_hash="h",
    )
    doc = json.loads(params)
    assert doc["format"] == "imagagen"
    assert doc["prompt"] == "hello"
    assert doc["steps"] == 10
    assert doc["sampler"] == "Euler a"
    assert doc["cfg"] == 7.0
    assert (doc["width"], doc["height"]) == (64, 64)
    assert doc["version"] == metadata._VERSION


def test_multiline_prompt_and_negative_roundtrip():
    # the legacy text format lost a multi-line negative prompt (and shifted
    # the meta line); the JSON document must keep both verbatim
    params = metadata.build_parameters(
        prompt="line one\nline two", negative_prompt="bad one\nbad two",
        steps=10, sampler="Euler a", scheduler="Karras", cfg=7.0, seed=1,
        width=64, height=64, model_name="m", model_hash="h",
    )
    parsed = metadata.parse_parameters(params)
    assert parsed["prompt"] == "line one\nline two"
    assert parsed["negative_prompt"] == "bad one\nbad two"
    # the other fields must survive too (no displaced meta line any more)
    assert parsed["steps"] == 10
    assert parsed["seed"] == 1
    assert parsed["model"] == "m"


def test_attach_and_read_png_metadata():
    img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    params = metadata.build_parameters(
        prompt="p", negative_prompt="n", steps=5, sampler="Euler a",
        scheduler="Karras", cfg=3.0, seed=9, width=4, height=4,
        model_name="m", model_hash="h",
    )
    img, pnginfo = metadata.attach_metadata(img, params)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    parsed = metadata.read_metadata(buf.getvalue())
    assert parsed["prompt"] == "p"
    assert parsed["negative_prompt"] == "n"
    assert parsed["steps"] == 5
    assert parsed["seed"] == 9
    assert (parsed["width"], parsed["height"]) == (4, 4)
    assert parsed["sampler"] == "Euler a"
    assert parsed["model"] == "m"


def test_read_metadata_plain_png_none():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buf, format="PNG")
    assert metadata.read_metadata(buf.getvalue()) is None


# ---------------- LoRA tokens embedded in the prompt (A1111/Forge) ----------------

def test_extract_prompt_loras_variants():
    base = "photo of a fabulous christmas tree, sparkling christmas lights"
    cases = [
        f"{base}, <lora:EnvyZoomSliderXL1.0:-2.0>,<lora:DMD2fp32XL1.0:0.7>",
        f"<lora:DMD2fp32XL1.0:0.7>, {base}, <lora:EnvyZoomSliderXL1.0:-2.0>",
        f"{base}, <lora:EnvyZoomSliderXL1.0:-2.0>\n<lora:DMD2fp32XL1.0:0.7>",
        f"{base}, <lora:EnvyZoomSliderXL1.0:-2.0>,\n<lora:DMD2fp32XL1.0:0.7>",
    ]
    for text in cases:
        loras, remainder = metadata.extract_prompt_loras(text)
        assert loras == [
            ("EnvyZoomSliderXL1.0", -2.0), ("DMD2fp32XL1.0", 0.7),
        ] or loras == [
            ("DMD2fp32XL1.0", 0.7), ("EnvyZoomSliderXL1.0", -2.0),
        ]
        assert "<lora" not in remainder


def test_extract_prompt_loras_preserves_order_and_weight():
    loras, remainder = metadata.extract_prompt_loras(
        "a cat <lora:Foo:1.0> and <lora:Bar:0.25> a dog"
    )
    assert loras == [("Foo", 1.0), ("Bar", 0.25)]
    assert remainder == "a cat  and  a dog"


def test_extract_prompt_loras_none():
    loras, remainder = metadata.extract_prompt_loras("a plain prompt")
    assert loras == []
    assert remainder == "a plain prompt"


# --- PNG text chunk position -------------------------------------------------


def _png_trailing_text(params_text: str) -> bytes:
    """Build a minimal 2x2 RGB PNG whose ``parameters`` tEXt chunk is written
    AFTER the IDAT, as external crop tools (and some Forge re-saves) do.
    Pillow only parses such chunks once the image is decoded, which is the
    case ``read_metadata`` must handle."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00\xff" * 2 for _ in range(2))
    idat = zlib.compress(raw)
    text = b"parameters" + b"\x00" + params_text.encode()
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"tEXt", text)
        + chunk(b"IEND", b"")
    )


def test_read_metadata_trailing_text_chunk():
    """Metadata in a post-IDAT chunk is still found (regression: cropped
    Forge images came back with no metadata at all)."""
    params = "a thing\nNegative prompt: blurry\nSteps: 2, Sampler: LCM, Seed: 3, Size: 2x2"
    meta = metadata.read_metadata(_png_trailing_text(params))
    assert meta is not None
    assert meta["prompt"] == "a thing"
    assert meta["negative_prompt"] == "blurry"
    assert meta["steps"] == 2
    assert meta["width"] == 2 and meta["height"] == 2


def test_parse_lora_hashes_quoted_single():
    # the form the reference Forge images use (whole value in quotes)
    text = (
        "a cat\nSteps: 12, Sampler: LCM, Model: m, "
        'Lora hashes: "DMD2fp32XL1.0: 21251344fb41", Version: f2.0.1'
    )
    p = metadata.parse_parameters(text)
    assert p["lora_hashes"] == {"DMD2fp32XL1.0": "21251344fb41"}
    assert p["version"] == "f2.0.1"
    assert "Lora hashes" not in p
    assert p["model"] == "m"


def test_parse_lora_hashes_bare_single():
    p = metadata.parse_parameters(
        "a cat\nSteps: 2, Sampler: LCM, Model: m, "
        "Lora hashes: DMD2fp32XL1.0: 21251344fb41, Version: f2"
    )
    assert p["lora_hashes"] == {"DMD2fp32XL1.0": "21251344fb41"}
    assert p["version"] == "f2"


def test_parse_lora_hashes_bare_multiple():
    # Forge joins entries with the same ", " as the meta line itself, so
    # the value spans parts; the later entries must not leak as keys
    p = metadata.parse_parameters(
        "a cat\nSteps: 2, Sampler: LCM, "
        "Lora hashes: Alpha: aaaaaaaaaaaa, Beta: bbbbbbbbbbbb, Version: f2"
    )
    assert p["lora_hashes"] == {
        "Alpha": "aaaaaaaaaaaa",
        "Beta": "bbbbbbbbbbbb",
    }
    assert p["version"] == "f2"
    assert "Beta" not in p


def test_parse_no_lora_hashes():
    p = metadata.parse_parameters("a cat\nSteps: 2, Sampler: LCM, Model: m")
    assert "lora_hashes" not in p


def test_forge_lora_hash_is_sha256_of_basename():
    import hashlib

    # reference value: dev/references/cute-kitten-dmd2.png
    assert (
        metadata.forge_lora_hash("DMD2fp32XL1.0.safetensors")
        == hashlib.sha256(b"DMD2fp32XL1.0.safetensors").hexdigest()[:12]
        == "21251344fb41"
    )
    # the path is ignored, the extension is part of the name
    assert metadata.forge_lora_hash(
        "/some/where/Deep.safetensors") == metadata.forge_lora_hash("Deep.safetensors")


def test_disabled_loras_roundtrip():
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=12, sampler="Euler a",
        scheduler="Polyexponential", cfg=1.5, seed=1, width=64,
        height=64, model_name="m", model_hash="h",
        loras=[("Foo", 0.8)],
        disabled_loras=[("Bar", -0.3), ("Qux/Deep", 1.0)],
    )
    # both lists travel as name@weight strings, applied vs disabled
    parsed = metadata.parse_parameters(params)
    assert parsed["loras"] == ["Foo@0.8"]
    assert parsed["disabled_loras"] == ["Bar@-0.3", "Qux/Deep@1"]


def test_disabled_loras_absent_by_default():
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=12, sampler="Euler a",
        scheduler="Polyexponential", cfg=1.5, seed=1, width=64,
        height=64, model_name="m", model_hash="h", loras=[("Foo", 0.8)],
    )
    assert "disabled_loras" not in metadata.parse_parameters(params)


def test_model_config_roundtrip():
    """The generating configuration's NAME rides in the PNG so an import can
    re-select it; absent (foreign images) = key omitted."""
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=10, sampler="Euler a",
        scheduler="Karras", cfg=7.0, seed=1, width=64, height=64,
        model_name="m", model_hash="h", model_config="My A + DMD2 Lora",
    )
    assert metadata.parse_parameters(params)["model_config"] == "My A + DMD2 Lora"
    params = metadata.build_parameters(
        prompt="p", negative_prompt="", steps=10, sampler="Euler a",
        scheduler="Karras", cfg=7.0, seed=1, width=64, height=64,
        model_name="m", model_hash="h",
    )
    assert "model_config" not in metadata.parse_parameters(params)


def test_parse_lora_entries():
    assert metadata.parse_lora_entries(["dmd2@0.7", "a/b@1", "bare"]) == [
        ("dmd2", 0.7), ("a/b", 1.0), ("bare", 1.0)]
    assert metadata.parse_lora_entries([]) == []
    assert metadata.parse_lora_entries(None) == []
