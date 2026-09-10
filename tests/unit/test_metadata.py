import io

import numpy as np
from PIL import Image

from imagagen import metadata


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


def test_parameters_format_like_forge():
    params = metadata.build_parameters(
        prompt="hello", negative_prompt="", steps=10, sampler="Euler a",
        scheduler="Karras", cfg=7.0, seed=1, width=64, height=64,
        model_name="m", model_hash="h",
    )
    lines = params.splitlines()
    assert lines[0] == "hello"
    assert lines[1].startswith("Steps: 10, Sampler: Euler a, Scheduler: Karras, CFG scale: 7, ")
    assert "Size: 64x64" in lines[1]
    assert "Version: " in lines[1]


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
