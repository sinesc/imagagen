import io
import json

from PIL import Image, PngImagePlugin

from imagagen import metadata
from imagagen.pngmeta import convert_to_imagagen, read_image_metadata
from imagagen.styles import Style, StyleStore


def _forge_text() -> str:
    return (
        "masterpiece, (x:1.2) <lora:mylora:0.8>, stylepos\n"
        "Negative prompt: ugly, styleneg\n"
        "Steps: 25, Sampler: Euler a, Scheduler: karras, CFG scale: 7, "
        "Seed: 42, Denoising strength: 0.7, Model: sd_xl.safetensors, "
        "Model hash: 36ed1679, Version: v1.9.1, VAE: fp16"
    )


def _save_png(path, params_text: str, size=(1024, 1024)):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", params_text)
    Image.new("RGB", size, "blue").save(path, pnginfo=info)


def test_forge_text_converted_to_imagagen_format(tmp_path):
    p = tmp_path / "forge.png"
    _save_png(p, _forge_text())
    meta, w, h = read_image_metadata(p)
    doc = convert_to_imagagen(meta, w, h)
    assert doc["format"] == "imagagen"
    # LoRA tokens come out of the prompt and into loras entries
    assert doc["loras"] == ["mylora@0.8"]
    assert "<lora:" not in doc["prompt"]
    assert doc["steps"] == 25 and doc["cfg"] == 7.0 and doc["seed"] == 42
    assert doc["denoise"] == 0.7
    assert (doc["width"], doc["height"]) == (1024, 1024)
    # Forge extras travel along verbatim (lenient format)
    assert doc["VAE"] == "fp16"
    assert doc["version"] == "v1.9.1"
    # canonical keys precede the extras
    keys = list(doc)
    assert keys.index("format") < keys.index("prompt") < keys.index("VAE")
    # output is valid JSON
    json.dumps(doc, ensure_ascii=False)


def test_forge_size_falls_back_to_image_size(tmp_path):
    p = tmp_path / "forge.png"
    _save_png(p, _forge_text(), size=(512, 768))  # no "Size:" key in text
    meta, w, h = read_image_metadata(p)
    doc = convert_to_imagagen(meta, w, h)
    assert (doc["width"], doc["height"]) == (512, 768)


def test_own_json_passes_through_verbatim(tmp_path):
    params = metadata.build_parameters(
        prompt="a fox # comment", negative_prompt="n", steps=20, sampler="Euler",
        scheduler="Simple", cfg=5.0, seed=7, width=64, height=64,
        model_name="sdxl/x", model_hash="abc",
    )
    p = tmp_path / "own.png"
    _save_png(p, params, size=(64, 64))
    meta, w, h = read_image_metadata(p)
    doc = convert_to_imagagen(meta, w, h)
    assert doc == meta  # already our format: untouched
    assert doc["prompt"] == "a fox # comment"


def test_styles_peeled_via_store(tmp_path):
    p = tmp_path / "forge.png"
    _save_png(p, _forge_text())
    store = StyleStore(tmp_path / "styles.csv")
    store.save(Style(name="S", prompt="stylepos", negative_prompt="styleneg"))
    meta, w, h = read_image_metadata(p)
    doc = convert_to_imagagen(meta, w, h, store)
    assert [s["name"] for s in doc["styles"]] == ["S"]
    assert "stylepos" not in doc["prompt"]
    assert "styleneg" not in doc["negative_prompt"]


def test_jpeg_comment_fallback(tmp_path):
    # A1111/Forge store the parameters text in the JPEG comment
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), "red").save(
        buf, format="JPEG", comment=("parameters: " + _forge_text()).encode()
    )
    p = tmp_path / "forge.jpg"
    p.write_bytes(buf.getvalue())
    meta, w, h = read_image_metadata(p)
    doc = convert_to_imagagen(meta, w, h)
    assert doc["format"] == "imagagen"
    assert doc["seed"] == 42
    assert (doc["width"], doc["height"]) == (64, 32)
