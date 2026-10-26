"""PNG import -> model configuration selection (/api/png-meta).

imagagen PNGs record the configuration NAME (``model_config``) and the
endpoint resolves it back to that configuration's id. A1111/Forge PNGs
record no configuration, so the endpoint scores the configurations of the
image's checkpoint by LoRA match: the configuration that also provides the
image's LoRAs (names AND weights) wins over a bare-checkpoint one. The UI
selects ``meta.config_id``.
"""

import io
import json
import time

import pytest
from PIL import Image
from starlette.testclient import TestClient

from imagagen import metadata
from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config

MOCK_ID = "mock/mock.safetensors"


@pytest.fixture
def client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    with TestClient(app) as c:
        yield c, tmp_path


def _wait_done(client, ws, job_id: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.receive_json(mode="text")
        except Exception:
            break
        if msg.get("type") == "done" and msg.get("job") == job_id:
            return
        if msg.get("type") == "error":
            raise AssertionError(f"generation error: {msg}")
    raise TimeoutError("no done event")


def _forge_png(text: str, w: int = 64, h: int = 64) -> bytes:
    """A PNG carrying legacy A1111/Forge-style `parameters` text."""
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), (4, 4, 4))
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", text)
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _config(client, name: str, loras: list[dict]) -> str:
    c, _ = client
    r = c.post("/api/model-configs", json={
        "name": name, "model": MOCK_ID,
        "t2i": {"loras": loras}, "i2i": {"loras": loras},
    })
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ------------------------------------------------- imagagen round-trip ---


def test_imagagen_png_records_and_restores_the_configuration(client):
    c, tmp = client
    cfg_id = _config(client, "Mock + DMD", [{"name": "DMD2fp32XL1.0", "weight": 0.7}])
    with c.websocket_connect("/api/ws") as ws:
        job = c.post("/api/generate", json={
            "prompt": "p", "model_id": cfg_id, "width": 64, "height": 64,
            "steps": 2, "seed": 5, "cfg": 7.0,
        }).json()["job_id"]
        _wait_done(c, ws, job)
    png = c.get(f"/api/thumb/{job}/0.png").content
    meta = metadata.read_metadata(png)
    assert meta["model_config"] == "Mock + DMD"
    assert meta["model_family"] == "sdxl"  # MockBackend mirrors the SDXL family
    # the LoRA travels in the metadata (config LoRAs are part of the request)
    assert meta["loras"] == ["DMD2fp32XL1.0@0.7"]
    j = c.post("/api/png-meta", content=png).json()
    assert j["meta"]["config_id"] == cfg_id


def test_imagagen_png_of_a_deleted_configuration_falls_back_to_scoring(client):
    c, _ = client
    cfg_id = _config(client, "Mock + DMD", [{"name": "DMD2fp32XL1.0", "weight": 0.7}])
    with c.websocket_connect("/api/ws") as ws:
        job = c.post("/api/generate", json={
            "prompt": "p", "model_id": cfg_id, "width": 64, "height": 64,
            "steps": 2, "seed": 5, "cfg": 7.0,
        }).json()["job_id"]
        _wait_done(c, ws, job)
    png = c.get(f"/api/thumb/{job}/0.png").content
    c.post("/api/model-configs/delete", json={"id": cfg_id})
    assert c.post("/api/png-meta", content=png).json()["meta"].get("config_id") is None


# ------------------------------------------------- Forge/A1111 imports ---


def test_forge_import_prefers_the_lora_matching_configuration(client):
    c, _ = client
    dmd = _config(client, "Mock (DMD)", [{"name": "DMD2fp32XL1.0", "weight": 0.7}])
    plain = _config(client, "Mock (plain)", [])
    png = _forge_png(
        "a cat <lora:DMD2fp32XL1.0:0.7>\n"
        "Negative prompt: bad\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7.0, Seed: 5, "
        "Size: 64x64, Model: mock.safetensors, Model hash: abcdef12")
    meta = c.post("/api/png-meta", content=png).json()["meta"]
    assert meta["loras"] == ["DMD2fp32XL1.0@0.7"]
    assert meta["config_id"] == dmd and plain != dmd


def test_forge_import_without_loras_picks_the_bare_configuration(client):
    c, _ = client
    _config(client, "Mock (DMD)", [{"name": "DMD2fp32XL1.0", "weight": 0.7}])
    plain = _config(client, "Mock (plain)", [])
    png = _forge_png(
        "a cat\nNegative prompt: bad\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7.0, Seed: 5, "
        "Size: 64x64, Model: mock.safetensors, Model hash: abcdef12")
    assert c.post("/api/png-meta", content=png).json()["meta"]["config_id"] == plain


def test_forge_import_weight_mismatch_does_not_pick_the_lora_configuration(client):
    c, _ = client
    _config(client, "Mock (DMD)", [{"name": "DMD2fp32XL1.0", "weight": 0.7}])
    plain = _config(client, "Mock (plain)", [])
    png = _forge_png(
        "a cat <lora:DMD2fp32XL1.0:1>\n"
        "Negative prompt: bad\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7.0, Seed: 5, "
        "Size: 64x64, Model: mock.safetensors, Model hash: abcdef12")
    meta = c.post("/api/png-meta", content=png).json()["meta"]
    # the LoRA stays a UI selection (weight differs), so the bare config wins
    assert meta["config_id"] == plain


def test_forge_import_unknown_model_matches_nothing(client):
    c, _ = client
    _config(client, "Mock (DMD)", [{"name": "DMD2fp32XL1.0", "weight": 0.7}])
    png = _forge_png(
        "a cat\nNegative prompt: bad\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7.0, Seed: 5, "
        "Size: 64x64, Model: someOtherModel")
    meta = c.post("/api/png-meta", content=png).json()["meta"]
    assert meta.get("config_id") is None
    # the UI keeps the current model and shows the "Checkpoint not found"
    # error; without family info the current model's family is assumed
    assert meta["model_missing"] is True
    assert "model_family" not in meta


# ---------------------------------------- moved/renamed checkpoints ---


def _imagagen_png(meta_doc: dict) -> bytes:
    """A PNG whose parameters chunk is the given imagagen JSON document
    (built with metadata.build_parameters, i.e. what a generation writes)."""
    from PIL import Image

    img, info = metadata.attach_metadata(Image.new("RGB", (64, 64), (1, 2, 3)),
                                         json.dumps(meta_doc))
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _imagagen_doc(model: str, **kw) -> dict:
    doc = json.loads(metadata.build_parameters(
        "a cat", "", 20, "Euler a", "Karras", 7.0, 1, 64, 64,
        model, "abc123", **kw))
    return doc


def test_imagagen_import_exact_checkpoint_path_required(client):
    """imagagen records the full model id: a same-named checkpoint in
    ANOTHER subfolder must not be matched by basename (that silently
    picks the wrong file after a folder reorganisation); the import is
    marked model_missing and keeps the family for the UI's fallback."""
    c, _ = client
    _config(client, "Mock (plain)", [])
    png = _imagagen_png(_imagagen_doc("elsewhere/mock",
                                      model_family="sdxl"))
    meta = c.post("/api/png-meta", content=png).json()["meta"]
    assert "config_id" not in meta
    assert meta["model_missing"] is True
    assert meta["model_family"] == "sdxl"


def test_imagagen_import_derives_family_from_the_model_path(client):
    """imagagen PNGs written before the model_family key: the model id is
    a path under models/<family>/, so its first component is the family
    when this install has that family (not a basename guess)."""
    c, _ = client
    doc = _imagagen_doc("sdxl/mock")  # no model_family key (older PNG)
    assert "model_family" not in doc
    meta = c.post("/api/png-meta", content=_imagagen_png(doc)).json()["meta"]
    assert meta["model_missing"] is True
    assert meta["model_family"] == "sdxl"  # mock registry family


def test_imagagen_import_family_not_derived_from_foreign_path(client):
    # "Optimizer/..." is a LoRA-style subfolder, not a family this install
    # has: no family is invented, the UI assumes the selected model's one
    c, _ = client
    doc = _imagagen_doc("Optimizer/mock")
    meta = c.post("/api/png-meta", content=_imagagen_png(doc)).json()["meta"]
    assert meta["model_missing"] is True
    assert "model_family" not in meta
