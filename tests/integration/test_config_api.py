"""Phase 4 configuration layer over the API (mock backend, no GPU).

Covers: model configurations in the model selector (defaults + files),
config CRUD, default-override resolution (cfg -1 / "use model default"
sampler+scheduler -> config values -> built-in; LoRAs prepended), LoRA
discovery, and preset CRUD. Resolution is observed through the PNG
metadata the queue writes into the generated thumbnails.
"""

import io
import json
import time

import pytest
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


def _wait_done(client, ws, job_id: int, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.receive_json(mode="text")
        except Exception:
            break
        if msg.get("type") == "done" and msg.get("job") == job_id:
            return msg
        if msg.get("type") == "error":
            raise AssertionError(f"generation error: {msg}")
    raise TimeoutError("no done event")


def _thumb_meta(client, job_id: int, index: int = 0) -> dict:
    data = client.get(f"/api/thumb/{job_id}/{index}.png").content
    m = metadata.read_metadata(data)
    assert m is not None
    return m


def _generate(client, ws, **kw) -> int:
    body = {"prompt": "p", "width": 64, "height": 64, "steps": 2,
            "seed": 5, "cfg": 7.0}
    body.update(kw)
    r = client.post("/api/generate", json=body)
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


# ------------------------------------------------------- model selector ---


def test_models_lists_default_entry(client):
    client, _ = client
    body = client.get("/api/models").json()
    assert len(body["models"]) == 1
    e = body["models"][0]
    assert e["id"] == MOCK_ID
    assert e["name"] == "Mock Model"
    assert e["model_id"] == MOCK_ID
    assert e["is_default"] is True
    assert e["t2i"] == {"sampler": None, "scheduler": None, "steps": None,
                        "cfg": None, "sag": False, "sag_scale": 0.5,
                        "sag_blur_sigma": 2.0, "sag_threshold": 1.0,
                        "soft": True, "soft_mask_blend_power": 1.0,
                        "soft_mask_blend_scale": 0.5,
                        "soft_inpaint_detail_preservation": 4.0,
                        "soft_composite_mask_influence": 0.0,
                        "soft_composite_difference_threshold": 0.5,
                        "soft_composite_difference_contrast": 2.0,
                        "loras": []}
    assert body["current"] == MOCK_ID


def test_model_config_crud(client):
    client, tmp_path = client
    # create from the current (default) entry
    r = client.post("/api/model-configs/create",
                    json={"name": "Mock tuned", "based_on": MOCK_ID})
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["id"] == "mock-tuned"
    assert created["model_id"] == MOCK_ID
    assert created["is_default"] is False

    # now the selector lists the file config (model A's default entry is
    # gone, the model is covered)
    body = client.get("/api/models").json()
    assert [e["id"] for e in body["models"]] == ["mock-tuned"]

    # update: t2i gets sampler/scheduler/CFG/LoRAs, i2i only a CFG
    r = client.post("/api/model-configs", json={
        "name": "Mock tuned", "model": MOCK_ID,
        "t2i": {"sampler": "DPM++ 2M", "scheduler": "Polyexponential",
                "cfg": 9.5, "loras": [{"name": "sdxl/foo", "weight": 0.7}]},
        "i2i": {"sampler": None, "scheduler": None, "cfg": 6.0, "loras": []},
    })
    assert r.status_code == 200, r.text
    assert (tmp_path / "user" / "model-configs" / "mock-tuned.json").is_file()

    # invalid bodies are rejected (pydantic: 422)
    assert client.post("/api/model-configs",
                       json={"name": "", "model": MOCK_ID}).status_code == 422

    # delete
    r = client.post("/api/model-configs/delete", json={"id": "mock-tuned"})
    assert r.json() == {"ok": True}
    assert not (tmp_path / "user" / "model-configs" / "mock-tuned.json").exists()
    body = client.get("/api/models").json()
    assert [e["id"] for e in body["models"]] == [MOCK_ID]


def test_editing_default_config_materializes_it(client):
    client, tmp_path = client
    # the default entry's id is the model path (with slashes)
    r = client.post("/api/model-configs", json={
        "id": MOCK_ID,
        "name": "Mock Model",
        "model": MOCK_ID,
        "t2i": {"sampler": None, "scheduler": None, "cfg": 6.5, "loras": []},
        "i2i": {"sampler": None, "scheduler": None, "cfg": None, "loras": []},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "mock-model"
    assert body["is_default"] is False
    assert (tmp_path / "user" / "model-configs" / "mock-model.json").is_file()
    # the selector now lists the file config instead of the in-memory
    # default (same name, model covered)
    models = client.get("/api/models").json()
    assert [(e["id"], e["name"], e["is_default"]) for e in models["models"]] == [
        ("mock-model", "Mock Model", False)]
    assert models["current"] == "mock-model"


def test_delete_default_config_rejected(client):
    client, _ = client
    # the default entry id is the model path (with slashes) -> body, not URL
    r = client.post("/api/model-configs/delete", json={"id": MOCK_ID})
    assert r.status_code == 400
    assert "cannot be deleted" in r.json()["error"]


# --------------------------------------------------------------- resolution
def test_cfg_sentinel_resolves_to_config_value(client):
    client, _ = client
    client.post("/api/model-configs", json={
        "name": "Tuned", "model": MOCK_ID,
        "t2i": {"sampler": None, "scheduler": None, "cfg": 9.5, "loras": []},
    })
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, cfg=-1, model_id="tuned")
        _wait_done(client, ws, job)
        m = _thumb_meta(client, job)
    assert m["cfg"] == 9.5  # config override, not the built-in 7.0


def test_steps_sentinel_resolves_to_config_value(client):
    client, _ = client
    client.post("/api/model-configs", json={
        "name": "Tuned", "model": MOCK_ID,
        "t2i": {"sampler": None, "scheduler": None, "steps": 9, "cfg": None, "loras": []},
    })
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, steps=-1, model_id="tuned")
        _wait_done(client, ws, job)
        m = _thumb_meta(client, job)
    assert m["steps"] == 9  # config override, not the request's -1


def test_cfg_sentinel_without_config_uses_builtin(client):
    client, _ = client
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, cfg=-1)
        _wait_done(client, ws, job)
        m = _thumb_meta(client, job)
    assert m["cfg"] == 7.0  # backend built-in


def test_sampler_scheduler_and_loras_resolution(client):
    client, _ = client
    client.post("/api/model-configs", json={
        "name": "Tuned", "model": MOCK_ID,
        "t2i": {"sampler": "DPM++ 2M", "scheduler": "Polyexponential",
                "cfg": 4.0, "loras": [{"name": "sdxl/foo", "weight": 0.7}]},
    })
    with client.websocket_connect("/api/ws") as ws:
        # everything "use model default" + no user loras -> config values
        job = _generate(client, ws, cfg=-1, model_id="tuned")
        _wait_done(client, ws, job)
        m = _thumb_meta(client, job)
        assert m["sampler"] == "DPM++ 2M"
        assert m["scheduler"] == "Polyexponential"
        assert m["cfg"] == 4.0
        assert m["loras"] == ["sdxl/foo@0.7"]

        # user loras are appended *after* the config's (prepended) ones
        job = _generate(client, ws, cfg=-1, model_id="tuned",
                        loras=[["sdxl/bar", 0.4]])
        _wait_done(client, ws, job)
        m = _thumb_meta(client, job)
        assert m["loras"] == ["sdxl/foo@0.7", "sdxl/bar@0.4"]

        # explicit user settings win over the config
        job = _generate(client, ws, cfg=3.0, sampler="Euler a",
                        scheduler="Karras", model_id="tuned")
        _wait_done(client, ws, job)
        m = _thumb_meta(client, job)
        assert (m["cfg"], m["sampler"], m["scheduler"]) == (3.0, "Euler a",
                                                            "Karras")


def test_inpaint_mode_uses_i2i_section(client, tmp_path):
    client, _ = client
    client.post("/api/model-configs", json={
        "name": "Tuned", "model": MOCK_ID,
        "t2i": {"sampler": None, "scheduler": None, "cfg": 9.0, "loras": []},
        "i2i": {"sampler": "DPM++ 2M SDE", "scheduler": "Karras",
                "cfg": 2.5, "loras": []},
    })
    # need a canvas for inpaint modes
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, format="PNG")
    assert client.post("/api/canvas/upload", content=buf.getvalue()).status_code == 200
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, cfg=-1, model_id="tuned",
                        mode="inpaint_not_masked")
        _wait_done(client, ws, job)
        m = _thumb_meta(client, job)
    assert m["cfg"] == 2.5        # i2i cfg, not the t2i 9.0
    assert m["sampler"] == "DPM++ 2M SDE"


# ------------------------------------------------------------------ loras --


def test_lora_discovery(client, tmp_path):
    client, _ = client
    assert client.get("/api/loras").json() == {"loras": [], "settings": {}}
    (tmp_path / "loras" / "sdxl").mkdir(parents=True)
    (tmp_path / "loras" / "sdxl" / "dmd2.safetensors").write_bytes(b"")
    (tmp_path / "loras" / "sd15").mkdir()
    (tmp_path / "loras" / "sd15" / "x.ckpt").write_bytes(b"")
    assert client.get("/api/loras").json() == {"loras": ["x", "dmd2"],
                                               "settings": {}}


def test_lora_settings(client, tmp_path):
    client, _ = client
    (tmp_path / "loras" / "sdxl").mkdir(parents=True)
    f = tmp_path / "loras" / "sdxl" / "dmd2.safetensors"
    f.write_bytes(b"")
    j = f.with_suffix(".json")

    # no settings file yet: the LoRA is listed, settings stay absent
    assert client.get("/api/loras").json()["settings"] == {}

    # a Forge-style file is read (missing fields default, extra fields
    # like "sd version" are ignored)
    j.write_text(json.dumps({
        "description": "", "sd version": "SDXL",
        "activation text": "foo", "preferred weight": 0.7,
        "notes": "n"}))
    got = client.get("/api/loras").json()["settings"]["dmd2"]
    assert got["preferred weight"] == 0.7
    assert got["activation text"] == "foo"
    assert got["negative text"] == ""

    # saving writes the .json next to the .safetensors and preserves the
    # unknown extra fields
    r = client.post("/api/loras/settings", json={
        "name": "dmd2", "description": "d", "preferred_weight": 0.5,
        "activation_text": "a", "negative_text": "n", "notes": "m"})
    assert r.status_code == 200, r.text
    data = json.loads(j.read_text())
    assert data == {"description": "d", "sd version": "SDXL",
                    "activation text": "a", "preferred weight": 0.5,
                    "negative text": "n", "notes": "m"}
    assert client.get("/api/loras").json()["settings"]["dmd2"]["description"] == "d"

    # unknown LoRA name -> 404; weight out of range -> 422
    assert client.post("/api/loras/settings", json={"name": "nope"}).status_code == 404
    assert client.post("/api/loras/settings",
                       json={"name": "dmd2", "preferred_weight": 99}).status_code == 422


# ---------------------------------------------------------------- presets --


def test_preset_crud_and_apply_payload(client, tmp_path):
    client, _ = client
    assert client.get("/api/presets").json() == {"presets": []}
    r = client.post("/api/presets", json={
        "name": "Big square",
        "width": 1024, "height": 1024, "steps": 30,
        "controls": {"width": True, "height": True, "steps": True},
    })
    assert r.status_code == 200, r.text
    assert (tmp_path / "user" / "presets" / "big-square.json").is_file()
    assert client.get("/api/presets").json() == {
        "presets": [{"name": "Big square", "group": None}]}

    # group is an optional label (null/absent = ungrouped); "" normalizes
    # to null
    r = client.post("/api/presets", json={"name": "Grouped", "group": "Touchup"})
    assert r.status_code == 200, r.text
    assert client.get("/api/presets/grouped").json()["preset"]["group"] == "Touchup"
    r = client.post("/api/presets", json={"name": "Blank group", "group": ""})
    assert r.status_code == 200
    assert client.get("/api/presets/blank-group").json()["preset"]["group"] is None

    body = client.get("/api/presets/big-square").json()
    p = body["preset"]
    assert p["name"] == "Big square"
    assert p["width"] == 1024
    assert p["cfg"] is None
    assert p["controls"]["width"] is True
    assert p["controls"]["seed"] is False
    # controlled() semantics: only checked settings with values
    controlled = {k: v for k, v in p.items()
                  if p["controls"].get(k) and v is not None}
    assert controlled == {"width": 1024, "height": 1024, "steps": 30}

    # update + delete
    r = client.post("/api/presets", json={
        "name": "Big square", "width": 512,
        "controls": {"width": True},
    })
    assert r.status_code == 200
    assert client.get("/api/presets/big-square").json()["preset"]["width"] == 512
    for name in ("Big square", "Grouped", "Blank group"):
        r = client.post("/api/presets/delete", json={"name": name})
        assert r.json() == {"ok": True}
    assert client.get("/api/presets").json() == {"presets": []}
    assert client.get("/api/presets/big-square").status_code == 404


def test_preset_with_model_default_sentinels(client):
    client, _ = client
    # the "set the form to use-model-defaults" preset: steps/cfg = -1
    r = client.post("/api/presets", json={
        "name": "Set defaults", "steps": -1, "cfg": -1.0,
        "controls": {"steps": True, "cfg": True}})
    assert r.status_code == 200, r.text
    p = client.get("/api/presets/set-defaults").json()["preset"]
    assert p["steps"] == -1
    assert p["cfg"] == -1.0
    # lookup by the *display* name works too (the UI select holds names,
    # the files are stored under the sanitized id)
    p2 = client.get("/api/presets/Set%20defaults").json()["preset"]
    assert p2 == p
    assert client.get("/api/presets/nope").status_code == 404


def test_preset_sampler_use_model_default(client):
    client, _ = client
    # "" is the main UI's "use model default" option for sampler/scheduler
    r = client.post("/api/presets", json={
        "name": "Reset sampler", "sampler": "", "scheduler": "",
        "controls": {"sampler": True, "scheduler": True}})
    assert r.status_code == 200, r.text
    p = client.get("/api/presets/reset-sampler").json()["preset"]
    assert p["sampler"] == ""
    assert p["scheduler"] == ""
    # mode has no empty state: "" is rejected
    r = client.post("/api/presets", json={"name": "bad", "mode": ""})
    assert r.status_code == 422
    # genuinely out-of-range values are still rejected
    r = client.post("/api/presets", json={"name": "bad", "steps": -2})
    assert r.status_code == 422


def test_preset_validation(client):
    client, _ = client
    r = client.post("/api/presets", json={"name": "x", "width": 9999})
    assert r.status_code == 422
    r = client.post("/api/presets", json={"name": "", "width": 512})
    assert r.status_code == 422
