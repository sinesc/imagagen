"""Soft inpainting (Forge builtin extension) through the API, against the
MockBackend: the enabled flag and the six options are MODEL CONFIGURATION
settings (per direction, enabled by default) — the generate body carries no
soft fields and the mock ignores them (it just blends init with
mask*denoise). The generated inpaint PNG metadata carries Forge's extra-
parameter lines; generate-mode PNGs carry none.
"""

import io
import json

from PIL import Image
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config

MOCK_ID = "mock/mock.safetensors"


def _client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    return TestClient(app)


SOFT = {
    "soft_mask_blend_power": 2.0,
    "soft_mask_blend_scale": 0.25,
    "soft_inpaint_detail_preservation": 8.0,
    "soft_composite_mask_influence": 0.5,
    "soft_composite_difference_threshold": 0.75,
    "soft_composite_difference_contrast": 4.0,
}


def _png_bytes(value=10, size=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (value, value, value)).save(buf, format="PNG")
    return buf.getvalue()


def _setup_canvas(c, size=64):
    assert c.post("/api/canvas/upload", content=_png_bytes(size=size)).status_code == 200
    # paint a small mask square in the centre (the client does the brush
    # work; the server only receives the synced state)
    img = Image.new("RGB", (size, size), (10, 10, 10))
    mask = Image.new("RGBA", (size, size), (128, 128, 128, 0))
    px = mask.load()
    mid = size // 2
    for y in range(mid - 2, mid + 2):
        for x in range(mid - 2, mid + 2):
            px[x, y] = (128, 128, 128, 255)
    buf_i, buf_m = io.BytesIO(), io.BytesIO()
    img.save(buf_i, format="PNG")
    mask.save(buf_m, format="PNG")
    assert c.post("/api/canvas/sync",
                  files={"image": ("canvas.png", buf_i.getvalue(), "image/png"),
                         "mask": ("mask.png", buf_m.getvalue(), "image/png")}
                  ).status_code == 200


def _wait_done(ws, job: str, timeout: float = 10.0):
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.receive_json(mode="text")
        except Exception:
            break
        if msg.get("type") == "done" and msg.get("job") == job:
            return msg
    raise TimeoutError(f"no done event for job {job}")


def _save_config(c, i2i: dict, name="Soft cfg"):
    r = c.post("/api/model-configs", json={
        "name": name, "model": MOCK_ID,
        "t2i": {"loras": []}, "i2i": {"loras": [], **i2i},
    })
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_inpaint_metadata_carries_config_soft_options(tmp_path):
    """The i2i section's soft flag + options ride into the request and the
    PNG metadata (Forge's extra-parameter labels)."""
    with _client(tmp_path) as c:
        _setup_canvas(c)
        _save_config(c, {"soft": True, **SOFT})
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "mode": "inpaint_masked", "denoise": 0.6,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert meta["Soft inpainting enabled"] == "True"
            assert meta["Soft inpainting schedule bias"] == 2.0
            assert meta["Soft inpainting preservation strength"] == 0.25
            assert meta["Soft inpainting transition contrast boost"] == 8.0
            assert meta["Soft inpainting mask influence"] == 0.5
            assert meta["Soft inpainting difference threshold"] == 0.75
            assert meta["Soft inpainting difference contrast"] == 4.0


def test_soft_disabled_in_config_writes_flag_false(tmp_path):
    # a disabled soft inpainting still writes the flag (like the Forge
    # extension), so the PNG records the configuration's state
    with _client(tmp_path) as c:
        _setup_canvas(c)
        _save_config(c, {"soft": False, "soft_mask_blend_power": 3.0})
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "mode": "inpaint_masked", "denoise": 0.6,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert meta["Soft inpainting enabled"] == "False"
            assert meta["Soft inpainting schedule bias"] == 3.0


def test_default_config_is_enabled_with_forge_defaults(tmp_path):
    """The in-memory default configuration (and old files without the key)
    has soft inpainting ENABLED with the Forge default options."""
    with _client(tmp_path) as c:
        _setup_canvas(c)
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "mode": "inpaint_masked", "denoise": 0.6,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert meta["Soft inpainting enabled"] == "True"
            assert meta["Soft inpainting schedule bias"] == 1.0
            assert meta["Soft inpainting preservation strength"] == 0.5
            assert meta["Soft inpainting transition contrast boost"] == 4.0
            assert meta["Soft inpainting mask influence"] == 0.0
            assert meta["Soft inpainting difference threshold"] == 0.5
            assert meta["Soft inpainting difference contrast"] == 2.0


def test_old_config_file_without_the_key_defaults_enabled(tmp_path):
    """A configuration file predating the soft keys loads with the flag
    enabled (pydantic defaults) and the options at the Forge defaults."""
    cfg_dir = tmp_path / "user" / "model-configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "legacy.json").write_text(json.dumps({
        "name": "Legacy", "model": MOCK_ID,
        "t2i": {"loras": []}, "i2i": {"loras": []},
    }), encoding="utf-8")
    with _client(tmp_path) as c:
        e = [x for x in c.get("/api/models").json()["models"]
             if x["name"] == "Legacy"][0]
        assert e["i2i"]["soft"] is True
        assert e["t2i"]["soft"] is True
        assert e["i2i"]["soft_mask_blend_scale"] == 0.5
        _setup_canvas(c)
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64, "model_id": "legacy",
                "mode": "inpaint_not_masked", "denoise": 0.6,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert meta["Soft inpainting enabled"] == "True"


def test_generate_mode_carries_no_soft_metadata(tmp_path):
    """Soft inpainting needs an init mask: generate-mode requests carry no
    soft object and the PNG has no soft keys."""
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
            })
            assert r.status_code == 200
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert "Soft inpainting enabled" not in meta


def test_config_rejects_out_of_range_soft_values(tmp_path):
    # outside the Forge slider ranges -> 422 (the config API validates)
    with _client(tmp_path) as c:
        assert c.post("/api/model-configs", json={
            "name": "Bad", "model": MOCK_ID,
            "t2i": {"loras": []},
            "i2i": {"loras": [], "soft_mask_blend_power": 99.0},
        }).status_code == 422
        assert c.post("/api/model-configs", json={
            "name": "Bad", "model": MOCK_ID,
            "t2i": {"loras": []},
            "i2i": {"loras": [], "soft_inpaint_detail_preservation": 0.5},
        }).status_code == 422


def test_ui_state_no_longer_carries_soft_fields(tmp_path):
    """The ui-state dropped the soft fields (the configuration is the
    source of truth); posting them is a no-op."""
    with _client(tmp_path) as c:
        r = c.post("/api/ui-state", json={
            "prompt": "p", "soft_inpainting": True, "soft_mask_blend_power": 3.5,
        })
        assert r.json() == {"ok": True}
        state = c.get("/api/ui-state").json()["state"]
        assert "soft_inpainting" not in state
        assert "soft_mask_blend_power" not in state
    on_disk = json.loads((tmp_path / "user" / "ui-state.json").read_text())
    assert "soft_inpainting" not in on_disk
