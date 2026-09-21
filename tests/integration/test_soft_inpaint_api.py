"""Soft inpainting (Forge builtin extension) through the API, against the
MockBackend: the request body carries the six options (null = off; the
mock ignores them, it just blends init with mask*denoise), the generated
PNG metadata carries Forge's extra-parameter lines, and ui-state persists
the checkbox + the six options with the Forge slider limits.
"""

import io
import json
import time

from PIL import Image
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    return TestClient(app)


SOFT = {
    "mask_blend_power": 2.0,
    "mask_blend_scale": 0.25,
    "inpaint_detail_preservation": 8.0,
    "composite_mask_influence": 0.5,
    "composite_difference_threshold": 0.75,
    "composite_difference_contrast": 4.0,
}


def _png_bytes(value=10, size=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (value, value, value)).save(buf, format="PNG")
    return buf.getvalue()


def _setup_canvas(c, size=64):
    assert c.post("/api/canvas/upload", content=_png_bytes(size=size)).status_code == 200
    assert c.post("/api/canvas/stroke", json={
        "points": [[size / 2, size / 2]], "size": 8, "target": "mask"}
    ).status_code == 200


def _wait_done(ws, job: str, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.receive_json(mode="text")
        except Exception:
            break
        if msg.get("type") == "done" and msg.get("job") == job:
            return msg
    raise TimeoutError(f"no done event for job {job}")


def test_generate_with_soft_inpainting_roundtrips_metadata(tmp_path):
    with _client(tmp_path) as c:
        _setup_canvas(c)
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "mode": "inpaint_masked", "denoise": 0.6,
                "soft_inpainting": SOFT,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            # the result PNG's metadata carries Forge's extra parameters
            thumb = c.get("/api/thumb/1/0.png")
            assert thumb.status_code == 200
            meta = c.post("/api/png-meta", content=thumb.content).json()["meta"]
            assert meta["Soft inpainting enabled"] == "True"
            assert meta["Soft inpainting schedule bias"] == "2"
            assert meta["Soft inpainting preservation strength"] == "0.25"
            assert meta["Soft inpainting transition contrast boost"] == "8"
            assert meta["Soft inpainting mask influence"] == "0.5"
            assert meta["Soft inpainting difference threshold"] == "0.75"
            assert meta["Soft inpainting difference contrast"] == "4"


def test_generate_with_soft_disabled_writes_flag_false(tmp_path):
    # a disabled soft inpainting still writes the flag (like the Forge
    # extension), so dropping such an image can switch the UI checkbox off
    with _client(tmp_path) as c:
        _setup_canvas(c)
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "mode": "inpaint_masked", "denoise": 0.6,
                "soft_inpainting": {"enabled": False, "mask_blend_power": 3},
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert meta["Soft inpainting enabled"] == "False"
            assert meta["Soft inpainting schedule bias"] == "3"


def test_generate_without_soft_has_no_soft_metadata(tmp_path):
    with _client(tmp_path) as c:
        _setup_canvas(c)
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "mode": "inpaint_masked", "denoise": 0.6,
                "soft_inpainting": None,
            })
            assert r.status_code == 200
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert "Soft inpainting enabled" not in meta


def test_generate_rejects_bad_soft_values(tmp_path):
    # outside the Forge slider ranges -> 422
    bad = dict(SOFT)
    bad["mask_blend_power"] = 99.0
    with _client(tmp_path) as c:
        assert c.post("/api/generate", json={
            "prompt": "t", "width": 64, "height": 64,
            "mode": "inpaint_masked", "soft_inpainting": bad,
        }).status_code == 422
    # a partial object: the Forge defaults fill in the missing fields
    with _client(tmp_path) as c:
        _setup_canvas(c)
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "t", "width": 64, "height": 64,
                "mode": "inpaint_masked",
                "soft_inpainting": {"mask_blend_power": 2.0},
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert meta["Soft inpainting schedule bias"] == "2"
            assert meta["Soft inpainting preservation strength"] == "0.5"


def test_ui_state_persists_soft_fields(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/ui-state", json={
            "soft_inpainting": True,
            "soft_mask_blend_power": 3.5,
            "soft_composite_mask_influence": 0.25,
        })
        assert r.json() == {"ok": True}
        state = c.get("/api/ui-state").json()["state"]
        assert state["soft_inpainting"] is True
        assert state["soft_mask_blend_power"] == 3.5
        assert state["soft_composite_mask_influence"] == 0.25
        # unspecified options keep the Forge defaults
        assert state["soft_inpaint_detail_preservation"] == 4.0
        assert state["soft_composite_difference_threshold"] == 0.5
    on_disk = json.loads((tmp_path / "user" / "ui-state.json").read_text())
    assert on_disk["soft_inpainting"] is True
    assert on_disk["soft_mask_blend_scale"] == 0.5


def test_ui_state_rejects_out_of_range_soft_values(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/api/ui-state", json={
            "soft_mask_blend_power": 99.0,
        }).status_code == 422
        assert c.post("/api/ui-state", json={
            "soft_inpaint_detail_preservation": 0.5,
        }).status_code == 422
