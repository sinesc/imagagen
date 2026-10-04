"""Self attention guidance (Forge builtin extension) through the API,
against the MockBackend: the ENABLED flag is a model-configuration
setting (per t2i/i2i checkbox), the request body carries only the three
options (null = no metadata; the mock ignores them), and the generated
PNG metadata carries Forge's sag_* extra-parameter lines plus the
resolved sag_enabled flag.
"""

import io
import time

from PIL import Image
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config

MODEL = "mock/mock.safetensors"


def _client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    return TestClient(app)


def _upload_canvas(c: TestClient) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (5, 6, 7)).save(buf, format="PNG")
    c.post("/api/canvas/upload", content=buf.getvalue(),
           headers={"content-type": "image/png"})


SAG = {
    "scale": 0.75,
    "blur_sigma": 3.0,
    "threshold": 1.5,
}


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


def _meta(c: TestClient, job: str) -> dict:
    return c.post("/api/png-meta",
                  content=c.get(f"/api/thumb/{job}/0.png").content).json()["meta"]


def test_sag_enabled_by_model_config_t2i(tmp_path):
    """t2i sag=true in the selected configuration -> sag_enabled True in
    the metadata for "generate" (with the client's option values)."""
    with _client(tmp_path) as c:
        r = c.post("/api/model-configs", json={
            "name": "SAG on", "model": MODEL, "t2i": {"sag": True},
        })
        assert r.status_code == 200, r.text
        cfg = r.json()
        assert cfg["t2i"]["sag"] is True and cfg["i2i"]["sag"] is False
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/models/select", json={"id": cfg["id"]})
            assert r.status_code == 200, r.text
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64, "sag": SAG,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = _meta(c, r.json()["job_id"])
        assert meta["sag_enabled"] == "True"
        assert meta["sag_scale"] == 0.75
        assert meta["sag_blur_sigma"] == 3.0
        assert meta["sag_threshold"] == 1.5


def test_sag_disabled_by_default_writes_flag_false(tmp_path):
    # the default (no file configuration) entry has sag off in both
    # sections; a disabled SAG still writes the flag (like the Forge
    # extension), the client no longer sends an enabled flag
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "sag": {"scale": 1.25},
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = _meta(c, r.json()["job_id"])
        assert meta["sag_enabled"] == "False"
        assert meta["sag_scale"] == 1.25
        # values still default to the Forge sliders
        assert meta["sag_blur_sigma"] == 2.0
        assert meta["sag_threshold"] == 1.0


def test_sag_flag_follows_the_mode_section(tmp_path):
    """generate uses the t2i section, inpaint modes the i2i section."""
    with _client(tmp_path) as c:
        r = c.post("/api/model-configs", json={
            "name": "SAG t2i only", "model": MODEL,
            "t2i": {"sag": True},  # i2i stays False
        })
        cfg = r.json()
        with c.websocket_connect("/api/ws") as ws:
            c.post("/api/models/select", json={"id": cfg["id"]})
            # t2i section: enabled
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64, "sag": SAG,
            })
            _wait_done(ws, r.json()["job_id"])
            assert _meta(c, r.json()["job_id"])["sag_enabled"] == "True"
            # i2i section: disabled (same config, inpaint mode)
            _upload_canvas(c)
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64, "sag": SAG,
                "mode": "inpaint_not_masked",
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            assert _meta(c, r.json()["job_id"])["sag_enabled"] == "False"


def test_generate_without_sag_has_no_sag_metadata(tmp_path):
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64, "sag": None,
            })
            assert r.status_code == 200
            _wait_done(ws, r.json()["job_id"])
            meta = _meta(c, r.json()["job_id"])
        assert "sag_enabled" not in meta


def test_generate_rejects_bad_sag_values(tmp_path):
    # outside the Forge slider ranges -> 422
    bad = dict(SAG)
    bad["scale"] = 5.5
    with _client(tmp_path) as c:
        assert c.post("/api/generate", json={
            "prompt": "t", "width": 64, "height": 64, "sag": bad,
        }).status_code == 422
    bad = dict(SAG)
    bad["threshold"] = -0.1
    with _client(tmp_path) as c:
        assert c.post("/api/generate", json={
            "prompt": "t", "width": 64, "height": 64, "sag": bad,
        }).status_code == 422
    # a partial object: the Forge defaults fill in the missing fields
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "t", "width": 64, "height": 64,
                "sag": {"scale": 0.25},
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = _meta(c, r.json()["job_id"])
        assert meta["sag_scale"] == 0.25
        assert meta["sag_blur_sigma"] == 2.0
        assert meta["sag_threshold"] == 1.0


def test_ui_state_persists_sag_options(tmp_path):
    # the enabled flag is no longer a UI field: only the three options
    # persist in ui-state
    with _client(tmp_path) as c:
        r = c.post("/api/ui-state", json={
            "sag_scale": 0.25,
            "sag_blur_sigma": 4.5,
            "sag_threshold": 2.5,
        })
        assert r.json() == {"ok": True}
        state = c.get("/api/ui-state").json()["state"]
        assert "sag" not in state
        assert state["sag_scale"] == 0.25
        assert state["sag_blur_sigma"] == 4.5
        assert state["sag_threshold"] == 2.5


def test_ui_state_rejects_out_of_range_sag_values(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/api/ui-state", json={"sag_scale": -2.5}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_scale": 5.5}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_blur_sigma": 10.5}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_threshold": -0.1}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_threshold": 4.1}).status_code == 422
