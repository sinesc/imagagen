"""Self attention guidance (Forge builtin extension) through the API,
against the MockBackend: the enabled flag AND the option values are
model-configuration settings (per t2i/i2i); the generate body carries no
sag fields, and the generated PNG metadata carries the resolved
sag_enabled flag plus Forge's sag_* option keys unconditionally.
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


T2I = {"sag": True, "sag_scale": 0.75, "sag_blur_sigma": 3.0,
       "sag_threshold": 1.5}
I2I = {"sag": False, "sag_scale": -0.5, "sag_blur_sigma": 8.0,
       "sag_threshold": 4.0}


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
    the metadata, with the configuration's option values (the body
    carries no sag fields at all)."""
    with _client(tmp_path) as c:
        r = c.post("/api/model-configs", json={
            "name": "SAG on", "model": MODEL, "t2i": T2I,
        })
        assert r.status_code == 200, r.text
        cfg = r.json()
        assert cfg["t2i"]["sag"] is True and cfg["i2i"]["sag"] is False
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/models/select", json={"id": cfg["id"]})
            assert r.status_code == 200, r.text
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
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
    # sections with the Forge option defaults
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = _meta(c, r.json()["job_id"])
        assert meta["sag_enabled"] == "False"
        # the Forge option defaults
        assert meta["sag_scale"] == 0.5
        assert meta["sag_blur_sigma"] == 2.0
        assert meta["sag_threshold"] == 1.0


def test_sag_flag_and_options_follow_the_mode_section(tmp_path):
    """generate uses the t2i section, inpaint modes the i2i section --
    for the flag AND the option values."""
    with _client(tmp_path) as c:
        r = c.post("/api/model-configs", json={
            "name": "SAG t2i only", "model": MODEL, "t2i": T2I, "i2i": I2I,
        })
        cfg = r.json()
        with c.websocket_connect("/api/ws") as ws:
            c.post("/api/models/select", json={"id": cfg["id"]})
            # t2i section: enabled, t2i options
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
            })
            _wait_done(ws, r.json()["job_id"])
            m = _meta(c, r.json()["job_id"])
            assert m["sag_enabled"] == "True"
            assert (m["sag_scale"], m["sag_blur_sigma"], m["sag_threshold"]) == \
                (0.75, 3.0, 1.5)
            # i2i section: disabled, i2i options
            _upload_canvas(c)
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "mode": "inpaint_not_masked",
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            m = _meta(c, r.json()["job_id"])
            assert m["sag_enabled"] == "False"
            assert (m["sag_scale"], m["sag_blur_sigma"], m["sag_threshold"]) == \
                (-0.5, 8.0, 4.0)


def test_model_config_rejects_bad_sag_option_values(tmp_path):
    # outside the Forge slider ranges -> 422
    bad = dict(T2I, sag_scale=5.5)
    with _client(tmp_path) as c:
        assert c.post("/api/model-configs", json={
            "name": "Bad", "model": MODEL, "t2i": bad,
        }).status_code == 422
    bad = dict(T2I, sag_threshold=-0.1)
    with _client(tmp_path) as c:
        assert c.post("/api/model-configs", json={
            "name": "Bad", "model": MODEL, "t2i": bad,
        }).status_code == 422
    bad = dict(T2I, sag_blur_sigma=10.5)
    with _client(tmp_path) as c:
        assert c.post("/api/model-configs", json={
            "name": "Bad", "model": MODEL, "t2i": bad,
        }).status_code == 422
