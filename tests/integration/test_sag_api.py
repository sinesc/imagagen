"""Self attention guidance (Forge builtin extension) through the API,
against the MockBackend: the request body carries the three options (null
= off; the mock ignores them), the generated PNG metadata carries Forge's
sag_* extra-parameter lines, and ui-state persists the checkbox + the
three options with the Forge slider limits.
"""

import json
import time

from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    return TestClient(app)


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


def test_generate_with_sag_roundtrips_metadata(tmp_path):
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64, "sag": SAG,
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            # the result PNG's metadata carries Forge's SAG extra
            # parameters
            thumb = c.get("/api/thumb/1/0.png")
            assert thumb.status_code == 200
            meta = c.post("/api/png-meta", content=thumb.content).json()["meta"]
        assert meta["sag_enabled"] == "True"
        assert meta["sag_scale"] == "0.75"
        assert meta["sag_blur_sigma"] == "3"
        assert meta["sag_threshold"] == "1.5"


def test_generate_with_sag_disabled_writes_flag_false(tmp_path):
    # a disabled SAG still writes the flag (like the Forge extension), so
    # dropping such an image can switch the UI checkbox back off
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64,
                "sag": {"enabled": False, "scale": 1.25},
            })
            assert r.status_code == 200, r.text
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
        assert meta["sag_enabled"] == "False"
        assert meta["sag_scale"] == "1.25"
        # values still default to the Forge sliders
        assert meta["sag_blur_sigma"] == "2"
        assert meta["sag_threshold"] == "1"


def test_generate_without_sag_has_no_sag_metadata(tmp_path):
    with _client(tmp_path) as c:
        with c.websocket_connect("/api/ws") as ws:
            r = c.post("/api/generate", json={
                "prompt": "test", "width": 64, "height": 64, "sag": None,
            })
            assert r.status_code == 200
            _wait_done(ws, r.json()["job_id"])
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
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
            meta = c.post("/api/png-meta",
                          content=c.get("/api/thumb/1/0.png").content).json()["meta"]
            assert meta["sag_scale"] == "0.25"
            assert meta["sag_blur_sigma"] == "2"
            assert meta["sag_threshold"] == "1"


def test_ui_state_persists_sag_fields(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/ui-state", json={
            "sag": True,
            "sag_scale": 0.25,
            "sag_blur_sigma": 4.5,
            "sag_threshold": 2.5,
        })
        assert r.json() == {"ok": True}
        state = c.get("/api/ui-state").json()["state"]
        assert state["sag"] is True
        assert state["sag_scale"] == 0.25
        assert state["sag_blur_sigma"] == 4.5
        assert state["sag_threshold"] == 2.5
    on_disk = json.loads((tmp_path / "user" / "ui-state.json").read_text())
    assert on_disk["sag"] is True
    # unspecified options keep the Forge defaults
    assert on_disk["sag_scale"] == 0.25


def test_ui_state_rejects_out_of_range_sag_values(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/api/ui-state", json={"sag_scale": -2.5}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_scale": 5.5}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_blur_sigma": 10.5}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_threshold": -0.1}).status_code == 422
        assert c.post("/api/ui-state", json={"sag_threshold": 4.1}).status_code == 422
