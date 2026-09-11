"""UI input state persistence: ui-state.json round-trip + restore on start.

The server serializes the web UI's form values (model, prompt, steps,
...) to <root>/ui-state.json and serves them back on the next boot, so
the UI picks up where the user left off.
"""

import json

from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    return TestClient(app)


def test_defaults_when_no_file(tmp_path):
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == ""
    assert state["steps"] == 25
    assert state["cfg"] == 7.0
    assert state["model"] is None
    assert not (tmp_path / "ui-state.json").exists()


def test_save_writes_file(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/ui-state", json={
            "prompt": "a red panda", "steps": 30, "cfg": 6.5,
            "model": "models/sdxl/x.safetensors", "sampler": "DPM++ 2M",
        })
        assert r.json() == {"ok": True}
    on_disk = json.loads((tmp_path / "ui-state.json").read_text())
    assert on_disk["prompt"] == "a red panda"
    assert on_disk["steps"] == 30
    assert on_disk["cfg"] == 6.5
    # unspecified fields keep their defaults
    assert on_disk["height"] == 1024


def test_restore_on_new_server(tmp_path):
    (tmp_path / "ui-state.json").write_text(json.dumps({
        "prompt": "kept across restart", "steps": 40, "seed": 777,
    }))
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == "kept across restart"
    assert state["steps"] == 40
    assert state["seed"] == 777


def test_invalid_values_rejected(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/api/ui-state", json={"steps": "abc"}).status_code == 422
        assert c.post("/api/ui-state", json={"width": 99999}).status_code == 422
        # cfg -1 is the "use model default" sentinel and valid (Phase 4)
        assert c.post("/api/ui-state", json={"cfg": -2}).status_code == 422
        assert c.post("/api/ui-state", json={"batch": 0}).status_code == 422
        assert c.post("/api/ui-state", content=b"not json").status_code == 400
        assert c.post("/api/ui-state", content=b"[1,2,3]").status_code == 400
        # rejected bodies never write the file
        assert not (tmp_path / "ui-state.json").exists()
        # empty body = reset to defaults
        assert c.post("/api/ui-state", content=b"").status_code == 200


def test_unknown_keys_ignored(tmp_path):
    (tmp_path / "ui-state.json").write_text(json.dumps({
        "prompt": "ok", "bogus-key": 42,
    }))
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == "ok"


def test_bad_value_falls_back_to_defaults(tmp_path):
    # one invalid field invalidates the whole file (all-or-nothing),
    # so a bad field never yields a half-restored form
    (tmp_path / "ui-state.json").write_text(json.dumps({
        "prompt": "ok", "steps": "nope",
    }))
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == ""
    assert state["steps"] == 25


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    (tmp_path / "ui-state.json").write_text("{corrupt json")
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == ""
    assert state["steps"] == 25


def test_sendbeacon_style_text_plain(tmp_path):
    """The tab-close path posts a text/plain Blob; content-type is ignored."""
    with _client(tmp_path) as c:
        r = c.post(
            "/api/ui-state",
            content=json.dumps({"prompt": "beacon"}),
            headers={"content-type": "text/plain"},
        )
        assert r.json() == {"ok": True}
    assert json.loads((tmp_path / "ui-state.json").read_text())["prompt"] == "beacon"
