"""UI input state persistence: ui-state.json round-trip + restore on start.

The server serializes the web UI's form values (model, prompt, steps,
...) to <root>/user/ui-state.json and serves them back on the next boot, so
the UI picks up where the user left off.
"""

import json

from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def us(p):
    f = p / "user" / "ui-state.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


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
    assert not (us(tmp_path)).exists()


def test_save_writes_file(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/ui-state", json={
            "prompt": "a red panda", "steps": 30, "cfg": 6.5,
            "model": "models/sdxl/x.safetensors", "sampler": "DPM++ 2M",
        })
        assert r.json() == {"ok": True}
    on_disk = json.loads((us(tmp_path)).read_text())
    assert on_disk["prompt"] == "a red panda"
    assert on_disk["steps"] == 30
    assert on_disk["cfg"] == 6.5
    # unspecified fields keep their defaults
    assert on_disk["height"] == 1024


def test_restore_on_new_server(tmp_path):
    (us(tmp_path)).write_text(json.dumps({
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
        # cfg/steps of -2 are below the -1 sentinel: rejected
        assert c.post("/api/ui-state", json={"cfg": -2}).status_code == 422
        assert c.post("/api/ui-state", json={"steps": -2}).status_code == 422
        assert c.post("/api/ui-state", json={"batch": 0}).status_code == 422
        assert c.post("/api/ui-state", content=b"not json").status_code == 400
        assert c.post("/api/ui-state", content=b"[1,2,3]").status_code == 400
        # rejected bodies never write the file
        assert not (us(tmp_path)).exists()
        # -1 is the "use model default" sentinel for both cfg and steps (Phase 4)
        assert c.post("/api/ui-state", json={"steps": -1, "cfg": -1.0}).status_code == 200
        # empty body = reset to defaults
        assert c.post("/api/ui-state", content=b"").status_code == 200


def test_invalid_model_falls_back_to_existing(tmp_path):
    # a persisted model pointing at a deleted/renamed configuration must
    # not be served: the UI would show an invalid dropdown value and
    # generation would fail with "unknown model configuration"
    configs = tmp_path / "user" / "model-configs"
    configs.mkdir(parents=True)
    (configs / "my-model.json").write_text(json.dumps({
        "name": "My Model",
        "model": "models/sdxl/x.safetensors",
    }))
    (us(tmp_path)).write_text(json.dumps({"model": "gone"}))
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
        ids = {m["id"] for m in c.get("/api/models").json()["models"]}
    assert state["model"] != "gone"
    assert state["model"] in ids
    # a valid model id is kept as-is
    (us(tmp_path)).write_text(json.dumps({"model": "my-model"}))
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["model"] == "my-model"


def test_unknown_keys_ignored(tmp_path):
    (us(tmp_path)).write_text(json.dumps({
        "prompt": "ok", "bogus-key": 42,
    }))
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == "ok"


def test_bad_value_falls_back_to_defaults(tmp_path):
    # one invalid field invalidates the whole file (all-or-nothing),
    # so a bad field never yields a half-restored form
    (us(tmp_path)).write_text(json.dumps({
        "prompt": "ok", "steps": "nope",
    }))
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == ""
    assert state["steps"] == 25


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    (us(tmp_path)).write_text("{corrupt json")
    with _client(tmp_path) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == ""
    assert state["steps"] == 25


def test_disabled_saving_skips_file(tmp_path):
    """save-ui-state = false: POSTs are validated but never written to disk."""
    app = create_app(
        config=Config(root=tmp_path, save_ui_state=False),
        backend=MockBackend(),
    )
    with TestClient(app) as c:
        r = c.post("/api/ui-state", json={"prompt": "not persisted"})
        assert r.json() == {"ok": True}
        assert not (us(tmp_path)).exists()


def test_disabled_saving_keeps_state_in_memory(tmp_path):
    """save-ui-state = false only disables the disk write: a live server
    still serves the latest state, so a browser reload restores it."""
    app = create_app(
        config=Config(root=tmp_path, save_ui_state=False),
        backend=MockBackend(),
    )
    with TestClient(app) as c:
        r = c.post("/api/ui-state", json={"prompt": "live only", "steps": 33})
        assert r.json() == {"ok": True}
        assert not (us(tmp_path)).exists()
        # what a browser reload fetches: the in-memory state
        state = c.get("/api/ui-state").json()["state"]
        assert state["prompt"] == "live only"
        assert state["steps"] == 33


def test_disabled_saving_still_loads_existing_file(tmp_path):
    (us(tmp_path)).write_text(json.dumps({"prompt": "old"}))
    app = create_app(
        config=Config(root=tmp_path, save_ui_state=False),
        backend=MockBackend(),
    )
    with TestClient(app) as c:
        state = c.get("/api/ui-state").json()["state"]
    assert state["prompt"] == "old"


def test_sendbeacon_style_text_plain(tmp_path):
    """The tab-close path posts a text/plain Blob; content-type is ignored."""
    with _client(tmp_path) as c:
        r = c.post(
            "/api/ui-state",
            content=json.dumps({"prompt": "beacon"}),
            headers={"content-type": "text/plain"},
        )
        assert r.json() == {"ok": True}
    assert json.loads((us(tmp_path)).read_text())["prompt"] == "beacon"
