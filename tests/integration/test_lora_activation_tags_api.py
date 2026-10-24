"""LoRA activation-text tags: request filtering, PNG metadata, ui-state.

A LoRA's "activation text" is comma separated; the UI shows one toggle per
tag under the negative prompt and sends the switched-off ones as
``disabled_activation_tags``. The remaining tags still go into the prompt,
the applied ones are recorded in the PNG, and the selection round-trips
through the ui-state.
"""

import json
import time

from starlette.testclient import TestClient

from imagagen import metadata
from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _setup(tmp_path):
    config = Config(root=tmp_path)
    d = config.loras_dir / "sdxl"
    d.mkdir(parents=True)
    for name in ("alpha", "beta"):
        (d / f"{name}.safetensors").write_bytes(b"")
    (d / "alpha.json").write_text(json.dumps({
        "activation text": "blue eyes, freckles",
        "negative text": "blurry",
    }))
    (d / "beta.json").write_text(json.dumps({
        "activation text": "freckles, 8k",
    }))
    return TestClient(create_app(config=config, backend=MockBackend()))


def _wait_done(c, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = c.get("/api/queue").json()["jobs"]
        if all(j["id"] != job_id or len(j["results"]) >= 1 for j in jobs):
            return
        time.sleep(0.02)
    raise TimeoutError


def _generate(c, **kw):
    body = {"prompt": "a cat", "width": 64, "height": 48, "steps": 1,
            "seed": 5, "batch_size": 1,
            "loras": [["alpha", 1.0], ["beta", 0.6]]}
    body.update(kw)
    r = c.post("/api/generate", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    _wait_done(c, job_id)
    return metadata.read_metadata(
        c.get(f"/api/thumb/{job_id}/0.png").content)


def test_tags_of_the_selected_loras_are_recorded(tmp_path):
    with _setup(tmp_path) as c:
        # no selection: every tag applies (deduped across LoRAs)
        meta = _generate(c)
        assert meta["activation_tags"] == ["blue eyes", "freckles", "8k"]


def test_disabled_tags_are_dropped_from_the_prompt(tmp_path):
    with _setup(tmp_path) as c:
        # the token counter applies the prompt exactly as generation does,
        # so switching a tag off must change the count it reports
        base = c.post("/api/prompt-tokens", json={
            "prompt": "a cat", "loras": [["alpha", 1.0]]}).json()
        fewer = c.post("/api/prompt-tokens", json={
            "prompt": "a cat", "loras": [["alpha", 1.0]],
            "disabled_activation_tags": ["blue eyes", "freckles"]}).json()
        assert fewer["tokens"] < base["tokens"]

        meta = _generate(c, disabled_activation_tags=["freckles"])
        assert meta["activation_tags"] == ["blue eyes", "8k"]


def test_all_tags_off_records_nothing(tmp_path):
    with _setup(tmp_path) as c:
        meta = _generate(c, disabled_activation_tags=[
            "blue eyes", "freckles", "8k"])
        assert "activation_tags" not in meta


def test_ui_state_round_trip(tmp_path):
    with _setup(tmp_path) as c:
        r = c.post("/api/ui-state", json={
            "loras": [["alpha", 1.0]],
            "disabled_activation_tags": ["freckles"]})
        assert r.status_code == 200
        state = c.get("/api/ui-state").json()["state"]
        assert state["disabled_activation_tags"] == ["freckles"]
        # an old state file without the key still loads (defaults to nothing
        # switched off)
        c.post("/api/ui-state", json={"loras": [["alpha", 1.0]]})
        state = c.get("/api/ui-state").json()["state"]
        assert state["disabled_activation_tags"] == []
