"""End-to-end API tests using the MockBackend (no GPU needed).

Exercises the full Phase 1 flow: generate -> job -> WS events -> thumbnails
-> canvas auto-set -> save, plus interrupt semantics.
"""

import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


@pytest.fixture
def client(tmp_path, monkeypatch):
    # small delay per step so interrupt-mid-flight is testable
    backend = MockBackend(steps_delay=0.02)
    app = create_app(config=Config(root=tmp_path), backend=backend)
    with TestClient(app) as c:
        yield c


def _wait_for(client, ws, want: str, timeout: float = 10.0) -> dict:
    """Consume WS events (which also arrive as they are produced) until one
    with the given type is found."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.receive_json(mode="text")
        except Exception:
            # socket closed: nothing more is coming
            break
        if msg.get("type") == want:
            return msg
        if time.time() > deadline:
            break
    raise TimeoutError(f"no {want!r} event within {timeout}s")


def test_health_and_dirs(client, tmp_path):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    for d in ("models", "models/segmentation", "loras", "user", "user/brushes",
              "user/wildcards", "user/presets", "user/model-configs", "output"):
        assert (tmp_path / d).is_dir(), d


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "imagagen" in r.text


def test_models_endpoint(client):
    # the model selector lists configurations (Phase 4); with no files in
    # model-configs/ that is just the in-memory default of the mock model
    body = client.get("/api/models").json()
    # family rides along (derived from the model) so the UI can decide which
    # family-specific controls to show without a second fetch
    assert body["models"] == [{
        "id": "mock/mock.safetensors",
        "name": "Mock Model",
        "model_id": "mock/mock.safetensors",
        "family": "sdxl",
        "distilled": True,
        "t2i": {"sampler": None, "scheduler": None, "steps": None, "cfg": None, "loras": []},
        "i2i": {"sampler": None, "scheduler": None, "steps": None, "cfg": None, "loras": []},
        "is_default": True,
    }]
    assert body["current"] == "mock/mock.safetensors"


def test_generate_single_and_save(client, tmp_path):
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "Photo of a turtle", "width": 64, "height": 64,
            "steps": 4, "seed": 123,
        })
        job_id = r.json()["job_id"]
        assert isinstance(job_id, int)
        # the client uses server_id to invalidate its finished-job
        # bookkeeping after a server restart (job ids restart at 1) and
        # done to keep the button from flipping for an already-settled job
        assert isinstance(r.json()["server_id"], str)
        assert r.json()["server_id"]
        assert isinstance(r.json()["done"], bool)

        # thumbnail appears, then job done
        done = _wait_for(client, ws, "done")
        assert done["job"] == job_id

        # first image was auto-set onto the canvas
        canvas = client.get("/api/canvas.png")
        assert canvas.status_code == 200

        # thumbnail is retrievable and carries metadata
        thumb = client.get(f"/api/thumb/{job_id}/0.png")
        assert thumb.status_code == 200
        assert b"parameters" in thumb.content

        # save with the same prompt -> numbered output
        r = client.post("/api/save", json={"prompt": "Photo of a turtle"})
        assert r.json()["path"] == "1-photo-of-a-turtle.png"
        assert (tmp_path / "output" / "1-photo-of-a-turtle.png").is_file()


def test_generate_batch_seeds_and_completion_order(client, tmp_path):
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "batch", "width": 64, "height": 64, "steps": 2,
            "seed": 100, "batch_size": 3,
        })
        job_id = r.json()["job_id"]
        _wait_for(client, ws, "done")
        for i in range(3):
            thumb = client.get(f"/api/thumb/{job_id}/{i}.png")
            assert thumb.status_code == 200
        # seeds ascend from base — verify via metadata of each thumbnail
        from imagagen import metadata
        parsed_seeds = []
        for i in range(3):
            data = client.get(f"/api/thumb/{job_id}/{i}.png").content
            m = metadata.read_metadata(data)
            parsed_seeds.append(m["seed"])
        assert sorted(parsed_seeds) == [100, 101, 102]


def test_generate_missing_wildcard_rejected(client, tmp_path):
    # an unresolved wildcard is a submit-time error: a 400 with a message
    # (terminal + UI status line), and no job is queued
    r = client.post("/api/generate", json={
        "prompt": "a __nope__ day", "width": 64, "height": 64, "steps": 2,
    })
    assert r.status_code == 400
    assert "not found" in r.json()["error"]
    assert "job_id" not in r.json()


def test_generate_recursion_wildcard_rejected(client, tmp_path):
    (tmp_path / "user" / "wildcards").mkdir(parents=True, exist_ok=True)
    (tmp_path / "user" / "wildcards" / "loop.txt").write_text("spins __loop__", encoding="utf-8")
    r = client.post("/api/generate", json={
        "prompt": "a __loop__ day", "width": 64, "height": 64, "steps": 2,
    })
    assert r.status_code == 400
    assert "recursion" in r.json()["error"]


def test_generate_wildcard_resolved_in_metadata(client, tmp_path):
    from imagagen import metadata

    (tmp_path / "user" / "wildcards").mkdir(parents=True, exist_ok=True)
    (tmp_path / "user" / "wildcards" / "mood.txt").write_text("sunny\nstormy\n", encoding="utf-8")
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "a __mood__ day", "width": 64, "height": 64, "steps": 2,
            "seed": 5, "batch_size": 2,
        })
        job_id = r.json()["job_id"]
        assert isinstance(job_id, int)
        _wait_for(client, ws, "done")
        for i in range(2):
            m = metadata.read_metadata(client.get(f"/api/thumb/{job_id}/{i}.png").content)
            # the metadata stores the fully resolved prompt, not the raw one
            assert "__mood__" not in m["prompt"]
            assert m["prompt"] in ("a sunny day", "a stormy day")


def test_interrupt_drops_partial(client, tmp_path):
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "slow", "width": 64, "height": 64, "steps": 300, "seed": 1,
            "batch_size": 1,
        })
        job_id = r.json()["job_id"]
        # let it start running (steps_delay 0.02 * 300 = 6s total)
        _wait_for(client, ws, "progress", timeout=5)
        client.post("/api/interrupt", json={"job": job_id})
        ev = _wait_for(client, ws, "job_interrupted")
        assert ev["job"] == job_id
        time.sleep(0.3)
        # no thumbnail should exist
        assert client.get(f"/api/thumb/{job_id}/0.png").status_code == 404


def test_canvas_upload_replaces_current_image(client):
    """Uploads create a new image entity and evict the previous one —
    including any client edits that had been synced to it."""
    import io

    from PIL import Image

    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    r = client.post("/api/canvas/upload", content=buf.getvalue(),
                    headers={"content-type": "image/png"})
    assert r.json()["ok"] is True
    data = client.get("/api/canvas.png").content
    assert Image.open(io.BytesIO(data)).size == (8, 8)

    # a client edit is synced onto the first image (pixel (4, 4) painted red)
    edited = img.copy()
    edited[4, 4] = (255, 0, 0)
    buf_i, buf_m = io.BytesIO(), io.BytesIO()
    Image.fromarray(edited).save(buf_i, format="PNG")
    Image.new("RGBA", (8, 8), (128, 128, 128, 0)).save(buf_m, format="PNG")
    r = client.post("/api/canvas/sync",
                    files={"image": ("canvas.png", buf_i.getvalue(), "image/png"),
                           "mask": ("mask.png", buf_m.getvalue(), "image/png")})
    assert r.json()["ok"] is True
    arr = np.asarray(Image.open(io.BytesIO(client.get("/api/canvas.png").content)))
    assert arr[4, 4, 0] == 255

    # upload a second image: a NEW entity, the first (and its edit) is evicted
    img2 = np.zeros((8, 8, 3), dtype=np.uint8)
    buf2 = io.BytesIO()
    Image.fromarray(img2).save(buf2, format="PNG")
    client.post("/api/canvas/upload", content=buf2.getvalue())
    arr = np.asarray(Image.open(io.BytesIO(client.get("/api/canvas.png").content)))
    assert arr[4, 4, 0] == 0  # the synced edit is gone with the evicted entity


def test_save_empty_canvas_rejected(client):
    r = client.post("/api/save", json={"prompt": "x"})
    assert r.json() == {"error": "canvas is empty"}


def test_generate_validation(client):
    r = client.post("/api/generate", json={"prompt": "x", "width": "abc"})
    assert r.status_code == 422


def test_loras_family_filter(client, tmp_path):
    # discovery scans loras/<family>/; the endpoint filters by the
    # selected model's family (no parameter: all families, as before)
    for family, name in (("sdxl", "FooXL"), ("flux", "Bar1.0")):
        d = tmp_path / "loras" / family
        d.mkdir(parents=True, exist_ok=True)
        (d / (name + ".safetensors")).touch()
    assert client.get("/api/loras").json()["loras"] == ["Bar1.0", "FooXL"]
    assert client.get("/api/loras?family=sdxl").json()["loras"] == ["FooXL"]
    assert client.get("/api/loras?family=flux").json()["loras"] == ["Bar1.0"]
    assert client.get("/api/loras?family=nope").json()["loras"] == []


def test_generate_records_disabled_loras(client):
    """Disabled LoRAs are recorded in the image metadata (for import) but
    are NOT applied — the backend only receives the enabled ones."""
    from imagagen import metadata
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "lora test", "width": 64, "height": 64,
            "steps": 2, "seed": 1,
            "loras": [("alpha", 0.8)],
            "disabled_loras": [("beta", -0.3), ("gamma", 1.0)],
        })
        job_id = r.json()["job_id"]
        _wait_for(client, ws, "done")
        m = metadata.read_metadata(
            client.get(f"/api/thumb/{job_id}/0.png").content)
        assert m["loras"] == ["alpha@0.8"]
        assert m["disabled_loras"] == ["beta@-0.3", "gamma@1"]
    # ui-state round-trip
    r = client.post("/api/ui-state", json={
        "loras": [["alpha", 0.8]], "disabled_loras": [["beta", -0.3]],
    })
    assert r.status_code == 200
    st = client.get("/api/ui-state").json()["state"]
    assert st["disabled_loras"] == [["beta", -0.3]]
