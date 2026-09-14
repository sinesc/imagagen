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
    for d in ("models", "models/segmentation", "loras", "brushes", "presets",
              "model-configs", "output"):
        assert (tmp_path / d).is_dir(), d


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "imagagen" in r.text


def test_models_endpoint(client):
    # the model selector lists configurations (Phase 4); with no files in
    # model-configs/ that is just the in-memory default of the mock model
    body = client.get("/api/models").json()
    assert body["models"] == [{
        "id": "mock/mock.safetensors",
        "name": "Mock Model",
        "model_id": "mock/mock.safetensors",
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


def test_canvas_upload_creates_new_image_not_undo_step(client):
    """Uploads create a new image entity (and evict the previous one), they
    do NOT push an undo step: the undo stack is the per-image draw stack."""
    img = np.full((8, 8, 3), 200, dtype=np.uint8)
    buf = __import__("io").BytesIO()
    from PIL import Image

    Image.fromarray(img).save(buf, format="PNG")
    r = client.post("/api/canvas/upload", content=buf.getvalue(),
                    headers={"content-type": "image/png"})
    assert r.json()["ok"] is True
    data = client.get("/api/canvas.png").content
    assert Image.open(__import__("io").BytesIO(data)).size == (8, 8)

    # stroke the first image (its draw stack now has one step)
    client.post("/api/canvas/stroke", json={
        "points": [[4.0, 4.0]], "size": 4, "color": "#ff0000", "target": "image"})

    # upload a second image: a NEW entity, the first is evicted
    img2 = np.zeros((8, 8, 3), dtype=np.uint8)
    buf2 = __import__("io").BytesIO()
    Image.fromarray(img2).save(buf2, format="PNG")
    client.post("/api/canvas/upload", content=buf2.getvalue())
    assert Image.open(__import__("io").BytesIO(client.get("/api/canvas.png").content)) \
        .getpixel((0, 0)) == (0, 0, 0)
    # the second image's draw stack is empty (nothing to undo), and the
    # stroke on the evicted first image is gone
    r = client.post("/api/canvas/undo", json={"target": "draw"})
    assert r.json()["undo_draw"] is False
    arr = np.asarray(Image.open(__import__("io").BytesIO(client.get("/api/canvas.png").content)))
    assert arr[0, 0, 0] == 0


def test_save_empty_canvas_rejected(client):
    r = client.post("/api/save", json={"prompt": "x"})
    assert r.json() == {"error": "canvas is empty"}


def test_generate_validation(client):
    r = client.post("/api/generate", json={"prompt": "x", "width": "abc"})
    assert r.status_code == 422
