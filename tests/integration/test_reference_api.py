"""Reference-image API (phase 0): request plumbing, caption preview,
ui-state persistence.

The UI checks a thumbnail as "reference image"; /api/generate resolves the
canvas entity to pixels in-process (like the img2img base), the queue
passes it to the backend, and the MockBackend echoes a caption so the
round trip (request -> result -> PNG metadata) is testable without a GPU.
"""

import io
import time

import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from imagagen import metadata
from imagagen.app import create_app
from imagagen.backends import MOCK_REFERENCE_CAPTION, MockBackend
from imagagen.config import Config


def _png_bytes(w=64, h=48, v=120):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), v, dtype=np.uint8)).save(
        buf, format="PNG")
    return buf.getvalue()


def _client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    return TestClient(app)


def _upload(c, **kw):
    return c.post("/api/canvas/upload", content=_png_bytes(**kw)).json()


def _wait_done(c, job_id, timeout=10.0):
    # the mock queue is fast; poll the job store via /api/queue
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = c.get("/api/queue").json()["jobs"]
        if all(j["id"] != job_id or len(j["results"]) >= 1 for j in jobs):
            return
        time.sleep(0.02)
    raise TimeoutError


def test_reference_round_trip(tmp_path):
    with _client(tmp_path) as c:
        up = _upload(c)
        image_id = up["image_id"]
        r = c.post("/api/generate", json={
            "prompt": "x", "width": 64, "height": 48, "steps": 1,
            "seed": 5, "batch_size": 1, "reference_image_id": image_id})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        _wait_done(c, job_id)
        # the result PNG carries the echoed caption
        png = c.get(f"/api/thumb/{job_id}/0.png").content
        meta = metadata.read_metadata(png)
        assert meta["reference_caption"] == MOCK_REFERENCE_CAPTION
        # the mock caption is Overview + Subject + Summary; default = all
        assert meta["reference_sections"] == ["Overview", "Subject",
                                              "Summary"]


def test_reference_sections_subset(tmp_path):
    with _client(tmp_path) as c:
        image_id = _upload(c)["image_id"]
        r = c.post("/api/generate", json={
            "prompt": "x", "width": 64, "height": 48, "steps": 1,
            "seed": 5, "batch_size": 1, "reference_image_id": image_id,
            "reference_sections": ["Subject"]})
        job_id = r.json()["job_id"]
        _wait_done(c, job_id)
        png = c.get(f"/api/thumb/{job_id}/0.png").content
        meta = metadata.read_metadata(png)
        # the full caption is recorded; only the selected sections were
        # applied (the mock echoes the applied titles on the result)
        assert meta["reference_caption"] == MOCK_REFERENCE_CAPTION
        assert meta["reference_sections"] == ["Subject"]


def test_generate_without_reference_has_no_caption_line(tmp_path):
    with _client(tmp_path) as c:
        _upload(c)
        r = c.post("/api/generate", json={
            "prompt": "x", "width": 64, "height": 48, "steps": 1,
            "seed": 5, "batch_size": 1})
        job_id = r.json()["job_id"]
        _wait_done(c, job_id)
        png = c.get(f"/api/thumb/{job_id}/0.png").content
        meta = metadata.read_metadata(png)
        assert "reference_caption" not in meta


def test_generate_with_unknown_reference_is_rejected(tmp_path):
    with _client(tmp_path) as c:
        _upload(c)
        r = c.post("/api/generate", json={
            "prompt": "x", "width": 64, "height": 48, "steps": 1,
            "seed": 5, "batch_size": 1, "reference_image_id": "nope:0"})
        assert r.status_code == 400
        assert "reference" in r.json()["error"]


def test_caption_preview_endpoint(tmp_path):
    with _client(tmp_path) as c:
        image_id = _upload(c)["image_id"]
        r = c.post("/api/reference/caption", json={"image_id": image_id})
        assert r.status_code == 200
        assert r.json()["caption"] == MOCK_REFERENCE_CAPTION
        r = c.post("/api/reference/caption", json={"image_id": "gone:1"})
        assert r.status_code == 400


def test_ui_state_persists_reference_image_id(tmp_path):
    with _client(tmp_path) as c:
        image_id = _upload(c)["image_id"]
        r = c.post("/api/ui-state", json={"prompt": "p",
                                          "reference_image_id": image_id})
        assert r.status_code == 200
        state = c.get("/api/ui-state").json()["state"]
        assert state["reference_image_id"] == image_id
        # the file carries it too (survives a server restart)
        saved = (tmp_path / "user" / "ui-state.json").read_text()
        assert image_id in saved


def test_ui_state_drops_dangling_reference_image_id(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/ui-state", json={"prompt": "p",
                                          "reference_image_id": "gone:9"})
        assert r.status_code == 200
        state = c.get("/api/ui-state").json()["state"]
        assert state["reference_image_id"] is None


def test_ui_state_persists_reference_sections(tmp_path):
    with _client(tmp_path) as c:
        image_id = _upload(c)["image_id"]
        c.post("/api/ui-state", json={"prompt": "p",
                                      "reference_image_id": image_id,
                                      "reference_sections": ["Subject"]})
        state = c.get("/api/ui-state").json()["state"]
        assert state["reference_sections"] == ["Subject"]


def test_ui_state_drops_reference_sections_with_the_entity(tmp_path):
    with _client(tmp_path) as c:
        c.post("/api/ui-state", json={"prompt": "p",
                                      "reference_image_id": "gone:9",
                                      "reference_sections": ["Subject"]})
        state = c.get("/api/ui-state").json()["state"]
        assert state["reference_image_id"] is None
        assert state["reference_sections"] is None
