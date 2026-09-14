"""Phase 3 API: generation modes (generate / inpaint masked / inpaint
not masked) against the MockBackend.

The mock paints R = horizontal ramp, G = vertical ramp, B = seed constant
(plus small noise), so a replaced region is recognized by the wide R/G
spread, while untouched regions keep their exact fill value.
"""

import io
import json

import numpy as np
import pytest
from PIL import Image
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


@pytest.fixture
def client(tmp_path):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    with TestClient(app) as c:
        yield c


def _wait_for(ws, want: str, timeout: float = 10.0) -> dict:
    """Consume WS events until one with the given type arrives."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            msg = ws.receive_json(mode="text")
        except Exception:
            break  # socket closed: nothing more is coming
        if msg.get("type") == want:
            return msg
    raise TimeoutError(f"no {want!r} event within {timeout}s")


def _png_bytes(w=32, h=24, value=10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (value, value, value)).save(buf, format="PNG")
    return buf.getvalue()


def _set_canvas(c, w=32, h=24, value=10):
    r = c.post("/api/canvas/upload", content=_png_bytes(w, h, value))
    assert r.status_code == 200


def _mask_dot(c, x: float, y: float, size: float = 8.0):
    r = c.post("/api/canvas/stroke", json={
        "points": [[x, y]], "size": size, "target": "mask"})
    assert r.status_code == 200, r.text


def _img(c) -> np.ndarray:
    r = c.get("/api/canvas.png")
    assert r.status_code == 200
    return np.asarray(Image.open(io.BytesIO(r.content)))


def _mask(c) -> np.ndarray | None:
    r = c.get("/api/mask.png")
    if r.status_code == 404:
        return None
    return np.asarray(Image.open(io.BytesIO(r.content)))


def _generate(c, ws, **kw) -> dict:
    body = {"prompt": "test", "width": 16, "height": 16, "batch_size": 1}
    body.update(kw)
    r = c.post("/api/generate", json=body)
    assert r.status_code == 200, r.text
    job = r.json()
    done = _wait_for(ws, "done")
    assert done["job"] == job["job_id"]
    return job


def _r_spread(img: np.ndarray, sel: np.ndarray) -> int:
    """R-channel max-min over the selected pixels: the mock's horizontal
    ramp makes any freshly generated region wide-spread, while a flat fill
    is (nearly) 0."""
    vals = img[sel, 0]
    return int(vals.max()) - int(vals.min()) if vals.size else 0


# ---------------------------------------------------------------- generate --


def test_generate_mode_retains_canvas_until_thumb_selected(client):
    """Generate on a non-empty canvas does NOT auto-replace it: the
    displayed image is retained (and published as the green-bordered
    source thumbnail) until the user selects a result thumbnail."""
    _set_canvas(client, 16, 16, 7)
    _mask_dot(client, 8, 8, 4)
    m0 = _mask(client)

    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="generate")

    img = _img(client)
    assert img.shape == (16, 16, 3)
    # the old flat fill is retained, mask untouched
    assert (img == 7).all()
    np.testing.assert_array_equal(_mask(client), m0)
    # selecting a result thumbnail puts it on the canvas (mask retained)
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150
    np.testing.assert_array_equal(_mask(client), m0)


def test_generate_mode_empty_canvas_auto_sets(client):
    # generate on an EMPTY canvas still auto-sets the first result
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws, width=16, height=16)
    img = _img(client)
    assert img.shape == (16, 16, 3)
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150


def test_generate_mode_default(client):
    _set_canvas(client)
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws)  # no mode -> generate
    # no mode specified: generate mode, which retains the existing canvas
    assert _img(client).shape == (24, 32, 3)


# ---------------------------------------------------------- inpaint masked --


def test_inpaint_masked_empty_mask_rejected(client):
    _set_canvas(client)
    r = client.post("/api/generate", json={"prompt": "x", "mode": "inpaint_masked"})
    assert r.status_code == 400
    assert r.json()["error"] == "mask is empty"


def test_inpaint_masked_empty_canvas_rejected(client):
    r = client.post("/api/generate", json={"prompt": "x", "mode": "inpaint_masked"})
    assert r.status_code == 400
    assert r.json()["error"] == "canvas is empty"


def test_inpaint_masked_replaces_only_mask_region(client):
    """The canvas is retained until a result thumbnail is selected; then
    only the masked region is replaced."""
    _set_canvas(client, 32, 32, 99)
    _mask_dot(client, 16, 16, 12)
    m0 = _mask(client)
    assert m0 is not None and m0.any()

    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=32, height=32,
                        mode="inpaint_masked", denoise=0.75)

    # no auto-set: the canvas keeps the original image and mask
    assert (np.asarray(_img(client)) == 99).all()
    np.testing.assert_array_equal(_mask(client), m0)
    # selecting the result: only the masked region is replaced
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (32, 32, 3)
    # inside the (fully masked) region: freshly generated gradient
    full = m0 == 255
    assert _r_spread(img, full) > 40
    # outside the mask: the original fill, untouched and exact
    assert (img[m0 == 0] == 99).all()
    # the mask is retained, unchanged
    np.testing.assert_array_equal(_mask(client), m0)


def test_inpaint_masked_thumbnail_is_blended_canvas(client):
    """Thumbnails show the post-processed canvas state (blended), not the
    raw generation: selecting one puts exactly that state on the canvas."""
    _set_canvas(client, 32, 32, 99)
    _mask_dot(client, 16, 16, 12)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=32, height=32, mode="inpaint_masked")

    thumb = client.get(f"/api/thumb/{job['job_id']}/0.png")
    assert thumb.status_code == 200
    t = np.asarray(Image.open(io.BytesIO(thumb.content)))
    client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    # identical to the selected canvas
    np.testing.assert_array_equal(t, _img(client))


def test_inpaint_source_thumbnail(client):
    """Inpaint jobs publish the source image snapshot (the UI shows it
    with a green border so the user can go back to it)."""
    _set_canvas(client, 16, 16, 55)
    _mask_dot(client, 8, 8, 6)
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "x", "width": 16, "height": 16,
            "mode": "inpaint_masked", "batch_size": 1})
        job = r.json()["job_id"]
        src = _wait_for(ws, "source_image")
        assert src["job"] == job
        assert src["url"] == f"/api/thumb/{job}/source.png"
        _wait_for(ws, "done")
    r = client.get(f"/api/thumb/{job}/source.png")
    assert r.status_code == 200
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(r.content))),
        np.full((16, 16, 3), 55, dtype=np.uint8))


def test_generate_source_thumbnail_with_canvas(client):
    """A generate job on a non-empty canvas publishes the canvas image as
    the source thumbnail (green border in the UI); the canvas is retained
    until a result is selected."""
    _set_canvas(client, 16, 16, 55)
    with client.websocket_connect("/api/ws") as ws:
        r = client.post("/api/generate", json={
            "prompt": "x", "width": 16, "height": 16, "batch_size": 1})
        job = r.json()["job_id"]
        src = _wait_for(ws, "source_image")
        assert src["job"] == job
        assert src["url"] == f"/api/thumb/{job}/source.png"
        _wait_for(ws, "done")
    r = client.get(f"/api/thumb/{job}/source.png")
    assert r.status_code == 200
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(r.content))),
        np.full((16, 16, 3), 55, dtype=np.uint8))


def test_generate_empty_canvas_has_no_source_thumbnail(client):
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws)
    assert client.get(f"/api/thumb/{job['job_id']}/source.png").status_code == 404


# ----------------------------------------------------- inpaint not masked --


def test_inpaint_not_masked_empty_canvas_rejected(client):
    r = client.post("/api/generate",
                    json={"prompt": "x", "mode": "inpaint_not_masked"})
    assert r.status_code == 400
    assert r.json()["error"] == "canvas is empty"


def test_inpaint_not_masked_same_size_replaces_canvas(client):
    _set_canvas(client, 16, 16, 7)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=16, height=16, mode="inpaint_not_masked")

    # no auto-set: the canvas keeps the original image
    assert (np.asarray(_img(client)) == 7).all()
    # selecting the result replaces it
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (16, 16, 3)
    # the whole canvas is the fresh gradient
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150
    # no mask existed -> the result canvas has an empty mask
    m = _mask(client)
    assert m is None or not m.any()


def test_inpaint_not_masked_outpaint_resizes_canvas_and_mask(client):
    _set_canvas(client, 20, 20, 7)
    _mask_dot(client, 10, 10, 8)
    m0 = _mask(client)

    # 2x: the 20x20 canvas is scaled and centered on a 40x40 canvas
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=40, height=40, mode="inpaint_not_masked")

    # no auto-set: the canvas (and its size) are unchanged
    assert (np.asarray(_img(client)) == 7).all()
    # selecting the result resizes the canvas and the mask along with it
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (40, 40, 3)
    # the canvas is the fresh generation at the new size (wide ramp)
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150
    # the mask was scaled 2x along with the image
    m = _mask(client)
    assert m is not None and m.shape == (40, 40)
    assert m[20, 20] > 0  # the original mask center doubled
    # LANCZOS may soften edges, but the support area clearly grew
    assert (m > 128).sum() >= (m0 > 128).sum()


def test_inpaint_unknown_mode_rejected(client):
    r = client.post("/api/generate", json={"prompt": "x", "mode": "bogus"})
    assert r.status_code == 400


# ------------------------------------------------------------ state + ui --


def test_uistate_persists_denoise_but_not_mode_or_preset(client):
    # the generation mode and the preset selection are never persisted
    # (the UI always opens in "generate" mode with the preset dropdown at
    # "—"); old-file keys are simply ignored
    r = client.post("/api/ui-state", json={
        "mode": "inpaint_not_masked", "denoise": 0.42, "preset": "Some preset"})
    assert r.status_code == 200
    st = client.get("/api/ui-state").json()["state"]
    assert "mode" not in st
    assert "preset" not in st
    assert st["denoise"] == 0.42
    on_disk = json.loads(client.app.state.ui_state.path.read_text())
    assert "mode" not in on_disk
    assert "preset" not in on_disk


def test_uistate_rejects_bad_denoise(client):
    assert client.post("/api/ui-state", json={"denoise": 1.5}).status_code == 422
    assert client.post("/api/ui-state", json={"denoise": -0.1}).status_code == 422
    # unknown keys (like an old "mode") are ignored, not rejected
    assert client.post("/api/ui-state", json={"mode": "bogus"}).status_code == 200


def test_generate_body_denoise_validated(client):
    _set_canvas(client)
    r = client.post("/api/generate",
                    json={"prompt": "x", "mode": "generate", "denoise": 3.0})
    assert r.status_code == 422


# ---------------------------------------------------- canvas metadata --
# Regression: the queue auto-sets the canvas from raw arrays; the result's
# PNG text chunks must be carried over, or "Save" writes a metadata-less
# PNG (which then yields nothing on drag&drop import).

def _canvas_meta(client) -> dict | None:
    from imagagen import metadata as md

    r = client.get("/api/canvas.png")
    assert r.status_code == 200
    return md.read_metadata(r.content)


def test_generate_carries_metadata_to_canvas_and_save(client, tmp_path):
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws, prompt="a test prompt", mode="generate")
    meta = _canvas_meta(client)
    assert meta is not None
    assert meta["prompt"] == "a test prompt"

    r = client.post("/api/save", json={"prompt": "a test prompt"})
    assert r.status_code == 200
    saved = (tmp_path / "output" / r.json()["path"]).read_bytes()
    from imagagen import metadata as md
    meta = md.read_metadata(saved)
    assert meta is not None
    assert meta["prompt"] == "a test prompt"


def test_inpaint_carries_metadata_to_canvas(client):
    _set_canvas(client)
    _mask_dot(client, 8, 8, 4)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, prompt="painted over", mode="inpaint_masked")
    # selecting the result puts its metadata onto the canvas
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    meta = _canvas_meta(client)
    assert meta is not None
    assert meta["prompt"] == "painted over"


def test_strokes_retain_canvas_metadata(client):
    # empty canvas: the first result is auto-set and carries its metadata
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws, prompt="before strokes", mode="generate")
    client.post("/api/canvas/stroke", json={
        "points": [[4.0, 4.0]], "size": 4, "target": "image"})
    meta = _canvas_meta(client)
    assert meta is not None
    assert meta["prompt"] == "before strokes"
