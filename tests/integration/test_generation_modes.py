"""Phase 3 API: generation modes (generate / inpaint masked / inpaint
not masked) against the MockBackend.

The mock paints R = horizontal ramp, G = vertical ramp, B = seed constant
(plus small noise), so a replaced region is recognized by the wide R/G
spread, while untouched regions keep their exact fill value.
"""

import io

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


def test_generate_mode_replaces_canvas_and_keeps_mask(client):
    _set_canvas(client, 16, 16, 7)
    _mask_dot(client, 8, 8, 4)
    m0 = _mask(client)

    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws, mode="generate")

    img = _img(client)
    assert img.shape == (16, 16, 3)
    # the whole canvas is the fresh mock gradient, not the old flat fill
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150
    # the mask (same size) is retained
    np.testing.assert_array_equal(_mask(client), m0)


def test_generate_mode_default(client):
    _set_canvas(client)
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws)  # no mode -> generate
    assert _img(client).shape == (16, 16, 3)


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
    _set_canvas(client, 32, 32, 99)
    _mask_dot(client, 16, 16, 12)
    m0 = _mask(client)
    assert m0 is not None and m0.any()

    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws, width=32, height=32,
                  mode="inpaint_masked", denoise=0.75)

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
    raw generation."""
    _set_canvas(client, 32, 32, 99)
    _mask_dot(client, 16, 16, 12)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=32, height=32, mode="inpaint_masked")

    thumb = client.get(f"/api/thumb/{job['job_id']}/0.png")
    assert thumb.status_code == 200
    t = np.asarray(Image.open(io.BytesIO(thumb.content)))
    # identical to the auto-set canvas
    np.testing.assert_array_equal(t, _img(client))


# ----------------------------------------------------- inpaint not masked --


def test_inpaint_not_masked_empty_canvas_rejected(client):
    r = client.post("/api/generate",
                    json={"prompt": "x", "mode": "inpaint_not_masked"})
    assert r.status_code == 400
    assert r.json()["error"] == "canvas is empty"


def test_inpaint_not_masked_same_size_replaces_canvas(client):
    _set_canvas(client, 16, 16, 7)
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws, width=16, height=16, mode="inpaint_not_masked")

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
        _generate(client, ws, width=40, height=40, mode="inpaint_not_masked")

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


def test_uistate_persists_mode_and_denoise(client):
    r = client.post("/api/ui-state", json={
        "mode": "inpaint_not_masked", "denoise": 0.42})
    assert r.status_code == 200
    st = client.get("/api/ui-state").json()["state"]
    assert st["mode"] == "inpaint_not_masked"
    assert st["denoise"] == 0.42


def test_uistate_rejects_bad_mode_and_denoise(client):
    assert client.post("/api/ui-state", json={"mode": "bogus"}).status_code == 422
    assert client.post("/api/ui-state", json={"denoise": 1.5}).status_code == 422
    assert client.post("/api/ui-state", json={"denoise": -0.1}).status_code == 422


def test_generate_body_denoise_validated(client):
    _set_canvas(client)
    r = client.post("/api/generate",
                    json={"prompt": "x", "mode": "generate", "denoise": 3.0})
    assert r.status_code == 422
