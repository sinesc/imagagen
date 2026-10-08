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
    # paint a mask square centred on (x, y) — the client does the brush
    # work; the server only receives the synced state
    img = c.get("/api/canvas.png")
    assert img.status_code == 200
    w, h = Image.open(io.BytesIO(img.content)).size
    img_p = Image.open(io.BytesIO(img.content)).convert("RGB")
    mask = Image.new("RGBA", (w, h), (128, 128, 128, 0))
    px = mask.load()
    r = size // 2
    for yy in range(max(0, int(y - r)), min(h, int(y + r + 1))):
        for xx in range(max(0, int(x - r)), min(w, int(x + r + 1))):
            px[xx, yy] = (128, 128, 128, 255)
    buf_i, buf_m = io.BytesIO(), io.BytesIO()
    img_p.save(buf_i, format="PNG")
    mask.save(buf_m, format="PNG")
    r = c.post("/api/canvas/sync",
               files={"image": ("canvas.png", buf_i.getvalue(), "image/png"),
                      "mask": ("mask.png", buf_m.getvalue(), "image/png")})
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
    body = {"prompt": "test", "width": 64, "height": 64, "batch_size": 1}
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
    # same size as the generation, so the (retained) mask stays unchanged
    _set_canvas(client, 64, 64, 7)
    _mask_dot(client, 32, 32, 4)
    m0 = _mask(client)

    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="generate")

    img = _img(client)
    assert img.shape == (64, 64, 3)
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
        _generate(client, ws, width=64, height=64)
    img = _img(client)
    assert img.shape == (64, 64, 3)
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150


def test_generate_mode_default(client):
    _set_canvas(client)
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws)  # no mode -> generate
    # no mode specified: generate mode, which retains the existing canvas
    assert _img(client).shape == (24, 32, 3)


# ------------------------------------------------------------- rotation --
# The mock's R channel is a horizontal ramp over the REQUEST (generation)
# size; counter-rotating it back to the canvas turns the ramp's axis (and
# a 180° job mirrors it). The blend box in the final canvas lands back on
# the original mask box (rotation and counter-rotation cancel).


# (the API rejects sizes below 64, so the "64x48" sizes of the unit tests
# become 96x64 here)

def test_generate_rotation_90_swaps_ramp_axes(client):
    _set_canvas(client, 96, 64, 5)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="generate", width=96, height=64,
                        rotation=90)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (64, 96, 3)
    # the R ramp ran along the generation's width (64) and is now rotated
    # to run along the rows: a column slice is wide-spread, a row slice flat
    col = img[:, 48, 0]
    assert int(col.max()) - int(col.min()) > 150
    row = img[30, :, 0]
    # mock noise only (96 samples of sigma-12 noise: well under the
    # 255-wide ramp a wrong axis would show)
    assert int(row.max()) - int(row.min()) <= 100


def test_generate_rotation_45_fits_and_extends(client):
    # generation ran at 114x114 (ceil((96+64)/sqrt(2))); the canvas is
    # exactly the requested size
    _set_canvas(client, 96, 64, 5)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="generate", width=96, height=64,
                        rotation=45)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    assert _img(client).shape == (64, 96, 3)


def test_generate_rotation_180_flips(client):
    _set_canvas(client, 96, 64, 5)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="generate", width=96, height=64,
                        rotation=180)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    # the R ramp is mirrored vs a rotation-0 result (increasing-x ramp for
    # rotation 0, so the inequality is reversed here)
    assert int(img[32, 8, 0]) > int(img[32, 87, 0])


def test_inpaint_masked_rotation_90_replaces_upright(client):
    _set_canvas(client, 64, 64, 99)
    _mask_dot(client, 32, 24, size=24)  # rows 12:36, cols 20:44
    m0 = _mask(client)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="inpaint_masked", width=64,
                        height=64, rotation=90, mask_padding=0)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (64, 64, 3)
    # inside the mask box the R ramp runs along the ROWS (the mock's
    # horizontal ramp was counter-rotated to vertical)
    col = img[13:35, 32, 0]
    assert int(col.max()) - int(col.min()) > 40
    row = img[24, 21:43, 0]
    assert int(row.max()) - int(row.min()) <= 60  # mock noise only
    # the mask content outside the box (with a safety margin) is untouched
    box = np.ones(img.shape[:2], bool)
    box[8:40, 16:48] = False
    assert (img[box] == 99).all()
    # the mask is retained, unchanged
    np.testing.assert_array_equal(_mask(client), m0)


def test_inpaint_masked_rotation_180_flips_content(client):
    _set_canvas(client, 64, 64, 99)
    _mask_dot(client, 32, 24, size=24)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="inpaint_masked", width=64,
                        height=64, rotation=180, mask_padding=0)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    # the R ramp inside the mask box is mirrored (an increasing-x ramp for
    # rotation 0) — ~81 signal vs ~17 per-pixel noise
    assert int(img[24, 22, 0]) > int(img[24, 42, 0])
    box = np.ones(img.shape[:2], bool)
    box[8:40, 16:48] = False
    assert (img[box] == 99).all()


def test_inpaint_masked_rotation_mismatched_sizes_keeps_canvas_size(client):
    # regression: in inpaint_masked the final canvas is the source canvas
    # size, not the requested W×H; counter-rotating to the requested size
    # produced an image based on a small area of the source
    _set_canvas(client, 96, 64, 99)
    _mask_dot(client, 48, 32, size=24)  # rows 20:44, cols 36:60
    m0 = _mask(client)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="inpaint_masked", width=128,
                        height=96, rotation=90, mask_padding=0)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    # the canvas keeps its own size (96x64), not the requested 128x96
    assert img.shape == (64, 96, 3)
    # inside the mask box the R ramp runs along the ROWS (counter-rotated)
    col = img[21:43, 48, 0]
    assert int(col.max()) - int(col.min()) > 40
    # outside the box (with a safety margin) the fill is untouched
    box = np.ones(img.shape[:2], bool)
    box[16:48, 32:64] = False
    assert (img[box] == 99).all()
    # the mask is retained, unchanged
    np.testing.assert_array_equal(_mask(client), m0)


def test_inpaint_not_masked_rotation_90_outpaints_upright(client):
    _set_canvas(client, 32, 24, 5)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="inpaint_not_masked", width=96,
                        height=64, rotation=90)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (64, 96, 3)
    # the whole canvas is freshly generated, and the R ramp (which ran
    # along the generation's width of 64) now runs along the rows
    col = img[:, 48, 0]
    assert int(col.max()) - int(col.min()) > 150


def test_rotation_rejected(client):
    assert client.post("/api/generate",
                       json={"prompt": "x", "rotation": 30}).status_code == 422
    assert client.post("/api/generate",
                       json={"prompt": "x", "rotation": 185}).status_code == 422


def test_rotation_metadata_round_trips(client):
    _set_canvas(client, 96, 64, 5)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="generate", width=96, height=64,
                        rotation=90)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    meta = _canvas_meta(client)
    # the user's requested size, not the 64x96 generation size
    assert meta["width"] == 96 and meta["height"] == 64
    assert meta["rotation"] == 90
    # a rotation-0 job records no rotation key at all
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="generate", width=96, height=64,
                        rotation=0)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    meta = _canvas_meta(client)
    assert "rotation" not in meta


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
    _set_canvas(client, 64, 64, 99)
    _mask_dot(client, 32, 32, 12)
    m0 = _mask(client)
    assert m0 is not None and m0.any()

    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=64, height=64,
                        mode="inpaint_masked", denoise=0.75)

    # no auto-set: the canvas keeps the original image and mask
    assert (np.asarray(_img(client)) == 99).all()
    np.testing.assert_array_equal(_mask(client), m0)
    # selecting the result: only the masked region is replaced
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (64, 64, 3)
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
    _set_canvas(client, 64, 64, 99)
    _mask_dot(client, 32, 32, 12)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=64, height=64, mode="inpaint_masked")

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
            "prompt": "x", "width": 64, "height": 64,
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
            "prompt": "x", "width": 64, "height": 64, "batch_size": 1})
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
    _set_canvas(client, 64, 64, 7)
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=64, height=64, mode="inpaint_not_masked")

    # no auto-set: the canvas keeps the original image
    assert (np.asarray(_img(client)) == 7).all()
    # selecting the result replaces it
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (64, 64, 3)
    # the whole canvas is the fresh gradient
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150
    # no mask existed -> the result canvas has an empty mask
    m = _mask(client)
    assert m is None or not m.any()


def test_inpaint_not_masked_outpaint_resizes_canvas_and_mask(client):
    _set_canvas(client, 32, 32, 7)
    _mask_dot(client, 16, 16, 8)
    m0 = _mask(client)

    # 2x: the 32x32 canvas is scaled and centered on a 64x64 canvas
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, width=64, height=64, mode="inpaint_not_masked")

    # no auto-set: the canvas (and its size) are unchanged
    assert (np.asarray(_img(client)) == 7).all()
    # selecting the result resizes the canvas and the mask along with it
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    img = _img(client)
    assert img.shape == (64, 64, 3)
    # the canvas is the fresh generation at the new size (wide ramp)
    assert _r_spread(img, np.ones(img.shape[:2], bool)) > 150
    # the mask was scaled 2x along with the image
    m = _mask(client)
    assert m is not None and m.shape == (64, 64)
    assert m[32, 32] > 0  # the original mask center doubled
    # LANCZOS may soften edges, but the support area clearly grew
    assert (m > 128).sum() >= (m0 > 128).sum()


def test_masks_are_per_image_after_outpaint(client):
    """Regression for the global-mask bug: the mask belongs to each image.
    After an outpaint (which resizes the canvas and with it the inherited
    mask), selecting the source image (the green-bordered source thumbnail)
    again must show the source's own mask at its own size — not the
    result's resized mask."""
    _set_canvas(client, 32, 24, 7)
    _mask_dot(client, 16, 12, 8)
    m0 = _mask(client)
    assert m0 is not None and m0[12, 16] == 255

    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="inpaint_not_masked")
    # the result inherits the mask, scaled 2x and centered along with the
    # image (32x24 -> 64x48 at oy=8 on a 64x64 canvas: 12*2+8 = 32)
    r = client.post("/api/canvas/thumb",
                    json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    m1 = _mask(client)
    assert m1.shape == (64, 64) and m1[32, 32] == 255
    # back to the source image: its own mask at its own size, intact
    r = client.post("/api/canvas/select", json={"id": "upload:0"})
    assert r.status_code == 200
    m2 = _mask(client)
    assert m2 is not None and m2.shape == (24, 32)
    np.testing.assert_array_equal(m2, m0)


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


def test_client_edits_retain_canvas_metadata(client):
    # empty canvas: the first result is auto-set and carries its metadata
    with client.websocket_connect("/api/ws") as ws:
        _generate(client, ws, prompt="before strokes", mode="generate")
    # a synced client edit (pixel painted) must keep the PNG metadata
    data = client.get("/api/canvas.png").content
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.load()[4, 4] = (255, 0, 0)
    w, h = img.size
    buf_i, buf_m = io.BytesIO(), io.BytesIO()
    img.save(buf_i, format="PNG")
    Image.new("RGBA", (w, h), (128, 128, 128, 0)).save(buf_m, format="PNG")
    client.post("/api/canvas/sync",
                files={"image": ("canvas.png", buf_i.getvalue(), "image/png"),
                       "mask": ("mask.png", buf_m.getvalue(), "image/png")})
    meta = _canvas_meta(client)
    assert meta is not None
    assert meta["prompt"] == "before strokes"


# ------------------------------------------------- auto size + oversample --


def test_auto_size_sentinel_requires_inpaint(client):
    _set_canvas(client, 64, 64, 99)
    _mask_dot(client, 32, 32, size=16)
    r = client.post("/api/generate", json={
        "prompt": "t", "width": -1, "height": 64, "mode": "generate"})
    assert r.status_code == 400


def test_auto_size_sentinel_and_oversample(client):
    _set_canvas(client, 128, 128, 99)
    _mask_dot(client, 40, 40, size=16)  # mask bbox 16x16
    # -1/-1: mask bbox (16) rounded up to 16, clamped to the 64px minimum
    with client.websocket_connect("/api/ws") as ws:
        job = _generate(client, ws, mode="inpaint_masked", width=-1,
                        height=-1, mask_padding=0)
    r = client.post("/api/canvas/thumb", json={"job": job["job_id"], "index": 0})
    assert r.status_code == 200
    meta = _canvas_meta(client)
    assert (meta["width"], meta["height"]) == (64, 64)
    # oversample scales the user-specified size before the 16-round-up
    with client.websocket_connect("/api/ws") as ws:
        job2 = _generate(client, ws, mode="inpaint_masked", width=64,
                         height=64, mask_padding=0, oversample=2.0)
    r = client.post("/api/canvas/thumb", json={"job": job2["job_id"], "index": 0})
    assert r.status_code == 200
    meta2 = _canvas_meta(client)
    assert (meta2["width"], meta2["height"]) == (128, 128)
