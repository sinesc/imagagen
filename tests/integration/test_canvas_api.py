"""Phase 2 API: strokes, mask buffer + ops, brushes, mask.png."""

import io

import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _client(tmp_path, brush: str | None = None):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    if brush:
        img = Image.new("L", (8, 8), 200)
        img.save(tmp_path / "brushes" / f"{brush}.png")
    return TestClient(app)


def _png_bytes(w=32, h=24, value=10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (value, value, value)).save(buf, format="PNG")
    return buf.getvalue()


def _set_canvas(c, data=None):
    r = c.post("/api/canvas/upload", content=data or _png_bytes())
    assert r.status_code == 200


def _img(c) -> np.ndarray:
    r = c.get("/api/canvas.png")
    assert r.status_code == 200
    return np.asarray(Image.open(io.BytesIO(r.content)))


def _mask(c) -> np.ndarray | None:
    r = c.get("/api/mask.png")
    if r.status_code == 404:
        return None
    assert r.status_code == 200
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("L"))


def test_stroke_paints_image(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 12, "opacity": 1.0,
            "color": "#ff0000", "target": "image",
        })
        assert r.json() == {"ok": True, "undo": True, "redo": False}
        img = _img(c)
        assert img[12, 16][0] == 255 and img[12, 16][1] == 0
        assert img[0, 0][0] == 10  # untouched corner


def test_stroke_empty_canvas_400(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/canvas/stroke", json={"points": [[1.0, 1.0]], "size": 10})
        assert r.status_code == 400


def test_stroke_bad_values_422(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        assert c.post("/api/canvas/stroke", json={
            "points": [], "size": 10}).status_code == 422  # need >= 1 point
        assert c.post("/api/canvas/stroke", json={
            "points": [[1.0, 1.0]], "size": 0}).status_code == 422
        assert c.post("/api/canvas/stroke", json={
            "points": [[1.0, 1.0]], "opacity": 2.0}).status_code == 422
        assert c.post("/api/canvas/stroke", json={
            "points": [[1.0, 1.0]], "color": "#zzz"}).status_code == 422


def test_stroke_unknown_brush_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "brush": "nope"})
        assert r.status_code == 400  # KeyError from the brush loader


def test_mask_roundtrip(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        assert _mask(c) is None or (_mask(c) == 0).all()
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "target": "mask"})
        m = _mask(c)
        assert m is not None and m[12, 16] == 255
        # shift-draw erases
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "target": "mask",
            "erase": True})
        assert _mask(c)[12, 16] == 0


def test_mask_dilate_erode_clear(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 8, "target": "mask"})
        before = int((_mask(c) > 0).sum())

        r = c.post("/api/canvas/mask", json={"op": "dilate", "px": 4})
        assert r.json()["ok"] is True
        assert int((_mask(c) > 0).sum()) > before

        c.post("/api/canvas/mask", json={"op": "erode", "px": 4})
        c.post("/api/canvas/mask", json={"op": "clear"})
        assert (_mask(c) == 0).all()

        # clear on an already-empty mask: ok but no change, no undo entry
        r = c.post("/api/canvas/mask", json={"op": "clear"})
        assert r.json()["changed"] is False


def test_mask_op_validation(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/api/canvas/mask", json={"op": "flip"}).status_code == 422
        r = c.post("/api/canvas/mask", json={"op": "dilate", "px": 4})
        assert r.status_code == 400  # empty canvas


def test_brushes_endpoint(tmp_path):
    with _client(tmp_path, brush="mybrush") as c:
        r = c.get("/api/brushes")
        assert r.status_code == 200
        assert r.json()["brushes"] == ["circle", "mybrush"]


def test_stroke_with_custom_brush(tmp_path):
    with _client(tmp_path, brush="soft") as c:
        _set_canvas(c)
        r = c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 16, "brush": "soft",
            "color": "#ffffff"})
        assert r.status_code == 200
        img = _img(c)
        assert img[12, 16][0] > 150  # luminance-200 brush painted


def test_undo_redo_over_api_with_mask(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "color": "#00ff00",
            "target": "image"})
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "target": "mask"})
        assert _mask(c)[12, 16] == 255

        c.post("/api/canvas/undo")
        assert _mask(c)[12, 16] == 0  # mask stroke reverted
        assert _img(c)[12, 16][1] == 255  # image stroke intact

        c.post("/api/canvas/redo")
        assert _mask(c)[12, 16] == 255

        c.post("/api/canvas/clear")
        assert c.get("/api/canvas.png").status_code == 404
        c.post("/api/canvas/undo")  # back to the painted canvas
        assert _img(c)[12, 16][1] == 255


def test_mask_retained_across_same_size_image(tmp_path):
    """Spec: the mask is retained across image generations (e.g. the first
    result auto-set into the canvas, or a thumb selection)."""
    with _client(tmp_path) as c:
        _set_canvas(c)
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "target": "mask"})
        assert _mask(c)[12, 16] == 255
        _set_canvas(c)  # new image, same size -> mask survives
        assert _mask(c)[12, 16] == 255


def test_mask_reset_on_size_change(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "target": "mask"})
        assert _mask(c)[12, 16] == 255
        _set_canvas(c, _png_bytes(64, 48))  # different size -> fresh mask
        assert (_mask(c) == 0).all()
