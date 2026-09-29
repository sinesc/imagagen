"""Phase 2 API: strokes, mask buffer + ops, brushes, mask.png."""

import io

import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from imagagen.app import MAX_BRUSH_SIZE, MAX_BRUSH_SIZE_ZOOMED, create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _client(tmp_path, brush: str | None = None):
    app = create_app(config=Config(root=tmp_path), backend=MockBackend())
    if brush:
        bdir = tmp_path / "user" / "brushes"
        bdir.mkdir(parents=True, exist_ok=True)
        img = Image.new("L", (8, 8), 200)
        img.save(bdir / f"{brush}.png")
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
        # the draw stack now has one step; the mask stack is untouched
        assert r.json() == {"ok": True, "undo_mask": False, "redo_mask": False,
                            "undo_draw": True, "redo_draw": False}
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
        # size boundary (must stay in sync with the frontend slider, app.js;
        # zoomed bounds cover "screen" mode: size/zoom, see MAX_ZOOM in app.py)
        assert c.post("/api/canvas/stroke", json={
            "points": [[1.0, 1.0]], "size": MAX_BRUSH_SIZE}).status_code == 200
        assert c.post("/api/canvas/stroke", json={
            "points": [[1.0, 1.0]],
            "size": MAX_BRUSH_SIZE_ZOOMED}).status_code == 200
        assert c.post("/api/canvas/stroke", json={
            "points": [[1.0, 1.0]],
            "size": MAX_BRUSH_SIZE_ZOOMED + 1}).status_code == 422


def test_stroke_unknown_brush_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "brush": "nope"})
        assert r.status_code == 400  # KeyError from the brush loader


def test_clone_paints_image(tmp_path):
    with _client(tmp_path) as c:
        left, right = (123, 45, 67), (10, 20, 30)
        arr = np.zeros((48, 64, 3), np.uint8)
        arr[:, :32] = left
        arr[:, 32:] = right
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        _set_canvas(c, buf.getvalue())
        r = c.post("/api/canvas/clone", json={
            "points": [[44.0, 24.0]], "offset": [-30.0, 0.0],
            "size": 20, "opacity": 1.0})
        # a clone stroke is a draw step; the mask stack is untouched
        assert r.json() == {"ok": True, "undo_mask": False, "redo_mask": False,
                            "undo_draw": True, "redo_draw": False}
        img = _img(c)
        assert img[24, 44].tolist() == list(left)  # copied from the left band
        assert img[0, 0].tolist() == list(left)  # untouched corner


def test_clone_empty_canvas_400(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/canvas/clone", json={
            "points": [[1.0, 1.0]], "offset": [1.0, 1.0]})
        assert r.status_code == 400


def test_clone_bad_values_422(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        assert c.post("/api/canvas/clone", json={
            "points": [], "offset": [1.0, 1.0]}).status_code == 422
        assert c.post("/api/canvas/clone", json={
            "points": [[1.0, 1.0]], "offset": [1.0]}).status_code == 422
        assert c.post("/api/canvas/clone", json={
            "points": [[1.0, 1.0]], "offset": [1.0, 1.0, 1.0]}).status_code == 422
        assert c.post("/api/canvas/clone", json={
            "points": [[1.0, 1.0]], "offset": [1.0, 1.0],
            "size": 0}).status_code == 422
        assert c.post("/api/canvas/clone", json={
            "points": [[1.0, 1.0]], "offset": [1.0, 1.0],
            "opacity": 2.0}).status_code == 422
        # size boundary (same as /api/canvas/stroke)
        assert c.post("/api/canvas/clone", json={
            "points": [[1.0, 1.0]], "offset": [1.0, 1.0],
            "size": MAX_BRUSH_SIZE_ZOOMED}).status_code == 200
        assert c.post("/api/canvas/clone", json={
            "points": [[1.0, 1.0]], "offset": [1.0, 1.0],
            "size": MAX_BRUSH_SIZE_ZOOMED + 1}).status_code == 422


def test_clone_unknown_brush_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/canvas/clone", json={
            "points": [[16.0, 12.0]], "offset": [5.0, 0.0],
            "size": 10, "brush": "nope"})
        assert r.status_code == 400  # KeyError from the brush loader


def test_smudge_smears_image(tmp_path):
    with _client(tmp_path) as c:
        left, right = (123, 45, 67), (10, 20, 30)
        arr = np.zeros((48, 64, 3), np.uint8)
        arr[:, :32] = left
        arr[:, 32:] = right
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        _set_canvas(c, buf.getvalue())
        # sequential smear: the left band is carried to the drag endpoint
        r = c.post("/api/canvas/smudge", json={
            "points": [[16.0, 24.0], [56.0, 24.0]], "size": 20,
            "opacity": 1.0})
        # a smudge stroke is a draw step; the mask stack is untouched
        assert r.json() == {"ok": True, "undo_mask": False, "redo_mask": False,
                            "undo_draw": True, "redo_draw": False}
        img = _img(c)
        assert img[24, 48].tolist() == list(left)  # sequential carry
        assert img[24, 10].tolist() == list(left)  # source region untouched


def test_smudge_empty_canvas_400(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/canvas/smudge", json={
            "points": [[1.0, 1.0], [5.0, 1.0]], "size": 10})
        assert r.status_code == 400


def test_smudge_bad_values_422(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        assert c.post("/api/canvas/smudge", json={
            "points": [], "size": 10}).status_code == 422
        assert c.post("/api/canvas/smudge", json={
            "points": [[1.0, 1.0], [5.0, 1.0]],
            "size": 0}).status_code == 422
        assert c.post("/api/canvas/smudge", json={
            "points": [[1.0, 1.0], [5.0, 1.0]],
            "opacity": 2.0}).status_code == 422
        # size boundary (same as /api/canvas/stroke)
        assert c.post("/api/canvas/smudge", json={
            "points": [[1.0, 1.0], [5.0, 1.0]],
            "size": MAX_BRUSH_SIZE_ZOOMED}).status_code == 200
        assert c.post("/api/canvas/smudge", json={
            "points": [[1.0, 1.0], [5.0, 1.0]],
            "size": MAX_BRUSH_SIZE_ZOOMED + 1}).status_code == 422


def test_smudge_unknown_brush_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/canvas/smudge", json={
            "points": [[16.0, 12.0], [40.0, 12.0]], "size": 10,
            "brush": "nope"})
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


def test_brush_png_endpoint(tmp_path):
    # the client's stroke preview fetches the exact brush shape (the
    # luminance PNG, 256 px largest dimension); 404 for unknown names
    with _client(tmp_path, brush="soft") as c:
        r = c.get("/api/brush/circle.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        a = np.asarray(Image.open(io.BytesIO(r.content)).convert("L"))
        assert a.shape == (256, 256)
        assert a[128, 128] == 255  # centre opaque
        assert a[0, 0] == 0  # corner transparent
        r2 = c.get("/api/brush/soft.png")
        a2 = np.asarray(Image.open(io.BytesIO(r2.content)).convert("L"))
        assert a2.shape == (256, 256)  # 8x8 source scaled to 256
        assert a2.max() == 200  # the brush file's luminance
        assert c.get("/api/brush/nope.png").status_code == 404


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

        # undo the mask stroke (independent of the draw stack)
        r = c.post("/api/canvas/undo", json={"target": "mask"})
        assert r.json()["undo_mask"] is False  # nothing left to undo in mask
        assert r.json()["undo_draw"] is True  # the image stroke still undoable
        assert _mask(c)[12, 16] == 0  # mask stroke reverted
        assert _img(c)[12, 16][1] == 255  # image stroke intact

        c.post("/api/canvas/redo", json={"target": "mask"})
        assert _mask(c)[12, 16] == 255

        # the draw stack is independent: undo it separately
        c.post("/api/canvas/undo", json={"target": "draw"})
        assert _img(c)[12, 16][1] == 10  # image stroke reverted
        assert _mask(c)[12, 16] == 255  # mask stroke still applied


def test_undo_redo_unknown_target_422(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        assert c.post("/api/canvas/undo", json={"target": "bogus"}).status_code == 422
        assert c.post("/api/canvas/redo", json={"target": "bogus"}).status_code == 422


def test_select_switches_and_rejects_unknown(tmp_path):
    with _client(tmp_path) as c:
        c.post("/api/canvas/upload", content=_png_bytes(32, 24, 1))  # upload:0
        # re-selecting the current id is a no-op that succeeds + returns flags
        r = c.post("/api/canvas/select", json={"id": "upload:0"})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert "undo_draw" in r.json() and "redo_mask" in r.json()
        # unknown id -> 400
        assert c.post("/api/canvas/select", json={"id": "nope"}).status_code == 400


def test_mask_png_follows_image_size(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "target": "mask"})
        assert _mask(c).shape == (24, 32)
        _set_canvas(c, _png_bytes(64, 48))  # mask.png now follows the image
        assert _mask(c).shape == (48, 64)


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


def test_new_image_inherits_mask_scaled_to_new_size(tmp_path):
    """The mask is per-image: a new image inherits the previous one's mask,
    scaled (fit-within, centered) to the new size (not reset)."""
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24
        c.post("/api/canvas/stroke", json={
            "points": [[16.0, 12.0]], "size": 10, "target": "mask"})
        assert _mask(c)[12, 16] == 255
        _set_canvas(c, _png_bytes(64, 48))  # different size -> mask scaled
        m = _mask(c)
        assert m.shape == (48, 64)  # mask.png follows the image size
        assert m[24, 32] == 255  # the dot scaled 2x (12,16 -> 24,32)


def _png_with_meta(w=16, h=12) -> bytes:
    from imagagen import metadata

    buf = io.BytesIO()
    img, info = metadata.attach_metadata(
        Image.new("RGB", (w, h), (5, 5, 5)),
        metadata.build_parameters(
            "a test prompt", "bad thing", 20, "Euler a", "Karras", 6.5,
            42, w, h, "some_model", "abc123", loras=[("lora_a", 0.8)]),
    )
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def test_png_meta_with_metadata(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/png-meta", content=_png_with_meta(16, 12))
        assert r.status_code == 200
        j = r.json()
        assert j["width"] == 16
        assert j["height"] == 12
        m = j["meta"]
        assert m["prompt"] == "a test prompt"
        assert m["negative_prompt"] == "bad thing"
        assert m["steps"] == 20
        assert m["sampler"] == "Euler a"
        assert m["scheduler"] == "Karras"
        assert m["cfg"] == 6.5
        assert m["seed"] == 42
        assert m["width"] == 16 and m["height"] == 12
        assert m["model"] == "some_model"
        assert m["model_hash"] == "abc123"
        assert m["loras"] == ["lora_a@0.8"]


def test_png_meta_without_metadata(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/png-meta", content=_png_bytes(28, 20))
        assert r.status_code == 200
        j = r.json()
        assert j["meta"] is None
        assert j["width"] == 28
        assert j["height"] == 20


def test_png_meta_not_an_image(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/png-meta", content=b"this is not a png")
        assert r.status_code == 400


# ---------------- crop ----------------

def _two_band_png() -> bytes:
    arr = np.zeros((24, 32, 3), np.uint8)
    arr[:, :16] = (120, 0, 0)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_crop_crops_image_and_returns_urls(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c, _two_band_png())
        r = c.post("/api/canvas/crop", json={"x": 4, "y": 6, "width": 12, "height": 8})
        assert r.status_code == 200
        j = r.json()
        assert j["ok"] is True
        assert j["prev_image_id"] and j["crop_image_id"]
        assert j["prev_image_id"] != j["crop_image_id"]
        assert j["prev_url"].startswith("/api/canvas/image.png?id=")
        assert j["crop_url"].startswith("/api/canvas/image.png?id=")
        # the four undo/redo flags are present (canvas-mutating response)
        for k in ("undo_mask", "redo_mask", "undo_draw", "redo_draw"):
            assert isinstance(j[k], bool)
        # the canvas now shows the cropped image (12x8)
        img = _img(c)
        assert img.shape == (8, 12, 3)
        assert img[0, 0][0] == 120  # inside the red left band


def test_crop_pre_crop_image_restorable(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c, _two_band_png())
        j = c.post("/api/canvas/crop", json={"x": 0, "y": 0, "width": 16, "height": 24}).json()
        # the pre-crop image is servable by id and, when re-selected,
        # restores the full-size image
        prev_png = c.get(j["prev_url"])
        assert prev_png.status_code == 200
        assert np.asarray(Image.open(io.BytesIO(prev_png.content))).shape == (24, 32, 3)
        s = c.post("/api/canvas/select", json={"id": j["prev_image_id"]})
        assert s.status_code == 200 and s.json()["ok"] is True
        assert _img(c).shape == (24, 32, 3)


def test_crop_clamps_to_bounds(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c, _two_band_png())
        # region runs past the right/bottom edges: clamped to the image
        r = c.post("/api/canvas/crop", json={"x": 20, "y": 20, "width": 100, "height": 100})
        assert r.status_code == 200
        assert _img(c).shape == (4, 12, 3)


def test_crop_empty_region_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24
        r = c.post(
            "/api/canvas/crop", json={"x": 50, "y": 50, "width": 5, "height": 5}
        )
        assert r.status_code == 400


def test_crop_empty_canvas_400(tmp_path):
    with _client(tmp_path) as c:
        r = c.post(
            "/api/canvas/crop", json={"x": 0, "y": 0, "width": 5, "height": 5}
        )
        assert r.status_code == 400


def test_crop_bad_values_422(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/canvas/crop", json={"x": 0, "y": 0, "width": 0, "height": 5})
        assert r.status_code == 422
        assert c.post("/api/canvas/crop", json={"x": 0, "y": 0, "height": 5}).status_code == 422


def test_image_png_unknown_id_404(tmp_path):
    with _client(tmp_path) as c:
        assert c.get("/api/canvas/image.png?id=nope").status_code == 404
