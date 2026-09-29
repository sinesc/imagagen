"""Canvas API: state sync (the client is the editor), brush PNGs,
mask.png, select, crop and entity PNGs."""

import io

import numpy as np
from PIL import Image
from PIL import PngImagePlugin
from starlette.testclient import TestClient

from imagagen.app import create_app
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


def test_select_switches_and_rejects_unknown(tmp_path):
    with _client(tmp_path) as c:
        c.post("/api/canvas/upload", content=_png_bytes(32, 24, 1))  # upload:0
        # re-selecting the current id is a no-op that succeeds
        r = c.post("/api/canvas/select", json={"id": "upload:0"})
        assert r.status_code == 200 and r.json()["ok"] is True
        # unknown id -> 400
        assert c.post("/api/canvas/select", json={"id": "nope"}).status_code == 400


def test_mask_png_follows_image_size(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24
        c.post("/api/canvas/sync", files=_sync_payload(mask_px=(16, 12),
                                                        mask_value=255))
        assert _mask(c).shape == (24, 32)
        _set_canvas(c, _png_bytes(64, 48))  # mask.png now follows the image
        assert _mask(c).shape == (48, 64)


def test_mask_retained_across_same_size_image(tmp_path):
    """Spec: the mask is retained across image generations (e.g. the first
    result auto-set into the canvas, or a thumb selection)."""
    with _client(tmp_path) as c:
        _set_canvas(c)
        c.post("/api/canvas/sync", files=_sync_payload(mask_px=(16, 12),
                                                        mask_value=255))
        assert _mask(c)[12, 16] == 255
        _set_canvas(c)  # new image, same size -> mask survives
        assert _mask(c)[12, 16] == 255


def test_new_image_inherits_mask_scaled_to_new_size(tmp_path):
    """The mask is per-image: a new image inherits the previous one's mask,
    scaled (fit-within, centered) to the new size (not reset)."""
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24
        # a 2x2 mask block (a single pixel would resample down when the
        # mask is scaled to the new image size)
        payload = _sync_payload(mask_px=(16, 12), mask_value=255)
        mask = Image.open(io.BytesIO(payload["mask"][1])).convert("RGBA")
        px = mask.load()
        for yy, xx in ((11, 16), (12, 15), (12, 16), (11, 15)):
            px[xx, yy] = (128, 128, 128, 255)
        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        payload["mask"] = ("mask.png", buf.getvalue(), "image/png")
        c.post("/api/canvas/sync", files=payload)
        assert _mask(c)[12, 16] == 255
        _set_canvas(c, _png_bytes(64, 48))  # different size -> mask scaled
        m = _mask(c)
        assert m.shape == (48, 64)  # mask.png follows the image size
        assert m[24, 32] == 255  # the block scaled 2x (12,16 -> 24,32)


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


# ---------------- client-authoritative state sync ----------------

def _sync_payload(w=32, h=24, img_value=200, mask_value=128,
                 mask_px=(5, 5)) -> dict:
    """Multipart files as the client sends them: the image canvas blob and
    the mask value canvas (RGBA, alpha = value)."""
    img = Image.new("RGB", (w, h), (img_value, img_value, img_value))
    mask = Image.new("RGBA", (w, h), (128, 128, 128, 0))
    px = mask.load()
    px[mask_px] = (128, 128, 128, mask_value)
    buf_i, buf_m = io.BytesIO(), io.BytesIO()
    img.save(buf_i, format="PNG")
    mask.save(buf_m, format="PNG")
    return {
        "image": ("canvas.png", buf_i.getvalue(), "image/png"),
        "mask": ("mask.png", buf_m.getvalue(), "image/png"),
    }


def test_sync_replaces_image_and_mask(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24, value 10
        r = c.post("/api/canvas/sync", files=_sync_payload())
        assert r.status_code == 200 and r.json()["ok"]
        img = _img(c)
        assert (img == 200).all()
        # the client's RGBA value PNG decodes via its alpha channel
        np.testing.assert_array_equal(_mask(c)[5, 5], 128)
        assert (_mask(c) == 0).sum() == 32 * 24 - 1


def test_sync_empty_canvas_400(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/api/canvas/sync", files=_sync_payload()).status_code == 400


def test_sync_missing_parts_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post(
            "/api/canvas/sync",
            files={"image": ("canvas.png", _png_bytes(), "image/png")},
        )
        assert r.status_code == 400
        r = c.post(
            "/api/canvas/sync",
            files={
                "mask": ("mask.png", _png_bytes(32, 24, 128), "image/png")
            },
        )
        assert r.status_code == 400


def test_sync_size_mismatch_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)  # 32x24
        r = c.post("/api/canvas/sync", files=_sync_payload(w=16, h=16))
        assert r.status_code == 400
        # the state is untouched
        assert (_img(c) == 10).all()


def test_sync_preserves_png_info_chunks(tmp_path):
    buf = io.BytesIO()
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", "a prompt")
    Image.new("RGB", (32, 24), (10, 10, 10)).save(buf, format="PNG",
                                                   pnginfo=pnginfo)
    with _client(tmp_path) as c:
        _set_canvas(c, data=buf.getvalue())
        r = c.post("/api/canvas/sync", files=_sync_payload())
        assert r.status_code == 200
        data = c.get("/api/canvas.png").content
        assert Image.open(io.BytesIO(data)).info.get("parameters") == "a prompt"


def test_sync_bad_png_400(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post(
            "/api/canvas/sync",
            files={
                "image": ("canvas.png", b"not a png", "image/png"),
                "mask": ("mask.png", _png_bytes(32, 24, 0), "image/png"),
            },
        )
        assert r.status_code == 400
