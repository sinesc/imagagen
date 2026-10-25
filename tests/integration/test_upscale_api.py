"""Upscale API: /api/upscalers listing and POST /api/upscale (crop-style
new entity + pin + metadata/mask carry-over). Uses the pure-PIL built-in
upscalers (Lanczos/Nearest) so no GPU/model weights are needed."""

import io

import numpy as np
from PIL import Image, PngImagePlugin
from starlette.testclient import TestClient

from imagagen.app import create_app
from imagagen.backends import MockBackend
from imagagen.config import Config


def _client(tmp_path):
    return TestClient(create_app(config=Config(root=tmp_path),
                                 backend=MockBackend()))


def _png_bytes(w=200, h=150, chunks: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (w, h), (10, 120, 240))
    if chunks:
        info = PngImagePlugin.PngInfo()
        for k, v in chunks.items():
            info.add_text(k, v)
        img.save(buf, format="PNG", pnginfo=info)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def _set_canvas(c, data=None):
    r = c.post("/api/canvas/upload", content=data or _png_bytes())
    assert r.status_code == 200


def _sync_mask(c, mask: np.ndarray):
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    img_buf = io.BytesIO()
    Image.fromarray(np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
                    ).save(img_buf, format="PNG")
    r = c.post("/api/canvas/sync",
               files={"image": ("c.png", img_buf.getvalue(), "image/png"),
                      "mask": ("m.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200


def test_upscalers_lists_builtins_and_models(tmp_path):
    with _client(tmp_path) as c:
        up = tmp_path / "models" / "upscalers" / "realesrgan"
        up.mkdir(parents=True)
        (up / "RealESRGAN_x4plus.pth").write_bytes(b"not-a-real-model")
        r = c.get("/api/upscalers")
        assert r.status_code == 200
        items = r.json()["upscalers"]
        # built-ins first, scanned model ids are extensionless rel paths
        assert [i["id"] for i in items[:2]] == ["Lanczos", "Nearest"]
        assert {"id": "realesrgan/RealESRGAN_x4plus",
                "name": "realesrgan/RealESRGAN_x4plus",
                "builtin": False} in items


def test_upscale_lanczos_creates_pinned_entity(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c, _png_bytes(chunks={"parameters": "a test prompt"}))
        # a 10x10 mask block at (0,0) must ride along scaled up
        mask = np.zeros((150, 200), np.uint8)
        mask[0:10, 0:10] = 255
        _sync_mask(c, mask)

        r = c.post("/api/upscale", json={"upscaler": "Lanczos", "scale": 2.0})
        assert r.status_code == 200
        body = r.json()
        # forge size rule: int(w*s)//8*8 (200x150 -> 400x296, not 300!)
        assert (body["width"], body["height"]) == (400, 296)
        prev_id, eid = body["prev_image_id"], body["upscale_image_id"]
        assert prev_id == "upload:0" and eid == "upscale:0"

        # the result entity is served and decodes at the upscaled size
        r = c.get("/api/canvas/image.png", params={"id": eid})
        assert r.status_code == 200
        img = Image.open(io.BytesIO(r.content))
        assert img.size == (400, 296)
        # the SOURCE entity's PNG text chunks survive on the result
        assert img.info.get("parameters") == "a test prompt"

        # the result is the displayed canvas image...
        r = c.get("/api/canvas.png")
        assert Image.open(io.BytesIO(r.content)).size == (400, 296)
        # the mask was resized to the new entity's size (block roughly
        # doubles: LANCZOS edges may feather a little)
        m = np.asarray(Image.open(io.BytesIO(
            c.get("/api/mask.png").content)).convert("L"))
        assert m.shape == (296, 400)
        assert m[5, 5] > 0 and m[15, 15] > 0 and m[60, 60] == 0
        # ...and both entities stay alive (crop-style pins): switching the
        # canvas does not evict them
        q = c.get("/api/queue").json()
        assert q["current"] == eid
        assert c.post("/api/canvas/select",
                      json={"id": prev_id}).status_code == 200
        assert c.get("/api/canvas/image.png",
                     params={"id": eid}).status_code == 200
        assert c.get("/api/canvas/image.png",
                     params={"id": prev_id}).status_code == 200


def test_upscale_nearest_resizes(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/upscale", json={"upscaler": "Nearest", "scale": 3.0})
        assert r.status_code == 200
        # int(200*3)//8*8 = 600, int(150*3)//8*8 = 448
        assert (r.json()["width"], r.json()["height"]) == (600, 448)


def test_upscale_max_side_length_clamps(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/upscale", json={
            "upscaler": "Lanczos", "scale": 4.0, "max_side_length": 256})
        assert r.status_code == 200
        body = r.json()
        # forge limit_size_by_one_dimention(800, 600, 256) = (256, 192);
        # effective scale 1.28 -> int(200*1.28)//8*8 = 256 / 192
        assert (body["width"], body["height"]) == (256, 192)


def test_upscale_hard_pixel_limit(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c, _png_bytes(w=4000, h=3000))
        r = c.post("/api/upscale", json={"upscaler": "Lanczos", "scale": 8.0})
        assert r.status_code == 400
        assert "16384" in r.json()["error"]
        # an upscale that fails creates no entity and keeps the canvas
        q = c.get("/api/queue").json()
        assert q["current"] == "upload:0"


def test_upscale_rejects_bad_requests(tmp_path):
    with _client(tmp_path) as c:
        # empty canvas
        r = c.post("/api/upscale", json={"upscaler": "Lanczos", "scale": 2.0})
        assert r.status_code == 400
        _set_canvas(c)
        # unknown upscaler id
        r = c.post("/api/upscale", json={"upscaler": "does/not/exist",
                                         "scale": 2.0})
        assert r.status_code == 400
        assert "unknown upscaler" in r.json()["error"]
        # scale/tile ranges (pydantic)
        assert c.post("/api/upscale", json={"upscaler": "Lanczos",
                                            "scale": 0.5}).status_code == 422
        assert c.post("/api/upscale", json={"upscaler": "Lanczos",
                                            "scale": 9}).status_code == 422
        assert c.post("/api/upscale", json={"upscaler": "Lanczos",
                                            "scale": 2.0,
                                            "tile": 600}).status_code == 422


def test_upscale_entities_survive_switching(tmp_path):
    with _client(tmp_path) as c:
        _set_canvas(c)
        r = c.post("/api/upscale", json={"upscaler": "Lanczos", "scale": 2.0})
        eid = r.json()["upscale_image_id"]
        # switch back to the pre-upscale image, then to the upscale again
        assert c.post("/api/canvas/select",
                      json={"id": "upload:0"}).status_code == 200
        assert c.post("/api/canvas/select",
                      json={"id": eid}).status_code == 200
        assert Image.open(io.BytesIO(
            c.get("/api/canvas.png").content)).size == (400, 296)
