"""Server-side canvas store: per-image entities (the client is the editor).

Strokes, masks, undo/redo and morphology live client-side; the server
persists what it is shown via ``replace_state`` (POST /api/canvas/sync).
These tests cover entity management, mask inheritance, crop, brush PNG
serving and state replacement.

Pure-Python (numpy/PIL) -- no GPU, no model.
"""

import io

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from imagagen.canvas import Canvas


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def blank_canvas(w=64, h=48, value=10):
    c = Canvas()
    c.import_image(_png(np.full((h, w, 3), value, dtype=np.uint8)))
    return c


# ---------------- brush PNG serving (client previews) ----------------

def test_custom_brush_luminance_is_alpha(tmp_path):
    # 8x4 brush: left half white (opaque), right half black (transparent)
    left = np.full((8, 4), 255, dtype=np.uint8)
    right = np.zeros((8, 4), dtype=np.uint8)
    Image.fromarray(np.hstack([left, right])).save(tmp_path / "half.png")
    c = Canvas(brushes_dir=tmp_path)
    s = c.brush_shape("half", 32)
    assert s.shape == (32, 32)
    assert s[16, 4].min() > 0.9  # left half fully opaque
    assert s[16, -5:].max() < 0.1  # right half transparent


def test_circle_brush_shape_aa():
    c = Canvas()
    s = c.brush_shape("circle", 8)
    assert s.shape == (8, 8)
    assert s.max() == 1.0 and s.min() == 0.0
    assert 0 < s[0, 3] < 1  # edge pixels are antialiased


def test_brush_png_bytes():
    c = Canvas()
    a = np.asarray(Image.open(io.BytesIO(c.brush_png("circle"))).convert("L"))
    assert a.shape == (256, 256)
    assert a[128, 128] == 255  # centre fully opaque
    assert a[0, 0] == 0  # corner transparent
    with pytest.raises(KeyError):
        c.brush_png("nope")


def test_list_brushes(tmp_path):
    c = Canvas(brushes_dir=tmp_path)
    Image.new("L", (4, 4), 128).save(tmp_path / "b1.png")
    (tmp_path / "b2.png").write_bytes(b"fake")  # content irrelevant for listing
    (tmp_path / "notes.txt").write_bytes(b"not a brush")
    assert c.list_brushes() == ["circle", "b1", "b2"]


def test_list_brushes_default_dir_first(tmp_path):
    d, u = tmp_path / "res", tmp_path / "user"
    d.mkdir()
    u.mkdir()
    for name in ("a.png", "dup.png"):  # dup in both folders
        Image.new("L", (4, 4), 10).save(d / name)
    for name in ("b.png", "dup.png"):
        Image.new("L", (4, 4), 200).save(u / name)
    c = Canvas(brushes_dir=u, default_brushes_dir=d)
    # provided defaults first, then user; a duplicate is listed once
    assert c.list_brushes() == ["circle", "a", "dup", "b"]
    # the default file wins: its luminance is 10/255
    assert c.brush_shape("dup", 4).max() == pytest.approx(10 / 255.0)
    assert c.brush_shape("b", 4).max() == pytest.approx(200 / 255.0)


# ---------------- mask ----------------

def test_mask_is_at_image_size_and_served_directly():
    c = blank_canvas(200, 100)
    m = np.zeros((100, 200), dtype=np.uint8)
    m[40:60, 90:110] = 255
    assert c.set_mask(m)
    np.testing.assert_array_equal(c.mask, m)
    # the served mask.png is the mask itself (at the image's size)
    png = Image.open(io.BytesIO(c.to_mask_png_bytes()))
    assert png.size == (200, 100)


def test_set_mask_identical_returns_false():
    c = blank_canvas()
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 10:20] = 255
    assert c.set_mask(m)
    assert not c.set_mask(m.copy())  # identical -> no change


def test_set_mask_wrong_size_raises():
    c = blank_canvas()
    with pytest.raises(ValueError):
        c.set_mask(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        Canvas().set_mask(np.zeros((8, 8), dtype=np.uint8))


# ---------------- client-authoritative state push ----------------

def test_replace_state_swaps_image_and_mask():
    c = blank_canvas(64, 48, value=10)
    img = np.full((48, 64, 3), 200, dtype=np.uint8)
    img[5, 5] = (1, 2, 3)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[5, 5] = 128
    c.replace_state(img, mask)
    np.testing.assert_array_equal(c.image, img)
    np.testing.assert_array_equal(c.mask, mask)
    # the cold store follows the hot state
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(c.to_png_bytes()))), img
    )


def test_replace_state_preserves_png_info_chunks():
    arr = np.full((24, 32, 3), 10, dtype=np.uint8)
    buf = io.BytesIO()
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", "a prompt")
    Image.fromarray(arr).save(buf, format="PNG", pnginfo=pnginfo)
    c = Canvas()
    c.import_image(buf.getvalue())
    c.replace_state(arr.copy(), np.zeros((24, 32), dtype=np.uint8))
    assert c.png_info() == {"parameters": "a prompt"}


def test_replace_state_empty_canvas_raises():
    c = Canvas()
    with pytest.raises(ValueError):
        c.replace_state(np.zeros((8, 8, 3), dtype=np.uint8),
                        np.zeros((8, 8), dtype=np.uint8))


# ---------------- mask inheritance (new entity) ----------------

def test_new_image_inherits_source_mask():
    # a new image (upload / result) inherits the source image's current
    # mask transformed to the new size (aspect fit within, centered)
    c = blank_canvas(64, 48)
    m = np.zeros((48, 64), dtype=np.uint8)
    m[14:34, 22:42] = 255  # the "dot"
    c.set_mask(m)
    ma = c.mask.copy()
    src = c.current_id()
    c.anchor({src})  # keep the source alive across the import
    b = c.import_image(_png(np.zeros((24, 32, 3), dtype=np.uint8)))
    c.anchor({src, b})
    # inherited current mask: fit-within, centered (2x downscale -> 1:1)
    assert c.mask is not None and c.mask.shape == (24, 32)
    assert c.mask[12, 16] == 255  # the dot, at the 2x-downscaled position
    # ...and the source's mask is untouched
    c.set_current(src)
    np.testing.assert_array_equal(c.mask, ma)


def test_add_result_inherits_mask_from_source():
    c = blank_canvas(64, 48)
    m = np.zeros((48, 64), dtype=np.uint8)
    m[14:34, 22:42] = 255
    c.set_mask(m)
    src = c.current_id()
    # same-size result: the mask is copied 1:1
    eid = c.add_result(1, 0, _png(np.full((48, 64, 3), 7, np.uint8)), None,
                      source_id=src)
    ent = c._images[eid]
    np.testing.assert_array_equal(ent.mask, c._images[src].mask)
    # larger result (outpaint): the mask is transformed to the new size
    eid2 = c.add_result(1, 1, _png(np.full((96, 128, 3), 7, np.uint8)), None,
                       source_id=src)
    m2 = c._images[eid2].mask
    assert m2.shape == (96, 128)
    assert m2[48, 64] == 255  # the dot, at 2x


def test_add_result_without_source_has_empty_mask():
    c = Canvas()
    eid = c.add_result(1, 0, _png(np.full((16, 16, 3), 1, np.uint8)), None)
    ent = c._images[eid]
    assert ent.mask.shape == (16, 16) and (ent.mask == 0).all()
    # the mask still exists (empty): once displayed, mask.png is 200, not 404
    c.set_current(eid)
    assert c.to_mask_png_bytes() is not None


def test_mask_per_image_isolation():
    # a mask set on one image must not be clobbered by, nor leak into,
    # another image of a different size
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))  # 32x24
    c.anchor({a})
    m = np.zeros((24, 32), dtype=np.uint8)
    m[4:12, 4:12] = 255
    c.set_mask(m)
    ma = c.mask.copy()
    # a differently-sized image is imported: it inherits A's mask...
    b = c.import_image(_png(np.full((48, 64, 3), 50, np.uint8)))  # 64x48
    c.anchor({a, b})
    assert c.mask is not None and (c.mask > 0).any()  # the blob, scaled 2x
    # ...which B may now replace independently...
    c.replace_state(np.full((48, 64, 3), 50, np.uint8),
                    np.zeros((48, 64), dtype=np.uint8))
    # ...without touching A: back on A its mask is intact
    c.set_current(a)
    np.testing.assert_array_equal(c.mask, ma)
    # ...and B's replaced mask too
    c.set_current(b)
    assert (c.mask == 0).all()


# ---------------- upload / import ----------------

def test_upload_png_info_survives_edits():
    c = Canvas()
    c.import_image(_png_with_info(b"seed: 1"))
    assert c.png_info().get("parameters") == "seed: 1"
    c.replace_state(np.full((16, 16, 3), 9, np.uint8),
                    np.zeros((16, 16), dtype=np.uint8))
    assert c.png_info().get("parameters") == "seed: 1"
    data = c.to_png_bytes()
    assert Image.open(_bytes_io(data)).info.get("parameters") == "seed: 1"


# ---------------- crop ----------------

def test_crop_crops_image_and_mask_1_to_1():
    # left band coloured, right band left black; mask painted on the left
    arr = np.zeros((24, 32, 3), np.uint8)
    arr[:, :16] = 100
    c = Canvas()
    c.import_image(_png(arr))
    m = np.zeros((24, 32), dtype=np.uint8)
    m[4:12, 2:10] = 255
    c.set_mask(m)

    prev_id, eid = c.crop(4, 6, 12, 8)  # x 4..16, y 6..14
    assert c.current_id() == eid
    img = c.image
    assert img.shape == (8, 12, 3)  # (h, w, ch)
    assert img[0, 0].tolist() == [100, 100, 100]  # still inside the left band
    # the mask is cropped 1:1 with the image: the dot (rows 4..11, cols 2..9)
    # is shifted to the crop origin (6, 4)
    assert c.mask.shape == (8, 12)
    assert c.mask[0, 0] == 255  # original (6, 4)
    assert c.mask[5, 5] == 255  # original (11, 9)
    assert c.mask[7, 7] == 0  # original (13, 11): outside the dot
    # the pre-crop image is kept as a restorable entity, not mutated
    prev = c._images[prev_id]
    assert prev.image is None  # hot cache dropped (it left the canvas)
    assert prev.png  # cold store intact


def test_crop_clamps_to_image_bounds():
    arr = np.zeros((24, 32, 3), np.uint8)
    arr[:, :16] = 100
    c = Canvas()
    c.import_image(_png(arr))
    # rect runs past the right and bottom edges -> clamped to the image
    c.crop(20, 20, 100, 100)
    assert c.image.shape == (4, 12, 3)  # (24-20, 32-20)
    assert c.image[0, 0].tolist() == [0, 0, 0]  # orig (20,20): right band
    # rect starts outside on the top/left -> clamped to the origin
    c2 = Canvas()
    c2.import_image(_png(arr.copy()))
    c2.crop(-10, -10, 40, 40)
    assert c2.image.shape == (24, 30, 3)  # rect spans x -10..30 -> clamped to 0..30
    assert c2.image[0, 0].tolist() == [100, 100, 100]


def test_crop_empty_region_raises():
    c = blank_canvas(16, 16, value=1)
    with pytest.raises(ValueError):
        c.crop(5, 5, 0, 0)
    with pytest.raises(ValueError):
        c.crop(20, 20, 5, 5)  # entirely outside the image


def test_crop_empty_canvas_raises():
    c = Canvas()
    with pytest.raises(ValueError):
        c.crop(0, 0, 10, 10)


def test_crop_pins_pre_crop_and_result_against_eviction():
    arr = np.zeros((24, 32, 3), np.uint8)
    c = Canvas()
    c.import_image(_png(arr))
    prev_id, eid = c.crop(0, 0, 16, 24)
    # un-anchor everything and evict: both the pre-crop image and the crop
    # must survive because they are pinned
    c.anchor(set())
    c.evict_stale()
    assert prev_id in c._images
    assert eid in c._images
    assert c.current_id() == eid
    # releasing the pins evicts the one that is no longer current
    c.release_pins()
    assert prev_id not in c._images
    assert eid in c._images  # still current


def test_user_pin_survives_eviction_and_release_pins():
    """The UI's "Pin thumbnail" checkbox: user pins must survive both
    stale eviction and the crop-pin release a new generation performs
    (unlike crop pins), until explicitly unpinned."""
    c = blank_canvas()
    prev_id, crop_id = c.crop(0, 0, 32, 24)  # crop pins both entities
    c.set_current(prev_id)  # crop_id is no longer the current image
    c.user_pin(crop_id)  # keep the crop via the UI pin as well
    c.anchor(set())
    c.release_pins()  # new generation starts: crop pins go away
    assert prev_id in c._images  # still the current image
    assert crop_id in c._images  # saved by the user pin, not the crop pin
    assert c.user_pinned() == {crop_id}
    # unpinning lets the entity be evicted again
    c.user_unpin(crop_id)
    c.evict_stale()
    assert crop_id not in c._images


def test_user_pin_rejects_unknown_and_is_idempotent():
    c = blank_canvas()
    with pytest.raises(ValueError):
        c.user_pin("nope")
    eid = c.current_id()
    c.user_pin(eid)
    c.user_pin(eid)  # pinning twice is a no-op
    assert c.user_pinned() == {eid}
    c.user_unpin("nope")  # unknown ids are ignored
    c.user_unpin(eid)
    assert c.user_pinned() == set()


def test_crop_keeps_png_metadata():
    c = Canvas()
    c.import_image(_png_with_info(b"seed: 7"))
    c.crop(0, 0, 16, 24)
    assert c.png_info().get("parameters") == "seed: 7"
    data = c.to_png_bytes()
    assert Image.open(_bytes_io(data)).info.get("parameters") == "seed: 7"


def test_entity_png_serves_any_known_entity():
    arr = np.zeros((16, 16, 3), np.uint8)
    arr[:, :8] = 5
    c = Canvas()
    c.import_image(_png(arr))
    prev_id, eid = c.crop(0, 0, 8, 16)
    for eid_, w in ((prev_id, 16), (eid, 8)):
        data = c.entity_png(eid_)
        assert data is not None
        assert Image.open(_bytes_io(data)).size == (w, 16)
    assert c.entity_png("nope") is None


# ---------------- entity eviction ----------------

def test_upload_pinned_until_release_when_it_leaves_canvas():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    b = c.import_image(_png(np.full((24, 32, 3), 50, np.uint8)))
    # a left the canvas but is pinned like a crop: the UI keeps its upload
    # thumbnail restorable until the next generation releases the pin
    assert a in c._images and b in c._images and c.current_id() == b
    c.release_pins()  # what a new generation does when it drops the thumbs
    assert a not in c._images and b in c._images


def test_upload_replace_retires_the_superseded_entity():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    b = c.import_image(_png(np.full((24, 32, 3), 50, np.uint8)), replace_id=a)
    assert a not in c._images  # superseded by the re-drop, pin released
    assert b in c._images and c.current_id() == b


def test_upload_replace_transfers_the_user_pin():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    c.user_pin(a)
    b = c.import_image(_png(np.full((24, 32, 3), 50, np.uint8)), replace_id=a)
    assert a not in c._images
    assert c.user_pinned() == {b}  # the re-pointed thumbnail stays pinned
    c.release_pins()  # the user pin must survive the per-generation reset
    assert b in c._images


def test_upload_replace_skips_anchored_entity():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    c.anchor({a})  # a live job's source: its PNG must stay reachable
    b = c.import_image(_png(np.full((24, 32, 3), 50, np.uint8)), replace_id=a)
    assert a in c._images and b in c._images
    c.anchor(set())
    c.evict_stale()  # pin released by the replace, evicted once unanchored
    assert a not in c._images


def test_anchored_entity_survives_leaving_canvas():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    c.anchor({a})
    b = c.import_image(_png(np.full((24, 32, 3), 50, np.uint8)))
    assert a in c._images  # still referenced by a (live) job
    # an un-anchored upload still needs its crop-style pin released
    c.anchor(set())
    c.evict_stale()
    assert a in c._images and b in c._images
    c.release_pins()
    assert a not in c._images and b in c._images


def test_current_never_evicted():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    c.anchor(set())
    c.evict_stale()
    assert a in c._images  # the displayed image always survives


def test_set_current_unknown_id_raises():
    c = blank_canvas()
    with pytest.raises(ValueError):
        c.set_current("job:9:0")


# ---------------- helpers ----------------

def _png_with_info(text: bytes) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", text.decode())
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _bytes_io(b: bytes):
    return io.BytesIO(b)


def test_import_retains_post_idat_metadata():
    """A foreign image whose tEXt chunk comes after the IDAT (Forge crops)
    still keeps its text metadata in the canvas entity -- otherwise a
    crop/save in imagagen would silently drop it."""
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(
            ">I", zlib.crc32(body) & 0xFFFFFFFF
        )

    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00\xff" * 8 for _ in range(8))
    idat = zlib.compress(raw)
    text = b"parameters" + b"\x00" + b"prompt: a thing"
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat)
        + chunk(b"tEXt", text)
        + chunk(b"IEND", b"")
    )
    c = Canvas()
    c.import_image(png)
    info = c.png_info()
    assert info.get("parameters") == "prompt: a thing"
