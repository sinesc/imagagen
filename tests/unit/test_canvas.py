"""Canvas v2: per-image draw history, global image-independent mask, mask
morphology, mask scaling, entity eviction.

Pure-Python (numpy/PIL/scipy) -- no GPU, no model.
"""

import numpy as np
import pytest
from PIL import Image

from imagagen.canvas import Canvas, DRAW_HISTORY, MASK_HISTORY


def _png(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def blank_canvas(w=64, h=48, value=10):
    c = Canvas()
    c.import_image(_png(np.full((h, w, 3), value, dtype=np.uint8)))
    return c


def test_stamp_positions_click_is_one_dot():
    assert Canvas.stamp_positions([(5.0, 5.0)], 2.0) == [(5.0, 5.0)]


def test_stamp_positions_spacing_and_endpoint():
    # straight 100 px line, spacing 10 -> ~10 stamps, endpoints covered
    pts = [(0.0, 0.0), (100.0, 0.0)]
    pos = Canvas.stamp_positions(pts, 10.0)
    assert pos[0] == (0.0, 0.0)
    assert pos[-1] == (100.0, 0.0)  # the last point is always stamped
    gaps = [
        np.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(pos, pos[1:], strict=False)
        if a != b
    ]
    assert all(g >= 9.999 for g in gaps)  # consecutive stamps >= spacing
    assert len(pos) == 11  # 0,10,...,100


def test_stamp_positions_diagonal():
    pos = Canvas.stamp_positions([(0.0, 0.0), (40.0, 30.0)], 10.0)
    assert abs(np.hypot(pos[-1][0], pos[-1][1]) - 50.0) < 1e-9
    gaps = [
        np.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(pos, pos[1:], strict=False)
        if a != b
    ]
    assert all(g >= 9.999 for g in gaps)


def test_stroke_paints_with_color_and_opacity():
    c = blank_canvas(64, 48, value=0)
    changed = c.stroke([(32.0, 24.0)], size=20, opacity=1.0, color=(255, 0, 0))
    assert changed
    img = c.image
    assert img[24, 32].tolist() == [255, 0, 0]  # centre is full colour
    assert img[0, 0].tolist() == [0, 0, 0]  # far corner untouched
    # opacity 0.5 over black -> mid grey
    c2 = blank_canvas(64, 48, value=0)
    c2.stroke([(32.0, 24.0)], size=20, opacity=0.5, color=(255, 0, 0))
    assert 120 <= c2.image[24, 32, 0] <= 135


def test_stroke_covers_whole_line_including_endpoints():
    c = blank_canvas(128, 16, value=0)
    c.stroke([(10.0, 8.0), (117.0, 8.0)], size=8, opacity=1.0, color=(255, 255, 255))
    img = c.image[:, :, 0]
    # a full-width painted band, incl. both ends (endpoint stamps)
    row = img[8]
    assert row[10] > 128 and row[117] > 128
    assert img[0, 64] == 0  # row far above the stroke untouched


def test_stroke_outside_canvas_clips():
    c = blank_canvas(16, 16, value=0)
    assert c.stroke([(-8.0, -8.0)], size=32, opacity=1.0, color=(1, 2, 3))
    img = c.image
    assert img[0, 0][0] == 1  # the part inside the canvas was painted
    assert img[15, 15][0] in (0, 1)  # still near-black, no crash


def test_stroke_empty_canvas_raises():
    c = Canvas()
    with pytest.raises(ValueError):
        c.stroke([(1.0, 1.0)], size=10)


def test_stroke_unknown_brush_raises():
    c = blank_canvas()
    with pytest.raises(KeyError):
        c.stroke([(32.0, 24.0)], size=10, brush="nope")


def test_custom_brush_luminance_is_alpha(tmp_path):
    # 8x4 brush: left half white (paints), right half black (never paints)
    left = np.full((8, 4), 255, dtype=np.uint8)
    right = np.zeros((8, 4), dtype=np.uint8)
    Image.fromarray(np.hstack([left, right])).save(tmp_path / "half.png")
    c = blank_canvas(64, 48, value=0)
    c._brushes_dir = tmp_path
    c.stroke([(32.0, 24.0)], size=32, opacity=1.0, color=(255, 255, 255), brush="half")
    img = c.image[:, :, 0]
    assert img[24, 20] == 255  # left half of the brush painted
    assert img[24, 44] == 0  # right half (black) never painted


def test_circle_brush_shape_aa():
    c = Canvas()
    s = c.brush_shape("circle", 8)
    assert s.shape == (8, 8)
    assert s.max() == 1.0 and s.min() == 0.0
    assert 0 < s[0, 3] < 1  # edge pixels are antialiased


def test_list_brushes(tmp_path):
    c = Canvas(brushes_dir=tmp_path)
    Image.new("L", (4, 4), 128).save(tmp_path / "b1.png")
    (tmp_path / "b2.png").write_bytes(b"fake")  # content irrelevant for listing
    (tmp_path / "notes.txt").write_bytes(b"not a brush")
    assert c.list_brushes() == ["circle", "b1", "b2"]


# ---------------- mask ----------------

def test_mask_stroke_paints_and_erases():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=20, opacity=1.0, target="mask")
    m = c.mask
    assert m[24, 32] == 255
    assert m[0, 0] == 0
    # shift = erase: the identical stroke removes exactly what it painted
    c.stroke([(32.0, 24.0)], size=20, opacity=1.0, erase=True, target="mask")
    assert c.mask[24, 32] == 0
    assert (c.mask == 0).all()  # AA fringe erased too (same shape, same opacity)


def test_mask_stroke_opacity_clips():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=20, opacity=0.5, target="mask")
    assert c.mask[24, 32] == 127  # 0.5 * 255, truncated to int
    c.stroke([(32.0, 24.0)], size=20, opacity=0.5, target="mask")
    assert c.mask[24, 32] >= 254  # additive, capped (127 + 127.5 -> 254)


def test_mask_stroke_does_not_touch_image():
    c = blank_canvas(value=42)
    c.stroke([(32.0, 24.0)], size=20, target="mask")
    assert (c.image == 42).all()


def test_mask_dilate_and_erode():
    c = blank_canvas(64, 48)
    c.stroke([(32.0, 24.0)], size=10, target="mask")
    before = (c.mask > 0).sum()
    c.mask_dilate(5)
    after = (c.mask > 0).sum()
    # a ~10 px disc dilated by 5 -> ~20 px disc: area grows ~4x
    assert after > 3.0 * before
    assert c.mask.max() == 255  # the grown ring is filled solid
    c.mask_erode(5)
    # erode by 5 undoes the dilate (support back to the original ~10px disc)
    assert (c.mask > 0).sum() <= before + 10


def test_mask_dilate_keeps_soft_alpha_inside():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=20, opacity=0.5, target="mask")
    centre = c.mask[24, 32]
    c.mask_dilate(10)
    assert c.mask[24, 32] == centre  # original soft value preserved
    # the grown ring takes the alpha of the nearest original pixel (the
    # feather shifts outward) instead of being filled solid
    assert c.mask[24, 50] == centre


def test_mask_dilate_closes_feather_dip():
    # 5x5 solid core, 1 px feather ring at 50, rest 0. The old
    # binary-dilate + solid-fill left the feather as a low-opacity band
    # between the core and the new solid ring.
    c = blank_canvas(9, 9)
    m = np.zeros((9, 9), dtype=np.uint8)
    m[2:7, 2:7] = 255
    m[1:8, 1:8] = np.where(m[1:8, 1:8] == 0, 50, m[1:8, 1:8])
    c._mask = m.copy()

    assert c.mask_dilate(1)
    # the 255 contour grew 1 px (7x7 solid), the 50 feather shifted out
    # to the outer edge; no low-opacity band inside the mask
    dilated = c.mask.copy()
    assert list(dilated[4]) == [50, 255, 255, 255, 255, 255, 255, 255, 50]
    assert (dilated >= m).all()  # dilate never lowers a pixel

    # symmetric: erode never raises a pixel and shrinks the solid core
    assert c.mask_erode(1)
    assert list(c.mask[4]) == [50, 50, 255, 255, 255, 255, 255, 50, 50]
    assert (c.mask <= dilated).all()


def test_mask_clear():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, target="mask")
    assert c.mask_clear()
    assert (c.mask == 0).all()
    assert not c.mask_clear()  # already empty: no-op, no undo step


def test_morphology_noop_on_empty_mask():
    c = blank_canvas()
    n = len(c._mask_undo)
    assert not c.mask_dilate(5)
    assert not c.mask_erode(5)
    assert len(c._mask_undo) == n  # no-ops record no undo step


# ---------------- mask scaling (image-independent) ----------------

def test_mask_scaled_follows_image_size():
    # native 100x100 mask, displayed over a 200x100 image -> fit-within,
    # centered: scale = min(200/100, 100/100) = 1, offset x = (200-100)//2 = 50
    c = blank_canvas(200, 100)
    m = np.zeros((100, 100), dtype=np.uint8)
    m[40:60, 40:60] = 255
    c.set_mask(m)
    scaled = c.mask_scaled()
    assert scaled.shape == (100, 200)  # (H, W) of the image
    assert scaled[40:60, 90:110].max() == 255  # mask sits at x = 50..150
    assert scaled[40:60, 0:49].max() == 0  # left remainder unmasked
    # the served mask.png is at the image size
    import io

    png = Image.open(io.BytesIO(c.to_mask_png_bytes()))
    assert png.size == (200, 100)


def test_mask_stroke_remaps_to_native():
    # image 200x100, native mask 100x100 (scale 1, ox=50): a stroke at canvas
    # x=100 (centre) lands on native x=50
    c = blank_canvas(200, 100)
    m = np.zeros((100, 100), dtype=np.uint8)
    c.set_mask(m)
    c.stroke([(100.0, 50.0)], size=10, target="mask")
    assert c.mask.shape == (100, 100)  # native size unchanged
    assert c.mask[50, 50] == 255  # remapped from canvas (100,50) to native (50,50)
    assert c.mask[50, 90] == 0  # canvas x=140 is native x=90, untouched


def test_dilate_runs_on_native_resolution():
    c = blank_canvas(200, 100)
    m = np.zeros((100, 100), dtype=np.uint8)
    m[48:52, 48:52] = 255
    c.set_mask(m)
    before = (c.mask > 0).sum()
    c.mask_dilate(3)
    assert (c.mask > 0).sum() > before  # dilated in native px


def test_set_mask_identical_pushes_no_step():
    c = blank_canvas()
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 10:20] = 255
    assert c.set_mask(m)
    n = len(c._mask_undo)
    assert not c.set_mask(m.copy())  # identical -> no step
    assert len(c._mask_undo) == n


# ---------------- undo/redo: per-image draw + global mask ----------------

def _two_images(c):
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    c.anchor({a})  # keep A alive while B is imported
    b = c.import_image(_png(np.full((48, 64, 3), 50, np.uint8)))
    c.anchor({a, b})  # both now job-referenced
    return a, b


def test_draw_and_mask_stacks_independent():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, color=(255, 0, 0), target="image")
    c.stroke([(32.0, 24.0)], size=10, target="mask")
    assert c.mask[24, 32] == 255

    c.undo("mask")  # reverts the mask stroke only
    assert c.mask[24, 32] == 0
    assert c.image[24, 32][0] == 255  # image stroke intact
    assert c.can_undo_draw()  # draw stack still has the image stroke

    c.undo("image")  # reverts the image stroke only
    assert c.image[24, 32][0] == 10

    c.redo("image")
    assert c.image[24, 32][0] == 255
    c.redo("mask")
    assert c.mask[24, 32] == 255
    assert not c.can_redo_draw() and not c.can_redo_mask()


def test_new_stroke_clears_only_its_redo():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, target="image")
    c.stroke([(5.0, 5.0)], size=10, target="mask")
    c.undo("image")
    c.undo("mask")
    assert c.can_redo_draw() and c.can_redo_mask()
    # a new image stroke clears only the draw redo, not the mask redo
    c.stroke([(9.0, 9.0)], size=10, target="image")
    assert not c.can_redo_draw()
    assert c.can_redo_mask()


def test_per_image_draw_history_restored_on_reselect():
    c = Canvas()
    a, b = _two_images(c)
    c.set_current(a)
    c.stroke([(8.0, 8.0)], size=8, target="image")
    c.set_current(b)
    c.stroke([(8.0, 8.0)], size=8, target="image")
    # each image kept its own single-stroke history
    assert len(c._images[a].undo) == 1 and len(c._images[b].undo) == 1
    # back to A: its pre-stroke pixels restore on undo, B's stack untouched
    c.set_current(a)
    c.undo("image")
    assert (c.image == 100).all()  # A restored to its original value
    assert len(c._images[b].undo) == 1  # B's stack intact


def test_selection_is_not_an_undo_step():
    c = Canvas()
    a, b = _two_images(c)
    c.set_current(a)
    c.stroke([(8.0, 8.0)], size=8, target="image")
    c.set_current(b)  # switching images pushes nothing
    assert not c.can_undo_draw()  # B has an empty draw stack
    c.set_current(a)
    assert c.can_undo_draw()  # A's stack is restored, not reset


def test_mask_retained_across_image_switch():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, target="mask")
    assert c.mask[24, 32] == 255
    # switching to a differently-sized image keeps (scales) the mask
    c.import_image(_png(np.zeros((32, 24, 3), dtype=np.uint8)))
    assert (c.mask_scaled() > 0).any()


def test_upload_png_info_survives_strokes():
    c = Canvas()
    png = _png_with_info(b"seed: 1")
    c.import_image(png)
    assert c.png_info().get("parameters") == "seed: 1"
    c.stroke([(8.0, 8.0)], size=10, target="image")
    assert c.png_info().get("parameters") == "seed: 1"
    data = c.to_png_bytes()
    assert Image.open(_bytes_io(data)).info.get("parameters") == "seed: 1"


# ---------------- history caps ----------------

def test_draw_history_capped():
    c = blank_canvas(32, 32)
    for i in range(DRAW_HISTORY + 20):
        x = 4.0 + (i % 4) * 6.0
        col = (i // 4) % 2 * 255  # flip the colour every 4 -> each stroke changes
        c.stroke([(x, 16.0)], size=4, color=(col, col, col), target="image")
    assert len(c._images[c.current_id()].undo) == DRAW_HISTORY


def test_mask_history_capped():
    c = blank_canvas(64, 64)
    for i in range(MASK_HISTORY + 20):
        # alternate paint/erase at the same spot -> every stroke changes
        c.stroke([(32.0, 32.0)], size=8, target="mask", erase=(i % 2 == 1))
    assert len(c._mask_undo) == MASK_HISTORY


# ---------------- entity eviction ----------------

def test_upload_evicted_when_it_leaves_canvas():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    b = c.import_image(_png(np.full((24, 32, 3), 50, np.uint8)))
    # a left the canvas and has no job reference -> evicted on import of b
    assert a not in c._images
    assert b in c._images and c.current_id() == b


def test_anchored_entity_survives_leaving_canvas():
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))
    c.anchor({a})
    b = c.import_image(_png(np.full((24, 32, 3), 50, np.uint8)))
    assert a in c._images  # still referenced by a (live) job
    # un-anchoring + evict drops it
    c.anchor(set())
    c.evict_stale()
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
    import io

    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", text.decode())
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _bytes_io(b: bytes):
    import io

    return io.BytesIO(b)
