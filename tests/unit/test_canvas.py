"""Canvas v2: per-image draw history, per-image mask (inherited from the
source image), mask morphology, entity eviction.

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


# ---------------- clone ----------------

def test_clone_copies_source_pixels():
    # left band L, right band R; clone from the left onto the right with an
    # offset that maps the destination footprint back into the left band
    left, right = (123, 45, 67), (10, 20, 30)
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = left
    arr[:, 32:] = right
    c = Canvas()
    c.import_image(_png(arr))
    changed = c.clone([(44.0, 24.0)], offset=(-30, 0), size=20, opacity=1.0)
    assert changed
    assert c.image[24, 44].tolist() == list(left)  # destination = source px
    assert c.image[24, 10].tolist() == list(left)  # source region untouched
    assert c.image[24, 55].tolist() == list(right)  # outside the footprint
    assert c.image[0, 63].tolist() == list(right)  # far corner untouched


def test_clone_samples_the_pre_stroke_image():
    # two stamps of one stroke whose source regions overlap the other
    # stamp's destination: the second stamp must still read the ORIGINAL
    # pixels, not the first stamp's output (progressive sampling bug)
    a, b, ccol = (100, 0, 0), (0, 100, 0), (0, 0, 100)
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :16] = a
    arr[:, 16:40] = b
    arr[:, 40:] = ccol
    c = Canvas()
    c.import_image(_png(arr))
    changed = c.clone([(24.0, 24.0), (25.0, 24.0)], offset=(14, 0), size=20)
    assert changed
    # stamp 2 (centre 25) samples x 29..49, which overlaps stamp 1's
    # destination x 14..34: pre-stroke it reads band B there; a
    # progressively-edited image would hand it stamp 1's band-C output
    assert c.image[24, 16].tolist() == list(b)
    assert c.image[24, 20].tolist() == list(b)


def test_clone_source_off_edge_clips():
    # source shifted off the left edge: no crash, the in-range portion is
    # copied 1:1 and off-canvas source pixels leave the destination as-is
    left, rest = (100, 0, 0), (0, 100, 0)
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :4] = left
    arr[:, 4:] = rest
    c = Canvas()
    c.import_image(_png(arr))
    changed = c.clone([(4.0, 24.0)], offset=(-6, 0), size=12, opacity=1.0)
    assert changed
    # dest x 6,8 <- source x 0,2 (inside: band `left`), full alpha there
    assert c.image[24, 6].tolist() == list(left)
    assert c.image[24, 8].tolist() == list(left)
    # dest x 2 <- source x -4 (off-canvas): destination unchanged
    assert c.image[24, 2].tolist() == list(left)


def test_clone_opacity_does_not_stack_overlapping_stamps():
    # one continuous stroke = one paint at the stroke's opacity: the
    # overlapping stamps must NOT accumulate (0.5 stays 0.5 at the core)
    b, a = (0, 100, 0), (100, 0, 0)
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = b
    arr[:, 32:] = a
    c = Canvas()
    c.import_image(_png(arr))
    changed = c.clone([(16.0, 24.0), (17.0, 24.0)], offset=(20, 0), size=20,
                      opacity=0.5)
    assert changed
    # core pixel covered by BOTH stamps: exactly the 0.5 blend, not the
    # ~0.75+ a per-stamp sequential blend would stack up to
    assert c.image[24, 20].tolist() == [50, 50, 0]


def test_clone_opacity_blending():
    a, b = (100, 0, 0), (0, 100, 0)
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = a
    arr[:, 32:] = b
    # opacity 0.5: destination centre is the midpoint of source + destination
    c = Canvas()
    c.import_image(_png(arr))
    c.clone([(32.0, 24.0)], offset=(-30, 0), size=20, opacity=0.5)
    assert c.image[24, 32].tolist() == [50, 50, 0]
    # opacity 1.0: hard copy at the centre (full brush alpha)
    c2 = Canvas()
    c2.import_image(_png(arr))
    c2.clone([(32.0, 24.0)], offset=(-30, 0), size=20, opacity=1.0)
    assert c2.image[24, 32].tolist() == list(a)


def test_clone_noop_returns_false():
    # offset (0,0) copies identical pixels: nothing changes, no history step
    arr = np.full((48, 64, 3), 10, np.uint8)
    c = Canvas()
    c.import_image(_png(arr))
    assert not c.clone([(32.0, 24.0)], offset=(0, 0), size=20)
    assert not c.can_undo_draw() and not c.can_redo_draw()
    assert (c.image == arr).all()


def test_clone_empty_canvas_raises():
    c = Canvas()
    with pytest.raises(ValueError):
        c.clone([(1.0, 1.0)], offset=(1, 1), size=10)


def test_clone_history_is_one_draw_step():
    left, right = (123, 45, 67), (10, 20, 30)
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = left
    arr[:, 32:] = right
    c = Canvas()
    c.import_image(_png(arr))
    c.clone([(44.0, 24.0)], offset=(-30, 0), size=20)
    assert c.can_undo_draw() and not c.can_redo_draw()
    assert len(c._images[c.current_id()].undo) == 1
    c.undo("image")
    assert (c.image == arr).all()
    assert not c.can_undo_draw() and c.can_redo_draw()
    c.redo("image")
    assert c.image[24, 44].tolist() == list(left)
    assert not c.can_redo_draw()


# ---------------- smudge ----------------

def test_smudge_single_point_is_noop():
    # smudge is a drag: one point has no previous stamp to sample
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = 255
    c = Canvas()
    c.import_image(_png(arr))
    assert not c.smudge([(30.0, 24.0)], size=20)
    assert (c.image == arr).all()
    assert not c.can_undo_draw() and not c.can_redo_draw()


def test_smudge_uniform_image_is_noop():
    # dragging a uniform image moves identical pixels: nothing changes
    arr = np.full((48, 64, 3), 10, np.uint8)
    c = Canvas()
    c.import_image(_png(arr))
    assert not c.smudge([(20.0, 24.0), (40.0, 24.0)], size=20)
    assert not c.can_undo_draw() and not c.can_redo_draw()


def test_smudge_is_sequential():
    # white left half, black right half. A SEQUENTIAL smear carries the
    # white the whole way (each stamp drags what the previous one moved);
    # sampling the pre-stroke image would stop short of the white region.
    # The source region is untouched (a copy, not a move).
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = 255
    c = Canvas()
    c.import_image(_png(arr))
    changed = c.smudge([(16.0, 24.0), (56.0, 24.0)], size=20, opacity=1.0)
    assert changed
    img = c.image
    assert img[24, 48].tolist() == [255, 255, 255]  # well past the white half
    assert img[24, 56].tolist() == [255, 255, 255]  # the drag endpoint
    assert img[24, 63].tolist() == [255, 255, 255]  # up to the canvas edge
    assert img[24, 10].tolist() == [255, 255, 255]  # source region untouched
    assert img[0, 60].tolist() == [0, 0, 0]  # outside the stamp rows


def test_smudge_off_edge_clips():
    # dragging into/past the right edge: no crash, the in-bounds part works
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = 255
    c = Canvas()
    c.import_image(_png(arr))
    changed = c.smudge([(32.0, 24.0), (80.0, 24.0)], size=20, opacity=1.0)
    assert changed
    assert c.image[24, 60].tolist() == [255, 255, 255]
    assert c.image[24, 63].tolist() == [255, 255, 255]
    # up-and-left corner stamp clipped the same way
    arr2 = np.zeros((48, 64, 3), np.uint8)
    arr2[:, :32] = 255
    c2 = Canvas()
    c2.import_image(_png(arr2))
    changed = c2.smudge([(4.0, 6.0), (40.0, 6.0)], size=20, opacity=1.0)
    assert changed
    assert c2.image[6, 38].tolist() == [255, 255, 255]


def test_smudge_blends_by_opacity():
    # A=(200,0,0) left of x=16, black elsewhere. A 0.5-opacity drag decays
    # as it propagates (each stamp blends the previous one's output), and
    # the original blob keeps its value (it copies A onto A there).
    a = (200, 0, 0)
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :16] = a
    c = Canvas()
    c.import_image(_png(arr))
    changed = c.smudge([(12.0, 24.0), (24.0, 24.0)], size=20, opacity=0.5)
    assert changed
    img = c.image
    assert img[24, 8].tolist() == list(a)  # inside the original blob: A
    assert img[24, 13].tolist() == list(a)  # copies A onto A
    # the trail decays: 154 -> 100 -> 0, strictly between source and black
    assert img[24, 20][0] == 154
    assert img[24, 22][0] == 100
    assert img[24, 24][0] == 45
    assert img[24, 30][0] == 0


def test_smudge_empty_canvas_raises():
    c = Canvas()
    with pytest.raises(ValueError):
        c.smudge([(1.0, 1.0), (5.0, 1.0)], size=10)


def test_smudge_history_is_one_draw_step():
    arr = np.zeros((48, 64, 3), np.uint8)
    arr[:, :32] = 255
    c = Canvas()
    c.import_image(_png(arr))
    c.smudge([(16.0, 24.0), (56.0, 24.0)], size=20)
    assert c.can_undo_draw() and not c.can_redo_draw()
    assert len(c._images[c.current_id()].undo) == 1
    c.undo("image")
    assert (c.image == arr).all()
    assert not c.can_undo_draw() and c.can_redo_draw()
    c.redo("image")
    assert c.image[24, 56].tolist() == [255, 255, 255]
    assert not c.can_redo_draw()


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


def test_mask_stroke_opacity_does_not_stack():
    # like image/clone strokes, one stroke paints at a fixed alpha:
    # overlapping stamps (and repeated strokes) do not stack the opacity
    c = blank_canvas()
    c.stroke([(32.0, 24.0), (52.0, 24.0)], size=20, opacity=0.5, target="mask")
    assert c.mask[24, 32] == 127  # 0.5 * 255, truncated to int
    assert c.mask[24, 44] == 127  # second stamp of the same stroke
    c.stroke([(32.0, 24.0)], size=20, opacity=0.5, target="mask")
    assert c.mask[24, 32] == 127  # a second stroke does not add on


def test_mask_erase_caps_alpha_by_opacity():
    # erasing caps the remaining alpha at 255 * (1-opacity): 0.5 caps at
    # 127.5, not subtractive 255 - 127.5
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=20, opacity=1.0, target="mask")
    assert c.mask[24, 32] == 255
    c.stroke([(32.0, 24.0)], size=20, opacity=0.5, erase=True, target="mask")
    assert c.mask[24, 32] == 127
    # and like add, erase does not stack
    c.stroke([(32.0, 24.0)], size=20, opacity=0.5, erase=True, target="mask")
    assert c.mask[24, 32] == 127


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
    c.set_mask(m)

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
    n = len(c._images[c.current_id()].mask_undo)
    assert not c.mask_dilate(5)
    assert not c.mask_erode(5)
    assert len(c._images[c.current_id()].mask_undo) == n  # no-ops push nothing


# ---------------- mask is at the image's size (1:1) ----------------

def test_mask_is_at_image_size_and_served_directly():
    c = blank_canvas(200, 100)
    m = np.zeros((100, 200), dtype=np.uint8)
    m[40:60, 90:110] = 255
    assert c.set_mask(m)
    np.testing.assert_array_equal(c.mask, m)
    # the served mask.png is the mask itself (at the image's size)
    import io

    png = Image.open(io.BytesIO(c.to_mask_png_bytes()))
    assert png.size == (200, 100)


def test_mask_stroke_is_1_to_1_with_canvas():
    # no remap: a stroke at canvas (x, y) lands on mask pixel (x, y);
    # the whole canvas is maskable, edges included (the regression this
    # fixes: with a drifting native size the top/bottom bands were not)
    c = blank_canvas(200, 100)
    c.stroke([(100.0, 50.0)], size=10, target="mask")
    assert c.mask.shape == (100, 200)
    assert c.mask[50, 100] == 255
    c.stroke([(100.0, 1.0)], size=4, target="mask")
    assert c.mask[0, 100] >= 128


def test_dilate_runs_on_image_pixels():
    c = blank_canvas(200, 100)
    m = np.zeros((100, 200), dtype=np.uint8)
    m[48:52, 48:52] = 255
    c.set_mask(m)
    before = (c.mask > 0).sum()
    c.mask_dilate(3)
    assert (c.mask > 0).sum() > before


def test_set_mask_identical_pushes_no_step():
    c = blank_canvas()
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 10:20] = 255
    assert c.set_mask(m)
    ent = c._images[c.current_id()]
    n = len(ent.mask_undo)
    assert not c.set_mask(m.copy())  # identical -> no step
    assert len(ent.mask_undo) == n


# ---------------- undo/redo: per-image draw + per-image mask ----------------

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


def test_new_image_inherits_source_mask_and_history():
    # a new image (upload / result) inherits the source image's mask state:
    # the current mask transformed to the new size plus both history
    # stacks (cold frames at the source's size, transformed on first use)
    c = blank_canvas(64, 48)
    c.stroke([(32.0, 24.0)], size=20, target="mask")
    ma = c.mask.copy()
    src = c.current_id()
    c.anchor({src})  # keep the source alive across the import
    b = c.import_image(_png(np.zeros((24, 32, 3), dtype=np.uint8)))
    c.anchor({src, b})
    # inherited current mask: fit-within, centered (2x downscale -> 1:1)
    assert c.mask is not None and c.mask.shape == (24, 32)
    assert c.mask[12, 16] == 255  # the dot, at the 2x-downscaled position
    # history inherited: undo walks back through the source's strokes
    assert c.can_undo_mask()
    c.undo("mask")
    assert c.mask is not None and c.mask.shape == (24, 32)
    assert (c.mask == 0).all()
    c.redo("mask")
    assert c.mask[12, 16] == 255
    # ...and the source's mask state is untouched
    c.set_current(src)
    np.testing.assert_array_equal(c.mask, ma)


def test_add_result_inherits_mask_from_source():
    c = blank_canvas(64, 48)
    c.stroke([(32.0, 24.0)], size=20, target="mask")
    src = c.current_id()
    # same-size result: the mask is copied 1:1
    eid = c.add_result(1, 0, _png(np.full((48, 64, 3), 7, np.uint8)), None,
                      source_id=src)
    ent = c._images[eid]
    np.testing.assert_array_equal(ent.mask, c._images[src].mask)
    assert len(ent.mask_undo) == 1 and len(ent.mask_redo) == 0
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
    assert ent.mask_undo == [] and ent.mask_redo == []
    # the mask still exists (empty): once displayed, mask.png is 200, not 404
    c.set_current(eid)
    assert c.to_mask_png_bytes() is not None


def test_mask_per_image_isolation():
    # the regression this rework fixes: a mask painted on one image must
    # not be clobbered by, nor leak into, another image of a different size
    c = Canvas()
    a = c.import_image(_png(np.full((24, 32, 3), 100, np.uint8)))  # 32x24
    c.anchor({a})
    c.stroke([(8.0, 6.0)], size=8, target="mask")
    ma = c.mask.copy()
    assert (ma > 0).any()
    # a differently-sized image is imported: it inherits A's mask...
    b = c.import_image(_png(np.full((48, 64, 3), 50, np.uint8)))  # 64x48
    c.anchor({a, b})
    assert c.mask is not None and (c.mask > 0).any()  # the blob, scaled 2x
    # ...which B may now edit independently
    assert c.mask_clear()
    # ...without touching A: back on A its mask and history are intact
    c.set_current(a)
    np.testing.assert_array_equal(c.mask, ma)
    assert len(c._images[a].mask_undo) == 1
    # ...and B's edited mask and (longer, inherited+own) history too
    c.set_current(b)
    assert (c.mask == 0).all()
    assert len(c._images[b].mask_undo) == 2
    # undo on B walks back through the inherited frame (lazy transform)
    c.undo("mask")
    assert (c.mask > 0).any()  # the inherited blob, at B's size
    c.undo("mask")
    assert (c.mask == 0).all()


def test_upload_png_info_survives_strokes():
    c = Canvas()
    png = _png_with_info(b"seed: 1")
    c.import_image(png)
    assert c.png_info().get("parameters") == "seed: 1"
    c.stroke([(8.0, 8.0)], size=10, target="image")
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
    # the crop's own history stacks start empty
    ent = c._images[eid]
    assert ent.undo == [] and ent.redo == []
    assert ent.mask_undo == [] and ent.mask_redo == []
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
    c = Canvas()
    c.import_image(_png(np.zeros((16, 16, 3), np.uint8)))
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
    assert len(c._images[c.current_id()].mask_undo) == MASK_HISTORY


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
