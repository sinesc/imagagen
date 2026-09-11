"""Canvas v2: brush strokes, mask buffer, mask morphology, undo/redo.

Pure-Python (numpy/PIL/scipy) -- no GPU, no model.
"""

import numpy as np
import pytest
from PIL import Image

from imagagen.canvas import Canvas


def blank_canvas(w=64, h=48, value=10):
    c = Canvas()
    c.set_image(np.full((h, w, 3), value, dtype=np.uint8))
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
    assert c.mask[24, 50] == 255  # grown ring filled solid


def test_mask_clear():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, target="mask")
    assert c.mask_clear()
    assert (c.mask == 0).all()
    assert not c.mask_clear()  # already empty: no-op, no undo step


def test_morphology_noop_on_empty_mask():
    c = blank_canvas()
    n = len(c._undo)
    assert not c.mask_dilate(5)
    assert not c.mask_erode(5)
    assert len(c._undo) == n  # no-ops record no undo step


# ---------------- undo/redo incl. mask ----------------

def test_undo_redo_tracks_image_and_mask():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, color=(255, 0, 0), target="image")
    c.stroke([(32.0, 24.0)], size=10, target="mask")
    assert c.mask[24, 32] == 255

    c.undo()  # reverts the mask stroke
    assert c.mask[24, 32] == 0
    assert c.image[24, 32][0] == 255  # image stroke intact

    c.undo()  # reverts the image stroke
    assert c.image[24, 32][0] == 10

    c.redo()
    c.redo()
    assert c.image[24, 32][0] == 255
    assert c.mask[24, 32] == 255
    assert not c.can_redo()


def test_new_stroke_clears_redo():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, target="image")
    c.undo()
    assert c.can_redo()
    c.stroke([(5.0, 5.0)], size=10, target="image")
    assert not c.can_redo()


def test_set_image_resets_mask():
    c = blank_canvas()
    c.stroke([(32.0, 24.0)], size=10, target="mask")
    assert c.mask[24, 32] == 255
    c.set_image(np.zeros((48, 64, 3), dtype=np.uint8))
    assert (c.mask == 0).all()


def test_set_image_retains_mask_when_given():
    c = blank_canvas()
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 10:20] = 255
    c.set_image(np.zeros((48, 64, 3), dtype=np.uint8), mask=m)
    assert (c.mask[10:20, 10:20] == 255).all()


def test_png_info_survives_strokes():
    c = Canvas()
    png = _png_with_info(b"seed: 1")
    c.set_from_png_bytes(png)
    assert c.png_info().get("parameters") == "seed: 1"
    c.stroke([(32.0, 24.0)], size=10, target="image")
    assert c.png_info().get("parameters") == "seed: 1"
    data = c.to_png_bytes()
    assert Image.open(_bytes_io(data)).info.get("parameters") == "seed: 1"


def test_history_is_capped():
    from imagagen.canvas import MAX_HISTORY

    c = blank_canvas()
    for i in range(MAX_HISTORY + 20):
        c.set_image(np.full((8, 8, 3), i % 256, dtype=np.uint8))
    assert len(c._undo) == MAX_HISTORY
    # still undoable
    for _ in range(MAX_HISTORY):
        c.undo()
    assert not c.can_undo()


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
