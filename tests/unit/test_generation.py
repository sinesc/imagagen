"""Unit tests for generation modes: inpaint geometry and canvas mapping."""

import numpy as np
import pytest
from PIL import Image

from imagagen.backends import GenerateRequest
from imagagen.canvas import Canvas
from imagagen.generation import (
    INPAINT_PADDING,
    GenerationError,
    InpaintPlan,
    crop_mask,
    crop_padded,
    fit_ar,
    masked_bbox,
    paste_centered,
    resize_shift_mask,
    with_inpaint,
)

# ------------------------------------------------------------- masked_bbox --


def _mask_with_region(rows, cols, shape=(40, 40)):
    m = np.zeros(shape, dtype=np.uint8)
    m[rows[0] : rows[1], cols[0] : cols[1]] = 255
    return m


def test_masked_bbox_exact_ar():
    # mask 20x10, target ratio 2:1 -> the box is the mask itself
    m = _mask_with_region(rows=(10, 20), cols=(5, 25))
    assert masked_bbox(m, 64, 32) == (5, 10, 20, 10)


def test_masked_bbox_grows_width_for_tall_mask():
    # mask 10x20 (taller than target ratio 2:1) -> grow the width to 40x20,
    # centered on the mask (center 15, 15)
    m = _mask_with_region(rows=(5, 25), cols=(10, 20))
    assert masked_bbox(m, 64, 32) == (-5, 5, 40, 20)


def test_masked_bbox_grows_height_for_wide_mask():
    # mask 40x10 (wider than target ratio 1:2) -> grow the height to 40x80,
    # centered on the mask (center 20, 20) -> top edge hangs off the canvas
    m = _mask_with_region(rows=(15, 25), cols=(0, 40))
    assert masked_bbox(m, 32, 64) == (0, -20, 40, 80)


def test_masked_bbox_may_extend_beyond_canvas():
    # mask hugging the top edge; a 1:1 target ratio grows the 8x4 box to
    # 8x8 centered on it, so the top half hangs off the canvas
    m = _mask_with_region(rows=(0, 4), cols=(0, 8), shape=(20, 20))
    x0, y0, w, h = masked_bbox(m, 64, 64)
    assert (x0, y0, w, h) == (0, -2, 8, 8)
    assert y0 < 0


def test_masked_bbox_empty_raises():
    with pytest.raises(GenerationError):
        masked_bbox(np.zeros((8, 8), dtype=np.uint8), 32, 32)


# ------------------------------------------------------------------- fit_ar --


def test_fit_ar_spec_example():
    # concept.md: 100x100 -> 250x200 scales to 200x200 centered at +25 right
    assert fit_ar(100, 100, 250, 200) == (200, 200, 25, 0)


def test_fit_ar_downscale():
    assert fit_ar(250, 200, 100, 100) == (100, 80, 0, 10)


def test_fit_ar_same_size():
    assert fit_ar(100, 200, 100, 200) == (100, 200, 0, 0)


def test_fit_ar_ar_preserved_both_axes():
    # 3:1 canvas into a 100x100 target: width is the limiting axis
    sw, sh, ox, oy = fit_ar(300, 100, 100, 100)
    assert (sw, sh, ox, oy) == (100, 33, 0, 33)


# ------------------------------------------------------------- crop helpers --


def test_crop_padded_inside():
    img = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    np.testing.assert_array_equal(crop_padded(img, 1, 0, 2, 2), img[:, 1:3])


def test_crop_padded_edge_replication():
    # left half: row values 0/10/20/30; right half: red
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    for y in range(4):
        img[y, :2, 0] = y * 10
    img[:, 2:] = np.dstack([255, 0, 0])
    # box (0,-2,8,8): 2 rows above and below, 4 cols right are off-canvas
    out = crop_padded(img, 0, -2, 8, 8)
    assert out.shape == (8, 8, 3)
    # actual rows in the middle, left half carries the row values
    for y in range(4):
        assert out[2 + y, 0, 0] == y * 10
        assert out[2 + y, 2, 0] == 255
    # top padding replicates row 0, bottom padding replicates row 3
    assert out[0, 0, 0] == 0 and out[1, 0, 0] == 0
    assert out[6, 0, 0] == 30 and out[7, 0, 0] == 30
    # right padding replicates the last (red) column
    assert (out[:, 4:, 0] == 255).all()
    # top/bottom padding also replicates the red half
    assert (out[:2, 4:, 0] == 255).all() and (out[6:, 4:, 0] == 255).all()


def test_crop_mask_zero_outside():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 2:5] = 200
    out = crop_mask(m, -2, -2, 8, 8)
    assert out.shape == (8, 8)
    assert out[2:5, 2:5].max() == 200 and out[0:2, :].max() == 0


def test_resize_shift_mask_spec_example():
    # 100x100 -> 250x200: mask scaled 2x, shifted 25 px right
    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:20, 10:20] = 255
    out = resize_shift_mask(m, 200, 200, 25, 0, 250, 200)
    assert out.shape == (200, 250)
    # the block lands at rows 20:40, cols 45:65; LANCZOS feathers ~2px of
    # the border, so assert on the interior
    assert out[30, 55] == 255
    assert out[22:38, 47:63].min() > 200
    assert out[30, 25:38].max() == 0  # left of the shifted region
    assert out[30, 38:45].max() < 128  # feathered transition (LANCZOS)
    assert out[:, 226:].max() == 0  # right of the placement


def test_paste_centered_edge_replicated_border():
    # R = per-column gradient, bottom row overridden: makes every
    # replicated edge distinct
    scaled = np.zeros((4, 4, 3), dtype=np.uint8)
    for x in range(4):
        scaled[:, x, 0] = x * 10
    scaled[3, :, 0] = 99
    out = paste_centered(scaled, 8, 6)
    assert out.shape == (6, 8, 3)
    np.testing.assert_array_equal(out[1:5, 2:6, 0], scaled[:, :, 0])
    # left/right borders replicate the edge column *of that row* (the bottom
    # row of the image is all 99, its borders too)
    assert (out[1:4, :2, 0] == 0).all() and (out[4, :2, 0] == 99).all()
    assert (out[1:4, 6:, 0] == 30).all() and (out[4, 6:, 0] == 99).all()
    # top/bottom borders replicate the edge rows, corners included
    np.testing.assert_array_equal(out[0, :, 0], [0, 0, 0, 10, 20, 30, 30, 30])
    np.testing.assert_array_equal(out[5, :, 0], [99, 99, 99, 99, 99, 99, 99, 99])
    # no border: identity
    np.testing.assert_array_equal(paste_centered(scaled, 4, 4), scaled)


# -------------------------------------------------------------------- plans --


def _png(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _canvas(w=64, h=48):
    c = Canvas()
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    c.import_image(_png(img))
    return c


def test_inpaint_masked_mask_is_at_image_size():
    # the mask is at the canvas image's size (1:1): inpaint uses it as-is
    c = _canvas(64, 48)
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:22, 10:22] = 255
    c.set_mask(m)
    req = GenerateRequest(prompt="x", width=64, height=64)
    plan = with_inpaint(req, "inpaint_masked", c, 0.8, padding=0)
    assert isinstance(plan, InpaintPlan)
    # the plan's base mask is the mask itself (image size)
    assert plan.base_mask.shape == (48, 64)
    # the bbox crop is fully inside the mask blob
    assert req.init_mask is not None
    assert req.init_mask.shape == (64, 64) and req.init_mask.max() == 255
    assert req.init_mask[32, 32] == 255


def test_generate_mode_is_noop():
    req = GenerateRequest(prompt="x", width=32, height=32)
    assert with_inpaint(req, "generate", _canvas(), 0.75) is None
    assert req.init_image is None


def test_unknown_mode_raises():
    with pytest.raises(GenerationError):
        with_inpaint(GenerateRequest(prompt="x"), "bogus", _canvas(), 1.0)


def test_inpaint_requires_canvas():
    with pytest.raises(GenerationError):
        with_inpaint(GenerateRequest(prompt="x"), "inpaint_masked", Canvas(), 1.0)


def test_inpaint_masked_requires_mask():
    with pytest.raises(GenerationError):
        with_inpaint(GenerateRequest(prompt="x", width=32, height=32),
                     "inpaint_masked", _canvas(), 1.0)


def test_inpaint_masked_populates_request():
    c = _canvas()
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 12:32] = 255
    c.set_mask(m)
    req = GenerateRequest(prompt="x", width=64, height=64)
    plan = with_inpaint(req, "inpaint_masked", c, 0.8, padding=0)
    assert isinstance(plan, InpaintPlan)
    # mask 20x10 at (12,10); target ratio 1:1 -> grow height to a 20x20 box
    # centered on the mask (center 22,15)
    assert plan.bbox == (12, 5, 20, 20)
    assert req.init_image.shape == (64, 64, 3)
    assert req.init_mask.shape == (64, 64)
    assert req.denoise == 0.8
    # the mask region made it into the (upscaled) init mask: the box
    # center is inside the mask, the corner is outside
    assert req.init_mask.max() == 255 and req.init_mask.min() == 0
    assert req.init_mask[32, 32] == 255 and req.init_mask[4, 4] == 0
    # the crop is inside the canvas (uniform value) -> no padding artifacts
    assert (req.init_image == 100).all()


def test_inpaint_masked_context_padding():
    """The box gets a context ring (capped at the mask extent) so the model
    sees the surroundings; the mask sits in the middle of a zero ring."""
    # small mask: padding capped at extent 20 -> box 60x60 around the
    # tight (12,5,20,20)
    c = _canvas()
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 12:32] = 255
    c.set_mask(m)
    req = GenerateRequest(prompt="x", width=64, height=64)
    plan = with_inpaint(req, "inpaint_masked", c, 0.8)
    assert plan.bbox == (12 - 20, 5 - 20, 20 + 40, 20 + 40)
    # mask centred in the padded box: the box centre is masked, the box
    # corners (pure context) are not
    assert req.init_mask[32, 32] == 255
    assert req.init_mask[4, 4] == 0 and req.init_mask[59, 59] == 0

    # large mask (>= INPAINT_PADDING): full 256 px ring
    c2 = _canvas(512, 512)
    m2 = np.zeros((512, 512), dtype=np.uint8)
    m2[100:400, 100:300] = 255  # 300 tall x 200 wide
    c2.set_mask(m2)
    req2 = GenerateRequest(prompt="x", width=64, height=64)
    plan2 = with_inpaint(req2, "inpaint_masked", c2, 0.8)
    # tight 1:1 box is 300x300 centred on the mask: left 50, top 100;
    # the ring adds 256 on every side
    x0, y0, w, h = plan2.bbox
    assert (w, h) == (2 * INPAINT_PADDING + 300, 2 * INPAINT_PADDING + 300)
    assert x0 == 50 - INPAINT_PADDING and y0 == 100 - INPAINT_PADDING


def test_inpaint_masked_postprocess_blends_only_mask_region():
    c = _canvas()
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 12:32] = 255
    c.set_mask(m)
    req = GenerateRequest(prompt="x", width=20, height=10)
    plan = with_inpaint(req, "inpaint_masked", c, 1.0)
    gen = np.full((10, 20, 3), 250, dtype=np.uint8)
    out = plan.postprocess(gen)
    assert out.shape == (48, 64, 3)
    # mask region: fully replaced (alpha 255)
    assert (out[10:20, 12:32] == 250).all()
    # everything else untouched
    assert (np.delete(out, list(range(10, 20)), axis=0) == 100).all()
    # the mask itself is untouched (the result inherits the source's)


def test_inpaint_not_masked_populates_request():
    c = _canvas()  # 64x48
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 12:32] = 255
    c.set_mask(m)
    req = GenerateRequest(prompt="x", width=64, height=48)
    plan = with_inpaint(req, "inpaint_not_masked", c, 0.75)
    # same size: no scale, no border
    assert plan.new_mask.shape == (48, 64)
    np.testing.assert_array_equal(plan.new_mask, m)
    assert req.init_image.shape == (48, 64, 3)
    # init mask is inverted: protect where the user masked
    np.testing.assert_array_equal(req.init_mask, 255 - m)


def test_inpaint_not_masked_outpaint_geometry():
    c = _canvas()  # 64x48
    req = GenerateRequest(prompt="x", width=128, height=48)
    plan = with_inpaint(req, "inpaint_not_masked", c, 1.0)
    # 2x horizontally, centered, no vertical change
    assert req.init_image.shape == (48, 128, 3)
    # the image sits in the middle: cols 32..96, borders edge-replicated
    np.testing.assert_array_equal(req.init_image[:, 32:96], 100)
    assert (req.init_image[:, :32] == 100).all()
    assert (req.init_image[:, 96:] == 100).all()
    assert plan.new_mask.shape == (48, 128)
    assert not plan.new_mask.any()


def test_inpaint_not_masked_postprocess_resizes_rounded_output():
    # the SDXL backend rounds generation sizes to multiples of 8; the canvas
    # must still come out exactly at the requested size
    c = _canvas(100, 100)
    plan = InpaintPlan("inpaint_not_masked", c.image, None,
                       new_mask=np.zeros((100, 100), dtype=np.uint8))
    gen = np.full((96, 96, 3), 42, dtype=np.uint8)
    out = plan.postprocess(gen)
    assert out.shape == (100, 100, 3)
