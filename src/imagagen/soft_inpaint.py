"""Soft inpainting, ported from Forge's builtin extension `soft-inpainting`.

Instead of hard-locking the protected (unmasked) region to the init latent
at every step, soft inpainting:

1. blends the per-step x0 estimate with the init latent according to the
   mask opacity and the current sigma (``latent_blend`` +
   ``get_modified_nmask``); the vector magnitudes are interpolated under a
   "detail preservation" exponent so the blend does not wash out contrast,
   and the final solver state is NOT hard-locked;
2. after decoding, composites the original image back over the generation
   wherever the latents did not change much (``adaptive_change_alpha``:
   a per-latent-pixel distance threshold, locally percentile-normalised
   and biased by the original mask).

Option names, ranges and defaults match the Forge extension exactly.
Only the per-pixel ``weighted_histogram_filter`` was rewritten
(vectorised numpy); it is numerically equivalent to Forge's Python
per-pixel loop.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel, Field

# Forge builtin extension defaults: (power, scale, detail, mask_inf,
# threshold, contrast)
DEFAULTS = (1, 0.5, 4, 0, 0.5, 2)


class SoftInpainting(BaseModel):
    """The six Forge soft-inpainting sliders."""

    mask_blend_power: float = Field(DEFAULTS[0], ge=0, le=8)
    mask_blend_scale: float = Field(DEFAULTS[1], ge=0, le=8)
    inpaint_detail_preservation: float = Field(DEFAULTS[2], ge=1, le=32)
    composite_mask_influence: float = Field(DEFAULTS[3], ge=0, le=1)
    composite_difference_threshold: float = Field(DEFAULTS[4], ge=0, le=8)
    composite_difference_contrast: float = Field(DEFAULTS[5], ge=0, le=8)


# ---------------------------------------------------------- latent blend --


def latent_blend(s: SoftInpainting, a, b, t):
    """Interpolate two latent vectors according to ``t`` (0 -> a, 1 -> b),
    interpolating the vector magnitudes separately.

    The "detail_preservation" exponent biases the magnitude interpolation
    towards the larger of the two magnitudes, which keeps the contrast
    linear interpolation would wash out (ported verbatim from Forge).
    """
    import torch

    if len(t.shape) == 3:
        # [4][w][h] to [1][4][w][h]
        t2 = t.unsqueeze(0)
        # [4][w][h] to [1][1][w][h] (the [4] is redundant: all channels
        # carry the same mask, Forge uses channel 0)
        t3 = t[0].unsqueeze(0).unsqueeze(0)
    else:
        t2 = t
        t3 = t[:, 0][:, None]

    # linearly interpolate the image vectors
    image_interp = a * (1 - t2) + b * t2
    result_type = image_interp.dtype

    # 64-bit operations for the magnitude pass (large exponents)
    current_magnitude = torch.norm(image_interp, p=2, dim=1, keepdim=True)
    current_magnitude = current_magnitude.to(torch.float64) + 1e-5

    a_magnitude = torch.norm(a, p=2, dim=1, keepdim=True).to(torch.float64)
    a_magnitude = a_magnitude.pow(s.inpaint_detail_preservation) * (1 - t3)
    b_magnitude = torch.norm(b, p=2, dim=1, keepdim=True).to(torch.float64)
    b_magnitude = b_magnitude.pow(s.inpaint_detail_preservation) * t3
    desired_magnitude = (a_magnitude + b_magnitude)
    desired_magnitude = desired_magnitude.pow(1 / s.inpaint_detail_preservation)

    return image_interp * (desired_magnitude / current_magnitude).to(result_type)


def get_modified_nmask(s: SoftInpainting, nmask, sigma):
    """Scale the mask (0 = fully protected, 1 = fully inpaint) according to
    the current denoising sigma (ported from Forge).

    Bringing the transparency to a power simulates N blending operations
    with any positive real N, shifting the balance between the denoiser and
    the original latents across the sigma schedule.
    """
    import torch

    return torch.pow(nmask, (sigma ** s.mask_blend_power) * s.mask_blend_scale)


# ------------------------------------------------ pixel composite (numpy) --


def smootherstep(x):
    """The smootherstep function (input clamped to 0-1): a flat-topped
    sigmoid-like curve (ported from Forge)."""
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def get_gaussian_kernel(stddev_radius: float = 1.0, max_radius: int = 2):
    """A Gaussian kernel with thresholded edges: taper the ends to zero one
    pixel beyond ``max_radius`` while keeping the peak at 1 (ported from
    Forge). Returns (kernel (2r+1, 2r+1), center)."""

    def gaussian(sqr_mag):
        return math.exp(-sqr_mag / (stddev_radius * stddev_radius))

    zero_radius = max_radius + 1.0
    gauss_zero = gaussian(zero_radius * zero_radius)
    gauss_kernel_scale = 1 / (1 - gauss_zero)

    size = max_radius * 2 + 1
    ys, xs = np.mgrid[0:size, 0:size]
    d2 = (xs - max_radius) ** 2 + (ys - max_radius) ** 2
    kernel = np.exp(-d2 / (stddev_radius * stddev_radius)) - gauss_zero
    kernel *= gauss_kernel_scale
    np.maximum(kernel, 0.0, out=kernel)
    return kernel, max_radius


def weighted_histogram_filter(
    img: np.ndarray,
    kernel: np.ndarray,
    kernel_center: int,
    percentile_min: float = 0.0,
    percentile_max: float = 1.0,
    min_width: float = 1.0,
) -> np.ndarray:
    """Generalised convolution filter capable of applying weighted mean,
    median, maximum, and minimum filters parametrically using an arbitrary
    kernel (Forge's weighted_histogram_filter, vectorised).

    For every pixel the kernel window is collected (out-of-bounds samples
    excluded, their weights dropped), sorted by value, and the weighted
    average across the ``[percentile_min, percentile_max]`` slice of the
    cumulative weight stack is returned (the stack widened to at least
    ``min_width``). Numerically equivalent to Forge's per-pixel Python
    loop (checked by test against it).
    """
    H, W = img.shape
    k = kernel.shape[0]
    kc = kernel_center
    # vals[n] / wts[n] hold, for each destination pixel, the image value and
    # its kernel weight at the pixel offset by the nth kernel step.
    # Out-of-bounds: value 0 AND weight 0 (Forge drops those samples from
    # the histogram entirely; a nonzero weight would skew the stack).
    vals = np.zeros((k * k, H, W), dtype=np.float64)
    wts = np.zeros((k * k, H, W), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            dr, dc = i - kc, j - kc
            r0, r1 = max(0, -dr), min(H, H - dr)
            c0, c1 = max(0, -dc), min(W, W - dc)
            n = i * k + j
            vals[n][r0:r1, c0:c1] = img[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            wts[n][r0:r1, c0:c1] = kernel[i][j]

    # per-pixel sort; ties are irrelevant (equal values integrate the same).
    # vals has shape (N, H, W), order (N, H, W): after the take, sv[i, j, k]
    # is the value at rank order[i, j, k] of pixel (j, k)'s sorted window.
    order = np.argsort(vals, axis=0)
    sv = np.take_along_axis(vals, order, axis=0)
    sw = np.take_along_axis(wts, order, axis=0)
    cum = np.concatenate(
        [np.zeros((1, H, W), dtype=np.float64), np.cumsum(sw, axis=0)], axis=0
    )

    total = cum[-1]
    window_min = total * percentile_min
    window_max = total * percentile_max
    narrow = (window_max - window_min) < min_width
    center = (window_min + window_max) / 2
    wm = np.where(narrow, center - min_width / 2, window_min)
    wx = np.where(narrow, center + min_width / 2, window_max)
    over = wx > total
    wx = np.where(over, total, wx)
    wm = np.where(over, total - min_width, wm)
    under = wm < 0
    wm = np.where(under, 0.0, wm)
    wx = np.where(under, min_width, wx)

    overlap = np.clip(np.minimum(wx, cum[1:]) - np.maximum(wm, cum[:-1]), 0, None)
    out = np.zeros((H, W), dtype=np.float64)
    np.divide((sv * overlap).sum(axis=0), overlap.sum(axis=0), out=out,
              where=overlap.sum(axis=0) > 0)
    return out


def adaptive_change_alpha(
    s: SoftInpainting,
    nmask,
    latent_orig,
    latent_processed,
    w: int,
    h: int,
) -> np.ndarray:
    """Forge's ``apply_adaptive_masks`` mask computation (batch = 1).

    Returns a float64 (h, w) map, 0 = keep the original pixels,
    1 = use the generated pixels. The per-latent-pixel distance between
    the processed and the original latent is locally percentile-normalised
    (two weighted-histogram passes), thresholded at
    ``composite_difference_threshold`` scaled by the original mask, shaped
    by ``composite_difference_contrast`` + smootherstep, upscaled to the
    image size (LANCZOS, Forge's ``images.resize_image(2, ...)``) and
    de-aliased with a Gaussian blur (radius 4).

    ``nmask``: (4, lh, lw) tensor/array, 1 = inpaint region. The latents
    are (1, 4, lh, lw) in the VAE-scaled sampling space (as decoded from).
    """
    import torch

    def as_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    shape = tuple(nmask.shape)  # works for torch tensors and ndarrays alike
    if len(shape) == 3:         # (4, lh, lw): all channels identical
        latent_mask = as_np(nmask[0])
    else:                        # (lh, lw)
        latent_mask = as_np(nmask)
    latent_mask = latent_mask.astype(np.float64)
    # 0 = inpaint region, 1 = protected: how much the original mask biases
    # the difference threshold (scaled distances for thresholding)
    mask_scalar = 1.0 - np.clip(latent_mask, 0, 1) ** (s.mask_blend_scale / 2)
    mask_scalar = (0.5 * (1 - s.composite_mask_influence)
                   + mask_scalar * s.composite_mask_influence)
    mask_scalar = mask_scalar / (1.00001 - mask_scalar)

    diff = as_np(latent_processed - latent_orig)
    dist = np.linalg.norm(diff, axis=1)[0]

    kernel, kernel_center = get_gaussian_kernel(stddev_radius=1.5, max_radius=2)
    dist = weighted_histogram_filter(dist, kernel, kernel_center,
                                     percentile_min=0.9, percentile_max=1, min_width=1)
    dist = weighted_histogram_filter(dist, kernel, kernel_center,
                                     percentile_min=0.25, percentile_max=0.75,
                                     min_width=1)

    # the distance at which the opacity of the original decreases to 50%
    half_weighted_distance = s.composite_difference_threshold * mask_scalar
    with np.errstate(divide="ignore", invalid="ignore"):
        converted = dist / half_weighted_distance
    converted = 1.0 / (1.0 + np.nan_to_num(converted, nan=0.0, posinf=np.inf)
                       ** s.composite_difference_contrast)
    converted = 1.0 - smootherstep(converted)
    converted = np.clip(converted, 0, 1)

    im = Image.fromarray((255.0 * converted).astype(np.uint8))
    im = im.resize((w, h), Image.LANCZOS)
    # remove aliasing artifacts (Forge: GaussianBlur radius 4)
    im = im.filter(ImageFilter.GaussianBlur(radius=4))
    return np.asarray(im, dtype=np.float64) / 255.0
