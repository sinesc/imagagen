"""Unit tests for soft inpainting (port of Forge's builtin extension).

The port must behave like the original: defaults/ranges match the Forge
sliders, the latent blend is exact at the endpoints, and the vectorised
weighted histogram filter is numerically identical to Forge's per-pixel
Python reference (kept verbatim here as the oracle).
"""

import math

import numpy as np
import pytest
import torch

from imagagen.metadata import build_parameters, parse_parameters
from imagagen.soft_inpaint import (
    DEFAULTS,
    SoftInpainting,
    adaptive_change_alpha,
    get_gaussian_kernel,
    get_modified_nmask,
    latent_blend,
    smootherstep,
    weighted_histogram_filter,
)

# ------------------------------------------------- Forge reference (oracle) --


def forge_get_gaussian_kernel(stddev_radius=1.0, max_radius=2):
    """Forge's get_gaussian_kernel, verbatim."""

    def gaussian(sqr_mag):
        return math.exp(-sqr_mag / (stddev_radius * stddev_radius))

    zero_radius = max_radius + 1.0
    gauss_zero = gaussian(zero_radius * zero_radius)
    gauss_kernel_scale = 1 / (1 - gauss_zero)
    kernel = []
    for x in range(max_radius * 2 + 1):
        row = []
        for y in range(max_radius * 2 + 1):
            value = gaussian((x - max_radius) ** 2 + (y - max_radius) ** 2)
            row.append(max(value - gauss_zero, 0) * gauss_kernel_scale)
        kernel.append(row)
    return kernel, max_radius


class _ForgeElement:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight


def forge_weighted_histogram_filter(img, kernel, kernel_center,
                                    percentile_min=0.0, percentile_max=1.0,
                                    min_width=1.0):
    """Forge's weighted_histogram_filter single-pixel loop, verbatim port
    (the scalar reference the vectorised version must match)."""
    result = np.zeros(img.shape)
    height, width = img.shape
    kernel_width = kernel_center * 2 + 1
    for x in range(height):
        for y in range(width):
            values = []
            for i in range(kernel_width):
                for j in range(kernel_width):
                    xx = x + i - kernel_center
                    yy = y + j - kernel_center
                    if xx < 0 or xx >= height or yy < 0 or yy >= width:
                        continue
                    values.append(_ForgeElement(img[xx, yy], kernel[i][j]))
            values.sort(key=lambda e: e.value)
            total_weight = sum(e.weight for e in values)
            window_min = total_weight * percentile_min
            window_max = total_weight * percentile_max
            window_width = window_max - window_min
            if window_width < min_width:
                window_center = (window_min + window_max) / 2
                window_min = window_center - min_width / 2
                window_max = window_center + min_width / 2
                if window_max > total_weight:
                    window_max = total_weight
                    window_min = total_weight - min_width
                if window_min < 0:
                    window_min = 0
                    window_max = min_width
            value, value_weight, weight_so_far = 0.0, 0.0, 0.0
            for e in values:
                start_weight = max(window_min, weight_so_far)
                end_weight = min(window_max, weight_so_far + e.weight)
                overlap = max(end_weight - start_weight, 0)
                value += e.value * overlap
                value_weight += overlap
                weight_so_far += e.weight
            result[x, y] = value / value_weight if value_weight != 0 else 0
    return result


# -------------------------------------------------------------- defaults --


def test_defaults_match_forge():
    s = SoftInpainting()
    assert s.enabled is True  # client always sends the object; flag toggles
    assert (s.mask_blend_power, s.mask_blend_scale,
            s.inpaint_detail_preservation, s.composite_mask_influence,
            s.composite_difference_threshold, s.composite_difference_contrast
            ) == (1.0, 0.5, 4.0, 0.0, 0.5, 2.0)
    assert DEFAULTS == (1, 0.5, 4, 0, 0.5, 2)


def test_ranges_match_forge_sliders():
    # Forge sliders: (0,8,0.1) (0,8,0.05) (1,32,0.5) (0,1,0.05) (0,8,0.25) (0,8,0.25)
    with pytest.raises(ValueError):
        SoftInpainting(mask_blend_power=8.01)
    with pytest.raises(ValueError):
        SoftInpainting(inpaint_detail_preservation=0.5)
    with pytest.raises(ValueError):
        SoftInpainting(composite_mask_influence=1.5)
    SoftInpainting(mask_blend_power=8, inpaint_detail_preservation=32,
                   composite_mask_influence=1, composite_difference_contrast=8)


# --------------------------------------------------------- latent blend --


def test_latent_blend_endpoints():
    s = SoftInpainting()
    a = torch.randn(1, 4, 8, 8) * 2
    b = torch.randn(1, 4, 8, 8) * 2
    t0 = torch.zeros(4, 8, 8)
    t1 = torch.ones(4, 8, 8)
    assert torch.allclose(latent_blend(s, a, b, t0), a, atol=1e-4)
    assert torch.allclose(latent_blend(s, a, b, t1), b, atol=1e-4)


def test_latent_blend_detail_preservation_biases_magnitude_up():
    # blending a magnitude-10 vector with a magnitude-1 vector at t=0.5:
    # the linear midpoint is 5.5, the detail-preservation exponent (4)
    # biases the magnitude towards the maximum (Forge's "contrast boost")
    s = SoftInpainting()
    a = torch.tensor([[[[10.0, 0.0, 0.0, 0.0]]]])
    b = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    t = torch.tensor([[[[0.5] * 4]]])
    r = latent_blend(s, a, b, t)
    mag = r.abs().sum().item()
    assert mag > 7  # clearly above the linear midpoint


# ---------------------------------------------------- get_modified_nmask --


def test_get_modified_nmask_power_and_scale():
    # t = nmask ** (sigma^power * scale); sigma=4, power=1, scale=0.5 -> 0.5**2
    s = SoftInpainting()
    n = torch.full((4, 4, 4), 0.5)
    assert torch.allclose(get_modified_nmask(s, n, 4.0), torch.full_like(n, 0.25))
    # power 0: sigma-independent, t = nmask ** scale
    s0 = SoftInpainting(mask_blend_power=0)
    assert torch.allclose(get_modified_nmask(s0, n, 99.0),
                          torch.full_like(n, 0.5 ** 0.5))


# ------------------------------------------------- histogram filter port --


@pytest.mark.parametrize("percentiles", [(0.9, 1.0, 1.0), (0.25, 0.75, 1.0),
                                         (0.0, 1.0, 1.0), (0.4, 0.6, 1.0)])
def test_weighted_histogram_filter_matches_forge(percentiles):
    """The vectorised filter must be numerically identical to Forge's
    per-pixel reference, including the image borders (where Forge drops
    out-of-bounds samples from the histogram entirely)."""
    rng = np.random.default_rng(0)
    kref, c = forge_get_gaussian_kernel(1.5, 2)
    k, kc = get_gaussian_kernel(1.5, 2)
    assert np.allclose(np.asarray(kref, float), k, atol=1e-12)
    assert kc == c
    for _ in range(4):
        img = rng.random((17, 19)) * 10
        pm, px, mw = percentiles
        ref = forge_weighted_histogram_filter(img, np.asarray(kref, float), c,
                                              pm, px, mw)
        vec = weighted_histogram_filter(img, k, kc, pm, px, mw)
        assert vec.shape == img.shape
        assert np.abs(ref - vec).max() < 1e-9


def test_weighted_histogram_filter_constant_image():
    k, c = get_gaussian_kernel(1.5, 2)
    img = np.full((9, 11), 3.7)
    for pm, px in [(0.9, 1.0), (0.25, 0.75), (0.0, 1.0)]:
        out = weighted_histogram_filter(img, k, c, pm, px, 1.0)
        assert np.allclose(out, 3.7, atol=1e-6)


# ------------------------------------------------------------ misc maths --


def test_smootherstep_flat_tops():
    x = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    y = smootherstep(x)
    # flat-topped sigmoid: 0 at 0, 1 at 1, 0.5 at 0.5, flat near both ends
    assert y[0] == 0 and y[-1] == 1
    assert abs(y[2] - 0.5) < 1e-9
    assert y[1] < 0.01
    assert y[3] > 0.99


# -------------------------------------------------- adaptive change alpha --


def _rand_latent(shape=(16, 16)):
    return torch.randn(1, 4, *shape)


def test_adaptive_change_alpha_no_change_is_zero():
    # nmask: 1 = inpaint region, 0 = protected
    s = SoftInpainting()
    lat = _rand_latent()
    a = adaptive_change_alpha(s, torch.zeros(4, 16, 16), lat, lat.clone(), 128, 128)
    assert a.shape == (128, 128)
    assert a.max() < 0.05


def test_adaptive_change_alpha_full_change_is_one():
    # with the default mask influence (0) the mask is ignored, so a large
    # change anywhere reads as "changed" (alpha ~1, generated content)
    s = SoftInpainting()
    lat = _rand_latent()
    for nmask in (torch.zeros(4, 16, 16), torch.ones(4, 16, 16)):
        a = adaptive_change_alpha(s, nmask, lat, lat + 1.0, 128, 128)
        assert a.min() > 0.95


def test_adaptive_change_alpha_protected_region_keeps_original():
    # with mask influence 0 the threshold is ~unscaled; a small change in a
    # fully protected region (nmask=1) stays below the 0.5 threshold, so the
    # original pixels are kept (alpha near 0)
    s = SoftInpainting()
    lat = _rand_latent()
    nmask = torch.ones(4, 16, 16)
    a = adaptive_change_alpha(s, nmask, lat, lat + 0.05, 128, 128)
    assert a.max() < 0.5


def test_adaptive_change_alpha_mask_influence_follows_mask():
    # mask influence 1 ("follow the mask"): in the inpaint region (nmask=1)
    # the threshold collapses to ~0, so even a tiny change reads as
    # "changed" (alpha ~1, generated content); in the protected region
    # (nmask=0) the threshold becomes huge, so the same tiny change keeps
    # the original (alpha ~0)
    s = SoftInpainting(composite_mask_influence=1.0)
    lat = _rand_latent()
    a = adaptive_change_alpha(s, torch.ones(4, 16, 16), lat, lat + 0.01, 64, 64)
    assert a.min() > 0.9
    b = adaptive_change_alpha(s, torch.zeros(4, 16, 16), lat, lat + 0.01, 64, 64)
    assert b.max() < 0.2


# ---------------------------------------------------------------- metadata --


def test_metadata_soft_inpainting_roundtrip():
    s = SoftInpainting(mask_blend_power=2.5, composite_difference_contrast=3)
    text = build_parameters(
        prompt="a bird", negative_prompt="blurry", steps=20, sampler="Euler a",
        scheduler="Karras", cfg=7, seed=1, width=512, height=512,
        model_name="m", model_hash="abc", soft_inpainting=s,
    )
    parsed = parse_parameters(text)
    assert parsed["Soft inpainting enabled"] == "True"
    assert parsed["Soft inpainting schedule bias"] == 2.5
    assert parsed["Soft inpainting preservation strength"] == 0.5
    assert parsed["Soft inpainting transition contrast boost"] == 4.0
    assert parsed["Soft inpainting mask influence"] == 0.0
    assert parsed["Soft inpainting difference threshold"] == 0.5
    assert parsed["Soft inpainting difference contrast"] == 3.0
    # other fields survive
    assert parsed["prompt"] == "a bird"
    assert parsed["steps"] == 20
    # without the flag: no soft keys at all
    plain = parse_parameters(build_parameters(
        prompt="a bird", negative_prompt="", steps=20, sampler="Euler a",
        scheduler="Karras", cfg=7, seed=1, width=512, height=512,
        model_name="m", model_hash="abc",
    ))
    assert "Soft inpainting enabled" not in plain
