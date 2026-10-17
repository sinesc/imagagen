"""Unit tests for self attention guidance (port of Forge's builtin
extension ``sd_forge_sag``).

The port must behave like the original: the pydantic model carries the
Forge slider ranges/defaults, the per-key attention mass matches the
Forge ``attn.mean(1).sum(1)`` reference (the full-matrix computation kept
verbatim here as the oracle), the blur map degrades exactly the masked
region, and the to_q/to_k hooks capture only while ``record`` is set.
"""

import types

import pytest
import torch

from imagagen.sag import (
    SagRecorder,
    SelfAttentionGuidance,
    compute_sag_mask,
    compute_sag_mass,
    create_blur_map,
    gaussian_blur_2d,
    install_sag_hooks,
)


def forge_mass(q, k, heads):
    """Forge's mask numerator, verbatim in full-matrix form (the oracle).

    ``attn`` is the (heads, n, n) softmax matrix of the mid-block
    self-attention; ``create_blur_map`` reshapes it to
    ``(b, heads, n, n)`` and computes
    ``attn.mean(1).sum(1)`` = heads-averaged, query-summed mass per key.
    """
    n, dim = q.shape[1], q.shape[2]
    dh = dim // heads
    scale = dh ** -0.5
    qf = q[0].reshape(n, heads, dh).transpose(0, 1).float()
    kf = k[0].reshape(n, heads, dh).transpose(0, 1).float()
    sim = torch.einsum("b i d, b j d -> b i j", qf, kf) * scale
    sim = sim.softmax(dim=-1)
    return sim.mean(0).sum(0)


# ------------------------------------------------------------- pydantic --


def test_sag_defaults_match_forge_sliders():
    s = SelfAttentionGuidance()
    assert s.enabled is True
    assert s.scale == 0.5
    assert s.blur_sigma == 2.0
    assert s.threshold == 1.0


def test_sag_enabled_flag_roundtrips():
    # the client always sends the object (like Forge always publishes the
    # parameters), toggling only the flag; the backend treats a disabled
    # object as absent
    off = SelfAttentionGuidance(enabled=False, scale=2.0)
    assert off.enabled is False
    assert off.scale == 2.0
    with pytest.raises(ValueError):
        SelfAttentionGuidance(enabled="maybe")


def test_sag_ranges_match_forge_sliders():
    for kwargs, field in (
        ({"scale": -2.0}, "scale"), ({"scale": 5.0}, "scale"),
        ({"blur_sigma": 0.0}, "blur_sigma"), ({"blur_sigma": 10.0}, "blur_sigma"),
        ({"threshold": 0.0}, "threshold"), ({"threshold": 4.0}, "threshold"),
    ):
        s = SelfAttentionGuidance(**kwargs)
        assert getattr(s, field) == kwargs[field]
    with pytest.raises(ValueError):
        SelfAttentionGuidance(scale=-2.01)
    with pytest.raises(ValueError):
        SelfAttentionGuidance(scale=5.01)
    with pytest.raises(ValueError):
        SelfAttentionGuidance(blur_sigma=-0.01)
    with pytest.raises(ValueError):
        SelfAttentionGuidance(blur_sigma=10.01)
    with pytest.raises(ValueError):
        SelfAttentionGuidance(threshold=4.01)


# ------------------------------------------------------------------ mass --


def test_sag_mass_matches_forge_reference_single_chunk():
    torch.manual_seed(1)
    heads, n, dim = 4, 256, 64
    q, k = torch.randn(1, n, dim), torch.randn(1, n, dim)
    ref = forge_mass(q, k, heads)
    got = compute_sag_mass(q, k, heads)
    assert got.shape == (n,)
    assert got.dtype == torch.float32
    # fp32 summation-order difference only (values are far from any
    # threshold except by luck)
    assert (ref - got).abs().max() < 1e-5


def test_sag_mass_chunked_accumulation(monkeypatch):
    # the default chunk covers n=4096 in one block; force a tiny chunk so
    # each head accumulates over 1024 (4, n) blocks and the result must
    # still match the full-matrix reference (fp32 accumulation error only)
    import imagagen.sag as sag

    torch.manual_seed(2)
    heads, n, dim = 2, 4096, 32
    q, k = torch.randn(1, n, dim), torch.randn(1, n, dim)
    default = compute_sag_mass(q, k, heads)
    monkeypatch.setattr(sag, "_SAG_SIM_CHUNK_ELEMENTS", 4 * n)  # chunk = 4
    chunked = compute_sag_mass(q, k, heads)
    ref = forge_mass(q, k, heads)
    assert (ref - default).abs().max() < 1e-5
    assert (ref - chunked).abs().max() < 1e-3
    # uniform-random attention: per-key mass is ~1 on average
    assert abs(float(default.mean()) - 1.0) < 0.05


def test_sag_mask_threshold_semantics():
    # concentrated attention: the hot keys carry the mass, the cold ones
    # almost none -> the mask separates them unambiguously
    heads, n, dim = 2, 16, 8
    k = torch.zeros(1, n, dim)
    k[0, :2, :] = 1.0  # 2 "hot" keys
    q = torch.ones(1, n, dim)
    q[0, 0, 0] += 50.0  # one strongly focused query
    mask = compute_sag_mask(q, k, heads, 1.0)
    assert mask.dtype == torch.bool
    assert mask[:2].all()
    assert not mask[2:].any()
    # threshold 0 blurs everything; at the slider max (4) the cold keys
    # are still unmasked (their mass stays below 1)
    assert compute_sag_mask(q, k, heads, 0.0).all()
    assert not compute_sag_mask(q, k, heads, 4.0)[2:].any()


def test_sag_mask_accepts_fp16_inputs():
    torch.manual_seed(3)
    q, k = torch.randn(1, 64, 32).half(), torch.randn(1, 64, 32).half()
    mask = compute_sag_mask(q, k, 4, 1.0)
    assert mask.dtype == torch.bool
    assert mask.shape == (64,)


# --------------------------------------------------------------- blur map --


def test_blur_sigma_zero_is_identity():
    x0 = torch.randn(1, 4, 8, 8)
    mask = torch.ones(64, dtype=torch.bool)
    assert torch.equal(create_blur_map(x0, mask, 0.0), x0)
    # Forge's documented "0 leaves the image unchanged" (its own kernel
    # divides by zero at sigma 0 and turns NaN)


def test_blur_map_only_touches_masked_region():
    # x0 at full latent res (8x8), mask at mid-block res (4x4 = 16 keys)
    x0 = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    none = torch.zeros(16, dtype=torch.bool)
    full = torch.ones(16, dtype=torch.bool)
    # nothing masked -> x0 unchanged; everything masked -> the plain
    # Gaussian blur (a constant mask upsamples to itself)
    assert torch.equal(create_blur_map(x0, none, 2.0), x0)
    assert torch.allclose(create_blur_map(x0, full, 2.0),
                          gaussian_blur_2d(x0, 9, 2.0))
    # top-left 2x2 of the 4x4 mask -> the top-left 4x4 interior of the
    # 8x8 output is blurred, the bottom-right interior untouched
    mask = torch.zeros(16, dtype=torch.bool)
    mask.view(4, 4)[:2, :2] = True
    out = create_blur_map(x0, mask, 2.0)
    blur = gaussian_blur_2d(x0, 9, 2.0)
    assert torch.allclose(out[0, :, 1:3, 1:3], blur[0, :, 1:3, 1:3])
    assert torch.allclose(out[0, :, 5:7, 5:7], x0[0, :, 5:7, 5:7])


def test_blur_map_accepts_full_resolution_mask():
    # n == lh * lw: the ratio trick resolves to the 1:1 grid (the
    # interpolate at equal size is the identity)
    x0 = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    mask = torch.zeros(64, dtype=torch.bool)
    mask[:10] = True  # row 0 + cols 0..1 of row 1
    out = create_blur_map(x0, mask, 2.0)
    assert torch.allclose(out[0, :, 5:, 5:], x0[0, :, 5:, 5:])
    blur = gaussian_blur_2d(x0, 9, 2.0)
    assert torch.allclose(out[0, :, 0, :], blur[0, :, 0, :])  # masked row
    assert torch.allclose(out[0, :, 1, :2], blur[0, :, 1, :2])
    assert torch.allclose(out[0, :, 1, 2:], x0[0, :, 1, 2:])


def test_gaussian_blur_preserves_constants():
    # the 2D kernel is normalised: a constant image blurs to itself
    img = torch.full((1, 4, 16, 16), 3.25)
    out = gaussian_blur_2d(img, 9, 2.0)
    assert torch.allclose(out, img, atol=1e-6)


# ------------------------------------------------------------------- hooks --


def test_install_sag_hooks_captures_only_while_recording():
    class Attn(torch.nn.Module):
        def __init__(self, heads=4, dim=32):
            super().__init__()
            self.heads = heads
            self.to_q = torch.nn.Linear(dim, dim)
            self.to_k = torch.nn.Linear(dim, dim)

        def forward(self, x):
            return self.to_q(x), self.to_k(x)

    attn = Attn()
    # diffusers structure: the mid attention entry is a Transformer2DModel
    # wrapper, the Attention is transformer_blocks[0].attn1
    wrapper = types.SimpleNamespace(transformer_blocks=[
        types.SimpleNamespace(attn1=attn),
    ])
    fake = types.SimpleNamespace(
        mid_block=types.SimpleNamespace(attentions=[wrapper]),
    )
    rec = SagRecorder()
    handles = install_sag_hooks(fake, rec)
    assert rec.heads == attn.heads

    x = torch.randn(2, 7, 32)
    rec.record = False
    attn(x)
    assert rec.q is None and rec.k is None

    rec.record = True
    attn(x)
    assert torch.equal(rec.q, attn.to_q(x).detach())
    assert torch.equal(rec.k, attn.to_k(x).detach())

    # record off again: a later forward must not overwrite the capture
    rec.record = False
    attn(torch.randn(2, 5, 32))
    assert torch.equal(rec.q, attn.to_q(x).detach())

    for h in handles:
        h.remove()
    rec.record = True
    attn(torch.randn(2, 5, 32))
    assert torch.equal(rec.q, attn.to_q(x).detach())  # hooks removed
