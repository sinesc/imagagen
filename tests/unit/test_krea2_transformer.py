"""Unit tests for the Krea 2 transformer pieces that do NOT require the
full 12B checkpoint (the end-to-end numeric check is dev/smoke_krea2.py).

- the Hadamard rotation (ConvRot, group 256): orthogonality and the
  rotate-twice-is-identity property;
- :class:`~imagagen.backends.krea2.transformer.Int8Linear`: the int8 load path
  (weight + per-row weight_scale), the dequantized GEMM on the
  rotated input, and the runtime LoRA deltas on the rotated input;
- :class:`~imagagen.backends.krea2.transformer.Int4Linear`: the packed
  nibble layout (int8 [out, in/2]) unpacks to the exact int8 values, so
  the forward and the LoRA deltas are bit-identical to Int8Linear;
- :class:`~imagagen.backends.krea2.transformer.DeltaLinear`: the plain-bf16
  delta contract (same shape/dtype rules as the int8 path).
"""

import pytest
import torch
from torch import nn

from imagagen.backends.krea2 import _unpack_int4_row_major
from imagagen.backends.krea2.transformer import (
    DeltaLinear,
    Int4Linear,
    Int8Linear,
    _build_hadamard,
    _rotate,
)
from imagagen.gguf_ops import quant_dtype_scope


class TestHadamard:
    def test_matches_the_reference_h4_recursion(self):
        # the checkpoint weights are rotated with the ComfyUI ConvRot
        # matrix: h4 (row order [111-1, 11-11, 1-111, -1111]) kron-
        # recursed. The Sylvester H2 base has a DIFFERENT row order
        # (H4 != H2 (x) H2) and de-rotates the txtfusion weights into
        # garbage (pure-noise renders) -- pin the exact matrix.
        import math

        h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                           [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
        h = h4
        while h.shape[0] < 64:
            h = torch.kron(h, h4)
        torch.testing.assert_close(
            _build_hadamard(64).double(), h / math.sqrt(64),
            rtol=1e-9, atol=1e-9)
        with pytest.raises(ValueError):
            _build_hadamard(8)  # not a ConvRot group (reference raises)

    def test_orthogonal(self):
        h = _build_hadamard(256)
        assert h.shape == (256, 256)
        torch.testing.assert_close(h @ h.T, torch.eye(256),
                                   rtol=1e-5, atol=1e-5)

    def test_rotate_twice_is_identity(self):
        x = torch.randn(8, 768)
        torch.testing.assert_close(_rotate(_rotate(x, 256), 256), x,
                                   rtol=1e-5, atol=1e-5)

    def test_rotate_is_block_local(self):
        # groups of 256: a change in one group never leaks into another
        x = torch.randn(2, 512)
        y = x.clone()
        y[0, 5] += 1.0
        r_x, r_y = _rotate(x, 256), _rotate(y, 256)
        assert not torch.allclose(r_x[0, :256], r_y[0, :256])
        assert torch.allclose(r_x[0, 256:], r_y[0, 256:])

    def test_cached_matrices_share_the_identity_property(self):
        for size in (256, 1024):
            h = _build_hadamard(size)
            torch.testing.assert_close(h @ h.T, torch.eye(size),
                                       rtol=1e-5, atol=1e-5)


def _make_int8(in_features: int = 256, out_features: int = 64):
    """A loaded Int8Linear with a random int8 weight (the checkpoint's
    layout: int8 weight + fp32 per-row scale, built under the scope)."""
    with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
        m = Int8Linear(in_features, out_features, bias=False)
    w = torch.randint(-128, 127, (out_features, in_features),
                      dtype=torch.int8)
    scale = torch.rand(out_features, 1) + 0.5
    m.load_state_dict({
        "weight": nn.Parameter(w, requires_grad=False),
        "weight_scale": nn.Parameter(scale, requires_grad=False),
    })
    return m, w, scale


def _rel_err(out, expected) -> float:
    # bf16 end-to-end vs the fp32 reference: judge by relative L2 error
    # (the bf16 noise floor is well under 1%)
    return float((out - expected).norm() / expected.norm())


class TestInt8Linear:
    def test_forward_is_the_dequantized_gemm_on_rotated_input(self):
        m, w, scale = _make_int8()
        x = torch.randn(3, 256, dtype=torch.bfloat16)
        out = m(x)
        assert out.shape == (3, 64)
        assert out.dtype == torch.bfloat16
        # expected: rotate x (fp32 for the reference), dequantize W, GEMM
        x_rot = _rotate(x.float(), 256)
        expected = x_rot @ (w.float() * scale.float()).T
        assert _rel_err(out.float(), expected) < 0.01

    def test_rotation_matters(self):
        # without the activation rotation the numbers differ (the stored
        # weight is rotated along the input axis)
        m, w, scale = _make_int8()
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        unrotated = x.float() @ (w.float() * scale.float()).T
        assert not torch.allclose(m(x).float(), unrotated, atol=1e-3)

    def test_lora_deltas_are_the_plain_space_deltas(self):
        # the LoRA was trained on the DE-ROTATED weight
        # (W_orig = W @ H); the runtime must reproduce
        # x @ (W_orig + B A)^T, i.e. the A matrix is rotated into the
        # ConvRot space at install (A' = A @ H, H^2 = I)
        # zero base weight: the output IS the delta (exact extraction)
        with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
            m = Int8Linear(256, 64, bias=False)
        m.load_state_dict({
            "weight": nn.Parameter(
                torch.zeros(64, 256, dtype=torch.int8), requires_grad=False),
            "weight_scale": nn.Parameter(torch.ones(64, 1), requires_grad=False),
        })
        gen = torch.Generator().manual_seed(3)  # fixed: the "not the
        # old bug" margin below is seed-sensitive
        a = torch.randn(4, 256, generator=gen)
        b = torch.randn(64, 4, generator=gen)
        m.set_lora_deltas([(a, b, 0, 64)])
        x = torch.randn(2, 256, dtype=torch.bfloat16, generator=gen)
        out = m(x).float()
        # the emitted delta is the PLAIN-space one: x @ (B A)^T
        expected = x.float() @ (b @ a).T
        assert _rel_err(out, expected) < 0.01
        # and NOT the old bug: the unrotated delta on the rotated input
        # (x @ (B A H)^T) -- an H-scrambled delta of the same magnitude
        # but a different direction (invisible at low LoRA weights)
        x_rot = _rotate(x.float(), 256)
        wrong = x_rot @ a.T @ b.T
        # two random delta paths of equal norm sit at rel_err ~ sqrt(2)
        # when they are INDEPENDENT (the correct case); the old bug
        # (scrambled delta direction) lands near 1.0 -- the margin must
        # stay below sqrt(2) to keep the seed sensitivity benign
        assert _rel_err(out, wrong) > 1.2
        m.clear_lora_deltas()
        assert m.lora_deltas == []

    def test_convrot_group_disabled_for_small_widths(self):
        with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
            m = Int8Linear(64, 32, bias=False)  # 64 < 256: no rotation
        w = torch.randint(-128, 127, (32, 64), dtype=torch.int8)
        scale = torch.rand(32, 1) + 0.5
        m.load_state_dict({
            "weight": nn.Parameter(w, requires_grad=False),
            "weight_scale": nn.Parameter(scale, requires_grad=False),
        })
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        expected = x.float() @ (w.float() * scale.float()).T
        assert _rel_err(m(x).float(), expected) < 0.01


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """comfy_kitchen _pack_int4_row_major: (..., K) int4 values ->
    (..., K/2) int8, the even column in the low nibble."""
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _make_int4(w: torch.Tensor, scale: torch.Tensor) -> Int4Linear:
    """A loaded Int4Linear for the stored int8 weight ``w`` (nibble-packed
    here; the checkpoints store the ConvRot-rotated weight) and the fp32
    per-row ``scale``."""
    with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
        m = Int4Linear(w.shape[1], w.shape[0], bias=False)
    m.load_state_dict({
        "weight": nn.Parameter(_pack_int4(w), requires_grad=False),
        "weight_scale": nn.Parameter(scale, requires_grad=False),
    })
    return m


class TestInt4Linear:
    def test_forward_is_bit_identical_to_int8(self):
        # same stored bytes (the rotated weight, int4-packed vs int8):
        # the GPU unpack reproduces the exact int8 values, so the bf16
        # GEMM inputs are identical -> bit-equal
        m8, w, scale = _make_int8()
        w4 = w.clamp(-8, 7)  # int4 needs the same stored values, in range
        m8.load_state_dict({
            "weight": nn.Parameter(w4, requires_grad=False),
            "weight_scale": nn.Parameter(scale, requires_grad=False)})
        m4 = _make_int4(w4, scale)
        torch.testing.assert_close(_unpack_int4_row_major(m4.weight), w4)
        assert m4.weight.shape == (w.shape[0], w.shape[1] // 2)
        gen = torch.Generator().manual_seed(5)
        x = torch.randn(5, 256, generator=gen, dtype=torch.bfloat16)
        assert torch.equal(m4(x), m8(x))

    def test_lora_deltas_are_inherited(self):
        # zero base weight: the output IS the plain-space delta, exactly
        # as on Int8Linear (the A rotation is inherited)
        m4 = _make_int4(torch.zeros(64, 256, dtype=torch.int8),
                        torch.ones(64, 1))
        a = torch.randn(4, 256)
        b = torch.randn(64, 4)
        m4.set_lora_deltas([(a, b, 0, 64)])
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        expected = x.float() @ (b @ a).T
        assert _rel_err(m4(x).float(), expected) < 0.01
        m4.clear_lora_deltas()
        assert m4.lora_deltas == []

    def test_apply_moves_the_scales(self):
        m4 = _make_int4(torch.zeros(4, 8, dtype=torch.int8), torch.ones(4, 1))
        # a plain tensor scale is not a registered parameter: _apply must
        # transport it (the inherited Int8Linear hook)
        m4 = m4.to(torch.float16)
        assert m4.weight_scale.dtype == torch.float16


class TestDeltaLinear:
    def test_forward_base_and_deltas(self):
        m = DeltaLinear(32, 16, bias=False)
        m.weight = nn.Parameter(torch.randn(16, 32))
        x = torch.randn(2, 32)
        base = x @ m.weight.T
        a = torch.randn(3, 32) * 0.1
        b = torch.randn(16, 3) * 0.1
        m.set_lora_deltas([(a, b, 0, 16)])
        torch.testing.assert_close(m(x), base + x @ a.T @ b.T,
                                   rtol=1e-5, atol=1e-5)
        # a partial-row delta only touches its slice
        b2 = torch.randn(4, 3) * 0.1
        m.set_lora_deltas([(a, b2, 0, 4)])
        out = m(x)
        torch.testing.assert_close(out[:, :4], base[:, :4] + x @ a.T @ b2.T,
                                   rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(out[:, 4:], base[:, 4:],
                                   rtol=1e-5, atol=1e-5)
        m.clear_lora_deltas()
        assert m.lora_deltas == []
        torch.testing.assert_close(m(x), base, rtol=1e-5, atol=1e-5)

    def test_deltas_are_registered_parameters(self):
        m = DeltaLinear(8, 4, bias=False)
        m.set_lora_deltas([(torch.randn(2, 8), torch.randn(4, 2), 0, 4)])
        names = [n for n, _ in m.named_parameters() if n.startswith("lora_")]
        assert len(names) == 2
        m.clear_lora_deltas()
        assert not any(n.startswith("lora_") for n, _ in m.named_parameters())
