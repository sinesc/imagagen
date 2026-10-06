"""Unit tests for the Krea 2 transformer pieces that do NOT require the
full 12B checkpoint (the end-to-end numeric check is dev/smoke_krea2.py).

- the Hadamard rotation (ConvRot, group 256): orthogonality and the
  rotate-twice-is-identity property;
- :class:`~imagagen.backends.krea2.transformer.Int8Linear`: the int8 load path
  (weight + per-row weight_scale), the dequantized GEMM on the
  rotated input, and the runtime LoRA deltas on the rotated input;
- :class:`~imagagen.backends.krea2.transformer.DeltaLinear`: the plain-bf16
  delta contract (same shape/dtype rules as the int8 path).
"""

import torch
from torch import nn

from imagagen.gguf_ops import quant_dtype_scope
from imagagen.backends.krea2.transformer import (
    DeltaLinear,
    Int8Linear,
    _build_hadamard,
    _rotate,
)


class TestHadamard:
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
        a = torch.randn(4, 256)
        b = torch.randn(64, 4)
        m.set_lora_deltas([(a, b, 0, 64)])
        x = torch.randn(2, 256, dtype=torch.bfloat16)
        out = m(x).float()
        # the emitted delta is the PLAIN-space one: x @ (B A)^T
        expected = x.float() @ (b @ a).T
        assert _rel_err(out, expected) < 0.01
        # and NOT the old bug: the unrotated delta on the rotated input
        # (x @ (B A H)^T) -- an H-scrambled delta of the same magnitude
        # but a different direction (invisible at low LoRA weights)
        x_rot = _rotate(x.float(), 256)
        wrong = x_rot @ a.T @ b.T
        assert _rel_err(out, wrong) > 1.0
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
