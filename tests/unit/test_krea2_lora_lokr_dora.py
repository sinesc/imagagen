"""LoKr (Kronecker) and DoRA (weight-decompose) LoRA support for Krea 2.

Covers the two on-disk formats that shipped as "unsupported" and are now
parsed and applied as runtime deltas:

- **LoKr** (ai-toolkit, ``BenchesKII1.0.safetensors``): full-matrix
  ``lokr_w1`` + ``lokr_w2``; the weight delta is ``kron(w1, w2)``
  (ComfyUI's LoKr ``calculate_weight``: scale 1.0 when w2 is stored
  full). Applied as the nested grouped GEMM of ComfyUI's LoKr bypass
  (``set_lora_kron``); the test compares against the materialized
  ``x @ kron(w1, w2).T`` on the same weights. For the ConvRot int8
  blocks the w1 rotation must reproduce the PLAIN-space delta.
- **DoRA** (``HighcutKII1.0`` / ``StreamlineKII1.0``): a LoRA pair plus
  an INPUT-axis ``dora_scale``; the effective delta is
  ``c * (W + d) - W`` with ``c = dora_scale / ||W + d||_col``
  (ComfyUI's ``weight_decompose``, input-axis branch). The backend
  folds ``c`` into the delta's A columns; the test replicates the
  reference math in fp32 and compares. ConvRot needs no de-rotation
  (the group Hadamard rotates the norm axis and is orthogonal).

Rejections (named diagnosis, per the compat plan): factorized/Tucker
LoKr, LoHa, LoKr+DoRA on one module, output-axis DoRA, GGUF files.
"""

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from imagagen.backends.krea2 import Krea2Backend
from imagagen.backends.krea2.lora import parse_krea2_lora
from imagagen.backends.krea2.transformer import (
    DeltaLinear,
    Int8Linear,
    _rotate,
)
from imagagen.gguf_ops import quant_dtype_scope

BENCHES = "loras/krea2/BenchesKII1.0.safetensors"
HIGHCUT = "loras/krea2/HighcutKII1.0.safetensors"


def _int8_module(in_f: int, out_f: int, group: int | None, seed: int):
    """A loaded Int8Linear with a random int8 weight; when ``group`` is
    set the stored weight is Hadamard-rotated (the checkpoint layout).
    Returns (module, plain fp32 weight = dequantized @ H^T)."""
    g = torch.Generator().manual_seed(seed)
    plain = torch.randn(out_f, in_f, generator=g) * 0.05
    scale = plain.abs().amax(dim=1, keepdim=True) / 127.0
    w8 = (plain / scale).round().clamp(-127, 127).to(torch.int8)
    stored = w8 if group is None else _rotate(w8.float(), group).to(torch.int8)
    with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
        m = Int8Linear(in_f, out_f, bias=False, convrot_group=group)
    m.load_state_dict({
        "weight": nn.Parameter(stored, requires_grad=False),
        "weight_scale": nn.Parameter(scale, requires_grad=False),
    })
    # the PLAIN weight implied by the stored bytes: de-rotating the stored
    # int8 (H^2 = I) is exactly what the backend does for its DoRA norms.
    # NOT w8 * scale: _rotate(w8) is not integral, so the int8 cast below
    # truncates, and w8 would leave the fixture inconsistent with `stored`
    # (that inconsistency, not bf16 math, made the DoRA checks flaky).
    plain_q = (_rotate(stored.float(), group) if group else stored.float()) * scale
    return m, plain_q


class TestKronDelta:
    def test_int8_convrot_kron_matches_plain_kron(self):
        # K2 main-block shape: in = in_l*in_k with in_l = 8; the ConvRot
        # group (256) tiles the in_k axis (K2: 768 / 864), so in_k must
        # be a multiple of 256 here and the rotation folds into w2
        m, _plain = _int8_module(8 * 512, 128, group=256, seed=1)
        w1 = torch.randn(8, 8) * 0.2
        w2 = torch.randn(16, 512) * 0.2
        x = torch.randn(4, 8 * 512, dtype=torch.bfloat16)
        base = m(x).float()  # no LoRA installed yet
        m.set_lora_kron([(w1.to(torch.bfloat16), w2.to(torch.bfloat16))])
        out = m(x).float()
        # the EMITTED delta must equal the plain-space Kronecker product
        # on the rotated input (the int8 base's own quantization floor
        # is not part of this comparison)
        delta = out - base
        expected = x.float() @ torch.kron(w1, w2).float().T
        rel = (delta - expected).norm() / expected.norm()
        assert rel < 0.02
        m.clear_lora_deltas()
        assert m.lora_kron == [] and m.lora_deltas == []
        torch.testing.assert_close(m(x).float(), base, rtol=0, atol=0)

    def test_delta_linear_kron_matches_plain_kron(self):
        m = DeltaLinear(128, 64, bias=False)  # fp32 plain-space module
        with torch.no_grad():
            m.weight.copy_(torch.randn(64, 128) * 0.05)
        w1 = torch.randn(8, 8) * 0.2
        w2 = torch.randn(8, 16) * 0.2
        x = torch.randn(3, 128)
        base = m(x)
        m.set_lora_kron([(w1, w2)])
        delta = m(x) - base
        expected = x @ torch.kron(w1, w2).T
        rel = (delta - expected).norm() / expected.norm()
        assert rel < 0.01

    def test_kron_coexists_with_plain_deltas(self):
        m, _plain = _int8_module(8 * 512, 128, group=256, seed=2)
        w1 = torch.randn(8, 8) * 0.2
        w2 = torch.randn(16, 512) * 0.2
        a = torch.randn(4, 8 * 512) * 0.05
        b = torch.randn(128, 4) * 0.05
        x = torch.randn(2, 8 * 512, dtype=torch.bfloat16)
        base = m(x).float()  # no LoRA installed yet
        m.set_lora_deltas([(a, b, 0, 128)])
        m.set_lora_kron([(w1.to(torch.bfloat16), w2.to(torch.bfloat16))])
        # per-kind setters: both survive
        out = m(x).float()
        assert m.lora_deltas and m.lora_kron
        rot = _rotate(x.float(), 256)
        delta = (rot @ _rotate(a, 256).T) @ b.T
        kron = x.float() @ torch.kron(w1, w2).float().T
        rel = (out - base - delta - kron).norm() / (delta.norm() + kron.norm())
        assert rel < 0.02


class TestDoraFold:
    """``Krea2Backend._fold_dora`` is self-free: call it unbound."""

    def _check(self, m, plain, a, b, dora):
        a = a.float()
        b = b.float()
        d = b @ a  # (out, in) -- the LoRA delta in weight space
        n_in = plain.shape[1]
        norm = (plain + d).norm(dim=0)
        c = dora / (norm + torch.finfo(torch.float32).eps)
        # the backend rounds c to bf16 before applying it (ComfyUI's cast)
        expected_delta = c.to(torch.bfloat16).to(torch.float32) * d
        folded, _kron = Krea2Backend._fold_dora(
            None, m, [(a, b, 0, plain.shape[0])], [], dora)
        (fa, fb, r0, r1), = folded
        # the fold is the column scaling with c rounded to bf16 first
        # (the ComfyUI cast); tiny A entries carry a larger relative
        # rounding
        torch.testing.assert_close(fa.to(torch.float32),
                                   a * c.to(torch.bfloat16).to(torch.float32),
                                   rtol=0.1, atol=1e-4)
        x = torch.randn(3, n_in, dtype=torch.bfloat16)
        base = m(x).float()  # no LoRA installed yet
        m.set_lora_deltas([(fa, fb, 0, plain.shape[0])])
        # the EMITTED delta must be the DoRA-scaled one (comparing the
        # full output would fold in the int8 base's quantization floor)
        rel = ((m(x).float() - base) - x.float() @ expected_delta.T).norm() \
            / x.float().matmul(expected_delta.T).norm()
        assert rel < 0.05

    def test_dora_fold_plain_int8(self):
        m, plain = _int8_module(64, 32, group=None, seed=3)
        a = torch.randn(4, 64) * 0.05
        b = torch.randn(32, 4) * 0.05
        dora = torch.rand(64) * 0.4 + 0.8
        self._check(m, plain, a, b, dora)

    def test_dora_fold_convrot_is_rotation_invariant(self):
        m, plain = _int8_module(256, 128, group=256, seed=4)
        a = torch.randn(4, 256) * 0.05
        b = torch.randn(128, 4) * 0.05
        dora = torch.rand(256) * 0.4 + 0.8
        self._check(m, plain, a, b, dora)

    def test_fold_rejects_kron_and_empty_deltas(self):
        m, plain = _int8_module(64, 32, group=None, seed=5)
        w1 = torch.randn(4, 8)
        w2 = torch.randn(8, 8)
        with pytest.raises(ValueError, match="LoKr \\+ DoRA"):
            Krea2Backend._fold_dora(None, m, [], [(w1, w2)], torch.rand(64))
        with pytest.raises(ValueError, match="at least one LoRA delta"):
            Krea2Backend._fold_dora(None, m, [], [], torch.rand(64))


class TestParser:
    def _save(self, tmp_path, tensors, name="lora.safetensors"):
        p = tmp_path / name
        save_file({k: t.to(torch.bfloat16) for k, t in tensors.items()}, str(p))
        return p

    def test_real_lokr_asset(self):
        specs = parse_krea2_lora(BENCHES)
        assert len(specs) == 256
        assert all(s.kron is not None and s.dora_scale is None for s in specs)
        by = {s.module: s for s in specs}
        assert by["blocks.0.attn.wq"].kron[0].shape == (8, 8)
        assert by["blocks.0.attn.wq"].kron[1].shape == (768, 768)
        assert by["blocks.0.attn.wk"].kron[1].shape == (192, 768)
        assert by["blocks.0.mlp.gate"].kron[1].shape == (2048, 768)
        assert by["blocks.0.mlp.down"].kron[1].shape == (768, 2048)
        assert by["txtfusion.layerwise_blocks.0.mlp.down"].kron[1].shape == (320, 864)
        # ComfyUI: scale 1.0 for full-matrix LoKr (the sentinel alpha is
        # ignored because lokr_w2 is stored full)
        assert all(s.scale == 1.0 for s in specs)

    def test_real_dora_assets(self):
        specs = parse_krea2_lora(HIGHCUT)
        assert len(specs) == 256
        assert all(s.dora_scale is not None and s.kron is None for s in specs)
        by = {s.module: s for s in specs}
        # dora_scale is the INPUT axis: wk is 1536x6144 -> 6144 values
        assert by["blocks.0.attn.wk"].dora_scale.numel() == 6144
        assert by["blocks.0.attn.wk"].B.shape == (1536, 32)
        assert by["blocks.0.mlp.down"].dora_scale.numel() == 16384
        assert by["txtfusion.layerwise_blocks.0.mlp.down"].dora_scale.numel() == 6912
        # alpha 16 at rank 32 -> scale 0.5
        assert all(abs(s.scale - 0.5) < 1e-6 for s in specs)

    def test_lokr_factorized_rejected(self, tmp_path):
        p = self._save(tmp_path, {
            "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(8.0),
            "diffusion_model.blocks.0.attn.wq.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.wq.lokr_w2_a": torch.randn(768, 8),
            "diffusion_model.blocks.0.attn.wq.lokr_w2_b": torch.randn(8, 768),
        })
        with pytest.raises(ValueError, match="factorized/Tucker LoKr"):
            parse_krea2_lora(p)

    def test_loha_rejected(self, tmp_path):
        p = self._save(tmp_path, {
            "diffusion_model.blocks.0.attn.wq.loha_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.wq.loha_w2": torch.randn(8, 8),
        })
        with pytest.raises(ValueError, match="LoHa"):
            parse_krea2_lora(p)

    def test_lokr_plus_dora_rejected(self, tmp_path):
        p = self._save(tmp_path, {
            "diffusion_model.blocks.0.attn.wq.lokr_w1": torch.randn(8, 8),
            "diffusion_model.blocks.0.attn.wq.lokr_w2": torch.randn(768, 768),
            "diffusion_model.blocks.0.attn.wq.dora_scale": torch.ones(1, 6144),
        })
        with pytest.raises(ValueError, match="LoKr \\+ DoRA"):
            parse_krea2_lora(p)

    def test_output_axis_dora_rejected(self, tmp_path):
        # wk is 1536x6144: a 1536-long scale is the output axis -> reject
        p = self._save(tmp_path, {
            "diffusion_model.blocks.0.attn.wk.lora_down.weight": torch.randn(32, 6144),
            "diffusion_model.blocks.0.attn.wk.lora_up.weight": torch.randn(1536, 32),
            "diffusion_model.blocks.0.attn.wk.dora_scale": torch.ones(1536),
        })
        with pytest.raises(ValueError, match="input axis"):
            parse_krea2_lora(p)

    def test_gguf_rejected(self, tmp_path):
        p = tmp_path / "quant.safetensors"
        p.write_bytes(b"GGUF" + b"\x00" * 64)
        with pytest.raises(ValueError, match="GGUF"):
            parse_krea2_lora(p)
