"""Quantized Linear formats + checkpoint loaders (CPU).

- NF4 round-trip: ``bnb.quantize_4bit`` (blocksize 64, nested on/off) ->
  ``QuantState.as_dict(packed=True)`` -> reassemble with
  ``QuantState.from_dict`` / :class:`Nf4Param` exactly like the file
  loader -> :class:`Nf4Linear` dequantize is bit-exact vs
  ``dequantize_4bit`` and within NF4 tolerance of the original;
- :class:`Nf4Param.to()` moves the packed bytes AND every aux tensor
  together (device and dtype);
- :class:`Fp8Linear`: fp8 weight/bias, bf16 input -> equals ``F.linear``
  of the cast weight; LoRA deltas add correctly;
- :func:`sniff_flux_checkpoint` / :func:`load_flux_state_dict` on small
  synthetic safetensors (both layouts, incl. the
  ``model.diffusion_model.`` prefix + JSON quant-state blob, and the
  plain-bf16 file that must raise).
"""

import json
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from bitsandbytes import functional as bfn
from bitsandbytes.nn.modules import QuantState
from safetensors.torch import save_file as save_safetensors

from imagagen.gguf_ops import (
    Fp8Linear,
    GgufLinear,
    GgufParam,
    Nf4Linear,
    Nf4Param,
    QuantLinear,
    dequantize_tensor,
    load_flux_state_dict,
    quant_linear_scope,
    sniff_flux_checkpoint,
)


def _rand_bf16(out_f: int, in_f: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(out_f, in_f, generator=g).to(torch.bfloat16)


def _quantize(W, nested: bool):
    """bnb-quantize a bf16 weight the way the shipped files were made."""
    packed, qs = bfn.quantize_4bit(
        W, blocksize=64, compress_statistics=nested, quant_type="nf4")
    return packed.contiguous().view(-1, 1), qs


def _as_file_pieces(packed, qs):
    """Split a fresh bnb QuantState into the pieces a file holds."""
    d = qs.as_dict(packed=True)
    meta = json.loads(d["quant_state.bitsandbytes__nf4"].numpy().tobytes().decode())
    return d, meta


def _nf4param(packed, d, meta):
    """Build the :class:`Nf4Param` the file loader builds (aux from pieces)."""
    return Nf4Param(
        packed,
        d["absmax"],
        blocksize=meta["blocksize"],
        quant_type=meta["quant_type"],
        out_dtype=getattr(torch, meta["dtype"]),
        real_shape=meta["shape"],
        nested_absmax=d.get("nested_absmax"),
        nested_quant_map=d.get("nested_quant_map"),
        nested_blocksize=meta.get("nested_blocksize"),
        nested_dtype=getattr(torch, meta.get("nested_dtype", "float32")),
        nested_offset=meta.get("nested_offset"),
    )


@pytest.mark.parametrize("nested", [False, True])
def test_nf4_roundtrip(nested):
    W = _rand_bf16(64, 128, seed=0)
    packed, qs = _quantize(W, nested)
    d, meta = _as_file_pieces(packed, qs)

    # the state reassembled from the file pieces (like the loader does)
    qs2 = QuantState.from_dict(d, torch.device("cpu"))
    ref = bfn.dequantize_4bit(packed, quant_state=qs2)

    # and the Nf4Param/Nf4Linear built from the same pieces must be
    # bit-exact against dequantize_4bit on a real forward
    lin = Nf4Linear(128, 64)
    lin.weight = _nf4param(packed, d, meta)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 5, 128, generator=g).to(torch.bfloat16)
    out = lin(x)
    assert torch.equal(out, F.linear(x, ref))
    assert ref.shape == W.shape and ref.dtype == torch.bfloat16
    # within NF4 tolerance of the original weight
    assert (ref.float() - W.float()).abs().max() < 0.5


def test_nf4param_to_moves_data_and_aux():
    W = _rand_bf16(128, 64, seed=2)
    packed, qs = _quantize(W, nested=True)  # nested: the full aux set
    d, meta = _as_file_pieces(packed, qs)
    param = _nf4param(packed, d, meta)
    assert param.nested_absmax is not None and param.nested_offset is not None

    moved = param.to("cpu")  # CPU test env: a no-op device move
    assert type(moved) is Nf4Param
    assert moved.data.dtype == torch.uint8
    assert moved.absmax.dtype == torch.uint8  # nested: uint8 indices
    assert moved.nested_absmax.dtype == torch.float32
    for name, t in [("absmax", moved.absmax),
                    ("nested_absmax", moved.nested_absmax),
                    ("nested_quant_map", moved.nested_quant_map)]:
        assert t is not None and t.device == moved.data.device, name
    # bookkeeping survives the move
    assert moved.shape == (128, 64)
    assert (moved.blocksize, moved.quant_type, moved.out_dtype) == (
        64, "nf4", torch.bfloat16)
    assert moved.nested_blocksize == 256
    assert moved.nested_offset == param.nested_offset

    # and the dequantize is unaffected by the moved aux (bit-exact)
    ref = bfn.dequantize_4bit(packed, quant_state=QuantState.from_dict(
        d, torch.device("cpu")))
    lin = Nf4Linear(64, 128)
    lin.weight = moved
    x = torch.randn(1, 3, 64, generator=torch.Generator().manual_seed(3)).to(
        torch.bfloat16)
    assert torch.equal(lin(x), F.linear(x, ref))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_nf4param_to_moves_aux_to_cuda():
    # a near-full GPU OOMs even the context init: skip when that happens
    # (the move itself is a few KB, but the context is ~hundreds of MB)
    dev = None
    for i in range(torch.cuda.device_count()):
        try:
            probe = torch.zeros(1, device=f"cuda:{i}")
            del probe
        except torch.AcceleratorError:
            continue
        dev = f"cuda:{i}"
        break
    if dev is None:
        pytest.skip("no CUDA device with room for a context")
    W = _rand_bf16(128, 64, seed=4)
    packed, qs = _quantize(W, nested=True)
    d, meta = _as_file_pieces(packed, qs)
    param = _nf4param(packed, d, meta)
    moved = param.to(dev)
    for t in (moved.data, moved.absmax, moved.nested_absmax,
              moved.nested_quant_map):
        assert t.device.type == "cuda"
    assert param.data.device.type == "cpu"  # the original is untouched
    # dequantize on CUDA is bit-exact vs CPU
    lin = Nf4Linear(64, 128)
    lin.weight = moved
    x = torch.randn(1, 3, 64, generator=torch.Generator().manual_seed(5)).to(
        torch.bfloat16)
    assert torch.equal(lin(x.to(dev)).cpu(), lin(x))


def test_fp8linear_forward_and_lora():
    with quant_linear_scope("fp8", torch.device("cpu"), torch.bfloat16):
        lin = Fp8Linear(16, 8)
    g = torch.Generator().manual_seed(6)
    w = torch.randn(8, 16, generator=g).to(torch.float8_e4m3fn)
    b = torch.randn(8, generator=g).to(torch.float8_e4m3fn)
    lin.weight = w
    lin.bias = b
    x = torch.randn(2, 4, 16, generator=g).to(torch.bfloat16)
    out = lin(x)
    expected = F.linear(x, w.to(torch.bfloat16), b.to(torch.bfloat16))
    assert torch.equal(out, expected)

    # LoRA deltas add on top (cast to the computation dtype, bf16)
    A = torch.randn(2, 16, generator=g)
    B = torch.randn(8, 2, generator=g)
    lin.set_lora_deltas([(A, B, 0, 8)])
    assert lin.lora_deltas[0][0].dtype == torch.bfloat16
    out2 = lin(x)
    expected2 = expected + (x @ A.t().to(x.dtype)) @ B.t().to(x.dtype)
    assert torch.allclose(out2, expected2, atol=1e-2)


def test_quant_linear_scope_selects_class():
    with quant_linear_scope("nf4", torch.device("cpu"), torch.bfloat16):
        assert torch.nn.Linear is Nf4Linear
        m = torch.nn.Linear(4, 8)
    assert isinstance(m, Nf4Linear) and isinstance(m, QuantLinear)
    with quant_linear_scope("gguf", torch.device("cpu"), torch.bfloat16):
        m2 = torch.nn.Linear(4, 8)
    assert isinstance(m2, GgufLinear)
    # the scope restores the original class
    assert torch.nn.Linear is not Nf4Linear
    assert torch.nn.Linear is not GgufLinear


def _write_fp8_checkpoint(path):
    g = torch.Generator().manual_seed(7)
    tensors = {
        "img_in.weight": torch.randn(8, 16, generator=g).to(
            torch.float8_e4m3fn),
        "img_in.bias": torch.randn(8, generator=g).to(torch.float8_e4m3fn),
        "final_layer.linear.weight": torch.randn(4, 8, generator=g).to(
            torch.float8_e4m3fn),
    }
    save_safetensors(tensors, str(path))
    return tensors


def test_sniff_and_load_fp8(tmp_path):
    path = tmp_path / "model.safetensors"
    tensors = _write_fp8_checkpoint(path)
    assert sniff_flux_checkpoint(path) == "fp8"
    sd = load_flux_state_dict(path, "fp8")
    assert set(sd) == set(tensors)
    for k, v in sd.items():
        assert v.dtype == torch.float8_e4m3fn
        assert torch.equal(v, tensors[k])


def _write_nf4_checkpoint(path, nested: bool, seed: int):
    """A synthetic bnb-NF4 all-in-one pipeline file (transformer section
    with the ``model.diffusion_model.`` prefix + decoy vae/text-encoder
    sections the loader must ignore)."""
    W = _rand_bf16(128, 64, seed=seed)
    packed, qs = _quantize(W, nested)
    d, meta = _as_file_pieces(packed, qs)
    root = "model.diffusion_model.lin"
    tensors = {
        f"{root}.weight": packed,
        f"{root}.weight.absmax": d["absmax"],
        f"{root}.weight.quant_map": d["quant_map"],
        f"{root}.weight.quant_state.bitsandbytes__nf4": (
            d["quant_state.bitsandbytes__nf4"]),
        f"{root}.bias": torch.zeros(128, dtype=torch.bfloat16),
        "vae.decoder.conv.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        "text_encoders.t5xxl.shared.weight": torch.zeros(
            4, 4, dtype=torch.float8_e4m3fn),
    }
    if nested:
        tensors[f"{root}.weight.nested_absmax"] = d["nested_absmax"]
        tensors[f"{root}.weight.nested_quant_map"] = d["nested_quant_map"]
    save_safetensors(tensors, str(path))
    return packed, qs


@pytest.mark.parametrize("nested", [False, True])
def test_sniff_and_load_nf4(tmp_path, nested):
    path = tmp_path / "model.safetensors"
    packed, qs = _write_nf4_checkpoint(path, nested, seed=8)
    assert sniff_flux_checkpoint(path) == "nf4"
    sd = load_flux_state_dict(path, "nf4")
    # only the transformer section, prefix stripped, aux folded in
    assert set(sd) == {"lin.weight", "lin.bias"}
    param = sd["lin.weight"]
    assert isinstance(param, Nf4Param)
    assert param.shape == (128, 64)
    assert param.data.dtype == torch.uint8
    assert (param.blocksize, param.quant_type, param.out_dtype) == (
        64, "nf4", torch.bfloat16)
    assert (param.nested_absmax is not None) == nested
    assert sd["lin.bias"].dtype == torch.bfloat16

    # the loaded Nf4Param dequantizes bit-exact like the original state
    ref = bfn.dequantize_4bit(packed, quant_state=qs)
    lin = Nf4Linear(64, 128)
    lin.weight = param
    g = torch.Generator().manual_seed(9)
    x = torch.randn(2, 3, 64, generator=g).to(torch.bfloat16)
    assert torch.equal(lin(x), F.linear(x, ref))


def test_plain_bf16_checkpoint_is_unsupported(tmp_path):
    path = tmp_path / "plain.safetensors"
    save_safetensors(
        {"img_in.weight": torch.randn(8, 16, dtype=torch.bfloat16)},
        str(path))
    with pytest.raises(ValueError):
        sniff_flux_checkpoint(path)


def test_gguf_sniff_by_magic(tmp_path):
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 16)
    assert sniff_flux_checkpoint(p) == "gguf"


# ------------------------------------------------------------------ GGUF ----


def _write_gguf_q4_0_checkpoint(path, W, B):
    """A tiny GGUF file: one Q4_0 linear weight + one F32 bias. The byte
    layout is produced by the gguf package's own Q4_0 quantizer (the same
    layout the shipped ``.gguf`` files use)."""
    blocks = gguf.quants.Q4_0.quantize_blocks(W.numpy())
    w = gguf.GGUFWriter(str(path), "test")
    w.add_tensor(
        "lin.weight", blocks,
        raw_dtype=gguf.constants.GGMLQuantizationType.Q4_0)
    w.add_tensor("lin.bias", B.numpy())
    w.open_output_file(Path(path))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_gguf_q4_0_dequantize(tmp_path):
    path = tmp_path / "model.gguf"
    W = _rand_bf16(64, 32, seed=4).float()
    B = torch.randn(64, generator=torch.Generator().manual_seed(5)).float()
    _write_gguf_q4_0_checkpoint(path, W, B)

    sd = load_flux_state_dict(path, "gguf")
    assert set(sd) == {"lin.weight", "lin.bias"}
    param = sd["lin.weight"]
    assert isinstance(param, GgufParam)
    assert param.shape == (64, 32)
    assert param.data.dtype == torch.uint8

    # the dequantized weight matches the file's reference layout exactly
    # (gguf package Q4_0 dequantizer) and is within Q4_0 tolerance of the
    # original
    reader = gguf.GGUFReader(str(path))
    tensor = next(t for t in reader.tensors if str(t.name) == "lin.weight")
    ref = gguf.quants.Q4_0.dequantize(np.asarray(tensor.data))
    got = dequantize_tensor(param).float()
    assert torch.equal(got, torch.tensor(ref, dtype=torch.float32))
    assert (got - W).abs().max() <= W.abs().max()

    # fp32 biases pass through untouched
    assert torch.equal(dequantize_tensor(sd["lin.bias"]).float(), B)


def test_gguf_q4_0_linear(tmp_path):
    path = tmp_path / "model.gguf"
    out_f, in_f = 64, 32
    W = _rand_bf16(out_f, in_f, seed=6).float()
    B = torch.randn(out_f, generator=torch.Generator().manual_seed(7)).float()
    _write_gguf_q4_0_checkpoint(path, W, B)

    sd = load_flux_state_dict(path, "gguf")
    with quant_linear_scope("gguf", torch.device("cpu"), torch.bfloat16):
        lin = GgufLinear(in_f, out_f)
    lin.load_state_dict(
        {"weight": sd["lin.weight"], "bias": sd["lin.bias"]}, strict=False)

    x = torch.randn(3, in_f, generator=torch.Generator().manual_seed(8)).to(
        torch.bfloat16)
    deq = dequantize_tensor(sd["lin.weight"]).to(torch.bfloat16)
    assert torch.equal(lin(x), F.linear(x, deq, B.to(torch.bfloat16)))
