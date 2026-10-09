"""Unit tests for the Krea 2 checkpoint loader (``_load_krea2_state_dict``).

Covers the two checkpoint layouts:
- ``int8_tensorwise`` (keys without prefix, int8 weight + (out, 1)
  weight_scale, ``*.comfy_quant`` markers skipped);
- ``int4_convrot`` (``model.diffusion_model.`` prefix, nibble-packed int4
  weight (out, in/2) + 1-D per-row weight_scale, declared through the
  ``_quantization_metadata`` safetensors metadata): the blocks weights
  stay PACKED for the ``Int4Linear`` modules, the txtfusion weights are
  dequantized and de-rotated to plain bf16.
"""

import torch
from safetensors.torch import save_file
from torch import nn

from imagagen.backends.krea2 import (
    _load_krea2_state_dict,
    _unpack_int4_row_major,
)
from imagagen.backends.krea2.transformer import (
    DeltaLinear,
    Int4Linear,
    Int8Linear,
    Krea2Transformer,
    _rotate,
)
from imagagen.gguf_ops import quant_dtype_scope  # noqa: F401

_GROUP = 256  # the checkpoints' convrot_groupsize


def _pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Inverse of the loader's codec (comfy_kitchen _pack_int4_row_major):
    (..., K) int4 values -> (..., K/2) int8, even column in the low
    nibble."""
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _fake_int4_checkpoint(tmp_path, in_f: int = 512, out_f: int = 64):
    """A minimal int4_convrot checkpoint: one blocks projection (Int8Linear
    target), one txtfusion projection (DeltaLinear target), a plain tensor
    and a stray comfy_quant marker."""
    gen = torch.Generator().manual_seed(7)
    wq = torch.randint(-8, 8, (out_f, in_f), generator=gen, dtype=torch.int8)
    scale = torch.rand(out_f, generator=gen) + 0.5
    wtxt = torch.randint(-8, 8, (out_f, in_f), generator=gen, dtype=torch.int8)
    stxt = torch.rand(out_f, generator=gen) + 0.5
    sd = {
        "model.diffusion_model.blocks.0.attn.wq.weight":
            _pack_int4_row_major(wq),
        "model.diffusion_model.blocks.0.attn.wq.weight_scale": scale,
        "model.diffusion_model.blocks.0.attn.wq.comfy_quant":
            torch.zeros(3, dtype=torch.uint8),
        "model.diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight":
            _pack_int4_row_major(wtxt),
        "model.diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.weight_scale":
            stxt,
        "model.diffusion_model.first.weight": torch.ones(out_f, 64),
    }
    meta = {"_quantization_metadata": (
        '{"format": "int4_convrot", "layers": {"model.diffusion_model.blocks.'
        '0.attn.wq": {"format": "convrot_w4a4", "convrot_groupsize": 256, '
        '"quant_group_size": 64}}}')}
    path = str(tmp_path / "fake_int4.safetensors")
    save_file(sd, path, metadata=meta)
    return sd, wq, scale, wtxt, stxt


def test_unpack_roundtrip_signed_nibbles():
    # every nibble value, including the -8..-1 sign-extension boundary
    values = torch.arange(-8, 8, dtype=torch.int8).repeat(16).unsqueeze(0)
    packed = _pack_int4_row_major(values)
    assert packed.shape == (1, values.shape[1] // 2)
    torch.testing.assert_close(_unpack_int4_row_major(packed).squeeze(0),
                               values.squeeze(0))


def test_gpu_shift_codec_matches_the_loader_codec():
    """The Int4Linear per-op unpack (arithmetic shifts staying int8) is
    the exact inverse of the packer on every nibble pattern."""
    gen = torch.Generator().manual_seed(11)
    packed = torch.randint(-128, 128, (8, 128), generator=gen,
                           dtype=torch.int8)
    hi = packed >> 4
    lo = (packed << 4) >> 4
    unpacked = torch.stack([lo, hi], dim=-1).reshape(8, -1)
    assert unpacked.dtype == torch.int8
    torch.testing.assert_close(unpacked, _unpack_int4_row_major(packed))


def test_int4_checkpoint_load(tmp_path):
    sd, wq, scale, wtxt, stxt = _fake_int4_checkpoint(tmp_path)
    out, layout = _load_krea2_state_dict(str(tmp_path / "fake_int4.safetensors"))
    assert layout == "int4"

    # prefix stripped, comfy_quant marker skipped, consumed scales dropped
    assert "blocks.0.attn.wq.weight" in out
    assert "txtfusion.layerwise_blocks.0.attn.wq.weight" in out
    assert "first.weight" in out
    assert not any("comfy_quant" in k for k in out)
    assert not any(k.startswith("model.") for k in out)
    assert "txtfusion.layerwise_blocks.0.attn.wq.weight_scale" not in out

    # blocks: PACKED int8 (out, in/2) + (out, 1) scale, i.e. the Int4Linear
    # layout (unpacked on the GPU per op)
    w, s = out["blocks.0.attn.wq.weight"], out["blocks.0.attn.wq.weight_scale"]
    assert w.dtype == torch.int8 and w.shape == (64, 256)
    assert s.shape == (64, 1)
    torch.testing.assert_close(w, _pack_int4_row_major(wq))
    torch.testing.assert_close(s, scale.reshape(-1, 1))

    # txtfusion: dequantized AND de-rotated plain bf16 (DeltaLinear target)
    w = out["txtfusion.layerwise_blocks.0.attn.wq.weight"]
    assert w.dtype == torch.bfloat16
    expected = _rotate(wtxt.to(torch.float32) * stxt.reshape(-1, 1), _GROUP)
    torch.testing.assert_close(w.to(torch.float32), expected,
                               rtol=1e-2, atol=1e-2)


def _cast_plain_bf16(model) -> None:
    """The backend's post-load cast: plain (non-quantized) params to
    bf16, the quantized ones keep their dtype."""
    qpref = {name + "." for name, mod in model.named_modules()
             if isinstance(mod, Int8Linear)}
    for name, p in list(model.named_parameters()):
        if not any(name.startswith(pref) for pref in qpref):
            p.data = p.data.to(torch.bfloat16)


def test_int4_checkpoint_loads_a_real_transformer(tmp_path):
    """End-to-end wiring: a synthetic int4_convrot checkpoint built for a
    small Krea2Transformer(quant_cls=Int4Linear) loads with NO missing/
    unexpected keys and its forward is BIT-IDENTICAL to the int8
    transformer built from the same logical weights."""
    kw = dict(features=512, tdim=64, txtdim=256, heads=4, kvheads=4,
              multiplier=4, layers=2, txtlayers=2, txtheads=2,
              txtkvheads=2)
    with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
        ref = Krea2Transformer(**kw)
    gen = torch.Generator().manual_seed(3)
    sd = {f"model.diffusion_model.{k}": v
          for k, v in ref.state_dict().items()}
    layers = {}
    for name, mod in ref.named_modules():
        # the quantized set: the block Int8Linears plus the txtfusion
        # DeltaLinears (the projector stays plain, like the checkpoint)
        if isinstance(mod, Int8Linear) or (
                isinstance(mod, DeltaLinear)
                and name.startswith("txtfusion.")
                and "projector" not in name):
            shape = (mod.out_features, mod.in_features)
            q = torch.randint(-7, 8, shape, generator=gen, dtype=torch.int8)
            scale = q.abs().amax(dim=1).to(torch.float32) / 7 + 0.25
            key = f"model.diffusion_model.{name}"
            sd[f"{key}.weight"] = _pack_int4_row_major(q)
            sd[f"{key}.weight_scale"] = scale
            layers[name] = {"format": "convrot_w4a4",
                            "convrot_groupsize": _GROUP}
    import json
    meta = {"_quantization_metadata": json.dumps(
        {"format": "int4_convrot", "layers": layers})}
    path = str(tmp_path / "small_int4.safetensors")
    save_file(sd, path, metadata=meta)

    # the int8 reference: the SAME logical weights in the int8 layout
    # (the unpacked bytes + the (out, 1) scales; the txtfusion weights
    # folded to plain bf16 exactly like the loader does)
    ref_sd = {}
    sdq = {k.removeprefix("model.diffusion_model."): v for k, v in sd.items()}
    for k, v in sdq.items():
        if k.endswith(".weight_scale"):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        scale_key = None if base is None else f"{base}.weight_scale"
        if base is not None and scale_key in sdq:
            packed_w, scale = v, sdq[scale_key]
            if base.startswith("txtfusion."):
                w = _unpack_int4_row_major(packed_w)
                w = _rotate(w.to(torch.float32) * scale.reshape(-1, 1), 256)
                ref_sd[k] = nn.Parameter(w.to(torch.bfloat16),
                                         requires_grad=False)
            else:
                ref_sd[k] = nn.Parameter(_unpack_int4_row_major(packed_w),
                                         requires_grad=False)
                ref_sd[scale_key] = nn.Parameter(scale.reshape(-1, 1),
                                                 requires_grad=False)
        else:
            if k.endswith(".weight_scale") and k.startswith("blocks."):
                v = v.reshape(-1, 1)  # the int8 layout: (out, 1) scales
            ref_sd[k] = nn.Parameter(v, requires_grad=False)
    result = ref.load_state_dict(ref_sd, strict=False)
    assert not result.missing_keys, result.missing_keys[:5]
    assert not result.unexpected_keys, result.unexpected_keys[:5]

    loaded, layout = _load_krea2_state_dict(path)
    assert layout == "int4"
    # the quantized block projections KEEP the packed int4 layout,
    # txtfusion arrives as plain bf16 (dequantized + de-rotated)
    assert loaded["blocks.0.attn.wq.weight"].dtype == torch.int8
    assert loaded["blocks.0.attn.wq.weight"].shape[1] == 512 // 2
    assert loaded["blocks.0.attn.wq.weight_scale"].shape == (512, 1)
    assert loaded[
        "txtfusion.layerwise_blocks.0.attn.wq.weight"].dtype == torch.bfloat16

    with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
        model = Krea2Transformer(quant_cls=Int4Linear, **kw)
    result = model.load_state_dict(loaded, strict=False)
    assert not result.missing_keys, result.missing_keys[:5]
    assert not result.unexpected_keys, result.unexpected_keys[:5]
    _cast_plain_bf16(model)
    _cast_plain_bf16(ref)

    # the int4 resident model == the int8 model on the same logical
    # weights: the GPU unpack reproduces the exact int8 bytes, so the
    # forwards must match bit for bit
    model.eval()
    ref.eval()
    x = torch.randn(1, 16, 4, 4, dtype=torch.bfloat16)
    sigma = torch.tensor([0.7])
    ctx = torch.randn(1, 6, kw["txtlayers"] * kw["txtdim"],
                      dtype=torch.bfloat16)
    torch.testing.assert_close(model(x, sigma, ctx), ref(x, sigma, ctx))


def test_int8_checkpoint_passthrough(tmp_path):
    """The int8_tensorwise layout (no quantization metadata, (out, 1)
    scales) loads unchanged."""
    w = torch.randint(-128, 127, (64, 512), dtype=torch.int8)
    s = torch.rand(64, 1) + 0.5
    sd = {"blocks.0.attn.wq.weight": w, "blocks.0.attn.wq.weight_scale": s,
          "blocks.0.attn.wq.comfy_quant": torch.zeros(3, dtype=torch.uint8)}
    path = str(tmp_path / "fake_int8.safetensors")
    save_file(sd, path)
    out, layout = _load_krea2_state_dict(path)
    assert layout == "int8"
    assert set(out) == {"blocks.0.attn.wq.weight",
                        "blocks.0.attn.wq.weight_scale"}
    torch.testing.assert_close(out["blocks.0.attn.wq.weight"], w)
    torch.testing.assert_close(out["blocks.0.attn.wq.weight_scale"], s)
