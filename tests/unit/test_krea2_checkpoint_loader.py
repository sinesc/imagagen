"""Unit tests for the Krea 2 checkpoint loader (``_load_krea2_state_dict``).

Covers the two checkpoint layouts:
- ``int8_tensorwise`` (keys without prefix, int8 weight + (out, 1)
  weight_scale, ``*.comfy_quant`` markers skipped);
- ``int4_convrot`` (``model.diffusion_model.`` prefix, nibble-packed int4
  weight (out, in/2) + 1-D per-row weight_scale, declared through the
  ``_quantization_metadata`` safetensors metadata): the blocks weights
  unpack to the int8 layout, the txtfusion weights are additionally
  dequantized and de-rotated to plain bf16.
"""

import torch
from safetensors.torch import save_file

from imagagen.gguf_ops import quant_dtype_scope  # noqa: F401
from imagagen.krea2_backend import (
    _load_krea2_state_dict,
    _unpack_int4_row_major,
)
from imagagen.krea2_transformer import (
    DeltaLinear,
    Int8Linear,
    Krea2Transformer,
    _rotate,
)

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


def test_int4_checkpoint_load(tmp_path):
    sd, wq, scale, wtxt, stxt = _fake_int4_checkpoint(tmp_path)
    out = _load_krea2_state_dict(str(tmp_path / "fake_int4.safetensors"))

    # prefix stripped, comfy_quant marker skipped, consumed scales dropped
    assert "blocks.0.attn.wq.weight" in out
    assert "txtfusion.layerwise_blocks.0.attn.wq.weight" in out
    assert "first.weight" in out
    assert not any("comfy_quant" in k for k in out)
    assert not any(k.startswith("model.") for k in out)
    assert "txtfusion.layerwise_blocks.0.attn.wq.weight_scale" not in out

    # blocks: unpacked int8 + (out, 1) scale, i.e. the int8 checkpoint layout
    w, s = out["blocks.0.attn.wq.weight"], out["blocks.0.attn.wq.weight_scale"]
    assert w.dtype == torch.int8 and s.shape == (64, 1)
    torch.testing.assert_close(w, wq)
    torch.testing.assert_close(s, scale.reshape(-1, 1))

    # txtfusion: dequantized AND de-rotated plain bf16 (DeltaLinear target)
    w = out["txtfusion.layerwise_blocks.0.attn.wq.weight"]
    assert w.dtype == torch.bfloat16
    expected = _rotate(wtxt.to(torch.float32) * stxt.reshape(-1, 1), _GROUP)
    torch.testing.assert_close(w.to(torch.float32), expected,
                               rtol=1e-2, atol=1e-2)


def test_int4_checkpoint_loads_a_real_transformer(tmp_path):
    """End-to-end wiring: a synthetic int4_convrot checkpoint built for a
    small Krea2Transformer loads with NO missing/unexpected keys and the
    model runs a forward pass."""
    kw = dict(features=512, tdim=64, txtdim=256, heads=4, kvheads=4,
              multiplier=4, layers=2, txtlayers=2, txtheads=2,
              txtkvheads=2)
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

    loaded = _load_krea2_state_dict(path)
    # the quantized block projections keep the int8 layout, txtfusion
    # arrives as plain bf16 (dequantized + de-rotated)
    assert loaded["blocks.0.attn.wq.weight"].dtype == torch.int8
    assert loaded["blocks.0.attn.wq.weight_scale"].shape == (512, 1)
    assert loaded[
        "txtfusion.layerwise_blocks.0.attn.wq.weight"].dtype == torch.bfloat16

    model = Krea2Transformer(**kw)
    result = model.load_state_dict(loaded, strict=False)
    assert not result.missing_keys, result.missing_keys[:5]
    assert not result.unexpected_keys, result.unexpected_keys[:5]


def test_int8_checkpoint_passthrough(tmp_path):
    """The int8_tensorwise layout (no quantization metadata, (out, 1)
    scales) loads unchanged."""
    w = torch.randint(-128, 127, (64, 512), dtype=torch.int8)
    s = torch.rand(64, 1) + 0.5
    sd = {"blocks.0.attn.wq.weight": w, "blocks.0.attn.wq.weight_scale": s,
          "blocks.0.attn.wq.comfy_quant": torch.zeros(3, dtype=torch.uint8)}
    path = str(tmp_path / "fake_int8.safetensors")
    save_file(sd, path)
    out = _load_krea2_state_dict(path)
    assert set(out) == {"blocks.0.attn.wq.weight",
                        "blocks.0.attn.wq.weight_scale"}
    torch.testing.assert_close(out["blocks.0.attn.wq.weight"], w)
    torch.testing.assert_close(out["blocks.0.attn.wq.weight_scale"], s)
