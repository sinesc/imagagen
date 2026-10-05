"""The Krea 2 VAE key remap (imagagen.krea2_vae).

The full 194-key mapping is verified against BOTH the real checkpoint
(``infra/qwen_image_vae.safetensors``, ComfyUI/Wan naming) and the default
``AutoencoderKLQwenImage`` state dict: every key maps to an existing
diffusers key, the shapes match exactly, and coverage is complete in both
directions (no key left over on either side).
"""

import json
from pathlib import Path

import pytest
from safetensors import safe_open

from imagagen.krea2_vae import (
    comfy_wan_to_diffusers_key,
    remap_krea2_vae_state_dict,
)

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "infra" / "qwen_image_vae.safetensors"
CFG = ROOT / "res" / "qwen_image_vae.config.json"


@pytest.fixture(scope="module")
def ckpt_sd():
    """The raw checkpoint state dict (ComfyUI/Wan naming)."""
    with safe_open(str(CKPT), framework="pt") as f:
        return {k: f.get_slice(k).get_shape() for k in f.keys()}


@pytest.fixture(scope="module")
def diffusers_sd():
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_config(
        json.loads(CFG.read_text()))
    return {k: tuple(p.shape) for k, p in vae.state_dict().items()}


def test_checkpoint_has_194_keys(ckpt_sd):
    assert len(ckpt_sd) == 194


def test_full_mapping_matches_diffusers(ckpt_sd, diffusers_sd):
    remapped = {comfy_wan_to_diffusers_key(k): shape
                for k, shape in ckpt_sd.items()}
    # no collision: every source key maps to a distinct diffusers key
    assert len(remapped) == 194
    # every remapped key exists in the model AND with the exact shape
    for key, shape in remapped.items():
        assert key in diffusers_sd, f"remapped key not in model: {key}"
        assert diffusers_sd[key] == tuple(shape), \
            f"shape mismatch for {key}: {diffusers_sd[key]} != {tuple(shape)}"
    # full coverage: the model has exactly these keys, no more
    assert set(remapped) == set(diffusers_sd)


def test_roundtrip_through_remap_function(ckpt_sd, diffusers_sd):
    """remap_krea2_vae_state_dict agrees with the per-key function."""
    remapped = remap_krea2_vae_state_dict({k: None for k in ckpt_sd})
    assert set(remapped) == set(diffusers_sd)


def test_individual_key_rules(ckpt_sd, diffusers_sd):
    cases = {
        "conv1.weight": "quant_conv.weight",
        "conv2.bias": "post_quant_conv.bias",
        "encoder.conv1.weight": "encoder.conv_in.weight",
        "encoder.head.0.gamma": "encoder.norm_out.gamma",
        "encoder.head.2.weight": "encoder.conv_out.weight",
        # encoder down_blocks: flat list, index N == down_blocks.N
        # (one resnet per entry, sub-parts directly on the block)
        "encoder.downsamples.0.residual.0.gamma":
            "encoder.down_blocks.0.norm1.gamma",
        "encoder.downsamples.1.residual.2.weight":
            "encoder.down_blocks.1.conv1.weight",
        "encoder.downsamples.2.resample.1.weight":
            "encoder.down_blocks.2.resample.1.weight",
        "encoder.downsamples.3.residual.6.weight":
            "encoder.down_blocks.3.conv2.weight",
        "encoder.downsamples.6.shortcut.weight":
            "encoder.down_blocks.6.conv_shortcut.weight",
        # middle: 0 -> resnets.0, 1 -> attentions.0, 2 -> resnets.1
        "encoder.middle.0.residual.0.gamma":
            "encoder.mid_block.resnets.0.norm1.gamma",
        "encoder.middle.1.norm.gamma":
            "encoder.mid_block.attentions.0.norm.gamma",
        "encoder.middle.1.to_qkv.weight":
            "encoder.mid_block.attentions.0.to_qkv.weight",
        "encoder.middle.2.residual.2.weight":
            "encoder.mid_block.resnets.1.conv1.weight",
        # decoder up_blocks: resnets + the resample/time_conv pair
        "decoder.upsamples.0.residual.0.gamma":
            "decoder.up_blocks.0.resnets.0.norm1.gamma",
        "decoder.upsamples.3.resample.1.weight":
            "decoder.up_blocks.0.upsamplers.0.resample.1.weight",
        "decoder.upsamples.3.time_conv.weight":
            "decoder.up_blocks.0.upsamplers.0.time_conv.weight",
        "decoder.upsamples.4.residual.0.gamma":
            "decoder.up_blocks.1.resnets.0.norm1.gamma",
        "decoder.head.0.gamma": "decoder.norm_out.gamma",
        "decoder.head.2.weight": "decoder.conv_out.weight",
    }
    for src, dst in cases.items():
        assert comfy_wan_to_diffusers_key(src) == dst
        assert dst in diffusers_sd


def test_config_has_latent_mean_std():
    cfg = json.loads(CFG.read_text())
    assert cfg["z_dim"] == 16
    assert len(cfg["latents_mean"]) == 16
    assert len(cfg["latents_std"]) == 16
    assert all(s > 0 for s in cfg["latents_std"])
