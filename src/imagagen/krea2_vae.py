"""Krea 2 VAE: the Qwen-Image VAE in ComfyUI/Wan naming -> diffusers.

``infra/qwen_image_vae.safetensors`` (194 bf16 keys) is the Qwen-Image VAE
(Wan-style architecture, 16 latent channels, 8x downsample) stored in
ComfyUI's flat ``Wan21`` naming:

- root ``conv1`` / ``conv2``: the quant / post-quant 1x1 convs;
- ``encoder.conv1`` / ``decoder.conv1``: ``conv_in``;
- ``encoder.head.0`` (``gamma``) / ``decoder.head.0``: ``norm_out``;
  ``encoder.head.2`` / ``decoder.head.2``: ``conv_out``;
- ``encoder.downsamples.N``: a FLAT list where index ``N`` IS the
  ``down_blocks.N`` index: each entry is either one resnet (its sub-parts
  sit directly on the block: ``down_blocks.N.norm1``/``conv1``/``norm2``/
  ``conv2``) or a resample (``down_blocks.N.resample``);
- ``decoder.upsamples.N``: a FLAT list where group ``g`` occupies indices
  ``4g .. 4g+3``: ``4g``, ``4g+1``, ``4g+2`` are the resnets
  (``up_blocks.g.resnets.r``) and ``4g+3`` holds the resample (+
  ``time_conv`` in the decoder's up_blocks);
- ``encoder.middle.{0,1,2}`` / ``decoder.middle.{0,1,2}``: ``mid_block``
  (0 = resnets.0, 1 = attentions.0, 2 = resnets.1);
- the Wan residual sub-part indices map to the diffusers names:
  ``residual.0`` -> ``norm1``, ``residual.2`` -> ``conv1``,
  ``residual.3`` -> ``norm2``, ``residual.6`` -> ``conv2``,
  ``shortcut`` -> ``conv_shortcut`` (the ``gamma`` LayerNormConv scales
  map onto the diffusers ``normN.gamma`` / ``norm_out.gamma`` /
  ``conv_in.weight`` params).

:func:`comfy_wan_to_diffusers_key` converts one key; the committed unit
test (``tests/unit/test_krea2_vae.py``) verifies the FULL mapping against
both the real checkpoint and the default ``AutoencoderKLQwenImage`` state
dict (194 keys each, exact shape match, full coverage).

Latent normalization is per-channel MEAN/STD (not scaling/shift): the
solver-space latent is ``z = (vae_latent - mean) / std`` and the decode
applies ``vae_latent = z * std + mean`` (the reference
``autoencoder.py``). The mean/std live in the VAE config (the committed
``res/qwen_image_vae.config.json``, the class defaults).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn

#: the flat-list group stride of the Wan down/up block indices
_RESNET_STRIDE = 4


def comfy_wan_to_diffusers_key(key: str) -> str:
    """Map one ComfyUI/Wan VAE key to its ``AutoencoderKLQwenImage``
    counterpart (see the module docstring for the rules)."""
    if key.startswith("conv1."):
        return "quant_conv." + key[len("conv1."):]
    if key.startswith("conv2."):
        return "post_quant_conv." + key[len("conv2."):]

    pre = "encoder" if key.startswith("encoder.") else "decoder"
    rest = key[len(pre) + 1:]
    if rest.startswith("conv1."):
        return pre + ".conv_in." + rest[len("conv1."):]
    if rest.startswith("head.0."):
        return pre + ".norm_out." + rest[len("head.0."):]
    if rest.startswith("head.2."):
        return pre + ".conv_out." + rest[len("head.2."):]
    if rest.startswith("middle."):
        n = int(re.match(r"middle\.(\d+)\.", rest).group(1))
        r = rest[len(f"middle.{n}."):]
        if r.startswith("residual."):
            # middle.0 -> resnets.0, middle.2 -> resnets.1 (1 is the attn)
            idx = 0 if n == 0 else 1
            return pre + f".mid_block.resnets.{idx}." + _resmap(r)
        return pre + ".mid_block.attentions.0." + r
    if pre == "encoder":
        n = int(re.match(r"downsamples\.(\d+)\.", rest).group(1))
        return pre + f".down_blocks.{n}." + _resmap(rest[len(f"downsamples.{n}."):])
    # decoder flat upsamples: resample/time_conv sit at index 4g+3 (g = 0..2)
    n = int(re.match(r"upsamples\.(\d+)\.", rest).group(1))
    r = rest[len(f"upsamples.{n}."):]
    if r.startswith("resample.") or r.startswith("time_conv."):
        return f"decoder.up_blocks.{(n - 3) // _RESNET_STRIDE}.upsamplers.0." + r
    g, rr = divmod(n, _RESNET_STRIDE)
    return f"decoder.up_blocks.{g}.resnets.{rr}." + _resmap(r)


def _resmap(rest: str) -> str:
    """The Wan residual / shortcut sub-part names to the diffusers ones."""
    rest = re.sub(r"^residual\.0\.", "norm1.", rest)
    rest = re.sub(r"^residual\.2\.", "conv1.", rest)
    rest = re.sub(r"^residual\.3\.", "norm2.", rest)
    rest = re.sub(r"^residual\.6\.", "conv2.", rest)
    return re.sub(r"^shortcut\.", "conv_shortcut.", rest)


def remap_krea2_vae_state_dict(sd: dict) -> dict:
    """Remap a whole ComfyUI/Wan VAE state dict to diffusers keys."""
    return {comfy_wan_to_diffusers_key(k): t for k, t in sd.items()}


class ImageVAE(nn.Module):
    """The Wan video VAE presented as an image VAE.

    ``AutoencoderKLQwenImage`` expects/returns 5D video tensors
    ``(1, c, frames, h, w)``; imagagen works with 4D images
    ``(1, c, h, w)``. This wrapper does the reshape so the backend can
    treat it exactly like the other families' VAEs (``.dtype``,
    ``.parameters()``, ``.to()`` all pass through).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        # unused by this VAE's decode (it derives the latent tile sizes
        # and the blend from the sample sizes); stored so the backend can
        # set them uniformly across families
        self.tile_latent_min_size = 64
        self.tile_overlap_factor = 0.5

    @property
    def dtype(self):
        return next(self.vae.parameters()).dtype

    @property
    def config(self):
        return self.vae.config

    def encode(self, x):
        # the frame dim compresses 1 -> 1; squeeze it out of the latent
        # distribution too, so the backend sees 4D images end-to-end
        from diffusers.models.autoencoders.autoencoder_kl import (
            AutoencoderKLOutput)
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        dist = self.vae.encode(x.unsqueeze(2)).latent_dist
        mean, logvar = dist.mean.squeeze(2), dist.logvar.squeeze(2)
        return AutoencoderKLOutput(
            latent_dist=DiagonalGaussianDistribution(torch.cat([mean, logvar], 1)))

    def decode(self, x, return_dict=True):
        out = self.vae.decode(x.unsqueeze(2), return_dict=return_dict)
        if return_dict:
            out["sample"] = out["sample"].squeeze(2)
            return out
        return (out[0].squeeze(2),)

    # the backend drives slicing/tiling through the classic AutoencoderKL
    # API (``enable_slicing()``, ``use_tiling``, ``tile_sample_min_size``,
    # ``tile_latent_min_size``, ``tile_overlap_factor``); the QwenImage VAE
    # implements it with the height/width spellings, so map it
    def enable_slicing(self) -> None:
        self.vae.enable_slicing()

    @property
    def use_tiling(self):
        return self.vae.use_tiling

    @use_tiling.setter
    def use_tiling(self, value):
        self.vae.use_tiling = value

    @property
    def tile_sample_min_size(self):
        return self.vae.tile_sample_min_height

    @tile_sample_min_size.setter
    def tile_sample_min_size(self, value):
        self.vae.tile_sample_min_height = value
        self.vae.tile_sample_min_width = value



def load_krea2_vae(config, device):
    """Load ``infra/qwen_image_vae.safetensors`` into a bf16
    :class:`ImageVAE` (the Wan ``AutoencoderKLQwenImage`` reshaped to
    4D image I/O) on ``device`` (architecture from the committed
    ``res/qwen_image_vae.config.json`` — the class defaults, so nothing
    is fetched)."""
    from safetensors import safe_open
    from diffusers import AutoencoderKLQwenImage
    from diffusers.utils.logging import disable_progress_bar

    disable_progress_bar()

    cfg_path = config.root / "res" / "qwen_image_vae.config.json"
    with safe_open(str(config.infra_dir / "qwen_image_vae.safetensors"),
                   framework="pt") as f:
        sd = remap_krea2_vae_state_dict({k: f.get_tensor(k) for k in f.keys()})

    vae = AutoencoderKLQwenImage.from_config(
        json.loads(cfg_path.read_text()))
    missing = list(vae.load_state_dict(sd, strict=True).missing_keys)
    if missing:
        raise ValueError(f"Krea 2 VAE checkpoint mismatch: missing={missing[:5]}")
    return ImageVAE(vae).to(device, dtype=torch.bfloat16)


def _per_channel(vae, latent: torch.Tensor):
    """The config's (16,) mean/std, reshaped to broadcast over ``latent``
    ((1, 16, h, w), channel dim 1)."""
    shape = (1, 16) + (1,) * (latent.ndim - 2)
    mean = torch.tensor(vae.config.latents_mean, device=latent.device,
                        dtype=latent.dtype).reshape(shape)
    std = torch.tensor(vae.config.latents_std, device=latent.device,
                       dtype=latent.dtype).reshape(shape)
    return mean, std


def normalize_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    """Pixels-space -> solver-space latent: ``(vae_latent - mean) / std``
    (per channel; ``latent`` is (1, 16, h, w), the mean/std (16,))."""
    mean, std = _per_channel(vae, latent)
    return (latent - mean) / std


def denormalize_latent(vae, latent: torch.Tensor) -> torch.Tensor:
    """Solver-space -> pixels-space latent: ``z * std + mean``."""
    mean, std = _per_channel(vae, latent)
    return latent * std + mean
