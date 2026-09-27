"""Minimal `comfy` package stubs so the *unmodified* ComfyUI reference sources
can be imported with plain torch (the real comfy tree needs comfy_aimdo,
comfy_kitchen, folder_paths, psutil, ... which are not in this venv).

Every function copied below is a verbatim copy of the corresponding function
in dev/implementations/ComfyUI (marked with its source location), so that
`comfy/ldm/lumina/model.py` can be loaded AS-IS and used as the ground truth
for the imagagen port.

`install()` registers the stub modules in sys.modules and then loads the
reference NextDiT module from the given file path.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------- common_dit --
def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    # comfy/ldm/common_dit.py
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return torch.nn.functional.pad(img, pad, mode=padding_mode)


# ----------------------------------------------------------------- flux math --
def rope(pos, dim, theta):
    """comfy/ldm/flux/math.py::rope (fp64 path only -- supports_fp64 is True
    on this box, so the cpu fallback is not needed for the comparison)."""
    assert dim % 2 == 0
    device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def _apply_rope1(x, freqs_cis):
    # comfy/ldm/flux/math.py
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])

    return x_out.reshape(*x.shape).type_as(x)


def apply_rope(xq, xk, freqs_cis):
    # comfy/ldm/flux/math.py::_apply_rope (the in_training=True path)
    return _apply_rope1(xq, freqs_cis), _apply_rope1(xk, freqs_cis)


class EmbedND(nn.Module):
    # comfy/ldm/flux/layers.py
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


# --------------------------------------------------------------- mmdit/util --
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    # comfy/ldm/modules/diffusionmodules/util.py
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = timesteps.repeat_interleave(dim)
    return embedding


def default(x, y):
    return x if x is not None else y


class TimestepEmbedder(nn.Module):
    # comfy/ldm/modules/diffusionmodules/mmdit.py
    def __init__(self, hidden_size, frequency_embedding_size=256, output_size=None,
                 dtype=None, device=None, operations=None, max_period=10000):
        super().__init__()
        operations = default(operations, nn)
        if output_size is None:
            output_size = hidden_size
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, output_size, bias=True, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def forward(self, t, dtype, **kwargs):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, max_period=self.max_period).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


# ------------------------------------------------------------------ attention --
def optimized_attention_masked(q, k, v, heads, mask=None, attn_precision=None,
                               skip_reshape=False, skip_output_reshape=False, **kwargs):
    # comfy/ldm/modules/attention.py::attention_pytorch (the pytorch backend is
    # what a plain torch install would select)
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2), (q, k, v))

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


# ---------------------------------------------------------------------- utils --
def repeat_to_batch_size(tensor, batch_size, dim=0):
    # comfy/utils.py
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor


# ------------------------------------------------------------- patcher ext ----
class _Executor:
    def __init__(self, fn, obj):
        self.fn = fn
        self.obj = obj

    def execute(self, *args, **kwargs):
        # new_class_executor is handed the BOUND _forward method
        return self.fn(*args, **kwargs)


class WrapperExecutor:
    @staticmethod
    def new_class_executor(fn, obj, wrappers):
        return _Executor(fn, obj)


class WrappersMP:
    DIFFUSION_MODEL = "diffusion_model"


def get_all_wrappers(*args, **kwargs):
    return []


# ------------------------------------------------------------------- install ----
def install():
    """Register the stub `comfy` tree in sys.modules."""
    if "comfy" in sys.modules:
        return

    def mod(name):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        sys.modules[name] = m
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(mod(parent), child, m)
        return m

    comfy = mod("comfy")
    ldm = mod("comfy.ldm")
    mod("comfy.ldm.common_dit").pad_to_patch_size = pad_to_patch_size
    mm = mod("comfy.model_management")
    mm.in_training = True          # force the pure-torch rope/norm paths
    mm.supports_fp64 = lambda dev: True
    ops = mod("comfy.ops")
    ops.RMSNorm = nn.RMSNorm
    mod("comfy.quant_ops")
    diffusionmodules = mod("comfy.ldm.modules.diffusionmodules")
    mod("comfy.ldm.modules.diffusionmodules.mmdit").TimestepEmbedder = TimestepEmbedder
    mod("comfy.ldm.modules.attention").optimized_attention_masked = optimized_attention_masked
    flux = mod("comfy.ldm.flux")
    layers = mod("comfy.ldm.flux.layers")
    layers.EmbedND = EmbedND
    mathm = mod("comfy.ldm.flux.math")
    mathm.apply_rope = apply_rope
    pe = mod("comfy.patcher_extension")
    pe.WrapperExecutor = WrapperExecutor
    pe.WrappersMP = WrappersMP
    pe.get_all_wrappers = get_all_wrappers
    utils = mod("comfy.utils")
    utils.repeat_to_batch_size = repeat_to_batch_size
    chroma = mod("comfy.ldm.chroma_radiance")
    clayers = mod("comfy.ldm.chroma_radiance.layers")
    class _NerfEmbedder(nn.Module):  # unused by the latent-space NextDiT
        pass
    clayers.NerfEmbedder = _NerfEmbedder
    comfy.ldm = ldm
    comfy.utils = utils
    comfy.ops = ops
    comfy.model_management = mm


def load_reference(model_py: str):
    """import the UNMODIFIED comfy/ldm/lumina/model.py as a module."""
    install()
    spec = importlib.util.spec_from_file_location("comfy_ref_lumina_model", model_py)
    m = importlib.util.module_from_spec(spec)
    sys.modules["comfy_ref_lumina_model"] = m
    spec.loader.exec_module(m)
    return m
