"""Structural diff: run the REFERENCE SingleStreamDiT (dev/implementations/
krea-2/mmdit.py, CPU bf16) on the dequantized weights of my checkpoint, and
compare its output to my Krea2Transformer's (GPU bf16) output on the same
inputs. A structural bug in my port shows up here."""
import os
import sys

import torch

torch.compile = lambda f=None, **kw: (f if callable(f) else (lambda g: g))
# the reference decorates forwards with torch.compile(fullgraph=True); make it
# a no-op so the CPU run is plain eager
import time

import torch
from safetensors import safe_open

ROOT = "/workspaces/Projekte/imagagen"
CKPT = f"{ROOT}/models/krea2/krea2_Int8Extended.safetensors"
REF = f"{ROOT}/dev/implementations/krea-2"

sys.path.insert(0, f"{ROOT}/dev/scratch")  # einops shim
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, REF)

import mmdit as _mmdit
from mmdit import SingleMMDiTConfig, SingleStreamDiT  # reference


def _cpu_attention(q, k, v, mask=None, scale=0.08838834764831845, gqa=False):
    """CPU stand-in for the reference attention (enable_gqa is unavailable
    with an explicit mask on this torch build)."""
    if gqa:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    amask = None
    if mask is not None and not mask.all():
        # the reference passes a (B, 1, L, L) bool mask; SDPA on CPU wants
        # additive floats
        amask = torch.where(mask, torch.zeros((), dtype=q.dtype),
                            torch.full((), -1e9, dtype=q.dtype))
    x = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=amask, scale=scale)
    b, h, l, d = x.shape
    return x.transpose(1, 2).reshape(b, l, h * d)


_mmdit.attention = _cpu_attention

cfg = SingleMMDiTConfig(
    features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
    multiplier=4, layers=28, patch=2, channels=16,
    txtheads=20, txtkvheads=20, txtlayers=12)

# ---- dequantize my checkpoint to bf16 -------------------------------------
from imagagen.backends.krea2 import transformer as k2t

sd = {}
with safe_open(CKPT, framework="pt") as f:
    for k in f.keys():
        if k.endswith(".comfy_quant") or k.endswith(".weight_scale"):
            continue
        t = f.get_tensor(k)
        if t.dtype == torch.int8:
            scale = f.get_tensor(k[: -len(".weight")] + ".weight_scale")
            w = t.to(torch.float32) * scale.to(torch.float32)
            # the stored int8 weights are ConvRot-PRE-rotated (W @ H^T per
            # 256-group); the reference plain model wants the original W:
            # W = W_stored @ H (H is orthogonal). The 256-wide int8 GEMM
            # path instead rotates the ACTIVATION at runtime.
            if w.shape[1] % 256 == 0:
                h = k2t._hadamard(256, "cpu", torch.float32)
                shp = w.shape
                w = (w.view(w.shape[0], w.shape[1] // 256, 256) @ h).reshape(shp)
            t = w.to(torch.bfloat16)
        sd[k] = t.to(torch.bfloat16)
print("dequantized + unrotated", len(sd), "keys")

# ---- build the reference model and load -----------------------------------
with torch.device("meta"):
    ref = SingleStreamDiT(cfg)
ref.load_state_dict(sd, strict=True, assign=True)
ref = ref.to("cpu", dtype=torch.bfloat16).eval()
del sd
print("reference built + loaded")

# ---- my model --------------------------------------------------------------
from imagagen.config import load_config
from imagagen.models import ModelRegistry

icfg = load_config(ROOT)
reg = ModelRegistry(icfg, gpus=[1])
info = next(m for m in reg.models if m.family == "krea2")
backend = reg.backend_for(info.id)
device = backend.device

# ---- shared inputs ---------------------------------------------------------
torch.manual_seed(0)
x_lat = torch.randn(1, 16, 32, 32, dtype=torch.bfloat16)  # 256 px latent grid
p = 2
hp, wp = 16, 16
# the reference DiT eats PATCHIFIED tokens, feature order (c, ph, pw)
x_tok = x_lat.view(1, 16, hp, p, wp, p).permute(0, 2, 4, 1, 3, 5) \
    .reshape(1, hp * wp, 16 * p * p)
t = torch.tensor([0.5], dtype=torch.bfloat16)
ctx = torch.randn(1, 40, 12 * 2560, dtype=torch.bfloat16)

t0 = time.time()
with torch.no_grad():
    # 32x32 latent -> 16x16 = 256 image patches
    imgids = torch.zeros(16, 16, 3)
    imgids[..., 1] = torch.arange(16)[:, None]
    imgids[..., 2] = torch.arange(16)[None, :]
    pos = torch.cat([
        torch.zeros(1, 40, 3),
        imgids.reshape(1, 256, 3),
    ], dim=1).to(torch.float32)
    mask = torch.ones(1, 296, dtype=torch.bool)
    out_ref = ref(x_tok, ctx.reshape(1, 40, 12, 2560), t, pos, mask)
print(f"reference forward: {time.time() - t0:.1f}s")

t0 = time.time()
with torch.inference_mode():
    out_mine = backend.transformer(x_lat.to(device), t.to(device), ctx.to(device))
    out_mine = out_mine.cpu()
print(f"my forward: {time.time() - t0:.1f}s")

out_ref = (out_ref.view(1, hp, wp, 16, p, p).permute(0, 3, 1, 4, 2, 5)
           .reshape(1, 16, 32, 32))
d = (out_ref.float() - out_mine.float()).abs()
print(f"ref  : mean={out_ref.float().mean().item():.5f} std={out_ref.float().std().item():.5f}")
print(f"mine : mean={out_mine.float().mean().item():.5f} std={out_mine.float().std().item():.5f}")
print(f"diff : max={d.max().item():.5f} mean={d.mean().item():.5f} "
      f"rel={d.mean().item() / out_ref.float().std().item():.4f}")
import torch.nn.functional as Fn
print("cos:", Fn.cosine_similarity(
    out_ref.flatten()[None], out_mine.flatten()[None], dim=1).item())
