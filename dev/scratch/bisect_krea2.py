"""Layer-by-layer bisection between the reference SingleStreamDiT and my
Krea2Transformer on identical inputs."""
import sys

ROOT = "/workspaces/Projekte/imagagen"
sys.path.insert(0, f"{ROOT}/dev/scratch")
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/dev/implementations/krea-2")

import torch

torch.compile = lambda f=None, **kw: (f if callable(f) else (lambda g: g))

import mmdit as _mmdit
from mmdit import SingleMMDiTConfig, SingleStreamDiT


def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
    if gqa:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    amask = None
    if mask is not None and not mask.all():
        amask = torch.where(mask, torch.zeros((), dtype=q.dtype),
                            torch.full((), -1e9, dtype=q.dtype))
    x = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=amask, scale=scale)
    b, h, seq, d = x.shape
    return x.transpose(1, 2).reshape(b, seq, h * d)


_mmdit.attention = _cpu_attention

CKPT = f"{ROOT}/models/krea2/krea2_Int8Extended.safetensors"
from safetensors import safe_open

sd = {}
with safe_open(CKPT, framework="pt") as f:
    for k in f.keys():
        if k.endswith(".comfy_quant") or k.endswith(".weight_scale"):
            continue
        t = f.get_tensor(k)
        if t.dtype == torch.int8:
            scale = f.get_tensor(k[: -len(".weight")] + ".weight_scale")
            t = t.to(torch.bfloat16) * scale.to(torch.bfloat16)
        sd[k] = t.to(torch.bfloat16)

cfg = SingleMMDiTConfig(features=6144, tdim=256, txtdim=2560, heads=48,
                        kvheads=12, multiplier=4, layers=28, patch=2,
                        channels=16, txtheads=20, txtkvheads=20, txtlayers=12)
with torch.device("meta"):
    ref = SingleStreamDiT(cfg)
ref.load_state_dict(sd, strict=True, assign=True)
ref = ref.to("cpu", dtype=torch.bfloat16).eval()
del sd

from imagagen.backends.krea2 import transformer as k2t
from imagagen.config import load_config
from imagagen.models import ModelRegistry

icfg = load_config(ROOT)
reg = ModelRegistry(icfg, gpus=[1])
info = next(m for m in reg.models if m.family == "krea2")
backend = reg.backend_for(info.id)
device = backend.device
mine = backend.transformer

torch.manual_seed(0)
x_lat = torch.randn(1, 16, 32, 32, dtype=torch.bfloat16)
p = 2
hp, wp = 16, 16
x_tok = x_lat.view(1, 16, hp, p, wp, p).permute(0, 2, 4, 1, 3, 5) \
    .reshape(1, hp * wp, 16 * p * p)
t = torch.tensor([0.5], dtype=torch.bfloat16)
ctx = torch.randn(1, 40, 12, 2560, dtype=torch.bfloat16)

imgids = torch.zeros(16, 16, 3)
imgids[..., 1] = torch.arange(16)[:, None]
imgids[..., 2] = torch.arange(16)[None, :]
pos = torch.cat([torch.zeros(1, 40, 3), imgids.reshape(1, 256, 3)],
                dim=1).to(torch.float32)


def cmp(name, a, b):
    a, b = a.float().cpu(), b.float().cpu()
    d = (a - b).abs()
    rel = d.max().item() / max(b.abs().max().item(), 1e-9)
    flag = "OK " if rel < 0.02 else "BAD"
    print(f"{flag} {name:28s} maxabs={d.max().item():10.5f} "
          f"rel={rel:10.5f}")
    return d.max().item()


# temb
temb_ref = _mmdit.temb(t, 256, device="cpu", dtype=torch.bfloat16)
temb_mine = k2t.timestep_embedding(t, 256).unsqueeze(1)
cmp("temb", temb_ref, temb_mine)

# first / tmlp / tproj
img_ref = ref.first(x_tok)
img_mine = mine.first(x_tok.to(device)).cpu()
cmp("first", img_ref, img_mine)

t_ref = ref.tmlp(temb_ref)
t_mine = mine.tmlp(temb_mine.to(device)).cpu()
cmp("tmlp", t_ref, t_mine)
tvec_ref = ref.tproj(t_ref)
tvec_mine = mine.tproj(t_mine.to(device)).cpu()
cmp("tproj", tvec_ref, tvec_mine)

# txtfusion
tf_ref = ref.txtfusion(ctx)
tf_mine = mine.txtfusion(ctx.to(device)).cpu()
cmp("txtfusion", tf_ref, tf_mine)
cm_ref = ref.txtmlp(tf_ref)
cm_mine = mine.txtmlp(tf_mine.to(device)).cpu()
cmp("txtmlp", cm_ref, cm_mine)

# freqs: reference pads combined to 384; compare the first 296
freqs_ref = ref.posemb(pos)
freqs_mine = mine.pe_embedder(pos.to(device)).cpu()
cmp("freqs (unpadded)", freqs_ref[:, :296], freqs_mine)

# block 0
block_in_ref = torch.cat([cm_ref, img_ref], dim=1)
block_in_mine = torch.cat([cm_mine.to(device), img_mine.to(device)], dim=1)
out_ref = ref.blocks[0](block_in_ref, tvec_ref, freqs_ref[:, :296])
out_mine = mine.blocks[0](block_in_mine, tvec_mine.to(device),
                          freqs_mine.to(device)).cpu()
cmp("block0", out_ref, out_mine)

# ---- txtfusion deep dive ----
xf_ref = ref.txtfusion
xf_mine = mine.txtfusion
x = ctx
xr = ctx.reshape(40, 12, 2560)
xm = ctx.reshape(40, 12, 2560).to(device)
for i in range(2):
    rb = xf_ref.layerwise_blocks[i]
    mb = xf_mine.layerwise_blocks[i]
    pn_r = rb.prenorm(xr)
    pn_m = mb.prenorm(xm).cpu()
    cmp(f"lay{i}.prenorm", pn_r, pn_m)
    q_r, k_r, v_r = rb.attn.wq(pn_r), rb.attn.wk(pn_r), rb.attn.wv(pn_r)
    g_r = rb.attn.gate(pn_r)
    q_m = mb.attn.wq(pn_m.to(device)).cpu()
    k_m = mb.attn.wk(pn_m.to(device)).cpu()
    g_m = mb.attn.gate(pn_m.to(device)).cpu()
    cmp(f"lay{i}.wq", q_r, q_m)
    cmp(f"lay{i}.gate", g_r, g_m)
    vview_r = v_r.view(40, 12, 20, 128).transpose(1, 2)
    vview_m = v_r.view(40, 12, 20, 128).transpose(1, 2).to(device)
    qn_r, kn_r, _ = rb.attn.qknorm(q_r.view(40, 12, 20, 128).transpose(1, 2),
                                   k_r.view(40, 12, 20, 128).transpose(1, 2),
                                   vview_r)
    qn_m, kn_m = mb.attn.qknorm(q_m.view(40, 12, 20, 128).transpose(1, 2),
                                k_m.view(40, 12, 20, 128).transpose(1, 2))
    cmp(f"lay{i}.qnorm", qn_r, qn_m)
    out_r = _cpu_attention(qn_r, kn_r, vview_r)
    out_m = _cpu_attention(qn_m.to(device), kn_m.to(device), vview_m).cpu()
    cmp(f"lay{i}.sdpa", out_r, out_m)
    wo_r = rb.attn.wo(out_r * torch.sigmoid(g_r))
    wo_m = mb.attn.wo((out_m * torch.sigmoid(g_m)).to(device)).cpu()
    cmp(f"lay{i}.wo*gate", wo_r, wo_m)
    po_r = rb.postnorm(xr + wo_r)
    po_m = mb.postnorm(xm + wo_m.to(device)).cpu()
    cmp(f"lay{i}.postnorm", po_r, po_m)
    mlp_r = rb.mlp(po_r)
    mlp_m = mb.mlp(po_m.to(device)).cpu()
    cmp(f"lay{i}.mlp", mlp_r, mlp_m)
    xr = xr + mlp_r
    xm = (xm + mlp_m.to(device))
    cmp(f"lay{i}.out", xr, xm)

pr_r = xf_ref.projector(
    xr.reshape(1, 40, 12, 2560).permute(0, 1, 3, 2)).squeeze(-1)
pr_m = xf_mine.projector(
    xm.reshape(1, 40, 12, 2560).to(device).permute(0, 1, 3, 2)).squeeze(-1).cpu()
print("pr_r", pr_r.shape, "pr_m", pr_m.shape)
cmp("projector", pr_r, pr_m)
for i in range(2):
    rb = xf_ref.refiner_blocks[i]
    mb = xf_mine.refiner_blocks[i]
    rr = rb(pr_r)
    rm = mb(pr_m.to(device)).cpu()
    cmp(f"ref{i}", rr, rm)
    pr_r = rr

# block0 subparts (text path fixed? use the REF text stream)
b = ref.blocks[0]
b2 = mine.blocks[0]
block_in_ref2 = torch.cat([cm_ref, img_ref], dim=1)
block_in_mine2 = torch.cat([cm_ref.to(device), img_mine.to(device)], dim=1)
ps_ref = b.mod(tvec_ref)
ps_mine = [p.cpu() for p in b2.mod(tvec_mine.to(device))]
for i, nm in enumerate(("prescale", "preshift", "pregate",
                        "postscale", "postshift", "postgate")):
    cmp(f"mod.{nm}", ps_ref[i], ps_mine[i])
pn_ref = b.prenorm(block_in_ref2)
pn_mine = b2.prenorm(block_in_mine2.to(device)).cpu()
cmp("block0.prenorm", pn_ref, pn_mine)
mod_in_r = (1 + ps_ref[0]) * pn_ref + ps_ref[1]
mod_in_m = (1 + ps_mine[0]) * pn_mine + ps_mine[1]
cmp("block0.mod.in", mod_in_r, mod_in_m)
q_r, k_r = b.attn.wq(mod_in_r), b.attn.wk(mod_in_r)
q_m, k_m = b2.attn.wq(mod_in_m.to(device)).cpu(), \
    b2.attn.wk(mod_in_m.to(device)).cpu()
cmp("block0.wq", q_r, q_m)
vr = b.attn.wv(mod_in_r)
vview_b = vr.view(1, 296, 12, 128).transpose(1, 2)
qkn_ref = b.attn.qknorm(q_r.view(1, 296, 48, 128).transpose(1, 2),
                        k_r.view(1, 296, 12, 128).transpose(1, 2), vview_b)
qkn_mine = b2.attn.qknorm(
    q_m.view(1, 296, 48, 128).transpose(1, 2),
    k_m.view(1, 296, 12, 128).transpose(1, 2))
cmp("block0.qnorm", qkn_ref[0], qkn_mine[0])
qr_ref = _mmdit.ropeapply(qkn_ref[0], qkn_ref[1], freqs_ref[:, :296])
qr_mine = k2t._rope_apply(qkn_mine[0], freqs_mine[:, :, :296])
cmp("block0.rope.q", qr_ref[0], qr_mine)
vr = b.attn.wv(mod_in_r)
gr = b.attn.gate(mod_in_r)
v_r2 = vr.view(1, 296, 12, 128).transpose(1, 2)
qrk_r = (qr_ref[0], qr_ref[1])
qrk_m = (qr_mine, k2t._rope_apply(qkn_mine[1], freqs_mine[:, :, :296]))
v_m2 = b2.attn.wv(mod_in_m.to(device)).cpu().view(
    1, 296, 12, 128).transpose(1, 2)
out_r = _cpu_attention(qrk_r[0], qrk_r[1], v_r2, gqa=True)
out_m = _cpu_attention(qrk_m[0], qrk_m[1], v_m2, gqa=True)
cmp("block0.sdpa", out_r, out_m)
