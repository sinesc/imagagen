#!/usr/bin/env python3
"""Bisect the caption 'distorted image' hallucination: compare our
Qwen3-VL vision tower's merged + deepstack features against the HF
transformers Qwen3VL vision model on the SAME image (fp8 checkpoint
dequantized to bf16). Match => bug is in the text/caption path;
mismatch => bug is in our vision port."""
import sys

sys.path.insert(0, "src")
import numpy as np
import torch
from PIL import Image

dev = "cuda:1"
img = np.asarray(Image.open("dev/references/purple-flower.png").convert("RGB"))
arr = torch.from_numpy(img.astype(np.float32) / 255.0)[None]

from imagagen.encoders.qwen3vl_vision import load_vision_tower, process_qwen2vl_images

patches, grid = process_qwen2vl_images(arr)

# ---- ours (fp8 native, fp32 compute) ----
mine = load_vision_tower("infra/qwen_3vl_4b.safetensors", dev)
# replicate the tower forward up to (excluding) the merger to match the
# HF vision model's pre-merge output
v = mine.visual
with torch.no_grad():
    x = v.patch_embed(patches.to(dev))
    x = x + v.fast_pos_embed_interpolate(grid.to(dev)).to(x.dtype)
    rotary = v.rot_pos_emb(grid.to(dev)).to(x.device)
    seq_len = x.shape[0]
    x = x.reshape(seq_len, -1)
    rotary = rotary.reshape(seq_len, -1)
    emb = torch.cat((rotary, rotary), dim=-1)
    cos = emb.cos().unsqueeze(-2)
    sin = emb.sin().unsqueeze(-2)
    sh = sin.shape[-1] // 2
    pe = (cos, sin[..., :sh], -sin[..., sh:])
    import torch.nn.functional as Fnn
    cu = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(0, dtype=torch.int32)
    cu = Fnn.pad(cu, (1, 0), value=0)
    for blk in v.blocks:
        x = blk(x, cu_seqlens=cu, position_embeddings=pe)
m_mine = x.float()
ds_mine = []

# ---- HF reference (bf16 dequant) ----
import safetensors.torch as st
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

vconf = Qwen3VLVisionConfig(
    deepstack_visual_indexes=[5, 11, 17], depth=24, hidden_size=1024,
    intermediate_size=4096, num_heads=16, in_channels=3, patch_size=16,
    spatial_merge_size=2, temporal_patch_size=2,
    num_position_embeddings=2304, out_hidden_size=2560)
hf = Qwen3VLVisionModel(vconf).to(device=dev, dtype=torch.bfloat16).eval()

file = st.load_file("infra/qwen_3vl_4b.safetensors")
sd = {}
for k, t in file.items():
    if not k.startswith("model.visual."):
        continue
    kk = k[len("model.visual."):]
    if t.dtype == torch.float8_e4m3fn:
        t = t.to(torch.bfloat16)
    sd[kk] = t
missing, unexpected = hf.load_state_dict(sd, strict=False)
print("hf missing:", missing[:6], "unexpected:", unexpected[:6])

with torch.no_grad():
    out = hf(patches.to(dev, dtype=torch.bfloat16), grid.to(dev))
# HF returns (hidden, deepstack) in its forward
out = hf(patches.to(dev, dtype=torch.bfloat16), grid.to(dev))
m_hf = out.last_hidden_state
ds_hf = getattr(out, "deepstack_features", []) or []
m_hf = m_hf.float()

print("pre-merge shape mine", tuple(m_mine.shape), "hf", tuple(m_hf.shape))
cos = torch.nn.functional.cosine_similarity(m_mine, m_hf, dim=-1)
print(f"merged row cosine: mean={cos.mean():.4f} min={cos.min():.4f}")
for i, d in enumerate(ds_hf[:0]):
    c = torch.nn.functional.cosine_similarity(ds_mine[i], d.float(), dim=-1)
    print(f"deepstack[{i}] cosine: mean={c.mean():.4f} min={c.min():.4f}")
