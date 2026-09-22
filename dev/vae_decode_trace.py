"""Locate where untiled VAE decode at 3440x1440 holds ~6 GiB."""
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(root, "cache"))
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

device = torch.device("cuda:1")
pipe = StableDiffusionXLPipeline.from_single_file(
    os.path.join(root, "models/sdxl/epicrealismXL11.0.safetensors"), torch_dtype=torch.float16)
pipe.to(device)
cap = torch.cuda.get_device_capability(device)
vae_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float32
pipe.vae.to(dtype=vae_dtype)
pipe.vae.enable_slicing()

peaks = []
for name, mod in list(pipe.vae.decoder.named_modules()) + [("mid", pipe.vae.decoder.mid_block)]:
    if not isinstance(mod, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        continue
    def mk(n):
        def h(mod, inp, out):
            peaks.append((torch.cuda.max_memory_allocated(device), n, tuple(out.shape)))
        mod.register_forward_hook(h)
peaks.sort(key=lambda t: t[0], reverse=True)
for p, n, s in peaks[:8]:
    print(f"{p/2**30:8.3f} GiB  {n}  {s}")

procs = set()
for name, mod in pipe.vae.decoder.named_modules():
    if hasattr(mod, "processor"):
        procs.add(type(mod.processor).__name__)
print("vae decoder attn processors:", procs)
x = torch.randn(1, 4, 180, 430, device=device, dtype=vae_dtype)
with torch.inference_mode():
    pipe.vae._use_tiling = False
    torch.cuda.reset_peak_memory_stats(device)
    img = pipe.vae.decode(x, return_dict=False)[0]
    print("decode peak:", torch.cuda.max_memory_allocated(device) / 2**30, "GiB", flush=True)
print("done")
