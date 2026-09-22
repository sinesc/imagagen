"""Tiled decode only, with CUDA memory history to see live blocks at OOM."""
import os
import sys
import time

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
pipe.text_encoder.cpu()
pipe.text_encoder_2.cpu()
torch.cuda.empty_cache()
torch.cuda.synchronize()

gen = torch.Generator(device=device).manual_seed(1234)
x = torch.randn(1, 4, 180, 430, device=device, dtype=vae_dtype, generator=gen)
print("before decode: allocated", torch.cuda.memory_allocated(device) / 2**30, "GiB", flush=True)

torch.cuda.memory._record_memory_history(max_entries=100000, stacks="python")

with torch.inference_mode():
    try:
        pipe.vae.use_tiling = True
        t0 = time.perf_counter()
        img = pipe.vae.decode(x, return_dict=False)[0]
        torch.cuda.synchronize()
        print("tiled ok:", time.perf_counter() - t0, "s peak",
              torch.cuda.max_memory_allocated(device) / 2**30, "GiB", flush=True)
    except torch.OutOfMemoryError as e:
        print("OOM:", e, flush=True)
        snap = torch.cuda.memory._snapshot()
        import collections
        # live blocks grouped by (size, device)
        tot = 0
        by_size = collections.Counter()
        for b in snap.segments:
            for blk in b.get("blocks", []):
                if blk.get("state") == "active_allocated":
                    by_size[blk["size"]] += 1
                    tot += blk["size"]
        print("live total GiB:", tot / 2**30, flush=True)
        for s, c in by_size.most_common(15):
            print(f"  {s/2**20:10.1f} MiB x{c}")
        sys.exit(1)
