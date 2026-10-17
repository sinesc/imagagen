"""Compare tiled vs untiled VAE decode at 3440x1440: pixel diff + speed."""
import os
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

gen = torch.Generator(device=device).manual_seed(1234)
x = torch.randn(1, 4, 180, 430, device=device, dtype=vae_dtype, generator=gen)


def decode(tiled):
    pipe.vae.use_tiling = tiled
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        img = pipe.vae.decode(x, return_dict=False)[0]
    torch.cuda.synchronize()
    return (img.float().cpu().numpy(), time.perf_counter() - t0,
            torch.cuda.max_memory_allocated(device) / 2**30)


a, ta, pa = decode(False)
b, tb, pb = decode(True)
d = np.abs(a.astype(np.int16) - b.astype(np.int16))
print(f"untiled:   {ta:7.2f} s  peak {pa:7.2f} GiB")
print(f"tiled:     {tb:7.2f} s  peak {pb:7.2f} GiB")
print(f"diff: max {d.max()}  mean {d.mean():.4f}  "
      f"p>1 {(d > 1).mean() * 100:.4f}%  p>2 {(d > 2).mean() * 100:.4f}%")
