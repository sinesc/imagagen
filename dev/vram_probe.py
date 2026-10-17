"""Probe: measure peak VRAM of each stage of an SDXL generation at 3440x1440.

Mimics backends/sdxl exactly (fp16 UNet, bf16 VAE, slicing + attention
slicing) and reports per-stage torch.cuda.max_memory_allocated so we can
see where imagagen exceeds what Forge (partial offload + 'GPU weights'
reserve) needs.
"""
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(root, "cache"))

import numpy as np
import torch

device = torch.device("cuda:1")
model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    root, "models/sdxl/epicrealismXL11.0.safetensors")
w, h = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (3440, 1440)

from diffusers import StableDiffusionXLPipeline

print("loading pipeline ...", flush=True)
pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)
pipe.to(device)
cap = torch.cuda.get_device_capability(device)
vae_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float32
pipe.vae.to(dtype=vae_dtype)
pipe.vae.enable_slicing()


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        return self

    def __exit__(self, *a):
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device)
        print(f"== {self.name}: peak {peak/2**30:.3f} GiB", flush=True)
        torch.cuda.reset_peak_memory_stats(device)


with torch.inference_mode():
    lh, lw = h // 8, w // 8
    print(f"image {w}x{h} -> latent {lw}x{lh}", flush=True)

    emb = torch.zeros(1, 154, 2048, dtype=torch.float16, device=device)
    pooled = torch.zeros(1, 1280, dtype=torch.float16, device=device)
    time_ids = torch.tensor([h, w, 0, 0, h, w], device=device, dtype=torch.float16).unsqueeze(0)
    unet = pipe.unet
    x = torch.randn(1, 4, lh, lw, device=device, dtype=torch.float32)
    sigma = torch.tensor([1.5], device=device)
    t = torch.tensor([700.0], device=device)

    def unet_forward():
        x_in = (x / (sigma ** 2 + 1.0).sqrt()).to(torch.float16)
        return unet(x_in, t, encoder_hidden_states=emb,
                    added_cond_kwargs={"time_ids": time_ids, "text_embeds": pooled}).sample.float()

    unet_forward()  # warmup (allocator sizing)
    torch.cuda.reset_peak_memory_stats(device)
    with Stage(f"unet forward x2 (cfg) @ latent {lw}x{lh}"):
        unet_forward()
        unet_forward()
    e = unet_forward()
    with Stage("solver x0 fp32 bookkeeping"):
        x0 = x - sigma * e
    pipe.text_encoder.cpu()
    pipe.text_encoder_2.cpu()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    print(f"   (static unet+vae resident: {base/2**30:.2f} GiB)", flush=True)

    lat = (x0 / pipe.vae.config.scaling_factor).to(vae_dtype)
    with Stage("vae.decode (sliced, no tiling)"):
        img = pipe.vae.decode(lat, return_dict=False)[0]

    torch.cuda.reset_peak_memory_stats(device)
    with Stage("vae.decode (sliced + tiled)"):
        pipe.vae.enable_tiling()
        img2 = pipe.vae.decode(lat, return_dict=False)[0]

    torch.cuda.reset_peak_memory_stats(device)
    with Stage("vae.encode (init image)"):
        arr = (img2 / 2 + 0.5).clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0)
        arr = np.ascontiguousarray((arr * 255).astype(np.uint8))
        lat2 = pipe.vae.encode(
            torch.from_numpy(arr).to(device).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        ).to(dtype=pipe.vae.dtype).latent_dist.mode()

    print(f"total device reserved: {torch.cuda.memory_reserved(device)/2**30:.2f} GiB", flush=True)
    print("done", flush=True)
