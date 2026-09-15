"""Identify which UNet layer/shape OOMs at 3440x1440 and peak per stage."""
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(root, "cache"))

import numpy as np
import torch

device = torch.device("cuda:1")
model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "models/sdxl/epicrealismXL11.0.safetensors")
w, h = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (3440, 1440)

from diffusers import StableDiffusionXLPipeline

print("loading pipeline ...", flush=True)
pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)
pipe.to(device)
cap = torch.cuda.get_device_capability(device)
vae_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float32
pipe.vae.to(dtype=vae_dtype)
pipe.vae.enable_slicing()
pipe.enable_attention_slicing()

lh, lw = h // 8, w // 8
print(f"image {w}x{h} -> latent {lw}x{lh}", flush=True)

# which processor is installed?
procs = {type(p).__name__ for p in pipe.unet.attn_processors.values()}
print("attn processors:", procs, flush=True)

current = {}

def hook(name):
    def h(mod, inp):
        x = inp[0]
        current["name"] = name
        current["in"] = tuple(x.shape) if torch.is_tensor(x) else str(type(x))
    return h


names = {}
for name, mod in pipe.unet.named_modules():
    if hasattr(mod, "to_q"):
        names[name] = mod
        mod.register_forward_pre_hook(hook(name))
        mod.register_forward_hook(lambda mod, inp, out: current.update({"out": tuple(out.shape) if torch.is_tensor(out) else type(out).__name__}))

print("attn modules:", len(names), flush=True)
assert names

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


with torch.inference_mode():
    try:
        y = unet_forward()
        torch.cuda.synchronize()
        print("OK; unet forward peak:", torch.cuda.max_memory_allocated(device) / 2**30, "GiB", flush=True)
    except torch.OutOfMemoryError as e:
        print("OOM at module:", current["name"], flush=True)
        print("  input shape:", current["in"], flush=True)
        print(e, flush=True)
        sys.exit(1)
