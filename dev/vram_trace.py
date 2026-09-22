import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(root, "cache"))

import torch

device = torch.device("cuda:1")

_orig_baddbmm = torch.baddbmm
def traced_baddbmm(*a, **k):
    if a and torch.is_tensor(a[0]):
        s = tuple(a[0].shape)
        if s and s[0] == 20 and s[1] > 5000:
            print(f"[baddbmm] out_shape={s} q={tuple(a[1].shape)} k={tuple(a[2].shape)} "
                  f"alloc={torch.cuda.memory_allocated(device)/2**30:.2f} GiB", flush=True)
    return _orig_baddbmm(*a, **k)

torch.baddbmm = traced_baddbmm
torch.Tensor.baddbmm = traced_baddbmm

_orig_softmax = torch.softmax
def traced_softmax(x, *a, **k):
    if x is not None and torch.is_tensor(x) and x.dim() >= 3:
        print(f"[softmax] in={tuple(x.shape)} dtype={x.dtype}", flush=True)
    return _orig_softmax(x, *a, **k)
torch.softmax = traced_softmax

_orig_empty = torch.empty
def traced_empty(*a, **k):
    if a and torch.is_tensor(a[0]) is False:
        n = 1
        for x in a:
            if isinstance(x, int):
                n *= x
        if n > 2e9:
            print(f"[empty] {n/2**30:.2f} GiB elems? shape={a[:4]}", flush=True)
    return _orig_empty(*a, **k)

from diffusers import StableDiffusionXLPipeline

print("loading pipeline ...", flush=True)
pipe = StableDiffusionXLPipeline.from_single_file(
    sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "models/sdxl/epicrealismXL11.0.safetensors"),
    torch_dtype=torch.float16)
pipe.to(device)
cap = torch.cuda.get_device_capability(device)
vae_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float32
pipe.vae.to(dtype=vae_dtype)
pipe.vae.enable_slicing()
pipe.enable_attention_slicing()
print("processor:", {type(p).__name__ for p in pipe.unet.attn_processors.values()}, flush=True)

w, h = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (3440, 1440)
lh, lw = h // 8, w // 8
print(f"latent {lw}x{lh} = {lw*lh} tokens", flush=True)

emb = torch.zeros(1, 154, 2048, dtype=torch.float16, device=device)
pooled = torch.zeros(1, 1280, dtype=torch.float16, device=device)
time_ids = torch.tensor([h, w, 0, 0, h, w], device=device, dtype=torch.float16).unsqueeze(0)
x = torch.randn(1, 4, lh, lw, device=device, dtype=torch.float32)
sigma = torch.tensor([1.5], device=device)
t = torch.tensor([700.0], device=device)

current = {}
for name, mod in pipe.unet.named_modules():
    if hasattr(mod, "to_q"):
        def mk(n):
            def h(mod, inp):
                current["name"] = n
                current["in"] = tuple(inp[0].shape)
            return h
        mod.register_forward_pre_hook(mk(name))

def unet_forward():
    x_in = (x / (sigma ** 2 + 1.0).sqrt()).to(torch.float16)
    return pipe.unet(x_in, t, encoder_hidden_states=emb,
                     added_cond_kwargs={"time_ids": time_ids, "text_embeds": pooled}).sample.float()

with torch.inference_mode():
    try:
        unet_forward()
        torch.cuda.synchronize()
        print("OK peak:", torch.cuda.max_memory_allocated(device) / 2**30, "GiB", flush=True)
    except torch.OutOfMemoryError as e:
        print("OOM at:", current["name"], "in=", current.get("in"), flush=True)
        print(e, flush=True)
        sys.exit(1)
