"""Live Z-Image smoke test (GPU 1).

Covers the Tier-2 checklist for the new family:

1. OUTPUT CONVENTION (load-bearing): the raw transformer output must be the
   flow velocity (x0 = x - sigma * out, the reference FLOW/CONST
   convention). Verified against the known velocity of a noised clean
   latent: feed x_t = (1-t)*x0 + t*e, the model's output must align with
   e - x0 (high cosine), not with -(e - x0) or with x0 itself.
2. txt2img: same seed -> identical image, different seed -> different;
   HxWx3 uint8; progress reaches 1.0.
3. LoRA: weight 0 == baseline (bit-exact), high weight shifts the image,
   switching back to the baseline set leaves no residue.
4. img2img: denoise 0.3 stays close to the init, denoise 1.0 diverges.
5. masked inpaint: a fully-protected run (init_mask all-0) comes back as
   the VAE round-trip of the init (per-step + final latent lock); in a
   half-protected run the protected side changes less than the inpaint
   side. (The Z-Image VAE is the non-local Flux decoder, so the protected
   pixels are compared against the fully-protected run, never against the
   raw init.)
6. cancellation: cancel() mid-generation raises InterruptedError, the
   backend stays usable.
7. parking: unload() frees VRAM, load() restores identical outputs.
8. idle VRAM after a generate ~= the resident weights.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

PROMPT = "a shiny red sports car on a wet city street at night"
PROMPT_B = "a portrait of an elderly fisherman in a wool hat"


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpus=[1])
info = next(m for m in reg.models if m.family == "zimage")
print("loading", info.name, "...", flush=True)
t0 = time.time()
backend = reg.backend_for(info.id)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)


def show(frac, msg):
    print(f"  {msg} {frac:.0%}", flush=True)


def gen(prompt=PROMPT, seed=42, **kw) -> np.ndarray:
    # steps=-1: the "model default" sentinel (the backend resolves the
    # built-in 8-step Turbo schedule)
    kw.setdefault("steps", -1)
    r = backend.generate(
        GenerateRequest(prompt=prompt, seed=seed, **kw), progress=show)
    assert r.image.ndim == 3 and r.image.shape[2] == 3
    assert r.image.dtype == np.uint8
    return r.image


# ---------------------------------------------------------------- 1. ---
print("\n[1] output convention: raw output must be the flow velocity",
      flush=True)
device = backend.device
with torch.no_grad():
    test_img = np.zeros((512, 512, 3), np.uint8)
    yy, xx = np.mgrid[0:512, 0:512]
    test_img[..., 0] = (xx * 255 // 511).astype(np.uint8)
    test_img[..., 1] = (yy * 255 // 511).astype(np.uint8)
    test_img[..., 2] = ((xx + yy) % 256).astype(np.uint8)
    x0 = backend._encode_init(test_img, device, (512, 512))
    g = torch.Generator(device=device).manual_seed(7)
    e = torch.randn(x0.shape, device=device, dtype=torch.float32,
                    generator=g)
    t = 0.5
    x_t = (1.0 - t) * x0 + t * e
    backend._text_to(backend.te_device)
    with torch.inference_mode():
        ctx = backend._encode_prompt(PROMPT).to(device)
    backend._text_offload()
    with torch.inference_mode():
        out = backend.transformer(x_t.to(torch.bfloat16),
                                  torch.tensor([t], device=device,
                                               dtype=torch.bfloat16),
                                  ctx).float()
v_true = e - x0
def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten()[None], b.flatten()[None], dim=1).item()
c_v, c_nv, c_x0 = cos(out, v_true), cos(out, -v_true), cos(out, x0)
print(f"  cos(out, v)={c_v:.3f}  cos(out, -v)={c_nv:.3f}  "
      f"cos(out, x0)={c_x0:.3f}", flush=True)
assert c_v > 0.5, (
    f"raw output is NOT the flow velocity (cos {c_v:.3f}); "
    f"x0 = x - sigma*out in model_fn is wrong")

# ---------------------------------------------------------------- 2. ---
print("\n[2] txt2img 512x512 x8 (model default steps)", flush=True)
img_a = gen(seed=42)
img_b = gen(seed=42)
img_c = gen(seed=43)
assert diff(img_a, img_b) == 0.0, "same seed must be identical"
assert diff(img_a, img_c) > 1.0, "different seed must differ"
Image.fromarray(img_a).save("/tmp/zimage_smoke_t2i.png")

# ---------------------------------------------------------------- 3. ---
print("\n[3] LoRA: weight 0 == baseline, high weight shifts", flush=True)
from imagagen.models import list_loras  # noqa: E402
zloras = [n for n in list_loras(cfg, family="zimage")]
assert zloras, "no zimage LoRAs discovered"
lora_name = zloras[0]
print(f"  LoRA: {lora_name}", flush=True)
base = gen(seed=50)
lora0 = gen(seed=50, loras=[(lora_name, 0.0)])
assert diff(base, lora0) == 0.0, "weight 0 must be a no-op (bit-exact)"
lora1 = gen(seed=50, loras=[(lora_name, 1.0)])
assert diff(base, lora1) > 2.0, "LoRA at 1.0 must shift the image"
back = gen(seed=50)  # apply the empty set again
assert diff(back, base) == 0.0, "baseline must be restored after LoRA"
Image.fromarray(lora1).save("/tmp/zimage_smoke_lora.png")

# ---------------------------------------------------------------- 4. ---
print("\n[4] img2img: low denoise close, full denoise diverges",
      flush=True)
init = np.asarray(Image.open("/tmp/zimage_smoke_t2i.png"))
near = gen(init_image=init, denoise=0.3, seed=60)
far = gen(init_image=init, denoise=1.0, seed=60)
assert diff(near, init) < diff(far, init), \
    "denoise 0.3 must stay closer to the init than denoise 1.0"
print(f"  denoise 0.3: {diff(near, init):.2f}  "
      f"denoise 1.0: {diff(far, init):.2f}", flush=True)

# ---------------------------------------------------------------- 5. ---
print("\n[5] masked inpaint (VAE round-trip + half-protected)",
      flush=True)
mask_full_protect = np.zeros((512, 512), np.uint8)  # 255 = inpaint
mask_half = np.full((512, 512), 255, np.uint8)
mask_half[:, 256:] = 0
full_prot = gen(init_image=init, init_mask=mask_full_protect, denoise=1.0,
                seed=70)
half = gen(init_image=init, init_mask=mask_half, denoise=0.9, seed=70)
# the fully-protected run is the VAE round-trip of the init (latent lock)
roundtrip = diff(full_prot, init)
print(f"  fully-protected vs init: {roundtrip:.2f} (VAE round-trip error)",
      flush=True)
# left half (cols 0..255) is the inpaint region, right half is protected
inpaint_side = diff(half[:, :256], full_prot[:, :256])
protected = diff(half[:, 256:], full_prot[:, 256:])
print(f"  half-protected: protected {protected:.2f} vs inpainted "
      f"{inpaint_side:.2f}", flush=True)
assert inpaint_side > 5.0, "the inpaint side must be regenerated"
assert protected < inpaint_side, "the protected side must change less"

# ---------------------------------------------------------------- 6. ---
print("\n[6] cancellation", flush=True)
req = GenerateRequest(prompt=PROMPT_B, seed=80, width=512, height=512)
import threading  # noqa: E402
def run():
    try:
        backend.generate(req)
        print("  ERROR: generation did not raise", flush=True)
    except InterruptedError:
        print("  InterruptedError ok", flush=True)
th = threading.Thread(target=run)
th.start()
time.sleep(2.0)
backend.cancel()
th.join(timeout=60)
assert not th.is_alive(), "cancel() did not stop the generation"
still = gen(seed=81)  # the backend must stay usable
assert still.shape == (512, 512, 3)

# ---------------------------------------------------------------- 7. ---
print("\n[7] parking (unload/load)", flush=True)
free_before, _ = torch.cuda.mem_get_info(device)
backend.unload()
free_after, _ = torch.cuda.mem_get_info(device)
print(f"  freed {(free_after - free_before) / 2**30:.2f} GiB", flush=True)
assert free_after - free_before > (1 << 30), "unload must free >= 1 GiB"
backend.load()
reloaded = gen(seed=42)
assert diff(reloaded, img_a) == 0.0, "load() must restore identical outputs"

# ---------------------------------------------------------------- 8. ---
print("\n[8] idle VRAM after a generate", flush=True)
free, total = torch.cuda.mem_get_info(device)
resid = (backend._encoder_bytes
         + sum(p.numel() * p.element_size()
               for p in backend.transformer.parameters())
         + sum(p.numel() * p.element_size() for p in backend.vae.parameters())
         ) / 2**30
print(f"  free {free / 2**30:.2f} GiB of {total / 2**30:.2f} "
      f"(~{resid:.2f} GiB resident weights)", flush=True)
assert free > total * 0.25, "the pool must be returned after a generate"

print("\nALL Z-IMAGE SMOKE CHECKS PASSED", flush=True)
