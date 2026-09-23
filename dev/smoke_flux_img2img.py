"""Live Flux img2img/inpaint smoke test (GPU 1).

Mirrors dev/smoke_sdxl_inpaint.py for the flux model family:

1. txt2img baseline (regression: the path is unchanged)
2. img2img: denoise 0.75, no mask -> must differ from the init image
3. low strength: denoise 0.1 -> must stay closer to the init image than [2]
4. "inpaint masked" convention: init_mask all-255 (whole box regenerated),
   denoise 0.6 -> output must differ from the init image
5. mask lock: a FULLY protected run (init_mask all-0) must come back as
   exactly the VAE round-trip of the init image (the per-step and final
   latent lock is model-independent); and in a half-protected run the
   protected side must change less than the inpainted side. (Not bit-exact
   to the init image: the Flux VAE decode has a large receptive field, so
   the protected pixels see the regenerated half -- same as A1111/Forge.)
6a. tone preservation: denoise 0.55 must not shift the scene tone
   (catches an invalid img2img init state -- the init must be the valid
   flow-matching state sigma*noise + (1-sigma)*init, as in Forge's
   PredictionFlux 'const' noise_scaling)
6b. determinism: maskless img2img, same seed twice -> identical image
   (masked inpaint is intentionally not reproducible: the per-step lock
   noise is fresh each step, exactly like the SDXL backend)
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

cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpus=[1])
info = next(m for m in reg.models if m.family == "flux")
print("loading", info.name, "...", flush=True)
t0 = time.time()
backend = reg.backend_for(info.id)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)


def show(frac, msg):
    print(f"  {msg} {frac:.0%}", flush=True)


# 1. baseline txt2img
print("\n[1] txt2img 512x512 x15", flush=True)
req = GenerateRequest(prompt="a bird on a branch, forest", width=512,
                      height=512, steps=15, seed=42)
img0 = backend.generate(req, progress=show).image
Image.fromarray(img0).save("/tmp/smoke_flux_base.png")
print("  saved /tmp/smoke_flux_base.png", flush=True)


def gen(prompt, seed, denoise, mask=None):
    req = GenerateRequest(prompt=prompt, width=512, height=512, steps=15,
                         seed=seed, init_image=img0, init_mask=mask,
                         denoise=denoise)
    return backend.generate(req, progress=show).image


def backend_rt(arr):
    """The backend's own decode(encode(arr)): what a fully latent-locked
    generation of ``arr`` decodes to (the exact target of the lock)."""
    x = backend._encode_init(arr, backend.device, (512, 512))
    with torch.no_grad():
        d = (x / backend.vae.config.scaling_factor
             + backend.vae.config.shift_factor).to(torch.bfloat16)
        out = backend.vae.decode(d, return_dict=False)[0].float()
    return (out / 2 + 0.5).clamp(0, 1).cpu().numpy()[0].transpose(1, 2, 0) * 255


# 2. img2img (the "inpaint not masked" backend contract with no mask)
print("\n[2] img2img denoise 0.75", flush=True)
img2 = gen("a cat on a branch, forest", 43, 0.75)
Image.fromarray(img2).save("/tmp/smoke_flux_i2i.png")
d2 = np.abs(img2.astype(int) - img0.astype(int)).mean()
print(f"  mean abs diff vs init: {d2:.2f} (expect clearly > 0)", flush=True)
assert d2 > 5, f"img2img did not change the image (diff={d2:.2f})"

# 3. low strength: must stay closer to the init image than [2]
print("\n[3] img2img denoise 0.1", flush=True)
img3 = gen("a cat on a branch, forest", 43, 0.1)
d3 = np.abs(img3.astype(int) - img0.astype(int)).mean()
print(f"  mean abs diff vs init: {d3:.2f} (expect < {d2:.2f} from [2])",
      flush=True)
assert d3 < d2, f"low strength is NOT closer to init ({d3:.2f} >= {d2:.2f})"

# 4. whole-box regeneration (the "inpaint masked" backend contract)
print("\n[4] whole-box denoise 0.6 (inpaint-masked convention)", flush=True)
img4 = gen("a cat on a branch, forest", 43, 0.6,
           mask=np.full((512, 512), 255, np.uint8))
d4 = np.abs(img4.astype(int) - img0.astype(int)).mean()
print(f"  mean abs diff vs init: {d4:.2f} (expect clearly > 0)", flush=True)
assert d4 > 5, f"whole-box inpaint did not change the image (diff={d4:.2f})"

# 5a. fully protected (init_mask all-0): every latent is locked to the init
#     latent, so the output must be exactly the VAE round-trip of the init
#     image -- this is the definitive check that the lock is applied at
#     all (a broken lock would return a generation, ~30-40 away)
print("\n[5a] fully protected (mask all 0) -> exact VAE round-trip",
      flush=True)
rt0 = backend_rt(img0)
img5a = gen("a cat on a branch, forest", 44, 0.8, mask=np.zeros((512, 512), np.uint8))
d5a = np.abs(img5a.astype(int) - rt0.astype(int)).mean()
d5a0 = np.abs(img5a.astype(int) - img0.astype(int)).mean()
print(f"  |out - roundtrip| mean abs: {d5a:.2f} (expect ~0)", flush=True)
print(f"  |out - init image|  mean abs: {d5a0:.2f} (VAE fidelity)", flush=True)
assert d5a < 2.0, f"fully-protected run is not the locked round-trip: {d5a:.2f}"

# 5b. protected left half (the "inpaint not masked" backend contract):
#     the protected side must change less than the inpainted side. The
#     protected pixels are latent-locked but not bit-exact the init image:
#     the Flux VAE decode has a large receptive field, so they decode with
#     the regenerated right half mixed in (measured ~10-20 here; A1111/
#     Forge with the same VAE show the same shift)
print("\n[5b] protected left half, denoise 0.8 (inpaint-not-masked convention)",
      flush=True)
user_mask = np.zeros((512, 512), np.uint8)
user_mask[:, :256] = 255  # user masked the left half -> protect it
img5 = gen("a cat on a branch, forest", 44, 0.8, mask=255 - user_mask)
Image.fromarray(img5).save("/tmp/smoke_flux_inpaint_half.png")
left = np.abs(img5[:, :256].astype(int) - img0[:, :256].astype(int)).mean()
right = np.abs(img5[:, 256:].astype(int) - img0[:, 256:].astype(int)).mean()
print(f"  protected left:  mean abs diff {left:.2f} (expect < right)", flush=True)
print(f"  inpainted right: mean abs diff {right:.2f} (expect large)", flush=True)
assert right > 5, f"inpaint region unchanged: {right:.2f}"
assert left < right, (f"protected side changed MORE than the inpaint side "
                      f"({left:.2f} >= {right:.2f}): lock not applied")

# 6a. tone preservation at medium strength: the img2img init state must be
#     the valid flow-matching state sigma*noise + (1-sigma)*init (Forge
#     PredictionFlux, prediction_type='const'). The invalid SDXL-style
#     `init + sigma*noise` state is NOT a valid flow state: the model reads
#     it as the init image amplified by 1/(1-sigma) (~5x at the truncated
#     sigma 0.74 of denoise 0.55/15 steps) and redraws it with a heavy tone
#     shift and over-regeneration (pre-fix on this base: -18/255 brightness
#     and 64% of full-strength diff at denoise 0.55)
print("\n[6a] tone preservation, denoise 0.55", flush=True)
img6a = gen("a cat on a branch, forest", 45, 0.55)
bri = img6a.astype(int).mean() - img0.astype(int).mean()
d6a = np.abs(img6a.astype(int) - img0.astype(int)).mean()
print(f"  brightness shift {bri:+.2f}/255, mean abs diff {d6a:.2f} "
      f"({d6a / d2:.0%} of the [2] denoise 0.75 diff)", flush=True)
assert abs(bri) <= 10.0, (
    f"denoise 0.55 shifted the tone by {bri:+.1f}/255: the init state is "
    f"not the valid flow-matching state (sigma*noise + (1-sigma)*init)")
assert d6a <= 0.75 * d2, (
    f"denoise 0.55 regenerated {d6a / d2:.0%} of the denoise 0.75 diff: "
    f"too strong for half the strength")

# 6b. determinism: maskless img2img is seed-deterministic
print("\n[6b] determinism: maskless img2img, same seed twice", flush=True)
a = gen("a cat on a branch, forest", 99, 0.75)
b = gen("a cat on a branch, forest", 99, 0.75)
da = np.abs(a.astype(int) - b.astype(int)).max()
print(f"  max abs diff: {da} (expect 0)", flush=True)
assert da == 0, f"maskless img2img is not deterministic: max diff {da}"

free, total = torch.cuda.mem_get_info(1)
print(f"\nVRAM: {free // 2**20} MiB free of {total // 2**20} MiB "
      f"(peak allocated {torch.cuda.max_memory_allocated(1) // 2**30} GiB)",
      flush=True)
print("\nALL SMOKE TESTS PASSED", flush=True)
