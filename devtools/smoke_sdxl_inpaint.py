"""Live SDXL inpaint smoke test (GPU 1).

1. txt2img baseline
2. "inpaint masked" convention: init_mask all-255 (whole box regenerated),
   denoise 0.6 -> output must differ from the init image
3. "inpaint not masked" convention: user mask protects the LEFT half
   (init_mask = 255 - user) -> left half must stay close to the init
   image, right half must change
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpu=1)
info = reg.models[0]
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
Image.fromarray(img0).save("/tmp/smoke_bird.png")
print("  saved /tmp/smoke_bird.png", flush=True)

# 2. whole-box regeneration (the "inpaint masked" backend contract)
print("\n[2] whole-box denoise 0.6 (inpaint-masked convention)", flush=True)
req2 = GenerateRequest(prompt="a cat on a branch, forest", width=512,
                       height=512, steps=15, seed=43,
                       init_image=img0,
                       init_mask=np.full((512, 512), 255, np.uint8),
                       denoise=0.6)
res2 = backend.generate(req2, progress=show)
img2 = res2.image
Image.fromarray(img2).save("/tmp/smoke_inpaint_full.png")
d2 = np.abs(img2.astype(int) - img0.astype(int)).mean()
print(f"  mean abs diff vs init: {d2:.2f} (expect clearly > 0)", flush=True)
assert d2 > 5, f"whole-box inpaint did not change the image (diff={d2:.2f})"

# 3. protected left half (the "inpaint not masked" backend contract)
print("\n[3] protected left half, denoise 0.8 (inpaint-not-masked convention)",
      flush=True)
user_mask = np.zeros((512, 512), np.uint8)
user_mask[:, :256] = 255  # user masked the left half -> protect it
req3 = GenerateRequest(prompt="a cat on a branch, forest", width=512,
                       height=512, steps=15, seed=44,
                       init_image=img0, init_mask=255 - user_mask,
                       denoise=0.8)
img3 = backend.generate(req3, progress=show).image
Image.fromarray(img3).save("/tmp/smoke_inpaint_half.png")
left = np.abs(img3[:, :256].astype(int) - img0[:, :256].astype(int))
right = np.abs(img3[:, 256:].astype(int) - img0[:, 256:].astype(int))
print(f"  protected left:  mean abs diff {left.mean():.2f} (expect small)",
      flush=True)
print(f"  inpainted right: mean abs diff {right.mean():.2f} (expect large)",
      flush=True)
assert left.mean() < 3, f"protected region drifted: {left.mean():.2f}"
assert right.mean() > 5, f"inpaint region unchanged: {right.mean():.2f}"

print("\nALL SMOKE TESTS PASSED", flush=True)
