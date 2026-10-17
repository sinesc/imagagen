"""Effective-strength / tone check for the Flux img2img path.

Flow matching: sigma IS the noise level. Forge's reference implementation
(PredictionFlux, prediction_type='const') builds the img2img init state as
``sigma*noise + (1-sigma)*latent`` -- the only valid flow state at level
sigma. The state must NOT be `latent + sigma*noise` (not a valid flow
state; the model reads it as the init image amplified by 1/(1-sigma) and
denoises it into a bright, high-contrast, blurry reconstruction).

This probe measures, for a few requested denoises, how far the whole-box
inpaint moves from the init (mean abs diff) and how much the tone shifts
(brightness delta), on a bright scene and a dark nightsky scene (the
"burned halo" case).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpus=[1])
info = next(m for m in reg.models if m.family == "flux")
backend = reg.backend_for(info.id)


def txt2img(prompt, seed):
    req = GenerateRequest(prompt=prompt, width=512, height=512, steps=15,
                          seed=seed)
    return backend.generate(req).image


def run(base, prompt, d, seed=7):
    req = GenerateRequest(
        prompt=prompt, width=512, height=512, steps=15, seed=seed,
        init_image=base, init_mask=np.full((512, 512), 255, np.uint8),
        denoise=d)
    return backend.generate(req).image


bright = np.asarray(Image.open("/tmp/smoke_flux_base.png").convert("RGB"))
bright_prompt = "a bird on a branch, forest"
dark = txt2img("photo of a lone bare tree silhouette under a dark starry "
               "night sky, moonlight, low key", 5)
dark_prompt = "a lone bare tree under a dark starry night sky"
Image.fromarray(dark).save("/tmp/fluxden_dark_base.png")

for name, base, prompt in (("bright", bright, bright_prompt),
                           ("dark", dark, dark_prompt)):
    print(f"\n== {name} scene (init brightness {base.mean():.1f}/255) ==")
    full = run(base, prompt, 0.99)
    ref = np.abs(full.astype(int) - base.astype(int)).mean()
    for d in (0.55, 0.35, 0.1):
        img = run(base, prompt, d)
        dabs = np.abs(img.astype(int) - base.astype(int)).mean()
        dbri = img.astype(int).mean() - base.astype(int).mean()
        Image.fromarray(img).save(f"/tmp/fluxden_{name}_d{d}.png")
        print(f"  denoise {d:.2f}: diff {dabs:.1f} "
              f"({dabs / ref:.0%} of full-strength), "
              f"brightness {dbri:+.1f}/255")
    print(f"  (full-strength reference diff: {ref:.1f})")
