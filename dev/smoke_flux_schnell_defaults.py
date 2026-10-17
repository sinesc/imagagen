"""Smoke: schnell checkpoint with the "model default" sentinel (steps/cfg
= -1, distilled checked) -- the path that produced oversaturated neon
images before the backend-level sentinel resolution.

Expected: the schnell class (no guidance embedding) demotes distilled to
classic CFG and resolves the built-ins to 4 steps / cfg 1.0; the result
carries the effective values; the image is a normal photo, not neon
cartoon.

Usage (from the project root, with the server NOT running):
    .venv/bin/python dev/smoke_flux_schnell_defaults.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from imagagen.backends import GenerateRequest
from imagagen.config import load_config
from imagagen.models import ModelRegistry

MODEL = "flux1SchnellNF4"
SEED = 42
OUT = Path("/tmp/flux_schnell_default.png")


def main() -> int:
    reg = ModelRegistry(load_config())
    model = next(m for m in reg.models if m.id == MODEL)
    backend = reg.backend_for(model.id)
    assert isinstance(backend, type(reg.backend_for(model.id)))
    assert not backend.supports_distilled, "expected a schnell-class checkpoint"

    t0 = time.time()
    res = backend.generate(GenerateRequest(
        prompt="professional photo of a young couple playing with their "
               "dog in the park",
        width=1280, height=1024, steps=-1, cfg=-1.0, seed=SEED,
        distilled=True),
        progress=lambda f, s: print(f"  {s}", flush=True))
    dt = time.time() - t0

    a = res.image.astype(float)
    small = np.array(Image.fromarray(res.image).resize((16, 16))).astype(float)
    Image.fromarray(res.image).save(OUT)
    print(f"model:      {model.id}")
    print(f"time:       {dt:.1f}s")
    print(f"effective:  steps={res.steps} cfg={res.cfg} distilled={res.distilled}")
    print(f"std:        {a.std():.1f}  structure(16x16): {small.std():.1f}")
    print(f"saturation: mean max-min per pixel { (a.max(axis=2) - a.min(axis=2)).mean():.1f} / 255")
    print(f"saved:      {OUT}  (seed {res.seed})")
    assert res.steps == 4, f"expected 4 built-in steps, got {res.steps}"
    assert res.cfg == 1.0, f"expected built-in cfg 1.0, got {res.cfg}"
    assert res.distilled is False, "schnell class must run classic CFG"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
