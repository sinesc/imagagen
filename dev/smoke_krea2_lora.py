"""Focused smoke: Krea 2 LoRAs end to end (both on-disk formats).

- ``Real3dKII1.0`` (ComfyUI native naming, ``diffusion_model.`` prefix)
- ``rustic-ambience`` (diffusers naming, ``transformer.`` prefix)

For each: weight 0 == baseline (bit-exact), weight 1.0 shifts the image,
baseline restored afterwards (no residue).
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
from imagagen.models import ModelRegistry, list_loras  # noqa: E402

PROMPT = "a shiny red sports car on a wet city street at night"


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpus=[1])
info = next(m for m in reg.models if m.family == "krea2")
print("loading", info.name, "...", flush=True)
t0 = time.time()
backend = reg.backend_for(info.id)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

loras = list_loras(cfg, family="krea2")
print("loras:", loras, flush=True)
NATIVE = "Real3dKII1.0"
DIFFUSERS = "rustic-ambience"
assert NATIVE in loras and DIFFUSERS in loras


def gen(prompt=PROMPT, seed=42, **kw) -> np.ndarray:
    kw.setdefault("steps", -1)
    r = backend.generate(GenerateRequest(prompt=prompt, seed=seed, **kw))
    assert r.image.ndim == 3 and r.image.shape[2] == 3 and r.image.dtype == np.uint8
    return r.image


base = gen(seed=50)
Image.fromarray(base).save("/tmp/krlora_base.png")

for name in (NATIVE, DIFFUSERS):
    l0 = gen(seed=50, loras=[(name, 0.0)])
    assert diff(base, l0) == 0.0, f"{name}: weight 0 must be a no-op"
    print(f"{name}: weight 0 == baseline ok", flush=True)

    l1 = gen(seed=50, loras=[(name, 1.0)])
    d = diff(base, l1)
    assert d > 1.0, f"{name}: LoRA at 1.0 must shift the image"
    print(f"{name}: 1.0 shifts by {d:.2f}", flush=True)
    Image.fromarray(l1).save(f"/tmp/krlora_{name}.png")

    back = gen(seed=50)
    assert diff(back, base) == 0.0, f"{name}: baseline must be restored"
    print(f"{name}: baseline restored ok", flush=True)

# combined: one of each format at once
c1 = gen(seed=50, loras=[(NATIVE, 0.5), (DIFFUSERS, 0.5)])
d = diff(base, c1)
assert d > 0.5, "combined LoRAs must shift the image"
print(f"combined LoRAs shift by {d:.2f}", flush=True)
back2 = gen(seed=50)
assert diff(back2, base) == 0.0, "baseline must be restored after the combined set"
print("baseline restored ok", flush=True)

print("\nALL KREA2-LoRA SMOKE CHECKS PASSED", flush=True)
