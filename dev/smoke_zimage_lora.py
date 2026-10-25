"""Focused smoke: the export-naming LoRA lora_3784692__zit1 end to end.

- weight 0 == baseline (bit-exact)
- weight 1.0 shifts the image and parses/applies cleanly (pairs + .diff/.diff_b)
- switching back to the baseline leaves no residue (norms + biases restored)
- the other (official-naming) LoRA still works too
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
info = next(m for m in reg.models if m.family == "zimage")
print("loading", info.name, "...", flush=True)
t0 = time.time()
backend = reg.backend_for(info.id)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

loras = list_loras(cfg, family="zimage")
print("loras:", loras, flush=True)
EXPORT = "lora_3784692__zit1"
OFFICIAL = "ZIT-shine"
assert EXPORT in loras and OFFICIAL in loras


def gen(prompt=PROMPT, seed=42, **kw) -> np.ndarray:
    kw.setdefault("steps", -1)
    r = backend.generate(GenerateRequest(prompt=prompt, seed=seed, **kw))
    assert r.image.ndim == 3 and r.image.shape[2] == 3 and r.image.dtype == np.uint8
    return r.image


base = gen(seed=50)
Image.fromarray(base).save("/tmp/zlora_base.png")

# 1. the previously failing LoRA: weight 0 is a no-op ...
l0 = gen(seed=50, loras=[(EXPORT, 0.0)])
assert diff(base, l0) == 0.0, "weight 0 must be a no-op (bit-exact)"
print("weight 0 == baseline ok", flush=True)

# ... and weight 1.0 parses + applies + shifts the image
l1 = gen(seed=50, loras=[(EXPORT, 1.0)])
assert diff(base, l1) > 1.0, "export LoRA at 1.0 must shift the image"
print(f"export LoRA 1.0 shifts by {diff(base, l1):.2f}", flush=True)
Image.fromarray(l1).save("/tmp/zlora_export.png")

# 2. back to the baseline: no residue (deltas + diffs must all be cleared)
back = gen(seed=50)
assert diff(back, base) == 0.0, "baseline must be restored after the LoRA"
print("baseline restored ok", flush=True)

# 3. the official-naming LoRA still works
o1 = gen(seed=50, loras=[(OFFICIAL, 1.0)])
assert diff(base, o1) > 1.0, "official LoRA at 1.0 must shift the image"
print(f"official LoRA 1.0 shifts by {diff(base, o1):.2f}", flush=True)
back2 = gen(seed=50)
assert diff(back2, base) == 0.0, "baseline must be restored after the official LoRA"
print("baseline restored ok", flush=True)

# 4. combined: both LoRAs at once (merges across formats)
c1 = gen(seed=50, loras=[(EXPORT, 0.5), (OFFICIAL, 0.5)])
assert diff(base, c1) > 0.5, "combined LoRAs must shift the image"
print(f"combined LoRAs shift by {diff(base, c1):.2f}", flush=True)
back3 = gen(seed=50)
assert diff(back3, base) == 0.0, "baseline must be restored after the combined set"
print("baseline restored ok", flush=True)

print("\nALL EXPORT-LoRA SMOKE CHECKS PASSED", flush=True)
