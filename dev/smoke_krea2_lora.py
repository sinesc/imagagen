"""Focused smoke: Krea 2 LoRAs end to end (all on-disk formats), run
against the INT4 checkpoint (the packed Int4Linear path).

- ``Real3dKII1.0`` (ComfyUI native naming, ``diffusion_model.`` prefix)
- ``rustic-ambience`` (diffusers naming, ``transformer.`` prefix)
- ``HyperspaceKII1.0`` (ComfyUI flat native naming, ``lora_unet_*``;
  also targets the top-level projections: first/last/tmlp/tproj/txtmlp/
  txtfusion.projector)
- ``BenchesKII1.0`` (ai-toolkit LoKr: ``lokr_w1``/``lokr_w2``, applied
  as the Kronecker nested-GEMM delta)
- ``HighcutKII1.0`` / ``StreamlineKII1.0`` (flat native + ``dora_scale``
  DoRA: the weight-decompose ratio is folded into the delta columns)

For each: weight 1.0 shifts the image, then the baseline is restored
(bit-exact, no residue). 12 steps (the model converges well before the
28-step default).
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
info = next(m for m in reg.models if m.family == "krea2" and "int4" in m.id)
print("loading", info.name, "...", flush=True)
t0 = time.time()
backend = reg.backend_for(info.id)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

loras = list_loras(cfg, family="krea2")
print("loras:", loras, flush=True)
NATIVE = "Real3dKII1.0"
DIFFUSERS = "rustic-ambience"
FLAT = "HyperspaceKII1.0"
LOKR = "BenchesKII1.0"
DORA = "HighcutKII1.0"
DORA2 = "StreamlineKII1.0"
for n in (NATIVE, DIFFUSERS, FLAT, LOKR, DORA, DORA2):
    assert n in loras, n


def gen(prompt=PROMPT, seed=42, **kw) -> np.ndarray:
    kw.setdefault("steps", 12)
    r = backend.generate(GenerateRequest(prompt=prompt, seed=seed, **kw))
    assert r.image.ndim == 3 and r.image.shape[2] == 3 and r.image.dtype == np.uint8
    return r.image


base = gen(seed=50)
Image.fromarray(base).save("/tmp/krlora_base.png")

for name in (NATIVE, DIFFUSERS, FLAT, LOKR, DORA, DORA2):
    l1 = gen(seed=50, loras=[(name, 1.0)])
    d = diff(base, l1)
    assert d > 1.0, f"{name}: LoRA at 1.0 must shift the image"
    print(f"{name}: 1.0 shifts by {d:.2f}", flush=True)
    Image.fromarray(l1).save(f"/tmp/krlora_{name}.png")

# combined: one of each format at once. LoKr + DoRA on ONE module is a
# documented rejection (the DoRA column fold cannot express a Kron
# delta), so the sets mix each special format with the plain LoRAs.
c1 = gen(seed=50, loras=[(NATIVE, 0.5), (DIFFUSERS, 0.5), (FLAT, 0.5),
                         (LOKR, 0.5)])
d = diff(base, c1)
assert d > 0.5, "combined LoRAs must shift the image"
print(f"combined (incl. LoKr) shift by {d:.2f}", flush=True)

c2 = gen(seed=50, loras=[(NATIVE, 0.5), (DIFFUSERS, 0.5), (FLAT, 0.5),
                         (DORA, 0.5), (DORA2, 0.5)])
d = diff(base, c2)
assert d > 0.5, "combined LoRAs must shift the image"
print(f"combined (incl. DoRA) shift by {d:.2f}", flush=True)

# LoKr + DoRA in one set: named rejection, and the backend stays clean
# (the next identical plain set must still work)
try:
    gen(seed=50, loras=[(LOKR, 0.5), (DORA, 0.5)])
    raise AssertionError("LoKr + DoRA combination must be rejected")
except ValueError as e:
    assert "LoKr + DoRA" in str(e), e
    print(f"LoKr + DoRA rejected: {e}", flush=True)

# baseline restored bit-exact (all deltas cleared, no residue)
back = gen(seed=50)
assert diff(back, base) == 0.0, "baseline must be restored after the sets"
print("baseline restored ok", flush=True)

print("\nALL KREA2-LoRA SMOKE CHECKS PASSED", flush=True)
