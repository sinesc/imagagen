"""Smoke: prompt-change prefetch (prefetch-text-encoder) on a swapping family.

Runs against a Krea 2 checkpoint on GPU 1. The INT8 checkpoint's tower and
transformer never co-reside, so prepare_text_encode parks the transformer
to the CPU (the interesting swap layout); pass a different id substring
(e.g. ``int4``, where both DO fit) to check the no-swap layout instead.
Exercises every layout a generation can meet:

  1. baseline generation (normal in-generate encode swap)
  2. prefetch of a NEW prompt  -> "prepared": text tower on GPU
     (transformer parked only when the fit check says so)
  3. generation of that prompt -> encode runs on the resident tower, the
     encode path restores the transformer, sampling succeeds
  4. prefetch of an ALREADY ENCODED prompt -> "cached" (no moves)
  5. prefetch of a new prompt (prepared) + generation of a DIFFERENT,
     cached prompt -> the cache-hit path must restore the transformer
  6. prefetch while the model is fully parked (registry.unload_all) ->
     "not-resident" (lazy loading stays lazy)

Images must never be uniform (the classic symptom of a broken device
layout); after every generation the transformer is asserted GPU-resident.

Usage: .venv/bin/python dev/smoke_parking_prefetch.py [ckpt-id-substring]
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402


def gen(reg, model_id, prompt: str) -> np.ndarray:
    img, _ = reg.run(model_id, 1, GenerateRequest(
        prompt=prompt, width=512, height=512, steps=8, seed=7))
    arr = img.image
    if arr.std() < 2.0:
        raise SystemExit(f"BROKEN: uniform image for prompt {prompt!r}")
    return arr


cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpus=[1])
SUBSTR = sys.argv[1] if len(sys.argv) > 1 else "Int8"
info = next(m for m in reg.models
            if m.family == "krea2" and SUBSTR.lower() in m.id.lower())
print("loading", info.name, "...", flush=True)
t0 = time.time()
backend = reg.backend_for(info.id)
print(f"loaded in {time.time() - t0:.1f}s", flush=True)


def devices(tag: str) -> str:
    te = next(backend.te.parameters()).device
    tr = next(backend.transformer.parameters()).device
    print(f"  [{tag}] text-encoder={te} transformer={tr}", flush=True)
    return te, tr


def is_gpu(d) -> bool:
    return str(d) == "cuda:1"


# 1. baseline generation (normal swap path; leaves everything CPU-side
#    for the tower after the encode)
t0 = time.time()
gen(reg, info.id, "a shiny red sports car on a wet city street at night")
print(f"gen1 (baseline) {time.time() - t0:.1f}s", flush=True)
devices("after gen1")

# 2. prefetch a NEW prompt -> prepared layout: the tower is on the GPU;
#    the transformer is parked only when the fit check says it must be
#    (INT8: yes; INT4: both co-reside and nothing is parked)
action = reg.prefetch_text_encoder(
    info.id, "a golden retriever riding a bicycle", "")
print("prefetch(new) ->", action, flush=True)
assert action == "prepared", action
te, tr = devices("after prefetch(new)")
assert is_gpu(te), "prefetch did not load the text tower"
swapped_for_text = not is_gpu(tr)

# 3. generate the SAME prompt: encode on the resident tower, the encode
#    path must bring the transformer back before sampling
t0 = time.time()
gen(reg, info.id, "a golden retriever riding a bicycle")
print(f"gen2 (prefetched prompt) {time.time() - t0:.1f}s", flush=True)
te, tr = devices("after gen2")
assert is_gpu(tr), "transformer not restored after encode"
if swapped_for_text:
    print("  (encode-on-resident-tower restore path exercised)", flush=True)

# 4. prefetch an ENCODED prompt -> cached, nothing moves
action = reg.prefetch_text_encoder(
    info.id, "a shiny red sports car on a wet city street at night", "")
print("prefetch(cached) ->", action, flush=True)
assert action == "cached", action
devices("after prefetch(cached)")

# 5. prepared state + cache-HIT generation (the hit path must restore the
#    transformer if the prefetch parked it for a different prompt)
action = reg.prefetch_text_encoder(info.id, "a lighthouse at dawn", "")
assert action == "prepared", action
te, tr = devices("after prefetch(3rd)")
assert is_gpu(te), "prefetch did not load the text tower"
if is_gpu(tr):
    print("  (co-resident layout: cache-hit restore is a no-op)", flush=True)
t0 = time.time()
gen(reg, info.id, "a golden retriever riding a bicycle")  # encoded in gen2
print(f"gen3 (cache hit, prepared layout) {time.time() - t0:.1f}s",
      flush=True)
te, tr = devices("after gen3")
assert is_gpu(tr), "cache-hit path did not keep the transformer on the GPU"

# 6. a fully parked model is never woken up for a prompt alone
reg.unload_all()
action = reg.prefetch_text_encoder(info.id, "a yet another new prompt", "")
print("prefetch(parked) ->", action, flush=True)
assert action == "not-resident", action

print("PREFETCH SMOKE OK", flush=True)
