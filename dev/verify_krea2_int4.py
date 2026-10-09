"""Live check for the Krea 2 int4 residency path (plan phase 1, 1c).

Loads the real ``krea2_finetuned_int4`` checkpoint through the backend
and reports the measured transformer resident bytes (expect ~7 GiB with
the packed Int4Linear layout vs the ~12.8 GiB of the unpacked int8
path), then renders a small deterministic t2i image (the int4 unpack is
bit-identical to the int8 dequantize, so the render is the same render
the unpacked path produces) and checks determinism + sane VRAM.

Run: .venv/bin/python dev/verify_krea2_int4.py [GPU]  (default GPU 1)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpus=[gpu])

PROMPT = "a shiny red sports car on a wet city street at night"


def resident(backend) -> float:
    return sum(p.numel() * p.element_size()
               for p in backend.transformer.parameters()) / 2**30


def load_model(sub: str):
    info = next(m for m in reg.models
                if m.family == "krea2" and sub in m.id.lower())
    print(f"loading {info.name} ...", flush=True)
    t0 = time.time()
    backend = reg.backend_for(info.id)
    print(f"  loaded in {time.time() - t0:.1f}s, "
          f"transformer resident {resident(backend):.2f} GiB", flush=True)
    return backend


def gen(backend, seed=42):
    r = backend.generate(
        GenerateRequest(prompt=PROMPT, seed=seed, width=512, height=512,
                        steps=8, cfg=1.0),
        progress=lambda f, m: None)
    return r.image


backend = load_model("int4")
assert type(backend.transformer.blocks[0].attn.wq).__name__ == "Int4Linear", \
    "the int4 checkpoint must build Int4Linear block projections"
resid4 = resident(backend)
print(f"int4 resident: {resid4:.2f} GiB (expect ~7, the int8 layout is "
      f"~12.8)", flush=True)
assert 5.0 < resid4 < 9.0, f"unexpected int4 resident size {resid4:.2f} GiB"

free, total = torch.cuda.mem_get_info(backend.device)
print(f"VRAM after load: {free / 2**30:.2f} GiB free of "
      f"{total / 2**30:.2f}", flush=True)

img_a = gen(backend)
img_b = gen(backend)
delta = float(np.abs(img_a.astype(np.float32) - img_b.astype(np.float32)).mean())
print(f"render mean {img_a.mean():.1f}, determinism delta {delta}", flush=True)
assert delta == 0.0, "same seed must render identically"
assert 1.0 < img_a.mean() < 254.0, "the render must not be flat"

free_after, _ = torch.cuda.mem_get_info(backend.device)
print(f"VRAM after generate: {free_after / 2**30:.2f} GiB free", flush=True)

# LoRA on the packed int4 modules: the inherited plain-space delta
# contract must shift the image and leave NO residue when removed
from imagagen.models import list_loras  # noqa: E402

lora = list_loras(cfg, "krea2")[0]
print(f"applying LoRA {lora} ...", flush=True)
img_lora = backend.generate(GenerateRequest(
    prompt=PROMPT, seed=42, width=512, height=512, steps=8, cfg=1.0,
    loras=[(lora, 1.0)]), progress=lambda f, m: None).image
d_shift = float(np.abs(img_lora.astype(np.float32)
                       - img_a.astype(np.float32)).mean())
img_back = gen(backend)
d_back = float(np.abs(img_back.astype(np.float32)
                      - img_a.astype(np.float32)).mean())
print(f"LoRA shift {d_shift:.2f}, baseline restore delta {d_back}",
      flush=True)
assert d_shift > 1.0, "the LoRA must shift the int4 render"
assert d_back == 0.0, "removing the LoRA must restore the baseline exactly"

backend.unload()

# the int8 checkpoint for the side-by-side resident number
backend8 = load_model("int8extended")
resid8 = resident(backend8)
print(f"int8 resident: {resid8:.2f} GiB", flush=True)
img8 = gen(backend8)
print(f"int8 render mean {img8.mean():.1f}", flush=True)
backend8.unload()

print(f"\nOK: int4 {resid4:.2f} GiB vs int8 {resid8:.2f} GiB resident",
      flush=True)
print("ALL KREA 2 INT4 CHECKS PASSED", flush=True)
