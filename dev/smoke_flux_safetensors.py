"""Live Flux .safetensors (Kijai FP8 + bnb NF4) smoke test (GPU).

Loads each of the four ``models/flux/*.safetensors`` checkpoints through
the registry (alongside the GGUF one, which keeps working) and:

  1. asserts ``supports_distilled`` (dev files = True, schnell = False)
  2. generates one 512x512 / 10-step baseline image (fixed seed/prompt)
  3. applies the flux LoRA and regenerates (the runtime deltas must be
     active on every QuantLinear format)
  4. reports peak VRAM (the per-op dequantize keeps the model quantized on
     the GPU: ~6-7 GB resident, not ~24 GB bf16)
  5. saves PNGs to ``dev/smoke_flux_safetensors/`` for inspection

Cross-checks:
  - ``flux1DevNF4V2`` vs ``flux1DevFP8110GBKijai``: same seed/prompt, two
    quantizations of the same model -> small differences, same scene
  - the schnell runs exercise the nested (absmax-blocksize-256) NF4
    dequantize path on CUDA (``flux1SchnellNF4``) and the plain fp8 path

Usage (from the project root):
    .venv/bin/python dev/smoke_flux_safetensors.py [gpu-index]
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry, list_loras  # noqa: E402

LORA = "MetalSuitFlux1.0"
PROMPT = "a bird on a branch, forest"
SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "smoke_flux_safetensors"

# model id -> expected supports_distilled (dev files carry guidance_in,
# schnell files don't)
MODELS = {
    "flux1DevFP8110GBKijai": True,
    "flux1SchnellFP811GBKijai": False,
    "flux1DevNF4V2": True,
    "flux1SchnellNF4": False,
}


def diff(a, b) -> float:
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


def gen(backend, loras=()):
    req = GenerateRequest(prompt=PROMPT, width=512, height=512, steps=10,
                         seed=SEED, loras=list(loras))
    t0 = time.time()
    img = backend.generate(req).image
    dt = time.time() - t0
    return img, dt


def main(gpu: int | None) -> None:
    # T5/CLIP live on the CPU (encode on CPU, embeddings to the GPU): the
    # FP8 checkpoints are ~12 GB of weights, which will not share a 16 GB
    # card with the ~5 GB T5-XXL on the GPU.
    cfg = replace(load_config(ROOT), text_encoder_device="cpu")
    found = list_loras(cfg, family="flux")
    assert LORA in found, f"expected {LORA!r} in loras/flux/, got {found!r}"

    reg = ModelRegistry(cfg, gpus=[gpu])
    OUT_DIR.mkdir(exist_ok=True)
    baselines: dict[str, np.ndarray] = {}

    for name, distilled in MODELS.items():
        info = next(m for m in reg.models if m.id == name)
        print(f"\nloading {info.name} ...", flush=True)
        t0 = time.time()
        backend = reg.backend_for(info.id)
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
        assert backend.supports_distilled is distilled, (
            f"{name}: supports_distilled={backend.supports_distilled}, "
            f"expected {distilled}")

        dev = backend.device
        # clean per-model VRAM: sync + reset the peak counter right after
        # the load, so `resident` is the live size and `peak` is the
        # generation high-water mark (not the previous model's)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        resident = torch.cuda.memory_allocated(dev) / 2**30

        print(f"  baseline 512x512 x10 (seed {SEED})", flush=True)
        img, base_dt = gen(backend)
        Image.fromarray(img).save(OUT_DIR / f"{name}.png")
        assert not np.isnan(img.astype(float)).any(), f"{name}: NaN in baseline"
        assert img.std() > 10, f"{name}: baseline looks uniform (std={img.std():.1f})"

        print(f"  {LORA} @ 1.0 (expect visibly different)", flush=True)
        lora_img, lora_dt = gen(backend, loras=[(LORA, 1.0)])
        Image.fromarray(lora_img).save(OUT_DIR / f"{name}.lora.png")
        d = diff(lora_img, img)
        print(f"  LoRA mean abs diff vs baseline: {d:.2f} "
              f"({lora_dt / base_dt:.2f}x baseline time)", flush=True)
        assert d > 1, f"{name}: LoRA had no visible effect (diff={d:.2f})"

        torch.cuda.synchronize(dev)
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
        print(f"  VRAM: resident {resident:.2f} GiB, peak {peak:.2f} GiB "
              f"({base_dt:.1f}s / 10 steps)", flush=True)
        baselines[name] = img

    # same seed/prompt on the same (dev) model in two quantizations:
    # different numbers, same scene
    d = diff(baselines["flux1DevNF4V2"], baselines["flux1DevFP8110GBKijai"])
    print(f"\nflux1DevNF4V2 vs flux1DevFP8110GBKijai (same seed): "
          f"mean abs diff {d:.2f}", flush=True)
    assert 0 < d < 30, (
        f"same-seed NF4 vs FP8 baseline diverged (diff={d:.2f}): "
        f"expected small differences, same scene")

    print("\nALL SMOKE TESTS PASSED", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
