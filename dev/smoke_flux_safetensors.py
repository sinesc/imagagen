"""Live Flux .safetensors (Kijai FP8 + bnb NF4) smoke test (GPU).

Loads each of the four ``models/flux/*.safetensors`` checkpoints through
the registry (alongside the GGUF one, which keeps working) and:

  1. asserts ``supports_distilled`` (dev files = True, schnell = False)
  2. generates one 512x512 / 10-step baseline image (fixed seed/prompt)
  3. applies the flux LoRA and regenerates (the runtime deltas must be
     active on every QuantLinear format)
  4. generates one 1024x1280 / 20-step image with the schnell FP8
     checkpoint (the large-size case from status.txt: the FP8 transformer
     is ~11.1 GiB and does not share the 16 GB card with the T5-XXL text
     encoder, so this exercises the load-time CPU parking and the
     per-generation transformer swap)
  5. reports peak VRAM (the per-op dequantize keeps the model quantized on
     the GPU: ~7 GB for NF4, ~11 GB for FP8, not ~24 GB bf16)
  6. saves PNGs to ``dev/smoke_flux_safetensors/`` for inspection

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


def gen(backend, loras=(), width=512, height=512, steps=10):
    req = GenerateRequest(prompt=PROMPT, width=width, height=height,
                         steps=steps, seed=SEED, loras=list(loras))
    t0 = time.time()
    img = backend.generate(req).image
    dt = time.time() - t0
    return img, dt


def main(gpu: int | None) -> None:
    # The text encoders run on the GPU (the default text_encoder_device):
    # for the NF4 checkpoints they co-reside with the transformer, for the
    # FP8 ones the backend parks them on the CPU and swaps the transformer
    # out while they run (Forge's unet/encoder swap).
    cfg = load_config(ROOT)
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

        if name == "flux1SchnellFP811GBKijai":
            # the large-size case: transformer swap + untiled decode at
            # 1.3 MP next to the 11 GiB FP8 transformer
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            print(f"  big 1024x1280 x20 (seed {SEED})", flush=True)
            big_img, big_dt = gen(backend, width=1024, height=1280, steps=20)
            Image.fromarray(big_img).save(OUT_DIR / f"{name}_1024x1280.png")
            assert big_img.std() > 10, "big run looks uniform"
            peak_big = torch.cuda.max_memory_allocated(dev) / 2**30
            print(f"  big VRAM peak: {peak_big:.2f} GiB ({big_dt:.1f}s / 20 steps)",
                  flush=True)

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
