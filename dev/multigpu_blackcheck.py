"""Fresh-process multi-GPU black-image check (requires >= 2 free real GPUs).

Mimics the queue's worst case on a cold server: the very first batch of 2
images is dispatched one per GPU, so BOTH backends are constructed
concurrently and both generations run concurrently. A poison that hits the
first image and persists would show up here.

After the concurrent first batch, 2 more sequential images (one per GPU)
are generated to see whether a poisoned process stays poisoned.

Checks each image for NaN and for a collapsed (black/flat) render, saves
the PNGs, prints PASS/FAIL.

Usage (from the project root):
    .venv/bin/python dev/multigpu_blackcheck.py [outdir]
"""

import sys
import threading
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

import numpy as np
from PIL import Image

from imagagen.backends import GenerateRequest
from imagagen.config import load_config
from imagagen.models import ModelRegistry

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/blackcheck")
OUT.mkdir(parents=True, exist_ok=True)


def check(tag: str, arr: np.ndarray) -> bool:
    """Return True if the image looks healthy; save it for inspection."""
    f = OUT / f"{tag}.png"
    try:
        img = arr.astype(np.float64)
        nan = bool(np.isnan(img).any())
        mean = float(img.mean())
        # structure check (same heuristic as dev/gen.py): downscale to
        # 16x16 and compare its std to the raw std -- a flat image has
        # downscaled std ~ 0 while a real image keeps most of its std
        from PIL import Image as _I
        small = np.asarray(_I.fromarray(arr).resize((16, 16)))
        flat = small.std() < 3.0
    except Exception as e:  # noqa: BLE001 - report, don't crash the harness
        print(f"  {tag}: CHECK ERROR {e!r}")
        return False
    ok = (not nan) and (not flat) and 5.0 < mean < 250.0
    print(f"  {tag}: ok={ok} nan={nan} mean={mean:.1f} small16_std={small.std():.1f}"
          if not nan else f"  {tag}: ok=False NaN in image! mean={np.nanmean(img):.1f}")
    Image.fromarray(arr).save(f)
    return ok


def main() -> int:
    t0 = time.time()
    reg = ModelRegistry(load_config())
    gpus = reg.gpus
    assert len(gpus) >= 2, f"need >= 2 usable GPUs, got {gpus}"
    model_id = reg.models[0].id
    print(f"registry ready in {time.time()-t0:.1f}s: gpus={gpus} "
          f"primary={reg.primary} model={model_id}")

    req = lambda seed: GenerateRequest(  # noqa: E731
        prompt="a red cube on a wooden table",
        width=512, height=512, steps=20, cfg=6, seed=seed)

    results: dict[str, tuple[object, ModelBackend | None]] = {}
    errors: list[str] = []

    def worker(tag: str, gpu: int, seed: int):
        try:
            res, backend = reg.run(model_id, gpu, req(seed))
            results[tag] = (res, backend)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{tag} (gpu {gpu}): {e!r}")

    # --- phase 1: the cold first batch, one image per GPU, concurrent -----
    t1 = time.time()
    threads = [
        threading.Thread(target=worker, args=(f"p1-gpu{g}", g, 100 + g))
        for g in gpus[:2]
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=900)
    print(f"phase 1 (concurrent cold load + generate) took {time.time()-t1:.1f}s")
    for tag in sorted(results):
        res, backend = results[tag]
        vae_src = type(backend.pipe.vae).__name__
        sf = backend.pipe.vae.config.scaling_factor
        print(f"  {tag}: vae={vae_src} scaling_factor={sf} "
              f"dtype={backend.vae_dtype}")
        if not check(tag, res.image):
            print("FAIL (phase 1 black/NaN image)")
            return 1

    # --- phase 2: 2 more sequential images, one per GPU --------------------
    for g in gpus[:2]:
        worker(f"p2-gpu{g}", g, 200 + g)
    for tag in sorted(k for k in results if k.startswith("p2")):
        if not check(tag, results[tag][0].image):
            print("FAIL (phase 2 black/NaN image)")
            return 1

    if errors:
        print("FAIL errors:", errors)
        return 1
    print("PASS: all images healthy in this fresh process")
    return 0


if __name__ == "__main__":
    sys.exit(main())
