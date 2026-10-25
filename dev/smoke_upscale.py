"""GPU smoke test for image upscaling (imagagen/upscalers.py).

Runs the installed models/upscalers/ models on the GPU with the most free
VRAM and checks forge's sizing rule, timings and peak VRAM:

- RealESRGAN_x4plus 512x512 -> 2048 with tile 192/8 (the dialog default)
  and with tile 0 (untiled); x2 and x2.5 pass counts follow the forge
  loop (dest int(w*s)//8//8*8, max 3 passes, final Lanczos fit);
- SwinIR_4x 256 -> 1024 untiled (spandrel reports tiling DISCOURAGED);
- ScuNET (1x restoration model): exactly ONE model pass + Lanczos fit.

Writes /tmp/smoke_upscale_in.png (the 512 source) and
/tmp/smoke_upscale_x4.png (the tiled x4 result) for visual evaluation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from imagagen.config import load_config  # noqa: E402
from imagagen.gpus import gpu_order  # noqa: E402
from imagagen.upscalers import UpscalerManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _source_512() -> np.ndarray:
    """A real generated photo from output/, centre-cropped to 512x512."""
    sample = next((ROOT / "output").glob("*.png"), None)
    if sample is None:
        rng = np.random.default_rng(7)
        arr = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
        return arr
    img = Image.open(sample).convert("RGB")
    s = min(img.size)
    left = (img.width - s) // 2
    top = (img.height - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((512, 512),
                                                          Image.LANCZOS)
    return np.asarray(img)


def main() -> None:
    import torch

    config = load_config(ROOT)
    gpus = gpu_order(config.primary_gpu, config.ignore_gpus)
    if not gpus:
        print("SKIP: no usable GPU")
        return
    gpu = gpus[0]
    device = f"cuda:{gpu}"
    print(f"device: {device} ({torch.cuda.get_device_name(gpu)})")
    m = UpscalerManager(config.upscalers_dir, device=device)

    def gpu_peak(fn):
        torch.cuda.reset_peak_memory_stats(gpu)
        torch.cuda.empty_cache()
        t0 = time.time()
        out = fn()
        secs = time.time() - t0
        peak = torch.cuda.max_memory_allocated(gpu) / 2**30
        return out, secs, peak

    src = _source_512()
    Image.fromarray(src).save("/tmp/smoke_upscale_in.png")

    x4_tile, s1, p1 = gpu_peak(lambda: m.run(
        src, "RealESRGAN/RealESRGAN_x4plus", 4.0, tile=192, tile_overlap=8))
    print(f"realesrgan x4 tiled 512->2048: shape={x4_tile.shape} "
          f"{s1:.2f}s peak={p1:.2f}GiB")
    assert x4_tile.shape[:2] == (2048, 2048)
    Image.fromarray(x4_tile).save("/tmp/smoke_upscale_x4.png")

    x4_full, s2, p2 = gpu_peak(lambda: m.run(
        src, "RealESRGAN/RealESRGAN_x4plus", 4.0, tile=0))
    print(f"realesrgan x4 untiled 512->2048: shape={x4_full.shape} "
          f"{s2:.2f}s peak={p2:.2f}GiB")
    assert x4_full.shape[:2] == (2048, 2048)

    # forge sizing: x2 (one model pass to 2048, Lanczos fit to 1024) and
    # x2.5 (2048 -> 1280); verify via dims only, on a small crop
    small = src[:64, :64]
    o2 = m.run(small, "RealESRGAN/RealESRGAN_x4plus", 2.0, tile=0)
    o25 = m.run(small, "RealESRGAN/RealESRGAN_x4plus", 2.5, tile=0)
    print(f"x2 64->{o2.shape[0]}, x2.5 64->{o25.shape[0]}")
    assert o2.shape[:2] == (128, 128) and o25.shape[:2] == (160, 160)

    # scale 1 = one enhancement pass + exact refit to the //8*8 size
    o1 = m.run(small, "RealESRGAN/RealESRGAN_x4plus", 1.0, tile=0)
    assert o1.shape[:2] == (64, 64)
    print("scale 1: one pass + exact 64x64 refit ok")

    # SwinIR (spandrel tiling DISCOURAGED): untiled 256 -> 1024
    swin, s3, p3 = gpu_peak(lambda: m.run(
        src[:256, :256], "SwinIR_4x", 4.0, tile=0))
    print(f"swinir x4 untiled 256->1024: shape={swin.shape} "
          f"{s3:.2f}s peak={p3:.2f}GiB")
    assert swin.shape[:2] == (1024, 1024)

    # 1x restoration model: exactly one pass, then the Lanczos fit
    sc, s4, p4 = gpu_peak(lambda: m.run(
        src[:256, :256], "ScuNET", 2.0, tile=0))
    print(f"scunet x2 256->{sc.shape[0]} (1 pass + fit): {s4:.2f}s "
          f"peak={p4:.2f}GiB")
    assert sc.shape[:2] == (512, 512)

    print("OK")


if __name__ == "__main__":
    main()
