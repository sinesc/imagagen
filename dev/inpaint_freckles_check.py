"""Reproduce the Forge reference inpaint (dev/references/inpainting/).

Runs "inpaint masked" on original(forge).png with the exact settings from
forge-inpaint-normal.png's metadata (12 steps, Euler a, Polyexponential,
CFG 1.5, seed 2384249041, DMD2fp16XL1.0 @ 0.7, denoise 0.55, padding 64,
generation size 1280x1440 over the 1024x1280 source) and prints face-region
detail metrics for: original, forge reference, and the reproduction.

The freckles are high-frequency detail: a healthy inpaint regenerates them
(high Laplacian variance, close to the forge reference); a too-weakly
noised inpaint region smooths them away (low variance, "plastic" face).

usage: inpaint_freckles_check.py [output.png]
"""
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.generation import with_inpaint  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

REF = ROOT / "dev/references/inpainting"
PROMPT = (
    "cinematic film still woman at the beach, clear sky, "
    "beautiful turquoise ocean . shallow depth of field, vignette, "
    "highly detailed, high budget Hollywood movie, bokeh, cinemascope, "
    "moody, epic, gorgeous, film grain, grainy"
)
# face region for the detail metrics (source-image coordinates)
FACE = (slice(300, 620), slice(470, 700))


def _face_grey(a: np.ndarray) -> np.ndarray:
    y, x = FACE
    return np.asarray(
        Image.fromarray(a).convert("L").crop((x.start, y.start, x.stop, y.stop)),
        dtype=np.float64,
    )


def lap_var(a: np.ndarray) -> float:
    """Laplacian variance of the (face-region) grey image: texture detail."""
    g = _face_grey(a)
    k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
    sw = sliding_window_view(g, (3, 3))
    lap = (sw * k).sum(-2).sum(-1)
    return float(lap.var())


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(_face_grey(a) - _face_grey(b)).mean())


def main() -> None:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inpaint_repro.png")

    original = np.asarray(
        Image.open(REF / "original(forge).png").convert("RGB"))
    H, W = original.shape[:2]
    forge = np.asarray(
        Image.open(REF / "forge-inpaint-normal.png").convert("RGB"))

    # the user's face mask, approximated as an ellipse over face+neck
    mask_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask_img).ellipse([440, 275, 715, 645], fill=255)
    mask = np.asarray(mask_img)

    class Canvas:
        image = original

        def mask_scaled(self):
            return mask

    cfg = load_config(ROOT)
    reg = ModelRegistry(cfg, gpus=[1])
    info = reg.models[0]
    print("loading", info.name, "...", flush=True)
    backend = reg.backend_for(info.id)

    req = GenerateRequest(
        prompt=PROMPT,
        negative_prompt="",
        width=1280, height=1440,
        steps=12, cfg=1.5, seed=2384249041,
        sampler="Euler a", scheduler="Polyexponential",
        loras=[("DMD2fp16XL1.0", 0.7)],
    )
    plan = with_inpaint(req, "inpaint_masked", Canvas(), 0.55, padding=64)
    print("bbox:", plan.bbox, "init", req.init_image.shape, flush=True)
    res = backend.generate(req, progress=lambda f, s: print(
        f"  {s}", flush=True))
    out, _ = plan.postprocess(res.image)
    Image.fromarray(out).save(out_path)
    print("saved", out_path, flush=True)

    for name, img in (("original", original), ("forge", forge),
                      ("repro", out)):
        print(f"  {name:9s} lap_var {lap_var(img):8.1f}   "
              f"mad-vs-original {mad(img, original):6.2f}")


if __name__ == "__main__":
    main()
