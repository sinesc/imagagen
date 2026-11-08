"""Check that an inpaint region touching a canvas edge is fully inpainted.

The mask -> latent quantization decides how much init content stays locked per
latent cell: the mask is cropped to the inpaint box, resized to the generation
size, and the backends sample it at the latent grid (one sample per 8-px cell).
When the mask's outer boundary lands in the middle of a latent cell that cell
stays (partially) locked, so the outermost rows/columns of the inpaint region
keep the original content -- visible as "unchanged lines" at the mask edge,
worst where the mask touches the canvas edge (the inpaint region's boundary
then coincides with the canvas edge) and growing with the box->generation
downscale factor and the padding ring.

Metric: per canvas line (row for top/bottom, column for left/right) inside the
masked strip, the mean absolute difference to the original. Lines that were
inpainted differ a lot (denoise 0.75 regenerates the content); lines that kept
the init content differ ~0. The summary prints the mean diff of the 8 lines at
the outer mask edge vs the interior baseline.

usage: check_inpaint_edge_lock.py [family] [--edge bottom|top|left|right]
       [--steps N] [--denoise F] [--size WxH] [--padding N] [--out FILE]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.generation import with_inpaint  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

SRC = ROOT / "dev/references/purple-flower.png"
PROMPT = "a field of purple wildflowers, soft evening light, highly detailed"


class _Canvas:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask


def _mask_for_edge(edge: str, H: int, W: int) -> np.ndarray:
    """A 300x220 rect flush with the requested canvas edge."""
    mask = np.zeros((H, W), dtype=np.uint8)
    img = Image.fromarray(mask)
    if edge == "bottom":
        box = [240, H - 220, 240 + 300 - 1, H - 1]
    elif edge == "top":
        box = [240, 0, 240 + 300 - 1, 220 - 1]
    elif edge == "left":
        box = [0, 240, 220 - 1, 240 + 300 - 1]
    elif edge == "right":
        box = [W - 220, 240, W - 1, 240 + 300 - 1]
    else:
        raise SystemExit(f"bad edge {edge!r}")
    ImageDraw.Draw(img).rectangle(box, fill=255)
    return np.asarray(img)


def main() -> None:
    family, edge, out_path = "sdxl", "bottom", "/tmp/inpaint_edge.png"
    steps, denoise, pad = 12, 0.75, 8
    size = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--steps":
            steps = int(args[i + 1])
        elif a == "--denoise":
            denoise = float(args[i + 1])
        elif a == "--padding":
            pad = int(args[i + 1])
        elif a == "--out":
            out_path = args[i + 1]
        elif a == "--edge":
            edge = args[i + 1]
        elif a == "--size":
            w, h = args[i + 1].split("x")
            size = (int(w), int(h))
        else:
            family = a
        i += 2 if a.startswith("--") else 1

    original = np.asarray(Image.open(SRC).convert("RGB"))
    H, W = original.shape[:2]
    mask = _mask_for_edge(edge, H, W)
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    cfg = load_config(ROOT)
    reg = ModelRegistry(cfg, gpus=[1])
    info = next(m for m in reg.models if m.family == family)
    print(f"loading {info.name} ({family}) ...", flush=True)
    backend = reg.backend_for(info.id)

    req = GenerateRequest(
        prompt=PROMPT, negative_prompt="",
        width=(size[0] if size else -1), height=(size[1] if size else -1),
        steps=steps, cfg=7.0, seed=1234,
    )
    plan = with_inpaint(req, "inpaint_masked", _Canvas(original, mask),
                        denoise, padding=pad)
    print(f"edge {edge}  gen {req.width}x{req.height}  bbox {plan.bbox}  "
          f"mask box {(x0, y0, x1 - x0 + 1, y1 - y0 + 1)}", flush=True)
    res = backend.generate(req, progress=lambda f, s: None)
    out = plan.postprocess(res.image)
    Image.fromarray(out).save(out_path)

    vertical = edge in ("bottom", "top")
    # lines along the mask's outer edge, and the interior baseline
    if vertical:
        span = slice(x0 + 40, x1 - 40)
        outer = list(range(y1 - 7, y1 + 1)) if edge == "bottom" \
            else list(range(y0, y0 + 8))
        inner = list(range(y0 + 60, y1 - 60))
    else:
        span = slice(y0 + 40, y1 - 40)
        outer = list(range(x1 - 7, x1 + 1)) if edge == "right" \
            else list(range(x0, x0 + 8))
        inner = list(range(x0 + 60, x1 - 60))

    def line_diff(k: int) -> float:
        a = original[k, span] if vertical else original[span, k]
        b = out[k, span] if vertical else out[span, k]
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())

    print("\n line   meanabs(orig, out)")
    for k in outer:
        d = line_diff(k)
        print(f"{k:5d}  {d:8.2f}{'  <-- mostly unchanged' if d < 20 else ''}")
    baseline = float(np.mean([line_diff(k) for k in inner]))
    outer_mean = float(np.mean([line_diff(k) for k in outer]))
    print(f"\n{edge} edge: outer-8 mean abs diff {outer_mean:.2f} "
          f"(interior baseline {baseline:.2f}, ratio "
          f"{outer_mean / max(baseline, 1e-6):.2f})")
    print(f"saved {out_path}  (mask {(x1 - x0 + 1)}x{(y1 - y0 + 1)}, "
          f"gen {req.width}x{req.height}, padding {pad}, denoise {denoise}, "
          f"steps {steps})")


if __name__ == "__main__":
    main()
