"""Soft inpainting GPU A/B check (real SDXL, GPU 1).

Runs the same masked inpaint twice and compares the behaviour:

A  standard inpaint      -- protected content is hard-locked at every step
B  soft inpaint (defaults)
                            -- protected content is soft-blended with the
                               inpainted content per mask opacity, then a
                               difference-based pixel composite restores it

Expectation (same seed/prompt/settings otherwise):
  * the inpaint core changes in both
  * the protected core: A is the VAE round-trip of the base (small but
    nonzero diff), B is the EXACT base (the pixel composite restores the
    original pixels wherever the latents did not change)  <-- the feature
  * the transition band (blurred mask edge): A carries the VAE round-trip
    of the partially locked content, B is a content-aware blend (original
    kept where the generation resembles it, generated where it differs),
    so B should sit at or below A's diff with no crisp seam

Saves individual PNGs plus a side-by-side composite with an amplified
diff map of (B - base) restricted to the protected side.

Usage:  .venv3.13/bin/python devtools/soft_inpaint_check.py
(No other server/model should be holding GPU 1.)
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402
from imagagen.soft_inpaint import SoftInpainting  # noqa: E402

SIZE = 768
SEED = 42
STEPS = 20
PROMPT = "a small red bird perched on a mossy branch, forest background"
NEGATIVE = "blurry, low quality"


def blurred_circle_mask(size: int, radius: int, blur: int) -> np.ndarray:
    """User mask (inpaint region = white): a circle with soft edges.

    The soft edges are the point -- hard-edged masks give the soft path
    nothing to blend.
    """
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).ellipse(
        [size // 2 - radius, size // 2 - radius,
         size // 2 + radius, size // 2 + radius], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(blur))
    return np.asarray(m, dtype=np.uint8)


def mask_fraction(mask: np.ndarray) -> float:
    # .sum() on uint8 wraps (uint8 accumulation); values are 0..255
    return float(mask.astype(np.int64).sum()) / 255.0 / mask.size


def region_stats(out: np.ndarray, base: np.ndarray, mask: np.ndarray):
    """Mean abs diff vs base in the inpaint core / transition / protected
    core (mask thresholds chosen inside the blur ramp)."""
    d = np.abs(out.astype(np.int32) - base.astype(np.int32)).mean(axis=2)
    core = mask > 200
    prot = mask < 55
    trans = ~(core | prot)
    return (d[core].mean(), d[trans].mean(), d[prot].mean())


def show(frac, msg):
    print(f"  {msg} {frac:.0%}", flush=True)


def main() -> None:
    cfg = load_config(ROOT)
    reg = ModelRegistry(cfg, gpu=1)
    info = reg.models[0]
    print("loading", info.name, "...", flush=True)
    t0 = time.time()
    backend = reg.backend_for(info.id)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    # 0. base image
    print(f"\n[0] base txt2img {SIZE}x{SIZE} x{STEPS}", flush=True)
    req = GenerateRequest(prompt=PROMPT, negative_prompt=NEGATIVE, width=SIZE,
                          height=SIZE, steps=STEPS, seed=SEED)
    base = backend.generate(req, progress=show).image
    Image.fromarray(base).save("/tmp/soft_base.png")

    # soft-edged mask: inpaint the centre, protect (most of) the rest
    mask = blurred_circle_mask(SIZE, radius=SIZE // 4, blur=SIZE // 16)
    Image.fromarray(mask, mode="L").save("/tmp/soft_mask.png")
    print(f"  mask: {mask_fraction(mask):.0%} inpaint, "
          f"blurred edges (sigma={SIZE // 16})", flush=True)

    runs = {
        "A_hard": None,
        "B_soft_defaults": SoftInpainting(),
    }
    # a "softened" variant: gentle blend biased towards the inpainted
    # content (strong preservation strength, low schedule bias)
    runs["C_soft_inpaint_bias"] = SoftInpainting(
        mask_blend_power=0.5, mask_blend_scale=1.5,
        composite_mask_influence=0.5)

    out_dir = ROOT / "output" / "soft_inpaint"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, soft in runs.items():
        tag = "soft" if soft is not None else "hard"
        print(f"\n[{name}] {tag} inpaint, same seed", flush=True)
        req = GenerateRequest(prompt=PROMPT, negative_prompt=NEGATIVE,
                              width=SIZE, height=SIZE, steps=STEPS,
                              seed=SEED, init_image=base, init_mask=mask,
                              denoise=0.75, soft_inpainting=soft)
        img = backend.generate(req, progress=show).image
        p = out_dir / f"{name}.png"
        Image.fromarray(img).save(p)
        results[name] = img

    # ---- metrics ----------------------------------------------------------
    print("\nmean abs diff vs base (inpaint core / transition / protected):")
    for name, img in results.items():
        c, t, pr = region_stats(img, base, mask)
        print(f"  {name:22s} {c:6.2f} / {t:6.2f} / {pr:6.2f}")
    a = region_stats(results["A_hard"], base, mask)
    b = region_stats(results["B_soft_defaults"], base, mask)
    assert b[0] > 5, f"soft run did not inpaint the core: {b[0]:.2f}"
    assert b[2] < 1, f"soft run did not restore the protected core: {b[2]:.2f}"
    assert b[2] < a[2], (
        f"soft run ({b[2]:.2f}) should preserve the protected core at least "
        f"as well as the hard run ({a[2]:.2f})")
    print("\n  A protected core (VAE round-trip baseline): "
          f"{a[2]:.3f}")
    print("  B protected core (composite restore, expect ~0): "
          f"{b[2]:.3f}")
    print("  A transition band: "
          f"{a[1]:.3f}")
    print("  B transition band (content-aware, expect <= A): "
          f"{b[1]:.3f}")
    if b[1] > a[1] + 1.0:
        print("  WARNING: soft run's transition band drifts further from "
              "the base than the hard run (check the composite)")

    # ---- composite ---------------------------------------------------------
    h = SIZE // 2
    panels = [Image.fromarray(base).resize((h, h)),
              * (Image.fromarray(img).resize((h, h))
                 for img in results.values())]
    # amplified diff of B in the protected + transition area
    diff = (np.abs(results["B_soft_defaults"].astype(np.int32) -
                   base.astype(np.int32))).mean(axis=2)
    diff = (np.clip(diff * 8, 0, 255)).astype(np.uint8)
    panels.append(Image.fromarray(diff, mode="L").resize((h, h)).convert("RGB"))
    labels = ["base", "A_hard", "B_soft_defaults", "C_soft_inpaint_bias",
              "diff B-base x8"]
    comp = Image.new("RGB", (h * len(panels), h + 24))
    draw = ImageDraw.Draw(comp)
    for i, (label, im) in enumerate(zip(labels, panels, strict=True)):
        comp.paste(im, (i * h, 24))
        draw.text((i * h + 8, 6), label, fill=(255, 255, 0))
    cpath = out_dir / "composite.png"
    comp.save(cpath)
    print(f"\nsaved to {out_dir}\ncomposite: {cpath}")
    print("expected: B/C look like A in the centre, but the protected "
          "content (and the blurred mask edge) is pixel-identical to the "
          "base in B/C (composite restore) while A shows the VAE "
          "round-trip texture there; the diff panel should be ~black on "
          "the protected side")


if __name__ == "__main__":
    main()
