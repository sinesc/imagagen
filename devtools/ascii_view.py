"""Render a PNG as ASCII art + per-region color stats.

Useful in sessions where the agent has no vision: `python devtools/ascii_view.py output/1-foo.png`.

Usage:
    python devtools/ascii_view.py <image.png> [width] [height]
"""

import sys

import numpy as np
from PIL import Image


def ascii_view(path: str, w: int = 72, h: int = 32) -> None:
    a = np.asarray(Image.open(path).convert("RGB").resize((w, h))).astype(float)
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    chars = " .:-=+*#%@"
    idx = (lum / 255 * (len(chars) - 1)).round().astype(int).clip(0, len(chars) - 1)
    for row in idx:
        print("".join(chars[i] for i in row))

    # per-region avg colors: catches "blank corner", "shadow under object", etc.
    H, W, _ = a.shape
    print(f"  top-left  RGB: {a[:H//4, :W//4].mean(axis=(0, 1)).round(0)}   "
          f"top-right RGB: {a[:H//4, 3*W//4:].mean(axis=(0, 1)).round(0)}")
    print(f"  bot-left  RGB: {a[3*H//4:, :W//4].mean(axis=(0, 1)).round(0)}   "
          f"bot-right RGB: {a[3*H//4:, 3*W//4:].mean(axis=(0, 1)).round(0)}")
    print(f"  center    RGB: {a[H//2-10:H//2+10, W//2-10:W//2+10].mean(axis=(0, 1)).round(0)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 72
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    ascii_view(sys.argv[1], w, h)
