"""Generation modes and inpaint geometry.

Three modes (concept.md):

- "generate" (text to image): create a new image at the requested size.
- "inpaint masked": operate on the *aspect-preserving bounding box* of the
  mask plus context padding (``INPAINT_PADDING`` px each side, the A1111/
  Forge convention) -- the box has (approximately) the requested
  width:height ratio and encloses the mask region (it may extend beyond the
  canvas). The padding is there so the model *sees* the real surroundings
  and can match their colour/tone; without it the model invents its own
  background and a visible halo outlines the mask. The model generates at
  the requested image size; the result is downscaled to the box and blended
  into the canvas via the mask alpha (the padding is discarded). Canvas
  size and mask are unchanged.
- "inpaint not masked": scale the canvas AR-preserving so it "touches" the
  inside of the requested size (100x100 -> 250x200 scales to 200x200 and is
  centered at offset 25 right / 0 down), paste it onto a fresh target canvas
  (borders edge-replicated), and inpaint the whole target with the mask
  region protected. The canvas resizes to the requested size and the mask is
  scaled/shifted identically to the image. With no mask and unchanged AR this
  is plain img2img; a larger target is outpainting.

The backend always generates at the request's width x height with
``init_mask`` marking the *inpaint region* (255 = regenerate, the standard
SD/A1111 convention). "inpaint not masked" therefore passes the *inverted*
mask. The plan maps the backend output back into canvas space afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .backends import GenerateRequest


class GenerationError(Exception):
    """User-facing generation setup error (bad mode, empty canvas/mask)."""


# ---------------------------------------------------------------- geometry --


#: context padding (canvas px, each side) added to the inpaint box so the
#: model sees and matches the surroundings (A1111/Forge default is 256).
#: ``with_inpaint`` caps it at the mask's largest extent so tiny masks on
#: small canvases do not get a degenerate box several times the canvas.
INPAINT_PADDING = 256


def masked_bbox(
    mask: np.ndarray, width: int, height: int, padding: int = 0
) -> tuple[int, int, int, int]:
    """Box with the aspect ratio ``width:height`` enclosing the mask region
    (alpha > 0), optionally grown by ``padding`` px on every side (context
    for the model). Returns ``(x0, y0, w, h)`` in canvas pixels; the box is
    centered on the mask and may extend beyond the canvas bounds."""
    support = mask > 0
    if not support.any():
        raise GenerationError("mask is empty")
    rows = np.where(support.any(axis=1))[0]
    cols = np.where(support.any(axis=0))[0]
    ix0, iy0 = int(cols[0]), int(rows[0])
    ix1, iy1 = int(cols[-1] + 1), int(rows[-1] + 1)
    iw, ih = ix1 - ix0, iy1 - iy0
    ar = width / height
    if iw / ih > ar:
        # inner box is wider than the target ratio -> grow the height
        bw, bh = float(iw), iw / ar
    else:
        # inner box is taller (or exact) -> grow the width
        bw, bh = ih * ar, float(ih)
    if padding:
        bw += 2 * padding
        bh += 2 * padding
    cx, cy = (ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0
    x0 = int(round(cx - bw / 2.0))
    y0 = int(round(cy - bh / 2.0))
    return x0, y0, int(round(bw)), int(round(bh))


def fit_ar(w: int, h: int, W: int, H: int) -> tuple[int, int, int, int]:
    """AR-preserving scale of (w,h) so it "touches" the inside of (W,H):
    the largest size that fits, centered. Returns ``(sw, sh, ox, oy)``.

    e.g. (100,100) -> (250,200) gives (200,200,25,0)."""
    scale = min(W / w, H / h)
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    return sw, sh, (W - sw) // 2, (H - sh) // 2


def crop_padded(img: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Crop the box (x0,y0,w,h) from img; out-of-canvas areas are filled by
    replicating the nearest edge pixel (the A1111/Forge inpaint padding)."""
    H, W = img.shape[:2]
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + w, W), min(y0 + h, H)
    region = img[sy0:sy1, sx0:sx1]
    return np.pad(
        region,
        ((sy0 - y0, (y0 + h) - sy1), (sx0 - x0, (x0 + w) - sx1), (0, 0)),
        mode="edge",
    )


def crop_mask(mask: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Crop the box from a mask; out-of-canvas areas are 0 (no mask there)."""
    H, W = mask.shape
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + w, W), min(y0 + h, H)
    out = np.zeros((h, w), dtype=np.uint8)
    out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = mask[sy0:sy1, sx0:sx1]
    return out


def resize_image(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """LANCZOS resize of an HxWx3 uint8 array to (w, h)."""
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.LANCZOS))


def resize_shift_mask(
    mask: np.ndarray, sw: int, sh: int, ox: int, oy: int, W: int, H: int
) -> np.ndarray:
    """Scale a mask to (sw,sh) and place it on a zero (H,W) canvas at
    (ox,oy) -- the same transform the image undergoes in "inpaint not
    masked" (100x100 -> 250x200: scale 2x, shifted 25 right)."""
    m = Image.fromarray(mask, mode="L").resize((sw, sh), Image.LANCZOS)
    out = np.zeros((H, W), dtype=np.uint8)
    out[oy : oy + sh, ox : ox + sw] = np.asarray(m)
    return out


def paste_centered(scaled: np.ndarray, W: int, H: int) -> np.ndarray:
    """Center a scaled image on a fresh (H,W) canvas; the empty border is
    edge-replicated (a low-denoise outpaint then continues plausibly
    instead of showing a black halo)."""
    h, w = scaled.shape[:2]
    ox, oy = (W - w) // 2, (H - h) // 2
    return np.pad(
        scaled, ((oy, H - oy - h), (ox, W - ox - w), (0, 0)), mode="edge"
    )


# -------------------------------------------------------------------- plan --


@dataclass
class InpaintPlan:
    """Captured at job submit time; maps the backend's width x height output
    back to canvas state. The base canvas/mask are snapshotted because the
    canvas stays editable during generation -- per spec the generation
    overwrites in-flight edits (restorable via undo), so it must composite
    against the state it actually saw, not the latest one."""

    mode: str  # "inpaint_masked" | "inpaint_not_masked"
    base_image: np.ndarray  # canvas RGB at submit time
    base_mask: np.ndarray | None
    bbox: tuple[int, int, int, int] | None = None  # inpaint_masked
    new_mask: np.ndarray | None = None  # inpaint_not_masked

    def postprocess(self, gen: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Backend output -> (canvas_image, mask). ``mask`` None = keep the
        current mask when it fits the (possibly new) canvas size."""
        if self.mode == "inpaint_masked":
            assert self.bbox is not None
            x0, y0, w, h = self.bbox
            H, W = self.base_image.shape[:2]
            resized = resize_image(gen, w, h)
            out = self.base_image.copy()
            a = self.base_mask.astype(np.float32) / 255.0
            dx0, dy0 = max(x0, 0), max(y0, 0)
            dx1, dy1 = min(x0 + w, W), min(y0 + h, H)
            if dx1 > dx0 and dy1 > dy0:
                patch = resized[dy0 - y0 : dy1 - y0, dx0 - x0 : dx1 - x0].astype(np.float32)
                aa = a[dy0:dy1, dx0:dx1][..., None]
                reg = out[dy0:dy1, dx0:dx1].astype(np.float32)
                out[dy0:dy1, dx0:dx1] = np.clip(
                    reg * (1 - aa) + patch * aa, 0, 255
                ).astype(np.uint8)
            # mask is retained (possibly user-edited in the meantime)
            return out, None
        # inpaint_not_masked: the generation is the whole new canvas
        assert self.new_mask is not None
        if gen.shape[:2] != self.new_mask.shape[:2]:
            # the backend may round the generation size (SDXL: multiples of
            # 8); the canvas must come out exactly at the requested size
            gen = resize_image(gen, self.new_mask.shape[1], self.new_mask.shape[0])
        return gen, self.new_mask


def with_inpaint(
    request: GenerateRequest,
    mode: str,
    canvas,
    denoise: float,
    padding: int = INPAINT_PADDING,
) -> InpaintPlan | None:
    """Populate ``request.init_image``/``init_mask``/``denoise`` for an
    inpaint mode and return the post-processing plan (None for "generate").

    The mask passed to the backend marks the *inpaint region* (255 =
    regenerate): "inpaint masked" passes the user's mask, "inpaint not
    masked" passes the inverted one (mask content is protected).

    ``padding`` is the context ring around the mask box for "inpaint
    masked" (see ``INPAINT_PADDING``), capped at the mask's largest extent."""
    if mode == "generate":
        return None
    if mode not in ("inpaint_masked", "inpaint_not_masked"):
        raise GenerationError(f"unknown mode {mode!r}")
    request.denoise = denoise
    image = canvas.image
    if image is None:
        raise GenerationError("canvas is empty")
    mask = canvas.mask

    if mode == "inpaint_masked":
        if mask is None or not (mask > 0).any():
            raise GenerationError("mask is empty")
        # context ring, capped so a tiny mask never gets a huge box
        rows = np.where((mask > 0).any(axis=1))[0]
        cols = np.where((mask > 0).any(axis=0))[0]
        extent = int(max(cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1))
        x0, y0, w, h = masked_bbox(
            mask, request.width, request.height,
            padding=min(padding, extent),
        )
        request.init_image = resize_image(
            crop_padded(image, x0, y0, w, h), request.width, request.height
        )
        request.init_mask = resize_image(
            crop_mask(mask, x0, y0, w, h), request.width, request.height
        )
        return InpaintPlan("inpaint_masked", image, mask, bbox=(x0, y0, w, h))

    # inpaint_not_masked
    ch, cw = image.shape[:2]
    sw, sh, ox, oy = fit_ar(cw, ch, request.width, request.height)
    target = paste_centered(resize_image(image, sw, sh), request.width, request.height)
    new_mask = (
        resize_shift_mask(mask, sw, sh, ox, oy, request.width, request.height)
        if mask is not None and mask.any()
        else np.zeros((request.height, request.width), dtype=np.uint8)
    )
    request.init_image = target
    request.init_mask = np.uint8(255) - new_mask  # protect the mask region
    return InpaintPlan("inpaint_not_masked", image, mask, new_mask=new_mask)
