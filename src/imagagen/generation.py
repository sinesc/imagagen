"""Generation modes and inpaint geometry.

Three modes (concept.md):

- "generate" (text to image): create a new image at the requested size.
- "inpaint masked": operate on the *aspect-preserving bounding box* of the
  mask plus context padding (the UI's "Mask padding" setting, defaulting to
  ``INPAINT_PADDING`` px each side, the A1111/
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

A job may additionally carry a source rotation (``with_inpaint``,
``rotation``): the source is rotated into an upright *working frame* before
generation and the output is counter-rotated afterwards — back to the
source canvas size for "inpaint masked" (the canvas is unchanged in that
mode) and to the requested size otherwise. A "generate" job with a
non-zero rotation therefore carries a plan too; all plan geometry of
rotated jobs is in the working frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .backends import GenerateRequest


class GenerationError(Exception):
    """User-facing generation setup error (bad mode, empty canvas/mask)."""


# ---------------------------------------------------------------- geometry --


#: default context padding (canvas px, each side) added to the inpaint box
#: so the model sees and matches the surroundings (the A1111/Forge default
#: is 256; we use a much smaller default; the UI's "Mask padding" field
#: overrides it per request).
#: ``with_inpaint`` caps it at the mask's largest extent so tiny masks on
#: small canvases do not get a degenerate box several times the canvas.
INPAINT_PADDING = 8

#: the UI's rotation dropdown values (user degrees, positive = clockwise)
ROTATIONS = (-180, -135, -90, -45, 0, 45, 90, 135, 180)


def rotation_size(w: int, h: int, angle: int) -> tuple[int, int]:
    """Bounding-box (width, height) of a w\u00d7h rectangle rotated by `angle`
    (user degrees, positive = clockwise). 0/180: (w, h); ±90: (h, w);
    ±45/±135: both dimensions (w+h)/√2 (rounded up)."""
    a = abs(angle) % 180
    c, s = abs(math.cos(math.radians(a))), abs(math.sin(math.radians(a)))
    # round before ceil: cos(90°) is ~6e-17, which would otherwise push an
    # exact half-turn up by 1 px
    return max(1, math.ceil(round(w * c + h * s, 9))), \
        max(1, math.ceil(round(w * s + h * c, 9)))


def rotate_canvas(arr: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an HxW[x3] uint8 array by `angle` (user degrees, positive =
    clockwise), expanding to the rotated rectangle's bounding box.

    0/±90/180 are exact pixel permutations; ±45/±135 are bicubic-resampled and
    the uncovered corners (which only exist for ±45/±135) are filled by
    replicating the nearest edge pixel of `arr` (images) or zeroed (masks), so
    image and mask share the same pixel grid while the mask never invents
    coverage outside the original canvas."""
    angle = angle % 360
    if angle == 0:
        return arr.copy()
    if angle == 180:
        return np.flip(arr, axis=(0, 1))
    if angle in (90, 270):
        return np.rot90(arr, k=-1 if angle == 90 else 1)
    h, w = arr.shape[:2]
    cw, ch = rotation_size(w, h, angle)
    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)
    # The corners of the rotated rectangle reach past the original edges by at
    # most max(w, h)·|sin 2θ|/2 (= max(w,h)/2 at 45°). Rotate an edge-padded
    # oversize copy so every output pixel samples (edge-replicated) source.
    pad = int(math.ceil(max(w, h) * abs(math.sin(2 * th)) / 2)) + 2
    base = np.pad(
        arr, ((pad, pad), (pad, pad)) + ((0, 0),) * (arr.ndim - 2),
        mode="constant" if arr.ndim == 2 else "edge")
    bh, bw = base.shape[:2]
    cx, cy = (cw - 1) / 2.0, (ch - 1) / 2.0
    # Output pixel (x, y) -> source pixel: the working frame is the source
    # rotated by `angle` clockwise, so the inverse mapping rotates back by
    # -angle around the shared center.
    m = (
        c, s, -c * cx - s * cy + (bw - 1) / 2.0,
        -s, c, s * cx - c * cy + (bh - 1) / 2.0,
    )
    return np.asarray(Image.fromarray(base).transform(
        (cw, ch), Image.AFFINE, m, resample=Image.BICUBIC, fillcolor=0))


def unrotate(img: np.ndarray, angle: int, w: int, h: int) -> np.ndarray:
    """Inverse of :func:`rotate_canvas` for a known final size: counter-rotate
    by -`angle` and center-crop to (w, h). (A ±1 px centring error is possible
    for the ±45/±135 family — the crop is centred on the rotation centre.)"""
    if angle % 360 == 0:
        if img.shape[:2] != (h, w):
            return resize_image(img, w, h)
        return img
    out = rotate_canvas(img, -angle)
    oh, ow = out.shape[:2]
    x0, y0 = (ow - w) // 2, (oh - h) // 2
    return out[y0 : y0 + h, x0 : x0 + w].copy()


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
    back to canvas state. For a non-zero rotation all geometry (bbox, masks)
    is in the *rotated working frame*: the source canvas is rotated by
    `rotation` before generation and the output is counter-rotated back to
    the source canvas size for "inpaint masked" (canvas size is unchanged
    in that mode — the requested W×H only sizes the generation) and to
    `requested_size` otherwise. (The rest is unchanged: the canvas stays
    editable during generation — the generation overwrites in-flight edits,
    restorable via undo.)"""

    mode: str  # "generate" | "inpaint_masked" | "inpaint_not_masked"
    requested_size: tuple[int, int]  # the user's W×H (final canvas size)
    rotation: int = 0  # user angle, degrees, positive = clockwise
    # inpaint_masked, working frame
    bbox: tuple[int, int, int, int] | None = None
    work_image: np.ndarray | None = None  # rotated canvas (the blend target)
    # inpaint_masked: rotated mask (blend alpha); inpaint_not_masked:
    # rotated target mask (resize reference)
    work_mask: np.ndarray | None = None
    # inpaint_masked: the unrotated source canvas (w, h) — the final canvas
    # size (the requested W×H only sizes the generation in this mode)
    canvas_size: tuple[int, int] | None = None

    def postprocess(self, gen: np.ndarray) -> np.ndarray:
        """Backend output -> canvas image (the inpaint_masked blend, or the
        resized whole-canvas result), counter-rotated to the requested size
        for rotated jobs. The canvas mask is owned by the result entity
        (inherited from the source image)."""
        if self.mode == "inpaint_masked":
            assert self.bbox is not None and self.work_image is not None
            x0, y0, w, h = self.bbox
            resized = resize_image(gen, w, h)
            out = self.work_image.copy()
            H, W = out.shape[:2]
            a = self.work_mask.astype(np.float32) / 255.0
            dx0, dy0 = max(x0, 0), max(y0, 0)
            dx1, dy1 = min(x0 + w, W), min(y0 + h, H)
            if dx1 > dx0 and dy1 > dy0:
                patch = resized[dy0 - y0 : dy1 - y0, dx0 - x0 : dx1 - x0].astype(np.float32)
                aa = a[dy0:dy1, dx0:dx1][..., None]
                reg = out[dy0:dy1, dx0:dx1].astype(np.float32)
                out[dy0:dy1, dx0:dx1] = np.clip(
                    reg * (1 - aa) + patch * aa, 0, 255).astype(np.uint8)
        elif self.mode == "inpaint_not_masked":
            assert self.work_mask is not None
            if gen.shape[:2] != self.work_mask.shape[:2]:
                # the backend may round the generation size (SDXL: multiples
                # of 8); the canvas must come out exactly at the requested size
                gen = resize_image(gen, self.work_mask.shape[1],
                                   self.work_mask.shape[0])
            out = gen
        else:  # "generate": the raw output is the (possibly rotated) result
            out = gen
        if self.rotation:
            if self.mode == "inpaint_masked":
                # the final canvas is the source canvas (unchanged in this
                # mode), not the user's W×H — counter-rotating to the
                # requested size would center-crop a small area of the image
                fw, fh = self.canvas_size
            else:
                fw, fh = self.requested_size
            out = unrotate(out, self.rotation, fw, fh)
        return out


def with_inpaint(
    request: GenerateRequest,
    mode: str,
    canvas,
    denoise: float,
    rotation: int = 0,
    padding: int = INPAINT_PADDING,
) -> InpaintPlan | None:
    """Populate ``request.init_image``/``init_mask``/``denoise`` and return
    the post-processing plan (None only for a rotation-0 "generate").

    The mask passed to the backend marks the *inpaint region* (255 =
    regenerate): "inpaint masked" passes the user's mask, "inpaint not
    masked" passes the inverted one (mask content is protected).

    ``rotation`` (degrees, positive = clockwise, one of ``ROTATIONS``) rotates
    the source into an upright working frame before generation; the output is
    counter-rotated back afterwards. The user's W×H always refers to the
    *display* frame (what is on screen), so the generation size in the working
    frame is ``rotation_size(W, H, rotation)`` in every mode: for generate and
    inpaint_not_masked that also grows the frame so the rotated result fits,
    for inpaint_masked it rotates the crop's aspect ratio so it stays aligned
    with the rotated mask (an unrotated AR around a rotated mask wastes most
    of the frame on context). ``requested_size`` always keeps the user's W×H
    (it is what the PNG metadata records, alongside ``rotation``).

    ``padding`` is the context ring around the mask box for "inpaint
    masked" (see ``INPAINT_PADDING``), capped at the mask's largest extent."""
    if rotation not in ROTATIONS:
        raise GenerationError(f"invalid rotation {rotation!r}")
    if mode == "generate":
        if rotation == 0:
            return None
        fw, fh = request.width, request.height
        request.width, request.height = rotation_size(fw, fh, rotation)
        return InpaintPlan("generate", requested_size=(fw, fh),
                           rotation=rotation)
    if mode not in ("inpaint_masked", "inpaint_not_masked"):
        raise GenerationError(f"unknown mode {mode!r}")
    request.denoise = denoise
    image = canvas.image
    if image is None:
        raise GenerationError("canvas is empty")
    # the displayed image's own mask (at the canvas image's size)
    mask = canvas.mask
    work_image = rotate_canvas(image, rotation) if rotation else image
    work_mask = rotate_canvas(mask, rotation) if rotation else mask

    if mode == "inpaint_masked":
        if not (work_mask > 0).any():
            raise GenerationError("mask is empty")
        # context ring, capped at the rotated mask's largest extent
        rows = np.where((work_mask > 0).any(axis=1))[0]
        cols = np.where((work_mask > 0).any(axis=0))[0]
        extent = int(max(cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1))
        # the generation size is the user's W×H rotated into the working
        # frame, so the box's aspect ratio matches the rotated mask; the plan
        # keeps the user's W×H (the metadata records it alongside `rotation`,
        # so an import must not see an already-rotated size)
        fw, fh = request.width, request.height
        gw, gh = rotation_size(fw, fh, rotation)
        x0, y0, w, h = masked_bbox(work_mask, gw, gh,
                                   padding=min(padding, extent))
        request.width, request.height = gw, gh
        request.init_image = resize_image(
            crop_padded(work_image, x0, y0, w, h), gw, gh)
        request.init_mask = resize_image(
            crop_mask(work_mask, x0, y0, w, h), gw, gh)
        return InpaintPlan("inpaint_masked",
                           requested_size=(fw, fh),
                           rotation=rotation, bbox=(x0, y0, w, h),
                           work_image=work_image, work_mask=work_mask,
                           canvas_size=(image.shape[1], image.shape[0]))

    # inpaint_not_masked
    ch, cw = image.shape[:2]
    sw, sh, ox, oy = fit_ar(cw, ch, request.width, request.height)
    target = paste_centered(
        resize_image(image, sw, sh), request.width, request.height)
    new_mask = (
        resize_shift_mask(mask, sw, sh, ox, oy, request.width, request.height)
        if mask.any()
        else np.zeros((request.height, request.width), dtype=np.uint8)
    )
    fw, fh = request.width, request.height
    if rotation:
        # rotate the *whole target* (source + borders), so the source content
        # is upright in the working frame while the outpaint borders and the
        # discarded 45° corners stay edge-replicated
        request.width, request.height = rotation_size(fw, fh, rotation)
        request.init_image = rotate_canvas(target, rotation)
        work_mask = rotate_canvas(new_mask, rotation)
    else:
        request.init_image = target
        work_mask = new_mask
    request.init_mask = np.uint8(255) - work_mask  # protect the mask region
    return InpaintPlan("inpaint_not_masked", requested_size=(fw, fh),
                       rotation=rotation, work_mask=work_mask)
