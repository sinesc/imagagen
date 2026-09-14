"""Server-authoritative canvas: image + mask with undo/redo and brush strokes.

The canvas holds an RGB image (HxWx3 uint8) and, alongside it, a mask
(HxW uint8, 0..255 where >0 marks the inpaint region -- mask alpha is the
inpainting alpha per the spec). Every committed change (stroke end,
generation, mask ops, thumbnail selection, import) pushes a snapshot of
the *previous* (image, mask, png-info) state onto the undo stack, so a
whole stroke is one undo step and the mask is restored with the image.

Strokes arrive as point lists in canvas-pixel coordinates; stamps are
laid out along the polyline at a minimum spacing (1-D Poisson disk), the
standard image-editor behaviour. Brush shapes: a built-in antialiased
circle, or greyscale PNGs from brushes/ whose luminance is the alpha.
"""

from __future__ import annotations

import threading
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

MAX_HISTORY = 200  # ~200 full-res 1024x1024 snapshots (image+mask) in RAM
STAMP_SPACING_FRACTION = 1 / 12  # stamps at least size/12 px apart (>= 1 px)


class Canvas:
    def __init__(self, brushes_dir: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._image: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._extra_info: dict = {}  # PNG text chunks carried across set-from-bytes
        self._undo: list[tuple[np.ndarray | None, np.ndarray | None, dict]] = []
        self._redo: list[tuple[np.ndarray | None, np.ndarray | None, dict]] = []
        self._brushes_dir = brushes_dir
        self._brush_cache: dict[str, np.ndarray] = {}  # name -> float32 L array 0..1

    # --- state ---
    @property
    def image(self) -> np.ndarray | None:
        with self._lock:
            return None if self._image is None else self._image.copy()

    @property
    def mask(self) -> np.ndarray | None:
        """Mask alpha (0..255) or None when there is no canvas image."""
        with self._lock:
            return None if self._mask is None else self._mask.copy()

    def has_image(self) -> bool:
        with self._lock:
            return self._image is not None

    def size(self) -> tuple[int, int] | None:
        """(width, height) or None when empty."""
        with self._lock:
            if self._image is None:
                return None
            return (self._image.shape[1], self._image.shape[0])

    # --- mutations (each records undo) ---
    def _state(self) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """Current (image, mask, extra info); caller must hold the lock."""
        return (
            None if self._image is None else self._image.copy(),
            None if self._mask is None else self._mask.copy(),
            dict(self._extra_info),
        )

    def _commit_locked(
        self,
        image: np.ndarray | None,
        mask: np.ndarray | None,
        extra: dict | None = None,
    ) -> bool:
        """Replace state if it differs from current; record undo. Returns True
        when a snapshot was pushed (i.e. something actually changed)."""
        new = (
            None if image is None else image,
            None if mask is None else mask,
            dict(extra) if extra is not None else dict(self._extra_info),
        )
        cur = (
            None if self._image is None else self._image,
            None if self._mask is None else self._mask,
            self._extra_info,
        )
        same_img = (cur[0] is None and new[0] is None) or (
            cur[0] is not None
            and new[0] is not None
            and cur[0].shape == new[0].shape
            and np.array_equal(cur[0], new[0])
        )
        same_mask = (cur[1] is None and new[1] is None) or (
            cur[1] is not None
            and new[1] is not None
            and np.array_equal(cur[1], new[1])
        )
        if same_img and same_mask and cur[2] == new[2]:
            return False
        self._undo.append(cur)
        if len(self._undo) > MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()
        self._image, self._mask, self._extra_info = new
        return True

    def set_image(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
        info: dict | None = None,
    ) -> None:
        """Replace the whole image (thumb selection, upload, generation).

        A fresh image gets a fresh empty mask of the same shape; pass an
        explicit mask to retain one (Phase 3 inpaint resize does). Pass the
        PNG text chunks of the source image to carry its generation
        metadata (so a later save keeps it); None retains the current info."""
        image = np.ascontiguousarray(image)
        m = (
            mask
            if mask is not None
            else np.zeros(image.shape[:2], dtype=np.uint8)
        )
        with self._lock:
            self._commit_locked(image, m, info)

    def set_image_keep_mask(
        self, image: np.ndarray, info: dict | None = None
    ) -> None:
        """Replace the image, retaining the current mask when its shape
        matches (generation results: the mask is retained across image
        generations unless the user clears it or the size changes). Pass
        PNG text chunks to update the carried metadata; None retains the
        current info."""
        image = np.ascontiguousarray(image)
        with self._lock:
            m = (
                self._mask
                if self._mask is not None and self._mask.shape == image.shape[:2]
                else np.zeros(image.shape[:2], dtype=np.uint8)
            )
            self._commit_locked(image, m, info)

    def set_from_png_bytes(self, data: bytes, mask: np.ndarray | None = None) -> None:
        img = Image.open(BytesIO(data))
        arr = np.asarray(img.convert("RGB"))
        info = {k: v for k, v in (img.info or {}).items() if isinstance(v, str)}
        with self._lock:
            # an explicit mask (e.g. the "inpaint not masked" plan's
            # scaled mask, part of the resulting canvas state) wins; else
            # spec: the mask is retained across image generations (e.g.
            # a thumb selection, or the first-result auto-set on an empty
            # canvas) unless the user clears it -- but a size change
            # invalidates it
            if mask is not None:
                m = mask
            else:
                m = (
                    self._mask
                    if self._mask is not None and self._mask.shape == arr.shape[:2]
                    else np.zeros(arr.shape[:2], dtype=np.uint8)
                )
            self._commit_locked(arr, m, info)

    def clear(self) -> None:
        with self._lock:
            self._commit_locked(None, None)

    def undo(self) -> None:
        with self._lock:
            if not self._undo:
                return
            self._redo.append(self._state())
            self._image, self._mask, self._extra_info = self._undo.pop()

    def redo(self) -> None:
        with self._lock:
            if not self._redo:
                return
            self._undo.append(self._state())
            self._image, self._mask, self._extra_info = self._redo.pop()

    def can_undo(self) -> bool:
        with self._lock:
            return bool(self._undo)

    def can_redo(self) -> bool:
        with self._lock:
            return bool(self._redo)

    # --- brushes ---
    def list_brushes(self) -> list[str]:
        """'circle' plus every greyscale png in brushes/ (by file stem)."""
        names = ["circle"]
        if self._brushes_dir is not None and self._brushes_dir.is_dir():
            for p in sorted(self._brushes_dir.iterdir()):
                if p.suffix.lower() == ".png" and p.is_file():
                    names.append(p.stem)
        return names

    def brush_shape(self, name: str, size: float) -> np.ndarray:
        """Brush alpha shape for a stamp of the given size (px, = largest
        dimension), float32 0..1. 'circle' is built in; anything else is a
        greyscale PNG whose luminance is the alpha."""
        with self._lock:
            if name != "circle" and name not in self._brush_cache:
                p = (self._brushes_dir or Path("/nonexistent")) / f"{name}.png"
                if not p.is_file():
                    raise KeyError(f"unknown brush: {name}")
                img = ImageOps.grayscale(Image.open(p)).convert("L")
                self._brush_cache[name] = np.asarray(img, dtype=np.float32) / 255.0
            base = (
                None if name == "circle" else self._brush_cache.get(name)
            )
        side = max(1, int(round(size)))
        if name == "circle":
            r = size / 2.0
            # half-pixel antialiased disc
            yy, xx = np.mgrid[0:side, 0:side]
            d = np.hypot(xx - (side - 1) / 2.0, yy - (side - 1) / 2.0)
            return np.clip(r + 0.5 - d, 0, 1).astype(np.float32)
        assert base is not None
        w, h = base.shape[1], base.shape[0]
        scale = side / max(w, h)  # preserve aspect, largest dim = size
        img = Image.fromarray((base * 255).astype(np.uint8))
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.LANCZOS,
        )
        return np.asarray(img, dtype=np.float32) / 255.0

    # --- strokes ---
    @staticmethod
    def stamp_positions(
        points: list[tuple[float, float]], spacing: float
    ) -> list[tuple[float, float]]:
        """Evenly spaced stamp centres along a polyline: consecutive stamps
        are at least `spacing` apart (1-D Poisson disk), the last point is
        always stamped (a click = one dot)."""
        if not points:
            return []
        if len(points) == 1:
            return [points[0]]
        out: list[tuple[float, float]] = [points[0]]
        acc = 0.0  # distance since the last stamp
        last = points[0]
        for p in points[1:]:
            seg = np.hypot(p[0] - last[0], p[1] - last[1])
            while seg >= spacing - acc:
                t = (spacing - acc) / seg if seg > 0 else 1.0
                s = (last[0] + t * (p[0] - last[0]), last[1] + t * (p[1] - last[1]))
                out.append(s)
                last = s
                seg = np.hypot(p[0] - last[0], p[1] - last[1])
                acc = 0.0
            acc += seg
            last = p
        return out

    def stroke(
        self,
        points: list[tuple[float, float]],
        size: float,
        opacity: float = 1.0,
        brush: str = "circle",
        erase: bool = False,
        color: tuple[int, int, int] = (255, 255, 255),
        target: str = "image",
    ) -> bool:
        """Paint a stroke onto `target` ("image" or "mask"). Returns True when
        anything changed (i.e. an undo step was recorded)."""
        if target not in ("image", "mask"):
            raise ValueError(f"unknown stroke target: {target}")
        opacity = min(max(opacity, 0.0), 1.0)
        shape = self.brush_shape(brush, size)
        spacing = max(1.0, size * STAMP_SPACING_FRACTION)
        centers = self.stamp_positions(points, spacing)
        if not centers:
            return False

        changed = False
        with self._lock:
            if self._image is None:
                raise ValueError("cannot stroke an empty canvas")
            h, w = self._image.shape[:2]
            sh, sw = shape.shape
            # mask strokes are hard alpha (no soft blending), image strokes
            # use the shape's soft edges * opacity
            a = shape if target == "mask" else shape * opacity
            if target == "mask":
                assert self._mask is not None
                m = self._mask.astype(np.float32)
                for cx, cy in centers:
                    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
                    # clip to canvas
                    sx0, sy0 = max(-x0, 0), max(-y0, 0)
                    ex0, ey0 = min(sw, w - x0), min(sh, h - y0)
                    if ex0 <= sx0 or ey0 <= sy0:
                        continue
                    region = m[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0]
                    patch = (a[sy0:ey0, sx0:ex0] * 255.0) * opacity
                    m[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0] = np.clip(
                        region - patch if erase else region + patch, 0, 255
                    )
                new_mask = m.astype(np.uint8)
                changed = self._commit_locked(self._image, new_mask)
            else:
                img = self._image.astype(np.float32)
                cr, cg, cb = (min(max(c, 0), 255) for c in color)
                for cx, cy in centers:
                    x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
                    sx0, sy0 = max(-x0, 0), max(-y0, 0)
                    ex0, ey0 = min(sw, w - x0), min(sh, h - y0)
                    if ex0 <= sx0 or ey0 <= sy0:
                        continue
                    aa = a[sy0:ey0, sx0:ex0][..., None]
                    region = img[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0]
                    img[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0] = np.clip(
                        region * (1 - aa) + np.array([cr, cg, cb]) * aa, 0, 255
                    )
                new_img = img.astype(np.uint8)
                changed = self._commit_locked(new_img, self._mask)
        return changed

    # --- mask operations ---
    @staticmethod
    def _disk(px: float) -> np.ndarray | None:
        r = int(round(px))
        if r <= 0:
            return None
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        return (xx * xx + yy * yy) <= r * r

    def mask_dilate(self, px: float) -> bool:
        return self._mask_morphology(px, dilate=True)

    def mask_erode(self, px: float) -> bool:
        return self._mask_morphology(px, dilate=False)

    def _mask_morphology(self, px: float, dilate: bool) -> bool:
        from scipy import ndimage

        with self._lock:
            if self._mask is None or not self._mask.any():
                return False
            se = self._disk(px)
            if se is None:
                return False
            # grow/shrink the *grayscale* alpha (max/min over the disk
            # neighbourhood) instead of its binary support: the whole mask —
            # feather included — moves as one, so a dilated soft mask has no
            # low-opacity band between the solid core and the grown ring
            # (the feather just shifts to the outer edge). For a binary mask
            # this equals binary dilation/erosion.
            out = (
                ndimage.maximum_filter(self._mask, footprint=se)
                if dilate
                else ndimage.minimum_filter(self._mask, footprint=se)
            )
            changed = self._commit_locked(self._image, out)
        return changed

    def mask_clear(self) -> bool:
        with self._lock:
            if self._mask is None or not self._mask.any():
                return False
            return self._commit_locked(
                self._image, np.zeros_like(self._mask)
            )

    def set_mask(self, mask: np.ndarray) -> bool:
        """Replace the whole mask (e.g. a segmentation result).

        Records one undo step, like every other mask change. Raises
        ValueError when the canvas is empty or the mask size does not
        match the image.
        """
        mask = np.ascontiguousarray(mask)
        with self._lock:
            if self._image is None:
                raise ValueError("cannot set a mask on an empty canvas")
            if mask.shape != self._image.shape[:2]:
                raise ValueError("mask size does not match the canvas")
            return self._commit_locked(self._image, mask)

    # --- (de)serialization ---
    def to_png_bytes(self) -> bytes | None:
        with self._lock:
            img = self._image
            if img is None:
                return None
            buf = BytesIO()
            extra = self._extra_info
            if extra:
                info = PngImagePlugin.PngInfo()
                for k, v in extra.items():
                    info.add_text(k, v)
                Image.fromarray(img).save(buf, format="PNG", pnginfo=info)
            else:
                Image.fromarray(img).save(buf, format="PNG")
            return buf.getvalue()

    def to_mask_png_bytes(self) -> bytes | None:
        with self._lock:
            if self._mask is None:
                return None
            buf = BytesIO()
            Image.fromarray(self._mask, mode="L").save(buf, format="PNG")
            return buf.getvalue()

    def png_info(self) -> dict:
        with self._lock:
            return dict(self._extra_info)
