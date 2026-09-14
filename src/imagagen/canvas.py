"""Server-authoritative canvas: per-image draw history + a global mask.

Every canvas image is a persistent ``CanvasImage`` entity: a cold PNG store
plus (while it is the displayed image) a hot numpy cache, and its own
*draw* undo/redo stack of pre-stroke PNG snapshots. Image selection is not
an undoable change — selecting a different entity just switches which draw
stack is active, and its history is restored with it.

The mask is a single, image-independent object with its own native size and
a global undo/redo stack. When it is displayed (``/api/mask.png``) or used
for inpainting while the canvas image has a different size, it is scaled to
fit within the image (aspect preserved, centered) — ``fit_ar`` geometry.
Mask strokes arrive in canvas-pixel coordinates and are remapped into the
mask's native space.

History entries are PNG-compressed bytes (not raw numpy): a 1024x1024 entry
is ~100–300 KB instead of ~3.2 MB, so deep stacks stay affordable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from .generation import fit_ar, resize_shift_mask

MASK_HISTORY = 200  # global mask undo/redo depth
DRAW_HISTORY = 50  # per-image draw undo depth
STAMP_SPACING_FRACTION = 1 / 12  # stamps at least size/12 px apart (>= 1 px)


# ------------------------------------------------------------------ helpers --


def _encode_png(img: np.ndarray, info: dict | None = None) -> bytes:
    buf = BytesIO()
    if info:
        pnginfo = PngImagePlugin.PngInfo()
        for k, v in info.items():
            pnginfo.add_text(k, v)
        Image.fromarray(img).save(buf, format="PNG", pnginfo=pnginfo)
    else:
        Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def _decode_png(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"))


def _encode_mask(m: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(m, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _decode_mask(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(data)).convert("L"))


def _png_text_chunks(data: bytes) -> dict:
    img = Image.open(BytesIO(data))
    return {k: v for k, v in (img.info or {}).items() if isinstance(v, str)}


# -------------------------------------------------------------------- entity --


@dataclass
class CanvasImage:
    id: str
    png: bytes  # cold store (always present, in sync with `image` when hot)
    info: dict = field(default_factory=dict)  # PNG text chunks (metadata)
    plan_mask: bytes | None = None  # new mask to apply when set (inpaint-not-masked)
    image: np.ndarray | None = None  # hot cache: only while on the canvas
    undo: list[bytes] = field(default_factory=list)  # pre-stroke PNG snapshots
    redo: list[bytes] = field(default_factory=list)


class Canvas:
    def __init__(self, brushes_dir: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._images: dict[str, CanvasImage] = {}
        self._current: str | None = None
        self._anchored: set[str] = set()  # entity ids referenced by a live job
        self._mask: np.ndarray | None = None  # native size (image-independent)
        self._mask_undo: list[bytes | None] = []
        self._mask_redo: list[bytes | None] = []
        self._upload_counter = 0
        self._brushes_dir = brushes_dir
        self._brush_cache: dict[str, np.ndarray] = {}  # name -> float32 L array 0..1

    # --- state ---
    @property
    def image(self) -> np.ndarray | None:
        with self._lock:
            if self._current is None:
                return None
            img = self._images[self._current].image
            return None if img is None else img.copy()

    @property
    def mask(self) -> np.ndarray | None:
        """Native mask alpha (0..255) or None when there is no mask."""
        with self._lock:
            return None if self._mask is None else self._mask.copy()

    def mask_scaled(self) -> np.ndarray | None:
        """The mask at the current image's size (fit-within, centered), or
        None when there is no mask or no displayed image."""
        with self._lock:
            return self._mask_scaled_locked()

    def _mask_scaled_locked(self) -> np.ndarray | None:
        if self._mask is None or self._current is None:
            return None
        img = self._images[self._current].image
        if img is None:
            return None
        H, W = img.shape[:2]
        mh, mw = self._mask.shape[:2]
        if (mw, mh) == (W, H):
            return self._mask.copy()
        sw, sh, ox, oy = fit_ar(mw, mh, W, H)
        return resize_shift_mask(self._mask, sw, sh, ox, oy, W, H)

    def _mask_fit_params_locked(self) -> tuple[float, int, int] | None:
        """(scale, ox, oy) mapping mask-native -> current image pixels; None
        when there is no displayed image. scale is image-px per native-px."""
        if self._mask is None or self._current is None:
            return None
        img = self._images[self._current].image
        if img is None:
            return None
        H, W = img.shape[:2]
        mh, mw = self._mask.shape[:2]
        sw, sh, ox, oy = fit_ar(mw, mh, W, H)
        return (sw / mw, ox, oy)

    def has_image(self) -> bool:
        with self._lock:
            return self._current is not None

    def size(self) -> tuple[int, int] | None:
        """(width, height) of the displayed image, or None when empty."""
        with self._lock:
            if self._current is None:
                return None
            img = self._images[self._current].image
            if img is None:
                return None
            return (img.shape[1], img.shape[0])

    def current_id(self) -> str | None:
        with self._lock:
            return self._current

    # --- entity management ---
    def add_result(self, job_id: int, index: int, png: bytes, info: dict | None,
                   plan_mask: np.ndarray | None = None) -> str:
        """Register a generation result as a (cold) entity; returns its id.

        ``plan_mask`` is the mask that belongs to this result (inpaint
        not-masked); it is applied when the result is set. A result entity is
        always referenced by its (still live) job, so it is anchored
        immediately and won't be evicted until the job is dropped.
        """
        eid = f"job:{job_id}:{index}"
        with self._lock:
            self._images[eid] = CanvasImage(
                id=eid,
                png=png,
                info=dict(info or {}),
                plan_mask=_encode_mask(plan_mask) if plan_mask is not None else None,
            )
            self._anchored.add(eid)
        return eid

    def import_image(self, data: bytes) -> str:
        """Upload / drag&drop: create an ``upload:{n}`` entity and set it."""
        arr = _decode_png(data)
        info = _png_text_chunks(data)
        with self._lock:
            eid = f"upload:{self._upload_counter}"
            self._upload_counter += 1
            self._images[eid] = CanvasImage(id=eid, png=data, info=info, image=arr)
            self._enter_entity_locked(eid)
        return eid

    def set_current(self, id: str) -> str:
        """Switch the displayed image to entity ``id`` (no history step)."""
        with self._lock:
            if id not in self._images:
                raise ValueError(f"unknown canvas image: {id!r}")
            self._enter_entity_locked(id)
            return id

    def _enter_entity_locked(self, id: str) -> None:
        old = self._current
        if old is not None and old != id and old in self._images:
            self._images[old].image = None  # drop the outgoing hot cache
        ent = self._images[id]
        if ent.image is None:
            ent.image = _decode_png(ent.png)
        self._current = id
        # the mask is image-independent: keep the existing one, only create
        # an (empty, image-sized) mask when there is none yet
        h, w = ent.image.shape[:2]
        if self._mask is None:
            self._mask = np.zeros((h, w), dtype=np.uint8)
        # an "inpaint not masked" result carries the mask that belongs to it
        if ent.plan_mask is not None:
            self._set_mask_locked(_decode_mask(ent.plan_mask))
        self._evict_stale_locked()

    def anchor(self, ids: set[str]) -> None:
        """Record which entity ids are referenced by a live job."""
        with self._lock:
            self._anchored = set(ids)

    def evict_stale(self) -> None:
        with self._lock:
            self._evict_stale_locked()

    def _evict_stale_locked(self) -> None:
        for eid in list(self._images.keys()):
            if eid != self._current and eid not in self._anchored:
                del self._images[eid]

    # --- draw (image) history ---
    def can_undo_draw(self) -> bool:
        with self._lock:
            return bool(self._current and self._images[self._current].undo)

    def can_redo_draw(self) -> bool:
        with self._lock:
            return bool(self._current and self._images[self._current].redo)

    def _undo_draw_locked(self) -> bool:
        if self._current is None:
            return False
        ent = self._images[self._current]
        if not ent.undo:
            return False
        ent.redo.append(ent.png)
        prev = ent.undo.pop()
        ent.image = _decode_png(prev)
        ent.png = prev
        return True

    def _redo_draw_locked(self) -> bool:
        if self._current is None:
            return False
        ent = self._images[self._current]
        if not ent.redo:
            return False
        ent.undo.append(ent.png)
        nxt = ent.redo.pop()
        ent.image = _decode_png(nxt)
        ent.png = nxt
        return True

    # --- mask history ---
    def can_undo_mask(self) -> bool:
        with self._lock:
            return bool(self._mask_undo)

    def can_redo_mask(self) -> bool:
        with self._lock:
            return bool(self._mask_redo)

    def _mask_snapshot_locked(self) -> bytes | None:
        return None if self._mask is None else _encode_mask(self._mask)

    def _undo_mask_locked(self) -> bool:
        if not self._mask_undo:
            return False
        self._mask_redo.append(self._mask_snapshot_locked())
        prev = self._mask_undo.pop()
        self._mask = _decode_mask(prev) if prev is not None else None
        return True

    def _redo_mask_locked(self) -> bool:
        if not self._mask_redo:
            return False
        self._mask_undo.append(self._mask_snapshot_locked())
        nxt = self._mask_redo.pop()
        self._mask = _decode_mask(nxt) if nxt is not None else None
        return True

    def _set_mask_locked(self, mask: np.ndarray) -> bool:
        """Replace the native mask, recording one history step unless it is
        unchanged. The mask may be any size (it is scaled for display/use)."""
        if (
            self._mask is not None
            and mask.shape == self._mask.shape
            and np.array_equal(mask, self._mask)
        ):
            return False
        self._mask_undo.append(self._mask_snapshot_locked())
        if len(self._mask_undo) > MASK_HISTORY:
            self._mask_undo.pop(0)
        self._mask_redo.clear()
        self._mask = np.ascontiguousarray(mask)
        return True

    def set_mask(self, mask: np.ndarray) -> bool:
        with self._lock:
            return self._set_mask_locked(mask)

    def undo(self, target: str = "image") -> bool:
        with self._lock:
            if target == "mask":
                return self._undo_mask_locked()
            if target == "image":
                return self._undo_draw_locked()
            raise ValueError(f"unknown undo target: {target}")

    def redo(self, target: str = "image") -> bool:
        with self._lock:
            if target == "mask":
                return self._redo_mask_locked()
            if target == "image":
                return self._redo_draw_locked()
            raise ValueError(f"unknown redo target: {target}")

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
        """Paint a stroke onto `target` ("image" or "mask"). Returns True
        when anything changed (i.e. a history step was recorded)."""
        if target not in ("image", "mask"):
            raise ValueError(f"unknown stroke target: {target}")
        opacity = min(max(opacity, 0.0), 1.0)
        with self._lock:
            if target == "mask":
                return self._stroke_mask_locked(points, size, opacity, brush, erase)
            return self._stroke_image_locked(points, size, opacity, brush, color)

    def _stroke_image_locked(
        self, points, size, opacity, brush, color
    ) -> bool:
        if self._current is None or self._images[self._current].image is None:
            raise ValueError("cannot stroke an empty canvas")
        ent = self._images[self._current]
        img = ent.image
        shape = self.brush_shape(brush, size)
        spacing = max(1.0, size * STAMP_SPACING_FRACTION)
        centers = self.stamp_positions(points, spacing)
        if not centers:
            return False
        h, w = img.shape[:2]
        sh, sw = shape.shape
        a = shape * opacity
        cr, cg, cb = (min(max(c, 0), 255) for c in color)
        img2 = img.astype(np.float32)
        for cx, cy in centers:
            x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
            sx0, sy0 = max(-x0, 0), max(-y0, 0)
            ex0, ey0 = min(sw, w - x0), min(sh, h - y0)
            if ex0 <= sx0 or ey0 <= sy0:
                continue
            aa = a[sy0:ey0, sx0:ex0][..., None]
            region = img2[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0]
            img2[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0] = np.clip(
                region * (1 - aa) + np.array([cr, cg, cb]) * aa, 0, 255
            )
        new_img = img2.astype(np.uint8)
        if np.array_equal(new_img, img):
            return False
        ent.undo.append(ent.png)
        if len(ent.undo) > DRAW_HISTORY:
            ent.undo.pop(0)
        ent.redo.clear()
        ent.image = new_img
        ent.png = _encode_png(new_img, ent.info)
        return True

    def _stroke_mask_locked(self, points, size, opacity, brush, erase) -> bool:
        if self._current is None or self._images[self._current].image is None:
            raise ValueError("cannot stroke an empty canvas")
        if self._mask is None:
            h, w = self._images[self._current].image.shape[:2]
            self._mask = np.zeros((h, w), dtype=np.uint8)
        fit = self._mask_fit_params_locked()
        if fit is None:
            raise ValueError("cannot stroke an empty canvas")
        scale, ox, oy = fit
        npts = [((x - ox) / scale, (y - oy) / scale) for (x, y) in points]
        return self._stamp_mask(npts, size / scale, opacity, brush, erase)

    def _stamp_mask(self, points, size, opacity, brush, erase) -> bool:
        shape = self.brush_shape(brush, size)
        spacing = max(1.0, size * STAMP_SPACING_FRACTION)
        centers = self.stamp_positions(points, spacing)
        if not centers:
            return False
        # mask strokes are hard alpha (no soft blending); opacity still
        # scales the added/removed alpha
        a = shape
        mh, mw = self._mask.shape[:2]
        sh, sw = shape.shape
        m = self._mask.astype(np.float32)
        for cx, cy in centers:
            x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
            sx0, sy0 = max(-x0, 0), max(-y0, 0)
            ex0, ey0 = min(sw, mw - x0), min(sh, mh - y0)
            if ex0 <= sx0 or ey0 <= sy0:
                continue
            region = m[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0]
            patch = (a[sy0:ey0, sx0:ex0] * 255.0) * opacity
            m[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0] = np.clip(
                region - patch if erase else region + patch, 0, 255
            )
        new_mask = m.astype(np.uint8)
        return self._set_mask_locked(new_mask)

    # --- mask operations (run on the native mask) ---
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
            return self._set_mask_locked(out)

    def mask_clear(self) -> bool:
        with self._lock:
            if self._mask is None or not self._mask.any():
                return False
            return self._set_mask_locked(np.zeros_like(self._mask))

    # --- (de)serialization ---
    def to_png_bytes(self) -> bytes | None:
        with self._lock:
            if self._current is None:
                return None
            return self._images[self._current].png

    def to_mask_png_bytes(self) -> bytes | None:
        """The *scaled* mask (at the displayed image's size) as a greyscale
        PNG, or None when there is no mask or no image."""
        with self._lock:
            m = self._mask_scaled_locked()
            if m is None:
                return None
            return _encode_mask(m)

    def png_info(self) -> dict:
        with self._lock:
            if self._current is None:
                return {}
            return dict(self._images[self._current].info)
