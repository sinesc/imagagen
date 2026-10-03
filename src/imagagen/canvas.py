"""Server-side canvas store: per-image entities (the client is the editor).

The browser owns the live canvas (strokes, masks, undo/redo, morphology
are all computed client-side) and pushes the full state to the server via
``replace_state`` (``POST /api/canvas/sync``) before generation, saving,
cropping, segmentation and when idle. The server only persists what it is
shown: every canvas image is a ``CanvasImage`` entity — a cold PNG store
plus (while it is the displayed image) a hot numpy cache — and its own
mask at the image's own size (always present; zero when nothing is
masked).

A new entity (generation result, upload) inherits the *current* mask of
the image it derives from — the result's source image (the green-bordered
source thumbnail), for uploads the currently displayed image. The mask is
copied at creation and transformed to the new image's size (aspect fit
within, centered — the same mapping the image itself receives). A new
entity without a source (empty canvas) starts with a fresh empty mask.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from .generation import fit_ar, resize_shift_mask

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
    im = Image.open(BytesIO(data))
    if "A" in im.getbands():  # client value canvas: alpha = value
        im = im.getchannel("A")
    return np.asarray(im.convert("L"))


def _png_text_chunks(data: bytes) -> dict:
    img = Image.open(BytesIO(data))
    # text chunks after the image data (foreign tools, e.g. Forge crops) are
    # only parsed on load, or they'd be silently dropped from this info
    try:
        img.load()
    except Exception:
        pass
    return {k: v for k, v in (img.info or {}).items() if isinstance(v, str)}


def _transform_mask(mask: np.ndarray, W: int, H: int) -> np.ndarray:
    """mask (any size) -> (H, W) uint8: aspect fit within, centered — the
    same mapping the image itself receives in "inpaint not masked".
    Copied unchanged when the size already matches."""
    h, w = mask.shape[:2]
    if (w, h) == (W, H):
        return mask.copy()
    sw, sh, ox, oy = fit_ar(w, h, W, H)
    return resize_shift_mask(mask, sw, sh, ox, oy, W, H)


# -------------------------------------------------------------------- entity --


@dataclass
class CanvasImage:
    id: str
    png: bytes  # cold store (always present, in sync with `image` when hot)
    # Per-image mask: at this image's own size (always present; zero when
    # nothing is masked).
    mask: np.ndarray
    info: dict = field(default_factory=dict)  # PNG text chunks (metadata)
    image: np.ndarray | None = None  # hot cache: only while on the canvas


class Canvas:
    def __init__(
        self,
        brushes_dir: Path | None = None,
        default_brushes_dir: Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._images: dict[str, CanvasImage] = {}
        self._current: str | None = None
        self._anchored: set[str] = set()  # entity ids referenced by a live job
        self._pinned: set[str] = set()  # entity ids kept as crop thumbnails
        # entity ids the UI's "Pin thumbnail" checkbox keeps alive across
        # generations (until unchecked) — unlike crop pins, these are not
        # dropped when a new generation starts
        self._user_pinned: set[str] = set()
        self._upload_counter = 0
        self._crop_counter = 0
        # brush PNG lookup: the provided defaults (res/brushes/) are
        # searched first, then the user's brushes (user/brushes/)
        self._brushes_dir = brushes_dir
        self._default_brushes_dir = default_brushes_dir
        self._brush_cache: dict[str, np.ndarray] = {}  # name -> float32 L array 0..1

    def _brush_dirs(self) -> list[Path]:
        """Existing brush folders in search order (defaults, then user)."""
        return [d for d in (self._default_brushes_dir, self._brushes_dir)
                if d is not None and d.is_dir()]

    def _brush_file(self, name: str) -> Path | None:
        """The PNG for brush `name` (first folder hit wins), or None."""
        for d in self._brush_dirs():
            p = d / f"{name}.png"
            if p.is_file():
                return p
        return None

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
        """The displayed image's mask (at its size), or None when there is
        no image."""
        with self._lock:
            if self._current is None:
                return None
            return self._images[self._current].mask.copy()

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

    def replace_state(self, img: np.ndarray, mask: np.ndarray) -> None:
        """Client-authoritative state push: replace the displayed entity's
        image and mask. PNG text chunks (metadata) are preserved."""
        with self._lock:
            if self._current is None:
                raise ValueError("cannot replace state on an empty canvas")
            ent = self._images[self._current]
            ent.image = np.ascontiguousarray(img)
            ent.mask = np.ascontiguousarray(mask)
            ent.png = _encode_png(ent.image, ent.info)

    # --- entity management ---
    def _inherit_mask_locked(self, src_id: str | None, W: int, H: int) -> np.ndarray:
        """The mask for a new entity of image size (W, H): the source
        image's current mask transformed to the new size (aspect fit
        within, centered; identity when the size matches). No source ->
        fresh empty (zero) mask."""
        if src_id is None or src_id not in self._images:
            return np.zeros((H, W), dtype=np.uint8)
        return _transform_mask(self._images[src_id].mask, W, H)

    def add_result(
        self, job_id: int, index: int, png: bytes, info: dict | None,
        source_id: str | None = None,
    ) -> str:
        """Register a generation result as a (cold) entity; returns its id.

        The result inherits its current mask from ``source_id`` (the job's
        source image, the green-bordered source thumbnail); see
        ``_inherit_mask_locked``. A result entity is always referenced by
        its (still live) job, so it is anchored immediately and won't be
        evicted until the job is dropped.
        """
        eid = f"job:{job_id}:{index}"
        with self._lock:
            w, h = Image.open(BytesIO(png)).size
            self._images[eid] = CanvasImage(
                id=eid,
                png=png,
                info=dict(info or {}),
                mask=self._inherit_mask_locked(source_id, w, h),
            )
            self._anchored.add(eid)
        return eid

    def import_image(self, data: bytes) -> str:
        """Upload / drag&drop: create an ``upload:{n}`` entity and set it.

        The new entity inherits the displayed image's current mask (the
        upload replaces it); see ``_inherit_mask_locked``."""
        arr = _decode_png(data)
        info = _png_text_chunks(data)
        with self._lock:
            eid = f"upload:{self._upload_counter}"
            self._upload_counter += 1
            h, w = arr.shape[:2]
            self._images[eid] = CanvasImage(
                id=eid, png=data, info=info, image=arr,
                mask=self._inherit_mask_locked(self._current, w, h),
            )
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
        self._evict_stale_locked()

    def anchor(self, ids: set[str]) -> None:
        """Record which entity ids are referenced by a live job."""
        with self._lock:
            self._anchored = set(ids)

    def pin(self, id: str) -> None:
        """Keep entity ``id`` alive outside of a job, as a selectable
        thumbnail. Used by crop: the pre-crop image and the crop result both
        stay restorable even after they leave the canvas."""
        with self._lock:
            self._pin_locked(id)

    def release_pins(self) -> None:
        """Drop every crop pin and evict what is no longer referenced. Called
        when a new generation starts (the client clears its crop thumbnails
        at that point, so the pinned entities are no longer reachable).
        User pins ("Pin thumbnail") are left alone: those thumbnails and
        their images survive the generation by design."""
        with self._lock:
            self._pinned.clear()
            self._evict_stale_locked()

    def _pin_locked(self, id: str) -> None:
        if id in self._images:
            self._pinned.add(id)

    def user_pin(self, id: str) -> None:
        """Keep entity ``id`` alive across generations (the UI's
        "Pin thumbnail" checkbox). Raises ValueError when unknown."""
        with self._lock:
            if id not in self._images:
                raise ValueError(f"unknown canvas image: {id!r}")
            self._user_pinned.add(id)

    def user_unpin(self, id: str) -> None:
        """Drop the user pin on entity ``id``; it may be evicted on the next
        pass. Unknown ids are ignored (idempotent)."""
        with self._lock:
            self._user_pinned.discard(id)

    def user_pinned(self) -> set[str]:
        """A copy of the user-pinned entity ids (for /api/queue)."""
        with self._lock:
            return set(self._user_pinned)

    def evict_stale(self) -> None:
        with self._lock:
            self._evict_stale_locked()

    def _evict_stale_locked(self) -> None:
        for eid in list(self._images.keys()):
            if (eid != self._current and eid not in self._anchored
                    and eid not in self._pinned
                    and eid not in self._user_pinned):
                del self._images[eid]

    def set_mask(self, mask: np.ndarray) -> bool:
        """Replace the displayed image's mask (at its size); returns True
        when it changed. (The segmentation endpoint is the only caller —
        user mask edits arrive via ``replace_state``.)"""
        with self._lock:
            if self._current is None or self._images[self._current].image is None:
                raise ValueError("cannot set a mask on an empty canvas")
            ent = self._images[self._current]
            if mask.shape[:2] != ent.image.shape[:2]:
                raise ValueError("mask must be at the image's size")
            if mask.shape == ent.mask.shape and np.array_equal(mask, ent.mask):
                return False
            ent.mask = np.ascontiguousarray(mask)
            return True

    # --- brushes ---
    def list_brushes(self) -> list[str]:
        """'circle' plus every greyscale png (by file stem), the provided
        defaults (res/brushes/) first, then the user's (user/brushes/);
        a name in both folders is listed once (the default file wins)."""
        names = ["circle"]
        seen: set[str] = set()
        for d in self._brush_dirs():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() == ".png" and p.is_file() \
                        and p.stem not in seen:
                    names.append(p.stem)
                    seen.add(p.stem)
        return names

    def brush_shape(self, name: str, size: float) -> np.ndarray:
        """Brush alpha shape for a stamp of the given size (px, = largest
        dimension), float32 0..1. 'circle' is built in; anything else is a
        greyscale PNG whose luminance is the alpha."""
        with self._lock:
            if name != "circle" and name not in self._brush_cache:
                p = self._brush_file(name)
                if p is None:
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

    def brush_png(self, name: str) -> bytes:
        """The brush's greyscale PNG bytes (luminance = alpha) at 256 px
        largest dimension, for the client's brush-shape previews.
        Unknown names raise KeyError."""
        shape = self.brush_shape(name, 256)
        img = Image.fromarray((shape * 255.0).astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # --- crop ---
    def crop(self, x: float, y: float, w: float, h: float) -> tuple[str, str]:
        """Crop the displayed image to the ``(x, y, w, h)`` rect (canvas px,
        clamped to the image bounds). The pre-crop image is kept (pinned) as a
        restorable entity; the crop becomes a new entity, set current and
        pinned as well. Returns ``(prev_image_id, crop_image_id)``."""
        with self._lock:
            if self._current is None or self._images[self._current].image is None:
                raise ValueError("cannot crop an empty canvas")
            ent = self._images[self._current]
            img = ent.image
            H, W = img.shape[:2]
            x0 = int(round(min(max(x, 0.0), W)))
            y0 = int(round(min(max(y, 0.0), H)))
            x1 = int(round(min(max(x + w, 0.0), W)))
            y1 = int(round(min(max(y + h, 0.0), H)))
            if x1 - x0 < 1 or y1 - y0 < 1:
                raise ValueError("crop region is empty")
            sub = img[y0:y1, x0:x1].copy()
            # the mask is at the image's size, so it is cropped 1:1 with the
            # image
            crop_mask = ent.mask[y0:y1, x0:x1].copy()
            prev_id = self._current
            eid = f"crop:{self._crop_counter}"
            self._crop_counter += 1
            self._images[eid] = CanvasImage(
                id=eid, png=_encode_png(sub, ent.info), info=dict(ent.info),
                image=sub, mask=crop_mask,
            )
            # keep both the pre-crop image and the crop restorable as
            # thumbnails (pinned before entering, so eviction sees them)
            self._pin_locked(prev_id)
            self._pin_locked(eid)
            self._enter_entity_locked(eid)
            return prev_id, eid

    def entity_png(self, id: str) -> bytes | None:
        """The cold PNG of any known entity (crop/result thumbnails), or
        None when it no longer exists."""
        with self._lock:
            ent = self._images.get(id)
            return None if ent is None else ent.png

    def entity_info(self, id: str) -> dict | None:
        """The PNG text chunks (metadata) of any known entity, as a copy,
        or None when it no longer exists."""
        with self._lock:
            ent = self._images.get(id)
            return None if ent is None else dict(ent.info)

    # --- (de)serialization ---
    def to_png_bytes(self) -> bytes | None:
        with self._lock:
            if self._current is None:
                return None
            return self._images[self._current].png

    def to_mask_png_bytes(self) -> bytes | None:
        """The displayed image's mask as a greyscale PNG, or None when
        there is no image."""
        with self._lock:
            if self._current is None:
                return None
            return _encode_mask(self._images[self._current].mask)

    def png_info(self) -> dict:
        with self._lock:
            if self._current is None:
                return {}
            return dict(self._images[self._current].info)
