"""Server-authoritative canvas: per-image draw history + per-image mask.

Every canvas image is a persistent ``CanvasImage`` entity: a cold PNG store
plus (while it is the displayed image) a hot numpy cache, and its own
*draw* and *mask* undo/redo stacks of pre-change PNG snapshots. Image
selection is not an undoable change — selecting a different entity just
switches which history is active, and its state is restored with it.

Every entity owns a mask at its own image's size (always present; zero
when nothing is masked): it is served/displayed directly and mask strokes
map 1:1 (the whole canvas is maskable). A new entity (generation result,
upload) inherits the mask state of the image it derives from — the
result's source image (the green-bordered source thumbnail), for uploads
the currently displayed image: current mask plus both history stacks. The
current mask is materialized at creation, transformed to the new image's
size (aspect fit within, centered — the same mapping the image itself
receives); inherited history frames stay cold PNG bytes at the source
image's size and are transformed on first use. A new entity without a
source (empty canvas) starts with a fresh empty mask.

History entries are PNG-compressed bytes (not raw numpy): a 1024x1024
entry is ~100–300 KB instead of ~3.2 MB, so deep stacks stay affordable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

from .generation import fit_ar, resize_shift_mask

MASK_HISTORY = 200  # per-image mask undo/redo depth
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
    # nothing is masked), plus its own undo/redo stacks of PNG bytes.
    # Frames inherited from the source image stay at the source image's
    # size and are transformed to this image's size on first use.
    mask: np.ndarray
    info: dict = field(default_factory=dict)  # PNG text chunks (metadata)
    image: np.ndarray | None = None  # hot cache: only while on the canvas
    undo: list[bytes] = field(default_factory=list)  # pre-stroke PNG snapshots
    redo: list[bytes] = field(default_factory=list)
    mask_undo: list[bytes] = field(default_factory=list)
    mask_redo: list[bytes] = field(default_factory=list)


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
    def _inherit_mask_state_locked(
        self, src_id: str | None, W: int, H: int
    ) -> tuple[np.ndarray, list, list]:
        """(mask, mask_undo, mask_redo) for a new entity of image size
        (W, H). Inherits the source image's mask state: the current mask
        transformed to the new size (aspect fit within, centered; identity
        when the size matches) plus both history stacks by reference (cold
        PNG bytes at the source image's size, transformed on first use).
        No source -> fresh empty (zero) mask."""
        if src_id is None or src_id not in self._images:
            return np.zeros((H, W), dtype=np.uint8), [], []
        src = self._images[src_id]
        return (
            _transform_mask(src.mask, W, H),
            list(src.mask_undo),
            list(src.mask_redo),
        )

    def add_result(
        self, job_id: int, index: int, png: bytes, info: dict | None,
        source_id: str | None = None,
    ) -> str:
        """Register a generation result as a (cold) entity; returns its id.

        The result inherits its mask state (current mask + history) from
        ``source_id`` (the job's source image, the green-bordered source
        thumbnail); see ``_inherit_mask_state_locked``. A result entity is
        always referenced by its (still live) job, so it is anchored
        immediately and won't be evicted until the job is dropped.
        """
        eid = f"job:{job_id}:{index}"
        with self._lock:
            w, h = Image.open(BytesIO(png)).size
            mask, m_undo, m_redo = self._inherit_mask_state_locked(source_id, w, h)
            self._images[eid] = CanvasImage(
                id=eid,
                png=png,
                info=dict(info or {}),
                mask=mask,
                mask_undo=m_undo,
                mask_redo=m_redo,
            )
            self._anchored.add(eid)
        return eid

    def import_image(self, data: bytes) -> str:
        """Upload / drag&drop: create an ``upload:{n}`` entity and set it.

        The new entity inherits the displayed image's mask state (the
        upload replaces it); see ``_inherit_mask_state_locked``."""
        arr = _decode_png(data)
        info = _png_text_chunks(data)
        with self._lock:
            eid = f"upload:{self._upload_counter}"
            self._upload_counter += 1
            h, w = arr.shape[:2]
            mask, m_undo, m_redo = self._inherit_mask_state_locked(
                self._current, w, h
            )
            self._images[eid] = CanvasImage(
                id=eid, png=data, info=info, image=arr,
                mask=mask, mask_undo=m_undo, mask_redo=m_redo,
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
        at that point, so the pinned entities are no longer reachable)."""
        with self._lock:
            self._pinned.clear()
            self._evict_stale_locked()

    def _pin_locked(self, id: str) -> None:
        if id in self._images:
            self._pinned.add(id)

    def evict_stale(self) -> None:
        with self._lock:
            self._evict_stale_locked()

    def _evict_stale_locked(self) -> None:
        for eid in list(self._images.keys()):
            if (eid != self._current and eid not in self._anchored
                    and eid not in self._pinned):
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

    # --- mask history (per image) ---
    def can_undo_mask(self) -> bool:
        with self._lock:
            return bool(self._current and self._images[self._current].mask_undo)

    def can_redo_mask(self) -> bool:
        with self._lock:
            return bool(self._current and self._images[self._current].mask_redo)

    def _mask_snapshot_locked(self, ent: CanvasImage) -> bytes:
        return _encode_mask(ent.mask)

    def _materialize_mask_locked(self, ent: CanvasImage, data: bytes) -> np.ndarray:
        """History frame (PNG bytes) -> mask at the entity's image size.
        Frames pushed on this entity are already at its size; frames
        inherited from the source image are at the source's size and get
        transformed (aspect fit within, centered) here, on first use."""
        m = _decode_mask(data)
        h, w = ent.image.shape[:2]
        if m.shape[:2] == (h, w):
            return m
        return _transform_mask(m, w, h)

    def _undo_mask_locked(self) -> bool:
        if self._current is None:
            return False
        ent = self._images[self._current]
        if not ent.mask_undo:
            return False
        ent.mask_redo.append(self._mask_snapshot_locked(ent))
        prev = ent.mask_undo.pop()
        ent.mask = self._materialize_mask_locked(ent, prev)
        return True

    def _redo_mask_locked(self) -> bool:
        if self._current is None:
            return False
        ent = self._images[self._current]
        if not ent.mask_redo:
            return False
        ent.mask_undo.append(self._mask_snapshot_locked(ent))
        nxt = ent.mask_redo.pop()
        ent.mask = self._materialize_mask_locked(ent, nxt)
        return True

    def _set_mask_locked(self, ent: CanvasImage, mask: np.ndarray) -> bool:
        """Replace the entity's mask (at the image's size), recording one
        history step unless it is unchanged."""
        if mask.shape == ent.mask.shape and np.array_equal(mask, ent.mask):
            return False
        ent.mask_undo.append(self._mask_snapshot_locked(ent))
        if len(ent.mask_undo) > MASK_HISTORY:
            ent.mask_undo.pop(0)
        ent.mask_redo.clear()
        ent.mask = np.ascontiguousarray(mask)
        return True

    def set_mask(self, mask: np.ndarray) -> bool:
        with self._lock:
            if self._current is None or self._images[self._current].image is None:
                raise ValueError("cannot set a mask on an empty canvas")
            ent = self._images[self._current]
            if mask.shape[:2] != ent.image.shape[:2]:
                raise ValueError("mask must be at the image's size")
            return self._set_mask_locked(ent, mask)

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

    def clone(
        self,
        points: list[tuple[float, float]],
        offset: tuple[float, float],
        size: float,
        opacity: float = 1.0,
        brush: str = "circle",
    ) -> bool:
        """Clone (stamp) a stroke: paint the pre-stroke image translated by
        `offset` over the stroke's stamp footprint, blended by the footprint
        coverage x `opacity` (overlapping stamps do NOT stack the opacity).
        Samples the *pre-stroke* image (deterministic and matches the
        client's live preview). Returns True when anything changed
        (a draw-history step was recorded)."""
        opacity = min(max(opacity, 0.0), 1.0)
        with self._lock:
            return self._stroke_clone_locked(points, offset, size, opacity, brush)

    def smudge(
        self,
        points: list[tuple[float, float]],
        size: float,
        opacity: float = 1.0,
        brush: str = "circle",
    ) -> bool:
        """Smudge (smear) a stroke over the displayed image. Sequential:
        each stamp drags the brush-shaped region at the previous stamp
        centre to the current one (integer displacement, same size, so no
        resampling) and alpha-blends it by the brush shape x `opacity`,
        accumulating on a copy of the pre-stroke image. Returns True when
        anything changed (a draw-history step was recorded)."""
        opacity = min(max(opacity, 0.0), 1.0)
        with self._lock:
            return self._stroke_smudge_locked(points, size, opacity, brush)

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
        a = shape
        cr, cg, cb = (min(max(c, 0), 255) for c in color)
        # non-stacking, like mask/clone: the stroke's coverage is the MAX
        # brush alpha over all stamp footprints, and the colour is blended
        # in ONCE with coverage x opacity (overlapping stamps must not
        # stack, so a stroke's core is `opacity`, not ~1.0)
        cov = np.zeros((h, w), np.float32)
        for cx, cy in centers:
            x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
            sx0, sy0 = max(-x0, 0), max(-y0, 0)
            ex0, ey0 = min(sw, w - x0), min(sh, h - y0)
            if ex0 <= sx0 or ey0 <= sy0:
                continue
            region = cov[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0]
            np.maximum(region, a[sy0:ey0, sx0:ex0], out=region)
        img2 = img.astype(np.float32)
        aa = cov * opacity
        img2 = np.clip(
            img2 * (1 - aa[..., None]) + np.array([cr, cg, cb], np.float32)
            * aa[..., None],
            0,
            255,
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

    def _stroke_clone_locked(self, points, offset, size, opacity, brush) -> bool:
        """Apply a clone stroke to the displayed image (see `clone`)."""
        if self._current is None or self._images[self._current].image is None:
            raise ValueError("cannot stroke an empty canvas")
        ent = self._images[self._current]
        img = ent.image
        src = img  # pre-stroke image, read-only during the loop
        shape = self.brush_shape(brush, size)
        spacing = max(1.0, size * STAMP_SPACING_FRACTION)
        centers = self.stamp_positions(points, spacing)
        if not centers:
            return False
        h, w = img.shape[:2]
        sh, sw = shape.shape
        ox, oy = int(round(offset[0])), int(round(offset[1]))
        # every stamp copies the pre-stroke image at (dest + offset): with
        # the frozen stroke offset the source pixel of any destination pixel
        # is fixed for the whole stroke, so accumulate the STROKE's coverage
        # over all stamp footprints, then blend the translated image ONCE
        # with coverage x opacity (overlapping stamps must not stack the
        # opacity, e.g. 0.5 opacity stays 0.5 at the stroke core)
        cov = np.zeros((h, w), np.float32)
        for cx, cy in centers:
            # destination footprint origin (top-left) in image coords
            dx0, dy0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
            # shape-local sub-rect valid for BOTH the destination and the
            # source (= destination shifted by (ox, oy)); the copy is a pure
            # integer translation so the two regions align 1:1
            j0 = max(0, -dx0, -dx0 - ox)
            j1 = min(sw, w - dx0, w - dx0 - ox)
            i0 = max(0, -dy0, -dy0 - oy)
            i1 = min(sh, h - dy0, h - dy0 - oy)
            if j1 <= j0 or i1 <= i0:
                continue  # stamp (or its source) is entirely off-canvas
            w_cov = cov[dy0 + i0:dy0 + i1, dx0 + j0:dx0 + j1]
            w_cov = 1 - (1 - w_cov) * (1 - shape[i0:i1, j0:j1])
            cov[dy0 + i0:dy0 + i1, dx0 + j0:dx0 + j1] = w_cov
        ys, xs = np.nonzero(cov)
        if not len(ys):
            return False
        # blend window: where cov > 0 the source (dest + offset) is inside
        # the canvas by construction, so the shifted read is always valid
        py0, py1 = max(int(ys.min()), -oy, 0), min(int(ys.max()), h - 1 - oy)
        px0, px1 = max(int(xs.min()), -ox, 0), min(int(xs.max()), w - 1 - ox)
        if py1 < py0 or px1 < px0:
            return False
        img2 = img.astype(np.float32)
        aa = cov[py0:py1 + 1, px0:px1 + 1] * opacity
        s = src[py0 + oy:py1 + 1 + oy, px0 + ox:px1 + 1 + ox]
        img2[py0:py1 + 1, px0:px1 + 1] = np.clip(
            img2[py0:py1 + 1, px0:px1 + 1] * (1 - aa[..., None])
            + s * aa[..., None],
            0,
            255,
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

    def _stroke_smudge_locked(self, points, size, opacity, brush) -> bool:
        """Apply a smudge stroke to the displayed image (see `smudge`)."""
        if self._current is None or self._images[self._current].image is None:
            raise ValueError("cannot stroke an empty canvas")
        ent = self._images[self._current]
        img = ent.image
        shape = self.brush_shape(brush, size)
        spacing = max(1.0, size * STAMP_SPACING_FRACTION)
        centers = self.stamp_positions(points, spacing)
        if len(centers) < 2:
            return False  # a smudge is a DRAG: one point moves nothing
        h, w = img.shape[:2]
        sh, sw = shape.shape
        # Sequential smear: stamp N samples stamp N-1's OUTPUT, so the stroke
        # accumulates on a working copy of the pre-stroke image (a single
        # click changes nothing). Integer displacement + same-size copy,
        # matching the client's incremental canvas drawImage preview.
        work = img.astype(np.float32)
        for prev, cur in zip(centers, centers[1:], strict=False):
            dx = int(round(cur[0] - prev[0]))
            dy = int(round(cur[1] - prev[1]))
            if dx == 0 and dy == 0:
                continue
            # destination footprint origin (top-left) in image coords;
            # the source is the same footprint shifted back by (dx, dy);
            # the shape-local sub-rect must be valid for BOTH
            x0, y0 = int(round(cur[0] - sw / 2)), int(round(cur[1] - sh / 2))
            j0 = max(0, -x0, dx - x0)
            j1 = min(sw, w - x0, w - x0 + dx)
            i0 = max(0, -y0, dy - y0)
            i1 = min(sh, h - y0, h - y0 + dy)
            if j1 <= j0 or i1 <= i0:
                continue  # stamp (or its source) is entirely off-canvas
            sub = (shape[i0:i1, j0:j1] * opacity)[..., None]
            dst = work[y0 + i0:y0 + i1, x0 + j0:x0 + j1]
            src = work[y0 + i0 - dy:y0 + i1 - dy, x0 + j0 - dx:x0 + j1 - dx]
            # the RHS reads `work` whole before this stamp's write, so
            # overlapping source/destination regions are safe
            work[y0 + i0:y0 + i1, x0 + j0:x0 + j1] = np.clip(
                dst * (1 - sub) + src * sub, 0, 255
            )
        new_img = work.astype(np.uint8)
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
        ent = self._images[self._current]
        # canvas-pixel coordinates are mask-native coordinates (1:1)
        return self._stamp_mask(ent, points, size, opacity, brush, erase)

    def _stamp_mask(self, ent, points, size, opacity, brush, erase) -> bool:
        shape = self.brush_shape(brush, size)
        spacing = max(1.0, size * STAMP_SPACING_FRACTION)
        centers = self.stamp_positions(points, spacing)
        if not centers:
            return False
        # like image strokes, mask strokes use non-stacking opacity: the
        # added alpha is the fixed brush x 255 x opacity (overlapping
        # stamps do not stack, so a stroke's look does not depend on how
        # long it is dragged); erasing blends the alpha toward brush x
        # 255 x (1-opacity) -- soft, exactly like the client preview
        a = shape
        mh, mw = ent.mask.shape[:2]
        sh, sw = shape.shape
        m = ent.mask.astype(np.float32)
        cap = 255.0 * (1.0 - opacity)
        for cx, cy in centers:
            x0, y0 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
            sx0, sy0 = max(-x0, 0), max(-y0, 0)
            ex0, ey0 = min(sw, mw - x0), min(sh, mh - y0)
            if ex0 <= sx0 or ey0 <= sy0:
                continue
            region = m[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0]
            if erase:
                # each stamp blends the alpha toward the brush-scaled cap
                # (shape x cap): f(m) = m - s x (m - min(m, cap)). f is
                # affine, so sequential stamps compose to coverage
                # 1 - prod(1 - s_i) -- the same union coverage the client
                # preview blends with, so commit == preview. A full erase
                # (opacity 1) zeroes the full-alpha core exactly; only the
                # soft edges keep a faint residue
                s = a[sy0:ey0, sx0:ex0]
                m[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0] = (
                    region - s * (region - np.minimum(region, cap))
                )
            else:
                patch = (a[sy0:ey0, sx0:ex0] * 255.0) * opacity
                m[y0 + sy0 : y0 + ey0, x0 + sx0 : x0 + ex0] = np.maximum(
                    region, patch
                )
        new_mask = m.astype(np.uint8)
        return self._set_mask_locked(ent, new_mask)

    # --- mask operations (run on the displayed image's mask) ---
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
            if self._current is None:
                return False
            ent = self._images[self._current]
            if not ent.mask.any():
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
                ndimage.maximum_filter(ent.mask, footprint=se)
                if dilate
                else ndimage.minimum_filter(ent.mask, footprint=se)
            )
            return self._set_mask_locked(ent, out)

    def mask_clear(self) -> bool:
        with self._lock:
            if self._current is None:
                return False
            ent = self._images[self._current]
            if not ent.mask.any():
                return False
            return self._set_mask_locked(ent, np.zeros_like(ent.mask))

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
            # image; the mask history does not follow (its frames are at the
            # source size and a crop is not the aspect-fit transform they
            # would need), so the new entity starts with an empty history
            crop_mask = ent.mask[y0:y1, x0:x1].copy()
            prev_id = self._current
            eid = f"crop:{self._crop_counter}"
            self._crop_counter += 1
            self._images[eid] = CanvasImage(
                id=eid, png=_encode_png(sub, ent.info), info=dict(ent.info),
                image=sub, mask=crop_mask, mask_undo=[], mask_redo=[],
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
