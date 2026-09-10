"""Server-authoritative canvas: image (+ mask from Phase 2) with undo/redo.

The canvas is a single numpy HxWx3 uint8 array. Every committed change
(stroke end, generation, mask ops, thumbnail selection, import) pushes a
snapshot of the *previous* state onto the undo stack.
"""

from __future__ import annotations

import threading
from io import BytesIO

import numpy as np
from PIL import Image

MAX_HISTORY = 200


class Canvas:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._image: np.ndarray | None = None
        self._undo: list[np.ndarray | None] = []
        self._redo: list[np.ndarray | None] = []

    # --- state ---
    @property
    def image(self) -> np.ndarray | None:
        with self._lock:
            return None if self._image is None else self._image.copy()

    def has_image(self) -> bool:
        with self._lock:
            return self._image is not None

    # --- mutations (each records undo) ---
    def _commit_locked(self, new_image: np.ndarray | None) -> None:
        if (
            self._image is not None
            and new_image is not None
            and self._image.shape == new_image.shape
        ):
            same = np.array_equal(self._image, new_image)
        else:
            same = False
        if same:
            return
        self._undo.append(self._image.copy() if self._image is not None else None)
        if len(self._undo) > MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()
        self._image = new_image

    def set_image(self, image: np.ndarray) -> None:
        with self._lock:
            self._commit_locked(image)

    def clear(self) -> None:
        with self._lock:
            self._commit_locked(None)

    def undo(self) -> None:
        with self._lock:
            if not self._undo:
                return
            self._redo.append(self._image.copy() if self._image is not None else None)
            self._image = self._undo.pop()

    def redo(self) -> None:
        with self._lock:
            if not self._redo:
                return
            self._undo.append(self._image.copy() if self._image is not None else None)
            self._image = self._redo.pop()

    def can_undo(self) -> bool:
        with self._lock:
            return bool(self._undo)

    def can_redo(self) -> bool:
        with self._lock:
            return bool(self._redo)

    # --- (de)serialization ---
    def to_png_bytes(self) -> bytes | None:
        img = self.image
        if img is None:
            return None
        buf = BytesIO()
        # keep any text chunks (metadata) that arrived with a set from PNG bytes
        extra = self._extra_info
        if extra:
            Image.fromarray(img).save(buf, format="PNG", pnginfo=_png_info_from(extra))
        else:
            Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    # carry PNG text chunks across set_image-from-bytes (metadata survives edits)
    _extra_info: dict = {}

    def set_from_png_bytes(self, data: bytes) -> None:
        img = Image.open(BytesIO(data))
        arr = np.asarray(img.convert("RGB"))
        info = {k: v for k, v in (img.info or {}).items() if isinstance(v, str)}
        with self._lock:
            self._extra_info = info
            self._commit_locked(arr)

    def png_info(self) -> dict:
        with self._lock:
            return dict(self._extra_info)


def _png_info_from(extra: dict) -> Image.PngInfo:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    for k, v in extra.items():
        info.add_text(k, v)
    return info
