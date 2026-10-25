"""Image upscaling (the toolbar's Upscale dialog).

Semantics follow stable-diffusion-webui-forge:

- ``Upscaler.upscale`` size loop (modules/upscaler.py): the final size is
  ``int(w * scale) // 8 * 8``; the model runs repeatedly (max 3 passes)
  until the image reaches that size, then a LANCZOS fit adjusts it exactly
  (so a 4x model at scale 2 overshoots first and shrinks back, and scale 1
  still runs one model pass -- a plain enhancement run).
- ``upscaler_utils.upscale_with_model`` + ``images.split_grid/combine_grid``
  (feathered tile-overlap masks) for tiled inference.
- ``limit_size_by_one_dimention`` (scripts/postprocessing_upscale.py) for
  the optional max-side-length clamp.

Models are .pth/.pt files under ``models/upscalers/`` loaded via spandrel
(the same library Forge and ComfyUI use; architecture auto-detected:
ESRGAN/RealESRGAN, SwinIR, ScuNET, DAT, HAT, ...). The two built-in
pseudo-upscalers "Lanczos"/"Nearest" are pure PIL resizes (Forge lists
them the same way).
"""

from __future__ import annotations

import logging
import math
import threading
from collections import namedtuple
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

LANCZOS = Image.Resampling.LANCZOS
NEAREST = Image.Resampling.NEAREST

# absolute pixel guard (VRAM protection); the dialog's max-side-length
# option is the user-facing knob, this is the never-exceed backstop
MAX_SIDE_HARD_LIMIT = 16384

UPSCALER_SUFFIXES = (".pth", ".pt")

# Forge's built-in pseudo-upscalers (no model file, pure PIL)
BUILTIN_UPSCALERS = ("Lanczos", "Nearest")


class UpscaleError(Exception):
    """User-facing upscale failure (bad request, load error, hard limit)."""


class UpscaleBusy(UpscaleError):
    """Another upscale is already running (one at a time per process)."""


# ------------------------------------------------------------------ grid ---
# port of forge modules/images.py Grid/split_grid/combine_grid


Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w",
                           "image_h", "overlap"])


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512,
               overlap: int = 64) -> Grid:
    w, h = image.size

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)
        if y + tile_h >= h:
            y = max(h - tile_h, 0)

        for col in range(cols):
            x = int(col * dx)
            if x + tile_w >= w:
                x = max(w - tile_w, 0)

            tile = image.crop((x, y, x + tile_w, y + tile_h))
            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid: Grid) -> Image.Image:
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, "L")

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32)
                             .reshape((1, grid.overlap))
                             .repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32)
                             .reshape((grid.overlap, 1))
                             .repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0),
                               mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)),
                               (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(
            combined_row.crop((0, 0, combined_row.width, grid.overlap)),
            (0, y), mask=mask_h)
        combined_image.paste(
            combined_row.crop((0, grid.overlap, combined_row.width, h)),
            (0, y + grid.overlap))

    return combined_image


# ------------------------------------------------------------ tensor conv ---


def pil_image_to_torch_bgr(img: Image.Image):
    """port of forge upscaler_utils.pil_image_to_torch_bgr"""
    import torch

    arr = np.array(img.convert("RGB"))
    arr = arr[:, :, ::-1]  # flip RGB to BGR
    arr = np.transpose(arr, (2, 0, 1))  # HWC to CHW
    arr = np.ascontiguousarray(arr) / 255  # rescale to [0, 1]
    return torch.from_numpy(arr)


def torch_bgr_to_pil_image(tensor) -> Image.Image:
    """port of forge upscaler_utils.torch_bgr_to_pil_image (batch dim of 1
    is squeezed away)"""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------- manager --


def limit_size_by_one_dimension(w: float, h: float,
                                limit: int) -> tuple[int, int]:
    """port of forge postprocessing_upscale.limit_size_by_one_dimention:
    shrink (w, h) so no side exceeds ``limit``, preserving the aspect."""
    if h > w and h > limit:
        w = limit * w // h
        h = limit
    elif w > limit:
        h = limit * h // w
        w = limit
    return int(w), int(h)


class UpscalerManager:
    """Discovers, loads and runs the upscalers.

    ``device`` is the torch device the models run on ("cuda:1" / "cpu");
    None means CPU. Loaded models are cached on CPU and moved to the
    target device for the duration of one upscale run (the models are
    small -- a few hundred MB at most -- so this parking pattern is cheap
    and keeps VRAM free while idle). Only one upscale runs at a time.
    """

    def __init__(self, upscalers_dir: Path, device: str | None = None) -> None:
        self._dir = upscalers_dir
        self._device = device or "cpu"
        self._models: dict[str, object] = {}  # id -> spandrel descriptor (CPU)
        self._busy = threading.Lock()

    # --- discovery ---
    def _scan(self) -> dict[str, Path]:
        from .models import _walk_files

        out: dict[str, Path] = {}
        if not self._dir.is_dir():
            return out
        for f in sorted(_walk_files(self._dir, UPSCALER_SUFFIXES)):
            rel = f.relative_to(self._dir).with_suffix("").as_posix()
            # built-in ids win: a model file may never shadow them
            if rel not in BUILTIN_UPSCALERS:
                out[rel] = f
        return out

    def list_upscalers(self) -> list[dict]:
        """Built-ins first, then the scanned models (id == display name)."""
        items = [{"id": name, "name": name, "builtin": True}
                 for name in BUILTIN_UPSCALERS]
        items += [{"id": rel, "name": rel, "builtin": False}
                  for rel in self._scan()]
        return items

    # --- sizing (forge Upscaler.upscale rule) ---
    def dest_size(self, w: int, h: int, scale: float,
                  max_side: int) -> tuple[int, int, float]:
        """(dest_w, dest_h, effective scale) for an upscale of a w x h
        image; applies the forge max-side clamp before the //8*8 rule."""
        if max_side > 0 and max(w, h) * scale > max_side:
            tw, th = limit_size_by_one_dimension(w * scale, h * scale,
                                                 max_side)
            scale = max(tw / w, th / h)
        dest_w = int(w * scale) // 8 * 8
        dest_h = int(h * scale) // 8 * 8
        return dest_w, dest_h, scale

    # --- model loading ---
    def _load(self, uid: str):
        desc = self._models.get(uid)
        if desc is not None:
            return desc
        path = self._scan().get(uid)
        if path is None:
            raise UpscaleError(f"unknown upscaler: {uid!r}")
        # parameter creation is serialized process-wide (same rule as the
        # backends: libraries monkey-patch global torch state on load)
        from .backends import _PARAM_INIT_LOCK

        with _PARAM_INIT_LOCK:
            from spandrel import ImageModelDescriptor, ModelLoader

            try:
                loaded = ModelLoader().load_from_file(str(path))
            except Exception as e:
                raise UpscaleError(
                    f"cannot load upscaler {uid!r}: {type(e).__name__}: {e}"
                ) from e
        if not isinstance(loaded, ImageModelDescriptor):
            raise UpscaleError(
                f"upscaler {uid!r} is not a single-image model")
        loaded.eval()
        loaded.cpu()  # the cache copy always lives on CPU
        self._models[uid] = loaded
        log.info("upscaler load: %s (%s, %dx, tiling %s)", uid,
                 loaded.architecture.id, loaded.scale,
                 loaded.tiling.name.lower())
        return loaded

    # --- run ---
    def run(self, arr: np.ndarray, upscaler_id: str, scale: float,
            max_side: int = 0, tile: int = 192, tile_overlap: int = 8,
            progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """Upscale an RGB uint8 HxWx3 array; returns the result array.

        ``progress(done, total)`` is called once per tile (and once for an
        untiled pass). Raises UpscaleError (unknown/broken upscaler, hard
        pixel limit) or UpscaleBusy (another run is active).
        """
        if not self._busy.acquire(blocking=False):
            raise UpscaleBusy("an upscale is already running")
        try:
            return self._run(arr, upscaler_id, scale, max_side, tile,
                             tile_overlap, progress)
        finally:
            self._busy.release()

    def _run(self, arr, upscaler_id, scale, max_side, tile, tile_overlap,
             progress):
        h, w = arr.shape[:2]
        dest_w, dest_h, eff_scale = self.dest_size(w, h, scale, max_side)
        if max(dest_w, dest_h) > MAX_SIDE_HARD_LIMIT:
            raise UpscaleError(
                f"upscale output would be {dest_w}x{dest_h}, exceeding the "
                f"{MAX_SIDE_HARD_LIMIT} px side limit; lower the scale or "
                f"set a max side length")
        img = Image.fromarray(arr)

        if upscaler_id in BUILTIN_UPSCALERS:
            resample = LANCZOS if upscaler_id == "Lanczos" else NEAREST
            if (img.width, img.height) != (dest_w, dest_h):
                img = img.resize((dest_w, dest_h), resample=resample)
            return np.asarray(img)

        desc = self._load(upscaler_id)

        # pass schedule: forge's loop (apply the model until the image
        # reaches the destination size, max 3 passes; scale 1 still runs
        # exactly one pass; a 1x model breaks after its first)
        ms = desc.scale
        sizes: list[tuple[int, int]] = []
        cur_w, cur_h = w, h
        for i in range(3):
            if cur_w >= dest_w and cur_h >= dest_h and (i > 0 or
                                                        eff_scale != 1):
                break
            sizes.append((cur_w, cur_h))
            if ms == 1:
                break
            cur_w *= ms
            cur_h *= ms
        total = sum(self._tile_count(sw, sh, tile) for sw, sh in sizes)

        desc.to(self._device)
        try:
            done = 0
            for _size in sizes:
                img, done = self._model_pass(desc, img, tile, tile_overlap,
                                             progress, done, total)
        finally:
            desc.cpu()

        if img.size != (dest_w, dest_h):
            img = img.resize((dest_w, dest_h), resample=LANCZOS)
        return np.asarray(img.convert("RGB"))

    def _tile_count(self, w: int, h: int, tile: int) -> int:
        if not self._tiles(w, h, tile):
            return 1
        overlap = self._overlap(tile, tile)
        cols = math.ceil((w - overlap) / (tile - overlap))
        rows = math.ceil((h - overlap) / (tile - overlap))
        return max(cols, 1) * max(rows, 1)

    @staticmethod
    def _tiles(w: int, h: int, tile: int) -> bool:
        """Tiling only when it strictly splits the image (tile >= the
        smaller side runs one untiled pass: a tile that is not smaller
        than the image brings Forge's clamping quirks and no VRAM win)."""
        return tile > 0 and tile < min(w, h)

    @staticmethod
    def _overlap(tile: int, overlap: int) -> int:
        # keep split_grid's stride positive (Forge clamps via the UI)
        return min(overlap, max(tile // 2, 0))

    def _model_pass(self, desc, img: Image.Image, tile: int, overlap: int,
                    progress, done: int,
                    total: int) -> tuple[Image.Image, int]:
        if not self._tiles(img.width, img.height, tile):
            out = self._run_patch(desc, img)
            done += 1
            if progress is not None:
                progress(done, total)
            return out, done

        grid = split_grid(img, tile, tile, self._overlap(tile, overlap))
        new_tiles = []
        sf = None
        for y, th, row in grid.tiles:
            new_row = []
            for x, tw, tile_img in row:
                out = self._run_patch(desc, tile_img)
                sf = out.width // tile_img.width
                new_row.append([x * sf, tw * sf, out])
                done += 1
                if progress is not None:
                    progress(done, total)
            new_tiles.append([y * sf, th * sf, new_row])
        new_grid = Grid(
            new_tiles,
            tile_w=grid.tile_w * sf,
            tile_h=grid.tile_h * sf,
            image_w=grid.image_w * sf,
            image_h=grid.image_h * sf,
            overlap=grid.overlap * sf,
        )
        return combine_grid(new_grid), done

    def _run_patch(self, desc, img: Image.Image) -> Image.Image:
        import torch

        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)
        tensor = tensor.to(device=desc.device, dtype=desc.dtype)
        # spandrel's __call__ runs under torch.inference_mode(), so the
        # output is an inference tensor: the conversion below (in-place
        # clamp_) must happen inside the same context
        with torch.inference_mode():
            out = desc(tensor)
            return torch_bgr_to_pil_image(out)
