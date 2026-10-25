"""UpscalerManager sizing rules + the tiled combine grid (no GPU and no
upscaler weights needed: the model pass is replaced with a deterministic
nearest x4 so the tile PLACEMENT is verifiable exactly)."""

import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from imagagen.upscalers import (
    UpscaleBusy,
    UpscaleError,
    UpscalerManager,
    limit_size_by_one_dimension,
    split_grid,
)


class _Fake4x(UpscalerManager):
    """UpscalerManager whose "model" is a plain 4x nearest resize, so
    _model_pass's split/combine geometry can be checked exactly: every
    tile comes from the same source, so overlapping tiles agree and the
    feathered blend must reproduce the whole-image resize."""

    def _run_patch(self, desc, img: Image.Image) -> Image.Image:
        return img.resize((img.width * 4, img.height * 4), Image.NEAREST)


def test_dest_size_forge_rule():
    m = UpscalerManager.__new__(UpscalerManager)  # no state needed
    # int(w*scale)//8*8 (200x150 x2 -> 400x296, NOT 300)
    assert m.dest_size(200, 150, 2.0, 0) == (400, 296, 2.0)
    # max-side clamp via limit_size_by_one_dimention first
    assert m.dest_size(200, 150, 4.0, 256) == (256, 192, 1.28)
    # a scale whose //8*8 target sits below the source size runs no pass
    assert m.dest_size(100, 100, 1.02, 0) == (96, 96, 1.02)


def test_limit_size_by_one_dimension():
    assert limit_size_by_one_dimension(800, 600, 256) == (256, 192)
    assert limit_size_by_one_dimension(600, 800, 256) == (192, 256)
    assert limit_size_by_one_dimension(200, 100, 256) == (200, 100)


def test_tiled_pass_reproduces_full_resize():
    """Regression: the combine grid must scale the tile ROW positions by
    the model's factor too (forge's newtiles.append([y * scale, ...]));
    without that the lower rows land at source-y offsets and the bottom
    of the result stays black. Also pins that the progress `total` matches
    the tile count actually processed (it used to be computed with the
    maximal overlap: 1720x720/tile 192/overlap 8 counted 119 but ran 40)."""
    m = _Fake4x.__new__(_Fake4x)
    src = np.arange(96 * 128 * 3, dtype=np.uint8).reshape(96, 128, 3)
    img = Image.fromarray(src)
    for tile, overlap in ((32, 8), (48, 16), (48, 1)):
        events = []
        out, done = m._model_pass(None, img, tile, overlap,
                                  lambda d, t, ev=events: ev.append((d, t)),
                                  0, 0)
        assert out.size == (512, 384)
        want = np.asarray(img.resize((512, 384), Image.NEAREST))
        assert np.array_equal(np.asarray(out), want), (tile, overlap)
        # the pass processes exactly the number of tiles _tile_count
        # promises and reports each one (run() passes this count as total)
        assert done == events[-1][0] == len(events) == m._tile_count(
            128, 96, tile, overlap), (tile, overlap, events[-1])


def test_tile_count_matches_split_grid():
    m = UpscalerManager.__new__(UpscalerManager)
    # the reported-user-case: 1720x720, tile 192, overlap 8 -> 40 tiles
    g = split_grid(Image.new("RGB", (1720, 720)), 192, 192,
                   m._overlap(192, 8))
    assert m._tile_count(1720, 720, 192, 8) == 40 == \
        sum(len(row_imgs) for _y, _h, row_imgs in g.tiles)
    for w, h, tile, ov in ((512, 512, 192, 8), (100, 60, 40, 20),
                           (64, 64, 64, 8), (300, 300, 100, 50)):
        if m._tiles(w, h, tile):
            g = split_grid(Image.new("RGB", (w, h)), tile, tile,
                           m._overlap(tile, ov))
            assert m._tile_count(w, h, tile, ov) == \
                sum(len(row_imgs) for _y, _h, row_imgs in g.tiles), \
                (w, h, tile, ov)


def test_tiling_guards():
    m = UpscalerManager.__new__(UpscalerManager)
    # tiling only when the tile strictly splits the image
    assert m._tiles(128, 128, 127) and not m._tiles(128, 128, 128)
    assert not m._tiles(128, 128, 0)
    # overlap never reaches half a tile (split_grid stride stays positive)
    assert m._overlap(64, 64) == 32
    assert m._overlap(64, 8) == 8


def test_run_builtins_and_busy_lock():
    m = UpscalerManager(Path("/nonexistent"), device=None)
    arr = np.zeros((16, 16, 3), np.uint8)
    out = m.run(arr, "Nearest", 2.0)
    assert out.shape[:2] == (32, 32)
    try:
        m.run(arr, "no/such/model", 2.0)
        raise AssertionError("expected UpscaleError")
    except UpscaleError as e:
        assert "unknown upscaler" in str(e)

    # busy lock: while one run is in flight a second one is UpscaleBusy
    # (on a GPU server the endpoint's registry lock serializes first;
    # this guard covers the CPU/mock path)
    real_run = m._run
    started = threading.Event()
    release = threading.Event()

    def slow_run(*a, **k):
        started.set()
        release.wait(5)
        return real_run(*a, **k)

    got = []

    def first():
        try:
            m.run(arr, "Nearest", 2.0)
            got.append("first-ok")
        except Exception as e:  # pragma: no cover - must not happen
            got.append(f"first-{e}")

    def second():
        try:
            m.run(arr, "Nearest", 2.0)
            got.append("second-ok")
        except UpscaleBusy:
            got.append("second-busy")

    m._run = slow_run
    try:
        t1 = threading.Thread(target=first)
        t1.start()
        assert started.wait(2)
        t2 = threading.Thread(target=second)
        t2.start()
        time.sleep(0.1)
        release.set()
        t1.join(5)
        t2.join(5)
    finally:
        m._run = real_run
    assert "first-ok" in got and "second-busy" in got
