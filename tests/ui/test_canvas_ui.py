"""Browser tests for the canvas UI (Chromium via Playwright).

Covers the parts that can only be verified against a real browser:
toolbar wiring, pointer strokes (brush + mask + shift-erase), zoom/pan,
tool-state persistence across reload, custom brush listing.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    canvas_px,
    dot,
    mask_arr,
    set_canvas,
    set_input,
    stroke,
    wait_until,
)
from PIL import Image

BG = (20, 30, 40)


def _open(page, base):
    page.goto(base)
    # toolbar is present and the brush list was fetched
    assert page.is_visible("#tool-brush")
    page.wait_for_function("brushNames.length >= 1")
    assert page.locator("#brush option").first.text_content() == "circle (default)"
    # no canvas yet: placeholder shown, <canvas> hidden
    assert page.is_visible("#canvas-empty")
    assert page.evaluate("document.getElementById('canvas').hidden")


def _show(page, base, w=200, h=150):
    """Put an image on the server canvas and make the UI render it."""
    set_canvas(base, w, h, BG)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")


def _centre(page) -> tuple[float, float]:
    return page.evaluate(
        """() => {
            const r = document.getElementById('canvas').getBoundingClientRect();
            return [r.left + r.width / 2, r.top + r.height / 2];
        }"""
    )


# ---------------- painting ----------------


def test_paint_with_brush_color(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    set_input(page, "#brush-color", "#ff0000")

    stroke(page, [(40, 75), (100, 75), (160, 75)])
    assert wait_until(lambda: canvas_px(base, 100, 75) == (255, 0, 0)), \
        f"centre: {canvas_px(base, 100, 75)}"
    # line ends are stamped too (first/last points), background untouched
    assert canvas_px(base, 40, 75) == (255, 0, 0)
    assert canvas_px(base, 160, 75) == (255, 0, 0)
    assert canvas_px(base, 10, 10) == BG


def test_paint_at_zoomed_transform(page, server):
    """Strokes land where the cursor is, at the current zoom level."""
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)
    set_input(page, "#brush-color", "#00ff00")

    # zoom in around the canvas centre, then paint
    cx, cy = _centre(page)
    for _ in range(4):
        page.mouse.move(cx, cy)
        page.mouse.wheel(0, -200)
    assert wait_until(lambda: page.evaluate("view.scale") > 5)

    # points near the centre: at 3-4x zoom, far points leave the canvas rect
    stroke(page, [(95, 70), (110, 70)])
    assert wait_until(lambda: canvas_px(base, 110, 70) == (0, 255, 0)), \
        f"painted at zoom: {canvas_px(base, 110, 70)}"
    assert canvas_px(base, 10, 10) == BG


# ---------------- mask tool ----------------


def test_mask_tool_shift_erase_and_morph(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 100, 80)

    # switch with the M hotkey
    page.keyboard.press("m")
    assert page.evaluate(
        "document.getElementById('tool-mask').classList.contains('active')")
    # the colour swatch stays live in mask mode (it edits the brush tool's
    # colour; mask strokes ignore colour server-side)
    assert not page.evaluate("document.getElementById('brush-color').disabled")

    dot(page, 50, 40)  # mask default 30 px -> disc, radius 15
    assert wait_until(lambda: mask_arr(base) is not None and mask_arr(base)[40, 50] == 255)
    assert canvas_px(base, 50, 40) == BG  # image untouched by mask strokes

    # shift = erase: punch a hole in the centre (leaves a ring). The eraser
    # must fully cover the pixel (alpha ~1) for destination-out to zero it.
    set_input(page, "#brush-size", "0.6")  # ~6.3 px -> radius ~3.1
    dot(page, 50, 40, shift=True)
    assert wait_until(lambda: mask_arr(base)[40, 50] == 0)
    ring = int((mask_arr(base) > 0).sum())
    assert ring > 100, f"ring too small: {ring}"

    # dilate button grows the mask (px field = 4)
    set_input(page, "#mask-px", "4")
    page.click("#mask-dilate")
    assert wait_until(lambda: int((mask_arr(base) > 0).sum()) > ring + 100)

    # clear mask button empties it
    page.click("#mask-clear")
    assert wait_until(lambda: bool((mask_arr(base) == 0).all()))

    # header undo/redo restore it
    page.click("#btn-undo")
    assert wait_until(lambda: int((mask_arr(base) > 0).sum()) > ring + 100)
    page.click("#btn-redo")
    assert wait_until(lambda: bool((mask_arr(base) == 0).all()))


# ---------------- zoom / pan ----------------


def test_stroke_preview_held_until_server_response(page, server):
    """No flicker on release: the on-screen preview must stay visible while
    the stroke request is in flight (regression: endStroke used to clear +
    redraw before the round trip, so the stroke vanished for ~100 ms)."""
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150 white
    set_input(page, "#brush-color", "#000000")

    # Delay the stroke response so the round-trip window is wide.
    page.evaluate("""() => {
        const orig = window.fetch;
        window.fetch = async (url, opts) => {
            const r = await orig(url, opts);
            if (String(url).includes('/canvas/stroke'))
                await new Promise((res) => setTimeout(res, 250));
            return r;
        };
    }""")
    stroke(page, [(90, 40), (110, 40)])  # ends with mouse.up

    def preview_px():
        return page.evaluate("""() => {
            const c = document.getElementById('canvas');
            const x = Math.round((view.ox + 100 * view.scale) * view.dpr);
            const y = Math.round((view.oy + 40 * view.scale) * view.dpr);
            const d = c.getContext('2d').getImageData(x, y, 1, 1).data;
            return [d[0], d[1], d[2]];
        }""")

    # inside the 250 ms delay: the preview must still be on screen
    assert preview_px() == [0, 0, 0], \
        f"preview lost during round trip: {preview_px()}"
    # ...and the server commits the stroke
    assert wait_until(lambda: canvas_px(base, 100, 40) == (0, 0, 0))
    # after the refresh the committed pixels are on screen
    assert preview_px() == [0, 0, 0]


def test_zoom_wheel_fit_pan(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)
    fit_scale = page.evaluate("view.scale")
    assert page.text_content("#zoom-label").strip().endswith("%")

    cx, cy = _centre(page)
    page.mouse.move(cx, cy)
    page.mouse.wheel(0, -300)
    assert page.evaluate("view.scale") > fit_scale * 1.3
    assert page.evaluate("view.scale") < fit_scale * 4  # sane growth

    # double-click refits (it also stamps two harmless dots)
    page.mouse.dblclick(cx, cy)
    assert abs(page.evaluate("view.scale") - fit_scale) < 0.01 * fit_scale

    # middle-button drag pans
    ox, oy = page.evaluate("([a, b]) => [a, b]", [page.evaluate("view.ox"),
                                                  page.evaluate("view.oy")])
    page.mouse.move(cx, cy)
    page.mouse.down(button="middle")
    page.mouse.move(cx + 60, cy + 40, steps=4)
    page.mouse.up(button="middle")
    new_ox, new_oy = page.evaluate("([a, b]) => [a, b]", [page.evaluate("view.ox"),
                                                          page.evaluate("view.oy")])
    assert new_ox - ox == pytest.approx(60, abs=2)
    assert new_oy - oy == pytest.approx(40, abs=2)


# ---------------- state persistence ----------------


def test_tool_state_persists_across_reload(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    page.keyboard.press("m")  # mask tool
    set_input(page, "#brush-size", "0.833")  # ~31.6 px (log scale)
    set_input(page, "#brush-opacity", "0.4")

    page.reload()
    page.wait_for_function(
        "document.getElementById('tool-mask').classList.contains('active')")
    assert page.evaluate("Tools.get().size") == pytest.approx(31.6, abs=0.6)
    assert page.evaluate("Tools.get().opacity") == pytest.approx(0.4, abs=0.01)
    # per-tool memory: switching back to brush restores its own settings
    page.keyboard.press("b")
    assert page.evaluate("Tools.get().size") == pytest.approx(10, abs=0.5)


def test_custom_brush_listed_and_used(page, server):
    base, root = server
    # a soft brush: falloff from the centre (luminance = alpha), written
    # before the page loads so the initial /api/brushes lists it
    yy, xx = np.mgrid[0:24, 0:24]
    d = np.hypot(xx - 11.5, yy - 11.5)
    a = (np.clip(1 - d / 12.0, 0, 1) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    (Path(root) / "brushes" / "soft24.png").write_bytes(buf.getvalue())

    _open(page, base)
    _show(page, base, 200, 150)
    page.wait_for_function("brushNames.includes('soft24')")
    page.select_option("#brush", "soft24")
    set_input(page, "#brush-color", "#00ff00")
    set_input(page, "#brush-size", "0.867")  # = 40 px

    # single stamp at the centre: strong in the middle, soft near the rim
    dot(page, 100, 75)
    assert wait_until(lambda: canvas_px(base, 100, 75)[1] > 150)
    centre = canvas_px(base, 100, 75)
    edge = canvas_px(base, 100, 93)  # 18 px out: inside a 40px stamp, near its rim
    assert edge[1] < centre[1], f"no falloff: centre={centre} edge={edge}"
