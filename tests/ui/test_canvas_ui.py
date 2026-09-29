"""Browser tests for the canvas UI (Chromium via Playwright).

Covers the parts that can only be verified against a real browser:
toolbar wiring, pointer strokes (brush + mask + shift-erase), zoom/pan,
tool-state persistence across reload, custom brush listing.
"""

import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from conftest import (
    canvas_px,
    combo_pick,
    dot,
    mask_arr,
    screen_pos,
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
    # mask strokes have no colour, like clone/smudge
    assert page.evaluate("document.getElementById('brush-color').disabled")

    dot(page, 50, 40)  # mask default 30 px -> disc, radius 15
    assert wait_until(lambda: mask_arr(base) is not None and mask_arr(base)[40, 50] == 255)
    assert canvas_px(base, 50, 40) == BG  # image untouched by mask strokes

    # shift = erase: punch a hole in the centre (leaves a ring). The eraser
    # must fully cover the pixel (alpha ~1) for destination-out to zero it.
    set_input(page, "#brush-size", "0")  # 10 px -> radius 5
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

    # toolbar undo/redo restore it
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


def test_holder_resize_keeps_cursor_mapping_in_sync(page, server):
    """A canvas-holder resize WITHOUT a window resize (dragging the sidebar
    resizer; the persisted sidebar_width restored late from /api/ui-state)
    must resync the canvas backing store: #canvas is a replaced element
    clamped to the holder (max-width/max-height: 100%), so a stale backing
    store is silently rescaled to fit the new box and the whole render
    (image, brush outline, strokes) drifts from the mouse by a factor of
    distance-from-top-left."""
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)
    before_w = page.evaluate("holderBox().w")

    # what a sidebar resizer drag / late ui-state restore does
    page.evaluate("document.getElementById('controls').style.width = '640px'")
    page.wait_for_timeout(400)  # let the ResizeObserver tick land

    holder_w, ratio = page.evaluate(
        """() => {
            const c = document.getElementById('canvas');
            const r = c.getBoundingClientRect();
            return [holderBox().w,
                    c.width / (r.width * (window.devicePixelRatio || 1))];
        }""")
    assert holder_w < before_w - 50  # the holder actually shrank
    # 1 backing pixel must map to exactly 1 device pixel (1 CSS px * dpr)
    assert abs(ratio - 1) < 0.01, \
        f"backing store not in step with the resized box: scale {ratio:.3f}"

    # and a stroke still lands exactly under the cursor
    set_input(page, "#brush-color", "#00ff00")
    dot(page, 90, 70)
    assert wait_until(lambda: canvas_px(base, 90, 70) == (0, 255, 0)), \
        f"off-target after holder resize: {canvas_px(base, 90, 70)}"
    assert canvas_px(base, 10, 10) == BG


def test_brush_outline_follows_cursor(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150, uniform BG = (20,30,40)
    assert page.evaluate("view.cursor") is None
    box = page.locator("#canvas").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    assert page.evaluate("view.cursor !== null")

    # brightest pixel in a small box around the right edge of the brush ring
    def ring_bright():
        return page.evaluate(
            """() => {
              const ctx = document.getElementById('canvas').getContext('2d');
              const dpr = view.dpr;
              const [mx, my] = view.cursor;
              const r = (Tools.get().size / 2) * view.scale * dpr;
              let best = 0;
              for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) {
                const d = ctx.getImageData(
                  Math.round(mx * dpr + r + dx), Math.round(my * dpr + dy), 1, 1
                ).data;
                best = Math.max(best, d[0] + d[1] + d[2]);
              }
              return best;
            }"""
        )

    assert ring_bright() > 300  # white ring pixel; BG sums to 90

    page.mouse.move(box["x"] + 10, box["y"] + 10)  # follows the cursor
    assert page.evaluate("view.cursor !== null")
    assert ring_bright() > 300

    page.mouse.move(box["x"] - 40, box["y"] - 40)  # off the canvas: hidden
    assert page.evaluate("view.cursor") is None


def test_cursor_tracks_without_flicker_during_commit(page, server):
    """While a stroke commit is in flight the preview stays on screen
    (view.holdStroke) AND the cursor outline keeps tracking the mouse
    (regression: tracking was disabled for the whole round trip, freezing
    the brush outline at the stroke end until the server update arrived)."""
    base, _ = server
    _open(page, base)
    _show(page, base)
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

    def preview_px(x, y):
        return page.evaluate(
            """([x, y]) => {
                const ctx = document.getElementById('canvas').getContext('2d');
                const d = ctx.getImageData(
                    Math.round((view.ox + x * view.scale) * view.dpr),
                    Math.round((view.oy + y * view.scale) * view.dpr), 1, 1).data;
                return [d[0], d[1], d[2]];
            }""",
            [x, y],
        )

    frame = "document.getElementById('canvas').toDataURL()"
    # inside the 250 ms delay: the preview is still on screen
    assert preview_px(100, 40) == [0, 0, 0]
    # ...and the cursor outline keeps tracking: moving the mouse changes
    # the frame (the outline moved) without flashing the preview off
    before = page.evaluate(frame)
    sx, sy = screen_pos(page, 150, 100)
    page.mouse.move(sx, sy, steps=3)
    assert page.evaluate(frame) != before, "cursor outline did not track"
    assert preview_px(100, 40) == [0, 0, 0], "preview flashed off"
    # the commit lands and the pixels stay
    assert wait_until(lambda: canvas_px(base, 100, 40) == (0, 0, 0))
    assert preview_px(100, 40) == [0, 0, 0]


def test_brush_screen_size_mode(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150
    page.keyboard.press("m")  # mask tool: default 30 px
    # zoom 2x (relative to the fit scale) anchored at the holder centre,
    # so the image centre stays put
    s0 = page.evaluate("view.scale")
    page.evaluate("viewZoomAt(2, holderBox().w / 2, holderBox().h / 2)")
    scale = page.evaluate("view.scale")
    assert scale == pytest.approx(2 * s0)

    def stamp_radius():
        ys, xs = np.nonzero(mask_arr(base) > 200)
        return (xs.max() - xs.min()) / 2

    # default: size in canvas px -> full 30 px stamp despite the zoom
    dot(page, 100, 75)
    assert wait_until(
        lambda: mask_arr(base) is not None and int((mask_arr(base) > 200).sum()) > 100)
    assert stamp_radius() == pytest.approx(15, rel=0.1)

    page.click("#mask-clear")
    assert wait_until(lambda: bool((mask_arr(base) == 0).all()))

    # screen mode: the 30 px slider value means 30 SCREEN px -> the canvas
    # footprint shrinks to size/scale
    page.check("#brush-screen")
    assert page.evaluate("brushSizeCanvas()") == pytest.approx(30 / scale)
    dot(page, 100, 75)
    assert wait_until(lambda: int((mask_arr(base) > 200).sum()) > 5)
    assert stamp_radius() == pytest.approx(15 / scale, rel=0.25, abs=1)

    # outline: the ring sits at slider/2 SCREEN px, not canvas px
    box = page.locator("#canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    bright = page.evaluate(
        """() => {
          const ctx = document.getElementById('canvas').getContext('2d');
          const dpr = view.dpr;
          const [mx, my] = view.cursor;
          const r = (Tools.get().size / 2) * dpr; // screen px
          let best = 0;
          for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) {
            const d = ctx.getImageData(
              Math.round(mx * dpr + r + dx), Math.round(my * dpr + dy), 1, 1
            ).data;
            best = Math.max(best, d[0] + d[1] + d[2]);
          }
          return best;
        }"""
    )
    assert bright > 300  # white ring pixel; BG sums to 90


# ---------------- clone tool ----------------

LEFT, RIGHT = (123, 45, 67), (10, 20, 30)


def _two_band(base, w=200, h=150, split=100):
    """Left half LEFT, right half RIGHT (clone source/destination bands)."""
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:, :split] = LEFT
    arr[:, split:] = RIGHT
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/canvas/upload", content=buf.getvalue())
        assert r.status_code == 200, r.text


def _screen_px(page, x, y):
    """The device-pixel on the display canvas at canvas-pixel (x, y)."""
    return page.evaluate(
        """([x, y]) => {
            const c = document.getElementById('canvas');
            const dx = Math.round((view.ox + x * view.scale) * view.dpr);
            const dy = Math.round((view.oy + y * view.scale) * view.dpr);
            const d = c.getContext('2d').getImageData(dx, dy, 1, 1).data;
            return [d[0], d[1], d[2]];
        }""",
        [x, y],
    )


def test_clone_toolbar_button_and_shortcut(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    assert page.is_visible("#tool-clone")
    # keyboard shortcut C selects the clone tool...
    page.keyboard.press("c")
    assert page.evaluate(
        "document.getElementById('tool-clone').classList.contains('active')")
    assert page.evaluate(
        "!document.getElementById('tool-brush').classList.contains('active')")
    # ...and the colour swatch is disabled (the clone tool has no colour)
    assert page.evaluate("document.getElementById('brush-color').disabled")
    # the brush outline follows the cursor like the other tools
    cx, cy = _centre(page)
    page.mouse.move(cx, cy)
    bright = page.evaluate(
        """() => {
            const ctx = document.getElementById('canvas').getContext('2d');
            const dpr = view.dpr;
            const [mx, my] = view.cursor;
            const r = (Tools.get().size / 2) * view.scale * dpr;
            let best = 0;
            for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) {
                const d = ctx.getImageData(
                    Math.round(mx * dpr + r + dx), Math.round(my * dpr + dy), 1, 1
                ).data;
                best = Math.max(best, d[0] + d[1] + d[2]);
            }
            return best;
        }"""
    )
    assert bright > 300  # white ring pixel; BG sums to 90
    # the toolbar button works too
    page.click("#tool-brush")
    assert page.evaluate(
        "document.getElementById('tool-brush').classList.contains('active')")
    page.click("#tool-clone")
    assert page.evaluate(
        "document.getElementById('tool-clone').classList.contains('active')")


def test_clone_shift_click_picks_source(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    page.keyboard.press("c")
    # shift+click picks the source; it is NOT a stroke (nothing committed)
    dot(page, 40, 75, shift=True)
    assert page.evaluate("view.cloneSource") == pytest.approx([40, 75], abs=0.5)
    assert canvas_px(base, 40, 75) == BG
    # the crosshair is rendered at the source point
    bright = page.evaluate(
        """() => {
            const ctx = document.getElementById('canvas').getContext('2d');
            const dpr = view.dpr;
            const x = Math.round((view.ox + view.cloneSource[0] * view.scale) * dpr);
            const y = Math.round((view.oy + view.cloneSource[1] * view.scale) * dpr);
            let best = 0;
            for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) {
                const d = ctx.getImageData(x + dx, y + dy, 1, 1).data;
                best = Math.max(best, d[0] + d[1] + d[2]);
            }
            return best;
        }"""
    )
    assert bright > 400  # white crosshair; BG sums to 90
    # the brush outline is drawn at the source too (cursor moved away, so
    # the ring under test is the source's, not the cursor's)
    page.mouse.move(*screen_pos(page, 130, 40))
    ring_bright = page.evaluate(
        """() => {
            const ctx = document.getElementById('canvas').getContext('2d');
            const dpr = view.dpr;
            const sx = (view.ox + view.cloneSource[0] * view.scale) * dpr;
            const sy = (view.oy + view.cloneSource[1] * view.scale) * dpr;
            const r = (Tools.get().size / 2) * view.scale * dpr;
            let best = 0;
            for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) {
                const d = ctx.getImageData(
                    Math.round(sx + r + dx), Math.round(sy + dy), 1, 1
                ).data;
                best = Math.max(best, d[0] + d[1] + d[2]);
            }
            return best;
        }"""
    )
    assert ring_bright > 300  # ring at the source point
    # shift+click again moves the source
    dot(page, 55, 80, shift=True)
    assert page.evaluate("view.cloneSource") == pytest.approx([55, 80], abs=0.5)
    assert canvas_px(base, 55, 80) == BG


def test_clone_drag_without_source_shows_hint(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    page.keyboard.press("c")
    # plain drag without a picked source: hint shown, nothing committed
    stroke(page, [(40, 75), (60, 75)])
    assert "shift-click" in page.text_content("#status-line").lower()
    assert canvas_px(base, 60, 75) == BG
    assert page.evaluate("view.stroke") is None


def test_clone_drag_copies_pixels(page, server):
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("c")
    dot(page, 30, 75, shift=True)  # source S in the left band
    assert page.evaluate("view.cloneSource") == pytest.approx([30, 75], abs=0.5)

    # drag on the right band: the offset S-D0 = (-110, 0) maps the cursor
    # back into the left band, so the stroke must paint the LEFT colour.
    # The preview is live: already on screen BEFORE mouse.up commits it.
    page.mouse.move(*screen_pos(page, 140, 75))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 160, 75), steps=5)
    assert _screen_px(page, 150, 75) == list(LEFT), \
        f"live preview under cursor: {_screen_px(page, 150, 75)}"
    page.mouse.up()
    assert wait_until(lambda: canvas_px(base, 150, 75) == LEFT), \
        f"clone centre: {canvas_px(base, 150, 75)}"
    assert canvas_px(base, 10, 10) == LEFT  # left band untouched
    assert canvas_px(base, 190, 140) == RIGHT  # right band corner untouched
    assert canvas_px(base, 30, 75) == LEFT  # source region untouched


def test_clone_opacity_is_not_stacked(page, server):
    # a continuous stroke at <1 opacity must paint the copied content at
    # that opacity: overlapping stamps do NOT accumulate (the stroke core
    # stays at the stroke's opacity, in the live preview AND the commit)
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("c")
    set_input(page, "#brush-opacity", "0.4")
    dot(page, 30, 75, shift=True)
    page.mouse.move(*screen_pos(page, 140, 75))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 160, 75), steps=5)
    exp = [0.4 * c + 0.6 * b for c, b in zip(LEFT, RIGHT)]
    got = _screen_px(page, 150, 75)
    assert all(abs(g - e) <= 1 for g, e in zip(got, exp)), \
        f"live preview stacked: {got}, want ~{exp}"
    page.mouse.up()
    assert wait_until(
        lambda: all(abs(int(g) - e) <= 1
                    for g, e in zip(canvas_px(base, 150, 75), exp))), \
        f"committed stroke stacked: {canvas_px(base, 150, 75)}, want ~{exp}"


def test_clone_ctrl_moves_source_relative(page, server):
    # CTRL gesture: with a source S set, holding CTRL makes the source
    # follow the cursor 1:1, at the relative offset frozen when CTRL went
    # down. Strokes pressed during the gesture use the moved source, and
    # the source reverts to S when CTRL is released (after the stroke).
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("c")
    dot(page, 30, 75, shift=True)  # source S

    page.mouse.move(*screen_pos(page, 140, 75))
    page.keyboard.down("Control")
    assert page.evaluate("view.cloneGesture") == pytest.approx([-110, 0])
    # moving the cursor moves the source 1:1 with it
    page.mouse.move(*screen_pos(page, 150, 75))
    assert page.evaluate("cloneSourcePoint()") == pytest.approx([40, 75], abs=0.5)
    # pressing now draws with the moved source: (150..170) <- (40..60) LEFT
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 170, 75), steps=5)
    assert _screen_px(page, 160, 75) == list(LEFT), "live preview (moved source)"
    page.mouse.up()
    # CTRL still held: the gesture continues; the next click stamps at the
    # same relative offset: (180..190) <- (70..80) LEFT
    page.mouse.move(*screen_pos(page, 180, 75))
    assert page.evaluate("cloneSourcePoint()") == pytest.approx([70, 75], abs=0.5)
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 190, 75), steps=3)
    page.mouse.up()
    # releasing CTRL reverts the source to the picked S
    page.keyboard.up("Control")
    assert page.evaluate("view.cloneGesture") is None
    assert page.evaluate("cloneSourcePoint()") == pytest.approx([30, 75], abs=0.5)
    assert canvas_px(base, 160, 75) == LEFT
    assert canvas_px(base, 180, 75) == LEFT


def test_clone_ctrl_release_mid_stroke_reverts_after(page, server):
    # releasing CTRL during a clone stroke only takes effect after the
    # stroke ends: the source reverts at mouseup, not mid-stroke
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("c")
    dot(page, 30, 75, shift=True)
    page.mouse.move(*screen_pos(page, 140, 75))
    page.keyboard.down("Control")
    page.mouse.move(*screen_pos(page, 150, 75))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 170, 75), steps=5)
    page.keyboard.up("Control")  # mid-stroke
    assert page.evaluate("view.cloneGesture") is not None
    assert _screen_px(page, 160, 75) == list(LEFT)
    page.mouse.up()
    assert wait_until(lambda: page.evaluate("view.cloneGesture") is None)
    assert page.evaluate("cloneSourcePoint()") == pytest.approx([30, 75], abs=0.5)


def test_clone_ctrl_without_source_is_inert(page, server):
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("c")
    page.mouse.move(*screen_pos(page, 100, 75))
    page.keyboard.down("Control")
    page.mouse.move(*screen_pos(page, 120, 75))
    assert page.evaluate("view.cloneGesture") is None
    assert page.evaluate("view.cloneSource") is None
    page.keyboard.up("Control")


def test_clone_shift_click_reanchors_ctrl_gesture(page, server):
    # shift-clicking during the gesture re-picks S and lands the moved
    # source on it (the relative offset resets to zero)
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("c")
    dot(page, 30, 75, shift=True)
    page.mouse.move(*screen_pos(page, 140, 75))
    page.keyboard.down("Control")
    dot(page, 40, 75, shift=True)  # new S, cursor on it
    assert page.evaluate("view.cloneSource") == pytest.approx([40, 75], abs=0.5)
    assert page.evaluate("cloneSourcePoint()") == pytest.approx([40, 75], abs=0.5)
    page.keyboard.up("Control")


def test_clone_preview_held_until_server_response(page, server):
    """No flicker on release: the on-screen clone preview must stay visible
    while the commit round trip is in flight (same invariant as strokes)."""
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("c")
    dot(page, 30, 75, shift=True)

    # Delay the clone response so the round-trip window is wide.
    page.evaluate("""() => {
        const orig = window.fetch;
        window.fetch = async (url, opts) => {
            const r = await orig(url, opts);
            if (String(url).includes('/canvas/clone'))
                await new Promise((res) => setTimeout(res, 250));
            return r;
        };
    }""")
    stroke(page, [(140, 75), (160, 75)])  # ends with mouse.up
    # after release the stroke must NOT keep following the cursor while the
    # commit is in flight: move the mouse away and the preview must stay
    # where it was
    page.mouse.move(*screen_pos(page, 170, 90))

    # inside the 250 ms delay: the preview (LEFT over the RIGHT band) must
    # still be on screen, not the un-stroked image underneath
    assert _screen_px(page, 150, 75) == list(LEFT), \
        f"preview lost during round trip: {_screen_px(page, 150, 75)}"
    # ...and it must not have extended after the release
    assert _screen_px(page, 170, 90) == list(RIGHT), \
        f"stroke followed the cursor after release: {_screen_px(page, 170, 90)}"
    # ...and the server commits the stroke
    assert wait_until(lambda: canvas_px(base, 150, 75) == LEFT)
    # after the refresh the committed pixels are on screen
    assert _screen_px(page, 150, 75) == list(LEFT)


# ---------------- smudge tool ----------------


def test_smudge_toolbar_button_and_shortcut(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    assert page.is_visible("#tool-smudge")
    # keyboard shortcut S selects the smudge tool...
    page.keyboard.press("s")
    assert page.evaluate(
        "document.getElementById('tool-smudge').classList.contains('active')")
    assert page.evaluate(
        "!document.getElementById('tool-brush').classList.contains('active')")
    # ...and the colour swatch is disabled (smudge smears existing pixels)
    assert page.evaluate("document.getElementById('brush-color').disabled")
    # the toolbar button works too
    page.click("#tool-brush")
    assert page.evaluate(
        "document.getElementById('tool-brush').classList.contains('active')")
    page.click("#tool-smudge")
    assert page.evaluate(
        "document.getElementById('tool-smudge').classList.contains('active')")


def test_smudge_single_click_is_noop(page, server):
    # smudge is a drag: a single click has no previous stamp to smear, so
    # nothing is committed
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("s")
    dot(page, 140, 75)
    assert canvas_px(base, 140, 75) == RIGHT
    assert page.evaluate("view.stroke") is None


def test_smudge_drag_smears(page, server):
    # a continuous smear carries the LEFT band along the drag: each stamp
    # drags what the previous one moved, so the white front (x=100) ends
    # up past the drag endpoint (x=140). A smear in the WRONG direction
    # (sampling ahead of the motion) would show RIGHT where LEFT is
    # expected. The preview is live: asserted BEFORE mouse.up commits.
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("s")

    page.mouse.move(*screen_pos(page, 60, 75))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 100, 75), steps=5)
    page.mouse.move(*screen_pos(page, 140, 75), steps=5)
    assert _screen_px(page, 130, 75) == list(LEFT), \
        f"live preview under cursor: {_screen_px(page, 130, 75)}"
    page.mouse.up()
    assert wait_until(lambda: canvas_px(base, 130, 75) == LEFT), \
        f"smudge trail: {canvas_px(base, 130, 75)}"
    assert canvas_px(base, 10, 10) == LEFT  # left band untouched
    assert canvas_px(base, 148, 75) == RIGHT  # ahead of the trail: untouched
    assert canvas_px(base, 190, 140) == RIGHT  # right band corner untouched


def test_smudge_preview_held_until_server_response(page, server):
    """No flicker on release: the on-screen smudge preview must stay visible
    while the commit round trip is in flight (same invariant as strokes)."""
    base, _ = server
    _open(page, base)
    _two_band(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    page.keyboard.press("s")

    # Delay the smudge response so the round-trip window is wide.
    page.evaluate("""() => {
        const orig = window.fetch;
        window.fetch = async (url, opts) => {
            const r = await orig(url, opts);
            if (String(url).includes('/canvas/smudge'))
                await new Promise((res) => setTimeout(res, 250));
            return r;
        };
    }""")
    stroke(page, [(40, 75), (90, 75)])  # ends with mouse.up
    # inside the 250 ms delay: the smear preview (LEFT out in the RIGHT
    # band) must still be on screen, not the un-stroked image underneath
    assert _screen_px(page, 80, 75) == list(LEFT), \
        f"preview lost during round trip: {_screen_px(page, 80, 75)}"
    # ...and the server commits the stroke
    assert wait_until(lambda: canvas_px(base, 80, 75) == LEFT)
    # after the refresh the committed pixels are on screen
    assert _screen_px(page, 80, 75) == list(LEFT)


# ---------------- state persistence ----------------


def test_smudge_state_persists_across_reload(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    page.keyboard.press("s")  # smudge tool
    set_input(page, "#brush-size", "0.342")  # 32 px (log scale)
    set_input(page, "#brush-opacity", "0.4")

    page.reload()
    page.wait_for_function(
        "document.getElementById('tool-smudge').classList.contains('active')")
    assert page.evaluate("Tools.get().size") == 32
    assert page.evaluate("Tools.get().opacity") == pytest.approx(0.4, abs=0.01)
    # per-tool memory: switching back to brush restores its own settings
    page.keyboard.press("d")
    assert page.evaluate("Tools.get().size") == pytest.approx(10, abs=0.5)


def test_tool_state_persists_across_reload(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    page.keyboard.press("m")  # mask tool
    set_input(page, "#brush-size", "0.342")  # 32 px (log scale)
    set_input(page, "#brush-opacity", "0.4")

    page.reload()
    page.wait_for_function(
        "document.getElementById('tool-mask').classList.contains('active')")
    assert page.evaluate("Tools.get().size") == 32
    assert page.evaluate("Tools.get().opacity") == pytest.approx(0.4, abs=0.01)
    # per-tool memory: switching back to brush restores its own settings
    page.keyboard.press("d")
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
    (Path(root) / "user" / "brushes" / "soft24.png").write_bytes(buf.getvalue())

    _open(page, base)
    _show(page, base, 200, 150)
    page.wait_for_function("brushNames.includes('soft24')")
    page.select_option("#brush", "soft24")
    set_input(page, "#brush-color", "#00ff00")
    set_input(page, "#brush-size", "0.408")  # = 40 px

    # single stamp at the centre: strong in the middle, soft near the rim
    dot(page, 100, 75)
    assert wait_until(lambda: canvas_px(base, 100, 75)[1] > 150)
    centre = canvas_px(base, 100, 75)
    edge = canvas_px(base, 100, 93)  # 18 px out: inside a 40px stamp, near its rim
    assert edge[1] < centre[1], f"no falloff: centre={centre} edge={edge}"


def test_soft_brush_preview_respects_shape(page, server):
    """While stroking with a custom (soft) brush the LIVE preview must show
    the brush's falloff, not a hard disc (the preview used to stamp a disc
    for every brush, which differed strongly from the server commit)."""
    base, root = server
    yy, xx = np.mgrid[0:24, 0:24]
    d = np.hypot(xx - 11.5, yy - 11.5)
    a = (np.clip(1 - d / 12.0, 0, 1) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    (Path(root) / "user" / "brushes" / "soft24.png").write_bytes(buf.getvalue())

    _open(page, base)
    _show(page, base, 200, 150)
    page.wait_for_function("brushNames.includes('soft24')")
    page.wait_for_function("brushImages['soft24'] && brushImages['soft24'].ok")
    page.select_option("#brush", "soft24")
    set_input(page, "#brush-color", "#ffffff")
    set_input(page, "#brush-size", "0.408")  # = 40 px

    def preview_px(x, y):
        # getImageData uses the element's backing store (no viewport offset)
        return page.evaluate(
            """([x, y]) => {
                const ctx = document.getElementById('canvas').getContext('2d');
                const px = Math.round((view.ox + x * view.scale) * view.dpr);
                const py = Math.round((view.oy + y * view.scale) * view.dpr);
                const d = ctx.getImageData(px, py, 1, 1).data;
                return [d[0], d[1], d[2]];
            }""",
            [x, y],
        )

    # start a stroke and HOLD it: the preview is on screen, the server has
    # not committed yet
    sx, sy = screen_pos(page, 100, 75)
    page.mouse.move(sx, sy)
    page.mouse.down()
    ex, ey = screen_pos(page, 104, 75)
    page.mouse.move(ex, ey, steps=3)

    centre = preview_px(100, 75)
    assert centre[0] > 200, f"centre should be near white: {centre}"
    # 18 px out: the soft brush's alpha is ~10% here; a hard disc would be
    # full white (255)
    rim = preview_px(100, 93)
    assert BG[0] < rim[0] < 150, \
        f"preview rim should show the soft falloff, not a hard disc: {rim}"

    page.mouse.up()
    assert wait_until(lambda: canvas_px(base, 100, 75)[0] > 200)
    # the commit matches the preview: soft at the rim, not a hard disc
    rim_s = canvas_px(base, 100, 93)
    assert rim_s[0] < 150, f"server stamp not soft either: {rim_s}"


# ---------------- generation modes ----------------


def test_generation_mode_denoise_ui(page, server):
    base, root = server
    _open(page, base)

    # denoise is only for the inpaint modes; mask padding only for
    # "inpaint masked" (it is the context ring around the mask)
    assert page.locator("#denoise").is_disabled()
    assert page.locator("#mask-padding").is_disabled()
    page.select_option("#mode", "inpaint_masked")
    assert not page.locator("#denoise").is_disabled()
    assert not page.locator("#mask-padding").is_disabled()
    set_input(page, "#denoise", "0.42")
    set_input(page, "#mask-padding", "64")
    page.select_option("#mode", "inpaint_not_masked")
    assert page.locator("#denoise").is_disabled() is False
    assert page.locator("#mask-padding").is_disabled()

    # denoise/mask-padding are persisted across a reload; the mode is not
    # and always comes back as "generate" (nothing to inpaint until the
    # user has generated an image first)
    page.evaluate("saveUiState()")
    page.reload()
    page.wait_for_function("brushNames.length >= 1")
    assert page.locator("#mode").input_value() == "generate"
    assert page.locator("#denoise").input_value() == "0.42"
    assert page.locator("#mask-padding").input_value() == "64"
    assert page.locator("#denoise").is_disabled()
    assert page.locator("#mask-padding").is_disabled()


def test_generate_source_thumbnail_retains_canvas(page, server):
    """A generate on a non-empty canvas keeps the displayed image until a
    result is selected; the source thumbnail (green border, highlighted)
    is how the user gets back to it."""
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)

    page.click("#btn-generate")

    # the source arrives before (or with) the first result
    page.wait_for_function("state.thumbs.length >= 2")
    first = page.locator("#thumbs img").first
    assert "source" in (first.get_attribute("class") or "")
    assert first.get_attribute("title") == "source image (click to restore)"
    # the canvas is showing the source, so IT is the selected thumbnail
    assert "selected" in (first.get_attribute("class") or "")
    # ...and the canvas was NOT auto-replaced
    assert canvas_px(base, 100, 75) == BG

    # selecting a result puts it on the canvas
    page.locator("#thumbs img").nth(1).click()
    assert wait_until(lambda: canvas_px(base, 100, 75) != BG)

    # clicking the source thumbnail restores the original image
    first.click()
    assert wait_until(lambda: canvas_px(base, 100, 75) == BG)


def test_inpaint_source_thumbnail(page, server):
    """An inpaint generation immediately shows a source thumbnail (green
    border); the canvas is retained until a result is selected, and
    clicking the source thumbnail restores the source image."""
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)
    # paint a mask dot in the centre
    page.click("#tool-mask")
    dot(page, 100, 75)
    assert wait_until(lambda: mask_arr(base) is not None and mask_arr(base).any())

    page.select_option("#mode", "inpaint_masked")
    set_input(page, "#denoise", "0.6")
    page.click("#btn-generate")

    # the source arrives before (or with) the first result, and it is
    # the selected thumbnail (the canvas shows the source, no auto-set)
    page.wait_for_function("state.thumbs.length >= 2")
    first = page.locator("#thumbs img").first
    assert "source" in (first.get_attribute("class") or "")
    assert "selected" in (first.get_attribute("class") or "")
    assert first.get_attribute("title") == "source image (click to restore)"
    assert canvas_px(base, 100, 75) == BG

    # selecting a result regenerates the masked centre... clicking the
    # source thumbnail brings the original image back
    page.locator("#thumbs img").nth(1).click()
    assert wait_until(lambda: canvas_px(base, 100, 75) != BG), \
        f"centre after result select: {canvas_px(base, 100, 75)}"
    first.click()
    assert wait_until(lambda: canvas_px(base, 100, 75) == BG), \
        f"centre after source restore: {canvas_px(base, 100, 75)}"


def test_inpaint_without_mask_shows_server_error(page, server):
    base, root = server
    _open(page, base)
    _show(page, base, 200, 150)
    # paint one dot so the canvas is non-empty (default tool: brush)
    dot(page, 100, 75)
    assert wait_until(lambda: canvas_px(base, 100, 75) != BG)

    # inpaint masked without a mask: the server rejects, and the message
    # (not a bare status code) shows up in the status line
    page.select_option("#mode", "inpaint_masked")
    page.click("#btn-generate")
    page.wait_for_function(
        "document.getElementById('status-line').textContent.includes('mask is empty')",
        timeout=15000,
    )


def test_progress_bar_reserves_space(page, server):
    """The progress area keeps its height at all times: the thumbnail strip
    (which sits below it) must not move when generation starts/stops."""
    base, root = server
    _open(page, base)
    _show(page, base, 200, 150)

    # a few thumbnails so the strip exists below the progress area
    page.evaluate(
        """async () => {
            await fetch('/api/generate', {method: 'POST',
              headers: {'content-type': 'application/json'},
              body: JSON.stringify({prompt: 't', width: 64, height: 64,
                                    batch_size: 3})});
        }"""
    )
    page.wait_for_function(
        "document.getElementById('thumbs').querySelectorAll('img').length >= 3",
        timeout=15000,
    )

    def thumbs_top() -> float:
        return page.evaluate(
            "() => document.getElementById('thumbs')"
            ".getBoundingClientRect().top"
        )

    top_idle = thumbs_top()
    page.evaluate("setGenerating(true)")
    page.evaluate(
        "() => { document.getElementById('progress-label')"
        ".textContent = 'sampling 12/25 steps'; }"
    )
    top_on = thumbs_top()
    page.evaluate("setGenerating(false)")  # back to the idle empty track
    top_off = thumbs_top()
    assert top_idle == top_on == top_off, (
        f"sidebar reflowed: idle={top_idle} generating={top_on} off={top_off}"
    )


# ---------------- PNG drag&drop ----------------


def _canvas_size(base):
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.get("/api/canvas.png")
        if r.status_code == 404:
            return None
        return Image.open(io.BytesIO(r.content)).size


def _meta_png(w=120, h=90, rgb=(9, 10, 11), prompt="the meta prompt"):
    """PNG bytes with an A1111-style parameters chunk."""
    import base64

    from imagagen import metadata as md

    img, info = md.attach_metadata(
        Image.new("RGB", (w, h), rgb),
        md.build_parameters(prompt, "the negative", 17, "Euler a", "Karras",
                            6.5, 7, w, h, "some_model", "abc123"),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _drop_png(page, selector, png: bytes, name="drop.png"):
    """Simulate a file drop of `png` onto the given element."""
    import base64

    page.evaluate(
        """async ([selector, b64, name]) => {
            const el = document.querySelector(selector);
            const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            const dt = new DataTransfer();
            dt.items.add(new File([bytes], name, {type: "image/png"}));
            el.dispatchEvent(new DragEvent("dragover",
                {dataTransfer: dt, bubbles: true, cancelable: true}));
            el.dispatchEvent(new DragEvent("drop",
                {dataTransfer: dt, bubbles: true, cancelable: true}));
        }""",
        [selector, base64.b64encode(png).decode(), name],
    )


def test_drop_png_on_canvas_sets_image_only(page, server):
    base, _ = server
    _open(page, base)
    page.fill("#prompt", "existing prompt")
    page.fill("#negative", "existing negative")

    _drop_png(page, "#canvas-holder", _meta_png(120, 90, rgb=BG))
    assert wait_until(lambda: _canvas_size(base) == (120, 90))
    assert canvas_px(base, 1, 1) == BG
    # the image is rendered, the drop highlight is gone... (the client
    # re-render is async after the upload commit, so wait for it)
    assert page.wait_for_function(
        "document.getElementById('canvas').hidden === false")
    assert not page.locator("#canvas-holder").evaluate(
        "el => el.classList.contains('drop')")
    # ...and NO metadata was applied to the controls
    assert page.input_value("#prompt") == "existing prompt"
    assert page.input_value("#negative") == "existing negative"
    assert page.input_value("#width") != "120"


def test_drop_png_on_prompt_applies_metadata_and_sets_canvas(page, server):
    base, _ = server
    _open(page, base)

    _drop_png(page, "#prompt", _meta_png(120, 90))
    assert wait_until(lambda: _canvas_size(base) == (120, 90))
    assert wait_until(
        lambda: page.input_value("#prompt") == "the meta prompt")
    assert page.input_value("#negative") == "the negative"
    assert page.input_value("#steps") == "17"
    assert page.input_value("#cfg") == "6.5"
    assert page.input_value("#seed") == "7"
    assert page.input_value("#width") == "120"
    assert page.input_value("#height") == "90"
    assert page.input_value("#sampler") == "Euler a"
    assert page.input_value("#scheduler") == "Karras"


def test_drop_png_without_metadata_only_sets_size(page, server):
    base, _ = server
    _open(page, base)
    page.fill("#prompt", "keep me")

    buf = io.BytesIO()
    Image.new("RGB", (112, 88), BG).save(buf, format="PNG")
    _drop_png(page, "#negative", buf.getvalue())
    assert wait_until(lambda: _canvas_size(base) == (112, 88))
    assert page.input_value("#width") == "112"
    assert page.input_value("#height") == "88"
    assert page.input_value("#prompt") == "keep me"


def _lora_png(model_name: str, loras: list[tuple[str, float]]) -> bytes:
    """PNG whose parameters chunk carries the given model name + LoRAs."""
    from imagagen import metadata as md

    img, info = md.attach_metadata(
        Image.new("RGB", (64, 64), BG),
        md.build_parameters("lora test", "", 20, "Euler a", "Karras", 7.0, 1,
                            64, 64, model_name, "abc123", loras=loras),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def test_drop_png_loras_skip_the_model_config_loras(page, server):
    """The model configuration's LoRAs are always prepended by the backend
    and therefore travel in the PNG metadata; on import they must not be
    re-added to the UI selection (they would load twice). The config to
    compare against is the one the import ends up selecting (the image's
    model if a configuration matches it, else the currently selected
    one)."""
    base, root = server
    (root / "loras" / "sdxl").mkdir(parents=True)
    (root / "loras" / "sdxl" / "dmd2.safetensors").write_bytes(b"")
    (root / "loras" / "sdxl" / "detail.safetensors").write_bytes(b"")
    # named like the shipped configs: same name as the model, so the
    # import's model match lands on this configuration
    cfg = {
        "name": "Mock Model",
        "model": "mock/mock.safetensors",
        "t2i": {"loras": [{"name": "dmd2", "weight": 0.7}]},
        "i2i": {"loras": [{"name": "dmd2", "weight": 0.7}]},
    }
    (root / "user" / "model-configs").mkdir(exist_ok=True)
    (root / "user" / "model-configs" / "mock-model.json").write_text(
        json.dumps(cfg), "utf-8")
    _open(page, base)
    combo_pick(page, page.locator("#model"), "mock-model")
    assert wait_until(lambda: page.evaluate("currentEntry()?.name") == "Mock Model")

    # the image was made with the config LoRA (dmd2@0.7, prepended) plus a
    # user-selected LoRA (detail@1.0): only the user's survives the import
    _drop_png(page, "#prompt", _lora_png("Mock Model", [("dmd2", 0.7), ("detail", 1.0)]))
    assert wait_until(lambda: page.evaluate("getLoras()") == [["detail", 1.0, True]])

    # a user-selected copy of a config LoRA at a DIFFERENT weight stays:
    # the match is on name AND weight
    _drop_png(page, "#prompt", _lora_png("Mock Model", [("dmd2", 0.7), ("dmd2", 0.9)]))
    assert wait_until(lambda: page.evaluate("getLoras()") == [["dmd2", 0.9, True]])

    # image from a model with no configuration here: the filter falls back
    # to the currently selected configuration
    _drop_png(page, "#prompt", _lora_png("Other Model", [("dmd2", 0.7), ("detail", 0.5)]))
    assert wait_until(lambda: page.evaluate("getLoras()") == [["detail", 0.5, True]])

    # an image with no Lora lines clears the previous selection (a prompt
    # that resolves to no LoRAs must not silently keep the user's old picks)
    _drop_png(page, "#prompt", _lora_png("Mock Model", []))
    page.wait_for_timeout(200)
    assert page.evaluate("getLoras()") == []


# ---------------- wheel modifiers (CTRL = size, SHIFT = opacity) ----------------


def _wheel(page, delta_y=0.0, delta_x=0.0, ctrl=False, shift=False):
    """Dispatch a synthetic wheel event on the canvas element."""
    page.evaluate(
        """([dy, dx, ctrl, shift]) => {
            document.getElementById('canvas').dispatchEvent(new WheelEvent('wheel', {
                bubbles: true, cancelable: true,
                deltaY: dy, deltaX: dx, ctrlKey: ctrl, shiftKey: shift,
            }));
        }""",
        [delta_y, delta_x, ctrl, shift],
    )


def test_ctrl_wheel_changes_brush_size(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    x, y = _centre(page)
    page.mouse.move(x, y)

    page.keyboard.press("d")  # brush tool: the sliders edit its state
    before = page.evaluate("Tools.get().size")
    scale_before = page.evaluate("view.scale")
    _wheel(page, delta_y=-120, ctrl=True)
    size_up = page.evaluate("Tools.get().size")
    _wheel(page, delta_y=120, ctrl=True)
    size_down = page.evaluate("Tools.get().size")
    assert size_up > before and size_down <= before
    # the slider + readout followed the change
    assert page.evaluate("document.getElementById('brush-size-val').textContent") \
        == str(size_down)
    # a plain wheel still zooms (and the ctrl/shift ones did not)
    assert page.evaluate("view.scale") == pytest.approx(scale_before)
    _wheel(page, delta_y=-120)
    assert page.evaluate("view.scale") > scale_before
    # size is clamped to the slider range (10..300)
    for _ in range(200):
        _wheel(page, delta_y=-120, ctrl=True)
    assert page.evaluate("Tools.get().size") == 300
    for _ in range(200):
        _wheel(page, delta_y=120, ctrl=True)
    assert page.evaluate("Tools.get().size") == 10


def test_shift_wheel_changes_opacity(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    x, y = _centre(page)
    page.mouse.move(x, y)

    set_input(page, "#brush-opacity", "0.5")
    _wheel(page, delta_y=-120, shift=True)
    op_up = page.evaluate("Tools.get().opacity")
    assert op_up == pytest.approx(0.6, abs=0.011)
    _wheel(page, delta_y=120, shift=True)
    assert page.evaluate("Tools.get().opacity") == pytest.approx(0.5, abs=0.011)
    # browsers deliver a shifted vertical wheel as HORIZONTAL delta: the
    # handler must fall back to deltaX
    _wheel(page, delta_x=-120, shift=True)
    assert page.evaluate("Tools.get().opacity") == pytest.approx(0.6, abs=0.011)
    # clamped to the slider range (0.05..1)
    for _ in range(200):
        _wheel(page, delta_y=-120, shift=True)
    assert page.evaluate("Tools.get().opacity") == 1
    for _ in range(300):
        _wheel(page, delta_y=120, shift=True)
    assert page.evaluate("Tools.get().opacity") == 0.05
