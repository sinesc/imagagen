"""Browser tests for the canvas UI (Chromium via Playwright).

Covers the parts that can only be verified against a real browser:
toolbar wiring, pointer strokes (brush + mask + shift-erase), zoom/pan,
tool-state persistence across reload, custom brush listing.
"""

import io
from pathlib import Path

import httpx
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


# ---------------- generation modes ----------------


def test_generation_mode_denoise_ui(page, server):
    base, root = server
    _open(page, base)

    # denoise is only for the inpaint modes
    assert page.locator("#denoise").is_disabled()
    page.select_option("#mode", "inpaint_masked")
    assert not page.locator("#denoise").is_disabled()
    set_input(page, "#denoise", "0.42")

    # denoise is persisted across a reload; the mode is not and always
    # comes back as "generate" (nothing to inpaint until the user has
    # generated an image first)
    page.evaluate("saveUiState()")
    page.reload()
    page.wait_for_function("brushNames.length >= 1")
    assert page.locator("#mode").input_value() == "generate"
    assert page.locator("#denoise").input_value() == "0.42"
    assert page.locator("#denoise").is_disabled()


def test_inpaint_source_thumbnail(page, server):
    """An inpaint generation immediately shows a source thumbnail (green
    border); clicking it restores the source image."""
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

    # the source arrives before (or with) the first result
    page.wait_for_function("state.thumbs.length >= 2")
    first = page.locator("#thumbs img").first
    assert "source" in (first.get_attribute("class") or "")
    assert first.get_attribute("title") == "inpaint source (click to restore)"

    # the masked centre got regenerated... clicking the source thumbnail
    # brings the original image back
    assert wait_until(lambda: canvas_px(base, 100, 75) != BG), \
        f"centre after inpaint: {canvas_px(base, 100, 75)}"
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
              body: JSON.stringify({prompt: 't', width: 64, height: 48,
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
    # the image is rendered, the drop highlight is gone...
    assert page.evaluate("document.getElementById('canvas').hidden === false")
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
