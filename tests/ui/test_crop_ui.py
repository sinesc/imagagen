"""Browser tests for the crop tool (Chromium via Playwright).

Covers the parts that can only be verified against a real browser: toolbar
wiring + control disabling, drag-to-crop with live preview, Esc abort,
out-of-bounds snapping, and the thumbnail behaviour (pre-crop image becomes
a restorable thumbnail, the crop becomes the immediately selected one).
"""

import io

import httpx
from conftest import canvas_px, screen_pos, set_canvas, wait_until
from PIL import Image

BG = (20, 30, 40)


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-crop")
    page.wait_for_function("brushNames.length >= 1")


def _show(page, base, w=200, h=150):
    """Put an image on the server canvas and make the UI render it."""
    set_canvas(base, w, h, BG)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")


def _canvas_size(base):
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.get("/api/canvas.png")
        if r.status_code == 404:
            return None
        return Image.open(io.BytesIO(r.content)).size


def _screen_rgb(page, x, y):
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


def test_crop_toolbar_button_and_shortcut(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    # keyboard shortcut P selects the crop tool
    page.keyboard.press("p")
    assert page.evaluate("document.getElementById('tool-crop').classList.contains('active')")
    assert page.evaluate("!document.getElementById('tool-brush').classList.contains('active')")

    # colour, size, screen and opacity are all disabled while crop is active
    for sel in ("#brush-color", "#brush-size", "#brush-screen", "#brush-opacity"):
        assert page.evaluate(f"document.querySelector('{sel}').disabled"), f"{sel} not disabled"

    # switching back re-enables them
    page.keyboard.press("d")
    assert page.evaluate("document.getElementById('tool-brush').classList.contains('active')")
    for sel in ("#brush-size", "#brush-screen", "#brush-opacity"):
        assert not page.evaluate(f"document.querySelector('{sel}').disabled"), (
            f"{sel} still disabled"
        )

    # the toolbar button works too
    page.click("#tool-crop")
    assert page.evaluate("document.getElementById('tool-crop').classList.contains('active')")


def test_crop_drag_previews_then_crops(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150
    page.click("#tool-crop")

    # start a drag from (50,30) to (150,100)
    page.mouse.move(*screen_pos(page, 50, 30))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 150, 100), steps=5)

    # live preview BEFORE release: inside the rect stays at BG, outside is dimmed
    inside = _screen_rgb(page, 100, 65)
    outside = _screen_rgb(page, 20, 20)
    assert all(abs(a - b) <= 2 for a, b in zip(inside, BG, strict=True)), (
        f"inside not undimmed: {inside}"
    )
    assert sum(outside) < sum(inside) - 20, f"outside should be dimmed: {outside} vs {inside}"

    page.mouse.up()

    # the crop commits: the canvas is now the 100x70 rectangle
    assert wait_until(lambda: _canvas_size(base) == (100, 70))

    # the pre-crop image became a thumbnail, the crop is a new, selected one
    assert wait_until(lambda: page.locator("#thumbs img").count() == 2)
    imgs = page.locator("#thumbs img")
    assert "selected" not in (imgs.nth(0).get_attribute("class") or "")
    # the title carries the dimensions once the image decoded: poll it, the
    # dimension-free title is a transient pre-decode state
    assert wait_until(lambda: imgs.nth(0).get_attribute("title")
                      == "crop 200\u00d7150 (click to restore)")
    assert "selected" in (imgs.nth(1).get_attribute("class") or "")

    # clicking the pre-crop thumbnail restores the full image
    imgs.nth(0).click()
    assert wait_until(lambda: _canvas_size(base) == (200, 150))
    assert canvas_px(base, 100, 75) == BG

    # ...and the crop thumbnail brings the crop back
    page.locator("#thumbs img").nth(1).click()
    assert wait_until(lambda: _canvas_size(base) == (100, 70))


def test_crop_snaps_to_image_bounds(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150, centred with margin around it
    # zoom out so the drag can start OUTSIDE the image but inside the canvas
    page.evaluate("viewZoomAt(0.5, holderBox().w / 2, holderBox().h / 2)")
    page.evaluate("viewZoomAt(0.5, holderBox().w / 2, holderBox().h / 2)")
    page.click("#tool-crop")

    # start the drag OUTSIDE the image (top-left) and end inside it: the
    # committed region must be the intersection, snapped to the image
    page.mouse.move(*screen_pos(page, -30, -20))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 120, 90), steps=5)
    page.mouse.up()

    assert wait_until(lambda: _canvas_size(base) == (120, 90))


def test_crop_esc_aborts_selection(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    page.click("#tool-crop")

    page.mouse.move(*screen_pos(page, 50, 30))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 150, 100), steps=5)
    assert page.evaluate("view.crop") is not None

    page.keyboard.press("Escape")
    assert page.evaluate("view.crop") is None

    # releasing after the abort commits nothing
    page.mouse.up()
    assert _canvas_size(base) == (200, 150)
    assert page.locator("#thumbs img").count() == 0
    assert canvas_px(base, 100, 75) == BG


def _approx_rect(r, w, h):
    """cropRect() live values carry sub-pixel jitter from the screen->canvas
    coordinate round-trip, so compare with a small tolerance."""
    return r is not None and abs(r["w"] - w) < 0.5 and abs(r["h"] - h) < 0.5


def test_crop_shift_locks_aspect_to_image(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150 -> image aspect 4:3
    page.click("#tool-crop")

    # SHIFT held for the whole drag: the rect locks to the image aspect.
    # Raw pointer rect would be 100x70; the locked rect is 100x75 (4:3).
    page.keyboard.down("Shift")
    page.mouse.move(*screen_pos(page, 50, 30))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 150, 100), steps=5)
    r = page.evaluate("cropRect()")
    assert _approx_rect(r, 100, 75), f"not aspect-locked: {r}"
    page.mouse.up()

    assert wait_until(lambda: _canvas_size(base) == (100, 75))
    page.keyboard.up("Shift")


def test_crop_shift_lock_applies_mid_drag(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150 -> aspect 4:3
    page.click("#tool-crop")

    # free-form drag first, press SHIFT mid-drag: the rect re-locks; release
    # SHIFT and the rect goes back to free-form resizing
    page.mouse.move(*screen_pos(page, 50, 30))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 150, 100), steps=5)
    r = page.evaluate("cropRect()")
    assert _approx_rect(r, 100, 70), f"free-form drag: {r}"

    page.keyboard.down("Shift")
    r = page.evaluate("cropRect()")
    assert _approx_rect(r, 100, 75), f"mid-drag lock: {r}"

    # vertical-dominant drag while locked: the Y travel drives the width
    page.mouse.move(*screen_pos(page, 150, 130), steps=4)
    r = page.evaluate("cropRect()")
    assert _approx_rect(r, 400 / 3, 100), f"vertical-dominant lock: {r}"

    page.keyboard.up("Shift")
    page.mouse.move(*screen_pos(page, 150, 100), steps=4)
    r = page.evaluate("cropRect()")
    assert _approx_rect(r, 100, 70), f"unlocked after SHIFT up: {r}"

    page.keyboard.press("Escape")
    page.mouse.up()
    assert _canvas_size(base) == (200, 150)


def test_copy_shift_locks_aspect_to_image(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150 -> aspect 4:3
    page.keyboard.press("y")  # copy tool
    assert page.evaluate("document.getElementById('tool-copy').classList.contains('active')")

    page.keyboard.down("Shift")
    page.mouse.move(*screen_pos(page, 50, 30))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 150, 100), steps=5)
    page.mouse.up()
    page.keyboard.up("Shift")

    clip = page.evaluate("view.clip ? [view.clip.w, view.clip.h] : null")
    assert clip == [100, 75], f"copy clip not aspect-locked: {clip}"
    assert _canvas_size(base) == (200, 150)  # copy never crops


def test_crop_shift_lock_freezes_at_image_edge(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)  # 200x150 -> aspect 4:3
    # zoom out so the drag can run past the image edge (like the bounds test)
    page.evaluate("viewZoomAt(0.5, holderBox().w / 2, holderBox().h / 2)")
    page.evaluate("viewZoomAt(0.5, holderBox().w / 2, holderBox().h / 2)")
    page.click("#tool-crop")

    page.keyboard.down("Shift")
    page.mouse.move(*screen_pos(page, 50, 30))
    page.mouse.down()
    # first position whose locked corner crosses BOTH image edges: the rect
    # is scaled down uniformly -> corner pinned at (200, 142.5), 150x112.5
    page.mouse.move(*screen_pos(page, 230, 160), steps=5)
    r1 = page.evaluate("cropRect()")
    assert _approx_rect(r1, 150, 112.5), f"corner not pinned to the edge: {r1}"

    # keep pushing right/down: BOTH dimensions freeze (the old behaviour let
    # the unclamped one keep growing and skewed the ratio)
    page.mouse.move(*screen_pos(page, 260, 175), steps=5)
    r2 = page.evaluate("cropRect()")
    assert _approx_rect(r2, 150, 112.5), f"grew past the edge: {r2} -> {r1}"

    page.keyboard.press("Escape")
    page.mouse.up()
    page.keyboard.up("Shift")
    assert _canvas_size(base) == (200, 150)


def test_crop_single_click_is_noop(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    page.click("#tool-crop")

    # a click without a drag selects an empty region: nothing is cropped
    sx, sy = screen_pos(page, 100, 75)
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.up()

    assert _canvas_size(base) == (200, 150)
    assert page.locator("#thumbs img").count() == 0
