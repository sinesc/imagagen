"""Browser tests for the thumbnail hover preview: the hovered thumbnail's
full image is overlaid on the canvas box (zoom-to-fit, centred) with a
0.25s fade -- purely visual (no canvas/mask/undo side effects)."""

from conftest import canvas_px, set_canvas, wait_until

BG = (20, 30, 40)


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-brush")
    page.wait_for_function("brushNames.length >= 1")


def _show(page, base, w=200, h=150):
    set_canvas(base, w, h, BG)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")


def _rect(page, sel):
    return page.locator(sel).evaluate(
        "el => { const r = el.getBoundingClientRect(); "
        "return [r.left, r.top, r.width, r.height]; }"
    )


def _opacity(page) -> float:
    return float(
        page.evaluate(
            "getComputedStyle(document.getElementById('thumb-preview')).opacity"
        )
    )


def test_thumb_hover_previews_full_image_over_canvas(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")  # source + result

    # idle: overlay not even in the DOM flow
    assert page.evaluate("document.getElementById('thumb-preview').hidden")

    page.hover("#thumbs img >> nth=0")
    # fades in to fully opaque (0.25s transition)
    assert wait_until(lambda: _opacity(page) > 0.95)
    assert not page.evaluate("document.getElementById('thumb-preview').hidden")

    # the container matches the canvas box exactly, and the image shown is
    # the hovered thumbnail's (full-resolution) URL
    pv, cv = _rect(page, "#thumb-preview"), _rect(page, "#canvas")
    assert all(abs(a - b) < 0.5 for a, b in zip(pv, cv))
    assert page.evaluate(
        "([t, u]) => t.endsWith(u)",
        [
            page.evaluate("document.getElementById('thumb-preview-img').src"),
            page.evaluate("state.thumbs[0].url"),
        ],
    )

    # zoom-to-fit: the (square) result is letterboxed, centred inside the
    # (200x150) canvas box -- fits in both axes
    ir = _rect(page, "#thumb-preview-img")
    assert ir[2] <= pv[2] + 1 and ir[3] <= pv[3] + 1
    assert abs((ir[0] + ir[2] / 2) - (pv[0] + pv[2] / 2)) < 1
    assert abs((ir[1] + ir[3] / 2) - (pv[1] + pv[3] / 2)) < 1

    # purely visual: the canvas (image, zoom/pan) and its pixels are
    # untouched, and the overlay never intercepts the pointer
    assert page.evaluate("document.getElementById('canvas').hidden === false")
    assert page.evaluate("view.img !== null")
    assert canvas_px(base, 100, 75) == BG
    assert page.evaluate(
        "getComputedStyle(document.getElementById('thumb-preview')).pointerEvents"
    ) == "none"

    # hovering the NEXT thumbnail swaps the preview image in place
    page.hover("#thumbs img >> nth=1")
    assert page.evaluate(
        "([t, u]) => t.endsWith(u)",
        [
            page.evaluate("document.getElementById('thumb-preview-img').src"),
            page.evaluate("state.thumbs[1].url"),
        ],
    )

    # moving the mouse away fades out, then the element is removed
    page.mouse.move(10, 750)
    assert wait_until(lambda: _opacity(page) < 0.05)
    assert wait_until(
        lambda: page.evaluate("document.getElementById('thumb-preview').hidden")
    )
