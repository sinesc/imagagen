"""Browser tests for the thumbnail hover preview: the hovered thumbnail's
full image is overlaid on the canvas box with a 0.25s fade -- purely visual
(no canvas/mask/undo side effects).

Two layouts:
 - match-the-canvas: when the thumbnail has the SAME dimensions as the image
   on the canvas and SHIFT is not held, the preview mirrors the canvas view
   exactly (same area, same zoom, same pan) so the images compare 1:1;
 - scale-to-fit: while SHIFT is held, or when the dimensions differ, the
   whole image is scaled into the box with the padding the zoom-to-fit button
   leaves around the canvas image (viewFit's 0.98 factor), centred."""

from conftest import canvas_px, set_canvas, set_input, stroke, wait_until

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


def _hover(page, base, nth=0):
    page.hover(f"#thumbs img >> nth={nth}")
    assert wait_until(lambda: _opacity(page) > 0.95)
    assert not page.evaluate("document.getElementById('thumb-preview').hidden")


def _state(page):
    return page.evaluate(
        "[thumbPreviewSize, document.getElementById('thumb-preview-img').className]"
    )


def _assert_match(page):
    """The preview image sits exactly where the canvas draws its own image."""
    assert _state(page)[1] == "match"
    g = page.evaluate(
        "({w,h,scale,ox,oy}) => [w*scale, h*scale, ox, oy]",
        page.evaluate("({w: view.w, h: view.h, scale: view.scale, ox: view.ox, oy: view.oy})"),
    )
    cv, ir = _rect(page, "#canvas"), _rect(page, "#thumb-preview-img")
    # +1: the canvas element's 1px border, mirrored by the preview container
    assert abs(ir[0] - (cv[0] + 1 + g[2])) < 0.5
    assert abs(ir[1] - (cv[1] + 1 + g[3])) < 0.5
    assert abs(ir[2] - g[0]) < 0.5
    assert abs(ir[3] - g[1]) < 0.5


def _assert_fit(page, iw, ih):
    """Scale-to-fit: the image element fills the 98% box (the padding the
    zoom-to-fit button leaves around the canvas image), centred; object-fit:
    contain scales the picture down inside it, so nothing is cropped."""
    assert _state(page)[1] == ""
    pv = _rect(page, "#thumb-preview")
    padw, padh = pv[2] - 2, pv[3] - 2  # container padding box (1px border)
    boxw, boxh = padw * 0.98, padh * 0.98
    ir = _rect(page, "#thumb-preview-img")
    assert abs(ir[2] - boxw) < 1 and abs(ir[3] - boxh) < 1
    assert abs(ir[0] - (pv[0] + 1 + (padw - boxw) / 2)) < 1
    assert abs(ir[1] - (pv[1] + 1 + (padh - boxh) / 2)) < 1
    # the picture itself is contained (never upscaled past the box)
    k = min(boxw / iw, boxh / ih)
    assert page.evaluate(
        "([iw, ih, k]) => { const i = document.getElementById('thumb-preview-img');"
        " return i.naturalWidth * k <= i.clientWidth + 1 &&"
        " i.naturalHeight * k <= i.clientHeight + 1; }",
        [iw, ih, k],
    )


def test_thumb_hover_previews_full_image_over_canvas(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")  # source + result

    # idle: overlay not even in the DOM flow
    assert page.evaluate("document.getElementById('thumb-preview').hidden")

    _hover(page, base, 0)

    # the container matches the canvas box exactly, and the image shown is
    # the hovered thumbnail's (full-resolution) URL
    pv, cv = _rect(page, "#thumb-preview"), _rect(page, "#canvas")
    assert all(abs(a - b) < 0.5 for a, b in zip(pv, cv, strict=True))
    assert page.evaluate(
        "([t, u]) => t.endsWith(u)",
        [
            page.evaluate("document.getElementById('thumb-preview-img').src"),
            page.evaluate("state.thumbs[0].url"),
        ],
    )

    # same dimensions as the canvas image and no SHIFT -> the preview mirrors
    # the canvas view: same area, same zoom, same pan
    assert _state(page)[0] == [200, 150]
    _assert_match(page)

    # purely visual: the canvas (image, zoom/pan) and its pixels are
    # untouched, and the overlay never intercepts the pointer
    assert page.evaluate("document.getElementById('canvas').hidden === false")
    assert page.evaluate("view.base !== null")
    assert canvas_px(base, 100, 75) == BG
    assert page.evaluate(
        "getComputedStyle(document.getElementById('thumb-preview')).pointerEvents"
    ) == "none"

    # hovering the NEXT thumbnail swaps the preview image in place
    _hover(page, base, 1)
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


def test_thumb_preview_shift_toggles_scale_to_fit(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")

    _hover(page, base, 0)
    _assert_match(page)

    # SHIFT while hovering -> scale-to-fit, live (the mouse never moves)
    page.keyboard.down("Shift")
    assert wait_until(lambda: _state(page)[1] == "")
    _assert_fit(page, 200, 150)
    # the canvas view itself is untouched by the preview
    assert page.evaluate("view.scale") > 0

    # releasing SHIFT mirrors the canvas view again
    page.keyboard.up("Shift")
    assert wait_until(lambda: _state(page)[1] == "match")
    _assert_match(page)

    # SHIFT held from the start -> scale-to-fit right away
    page.mouse.move(10, 750)
    assert wait_until(lambda: page.evaluate("document.getElementById('thumb-preview').hidden"))
    page.keyboard.down("Shift")
    _hover(page, base, 0)
    _assert_fit(page, 200, 150)
    page.keyboard.up("Shift")


def test_thumb_preview_tracks_the_canvas_zoom(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")
    _hover(page, base, 0)

    # zooming the canvas re-lays out the visible preview: it keeps mirroring
    # the canvas view (the mouse stays on the thumbnail, so the preview of a
    # zoom driven from the JS API stays open)
    page.evaluate("zoomCenter(1.4)")
    assert page.evaluate("view.scale") > 4
    _assert_match(page)

    # panning too
    page.evaluate("view.ox += 37; view.oy -= 21; viewRedraw()")
    _assert_match(page)


def test_thumb_preview_fits_when_dimensions_differ(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)

    # generate at a DIFFERENT size: the 300x100 result thumbnail does not
    # match the 200x150 image on the canvas
    set_input(page, "#width", 300)
    set_input(page, "#height", 100)
    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")
    assert page.evaluate("view.w === 200 && view.h === 150")

    # the result thumbnail (300x100) -> scale-to-fit
    _hover(page, base, 1)
    assert _state(page)[0] == [300, 100]
    _assert_fit(page, 300, 100)

    # the source thumbnail (200x150) matches the canvas -> mirrors the view
    _hover(page, base, 0)
    assert _state(page)[0] == [200, 150]
    _assert_match(page)


def test_thumb_titles_name_the_dimensions(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)

    # the hover preview is not the only way to see what a thumbnail is: its
    # title names the dimensions as well as the seed / kind. They only become
    # known once the file decoded, so the title is rebuilt on load.
    set_input(page, "#width", 300)
    set_input(page, "#height", 100)
    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")

    def titles():
        return page.eval_on_selector_all("#thumbs img", "els => els.map(e => e.title)")

    assert wait_until(lambda: "200\u00d7150" in titles()[0]
                      and "300\u00d7100" in titles()[1])
    assert titles()[0].startswith("source image 200\u00d7150")
    assert titles()[0].endswith("(click to restore)")
    assert f"seed {page.evaluate('state.thumbs[1].seed')} 300\u00d7100" in titles()[1]


def test_thumb_and_preview_track_local_edits(page, server):
    """The thumbnail AND hover preview of the entity on the canvas must
    reflect locally committed strokes: the strip <img> is never re-fetched
    and the browser memory-caches the entity PNG URL anyway, so app.js
    swaps in a blob snapshot of the local view base."""
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)
    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")
    # the source thumbnail (unshifted by the WS event) IS the entity the
    # canvas displays
    assert wait_until(lambda: page.evaluate(
        "currentEntityId === state.thumbs[0].image_id"))

    set_input(page, "#brush-color", "#ff0000")
    stroke(page, [(60, 75), (140, 75)])

    # the current entity's thumbnail was re-pointed at a local blob
    assert wait_until(lambda: page.evaluate(
        "state.thumbs.find(t => t.image_id === currentEntityId)"
        "?.url.startsWith('blob:')"))
    src = page.eval_on_selector("#thumbs img >> nth=0", "el => el.src")
    assert src.startswith("blob:")

    # the hover preview shows the SAME fresh snapshot, including the stroke
    _hover(page, base, 0)
    assert page.eval_on_selector("#thumb-preview-img", "el => el.src") == src
    px = page.evaluate(
        """() => { const im = document.getElementById('thumb-preview-img');
            const c = document.createElement('canvas');
            c.width = im.naturalWidth; c.height = im.naturalHeight;
            const g = c.getContext('2d'); g.drawImage(im, 0, 0);
            return Array.from(g.getImageData(100, 75, 1, 1).data); }""")
    assert px[0] > 200 and px[1] < 60 and px[2] < 60, f"stroke missing: {px}"

    # untouched: another entity's thumbnail keeps its server URL
    assert not page.eval_on_selector(
        "#thumbs img >> nth=1", "el => el.src").startswith("blob:")
