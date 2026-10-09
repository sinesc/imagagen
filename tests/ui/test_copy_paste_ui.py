"""Browser tests for copy / paste with a floating selection (Chromium).

Covers what only a real browser can verify: the toolbar wiring (Clone moved
to O, Copy = C, Paste = V, Ctrl+C / Ctrl+V), the copy rectangle, the floating
selection's lifecycle (drop, handles, erase, crop, apply, cancel, undo) and
the rule that the selection is an OVERLAY -- the server's canvas only changes
when it is applied.
"""

import io

import httpx
from conftest import canvas_px, dot, screen_pos, set_canvas, stroke, wait_until
from PIL import Image

BG = (20, 30, 40)
PATCH = (240, 220, 10)  # the copied area: a bright 40x40 square at (10,10)


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-copy")
    page.wait_for_function("brushNames.length >= 1")


def _upload(base, img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    with httpx.Client(base_url=base, timeout=10) as c:
        assert c.post("/api/canvas/upload", content=buf.getvalue()).status_code == 200


def _show(page, base, w=200, h=150):
    """Two-tone image on the server canvas, rendered in the UI."""
    img = Image.new("RGB", (w, h), BG)
    for x in range(10, 50):
        for y in range(10, 50):
            img.putpixel((x, y), PATCH)
    _upload(base, img)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")


def _canvas_size(base):
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.get("/api/canvas.png")
        if r.status_code == 404:
            return None
        return Image.open(io.BytesIO(r.content)).size


def _active(pid):
    return f"document.getElementById('{pid}').classList.contains('active')"


def _copy(page, base, x0=10, y0=10, x1=50, y1=50):
    """Drag the copy rectangle (x0,y0) -> (x1,y1)."""
    page.keyboard.press("c")
    page.mouse.move(*screen_pos(page, x0, y0))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, x1, y1), steps=5)
    page.mouse.up()
    assert wait_until(lambda: page.evaluate("view.clip") is not None)


def _paste(page, x, y):
    page.keyboard.press("v")
    page.mouse.click(*screen_pos(page, x, y))
    assert page.evaluate("view.sel") is not None


def _sel_state(page):
    return page.evaluate(
        "view.sel ? {sw: Math.round(view.sel.sw), sh: Math.round(view.sel.sh),"
        " angle: view.sel.angle} : null"
    )


def test_copy_paste_toolbar_and_hotkeys(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    # Clone moved from C to O; C arms the copy tool, V the paste tool
    page.keyboard.press("o")
    assert page.evaluate(_active("tool-clone"))
    page.keyboard.press("c")
    assert page.evaluate(_active("tool-copy"))
    page.keyboard.press("v")
    assert page.evaluate(_active("tool-paste"))
    page.keyboard.press("p")
    assert page.evaluate(_active("tool-crop"))
    page.keyboard.press("Control+c")
    assert page.evaluate(_active("tool-copy"))
    page.keyboard.press("Control+v")
    assert page.evaluate(_active("tool-paste"))

    # copy / paste use no brush: all four brush controls are disabled
    for sel in ("#brush-color", "#brush-size", "#brush-screen", "#brush-opacity"):
        assert page.evaluate(f"document.querySelector('{sel}').disabled"), f"{sel} not disabled"
    page.keyboard.press("d")
    for sel in ("#brush-size", "#brush-screen", "#brush-opacity"):
        assert not page.evaluate(f"document.querySelector('{sel}').disabled"), f"{sel} disabled"

    # paste has nothing to drop until an area was copied
    assert page.evaluate("document.getElementById('tool-paste').disabled")
    _copy(page, base)
    assert not page.evaluate("document.getElementById('tool-paste').disabled")


def test_copy_stores_the_area_only(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base, 10, 10, 50, 90)

    clip = page.evaluate("view.clip ? {w: view.clip.w, h: view.clip.h} : null")
    assert clip == {"w": 40, "h": 80}
    # copying changes nothing on the canvas (no new entity, no pixels touched)
    assert _canvas_size(base) == (200, 150)
    assert canvas_px(base, 100, 100) == BG
    assert page.evaluate("view.sel") is None
    assert "copied 40" in page.text_content("#status-line")


def test_paste_drop_is_an_overlay_until_applied(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 150, 100)

    sel = _sel_state(page)
    assert sel["sw"] == 40 and sel["sh"] == 40
    # centred on the click
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.cy)]") == [150, 100]
    # the apply / discard pair is there, flanking the selection's centre
    # (#sel-actions itself is a zero-size anchor, so check the buttons)
    assert page.is_visible("#sel-ok") and page.is_visible("#sel-cancel")

    # the labels must not wrap: the buttons shrink-to-fit inside a ZERO-size
    # anchor, so without white-space: nowrap the text breaks after the glyph
    # and overflows the button box (checked with the labels visible)
    fits = lambda: page.evaluate(
        "(() => ['sel-ok', 'sel-cancel'].map((id) => {"
        " const e = document.getElementById(id);"
        " return [e.scrollWidth <= e.clientWidth, e.scrollHeight <= e.clientHeight]; }))()"
    )

    # the pair drops its labels when it would cover the selection's handles
    # (a 40 px selection at this zoom), and shows them on a wide one
    word_display = lambda: page.evaluate(
        "getComputedStyle(document.querySelector('#sel-ok .sel-word')).display"
    )
    assert page.evaluate("document.getElementById('sel-actions').classList.contains('compact')")
    assert word_display() == "none"
    page.evaluate("view.sel.sw = 300; viewRedraw()")
    assert not page.evaluate("document.getElementById('sel-actions').classList.contains('compact')")
    assert word_display() != "none"
    assert fits() == [[True, True], [True, True]]
    page.evaluate("view.sel.sw = 40; viewRedraw()")
    assert fits() == [[True, True], [True, True]]

    # the drop is an overlay only: the server's canvas is untouched
    page.evaluate("forceSync()")
    assert canvas_px(base, 150, 100) == BG

    # Apply composites it (with one draw-undo step) and leaves paste mode
    page.click("#sel-ok")
    assert wait_until(lambda: page.evaluate("view.sel") is None)  # async: snapshot first
    assert page.evaluate("Tools.current") == "copy"
    page.evaluate("forceSync()")
    assert wait_until(lambda: canvas_px(base, 150, 100) == PATCH)
    assert _canvas_size(base) == (200, 150)

    page.keyboard.press("Control+z")
    assert wait_until(lambda: canvas_px(base, 150, 100) == BG)


def test_paste_cancel_discards(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 150, 100)

    page.click("#sel-cancel")
    assert page.evaluate("view.sel") is None
    assert page.is_hidden("#sel-ok")
    # discarding leaves paste mode, exactly like applying does
    assert page.evaluate(_active("tool-copy"))
    page.evaluate("forceSync()")
    assert canvas_px(base, 150, 100) == BG

    # Esc discards too (and leaves paste mode the same way)
    _paste(page, 150, 100)
    page.keyboard.press("Escape")
    assert page.evaluate("view.sel") is None
    assert page.evaluate(_active("tool-copy"))


def test_selection_handles_resize_and_rotate(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 100, 75)  # 40x40 at (80..120, 55..95)

    # right-edge handle: moves that edge, the left edge stays put
    page.mouse.move(*screen_pos(page, 120, 75))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 150, 75), steps=5)
    page.mouse.up()
    sel = _sel_state(page)
    assert sel["sw"] == 70
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.sh)]") == [115, 40]

    # top-left corner: free rotation around the centre
    before = _sel_state(page)["angle"]
    page.mouse.move(*screen_pos(page, 80, 55))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 115, 60), steps=5)
    page.mouse.up()
    after = _sel_state(page)["angle"]
    assert after != before
    assert abs(after) <= 2 * 3.141592653589793

    # bottom-right corner: aspect-locked scale, opposite corner pinned
    page.evaluate("view.sel.angle = 0")
    page.evaluate("view.sel.sw = 40; view.sel.sh = 40; view.sel.cx = 100; view.sel.cy = 75")
    page.mouse.move(*screen_pos(page, 120, 55))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 160, 35), steps=5)
    page.mouse.up()
    sel = _sel_state(page)
    assert sel["sw"] == sel["sh"]  # the aspect (1:1) is preserved
    assert sel["sw"] > 40
    # the pinned bottom-left corner did not move
    assert page.evaluate("[view.sel.cx - view.sel.sw / 2, view.sel.cy + view.sel.sh / 2]") == [80, 95]

    # the canvas never changed while all this happened
    assert _canvas_size(base) == (200, 150)


def test_draw_on_selection_and_shift_erase(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 150, 100)  # local (0,0) is world (130,80)

    def alpha(lx, ly):
        return page.evaluate(
            "([x, y]) => view.sel.img.getContext('2d').getImageData(x, y, 1, 1).data[3]",
            [lx, ly],
        )

    assert alpha(20, 20) == 255

    # a normal draw paints ON the selection (in its frame), not on the canvas.
    # Off the exact centre: the apply / discard pair sits there (and only its
    # buttons eat clicks, see #sel-actions)
    page.keyboard.press("d")
    page.evaluate("Tools.state.brush.color = '#ff0000'; Tools.state.brush.size = 20")
    stroke(page, [[140, 90]])
    assert alpha(10, 10) == 255
    assert page.evaluate(
        "([x, y]) => view.sel.img.getContext('2d').getImageData(x, y, 1, 1).data.slice(0, 3)",
        [10, 10],
    ) == [255, 0, 0]
    assert canvas_px(base, 140, 90) == BG

    # shift draws as an eraser: the affected pixels lose their alpha
    dot(page, 140, 90, shift=True)
    assert alpha(10, 10) == 0
    assert alpha(2, 2) == 255  # untouched corner
    assert canvas_px(base, 140, 90) == BG

    # Apply keeps the erased pixels transparent
    page.click("#sel-ok")
    assert wait_until(lambda: page.evaluate("view.sel") is None)
    page.evaluate("forceSync()")
    assert canvas_px(base, 140, 90) == BG
    assert canvas_px(base, 132, 82) == PATCH


def test_crop_and_clone_target_the_selection(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 100, 75)  # 40x40 at (80..120, 55..95)

    # crop with an active selection crops the SELECTION, not the canvas
    page.keyboard.press("p")
    page.mouse.move(*screen_pos(page, 90, 60))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 110, 90), steps=5)
    page.mouse.up()
    sel = _sel_state(page)
    assert sel == {"sw": 20, "sh": 30, "angle": 0}
    assert _canvas_size(base) == (200, 150)
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.cy)]") == [100, 75]

    # clone samples the SELECTION: paint a marker colour on it, then clone
    # from that point (the canvas under the target is the plain BG, so a
    # source that leaked to the canvas would show up as BG). Coordinates are
    # inside the CROPPED selection: 20x30 at (90..110, 60..90)
    page.keyboard.press("d")
    page.evaluate("Tools.state.brush.color = '#ff0000'; Tools.state.brush.size = 12")
    stroke(page, [[94, 66]])  # local (4, 6)
    page.keyboard.press("o")
    page.keyboard.down("Control")
    page.mouse.move(*screen_pos(page, 94, 66))
    page.mouse.down()
    page.mouse.up()
    page.keyboard.up("Control")
    page.evaluate("Tools.state.clone.size = 12")
    stroke(page, [[104, 82]])  # local (14, 22)
    rgb = page.evaluate(
        "(() => { const d = view.sel.img.getContext('2d').getImageData(14, 22, 1, 1).data;"
        " return [d[0], d[1], d[2]]; })()"
    )
    assert rgb == [255, 0, 0]
    assert canvas_px(base, 104, 82) == BG  # the canvas never saw any of it


def test_stroke_preview_lands_on_the_selection(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 100, 75)  # 40x40 at (80..120, 55..95)

    def screen_rgb(x, y):
        """The device pixel of the DISPLAY canvas at canvas px (x, y)."""
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

    page.keyboard.press("d")
    page.evaluate("Tools.state.brush.color = '#ff0000'; Tools.state.brush.size = 20")
    page.mouse.move(*screen_pos(page, 94, 66))  # selection-local (14, 16)
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 98, 70), steps=4)

    # the live preview rides ON the selection: the stroke point is red there...
    assert screen_rgb(94, 66) == [255, 0, 0]
    # ...and not as a second copy offset by half the selection's size (the
    # preview layer's origin is its centre, not its top-left): the matching
    # point of that offset copy is (114, 91) and must still be the plain
    # selection image (the patch colour), not the stroke colour
    assert screen_rgb(114, 91) == list(PATCH)

    page.mouse.up()
    # the commit lands in the selection image, not on the canvas
    rgb = page.evaluate(
        "(() => { const d = view.sel.img.getContext('2d').getImageData(14, 16, 1, 1).data;"
        " return [d[0], d[1], d[2]]; })()"
    )
    assert rgb == [255, 0, 0]
    page.evaluate("forceSync()")
    assert canvas_px(base, 94, 66) == BG


def test_erase_preview_is_live(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 100, 75)  # 40x40 patch at (80..120, 55..95)

    def screen_rgb(x, y):
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

    assert screen_rgb(94, 66) == list(PATCH)  # the selection covers it

    page.keyboard.press("d")
    page.evaluate("Tools.state.brush.size = 20")
    page.mouse.move(*screen_pos(page, 94, 66))  # selection-local (14, 16)
    page.keyboard.down("Shift")
    page.mouse.down()

    # the erase is visible WHILE dragging: the hole reveals the canvas image
    # (the preview layer replaces the selection image, it is not blitted on
    # top of it, where lowering its alpha would show nothing)
    assert screen_rgb(94, 66) == list(BG)
    assert screen_rgb(84, 60) == list(PATCH)  # outside the brush: intact

    page.mouse.up()
    page.keyboard.up("Shift")
    a = page.evaluate(
        "([x, y]) => view.sel.img.getContext('2d').getImageData(x, y, 1, 1).data[3]",
        [14, 16],
    )
    assert a == 0
    assert screen_rgb(94, 66) == list(BG)  # the commit matches the preview


def test_paste_mode_moves_the_selection_and_cursors(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 100, 75)  # 40x40 at (80..120, 55..95)

    cursor = lambda: page.evaluate("document.getElementById('canvas').style.cursor")

    # in paste mode the body is draggable: the cursor says move
    page.mouse.move(*screen_pos(page, 90, 65))
    assert cursor() == "move"

    # dragging it MOVES the selection instead of dropping a new copy
    page.mouse.move(*screen_pos(page, 100, 75))  # its centre
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 110, 85), steps=3)
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.cy)]") == [110, 85]
    page.mouse.move(*screen_pos(page, 120, 95), steps=3)
    page.mouse.up()
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.cy)]") == [120, 95]
    assert _sel_state(page)["sw"] == 40  # same selection, just moved

    # handle cursors, on the moved selection (100..140, 75..115)
    for x, y, want in [
        (140, 95, "ew-resize"),  # right edge
        (100, 95, "ew-resize"),  # left edge
        (120, 75, "ns-resize"),  # top edge
        (120, 115, "ns-resize"),  # bottom edge
        (100, 75, "grabbing"),  # top-left corner (rotate)
        (140, 115, "grabbing"),  # bottom-right corner (rotate)
        (140, 75, "nesw-resize"),  # top-right corner (aspect scale)
        (100, 115, "nesw-resize"),  # bottom-left corner (aspect scale)
    ]:
        page.mouse.move(*screen_pos(page, x, y))
        assert cursor() == want, f"({x},{y})"

    # the "move" cursor is limited to the selection's body: outside it the
    # canvas keeps its default cursor (the point stays clear of the apply /
    # discard pair, which covers a band across the selection's middle)
    page.mouse.move(*screen_pos(page, 170, 130))
    assert cursor() == ""
    page.mouse.move(*screen_pos(page, 105, 110))  # inside, off the buttons
    assert cursor() == "move"
    # leaving paste mode drops it back to the CSS default (crosshair)
    page.keyboard.press("d")
    assert cursor() == ""

    # and the move drag keeps working with the paste tool re-armed: inside
    # the selection a drag moves it, a click OUTSIDE it drops a new copy
    page.keyboard.press("v")
    page.mouse.move(*screen_pos(page, 120, 95))
    page.mouse.down()
    page.mouse.move(*screen_pos(page, 130, 95), steps=3)
    page.mouse.up()
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.cy)]") == [130, 95]

    # a click OUTSIDE the active selection no longer drops a new copy: the
    # selection stays where the user positioned it and the status line says
    # what has to happen first
    page.mouse.click(*screen_pos(page, 160, 120))
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.cy)]") == [130, 95]
    assert page.evaluate("document.getElementById('status-line').textContent") == (
        "Apply or cancel active selection first"
    )

    # discarding frees the tool again: re-arming paste, the next click drops
    # a new copy (discarding itself left the copy tool, like Apply does)
    page.keyboard.press("Escape")
    assert page.evaluate("view.sel") is None
    page.keyboard.press("v")
    page.mouse.click(*screen_pos(page, 160, 120))
    assert page.evaluate("[Math.round(view.sel.cx), Math.round(view.sel.cy)]") == [160, 120]


def test_selection_survives_tool_switch_and_dies_with_the_view(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _copy(page, base)
    _paste(page, 150, 100)

    for key in ("d", "s", "o", "m", "p"):
        page.keyboard.press(key)
        assert page.evaluate("view.sel") is not None, f"lost on {key}"

    # a new server-side view (new entity here) drops the floating selection,
    # while the copied area survives (it is the client's own buffer)
    set_canvas(base, 120, 100, BG)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    assert page.evaluate("view.sel") is None
    assert page.is_hidden("#sel-ok")
    assert page.evaluate("view.clip") is not None


def test_selection_flip_buttons(page, server):
    base, _ = server
    _open(page, base)

    # a patch with four DISTINCT corners, so a flip is visible on the canvas
    corners = {(10, 10): (255, 0, 0), (49, 10): (0, 255, 0),
               (10, 49): (0, 0, 255), (49, 49): (255, 255, 255)}
    img = Image.new("RGB", (200, 150), BG)
    for (x, y), c in corners.items():
        img.putpixel((x, y), c)
    _upload(base, img)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")

    _copy(page, base)
    _paste(page, 30, 30)  # selection rect (10..50, 10..50)

    # the flip pair is centred on the same anchor, above / below the pair
    geo = page.evaluate(
        "(() => { const r = (id) => document.getElementById(id).getBoundingClientRect();"
        " const a = r('sel-actions'), h = r('sel-flip-h'), v = r('sel-flip-v');"
        " const cx = (a.left + a.right) / 2;"
        " return [Math.abs((h.left + h.right) / 2 - cx) < 1,"
        " Math.abs((v.left + v.right) / 2 - cx) < 1,"
        " h.bottom <= a.top, v.top >= a.bottom, h.width > 0, v.width > 0]; })()"
    )
    assert geo == [True, True, True, True, True, True]

    # the flip labels follow the same fit rule as apply / discard: glyphs only
    # on a small selection (this one), full labels once the row fits
    flip_word = lambda: page.evaluate(
        "getComputedStyle(document.querySelector('#sel-flip-h .sel-word')).display"
    )
    assert flip_word() == "none"
    page.evaluate("view.sel.sw = 300; viewRedraw()")
    assert flip_word() != "none"
    assert "H-flip" in page.evaluate("document.getElementById('sel-flip-h').textContent")
    assert "V-flip" in page.evaluate("document.getElementById('sel-flip-v').textContent")
    page.evaluate("view.sel.sw = 40; viewRedraw()")
    assert flip_word() == "none"

    def apply_sel():
        page.click("#sel-ok")
        assert wait_until(lambda: page.evaluate("view.sel") is None)
        page.evaluate("forceSync()")  # the composite reaches the server async

    # H-flip: left <-> right
    page.click("#sel-flip-h")
    apply_sel()
    assert canvas_px(base, 10, 10) == (0, 255, 0)
    assert canvas_px(base, 49, 10) == (255, 0, 0)
    assert canvas_px(base, 10, 49) == (255, 255, 255)
    assert canvas_px(base, 49, 49) == (0, 0, 255)

    page.keyboard.press("Control+z")
    page.evaluate("forceSync()")
    assert wait_until(lambda: canvas_px(base, 10, 10) == (255, 0, 0))

    # V-flip: top <-> bottom (the copied area is still loaded)
    _paste(page, 30, 30)
    page.click("#sel-flip-v")
    apply_sel()
    assert canvas_px(base, 10, 10) == (0, 0, 255)
    assert canvas_px(base, 49, 10) == (255, 255, 255)
    assert canvas_px(base, 10, 49) == (255, 0, 0)
    assert canvas_px(base, 49, 49) == (0, 255, 0)

    # flipping edits the SELECTION only: discarding leaves the image untouched
    page.keyboard.press("Control+z")
    page.evaluate("forceSync()")
    assert wait_until(lambda: canvas_px(base, 10, 10) == (255, 0, 0))
    _paste(page, 30, 30)
    page.click("#sel-flip-h")
    page.click("#sel-flip-v")
    page.keyboard.press("Escape")
    assert page.evaluate("view.sel") is None
    page.evaluate("forceSync()")
    assert canvas_px(base, 10, 10) == (255, 0, 0)
    assert canvas_px(base, 49, 49) == (255, 255, 255)
