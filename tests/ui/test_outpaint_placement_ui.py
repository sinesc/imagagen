"""UI checks for the outpaint source-placement overlay: visibility rules,
drag-handle control of the placement, handle cursors, and ui-state
persistence (the mock server runs with the MockBackend)."""

import json
from pathlib import Path

from conftest import set_canvas, set_input, wait_until


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-brush")


def _show(page, base, w=128, h=64):
    """Put an image on the server canvas and make the UI render it."""
    set_canvas(base, w, h, (20, 30, 40))
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")


def _geom(page):
    return page.evaluate("outpaintGeom()")


def _placement(page):
    return page.evaluate("state.outpaintPlacement")


def _handle_screen(page, axis, sign):
    """Screen coords of a placement handle: axis "x" -> right (sign 1) /
    left (sign -1) edge handle, axis "y" -> bottom (1) / top (-1).
    None when that axis has no slack (no handles there)."""
    return page.evaluate(
        """([axis, sign]) => {
            const g = outpaintGeom();
            if (!g) return null;
            const hs = outpaintHandles(g).filter((h) => h.axis === axis);
            if (!hs.length) return null;
            const h = sign > 0 ? hs[hs.length - 1] : hs[0];
            const r = document.getElementById('canvas').getBoundingClientRect();
            return [r.left + view.ox + h.x * view.scale,
                    r.top + view.oy + h.y * view.scale];
        }""",
        [axis, sign],
    )


def _saved_placement(root):
    """The placement in ui-state.json once the debounced save landed."""
    p = Path(root) / "user" / "ui-state.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("placement")


# ------------------------------------------------------------- visibility --


def test_placement_overlay_visibility(page, server):
    base, _root = server
    _open(page, base)
    _show(page, base)
    set_input(page, "#width", "256")
    set_input(page, "#height", "64")

    # the overlay is outpaint-only: generate and inpaint_masked show nothing
    assert _geom(page) is None
    page.select_option("#mode", "inpaint_masked")
    assert _geom(page) is None

    page.select_option("#mode", "inpaint_not_masked")
    # 128x64 -> 256x64: horizontal slack 128, vertical axis filled
    g = _geom(page)
    assert g is not None and (g["W"], g["H"]) == (256, 64)
    assert g["gx"] == 128 and g["gy"] == 0
    # centered default: the box reaches 64 px past each side of the source
    assert abs(g["bx"] + 64) < 1e-6 and abs(g["ox"] - 64) < 1e-6

    # same AR (2:1 source into a 2:1 target): box == source, nothing to place
    set_input(page, "#width", "256")
    set_input(page, "#height", "128")
    assert _geom(page) is None

    # the box mirrors the server's oversample scale + 16-round-up
    set_input(page, "#width", "250")
    set_input(page, "#height", "64")
    set_input(page, "#oversample", "1.1")  # -> 288x80
    g = _geom(page)
    assert g is not None and (g["W"], g["H"]) == (288, 80)
    set_input(page, "#oversample", "1")

    # the -1 auto-size sentinel resolves server-side: not previewable
    set_input(page, "#width", "-1")
    assert _geom(page) is None


def test_placement_box_touches_source(page, server):
    """Geometry rule: the box touches the source on the axes WITHOUT slack
    (vertical here) and the source sits at ox/s inside the box."""
    base, _root = server
    _open(page, base)
    _show(page, base)
    set_input(page, "#width", "256")
    set_input(page, "#height", "64")
    page.select_option("#mode", "inpaint_not_masked")
    g = _geom(page)
    # vertical: filled -> the box edges coincide with the source edges
    assert abs(g["by"]) < 1e-9 and abs(g["bh"] - 64) < 1e-9
    # placement 0: source flush left inside the box
    page.evaluate("state.outpaintPlacement.x = 0; viewRedraw()")
    g = _geom(page)
    assert abs(g["bx"]) < 1e-9 and abs(g["ox"]) < 1e-9
    # placement 1: source flush right (its right edge coincides with the
    # box's right edge)
    page.evaluate("state.outpaintPlacement.x = 1; viewRedraw()")
    g = _geom(page)
    assert abs(g["bx"] + g["bw"] - 128) < 1e-6


# ------------------------------------------------------------------ drag --


def _drag_handle(page, axis, sign, dx, dy):
    """Press the handle and drag by (dx, dy) screen px; True if the drag
    started (the handle grabbed the pointer)."""
    sx, sy = _handle_screen(page, axis, sign)
    page.mouse.move(sx, sy)
    page.mouse.down()
    started = page.evaluate("view.outDrag !== null && view.outDrag !== undefined")
    page.mouse.move(sx + dx, sy + dy, steps=4)
    page.mouse.up()
    return started


def _anchor(page):
    """(view.ox, view.oy, box-screen-left, box-screen-top): the box's
    screen anchor must stay put across a placement drag."""
    return page.evaluate(
        """() => {
            const g = outpaintGeom();
            return [view.ox, view.oy,
                    view.ox + g.bx * view.scale, view.oy + g.by * view.scale];
        }""")


def test_placement_drag_moves_source(page, server):
    base, _root = server
    _open(page, base)
    _show(page, base)
    set_input(page, "#width", "256")
    set_input(page, "#height", "64")
    page.select_option("#mode", "inpaint_not_masked")
    assert _placement(page) == {"x": 0.5, "y": 0.5}
    ox0, oy0, bl0, bt0 = _anchor(page)
    assert _drag_handle(page, "x", 1, 40, 0)
    ox1, oy1, bl1, bt1 = _anchor(page)
    # the source (canvas pixels) follow the cursor 1:1: the view pans by
    # exactly the dragged screen delta
    assert abs((ox1 - ox0) - 40) < 1e-6
    assert oy1 == oy0  # an x-axis drag never touches the vertical offset
    # the box keeps its screen anchor: ox pans by the same amount the box
    # slides in world px, cancelling it. Tolerance: the box position is a
    # quantized TARGET-PX offset (int), i.e. 1 canvas px = view.scale
    # screen px of residual box jitter during the drag
    sc = page.evaluate("view.scale")
    assert abs(bl1 - bl0) <= 0.6 * sc and abs(bt1 - bt0) <= 0.6 * sc
    # placement mapping: 1 canvas display px (dx/scale) moves the source s
    # target px, i.e. placement by s/gx per canvas px
    exp = page.evaluate(
        """(dx) => 0.5 + dx / view.scale * outpaintGeom().s / outpaintGeom().gx""",
        40)
    assert abs(_placement(page)["x"] - exp) < 1e-6
    assert _placement(page)["y"] == 0.5
    # the vertical axis has no slack -> no vertical handles
    assert _handle_screen(page, "y", 1) is None
    assert _handle_screen(page, "y", -1) is None

    # vertical slack scenario: 128x64 -> 128x128 (box touches the source's
    # left AND right edges, handles above/below the source)
    x_before = _placement(page)["x"]
    set_input(page, "#width", "128")
    set_input(page, "#height", "128")
    g = _geom(page)
    assert g["gx"] == 0 and g["gy"] == 64
    assert _handle_screen(page, "x", 1) is None
    a1 = _anchor(page)  # re-anchor: the target change rebuilt the box, the
    ox1, oy1 = a1[0], a1[1]  # pre-change anchors no longer apply
    _drag_handle(page, "y", 1, 0, 40)
    exp = page.evaluate(
        "(dy) => 0.5 + dy / view.scale * outpaintGeom().s / outpaintGeom().gy",
        40)
    assert abs(_placement(page)["y"] - exp) < 1e-6
    assert _placement(page)["x"] == x_before  # the y drag never touches x
    # vertical source movement: box anchored, source follows the handle
    ox2, oy2, bl2, bt2 = _anchor(page)
    assert abs(ox2 - ox1) < 1e-6  # a y drag never pans horizontally
    assert abs((oy2 - oy1) - 40) < 1e-6
    # the box keeps its screen anchor during the drag (same quantization
    # tolerance as the x-axis case; the box is square here, so both edges
    # are anchored)
    assert abs(bl2 - a1[2]) <= 0.6 * sc and abs(bt2 - a1[3]) <= 0.6 * sc


def test_placement_drag_clamps(page, server):
    base, _root = server
    _open(page, base)
    _show(page, base)
    set_input(page, "#width", "256")
    set_input(page, "#height", "64")
    page.select_option("#mode", "inpaint_not_masked")
    # dragging far past the box clamps at 1 ...
    _drag_handle(page, "x", 1, 3000, 0)
    assert _placement(page)["x"] == 1
    # ... and at 0 via the left handle
    _drag_handle(page, "x", -1, -3000, 0)
    assert _placement(page)["x"] == 0
    # the drags panned the viewport (the source moved); refit so the canvas
    # centre is clear of the edge handles, then check the brush tool arms
    # as usual when NOT grabbing a handle
    page.evaluate("viewFit()")
    page.mouse.move(*_centre(page))
    page.mouse.down()
    assert page.evaluate("view.stroke !== null")
    page.mouse.up()


def _centre(page):
    return page.evaluate(
        """() => {
            const r = document.getElementById('canvas').getBoundingClientRect();
            return [r.left + r.width / 2, r.top + r.height / 2];
        }""")


def test_placement_handle_cursors(page, server):
    base, _root = server
    _open(page, base)
    _show(page, base)
    set_input(page, "#width", "256")
    set_input(page, "#height", "64")
    page.select_option("#mode", "inpaint_not_masked")
    def cur():
        return page.evaluate("document.getElementById('canvas').style.cursor")
    assert cur() == ""
    page.mouse.move(*_handle_screen(page, "x", 1))
    assert cur() == "ew-resize"
    page.mouse.move(*_centre(page))
    assert cur() == ""


# ------------------------------------------------------------- persistence --


def test_placement_persists_across_reload(page, server):
    base, root = server
    _open(page, base)
    _show(page, base)
    set_input(page, "#width", "256")
    set_input(page, "#height", "64")
    page.select_option("#mode", "inpaint_not_masked")
    _drag_handle(page, "x", 1, 40, 0)
    # the debounced ui-state save must land before the reload
    pl = wait_until(lambda: (_saved_placement(root) or [None])[0] is not None)
    assert pl
    saved = _saved_placement(root)
    assert abs(saved[0] - _placement(page)["x"]) < 1e-6
    page.reload()
    page.wait_for_function("document.getElementById('loading-overlay').hidden")
    assert abs(_placement(page)["x"] - saved[0]) < 1e-6
