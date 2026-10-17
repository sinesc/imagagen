"""Browser tests for the rotation dropdown (dims row).

Covers: the select's position in the width/height row and its options,
the ui-state round-trip, and an actual generate-with-rotation (the mock's
horizontal R ramp is mirrored in the selected result). Uses the
mock-backend server in a temp root.
"""

from conftest import set_canvas, wait_until


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-brush")
    page.wait_for_function("brushNames.length >= 1")


def test_rotation_select_layout(page, server):
    base, _ = server
    _open(page, base)
    # the select sits in the dims row, between the oversample and batch slots
    # each slot: its own id (the swap button) or the id of the control
    # nested in its label
    ids = page.evaluate(
        "() => [...document.getElementById('width')"
        ".closest('.row').children]"
        ".map(e => e.id || e.querySelector('input,select')?.id || '')")
    assert ids == ["width", "swap-dims", "height", "oversample", "rotation",
                    "batch"]
    opts = page.evaluate(
        "() => [...document.getElementById('rotation').options]"
        ".map(o => [o.value, o.textContent])")
    assert opts == [
        ["-180", "-180°"], ["-135", "-135°"], ["-90", "-90°"],
        ["-45", "-45°"], ["0", "0°"], ["45", "+45°"], ["90", "+90°"],
        ["135", "+135°"], ["180", "+180°"],
    ]
    # 0° is the default
    assert page.evaluate("() => document.getElementById('rotation').value") == "0"


def test_rotation_persists_across_reload(page, server):
    base, _ = server
    _open(page, base)
    page.select_option("#rotation", "90")
    # the select's onchange schedules a (0-delay) ui-state save
    assert wait_until(lambda: page.evaluate(
        "fetch('/api/ui-state').then(r => r.json())"
        ".then(j => j.state.rotation)") == 90)
    page.reload()
    assert wait_until(lambda: page.evaluate(
        "() => document.getElementById('rotation').value") == "90")


def test_generate_rotation_180_mirrors(page, server):
    base, _ = server
    _open(page, base)
    # a flat 96x64 canvas (the API's minimum is 64px), picked up by the
    # page on reload; the form asks for the same size
    set_canvas(base, 96, 64, 30)
    page.reload()
    assert wait_until(lambda: page.evaluate(
        "() => document.getElementById('rotation') !== null"))
    page.fill("#width", "96")
    page.fill("#height", "64")
    page.select_option("#rotation", "180")
    page.click("#btn-generate")
    # wait for the first result thumbnail (the 'image' event)
    assert wait_until(
        lambda: page.evaluate(
            "() => document.querySelectorAll('#thumbs img').length") >= 1,
        timeout=30)
    job = page.evaluate("() => state.job")
    # selecting the result puts the counter-rotated image on the canvas
    import httpx
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/canvas/thumb", json={"job": job, "index": 0})
        assert r.status_code == 200, r.text
        img = c.get("/api/canvas.png")
        assert img.status_code == 200
    import io

    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(img.content)))
    assert a.shape == (64, 96, 3)
    # the mock's R ramp is mirrored (increasing-x ramp for rotation 0, so
    # the inequality is reversed here)
    assert int(a[32, 8, 0]) > int(a[32, 87, 0])
