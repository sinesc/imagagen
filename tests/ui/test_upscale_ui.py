"""Browser tests for the Upscale dialog (Chromium via Playwright).

Covers the toolbar button (position + U hotkey), the dialog (upscaler
combobox with built-ins, scale slider with live output-size preview,
name-driven scale auto-set), the crop-style thumbnail flow after Ok, and
the error path (broken model keeps the dialog open). The mock server has
no GPU; the pure-PIL built-in upscalers exercise the full round trip.
"""

import io
from pathlib import Path

import httpx
import pytest
from conftest import combo_pick, combo_seq, set_canvas, set_input, wait_until
from PIL import Image

BG = (20, 30, 40)


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#btn-upscale")
    page.wait_for_function("brushNames.length >= 1")


def _show(page, base, w=200, h=150):
    set_canvas(base, w, h, BG)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")


def _open_dialog(page):
    page.evaluate("view.has && viewRedraw()")  # ensure no accidental edits
    page.click("#btn-upscale")
    page.wait_for_function(
        "document.getElementById('dialog-title').textContent === 'Upscale'")


def _canvas_size(base):
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.get("/api/canvas.png")
        if r.status_code == 404:
            return None
        return Image.open(io.BytesIO(r.content)).size


def _thumb_titles(page):
    return page.evaluate(
        "[...document.querySelectorAll('#thumbs img')].map(i => i.title)")


def test_upscale_button_sits_right_of_paste(page, server):
    base, _ = server
    _open(page, base)
    paste = page.locator("#tool-paste").bounding_box()
    ups = page.locator("#btn-upscale").bounding_box()
    assert ups["x"] > paste["x"] + paste["width"] - 2  # same toolbar row
    assert abs(ups["y"] - paste["y"]) < 4


def test_upscale_hotkey_opens_dialog_once(page, server):
    base, _ = server
    _open(page, base)
    page.keyboard.press("u")
    page.wait_for_function("document.getElementById('dialog-title').textContent === 'Upscale'")
    # while the dialog is open U must NOT re-open / reset it
    page.evaluate("document.querySelector('.ups-scale').value = '3'")
    page.keyboard.press("u")
    assert not page.locator("#dialog").is_hidden()
    assert page.evaluate("document.querySelector('.ups-scale').value") == "3"
    page.click("#dialog-cancel")


def test_upscale_combobox_lists_builtins_first(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _open_dialog(page)
    # built-in upscalers are listed first, ungrouped (combo_seq closes
    # the list with Escape, which also closes the dialog: fine here,
    # this test only reads the list)
    seq = combo_seq(page, page.locator(".ups-upscaler"))
    texts = [t for kind, t in seq if kind == "item"]
    assert texts[:2] == ["Lanczos", "Nearest"]


def test_upscale_dialog_flow_creates_new_image_and_thumb(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)

    _open_dialog(page)
    combo_pick(page, page.locator(".ups-upscaler"), "Lanczos")

    # default scale 4 -> live preview shows the forge size rule result
    assert page.evaluate("document.querySelector('.ups-dims').textContent") \
        == "800\u00d7600"
    set_input(page, ".ups-scale", 2)
    assert page.evaluate("document.querySelector('.ups-dims').textContent") \
        == "400\u00d7296"  # int(150*2)//8*8 = 296, not 300

    page.click("#dialog-ok")
    # crop-style: the pre-upscale image AND the result become thumbnails,
    # the result is selected; titles carry the dimensions after decode
    assert wait_until(lambda: any(
        t.startswith("upscaled image 400\u00d7296")
        for t in _thumb_titles(page)))
    assert _canvas_size(base) == (400, 296)
    assert page.evaluate("currentEntityId") == "upscale:0"
    titles = _thumb_titles(page)
    # like the crop session, BOTH thumbnails carry the session kind
    # (the 200x150 entry is the pre-upscale image, restorable)
    assert len(titles) == 2 and "upscaled image 200\u00d7150" in titles[0]
    # the selection highlight sits on the upscaled thumbnail
    assert page.evaluate(
        "document.querySelectorAll('#thumbs img').length === 2"
        " && document.querySelectorAll('#thumbs img.selected').length === 1"
        " && document.querySelector('#thumbs img.selected').title"
        ".startsWith('upscaled image')")
    # progress bar was cleared after the sync request settled
    assert page.evaluate("document.getElementById('progress-fill')"
                         ".style.width") == "0%"


def test_upscale_scale_autosets_from_model_name(page, server):
    base, tmp = server
    fake = tmp / "models" / "upscalers" / "real"
    fake.mkdir(parents=True)
    (fake / "RealESRGAN_x4plus.pth").write_bytes(b"not-a-real-model")
    _open(page, base)
    _show(page, base)
    _open_dialog(page)
    combo_pick(page, page.locator(".ups-upscaler"),
               "real/RealESRGAN_x4plus")
    # forge's set_scale_by_when_changing_upscaler: "x4" in the name -> 4
    assert page.evaluate("document.querySelector('.ups-scale').value") == "4"
    assert page.evaluate("document.querySelector('.ups-dims').textContent") \
        == "800\u00d7600"
    # the model cannot load -> the error stays IN the dialog (retry there)
    page.click("#dialog-ok")
    page.wait_for_function("!document.getElementById('dialog').hidden"
                           " && !!document.querySelector('.dialog-error')")
    err = page.locator(".dialog-error").text_content()
    assert "upscale failed" in err
    assert _canvas_size(base) == (200, 150)  # nothing changed
    page.click("#dialog-cancel")


def test_upscale_cancel_leaves_canvas(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _open_dialog(page)
    page.click("#dialog-cancel")
    assert _canvas_size(base) == (200, 150)
    assert page.evaluate("state.thumbs.length") == 0


def test_upscale_cancel_aborts_running_upscale(page, server):
    base, tmp = server
    # a REAL model (smallest installed weight, symlinked into the server
    # root) so the CPU run is slow enough to interrupt mid-tiling; the
    # pure-PIL built-ins have no tile boundary to abort at
    models_root = Path(__file__).resolve().parents[2] / "models" / "upscalers"
    files = sorted(models_root.rglob("*.pth"),
                   key=lambda p: p.stat().st_size) \
        if models_root.is_dir() else []
    if not files:
        pytest.skip("no upscaler weights on this host")
    model = next((f for f in files if "x2" in f.name.lower()), files[0])
    dst = tmp / "models" / "upscalers" / model.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(model.resolve())

    _open(page, base)
    _show(page, base, 64, 64)
    _open_dialog(page)
    combo_pick(page, page.locator(".ups-upscaler"), model.name[:-4])
    # tiny tiles -> many slow CPU tiles: plenty of runtime to abort in
    set_input(page, ".ups-tile", 8)
    set_input(page, ".ups-overlap", 4)
    page.click("#dialog-ok")
    # the OK button is disabled while the upscale request is in flight
    page.wait_for_function(
        "document.getElementById('dialog-ok').disabled === true")
    page.wait_for_timeout(1200)  # let the run actually start (model load)

    # Cancel mid-run aborts but does NOT close the dialog: the request
    # settles with an in-dialog "upscale aborted" note while it stays open
    page.click("#dialog-cancel")
    page.wait_for_function(
        "!document.getElementById('dialog').hidden"
        " && [...document.querySelectorAll('.dialog-error')]"
        ".some(d => d.textContent.includes('aborted'))", timeout=30000)
    # Ok is enabled again (retry possible)... Cancel now closes (idle)
    assert page.evaluate(
        "document.getElementById('dialog-ok').disabled") is False
    page.click("#dialog-cancel")
    page.wait_for_function(
        "document.getElementById('dialog').hidden === true")
    # nothing was applied: canvas untouched, no new entities/thumbs
    assert _canvas_size(base) == (64, 64)
    assert _thumb_titles(page) == []
    # the manager is free again: a built-in upscale still works
    assert httpx.post(f"{base}/api/upscale", json={
        "upscaler": "Lanczos", "scale": 2.0}, timeout=30).status_code == 200
