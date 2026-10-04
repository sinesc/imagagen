"""Browser tests for the self-attention-guidance (Forge builtin extension)
controls: the main UI carries no SAG row; the model configuration dialog
offers, per direction (SDXL entries only), a checkbox plus an options gear
opening the three-slider options dialog, and the values round-trip through
the saved configuration.
"""

import io

from PIL import Image

from conftest import wait_until
from imagagen import metadata as md
from imagagen.sag import SelfAttentionGuidance


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-brush")
    page.wait_for_function("brushNames.length >= 1")


def _dialog_visible(page) -> bool:
    return page.evaluate("document.getElementById('dialog') !== null && "
                         "!document.getElementById('dialog').hidden")


def _open_config_dialog(page):
    page.click("#cfg-edit")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)


def test_no_sag_row_in_main_ui(page, server):
    """The main UI has no SAG row: the soft-inpainting row sits directly
    above the Generate/Save buttons (the enable flag and options live in
    the model configuration)."""
    base, _ = server
    _open(page, base)
    rows = page.locator("#controls > .row")
    classes = [rows.nth(i).get_attribute("class") or "" for i in range(rows.count())]
    assert "actions" in classes[-1]
    assert "soft-row" in classes[-2]
    assert "sag-row" not in classes
    assert page.locator("#sag, #sag-cfg, #sag-note").count() == 0


def test_config_dialog_sag_checkbox_and_options_gear(page, server):
    """Per direction (SDXL entries only) the fieldset offers a checkbox and
    an options gear; the gear opens the three-slider dialog initialized
    from that section, and saving reopens the config dialog with the state
    intact (unsaved edits survive the detour)."""
    import httpx

    base, _ = server
    _open(page, base)
    _open_config_dialog(page)
    boxes = page.locator(".cfg-sag")
    gears = page.locator(".cfg-sag-cfg")
    # the mock model is an SDXL stand-in: both fieldsets offer both
    assert boxes.count() == 2 and gears.count() == 2
    assert not boxes.first.is_checked()

    # the gear opens the options dialog with the Forge defaults
    gears.first.click()
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Self attention guidance" in page.locator("#dialog-title").inner_text()
    sliders = page.locator("#dialog-body input[type=range]")
    assert sliders.count() == 3
    for i, d in enumerate(["0.5", "2", "1"]):
        assert sliders.nth(i).input_value() == d
    # edit a slider and save: the config dialog reopens...
    page.evaluate("(() => { const i = document.querySelectorAll('#dialog-body input[type=range]')[0]; "
                  "i.value = '1.25'; "
                  "i.dispatchEvent(new Event('input', { bubbles: true })); })()")
    page.click("#dialog-ok")
    # ...still in the model configuration dialog (not the main UI), with
    # the other fields' state intact
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Model configuration" in page.locator("#dialog-title").inner_text()

    # an unsaved edit to another field survives the detour: edit the CFG,
    # detour into the i2i options dialog, come back
    page.locator(".cfg-mode").nth(1).locator(".cfg-cfg").fill("4.25")
    page.locator(".cfg-sag-cfg").nth(1).click()
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Self attention guidance" in page.locator("#dialog-title").inner_text()
    page.click("#dialog-cancel")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Model configuration" in page.locator("#dialog-title").inner_text()
    assert page.locator(".cfg-mode").nth(1).locator(".cfg-cfg").input_value() == "4.25"

    # save: the configuration carries the t2i options edit + the CFG edit
    boxes.first.check()
    page.fill(".cfg-name", "SAG config")
    page.click("#dialog-ok")
    assert wait_until(lambda: not _dialog_visible(page), timeout=5)
    with httpx.Client(base_url=base, timeout=10) as c:
        m = c.get("/api/models").json()
        e = [x for x in m["models"] if x["name"] == "SAG config"][0]
        assert e["t2i"]["sag"] is True and e["i2i"]["sag"] is False
        assert e["t2i"]["sag_scale"] == 1.25
        assert e["t2i"]["sag_blur_sigma"] == 2.0
        assert e["t2i"]["sag_threshold"] == 1.0
        # the unsaved i2i CFG edit survived the detour into the i2i
        # options dialog
        assert e["i2i"]["cfg"] == 4.25
        # the i2i section kept the Forge option defaults
        assert (e["i2i"]["sag_scale"], e["i2i"]["sag_blur_sigma"],
                e["i2i"]["sag_threshold"]) == (0.5, 2.0, 1.0)

    # reopening prefills the checkbox and the edited options
    _open_config_dialog(page)
    assert page.locator(".cfg-sag").first.is_checked()
    page.locator(".cfg-sag-cfg").first.click()
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator("#dialog-body input[type=range]").first.input_value() == "1.25"


def test_config_dialog_sag_sdxl_only(page, server):
    """A non-SDXL entry gets no SAG checkbox or gear at all (omitted, not
    disabled)."""
    base, _ = server
    _open(page, base)
    page.evaluate(
        "openConfigDialog({name: 'Fake Flux', model_id: 'flux/f.safetensors', "
        "family: 'flux', distilled: true, is_default: true, id: 'fake', "
        "t2i: {sampler: null, scheduler: null, steps: null, cfg: null, "
        "loras: []}, i2i: {sampler: null, scheduler: null, steps: null, "
        "cfg: null, loras: []}})")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator(".cfg-sag").count() == 0
    assert page.locator(".cfg-sag-cfg").count() == 0
    page.click("#dialog-cancel")


def _sag_png(sag) -> bytes:
    """A minimal PNG carrying the given sag object in its parameters chunk
    (sag=None -> no sag keys at all, like a legacy/other-tool image)."""
    img, info = md.attach_metadata(
        Image.new("RGB", (64, 64), (9, 10, 11)),
        md.build_parameters("a photo of a frog", "blurry", 20, "Euler a",
                            "Karras", 7.0, 1, 64, 64, "some_model", "abc",
                            sag=sag),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _drop_png(page, selector: str, png: bytes, name="drop.png"):
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


def test_dropped_image_does_not_import_sag_settings(page, server):
    """SAG's enabled flag and option values are model-configuration
    settings: dropping an image with sag metadata must NOT modify the
    selected configuration (it only switches the model as usual)."""
    import httpx

    base, _ = server
    _open(page, base)
    with httpx.Client(base_url=base, timeout=10) as c:
        m = c.get("/api/models").json()
        before = m["models"][0]
    # the image carries sag metadata in both enabled states
    _drop_png(page, "#prompt", _sag_png(SelfAttentionGuidance(enabled=True, scale=2.0)))
    page.wait_for_timeout(500)
    _drop_png(page, "#prompt", _sag_png(SelfAttentionGuidance(enabled=False, scale=0.1)))
    page.wait_for_timeout(500)
    with httpx.Client(base_url=base, timeout=10) as c:
        m = c.get("/api/models").json()
        after = m["models"][0]
    assert after["t2i"] == before["t2i"]
    assert after["i2i"] == before["i2i"]
