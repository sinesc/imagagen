"""Browser tests for the self-attention-guidance (Forge builtin extension)
controls: the enable flag is a per-direction checkbox in the model
configuration dialog (SDXL entries only), the main UI keeps only the
options gear (dimmed for non-SDXL models), and the three-slider options
dialog with persistence.
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


def _ui_state_key(page, key: str):
    return page.evaluate(
        "fetch('/api/ui-state').then(r => r.json())"
        ".then(j => j.state['" + key + "'])"
    )


def test_sag_row_is_options_only(page, server):
    """The main UI no longer carries an enable checkbox: the row above the
    Generate/Save buttons (below the soft-inpainting row) shows the options
    note + gear; the enable flag lives in the model configuration."""
    base, _ = server
    _open(page, base)

    rows = page.locator("#controls > .row")
    classes = [rows.nth(i).get_attribute("class") or "" for i in range(rows.count())]
    assert "actions" in classes[-1]
    assert "sag-row" in classes[-2]
    assert "soft-row" in classes[-3]
    row = page.locator(".sag-row")
    # no checkbox in the row; note text left, gear right
    assert row.locator("input[type=checkbox]").count() == 0
    note = row.locator("#sag-note")
    assert "Self attention guidance options" in note.inner_text()
    assert row.locator("> :last-child").get_attribute("id") == "sag-cfg"
    box = note.bounding_box()
    gear = row.locator("#sag-cfg").bounding_box()
    assert gear["x"] > box["x"] + box["width"]
    # the mock model is an SDXL stand-in: the gear is enabled
    assert not page.locator("#sag-cfg").is_disabled()


def test_sag_options_dimmed_for_non_sdxl_models(page, server):
    """SAG is implemented for the SDXL UNet only; the mock registry only
    knows SDXL models, so flip the selected configuration's family and
    re-apply the family controls directly: the gear is disabled and the
    note text dims (#aaa)."""
    base, _ = server
    _open(page, base)
    note = page.locator("#sag-note")
    for family, disabled in [("flux", True), ("zimage", True), ("sdxl", False)]:
        page.evaluate(
            "([f]) => { state.entries[0].family = f; applyFamilyControls(); }",
            [family],
        )
        assert page.locator("#sag-cfg").is_disabled() is disabled
        assert note.evaluate("e => getComputedStyle(e).color") == (
            "rgb(170, 170, 170)" if disabled else "rgb(204, 204, 204)")


def test_sag_options_dialog(page, server):
    base, _ = server
    _open(page, base)
    # the gear opens the options dialog with the three Forge sliders
    page.click("#sag-cfg")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Self attention guidance" in page.locator("#dialog-title").inner_text()
    sliders = page.locator("#dialog-body input[type=range]")
    assert sliders.count() == 3
    # Forge slider ranges/steps
    expect = [
        ("-2", "5", "0.01"),    # scale
        ("0", "10", "0.01"),    # blur sigma
        ("0", "4", "0.01"),     # blur mask threshold
    ]
    for i, (mn, mx, step) in enumerate(expect):
        s = sliders.nth(i)
        assert s.get_attribute("min") == mn
        assert s.get_attribute("max") == mx
        assert s.get_attribute("step") == step
    # the Forge defaults
    for i, d in enumerate(["0.5", "2", "1"]):
        assert sliders.nth(i).input_value() == d

    # changing a slider updates its live value; saving persists to ui-state
    page.evaluate("(() => { const i = document.querySelectorAll('#dialog-body input[type=range]')[0]; "
                  "i.value = '1.25'; "
                  "i.dispatchEvent(new Event('input', { bubbles: true })); })()")
    assert sliders.nth(0).input_value() == "1.25"
    page.click("#dialog-ok")
    assert wait_until(lambda: not _dialog_visible(page), timeout=3)
    assert wait_until(lambda: _ui_state_key(page, "sag_scale") == 1.25, timeout=5)

    # the dialog reopens at the persisted value
    page.click("#sag-cfg")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator("#dialog-body input[type=range]").first.input_value() == "1.25"


def test_sag_options_persist_across_reload(page, server):
    """Only the options persist in ui-state (the enabled flag is a model
    configuration setting, not a UI field)."""
    base, _ = server
    _open(page, base)
    page.evaluate("(() => { state.sag.scale = 1.5; saveUiState(); })()")
    assert wait_until(lambda: _ui_state_key(page, "sag_scale") == 1.5)
    assert _ui_state_key(page, "sag") is None
    page.reload()
    page.wait_for_function("brushNames.length >= 1")
    assert wait_until(
        lambda: page.evaluate("state.sag.scale") == 1.5)


def test_config_dialog_sag_checkboxes_sdxl_only(page, server):
    """The model configuration dialog offers a self-attention-guidance
    checkbox in each direction's fieldset for SDXL entries only; the
    values round-trip through the saved configuration."""
    import httpx

    base, _ = server
    _open(page, base)
    page.click("#cfg-edit")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    boxes = page.locator(".cfg-sag")
    # the mock model is an SDXL stand-in: both fieldsets offer the checkbox
    assert boxes.count() == 2
    assert not boxes.first.is_checked()
    assert not boxes.last.is_checked()

    # check both and save; the configuration carries the values
    boxes.first.check()
    boxes.last.check()
    page.fill(".cfg-name", "SAG config")
    page.click("#dialog-ok")
    assert wait_until(lambda: not _dialog_visible(page), timeout=5)
    with httpx.Client(base_url=base, timeout=10) as c:
        m = c.get("/api/models").json()
        e = [x for x in m["models"] if x["name"] == "SAG config"][0]
        assert e["t2i"]["sag"] is True and e["i2i"]["sag"] is True

    # reopening prefills both checkboxes
    page.click("#cfg-edit")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator(".cfg-sag").first.is_checked()
    assert page.locator(".cfg-sag").last.is_checked()
    page.click("#dialog-cancel")

    # a non-SDXL entry gets no checkboxes at all (omitted, not disabled)
    page.evaluate(
        "openConfigDialog({name: 'Fake Flux', model_id: 'flux/f.safetensors', "
        "family: 'flux', distilled: true, is_default: true, id: 'fake', "
        "t2i: {sampler: null, scheduler: null, steps: null, cfg: null, "
        "loras: []}, i2i: {sampler: null, scheduler: null, steps: null, "
        "cfg: null, loras: []}})")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator(".cfg-sag").count() == 0
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


def test_dropped_image_imports_sag_options_only(page, server):
    """The enabled flag is a model configuration setting now, so dropping an
    image imports only the sag option values (in both enabled states; an
    image without the keys leaves the options as-is)."""
    base, _ = server
    _open(page, base)

    # drop an image generated with SAG off: the option values are imported
    _drop_png(page, "#prompt", _sag_png(SelfAttentionGuidance(enabled=False, scale=2.0)))
    assert wait_until(lambda: _ui_state_key(page, "sag_scale") == 2.0, timeout=5)

    # an image with SAG on imports its (default) values
    _drop_png(page, "#prompt", _sag_png(SelfAttentionGuidance()))
    assert wait_until(lambda: _ui_state_key(page, "sag_scale") == 0.5, timeout=5)

    # an image without any sag keys (legacy / other tool) leaves the
    # options as-is
    page.evaluate("(() => { state.sag.scale = 3.0; })()")
    _drop_png(page, "#prompt", _sag_png(None))
    page.wait_for_timeout(300)
    assert page.evaluate("state.sag.scale") == 3.0
