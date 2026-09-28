"""Browser tests for the self-attention-guidance row (Forge builtin
extension): checkbox left + options gear right in the row above the
Generate/Save buttons (below the soft-inpainting row, always enabled --
SAG works in all modes), and the three-slider options dialog with
persistence.
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


def test_sag_row_layout_and_always_enabled(page, server):
    base, _ = server
    _open(page, base)

    # the row sits between the soft-inpainting row and the Generate/Save
    # buttons, with the checkbox first and the gear last (right-aligned)
    rows = page.locator("#controls > .row")
    classes = [rows.nth(i).get_attribute("class") or "" for i in range(rows.count())]
    assert "actions" in classes[-1]
    assert "sag-row" in classes[-2]
    assert "soft-row" in classes[-3]
    row = page.locator(".sag-row")
    first = row.locator("> :first-child")
    assert first.evaluate("e => e.querySelector('input')?.id") == "sag"
    assert row.locator("> :last-child").get_attribute("id") == "sag-cfg"
    # the label sits to the RIGHT of the checkbox (one line, not stacked)
    box = page.locator("#sag").bounding_box()
    lab = row.locator(".soft-check span").bounding_box()
    assert abs(lab["y"] - box["y"]) < 5
    assert lab["x"] > box["x"] + box["width"]

    # unlike soft inpainting, SAG works in all modes: never disabled
    assert not page.locator("#sag").is_checked()
    assert not page.locator("#sag").is_disabled()
    page.select_option("#mode", "inpaint_masked")
    assert not page.locator("#sag").is_disabled()
    page.select_option("#mode", "inpaint_not_masked")
    assert not page.locator("#sag").is_disabled()
    page.select_option("#mode", "generate")
    assert not page.locator("#sag").is_disabled()


def test_sag_disabled_for_non_sdxl_models(page, server):
    """SAG is implemented for the SDXL UNet only; the Flux/Z-Image backends
    ignore the sag field, so the checkbox (and its options gear) must be
    disabled for non-SDXL models."""
    base, _ = server
    _open(page, base)

    # the mock model is an SDXL stand-in: the controls are enabled
    assert not page.locator("#sag").is_disabled()
    assert not page.locator("#sag-cfg").is_disabled()

    # the mock registry only knows SDXL models, so flip the selected
    # configuration's family and re-apply the family controls directly
    for family, disabled in [("flux", True), ("zimage", True), ("sdxl", False)]:
        page.evaluate(
            "([f]) => { state.entries[0].family = f; applyFamilyControls(); }",
            [family],
        )
        assert page.locator("#sag").is_disabled() is disabled
        assert page.locator("#sag-cfg").is_disabled() is disabled


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


def test_sag_checkbox_and_options_persist_across_reload(page, server):
    base, _ = server
    _open(page, base)
    page.check("#sag")
    page.evaluate("saveUiState()")
    assert wait_until(lambda: _ui_state_key(page, "sag") is True)
    page.reload()
    page.wait_for_function("brushNames.length >= 1")
    # (applyUiState runs when the ui-state fetch resolves; poll for it)
    assert wait_until(lambda: page.locator("#sag").is_checked())
    # SAG stays enabled in "generate" mode (no mode gating)
    assert not page.locator("#sag").is_disabled()


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


def test_dropped_image_restores_sag_checkbox_both_states(page, server):
    """Regression: the sag_enabled flag is written unconditionally, so a
    dropped image restores the checkbox in BOTH directions (and an image
    without the key leaves the checkbox as-is)."""
    base, _ = server
    _open(page, base)

    # SAG on -> drop an image generated with SAG off: the checkbox must
    # turn off (and the image's option values are imported)
    page.check("#sag")
    _drop_png(page, "#prompt", _sag_png(SelfAttentionGuidance(enabled=False, scale=2.0)))
    assert wait_until(lambda: not page.locator("#sag").is_checked(), timeout=5)
    assert wait_until(lambda: _ui_state_key(page, "sag") is False, timeout=5)
    assert wait_until(lambda: _ui_state_key(page, "sag_scale") == 2.0, timeout=5)

    # the inverse: SAG off -> drop an image with SAG on
    _drop_png(page, "#prompt", _sag_png(SelfAttentionGuidance()))
    assert wait_until(lambda: page.locator("#sag").is_checked(), timeout=5)
    assert wait_until(lambda: _ui_state_key(page, "sag_scale") == 0.5, timeout=5)

    # an image without any sag keys (legacy / other tool) leaves the
    # checkbox as-is
    _drop_png(page, "#prompt", _sag_png(None))
    page.wait_for_timeout(300)
    assert page.locator("#sag").is_checked()
