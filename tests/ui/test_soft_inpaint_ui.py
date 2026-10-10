"""Browser tests for the soft-inpainting controls (Forge builtin
extension): the main UI carries no soft row; the model configuration dialog
offers the checkbox + options gear (six-slider dialog) in the i2i fieldset
only (soft inpainting needs an init mask), for EVERY family — all four
implement soft inpainting. The flag is enabled by default, including for
old configuration files that predate the key.
"""

import json

import httpx

from conftest import wait_until


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


def _set_range(page, idx: int, value: str):
    # range inputs only accept integer fill values, so set a decimal via
    # the real input event
    page.evaluate(
        "(([i, v]) => { const el = document.querySelectorAll"
        "('#dialog-body input[type=range]')[i]; el.value = v; "
        "el.dispatchEvent(new Event('input', { bubbles: true })); })",
        [idx, value])


def test_no_soft_row_in_main_ui(page, server):
    base, _ = server
    _open(page, base)
    assert page.locator("#soft-inpaint, #soft-inpaint-cfg").count() == 0
    classes = [r.get_attribute("class") or ""
               for r in page.locator("#controls > .row").all()]
    assert "soft-row" not in classes


def test_config_dialog_soft_row_layout_and_default(page, server):
    """The i2i fieldset carries the soft row directly under steps/CFG,
    aligned with the SAG row (checkbox left, label right, gear last); the
    t2i fieldset has NO soft controls (soft inpainting needs an init mask,
    only the inpaint modes read the i2i section). The checkbox is CHECKED
    by default (soft inpainting is enabled by default)."""
    base, _ = server
    _open(page, base)
    _open_config_dialog(page)
    boxes = page.locator(".cfg-soft")
    gears = page.locator(".cfg-soft-cfg")
    # i2i only: exactly one checkbox + gear (in the i2i fieldset)
    assert boxes.count() == 1 and gears.count() == 1
    fs = page.locator(".cfg-mode").nth(1)
    assert fs.locator(".cfg-soft").count() == 1
    assert page.locator(".cfg-mode").nth(0).locator(".cfg-soft, .cfg-soft-cfg").count() == 0
    rows = fs.locator(".row")
    # sampler/scheduler, steps/CFG, soft, SAG (the mock model is SDXL)
    assert rows.nth(2).locator(".cfg-soft").count() == 1
    assert rows.nth(3).locator(".cfg-sag").count() == 1
    # the gear is the row's last element (right-aligned)
    assert rows.nth(2).locator("> :last-child").get_attribute("class") \
        == "mini icon cfg-soft-cfg"
    # aligned with the SAG row: same left edge, checkbox left of the
    # label text (one line, vertically centred)
    soft_box = page.locator(".cfg-soft-row").bounding_box()
    sag_box = page.locator(".cfg-sag-row").nth(1).bounding_box()
    assert abs(soft_box["x"] - sag_box["x"]) < 1
    cb = boxes.first.bounding_box()
    assert cb["x"] >= soft_box["x"]
    assert abs(cb["y"] + cb["height"] / 2 - (soft_box["y"] + soft_box["height"] / 2)) < 5
    # enabled by default (new/in-memory configurations)
    assert boxes.first.is_checked()
    page.click("#dialog-cancel")


def test_config_dialog_soft_row_for_every_family(page, server):
    """A non-SDXL entry gets the soft row (i2i) even without the SAG row
    (all four families implement soft inpainting)."""
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
    assert page.locator(".cfg-soft").count() == 1
    assert page.locator(".cfg-soft-cfg").count() == 1
    # a section without the key (the fake prefill) shows it checked
    assert page.locator(".cfg-soft").first.is_checked()
    page.click("#dialog-cancel")


def test_soft_options_dialog_and_config_roundtrip(page, server):
    """The gear opens the six-slider dialog with the Forge defaults; the
    save/cancel detour reopens the config dialog with unsaved edits intact;
    saving persists the options into the configuration; reopening prefills
    them."""
    base, _ = server
    _open(page, base)
    _open_config_dialog(page)
    page.locator(".cfg-soft-cfg").click()
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Soft inpainting" in page.locator("#dialog-title").inner_text()
    sliders = page.locator("#dialog-body input[type=range]")
    assert sliders.count() == 6
    # Forge slider ranges/steps
    expect = [
        ("0", "8", "0.1"),       # schedule bias
        ("0", "8", "0.05"),      # preservation strength
        ("1", "32", "0.5"),      # transition contrast boost
        ("0", "1", "0.05"),      # mask influence
        ("0", "8", "0.25"),      # difference threshold
        ("0", "8", "0.25"),      # difference contrast
    ]
    for i, (mn, mx, step) in enumerate(expect):
        s = sliders.nth(i)
        assert s.get_attribute("min") == mn
        assert s.get_attribute("max") == mx
        assert s.get_attribute("step") == step
    for i, d in enumerate(["1", "0.5", "4", "0", "0.5", "2"]):
        assert sliders.nth(i).input_value() == d
    # the pixel composite section heading divides the two groups
    assert "Pixel composite settings" in page.locator("#dialog-body").inner_text()

    # edit a slider and save: the config dialog reopens (not the main UI)
    _set_range(page, 0, "2.2")
    page.click("#dialog-ok")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Model configuration" in page.locator("#dialog-title").inner_text()

    # an unsaved edit to another field survives the detour: edit the t2i
    # CFG, detour into the i2i soft options, cancel
    page.locator(".cfg-mode").nth(0).locator(".cfg-cfg").fill("4.25")
    page.locator(".cfg-soft-cfg").click()
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Soft inpainting" in page.locator("#dialog-title").inner_text()
    page.click("#dialog-cancel")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Model configuration" in page.locator("#dialog-title").inner_text()
    assert page.locator(".cfg-mode").nth(0).locator(".cfg-cfg").input_value() == "4.25"

    # save: the configuration carries the i2i options edit + the t2i CFG
    # edit, the flag stays enabled, and the section kept the remaining
    # Forge defaults
    page.fill(".cfg-name", "Soft config")
    page.click("#dialog-ok")
    assert wait_until(lambda: not _dialog_visible(page), timeout=5)
    with httpx.Client(base_url=base, timeout=10) as c:
        m = c.get("/api/models").json()
        e = [x for x in m["models"] if x["name"] == "Soft config"][0]
    assert e["t2i"]["soft"] is True and e["i2i"]["soft"] is True
    assert e["i2i"]["soft_mask_blend_power"] == 2.2
    assert e["t2i"]["cfg"] == 4.25
    assert (e["i2i"]["soft_mask_blend_power"],
            e["i2i"]["soft_mask_blend_scale"],
            e["i2i"]["soft_inpaint_detail_preservation"],
            e["i2i"]["soft_composite_mask_influence"],
            e["i2i"]["soft_composite_difference_threshold"],
            e["i2i"]["soft_composite_difference_contrast"]) \
        == (2.2, 0.5, 4.0, 0.0, 0.5, 2.0)

    # reopening prefills the checkbox and the edited options
    _open_config_dialog(page)
    assert page.locator(".cfg-soft").first.is_checked()
    page.locator(".cfg-soft-cfg").first.click()
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator("#dialog-body input[type=range]").first.input_value() == "2.2"


def test_old_config_without_the_key_is_enabled(page, server):
    """A configuration file predating the soft keys loads with the flag
    enabled and the Forge default options."""
    base, tmp = server
    cfg_dir = tmp / "user" / "model-configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "legacy-cfg.json").write_text(json.dumps({
        "name": "Legacy cfg", "model": "mock/mock.safetensors",
        "t2i": {"sampler": None, "scheduler": None, "loras": []},
        "i2i": {"sampler": None, "scheduler": None, "loras": []},
    }), encoding="utf-8")
    with httpx.Client(base_url=base, timeout=10) as c:
        e = [x for x in c.get("/api/models").json()["models"]
             if x["name"] == "Legacy cfg"][0]
    # the pydantic defaults fill in the missing keys: enabled + Forge options
    assert e["i2i"]["soft"] is True
    assert e["t2i"]["soft"] is True
    assert e["i2i"]["soft_mask_blend_power"] == 1.0
    assert e["i2i"]["soft_composite_difference_contrast"] == 2.0
    # the config dialog shows the checkbox checked
    _open(page, base)
    page.evaluate("selectModel('legacy-cfg')")
    _open_config_dialog(page)
    assert page.locator(".cfg-soft").count() == 1
    assert page.locator(".cfg-soft").first.is_checked()
    page.click("#dialog-cancel")
