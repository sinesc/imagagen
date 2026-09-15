"""Browser tests for the Phase 4 configuration layer (Chromium via Playwright).

Covers what only a real browser can verify: the model selector listing
configurations, the model-config create/edit dialog, the LoRA list rows,
and preset select/application. Uses the mock-backend server in a temp root.
"""

import time

import httpx
from conftest import wait_until


def _open(page, base, extra="state.entries.length >= 1"):
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1")
    page.wait_for_function(extra)


def _dialogs(page, prompt_reply: str):
    """One handler for both prompt() (reply with prompt_reply) and
    confirm() (accept). Register it before triggering the dialogs."""

    def handle(d):
        if d.type == "prompt":
            d.accept(prompt_reply)
        else:
            d.accept()

    page.on("dialog", handle)


# ---------------- model configurations ----------------


def test_model_selector_lists_default_entry(page, server):
    base, _ = server
    _open(page, base)
    texts = page.eval_on_selector_all("#model option", "els => els.map(e => e.textContent)")
    assert texts == ["Mock Model (default)"]
    # sampler/scheduler offer "Model default" first
    first = page.eval_on_selector_all("#sampler option", "els => els.map(e => e.textContent)")
    assert first[0] == "Model default"
    assert "Euler a" in first


def test_create_and_edit_model_config(page, server):
    base, root = server
    _open(page, base)
    # create dialog: name input prefilled with "<selected> (variant)"
    page.click("#cfg-create")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    assert "variant" in page.locator(".cfg-name").input_value()

    # confirm -> the file configuration exists and is selected
    page.locator(".cfg-name").fill("Mock tuned")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    wait_until(lambda: any("Mock tuned" in (t or "")
                           for t in page.eval_on_selector_all(
                               "#model option", "els => els.map(e => e.textContent)")))
    assert (root / "model-configs" / "mock-tuned.json").is_file()
    assert page.evaluate("document.getElementById('model').value") == "mock-tuned"

    # edit dialog: configure the t2i section (CFG override + sampler)
    page.click("#cfg-edit")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    assert page.locator(".cfg-mode").count() == 2
    page.locator(".cfg-mode").first.locator(".cfg-cfg").fill("4.25")
    page.locator(".cfg-mode").first.locator(".cfg-steps").fill("12")
    page.locator(".cfg-mode").first.locator(".cfg-sampler").select_option("DPM++ 2M")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    import json

    on_disk = json.loads((root / "model-configs" / "mock-tuned.json").read_text())
    assert on_disk["t2i"]["cfg"] == 4.25
    assert on_disk["t2i"]["steps"] == 12
    assert on_disk["t2i"]["sampler"] == "DPM++ 2M"

    # re-open: values round-trip
    page.click("#cfg-edit")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    assert page.locator(".cfg-mode").first.locator(".cfg-cfg").input_value() == "4.25"
    assert page.locator(".cfg-mode").first.locator(".cfg-steps").input_value() == "12"
    assert page.locator(".cfg-mode").first.locator(".cfg-sampler").input_value() == "DPM++ 2M"
    page.click("#dialog-cancel")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))


def test_edit_default_model_config_materializes_it(page, server):
    base, root = server
    _open(page, base)
    # the edit dialog also works on the in-memory default; saving
    # materializes it into a file configuration (the reported bug)
    page.click("#cfg-edit")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.locator(".cfg-mode").first.locator(".cfg-cfg").fill("4.5")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    wait_until(lambda: (root / "model-configs" / "mock-model.json").exists())
    import json

    on_disk = json.loads((root / "model-configs" / "mock-model.json").read_text())
    assert on_disk["t2i"]["cfg"] == 4.5
    # the default entry is replaced by the file configuration (same name,
    # no more "(default)" suffix) and the selector follows the new id
    texts = page.eval_on_selector_all("#model option", "els => els.map(e => e.textContent)")
    assert texts == ["Mock Model"]
    assert page.evaluate("document.getElementById('model').value") == "mock-model"


def test_delete_model_config(page, server):
    base, root = server
    _open(page, base)
    page.click("#cfg-create")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.locator(".cfg-name").fill("to-delete")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    assert (root / "model-configs" / "to-delete.json").is_file()

    _dialogs(page, "")  # confirm(delete)
    page.click("#cfg-delete")
    wait_until(lambda: not (root / "model-configs" / "to-delete.json").exists())
    texts = page.eval_on_selector_all("#model option", "els => els.map(e => e.textContent)")
    assert texts == ["Mock Model (default)"]


# ---------------- LoRAs ----------------


def test_lora_rows(page, server):
    base, root = server
    (root / "loras" / "sdxl").mkdir(parents=True)
    (root / "loras" / "sdxl" / "dmd2.safetensors").write_bytes(b"")
    _open(page, base, "state.loras.length >= 1")

    # add a row, pick the discovered LoRA, set a weight
    page.click("#lora-add")
    page.locator("#loras .lora-row").locator(".lora-name").select_option("dmd2")
    page.locator("#loras .lora-row").locator(".lora-weight").fill("0.8")
    assert page.evaluate("getLoras()") == [["dmd2", 0.8]]

    # the choice persists in ui-state (debounced saves: wait for the final
    # one, since an earlier save may still carry an empty lora list)
    import json

    def loras_persisted():
        f = root / "ui-state.json"
        if not f.is_file():
            return False
        try:
            return json.loads(f.read_text())["loras"] == [["dmd2", 0.8]]
        except (json.JSONDecodeError, KeyError):
            return False

    wait_until(loras_persisted)

    # remove the row again
    page.locator("#loras .lora-row .lora-rm").click()
    assert page.evaluate("getLoras()") == []

    # in the config dialog the lora dropdown must dominate the row (the
    # .cfg-mode width:100% rule used to squash it to ~15px and stretch the
    # weight input instead)
    page.click("#cfg-edit")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.locator(".cfg-mode").first.locator("button:has-text('+ LoRA')").first.click()
    wait_until(lambda: page.locator(".cfg-mode .lora-row").count() >= 1)
    sel_w, in_w = page.locator(".cfg-mode").first.locator(".lora-row").evaluate(
        """r => [
            r.querySelector('.lora-name').getBoundingClientRect().width,
            r.querySelector('.lora-weight').getBoundingClientRect().width,
        ]""")
    assert sel_w > 4 * in_w, (sel_w, in_w)
    page.click("#dialog-cancel")


# ---------------- presets ----------------


def test_preset_apply_and_manage(page, server):
    base, root = server
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/presets", json={
            "name": "Small", "width": 512, "steps": 30,
            "controls": {"width": True, "steps": True},
        })
        assert r.status_code == 200, r.text
    _open(page, base, "state.presets.length >= 1")

    # selecting the preset applies only its checked settings
    page.select_option("#preset", "Small")
    wait_until(lambda: page.evaluate("document.getElementById('width').value") == "512")
    wait_until(lambda: page.evaluate("document.getElementById('steps').value") == "30")
    # unchecked settings untouched
    assert page.evaluate("document.getElementById('height').value") == "1024"

    # management dialog: edit + delete
    page.click("#preset-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    assert page.locator(".preset-manage-row").count() == 1
    # the close-only list dialog has no redundant Cancel button
    assert not page.locator("#dialog-cancel").is_visible()
    _dialogs(page, "")
    page.locator(".preset-manage-row .mini").last.click()
    wait_until(lambda: page.evaluate("state.presets.length === 0"))


def test_preset_dropdown_resets_after_apply_and_not_persisted(page, server):
    base, _ = server
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/presets", json={
            "name": "Small", "width": 512,
            "controls": {"width": True},
        })
        assert r.status_code == 200, r.text
        # an old state file may still carry a preset selection: ignored
        r = c.post("/api/ui-state", json={"preset": "Small", "width": 777})
        assert r.status_code == 200
    _open(page, base, "state.presets.length >= 1")
    # the dropdown starts at "—" even though the persisted state named one
    assert page.input_value("#preset") == ""
    # width was persisted, so the restore happened (sanity)
    wait_until(lambda: page.input_value("#width") == "777")

    # selecting applies the preset AND resets the dropdown to "—"
    page.select_option("#preset", "Small")
    wait_until(lambda: page.input_value("#width") == "512")
    wait_until(lambda: page.input_value("#preset") == "")

    # the same preset can be selected again straight away (change fires
    # because the dropdown was reset, not stuck on "Small")
    page.select_option("#preset", "Small")
    wait_until(lambda: page.input_value("#preset") == "")
    assert page.input_value("#width") == "512"

    # the selection itself was not persisted
    wait_until(lambda: (
        httpx.get(base + "/api/ui-state", timeout=10).json()["state"].get("preset") is None))


def test_preset_dialog_sampler_use_model_default(page, server):
    base, _ = server
    _open(page, base)

    # the dialog's sampler/scheduler offer the same "Model default"
    # option as the main UI, and saving it stores "" (not null)
    page.click("#preset-manage")
    page.click("text=+ New preset")
    box = page.locator(".preset-dialog")
    box.locator("label input").first.fill("Reset sampler")
    srow = box.locator(".preset-row", has_text="sampler").first
    assert srow.locator(".preset-val option").first.text_content() == "Model default"
    srow.locator(".preset-cb").check()
    srow.locator(".preset-val").select_option("Euler a")
    # scheduler stays at "Model default" (index 0), checked as well
    hrow = box.locator(".preset-row", has_text="scheduler").first
    hrow.locator(".preset-cb").check()
    page.click("#dialog-ok")
    # saving returns to the management list (the new preset is re-listed)
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Setting presets")
    with httpx.Client(base_url=base, timeout=10) as c:
        p = c.get("/api/presets/reset-sampler").json()["preset"]
    assert p["sampler"] == "Euler a"
    assert p["scheduler"] == ""

    # applying resets a concrete scheduler to "Model default"
    page.select_option("#scheduler", "Karras")
    page.select_option("#preset", "Reset sampler")
    wait_until(lambda: page.input_value("#scheduler") == "")
    assert page.locator("#scheduler option:checked").text_content() == "Model default"
    assert page.input_value("#sampler") == "Euler a"


def test_preset_soft_inpaint_sag_checkbox_rows(page, server):
    # the dialog gains two rows, each with the usual controls checkbox
    # PLUS a second checkbox that is the enable/disable value itself;
    # applying the preset toggles the sidebar checkboxes
    base, _ = server
    _open(page, base)
    page.click("#preset-manage")
    page.click("text=+ New preset")
    box = page.locator(".preset-dialog")
    box.locator("label input").first.fill("Enable SAG")
    srow = box.locator(".preset-row", has_text="soft inpainting")
    assert srow.locator("input[type=checkbox]").count() == 2
    srow.locator(".preset-cb").check()        # the preset controls it
    srow.locator(".preset-val").check()       # value: enable
    arow = box.locator(".preset-row", has_text="self attention guidance")
    assert arow.locator("input[type=checkbox]").count() == 2
    arow.locator(".preset-cb").check()        # value stays unchecked:
    # the preset actively disables SAG
    page.click("#dialog-ok")
    # saving returns to the management list (the new preset is re-listed)
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Setting presets")
    with httpx.Client(base_url=base, timeout=10) as c:
        p = c.get("/api/presets/enable-sag").json()["preset"]
    assert p["soft_inpaint"] is True and p["sag"] is False
    assert p["controls"]["soft_inpaint"] is True and p["controls"]["sag"] is True

    # the edit dialog prefills both checkboxes (the management list is
    # still open from the save)
    wait_until(lambda: page.locator(".preset-manage-row", has_text="Enable SAG").count() == 1)
    page.locator(".preset-manage-row", has_text="Enable SAG") \
        .locator(".mini").first.click()
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Edit preset")
    # the edit dialog saves, so it keeps its Cancel button
    assert page.locator("#dialog-cancel").is_visible()
    box = page.locator(".preset-dialog")
    srow = box.locator(".preset-row", has_text="soft inpainting")
    assert srow.locator(".preset-cb").is_checked()
    assert srow.locator(".preset-val").is_checked()
    arow = box.locator(".preset-row", has_text="self attention guidance")
    assert arow.locator(".preset-cb").is_checked()
    assert not arow.locator(".preset-val").is_checked()
    page.click("#dialog-cancel")
    # cancelling also returns to the management list
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Setting presets")
    # close the management list to get back to the main UI
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))

    # applying toggles the sidebar checkboxes: SAG was actively disabled
    # (checked -> off), soft inpainting enabled (off -> checked, even
    # though it is disabled in "generate" mode)
    page.check("#sag")
    page.select_option("#preset", "Enable SAG")
    wait_until(lambda: not page.is_checked("#sag"))
    wait_until(lambda: page.is_checked("#soft-inpaint"))


def test_preset_multiword_name_select(page, server):  
    # regression: preset names containing spaces 404'd on select because the
    # store looked up by sanitized file id while the UI holds display names
    base, _ = server
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/presets", json={
            "name": "Set defaults", "steps": -1, "cfg": -1.0,
            "controls": {"steps": True, "cfg": True},
        })
        assert r.status_code == 200, r.text
    _open(page, base, "state.presets.length >= 1")
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text)
            if m.type == "error" else None)
    page.on("pageerror", lambda e: console_errors.append(str(e)))
    page.select_option("#preset", "Set defaults")
    wait_until(lambda: page.evaluate("document.getElementById('steps').value") == "-1")
    wait_until(lambda: page.evaluate("document.getElementById('cfg').value") == "-1")
    time.sleep(0.3)
    # the fetch no longer 404s and applyPreset no longer throws
    assert console_errors == []


def test_preset_groups_in_dropdown_and_dialog(page, server):
    # the main UI's preset dropdown renders one optgroup per group;
    # ungrouped presets are plain options. The edit dialog has a "group"
    # field under the name.
    base, _ = server
    with httpx.Client(base_url=base, timeout=10) as c:
        for name, group in [("Outpaint A", "Outpaint 1440p"),
                            ("Outpaint B", "Outpaint 1440p"),
                            ("Loose", None)]:
            r = c.post("/api/presets", json={
                "name": name, "group": group,
                "width": 512, "controls": {"width": True},
            })
            assert r.status_code == 200, r.text
    _open(page, base, "state.presets.length >= 3")

    # "Loose" is a plain option (not in an optgroup); the group's presets
    # sit under the optgroup in name order
    loose = page.locator("#preset > option", has_text="Loose")
    assert loose.count() == 1
    og = page.locator('#preset optgroup[label="Outpaint 1440p"]')
    assert og.count() == 1
    assert og.locator("option").count() == 2
    assert og.locator("option").all_inner_texts() == ["Outpaint A", "Outpaint B"]

    # the dialog offers the group field under the name; saving stores it
    page.click("#preset-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.click("text=+ New preset")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    box = page.locator(".preset-dialog")
    labels = box.locator("label").all_inner_texts()
    assert labels[0].strip().startswith("name")
    assert labels[1].strip().startswith("group")
    box.locator("label").nth(0).locator("input").fill("New grouped")
    box.locator("label").nth(1).locator("input").fill("Outpaint 1440p")
    page.click("#dialog-ok")
    # saving returns to the management list
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Setting presets")
    with httpx.Client(base_url=base, timeout=10) as c:
        p = c.get("/api/presets/new-grouped").json()["preset"]
    assert p["group"] == "Outpaint 1440p"
    # it now shows up in the existing optgroup
    wait_until(lambda: page.locator(
        '#preset optgroup[label="Outpaint 1440p"] option').count() == 3)
