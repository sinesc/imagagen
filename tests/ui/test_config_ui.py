"""Browser tests for the Phase 4 configuration layer (Chromium via Playwright).

Covers what only a real browser can verify: the model selector listing
configurations, the model-config create/edit dialog, the LoRA list rows,
and preset select/application. Uses the mock-backend server in a temp root.
"""

import time

import httpx
from conftest import combo_items, combo_pick, combo_seq, wait_until


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
    # the model field is a searchable combobox: click opens its list
    assert combo_items(page, page.locator("#model")) == ["Mock Model (default)"]
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
    # creating one drops straight into its edit dialog (wait for the TITLE:
    # the hidden flag flips false -> true -> false in one go)
    wait_until(lambda: page.text_content("#dialog-title") == "Model configuration")
    assert page.locator(".cfg-name").input_value() == "Mock tuned"
    assert page.locator(".cfg-mode").count() == 2
    # the model field is a searchable combobox, NOT a <select>: assert on the
    # loaded model list (an "#model option" selector never matched anything,
    # so this check silently never ran -- and the edit dialog still covers the
    # combobox here, so opening its list is not an option either)
    wait_until(lambda: "Mock tuned" in page.evaluate("state.entries.map(e => e.name)"))
    assert (root / "user" / "model-configs" / "mock-tuned.json").is_file()
    assert page.evaluate("document.getElementById('model').value") == "mock-tuned"

    # the auto-opened edit dialog configures the t2i section (CFG override + sampler)
    assert page.locator(".cfg-distilled").count() == 0  # mock model is SDXL
    page.locator(".cfg-mode").first.locator(".cfg-cfg").fill("4.25")
    page.locator(".cfg-mode").first.locator(".cfg-steps").fill("12")
    page.locator(".cfg-mode").first.locator(".cfg-sampler").select_option("DPM++ 2M")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    import json

    on_disk = json.loads((root / "user" / "model-configs" / "mock-tuned.json").read_text())
    assert on_disk["distilled"] is True  # unchanged default (inert for SDXL)
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
    wait_until(lambda: (root / "user" / "model-configs" / "mock-model.json").exists())
    import json

    on_disk = json.loads((root / "user" / "model-configs" / "mock-model.json").read_text())
    assert on_disk["t2i"]["cfg"] == 4.5
    # the default entry is replaced by the file configuration (same name,
    # no more "(default)" suffix) and the selector follows the new id
    assert combo_items(page, page.locator("#model")) == ["Mock Model"]
    assert page.evaluate("document.getElementById('model').value") == "mock-model"


def test_delete_model_config(page, server):
    base, root = server
    _open(page, base)
    page.click("#cfg-create")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.locator(".cfg-name").fill("to-delete")
    page.click("#dialog-ok")
    # the edit dialog opens on the new configuration; close it before deleting
    wait_until(lambda: page.text_content("#dialog-title") == "Model configuration")
    page.click("#dialog-cancel")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    assert (root / "user" / "model-configs" / "to-delete.json").is_file()

    _dialogs(page, "")  # confirm(delete)
    page.click("#cfg-delete")
    wait_until(lambda: not (root / "user" / "model-configs" / "to-delete.json").exists())
    assert combo_items(page, page.locator("#model")) == ["Mock Model (default)"]


# ---------------- LoRAs ----------------


def test_lora_rows(page, server):
    base, root = server
    (root / "loras" / "sdxl").mkdir(parents=True)
    (root / "loras" / "sdxl" / "dmd2.safetensors").write_bytes(b"")
    _open(page, base, "state.loras.length >= 1")

    # add a row, pick the discovered LoRA, set a weight
    page.click("#lora-add")
    combo_pick(page, page.locator("#loras .lora-row .lora-name"), "dmd2")
    page.locator("#loras .lora-row").locator(".lora-weight").fill("0.8")
    assert page.evaluate("getLoras()") == [["dmd2", 0.8, True]]

    # the choice persists in ui-state (debounced saves: wait for the final
    # one, since an earlier save may still carry an empty lora list)
    import json

    def loras_persisted():
        f = root / "user" / "ui-state.json"
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
    combo_pick(page, page.locator("#preset"), "Small")
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


def test_preset_edit_renames_and_saves_in_place(page, server):
    base, _ = server
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/presets", json={
            "name": "SDXL 1024x1280 Standard", "width": 1024, "height": 1280,
            "controls": {"width": True, "height": True},
        })
        assert r.status_code == 200, r.text
    _open(page, base, "state.presets.length >= 1")

    # edit the preset and change its name: it renames, no second preset
    page.click("#preset-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.locator(".preset-manage-row .mini").first.click()
    wait_until(lambda: page.locator("#dialog-title").inner_text() == "Edit preset")
    page.locator(".preset-dialog label input").first.fill("1024x1280 Portrait")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate(
        "state.presets.map((p) => p.name).join(',')") == "1024x1280 Portrait")

    # saving an edit without changing the name still works (in place)
    page.locator(".preset-manage-row .mini").first.click()
    wait_until(lambda: page.locator("#dialog-title").inner_text() == "Edit preset")
    page.locator(".preset-row", has_text="width").first.locator(".preset-val").fill("768")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate(
        "state.presets.map((p) => p.name).join(',')") == "1024x1280 Portrait")
    with httpx.Client(base_url=base, timeout=10) as c:
        p = c.get("/api/presets/1024x1280-portrait", timeout=10).json()["preset"]
    assert p["width"] == 768 and p["height"] == 1280


def test_preset_selection_kept_cleared_on_edit_and_not_persisted(page, server):
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
    # the box starts at "—" even though the persisted state named one
    assert page.evaluate("document.getElementById('preset').value") == ""
    # width was persisted, so the restore happened (sanity)
    wait_until(lambda: page.input_value("#width") == "777")

    # selecting applies the preset AND keeps the selection (the form now
    # matches it)
    combo_pick(page, page.locator("#preset"), "Small")
    wait_until(lambda: page.input_value("#width") == "512")
    assert page.evaluate("document.getElementById('preset').value") == "Small"

    # the same preset can be selected again straight away (a pick always
    # fires change, even for the already-selected value)
    combo_pick(page, page.locator("#preset"), "Small")
    assert page.input_value("#width") == "512"

    # editing a field the preset covers clears the selection
    page.fill("#width", "400")
    wait_until(lambda: page.evaluate(
        "document.getElementById('preset').value") == "")

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

    # close the management list to get back to the main UI
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))

    # applying resets a concrete scheduler to "Model default"
    page.select_option("#scheduler", "Karras")
    combo_pick(page, page.locator("#preset"), "Reset sampler")
    wait_until(lambda: page.input_value("#scheduler") == "")
    assert page.locator("#scheduler option:checked").text_content() == "Model default"
    assert page.input_value("#sampler") == "Euler a"


def test_preset_manage_filter_survives_edit(page, server):
    base, _ = server
    with httpx.Client(base_url=base, timeout=10) as c:
        for name in ("Alpha One", "Beta Two"):
            r = c.post("/api/presets", json={
                "name": name, "width": 512, "controls": {"width": True}})
            assert r.status_code == 200, r.text
    _open(page, base, "state.presets.length >= 2")

    page.click("#preset-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    assert page.locator(".preset-manage-row").count() == 2

    # filter the list down to one preset (the filter is debounced)
    page.locator("#dialog-filter input").fill("alpha")
    wait_until(lambda: page.locator(".preset-manage-row").count() == 1)

    # edit that preset and save it unchanged: the management list reopens
    page.locator(".preset-manage-row .mini").first.click()
    wait_until(lambda: page.locator("#dialog-title").inner_text() == "Edit preset")
    page.click("#dialog-ok")
    wait_until(lambda: page.locator("#dialog-title").inner_text() == "Setting presets")

    # the previous filter is restored: list still filtered, text in the box
    assert page.locator(".preset-manage-row").count() == 1
    assert page.locator("#dialog-filter input").input_value() == "alpha"

    page.click("#dialog-ok")


def test_preset_clone_prefills_new_preset(page, server):
    # the clone button between edit and delete opens the create dialog with
    # the source's fields prefilled (name made unique, group/values/checkboxes
    # copied) and saving leaves the source untouched
    base, _ = server
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/presets", json={
            "name": "Src", "group": "Grp", "width": 768, "sampler": "Euler a",
            "controls": {"width": True, "sampler": True}})
        assert r.status_code == 200, r.text
        r = c.post("/api/presets", json={"name": "Src copy", "width": 512,
                                         "controls": {"width": True}})
        assert r.status_code == 200, r.text
    _open(page, base, "state.presets.length >= 2")

    page.click("#preset-manage")
    wait_until(lambda: page.locator(".preset-manage-row").count() == 2)
    # find the "Src" row by its exact name ("Src copy" also matches "Src")
    idx = page.evaluate(
        "[...document.querySelectorAll('.preset-manage-name')]"
        ".findIndex(e => e.textContent === 'Src')")
    row = page.locator(".preset-manage-row").nth(idx)
    assert row.locator(".preset-manage-name").inner_text() == "Src"
    assert [b.get_attribute("title") for b in row.locator("button").all()] == [
        "edit", "clone", "delete"]
    row.locator("button[title=clone]").click()

    # a new-preset dialog (not "Edit preset") prefilled from the source.
    # "Src copy" already exists, so the name becomes "Src copy 2"
    wait_until(lambda: page.locator("#dialog-title").inner_text() == "New preset")
    box = page.locator(".preset-dialog")
    name_in = box.locator("label input").first
    assert name_in.input_value() == "Src copy 2"
    assert box.locator("label").nth(1).locator("input").input_value() == "Grp"
    wrow = box.locator(".preset-row", has_text="width").first
    assert wrow.locator(".preset-cb").is_checked()
    assert wrow.locator(".preset-val").input_value() == "768"
    srow = box.locator(".preset-row", has_text="sampler").first
    assert srow.locator(".preset-cb").is_checked()
    assert srow.locator(".preset-val").input_value() == "Euler a"
    # unchecked/unset fields stay at their defaults
    assert not box.locator(
        ".preset-row", has_text="height"
    ).first.locator(".preset-cb").is_checked()

    # editing the clone's width leaves the source alone
    wrow.locator(".preset-val").fill("832")
    page.click("#dialog-ok")
    wait_until(lambda: page.locator("#dialog-title").inner_text() == "Setting presets")
    with httpx.Client(base_url=base, timeout=10) as c:
        clone = c.get("/api/presets/src-copy-2").json()["preset"]
        src = c.get("/api/presets/src").json()["preset"]
        copy = c.get("/api/presets/src-copy").json()["preset"]
    assert clone["name"] == "Src copy 2" and clone["group"] == "Grp"
    assert clone["width"] == 832 and clone["sampler"] == "Euler a"
    assert clone["controls"]["width"] and clone["controls"]["sampler"]
    assert not clone["controls"]["height"]
    assert src["width"] == 768 and copy["width"] == 512

    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))


def test_manage_dialogs_scroll_only_their_list(page, server):
    # long lists must not scroll the whole dialog: the title + filter stay
    # pinned at the top, the "+ New …" button and the actions at the bottom,
    # only .preset-manage scrolls (and the box still grows to its 88vh cap)
    base, _ = server
    with httpx.Client(base_url=base, timeout=10) as c:
        for i in range(40):
            r = c.post("/api/presets", json={
                "name": f"P{i:02d}", "width": 512, "controls": {"width": True}})
            assert r.status_code == 200, r.text
            r = c.post("/api/styles", json={"name": f"S{i:02d}", "prompt": "x"})
            assert r.status_code == 200, r.text
    _open(page, base, "state.presets.length >= 40 && state.styles.length >= 40")

    def check(open_selector, new_btn):
        page.click(open_selector)
        wait_until(lambda: page.locator(".preset-manage-row").count() >= 40)
        box = page.locator("#dialog-box")
        # the box itself never scrolls, its list does
        assert box.evaluate("e => e.scrollHeight <= e.clientHeight")
        lst = page.locator(".preset-manage")
        assert lst.evaluate("e => e.scrollHeight > e.clientHeight")
        # the dialog grew to its height cap but stays inside the viewport
        bb = box.bounding_box()
        assert bb["height"] > 0.8 * 800 and bb["y"] > 0 and bb["y"] + bb["height"] < 800
        # title row, filter, new button and Close are all inside the box
        for sel in ("#dialog-title-row", ".dialog-actions") + ((new_btn,) if new_btn else ()):
            r = page.locator(sel).bounding_box()
            assert r["y"] >= bb["y"] and r["y"] + r["height"] <= bb["y"] + bb["height"] + 1, sel
        # scrolling the list moves the rows but not the pinned parts
        btn_bb = new_btn and page.locator(new_btn).bounding_box()
        page.locator(".preset-manage").evaluate("e => e.scrollTop = e.scrollHeight")
        wait_until(lambda: page.locator(".preset-manage").evaluate("e => e.scrollTop") > 0)
        assert box.evaluate("e => e.scrollHeight <= e.clientHeight")
        if new_btn:
            assert page.locator(new_btn).bounding_box() == btn_bb
        assert page.locator("#dialog-title-row").bounding_box()["y"] >= bb["y"]
        page.click("#dialog-ok")  # Close
        wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))

    check("#preset-manage", "text=+ New preset")
    check("#style-manage", "text=+ New style")
    # the LoRA list has no new button, so only the actions are pinned
    page.evaluate("state.loras = Array.from({length: 40}, (_, i) => 'lora' + i)")
    check("#lora-manage", None)


def test_preset_dialog_has_no_soft_inpaint_row(page, server):
    # soft inpainting is not a preset field any more: its enable flag and
    # options live in the model configuration (like SAG), so the preset
    # dialog offers no checkbox row for it (and none for SAG either)
    base, _ = server
    _open(page, base)
    page.click("#preset-manage")
    page.click("text=+ New preset")
    box = page.locator(".preset-dialog")
    assert box.locator(".preset-row", has_text="soft inpainting").count() == 0
    assert box.locator(".preset-row", has_text="self attention guidance").count() == 0
    # the sidebar has no soft inpainting checkbox either (moved to the
    # model configuration dialog)
    assert page.locator("#soft-inpaint, #soft-inpaint-cfg").count() == 0
    page.click("#dialog-cancel")
    # cancelling the new-preset dialog returns to the preset manager (that
    # dialog's onClose re-opens the list), so #dialog stays visible: the old
    # "dialog is hidden again" wait never succeeded (10 s dead timeout)
    wait_until(lambda: page.text_content("#dialog-title") == "Setting presets")
    assert page.locator(".preset-dialog").count() == 0


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
    combo_pick(page, page.locator("#preset"), "Set defaults")
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

    # "Loose" is a plain item (no group header); the group's presets sit
    # under their group header in name order
    assert combo_seq(page, page.locator("#preset")) == [
        ["item", "Loose"],
        ["group", "Outpaint 1440p"],
        ["item", "Outpaint A"],
        ["item", "Outpaint B"],
    ]

    # the dialog offers the group field under the name; saving stores it
    page.click("#preset-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.click("text=+ New preset")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    box = page.locator(".preset-dialog")
    labels = box.locator("label").all_inner_texts()
    # (case-insensitive: dialog labels are capitalised)
    assert labels[0].strip().lower().startswith("name")
    assert labels[1].strip().lower().startswith("group")
    box.locator("label").nth(0).locator("input").fill("New grouped")
    box.locator("label").nth(1).locator("input").fill("Outpaint 1440p")
    page.click("#dialog-ok")
    # saving returns to the management list
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Setting presets")
    with httpx.Client(base_url=base, timeout=10) as c:
        p = c.get("/api/presets/new-grouped").json()["preset"]
    assert p["group"] == "Outpaint 1440p"
    # close the management list to get back to the main UI
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    # it now shows up in the existing group (name order inside)
    wait_until(lambda: combo_seq(page, page.locator("#preset")) == [
        ["item", "Loose"],
        ["group", "Outpaint 1440p"],
        ["item", "New grouped"],
        ["item", "Outpaint A"],
        ["item", "Outpaint B"],
    ])


# ---------------- PNG import restores the configuration ----------------


def _seed_import_setup(root):
    """Two configurations for the mock model (bare + DMD2@0.7) and the LoRA
    files the import references (dummy contents: only listed, never loaded)."""
    loras = root / "loras" / "sdxl"
    loras.mkdir(parents=True, exist_ok=True)
    for n in ("DMD2fp32XL1.0", "Extra"):
        (loras / f"{n}.safetensors").write_bytes(b"")


def _create_config(base, name, loras):
    r = httpx.post(base + "/api/model-configs", timeout=10, json={
        "name": name, "model": "mock/mock.safetensors",
        "t2i": {"loras": loras}, "i2i": {"loras": loras}})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_png_import_selects_the_matching_configuration(page, server):
    """applyPngMeta selects meta.config_id (the server resolves it: exact
    configuration name for imagagen PNGs, best checkpoint+LoRA match for
    A1111/Forge PNGs) and drops the configuration's own LoRAs from the UI
    list (they ride on the configuration, so applying them again would load
    them twice)."""
    base, root = server
    _seed_import_setup(root)
    # "A Mock (plain)" sorts first, so it is the initial selection
    _create_config(base, "A Mock (plain)", [])
    dmd = _create_config(base, "Mock (DMD)",
                         [{"name": "DMD2fp32XL1.0", "weight": 0.7}])
    _open(page, base)
    assert page.evaluate("document.getElementById('model').value") == "a-mock-plain"
    page.evaluate(
        """async () => {
            await applyPngMeta({
              model: "mock.safetensors", config_id: "mock-dmd",
              prompt: "a cat", loras: ["DMD2fp32XL1.0@0.7", "Extra@1"],
            }, 64, 64);
        }"""
    )
    assert page.evaluate("document.getElementById('model').value") == dmd
    # the config's DMD2@0.7 is dropped (the configuration provides it),
    # the extra UI LoRA stays
    wait_until(lambda: page.eval_on_selector_all(
        "#loras .lora-row", "els => els.map(e => e.querySelector('.lora-name').value)"
    ) == ["Extra"])


def test_png_import_without_config_id_keeps_the_model(page, server):
    """No config_id (metadata without a model, server match failed): the
    current configuration stays selected."""
    base, root = server
    _seed_import_setup(root)
    plain = _create_config(base, "Mock (plain)", [])
    _open(page, base)
    page.evaluate("async () => { await applyPngMeta({prompt: 'a cat'}, 64, 64); }")
    assert page.evaluate("document.getElementById('model').value") == plain
