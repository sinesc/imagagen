"""Browser tests for the Phase 4 configuration layer (Chromium via Playwright).

Covers what only a real browser can verify: the model selector listing
configurations, the model-config create/edit dialog, the LoRA list rows,
and preset select/application. Uses the mock-backend server in a temp root.
"""

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
    # sampler/scheduler offer "Use model default" first
    first = page.eval_on_selector_all("#sampler option", "els => els.map(e => e.textContent)")
    assert first[0] == "Use model default"
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
    page.locator(".cfg-mode").first.locator(".cfg-sampler").select_option("DPM++ 2M")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    import json

    on_disk = json.loads((root / "model-configs" / "mock-tuned.json").read_text())
    assert on_disk["t2i"]["cfg"] == 4.25
    assert on_disk["t2i"]["sampler"] == "DPM++ 2M"

    # re-open: values round-trip
    page.click("#cfg-edit")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    assert page.locator(".cfg-mode").first.locator(".cfg-cfg").input_value() == "4.25"
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
    page.locator("#loras .lora-row").locator(".lora-name").select_option("sdxl/dmd2")
    page.locator("#loras .lora-row").locator(".lora-weight").fill("0.8")
    assert page.evaluate("getLoras()") == [["sdxl/dmd2", 0.8]]

    # the choice persists in ui-state (debounced saves: wait for the final
    # one, since an earlier save may still carry an empty lora list)
    import json

    def loras_persisted():
        f = root / "ui-state.json"
        if not f.is_file():
            return False
        try:
            return json.loads(f.read_text())["loras"] == [["sdxl/dmd2", 0.8]]
        except (json.JSONDecodeError, KeyError):
            return False

    wait_until(loras_persisted)

    # remove the row again
    page.locator("#loras .lora-row .lora-rm").click()
    assert page.evaluate("getLoras()") == []


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
    # delete (confirm() auto-accept)
    _dialogs(page, "")
    page.locator(".preset-manage-row .mini").last.click()
    wait_until(lambda: page.evaluate("state.presets.length === 0"))
