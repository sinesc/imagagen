"""Browser tests for the LoRA configuration UI (Chromium via Playwright).

Covers the "LoRA configuration" dialog (manage list + edit dialog), the
settings .json next to the .safetensors, and the preferred-weight default
when a LoRA is selected in the generation form. Uses the mock-backend
server in a temp root.
"""
import json

from conftest import combo_pick


def _seed_lora(root, name="DMD", weight=0.7):
    d = root / "loras" / "sdxl"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.safetensors").write_bytes(b"")
    (d / f"{name}.json").write_text(json.dumps({
        "description": "forge desc", "sd version": "SDXL",
        "activation text": "act", "preferred weight": weight,
        "negative text": "neg", "notes": "note",
    }))
    return d / f"{name}.json"


def test_lora_settings_ui(page, server):
    base, root = server
    jpath = _seed_lora(root)
    page.goto(base)
    page.wait_for_function(
        "state.loras.includes('DMD') && !!state.loraSettings['DMD']"
    )

    # selecting a LoRA defaults the weight to its preferred weight
    page.click("#lora-add")
    combo_pick(page, page.locator(".lora-row .lora-name"), "DMD")
    assert page.locator(".lora-weight").input_value() == "0.7"

    # the manage dialog: one row per LoRA with an edit button only
    # (no delete, no "+ New LoRA")
    page.click("#lora-manage")
    page.wait_for_function(
        "document.getElementById('dialog-title').textContent === 'LoRA configuration'"
    )
    row = page.locator("#dialog-body .preset-manage-row")
    assert row.count() == 1
    assert "DMD" in row.locator(".preset-manage-name").text_content()
    assert row.locator("button").count() == 1
    assert row.locator("button").get_attribute("title") == "edit"

    # the edit dialog: prefilled fields, readonly name, no delete anywhere
    row.locator("button").click()
    page.wait_for_function(
        "document.getElementById('dialog-title').textContent === 'Edit LoRA'"
    )
    body = "#dialog-body .style-dialog"
    name_in = page.locator(f"{body} label:has-text('name') input")
    assert name_in.input_value() == "DMD"
    assert name_in.is_disabled() or name_in.get_attribute("readonly") is not None
    page.locator(
        f"{body} label:has-text('description') textarea"
    ).wait_for(timeout=500)
    assert page.locator(f"{body} label:has-text('description') textarea").input_value() == "forge desc"
    assert page.locator(f"{body} label:has-text('preferred weight') input").input_value() == "0.7"
    assert page.locator(f"{body} label:has-text('activation text') textarea").input_value() == "act"
    assert page.locator(f"{body} label:has-text('negative text') textarea").input_value() == "neg"
    assert page.locator(f"{body} label:has-text('notes') textarea").input_value() == "note"

    # save: writes the .json next to the .safetensors, preserving unknown
    # extra fields like "sd version"
    page.locator(f"{body} label:has-text('description') textarea").fill("edited")
    page.locator(f"{body} label:has-text('preferred weight') input").fill("0.35")
    page.click("#dialog-ok")
    # saving returns to the management list dialog
    page.wait_for_function(
        "document.getElementById('dialog-title').textContent === 'LoRA configuration'",
        timeout=5000,
    )
    data = json.loads(jpath.read_text())
    assert data["description"] == "edited"
    assert data["preferred weight"] == 0.35
    assert data["sd version"] == "SDXL"

    # the fresh preferred weight is used for new selections
    page.click("#dialog-ok")  # Close the manage dialog
    page.wait_for_function("document.getElementById('dialog').hidden === true")
    page.locator(".lora-row .lora-rm").click()
    page.click("#lora-add")
    combo_pick(page, page.locator(".lora-row .lora-name"), "DMD")
    assert page.locator(".lora-weight").input_value() == "0.35"
