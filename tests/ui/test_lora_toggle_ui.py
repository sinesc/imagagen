"""Browser tests for the LoRA on/off toggle.

Covers: the toggle button in each sidebar LoRA row (position, disabled UI
state of the dropdown), the generate body (enabled vs disabled_loras),
the ui-state round-trip, and restoring the disabled state on PNG import.
Uses the mock-backend server in a temp root.
"""

import io
import json

from conftest import combo_pick, set_input, wait_until


def _open(page, base, root, loras=("alpha", "beta")):
    (root / "loras" / "sdxl").mkdir(parents=True, exist_ok=True)
    for n in loras:
        (root / "loras" / "sdxl" / f"{n}.safetensors").write_bytes(b"")
    page.goto(base)
    page.wait_for_function(
        "brushNames.length >= 1 && state.loras.length >= " + str(len(loras)))


def _add_lora(page, name, weight=None):
    page.click("#lora-add")
    combo_pick(page, page.locator("#loras .lora-row").last, name)
    if weight is not None:
        set_input(page, "#loras .lora-row:last-child .lora-weight", str(weight))


def _toggle(page, index=-1):
    page.locator("#loras .lora-row").nth(index).locator("button.lora-toggle").click()


def _row_state(page, index=-1):
    """[enabled, comboDisabled] of a sidebar LoRA row."""
    return page.evaluate(
        """(i) => {
            const rows = document.querySelectorAll('#loras .lora-row');
            const r = i < 0 ? rows[rows.length + i] : rows[i];
            return [
                !r.classList.contains('off'),
                r.querySelector('.combo-input').disabled,
            ];
        }""", index)


def test_toggle_button_state_and_position(page, server):
    base, root = server
    _open(page, base, root)
    _add_lora(page, "alpha", 0.8)
    _add_lora(page, "beta", 0.5)
    wait_until(lambda: page.locator("#loras .lora-row").count() == 2)

    # the toggle sits between the weight input and the − button:
    # [combo, weight, toggle, remove]
    assert page.eval_on_selector(
        "#loras .lora-row:last-child",
        "r => [...r.children].map(c => c.classList.contains('lora-name') ? 'combo'\n"
        "  : c.classList.contains('lora-toggle') ? 'toggle'\n"
        "  : c.classList.contains('lora-weight') ? 'weight' : 'rm')") == \
        ["combo", "weight", "toggle", "rm"]

    # off: the row is dimmed and the dropdown uses the disabled UI state
    _toggle(page)
    assert _row_state(page) == [False, True]
    assert _row_state(page, 0) == [True, False]  # the other row is untouched

    # on again
    _toggle(page)
    assert _row_state(page) == [True, False]


def test_generate_sends_enabled_and_disabled_loras(page, server):
    base, root = server
    _open(page, base, root)
    _add_lora(page, "alpha", 0.8)
    _add_lora(page, "beta", 0.5)
    _toggle(page, 1)
    assert _row_state(page, 1) == [False, True]

    seen = {}

    def on_request(req):
        if req.url.endswith("/api/generate"):
            seen["body"] = json.loads(req.post_data)

    page.on("request", on_request)
    page.click("#btn-generate")
    wait_until(lambda: "body" in seen, timeout=30)
    # only the enabled LoRA is applied; the disabled one is recorded
    assert seen["body"]["loras"] == [["alpha", 0.8]]
    assert seen["body"]["disabled_loras"] == [["beta", 0.5]]


def test_ui_state_keeps_loras_disabled(page, server):
    base, root = server
    _open(page, base, root)
    _add_lora(page, "alpha", 0.8)
    _add_lora(page, "beta", 0.5)
    _toggle(page, 1)
    # ui-state is saved debounced; give it a beat before reloading
    page.wait_for_timeout(500)
    page.reload()
    wait_until(
        lambda: page.evaluate(
            "document.querySelectorAll('#loras .lora-row').length") == 2)
    assert _row_state(page, 0) == [True, False]
    assert _row_state(page, 1) == [False, True]
    # and the weights survived
    assert page.eval_on_selector_all(
        "#loras .lora-row .lora-weight", "els => els.map(e => e.value)") == \
        ["0.8", "0.5"]


def _loras_png(model_name, loras, disabled_loras):
    """PNG whose parameters chunk carries the given LoRA lines."""
    from PIL import Image

    from imagagen import metadata as md
    img, info = md.attach_metadata(
        Image.new("RGB", (64, 64), (240, 10, 40)),
        md.build_parameters("lora import test", "", 20, "Euler a", "Karras",
                            7.0, 1, 64, 64, model_name, "abc123",
                            loras=loras, disabled_loras=disabled_loras),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _drop_png(page, selector, png: bytes, name="drop.png"):
    import base64
    page.evaluate(
        """async ([sel, b64, name]) => {
            const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            const dt = new DataTransfer();
            dt.items.add(new File([bytes], name, {type: "image/png"}));
            const el = document.querySelector(sel);
            el.dispatchEvent(new DragEvent("dragover",
                {dataTransfer: dt, bubbles: true, cancelable: true}));
            el.dispatchEvent(new DragEvent("drop",
                {dataTransfer: dt, bubbles: true, cancelable: true}));
        }""",
        [selector, base64.b64encode(png).decode(), name],
    )


def test_import_restores_disabled_loras(page, server):
    base, root = server
    _open(page, base, root)
    _drop_png(page, "#prompt", _loras_png(
        "Mock Model", [("alpha", 0.8)], [("beta", 0.5), ("gamma", -0.3)]))
    # enabled rows first, disabled ones after (in their recorded order)
    assert wait_until(
        lambda: page.evaluate("getLoras()") ==
        [["alpha", 0.8, True], ["beta", 0.5, False], ["gamma", -0.3, False]])
    assert _row_state(page, 1) == [False, True]
    assert _row_state(page, 2) == [False, True]


def test_screenshot_alignment(page, server):
    """Visual check of the toggle button: same size as the − button, same
    vertical alignment (the plan calls for a screenshot review)."""
    base, root = server
    _open(page, base, root)
    _add_lora(page, "alpha", 0.8)
    _add_lora(page, "beta", 0.5)
    _toggle(page, 1)
    page.wait_for_timeout(200)
    page.locator("#controls").screenshot(path="/tmp/lora_toggle_rows.png")
