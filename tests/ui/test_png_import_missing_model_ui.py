"""Browser tests for PNG imports whose checkpoint/LoRAs cannot be found.

An imagagen PNG records the model id (a path under models/<family>/), the
checkpoint FAMILY (model_family) and its LoRAs by name. Imports from
another install (different folder layout) or after a checkpoint rename
must communicate that instead of silently guessing:

* the recorded model id must exist HERE (no basename guessing);
* when it cannot be found, the image's family decides the fallback: same
  family -> keep the selected checkpoint with a red "Checkpoint not found,
  retaining selected checkpoint" status, other family -> select a
  configuration of the image's family;
* without any family info (foreign or old imagagen PNGs) the selected
  model's family is assumed (the model stays selected, red status);
* a LoRA whose file is discovered under NO family dir shows ❌ on the
  enable/disable button (not the foreign-family padlock 🔐) with the
  dropdown left ENABLED so the user can pick an existing file.
"""

import base64
import io

from conftest import combo_pick, wait_until
from PIL import Image

from imagagen import metadata as md


def _open(page, base, root, loras=("alpha", "beta")):
    (root / "loras" / "sdxl").mkdir(parents=True, exist_ok=True)
    for n in loras:
        (root / "loras" / "sdxl" / f"{n}.safetensors").write_bytes(b"")
    page.goto(base)
    page.wait_for_function(
        "brushNames.length >= 1 && state.loras.length >= " + str(len(loras)))


def _switch_family(page, family):
    """Simulate a model switch to another family (the mock registry only
    has one model, so inject a matching entry like the family-lock tests
    do)."""
    page.evaluate(
        """async (family) => {
            state.entries.push({
                id: 'fam-' + family, model_id: family + '/mock', name: 'fam',
                family, is_default: true,
            });
            document.getElementById('model').value = 'fam-' + family;
            await refreshLoras();
        }""", family)


def _meta_png(model, loras=(), model_family=None, forge_text=None):
    """PNG carrying imagagen JSON metadata (or raw Forge-style text when
    forge_text is given)."""
    if forge_text is None:
        params = md.build_parameters(
            "import test", "", 20, "Euler a", "Karras", 7.0, 1, 64, 64,
            model, "abc123", loras=list(loras), model_family=model_family)
    else:
        params = forge_text
    img, info = md.attach_metadata(Image.new("RGB", (64, 64), (8, 8, 8)),
                                   params)
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _drop_png(page, selector, png: bytes, name="drop.png"):
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


def _status(page):
    return page.evaluate(
        "(() => { const s = document.getElementById('status-line');"
        " return [s.textContent, s.classList.contains('error')]; })()")


def _row_full(page, index=-1):
    """[enabled, comboDisabled, toggleDisabled, toggleText, familyOff,
    name] of a sidebar LoRA row (see test_lora_toggle_ui; the name comes
    from the combobox VALUE, not the input's display text, which groups
    subfolder LoRAs under their folder)."""
    return page.evaluate(
        """(i) => {
            const rows = document.querySelectorAll('#loras .lora-row');
            const r = i < 0 ? rows[rows.length + i] : rows[i];
            const tog = r.querySelector('button.lora-toggle');
            return [
                !r.classList.contains('off'),
                r.querySelector('.combo-input').disabled,
                tog.disabled,
                tog.textContent,
                r.dataset.familyOff || null,
                r.querySelector('.lora-name').value,
            ];
        }""", index)


def test_missing_checkpoint_same_family_retains(page, server):
    """The recorded path does not exist here but the image's family equals
    the selected model's: the selection stays and the status reports it."""
    base, root = server
    _open(page, base, root)
    model_before = page.evaluate("$('model').value")
    _drop_png(page, "#prompt", _meta_png(
        "another-install/here/mock", loras=[("alpha", 0.8)],
        model_family="sdxl"))
    assert wait_until(
        lambda: _status(page) ==
        ["Checkpoint not found, retaining selected checkpoint", True])
    assert page.evaluate("$('model').value") == model_before
    # the LoRA itself exists: a normal enabled row (no lock, no ❌)
    assert wait_until(lambda: page.evaluate("getLoras()") ==
                      [["alpha", 0.8, True]])
    assert _row_full(page, 0) == [True, False, False, "\u25d0", None, "alpha"]


def test_missing_checkpoint_other_family_switches(page, server):
    """The image's family differs from the selected model's: select the
    first configuration of the image's family instead of keeping a model
    the image's LoRAs could never load for."""
    base, root = server
    _open(page, base, root)
    _switch_family(page, "flux")
    _drop_png(page, "#prompt", _meta_png(
        "gone/mock", model_family="sdxl"))
    assert wait_until(
        lambda: page.evaluate("$('model').value") ==
        page.evaluate("state.entries[0].id"))
    text, err = _status(page)
    assert err
    assert text.startswith("Checkpoint not found, selected") and \
        "(family sdxl)" in text


def test_legacy_png_family_derived_from_model_path(page, server):
    """imagagen PNGs written before the model_family key: the model id is
    a path whose first component is a family this install knows, so the
    family fallback still works (same family -> retain + message)."""
    base, root = server
    _open(page, base, root)
    _drop_png(page, "#prompt", _meta_png("sdxl-elsewhere/mock"))
    # the mock registry's family is "sdxl"; the path prefix "sdxl-elsewhere"
    # is NOT a family -> no derivation, but the current model's family is
    # assumed (requirement 2) and the selection retained
    assert wait_until(
        lambda: _status(page) ==
        ["Checkpoint not found, retaining selected checkpoint", True])
    _drop_png(page, "#prompt", _meta_png("sdxl/deeply/nested/mock"))
    assert wait_until(
        lambda: _status(page) ==
        ["Checkpoint not found, retaining selected checkpoint", True])


def test_foreign_png_unknown_model_assumes_current_family(page, server):
    """A1111/Forge PNG, unknown checkpoint: no family info at all -- the
    current model's family is assumed, the model stays selected."""
    base, root = server
    _open(page, base, root)
    _drop_png(page, "#prompt", _meta_png(
        None, forge_text="a cat\nNegative prompt: bad\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7.0, Seed: 5, "
        "Size: 64x64, Model: someOtherModel"))
    assert wait_until(
        lambda: _status(page) ==
        ["Checkpoint not found, retaining selected checkpoint", True])
    assert page.evaluate(
        "state.entries.some(e => e.id === $('model').value)")


def test_missing_lora_shows_error_toggle(page, server):
    """The checkpoint resolves exactly; one LoRA path does not exist here:
    that row gets ❌ (force-off, title explains), the dropdown stays
    enabled, and picking an existing file fixes the row."""
    base, root = server
    _open(page, base, root)
    _drop_png(page, "#prompt", _meta_png(
        "mock/mock.safetensors",  # the mock model id: exact match
        loras=[("alpha", 0.8), ("Optimizer/DMD2fp32XL1.0", 0.7)]))
    assert wait_until(lambda: page.evaluate("getLoras().length") == 2)
    # no checkpoint complaint: the import itself succeeded normally
    assert wait_until(lambda: "imported" in _status(page)[0] and
                      _status(page)[1] is False)
    assert _row_full(page, 0) == [True, False, False, "\u25d0", None, "alpha"]
    missing_row = wait_until(
        lambda: _row_full(page, 1) ==
        [False, False, True, "\u274c", "1", "Optimizer/DMD2fp32XL1.0"])
    assert missing_row, _row_full(page, 1)
    title = page.locator("#loras .lora-row").nth(1).locator(
        "button.lora-toggle").get_attribute("title")
    assert "could not be found" in title
    # the dropdown is openable (not disabled like the family padlock):
    # selecting an existing file re-enables the auto-disabled row
    combo_pick(page, page.locator("#loras .lora-row").nth(1), "beta")
    assert wait_until(lambda: _row_full(page, 1) ==
                      [True, False, False, "\u25d0", None, "beta"])
    assert page.evaluate("getLoras()") == [
        ["alpha", 0.8, True], ["beta", 0.7, True]]


def test_foreign_family_lora_still_gets_padlock(page, server):
    """A LoRA discovered under ANOTHER family's dir keeps the padlock (it
    is a family mismatch, not a missing file), dropdown disabled as
    before."""
    base, root = server
    (root / "loras" / "flux").mkdir(parents=True, exist_ok=True)
    (root / "loras" / "flux" / "gamma.safetensors").write_bytes(b"")
    _open(page, base, root)
    _drop_png(page, "#prompt", _meta_png(
        "mock/mock.safetensors", loras=[("gamma", 0.6)]))
    assert wait_until(lambda: page.evaluate("getLoras().length") == 1)
    assert wait_until(lambda: _row_full(page, 0) ==
                      [False, True, True, "\U0001f512", "1", "gamma"])
