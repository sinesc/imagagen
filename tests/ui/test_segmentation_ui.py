"""Person mask generation UI (Phase 5): the segment dialog (model dropdown,
class checkboxes, threshold, feather) and the mask it writes.

Drives a real Chromium against a mock-backend server that carries the
*real* selfie tflite model (copied from the repo's models/segmentation/
if present): the mask pixels are asserted server-side (authoritative).
Skips when the model file, mediapipe, or Playwright's Chromium is
missing.
"""

import shutil
from pathlib import Path

import pytest
from conftest import flush, mask_arr, wait_until

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "segmentation" / "selfie_multiclass_256x256.tflite"
REF_IMAGE = REPO / "dev/references" / "woman-with-dog-mask-generator-testing.png"


@pytest.fixture
def seg_env(server):
    """Mock server whose models/segmentation/ holds the real selfie model."""
    if not MODEL.is_file() or not REF_IMAGE.is_file():
        pytest.skip("selfie model or reference image not present")
    shutil.copy(MODEL, server[1] / "models" / "segmentation" / MODEL.name)
    from io import BytesIO

    import httpx
    from PIL import Image

    buf = BytesIO()
    Image.open(REF_IMAGE).convert("RGB").save(buf, format="PNG")
    with httpx.Client(base_url=server[0], timeout=10) as c:
        r = c.post("/api/canvas/upload", content=buf.getvalue())
        assert r.status_code == 200, r.text
    yield server


def test_segment_mask_flow(seg_env, page):
    base, _ = seg_env
    page.goto(base)
    page.wait_for_selector("#toolbar")

    # the model dropdown lives in the dialog now, not in the toolbar
    assert page.locator("#toolbar select").filter(
        has=page.locator("#seg-model")).count() == 0

    # open the dialog: model dropdown, 5 class checkboxes (background
    # excluded), threshold + feather inputs
    page.click("#seg-run")
    page.wait_for_selector("#dialog:not([hidden])")
    model_sel = page.locator("#dialog select.seg-model")
    assert wait_until(
        lambda: model_sel.locator("option").count() >= 1, timeout=30
    ), "segmentation model dropdown not populated"
    assert "selfie_multiclass_256x256" in model_sel.inner_text()
    boxes = page.locator("#dialog .seg-class input")
    assert boxes.count() == 5
    assert page.locator("#dialog input.seg-threshold").count() == 1
    assert page.locator("#dialog input.seg-feather").count() == 1

    # keep only hair + face: uncheck body-skin, clothes, others
    for label in ("body-skin", "clothes", "others"):
        boxes.nth(_index_of(boxes, label)).uncheck()
    # a feathered edge (the mask core stays fully opaque, the border softens)
    page.fill("#dialog input.seg-feather", "6")
    page.click("#dialog-ok")  # "Generate"; the dialog stays open on failure
    assert wait_until(
        lambda: page.locator("#dialog").evaluate("el => el.hidden"), timeout=30
    ), "dialog should close after a successful segmentation"

    assert wait_until(
        lambda: (m := mask_arr(base)) is not None and m.max() > 0, timeout=30
    ), "mask was not written"
    mask = mask_arr(base)
    # hair+face only: the face/skirt (clothes) region must stay unmasked.
    # The woman's face is roughly at (700, 180); her skirt at (700, 700).
    assert mask[180, 700] == 255, "face region should be fully opaque"
    assert mask[700, 700] == 0, "clothes region should not be masked"
    # the feather produced partial-alpha edge pixels (a soft border)
    assert ((mask > 0) & (mask < 255)).sum() > 0, "feathered edge expected"
    # most of the image stays unmasked (a person is <50% of the frame)
    assert (mask > 0).mean() < 0.5

    # the mask tool is armed automatically: the mask overlay is only drawn
    # in mask mode, so the user would otherwise not see the result
    assert wait_until(
        lambda: page.locator("#tool-mask.active").count() == 1, timeout=30
    ), "segmentation should switch to the mask tool"

    # the mask write is one undo step. Undo follows the active tool, which
    # segmentation just made the mask tool (so its stack holds the write).
    page.click("#btn-undo")
    flush(page)  # undo re-dirties asynchronously
    assert wait_until(
        lambda: (m := mask_arr(base)) is not None and m.max() == 0)


def test_segment_dialog_values_are_persisted(seg_env, page):
    """Threshold + feather + the chosen model ride in the ui-state."""
    base, _ = seg_env
    page.goto(base)
    page.wait_for_selector("#toolbar")
    page.click("#seg-run")
    page.wait_for_selector("#dialog:not([hidden])")
    page.fill("#dialog input.seg-threshold", "0.7")
    page.fill("#dialog input.seg-feather", "12")
    page.click("#dialog-ok")
    assert wait_until(
        lambda: page.locator("#dialog").evaluate("el => el.hidden"), timeout=30)

    def ui_state():
        import httpx

        # GET /api/ui-state wraps the state object in {"state": {...}}
        return httpx.get(f"{base}/api/ui-state", timeout=10).json()["state"]

    assert wait_until(
        lambda: ui_state().get("seg_threshold") == 0.7
        and ui_state().get("seg_feather") == 12
        and ui_state().get("seg_model") == "selfie_multiclass_256x256.tflite",
        timeout=30,
    ), f"segmentation values not persisted: {ui_state()}"

    # a fresh page restores them into the dialog
    page.reload()
    page.wait_for_selector("#toolbar")
    page.click("#seg-run")
    page.wait_for_selector("#dialog:not([hidden])")
    assert page.input_value("#dialog input.seg-threshold") == "0.7"
    assert page.input_value("#dialog input.seg-feather") == "12"


def test_segment_hotkey_opens_dialog_and_enter_submits(seg_env, page):
    """T opens the segment dialog; ENTER in it fires Generate."""
    base, _ = seg_env
    page.goto(base)
    page.wait_for_selector("#toolbar")
    # the boot fetch of /api/segmentation/models is async; pressing T before
    # it lands would open the "no models found" fallback dialog
    assert wait_until(lambda: page.evaluate("state.segModels.length") > 0, timeout=30)
    page.keyboard.press("t")
    page.wait_for_selector("#dialog:not([hidden])")
    assert page.locator("#dialog .seg-class input").count() == 5

    # Enter submits (all classes are on by default); the dialog closes on
    # success and the mask is written
    page.keyboard.press("Enter")
    assert wait_until(
        lambda: page.locator("#dialog").evaluate("el => el.hidden"), timeout=30
    ), "Enter should submit the dialog"
    assert wait_until(
        lambda: (m := mask_arr(base)) is not None and m.max() > 0, timeout=30
    ), "mask was not written"


def _index_of(boxes, label) -> int:
    for i in range(boxes.count()):
        if label in boxes.nth(i).evaluate("el => el.parentElement.textContent"):
            return i
    raise AssertionError(f"no checkbox for {label!r}")
