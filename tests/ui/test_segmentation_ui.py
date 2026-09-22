"""Person mask generation UI (Phase 5): model dropdown + class dialog.

Drives a real Chromium against a mock-backend server that carries the
*real* selfie tflite model (copied from the repo's models/segmentation/
if present): the mask pixels are asserted server-side (authoritative).
Skips when the model file, mediapipe, or Playwright's Chromium is
missing.
"""

import shutil
from pathlib import Path

import pytest
from conftest import mask_arr, wait_until

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

    # the model dropdown is populated from /api/segmentation/models
    sel = page.locator("#seg-model")
    assert wait_until(
        lambda: sel.locator("option").count() >= 1, timeout=30
    ), "segmentation model dropdown not populated"
    assert "selfie_multiclass_256x256" in sel.inner_text()

    # open the class dialog: 5 checkboxes (background excluded)
    page.click("#seg-run")
    page.wait_for_selector("#dialog:not([hidden])")
    boxes = page.locator("#dialog .seg-class input")
    assert boxes.count() == 5

    # keep only hair + face: uncheck body-skin, clothes, others
    for label in ("body-skin", "clothes", "others"):
        boxes.nth(_index_of(boxes, label)).uncheck()
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
    assert mask[180, 700] > 0, "face region should be masked"
    assert mask[700, 700] == 0, "clothes region should not be masked"
    # most of the image stays unmasked (a person is <50% of the frame)
    assert (mask > 0).mean() < 0.5

    # the mask write is one undo step. Undo follows the active tool, so
    # switch to the mask tool first (the default is the brush tool, whose
    # draw stack is empty).
    page.keyboard.press("m")
    page.click("#btn-undo")
    assert wait_until(
        lambda: (m := mask_arr(base)) is not None and m.max() == 0)


def _index_of(boxes, label) -> int:
    for i in range(boxes.count()):
        if label in boxes.nth(i).evaluate("el => el.parentElement.textContent"):
            return i
    raise AssertionError(f"no checkbox for {label!r}")
