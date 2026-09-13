"""Person mask generation (Phase 5 segmentation).

The MediaPipe segmenter is injected with fakes: these tests cover the
service's model discovery, class validation, union/alpha math, and the
canvas mask write (undo), without mediapipe or a real .tflite file. The
real model + inference is exercised by devtools/segmentation_check.py
and the (skipping) browser test in tests/ui/test_segmentation_ui.py.
"""

import numpy as np
import pytest

from imagagen.canvas import Canvas
from imagagen.segmentation import SegmentationError, SegmentationService


class FakeM:
    """Stands in for mediapipe's MatArray (has .numpy_view())."""

    def __init__(self, arr):
        self.arr = np.asarray(arr, dtype=np.float32)

    def numpy_view(self):
        return self.arr


class FakeSegmenter:
    """Returns pre-baked per-class confidence masks for one model id."""

    def __init__(self, labels, confs):
        self.labels = list(labels)
        self.confs = confs
        self.segments = 0

    def segment(self, image):
        self.segments += 1
        return type("Out", (), {"confidence_masks": [FakeM(c) for c in self.confs]})

    def close(self):
        pass


def make_service(tmp_path, models):
    """models: {filename: (labels, confs)} -> service with a fake factory."""
    (tmp_path / "models/segmentation").mkdir(parents=True, exist_ok=True)
    for name in models:
        (tmp_path / "models/segmentation" / name).write_bytes(b"tflite")

    def factory(path):
        key = path.name
        labels, confs = models[key]
        return FakeSegmenter(labels, confs)

    return SegmentationService(tmp_path / "models/segmentation", segmenter_factory=factory)


# ---------------- list_models ----------------

def test_list_models_empty_dir(tmp_path):
    (tmp_path / "seg").mkdir()
    svc = SegmentationService(tmp_path / "seg", segmenter_factory=lambda p: None)
    assert svc.list_models() == []


def test_list_models_missing_dir(tmp_path):
    svc = SegmentationService(tmp_path / "nope", segmenter_factory=lambda p: None)
    assert svc.list_models() == []


def test_list_models_discovers_tflite_with_labels(tmp_path):
    svc = make_service(tmp_path, {
        "selfie_multiclass_256x256.tflite": (
            ["background", "hair", "body-skin", "face-skin", "clothes", "others"],
            [],
        ),
    })
    # non-tflite files are ignored
    (tmp_path / "models/segmentation/notes.txt").write_text("x")
    models = svc.list_models()
    assert [m.id for m in models] == ["selfie_multiclass_256x256.tflite"]
    m = models[0]
    assert m.name == "selfie_multiclass_256x256"
    assert m.labels == (
        "background", "hair", "body-skin", "face-skin", "clothes", "others")


def test_list_models_skips_unloadable(tmp_path):
    (tmp_path / "models/segmentation").mkdir(parents=True)
    (tmp_path / "models/segmentation/bad.tflite").write_bytes(b"x")
    (tmp_path / "models/segmentation/good.tflite").write_bytes(b"x")

    def factory(path):
        if path.name == "bad.tflite":
            raise RuntimeError("corrupt model")
        return FakeSegmenter(["background", "hair"], [[], []])

    svc = SegmentationService(tmp_path / "models/segmentation", segmenter_factory=factory)
    assert [m.id for m in svc.list_models()] == ["good.tflite"]


# ---------------- generate ----------------

def test_generate_unions_classes_by_max_confidence(tmp_path):
    H, W = 4, 5
    # two classes with overlapping support: the union must take the max
    hair = np.zeros((H, W), np.float32)
    hair[0, :3] = 0.5
    face = np.zeros((H, W), np.float32)
    face[0, 2] = 0.9  # overlaps hair's support at (0,2)
    face[3, 4] = 0.3  # above the 0.25 confidence threshold
    svc = make_service(tmp_path, {
        "m.tflite": (["background", "hair", "face"], [[0.0], hair, face]),
    })
    rgb = np.zeros((H, W, 3), np.uint8)
    mask = svc.generate("m.tflite", ["hair", "face"], rgb)
    assert mask.shape == (H, W) and mask.dtype == np.uint8
    expected = (np.maximum(hair, face) / 0.9 * 255).astype(np.uint8)
    assert np.array_equal(mask, expected)
    # the overlap pixel keeps the higher confidence (0.9, the peak -> 255)
    assert mask[0, 2] == 255
    # and the peak normalizes the rest proportionally (0.5 -> 0.5/0.9)
    assert mask[0, 0] == 141


def test_generate_peak_normalized_to_full_opacity(tmp_path):
    # the models' confidences are not calibrated to 1.0 (solid regions
    # top out ~0.7-0.9): the highest confidence of the run must map to
    # a fully opaque mask
    conf = np.zeros((2, 3), np.float32)
    conf[0, :] = [0.2, 0.8, 0.4]
    svc = make_service(tmp_path, {"m.tflite": (["background", "hair"], [[0.0], conf])})
    mask = svc.generate("m.tflite", ["hair"], np.zeros((2, 3, 3), np.uint8))
    assert mask.max() == 255  # 0.8 -> 1.0
    assert mask[0, 1] == 255  # the peak pixel is fully opaque
    assert mask[0, 2] == 127  # 0.4 / 0.8 * 255 = 127.5, truncated on cast
    assert mask[0, 0] == 0    # 0.2 is below the 0.25 threshold
    assert mask[1].tolist() == [0, 0, 0]


def test_generate_single_class_is_confidence_alpha(tmp_path):
    conf = np.full((2, 2), 1.0, np.float32)
    svc = make_service(tmp_path, {"m.tflite": (["background", "hair"], [[0.0], conf])})
    mask = svc.generate("m.tflite", ["hair"], np.zeros((2, 2, 3), np.uint8))
    assert (mask == 255).all()


def test_generate_sub_threshold_confidence_is_zero(tmp_path):
    # the real model scores ~0.017 (baseline) on every pixel for every
    # class: below CONFIDENCE_THRESHOLD a class is "not there". An all-
    # sub-threshold run stays all zero (peak-normalization must not
    # divide by zero / rescale noise).
    conf = np.full((2, 2), 0.1, np.float32)
    svc = make_service(tmp_path, {"m.tflite": (["background", "hair"], [[0.0], conf])})
    mask = svc.generate("m.tflite", ["hair"], np.zeros((2, 2, 3), np.uint8))
    assert (mask == 0).all()


def test_generate_rejects_unknown_model(tmp_path):
    svc = make_service(tmp_path, {"m.tflite": (["background"], [[0.0]])})
    with pytest.raises(SegmentationError, match="unknown segmentation model"):
        svc.generate("other.tflite", ["background"], np.zeros((2, 2, 3), np.uint8))


def test_generate_rejects_unknown_class(tmp_path):
    svc = make_service(tmp_path, {"m.tflite": (["background", "hair"], [[0.0], [0.0]])})
    with pytest.raises(SegmentationError, match="unknown class.*cats"):
        svc.generate("m.tflite", ["hair", "cats"], np.zeros((2, 2, 3), np.uint8))


def test_generate_rejects_empty_selection(tmp_path):
    svc = make_service(tmp_path, {"m.tflite": (["background"], [[0.0]])})
    with pytest.raises(SegmentationError, match="no classes selected"):
        svc.generate("m.tflite", [], np.zeros((2, 2, 3), np.uint8))


# ---------------- canvas mask write ----------------

def test_canvas_set_mask_replaces_and_undoes():
    c = Canvas()
    img = np.full((4, 4, 3), 9, np.uint8)
    c.set_image(img)
    c.set_mask(np.full((4, 4), 30, np.uint8))  # some pre-existing mask
    assert (c.mask == 30).all()

    new = np.zeros((4, 4), np.uint8)
    new[1, 1] = 255
    assert c.set_mask(new)  # changed -> one undo step
    assert np.array_equal(c.mask, new)

    c.undo()
    assert (c.mask == 30).all()
    c.redo()
    assert np.array_equal(c.mask, new)
    # the image was not touched
    assert np.array_equal(c.image, img)


def test_canvas_set_mask_errors():
    c = Canvas()
    with pytest.raises(ValueError, match="empty canvas"):
        c.set_mask(np.zeros((4, 4), np.uint8))
    c.set_image(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError, match="does not match"):
        c.set_mask(np.zeros((3, 3), np.uint8))
