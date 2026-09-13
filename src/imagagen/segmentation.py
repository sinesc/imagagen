"""Person mask generation via MediaPipe image segmenter models.

Port of the logic of the `a-person-mask-generator` Forge extension: a
MediaPipe *image segmenter* tflite model (e.g.
``selfie_multiclass_256x256.tflite``, whose classes are
background / hair / body-skin / face-skin / clothes / others) runs on the
current canvas image and the per-class *confidence* masks of the selected
classes are unioned (element-wise maximum) into the canvas mask buffer
as 0..255 alpha. Unlike the reference implementation (which thresholds
confidence at 0.25 into a binary mask) we keep the confidence as the
alpha, so the mask gets soft edges and the inpaint mask-blending uses
them. The union is then peak-normalized per run (highest confidence ->
full opacity), because the models' confidences are not calibrated to
1.0 — a "solid" region typically only reaches ~0.7–0.9, which would
render the mask core as semi-transparent.

Every ``*.tflite`` file in ``models/segmentation/`` is offered to the UI;
the class names come from the model itself (``ImageSegmenter.labels``),
so any multiclass segmenter works, not just the selfie one. MediaPipe is
imported lazily: nothing torch/mediapipe-related loads at import time or
server start (the mock-backend test suite never touches it).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Quiet the tflite runtime / glog chatter before mediapipe is first
# imported (otherwise every segmenter load prints "Created TensorFlow
# Lite XNNPACK delegate for CPU" and glog warnings to the console).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

log = logging.getLogger(__name__)


# The selfie model's per-class confidences are not sparse: every class
# scores ~0.017 (a baseline) on every pixel, with boundary pixels up to
# ~0.6. Below the threshold a class counts as "not there"; above it the
# raw confidence is kept as the mask alpha, so the mask gets soft edges
# (the reference implementation uses this same 0.25 cut, but binarizes
# everything instead).
CONFIDENCE_THRESHOLD = 0.25


class SegmentationError(Exception):
    """User-facing segmentation failure (no models dir, bad model/class)."""


@dataclass(frozen=True)
class SegmentationModel:
    """One .tflite file in models/segmentation/."""

    id: str  # file name (what the UI sends back)
    name: str  # file stem (display)
    labels: tuple[str, ...]  # the model's class names, in index order


class SegmentationService:
    """Loads segmenter models from ``models/segmentation/`` and masks images.

    Segmenters are created lazily on first use and cached per model. A
    MediaPipe task is not thread-safe, so model loading *and* inference
    are serialized by one lock (segmentation requests come from FastAPI's
    threadpool). ``segmenter_factory`` is injectable for tests; the
    default builds a real MediaPipe ``ImageSegmenter``.
    """

    def __init__(self, models_dir: Path, segmenter_factory=None) -> None:
        self.models_dir = Path(models_dir)
        self._factory = segmenter_factory or self._mediapipe_factory
        self._lock = threading.Lock()
        self._segmenters: dict[str, object] = {}

    # --- segmenter factory / input image (overridable in tests) ---
    @staticmethod
    def _mediapipe_factory(path: Path) -> object:
        import mediapipe as mp

        # Attribute access on purpose: `mediapipe.tasks` is an alias of
        # the `mediapipe.tasks.python` package (mediapipe 1.x), so
        # `import mediapipe.tasks.vision` fails but `mp.tasks.vision` works.
        vision = mp.tasks.vision
        options = vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        return vision.ImageSegmenter.create_from_options(options)

    def _make_image(self, rgb: np.ndarray) -> object:
        """Wrap an HxWx3 uint8 numpy array as the segmenter's input image."""
        import mediapipe as mp

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # --- public API ---
    def segmenter(self, model_id: str) -> object:
        """The (cached) loaded segmenter for ``models_dir/model_id``."""
        with self._lock:
            seg = self._segmenters.get(model_id)
            if seg is None:
                path = self.models_dir / model_id
                if not path.is_file():
                    raise SegmentationError(
                        f"unknown segmentation model: {model_id} "
                        f"(files in models/segmentation/)")
                try:
                    seg = self._factory(path)
                except Exception as e:
                    raise SegmentationError(
                        f"cannot load segmentation model {model_id}: {e}") from e
                self._segmenters[model_id] = seg
            return seg

    def list_models(self) -> list[SegmentationModel]:
        """Every readable .tflite in the models dir, with its class labels.

        A file that fails to load is skipped (logged) so one bad model
        does not take the whole dropdown down.
        """
        if not self.models_dir.is_dir():
            return []
        out: list[SegmentationModel] = []
        for p in sorted(self.models_dir.iterdir()):
            if p.suffix.lower() != ".tflite" or not p.is_file():
                continue
            try:
                seg = self.segmenter(p.name)
            except SegmentationError as e:
                log.warning("segmentation: skipping %s: %s", p.name, e)
                continue
            out.append(
                SegmentationModel(id=p.name, name=p.stem, labels=tuple(seg.labels)))
        return out

    def close(self) -> None:
        """Close all cached segmenters (server shutdown)."""
        with self._lock:
            for seg in self._segmenters.values():
                try:
                    seg.close()
                except Exception:  # noqa: S112 - shutdown only
                    pass
            self._segmenters.clear()

    def generate(
        self, model_id: str, classes: list[str], rgb: np.ndarray
    ) -> np.ndarray:
        """Segment ``rgb`` (HxWx3 uint8) and return the union mask of the
        selected classes as HxW uint8 alpha.

        Alpha = confidence * 255, peak-normalized per run so the highest
        confidence of the union maps to full opacity (the models' raw
        confidences top out well below 1.0).

        Raises :class:`SegmentationError` for unknown models, unknown
        classes, or an empty selection.
        """
        if not classes:
            raise SegmentationError("no classes selected")
        seg = self.segmenter(model_id)
        labels = list(seg.labels)
        missing = sorted({c for c in classes if c not in labels})
        if missing:
            raise SegmentationError(
                f"unknown class(es) for {model_id}: {', '.join(missing)}")
        rgb = np.ascontiguousarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise SegmentationError("input image must be HxWx3")
        with self._lock:
            out = seg.segment(self._make_image(rgb))
            # confidence masks come back as (H, W, 1); flatten to (H, W)
            confs = []
            for m in out.confidence_masks:
                a = np.asarray(m.numpy_view(), dtype=np.float32)
                if a.size == rgb.size // 3:
                    a = a.reshape(rgb.shape[:2])
                confs.append(a)
        # Union: element-wise maximum of the selected classes' confidences
        # (dedup preserves the user's order; order does not matter for max),
        # then drop the sub-threshold baseline ("not there" pixels -> 0)
        mask = None
        for name in dict.fromkeys(classes):
            a = np.clip(confs[labels.index(name)], 0.0, 1.0)
            mask = a if mask is None else np.maximum(mask, a)
        assert mask is not None
        mask = np.where(mask > CONFIDENCE_THRESHOLD, mask, 0.0)
        # Peak-normalize: the models' confidences are not calibrated to
        # 1.0 (solid regions top out ~0.7-0.9), so scale the union so its
        # highest confidence is fully opaque. (Monotone, so the threshold
        # cut above is unaffected.)
        peak = float(mask.max())
        if peak > 0.0:
            mask = mask / peak
        return (mask * 255.0).astype(np.uint8)
