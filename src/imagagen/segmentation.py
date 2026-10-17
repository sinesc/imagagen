"""Person mask generation via MediaPipe image segmenter models.

Port of the logic of the `a-person-mask-generator` Forge extension: a
MediaPipe *image segmenter* tflite model (e.g.
``selfie_multiclass_256x256.tflite``, whose classes are
background / hair / body-skin / face-skin / clothes / others) runs on the
current canvas image and the per-class *confidence* masks of the selected
classes are unioned (element-wise maximum). The union is then
**binarized** at a user threshold: confidence at or below the threshold
produces no mask pixel, everything above it is fully opaque — a mask with
a hard, fully opaque core. An optional *feather* radius Gaussian-blurs
that binary mask, which softens the edge (and only the edge: the core
stays fully opaque) so the inpaint blend gets a smooth transition.
Every ``*.tflite`` file in ``models/segmentation/`` is offered to the UI;
the class names come from the model itself (``ImageSegmenter.labels``),
so any multiclass segmenter works, not just the selfie one. MediaPipe is
imported lazily: nothing torch/mediapipe-related loads at import time or
server start (the mock-backend test suite never touches it).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Best-effort quieting of *other* native glog/TF C++ log levels (mediapipe
# is imported lazily, so these must be set before first use). They cannot
# reach the TFLite "Created XNNPACK delegate" printf or the glog
# inference_feedback_manager line (see _quiet_stderr), which is why the
# model load itself is wrapped in an fd-2 redirect.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_stderr():
    """Point fd 2 (stderr) at /dev/null for the enclosed block.

    TFLite/MediaPipe's native code writes its ``Created TensorFlow Lite
    XNNPACK delegate`` notice and glog/absl warnings (e.g.
    ``inference_feedback_manager.cc``) straight to the stderr *file
    descriptor*; env vars and Python ``logging``/``warnings`` filters all
    operate above the fd and cannot reach them. Redirecting fd 2 around
    the C model load is the only reliable way to keep that noise out of
    the console. This is process-wide and inherently not thread-safe, but
    segmenter loading runs under ``SegmentationService._lock`` and the
    window is a few milliseconds, so the only output at risk is a stray
    stderr write in that instant.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with open(os.devnull, "wb") as devnull:
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
    os.close(saved_fd)


# Default cut for the union confidence: the selfie model's per-class
# confidences are not sparse — every class scores ~0.017 (a baseline) on
# every pixel, with boundary pixels up to ~0.6 and solid regions around
# 0.7–0.9. Confidence at or below the cut produces no mask; above it the
# pixel is fully opaque (the UI exposes this as the "Threshold" field).
CONFIDENCE_THRESHOLD = 0.25

# Upper bound for the feather radius in canvas px (mirrors the UI input).
MAX_FEATHER = 64


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
        # The native load writes to the stderr fd directly; hide it.
        with _quiet_stderr():
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
        self,
        model_id: str,
        classes: list[str],
        rgb: np.ndarray,
        threshold: float = CONFIDENCE_THRESHOLD,
        feather: float = 0.0,
    ) -> np.ndarray:
        """Segment ``rgb`` (HxWx3 uint8) and return the union mask of the
        selected classes as HxW uint8 alpha.

        The union is cut at ``threshold`` (a confidence in 0..1): at or
        below it a pixel stays transparent, above it it is fully opaque.
        The models' confidences are not calibrated to 1.0 (a solid region
        tops out around 0.7–0.9), which is why the binarized mask is fully
        opaque instead of carrying the confidence as alpha. ``feather``
        (canvas px, 0 = off) Gaussian-blurs the resulting binary mask,
        softening only its edge.

        Raises :class:`SegmentationError` for unknown models, unknown
        classes, an empty selection, or an out-of-range threshold/feather.
        """
        if not classes:
            raise SegmentationError("no classes selected")
        if not 0.0 <= threshold <= 1.0:
            raise SegmentationError(
                f"threshold must be between 0.0 and 1.0 (got {threshold})")
        if not 0.0 <= feather <= MAX_FEATHER:
            raise SegmentationError(
                f"feather must be between 0 and {MAX_FEATHER} px "
                f"(got {feather})")
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
        # (dedup preserves the user's order; order does not matter for max)
        mask = None
        for name in dict.fromkeys(classes):
            a = np.clip(confs[labels.index(name)], 0.0, 1.0)
            mask = a if mask is None else np.maximum(mask, a)
        assert mask is not None
        # Cut at the threshold and binarize: the confidence is NOT kept as
        # the alpha (a solid region only reaches ~0.7-0.9, which would
        # render the core semi-transparent), so above the cut a pixel is
        # fully opaque. The cut runs on the raw confidence on purpose: a
        # per-run peak-normalization would make the threshold relative and
        # turn an all-noise run (a class that is not in the image at all)
        # into a fully masked canvas.
        mask = np.where(mask > threshold, 255.0, 0.0)
        if feather > 0.0:
            # Blur the binary mask: the edge softens, the opaque core
            # (interior) stays fully opaque.
            from PIL import Image, ImageFilter

            im = Image.fromarray(mask.astype(np.uint8), mode="L")
            im = im.filter(ImageFilter.GaussianBlur(radius=feather))
            mask = np.asarray(im, dtype=np.float64)
        return mask.astype(np.uint8)
