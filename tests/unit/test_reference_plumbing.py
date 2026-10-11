"""Phase 0 reference-image plumbing.

The reference image rides on the request exactly like the img2img base
(in-process numpy, never JSON), the queue passes it through unchanged, the
MockBackend echoes a caption so the round trip is testable without a GPU,
and the PNG metadata records the caption the backend applied.
"""

import numpy as np

from imagagen import metadata
from imagagen.backends import (GenerateRequest, MOCK_REFERENCE_CAPTION,
                               MockBackend, PHASE_REFERENCE)

from test_queue import _queue, _wait_done


def test_reference_image_round_trips_through_the_queue_dump():
    # queue._run_task rebuilds the per-task request via
    # model_dump(exclude=seed) + model_validate: the numpy reference image
    # must survive that (exactly like init_image)
    ref = np.zeros((8, 6, 3), dtype=np.uint8)
    req = GenerateRequest(prompt="x", width=8, height=8, reference_image=ref)
    d = req.model_dump(exclude={"seed"})
    d.update(seed=7)
    back = GenerateRequest.model_validate(d)
    assert back.reference_image is not None
    assert np.array_equal(back.reference_image, ref)
    assert back.caption_only is False


def test_reference_image_defaults_none_for_json_callers():
    req = GenerateRequest.model_validate({"prompt": "x"})
    assert req.reference_image is None
    assert req.caption_only is False


class _RecordingBackend(MockBackend):
    def __init__(self):
        super().__init__()
        self.seen_reference = "unset"
        self.seen_caption_only = None

    def generate(self, request, progress=None):
        self.seen_reference = request.reference_image
        self.seen_caption_only = request.caption_only
        return super().generate(request, progress)


def test_queue_passes_reference_image_to_the_backend():
    backend = _RecordingBackend()
    events = []
    q, _ = _queue(backend, events)
    ref = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
    job = q.submit(
        "m", GenerateRequest(prompt="x", width=8, height=8, steps=1,
                             seed=3, reference_image=ref), 1)
    _wait_done(q, job.id)
    assert backend.seen_reference is not None
    assert np.array_equal(backend.seen_reference, ref)
    assert backend.seen_caption_only is False
    # the mock echoes the reference into the caption, and the caption
    # rides onto the PNG metadata through the queue
    png = job.results[0]
    from io import BytesIO

    meta = metadata.read_metadata(png)
    assert meta["reference_caption"] == MOCK_REFERENCE_CAPTION


def test_mock_backend_echoes_reference_caption():
    b = MockBackend()
    ref = np.zeros((8, 8, 3), dtype=np.uint8)
    r = b.generate(GenerateRequest(prompt="x", width=8, height=8, steps=1,
                                   seed=1, reference_image=ref))
    assert r.reference_caption == MOCK_REFERENCE_CAPTION
    r = b.generate(GenerateRequest(prompt="x", width=8, height=8, steps=1,
                                   seed=1))
    assert r.reference_caption is None


def test_caption_only_returns_the_caption_without_an_image():
    b = MockBackend()
    ref = np.zeros((8, 8, 3), dtype=np.uint8)
    r = b.generate(GenerateRequest(prompt="x", width=8, height=8, seed=1,
                                   reference_image=ref, caption_only=True))
    assert r.reference_caption == MOCK_REFERENCE_CAPTION
    assert r.image.size == 0


def test_metadata_records_the_reference_caption():
    doc = metadata.build_parameters(
        prompt="p", negative_prompt="n", steps=10, sampler="Euler a",
        scheduler="Karras", cfg=7.0, seed=1, width=8, height=8,
        model_name="m", model_hash="h", reference_caption="a cat on a mat")
    parsed = metadata.parse_parameters(doc)
    assert parsed["reference_caption"] == "a cat on a mat"
    # absent when no reference was used
    doc = metadata.build_parameters(
        prompt="p", negative_prompt="n", steps=10, sampler="Euler a",
        scheduler="Karras", cfg=7.0, seed=1, width=8, height=8,
        model_name="m", model_hash="h")
    assert "reference_caption" not in metadata.parse_parameters(doc)


def test_reference_phase_constant():
    # the queue maps it to the "Describing reference image" status text
    assert PHASE_REFERENCE == "phase:reference"
