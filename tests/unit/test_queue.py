import time

from imagagen.backends import GenerateRequest, MockBackend
from imagagen.canvas import Canvas
from imagagen.queue import GenerationQueue


class FakeRegistry:
    def __init__(self, backend):
        self._b = backend
        self._id = None

    def list(self):
        return []

    def models(self):
        return []

    def current_id(self):
        return self._id

    def current_backend(self):
        return self._b

    def backend_for(self, model_id):
        self._id = model_id
        return self._b


def _queue(backend, events):
    canvas = Canvas()
    q = GenerationQueue(registry=FakeRegistry(backend), canvas=canvas, emit=events.append)
    q.start()
    return q, canvas


def _wait_done(q, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = q.job(job_id)
        if job and job.completed == job.batch_size:
            return job
        time.sleep(0.02)
    raise TimeoutError


def test_batch_expands_and_seeds_ascending():
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events)
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=50), 3)
    _wait_done(q, job.id)
    assert set(job.results) == {0, 1, 2}
    # image events carry ascending seeds in index order
    img_events = [e for e in events if e["type"] == "image"]
    assert sorted(e["seed"] for e in img_events) == [50, 51, 52]
    urls = [e["url"] for e in img_events]
    assert all(u.startswith("/api/thumb/1/") for u in urls)


def test_seed_minus_one_random():
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events)
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=-1), 2)
    _wait_done(q, job.id)
    started = next(e for e in events if e["type"] == "job_started")
    assert started["base_seed"] >= 0
    assert job.base_seed == started["base_seed"]


def test_first_image_auto_sets_canvas():
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 2)
    _wait_done(q, job.id)
    assert canvas.has_image()
    assert any(e["type"] == "canvas_set" for e in events)
    # canvas_set emitted exactly once (first image only)
    assert sum(1 for e in events if e["type"] == "canvas_set") == 1


def test_interrupt_drops_pending_and_inflight():
    backend = MockBackend(steps_delay=0.05)
    events = []
    q, _ = _queue(backend, events)
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=50, seed=1), 4)
    # wait until the first task is running (a progress event appears)
    deadline = time.time() + 5
    while time.time() < deadline and not any(e["type"] == "progress" for e in events):
        time.sleep(0.01)
    q.interrupt(job.id)
    time.sleep(0.5)
    assert job.interrupted
    assert job.results == {}
    assert any(e["type"] == "job_interrupted" for e in events)
    assert not any(e["type"] == "image" for e in events)


def test_progress_events_report_fraction():
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events)
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=4, seed=1), 1)
    _wait_done(q, job.id)
    fracs = [e["fraction"] for e in events if e["type"] == "progress"]
    assert fracs == [0.25, 0.5, 0.75, 1.0]
    assert any(e["type"] == "done" for e in events)


def test_result_pngs_have_parameters_chunk():
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events)
    job = q.submit(
        "m",
        GenerateRequest(prompt="turtle", width=8, height=8, steps=2, seed=3, cfg=5.0),
        1,
    )
    _wait_done(q, job.id)
    png = job.results[0]
    assert b"parameters" in png
    assert b"turtle" in png
