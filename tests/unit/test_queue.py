import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from imagagen import metadata
from imagagen.backends import GenerateRequest, MockBackend
from imagagen.canvas import Canvas
from imagagen.generation import InpaintPlan
from imagagen.queue import GenerationQueue


def _png(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class FakeRegistry:
    """Minimal stand-in for the (multi-GPU) ModelRegistry.

    ``gpus`` / ``primary`` describe the pool the queue should dispatch
    over; ``uses`` records ``(gpu, seed)`` in the order generations start,
    so distribution across GPUs can be asserted deterministically.
    """

    def __init__(self, backend, gpus=(0,)):
        self._b = backend
        self.gpus = list(gpus)
        self.primary = self.gpus[0] if self.gpus else None
        self._id = None
        self.uses = []
        self.model_uses = []
        self.cancels = []
        self._lk = threading.Lock()

    def list(self):
        return []

    def models(self):
        return []

    def get(self, model_id):
        return SimpleNamespace(name="Fake Model")

    def current_id(self):
        return self._id

    def current_backend(self):
        return self._b

    def backend_for(self, model_id, gpu=None):
        self._id = model_id
        return self._b

    def run(self, model_id, gpu=None, request=None, progress=None):
        with self._lk:
            self._id = model_id
            self.uses.append((gpu, request.seed))
            self.model_uses.append((gpu, model_id))
        return self._b.generate(request, progress), self._b

    def cancel(self, gpu=None):
        self.cancels.append(gpu)
        self._b.cancel()


def _queue(backend, events, wildcards_dir=None, gpus=None):
    canvas = Canvas()
    q = GenerationQueue(
        registry=FakeRegistry(backend, gpus or (0,)), canvas=canvas,
        emit=events.append, wildcards_dir=wildcards_dir,
    )
    q.start(gpus=gpus)  # None -> FakeRegistry.gpus -> (0,)
    return q, canvas


def _wait_done(q, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = q.job(job_id)
        if job and job.completed == job.batch_size and job.done_emitted:
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
    # generate on an EMPTY canvas auto-sets the first result (there is
    # no displayed image to retain)
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 2)
    _wait_done(q, job.id)
    assert canvas.has_image()
    assert not any(e["type"] == "source_image" for e in events)
    assert any(e["type"] == "canvas_set" for e in events)
    # canvas_set emitted exactly once (first image only) and carries the
    # per-image undo/redo flags of the (fresh) result entity
    cs = [e for e in events if e["type"] == "canvas_set"]
    assert len(cs) == 1
    assert cs[0]["job"] == job.id
    assert cs[0]["undo_mask"] is False and cs[0]["redo_mask"] is False
    assert cs[0]["undo_draw"] is False and cs[0]["redo_draw"] is False


def test_generate_retains_nonempty_canvas_as_source_thumbnail():
    """Generate on a non-empty canvas does NOT auto-set: the displayed
    image is kept until the user selects a result, and is published as
    the source thumbnail."""
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    canvas.import_image(_png(np.full((8, 8, 3), 5, dtype=np.uint8)))
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 2)
    _wait_done(q, job.id)
    # canvas untouched (no auto-set, no canvas_set event)
    np.testing.assert_array_equal(canvas.image, np.full((8, 8, 3), 5, dtype=np.uint8))
    assert not any(e["type"] == "canvas_set" for e in events)
    # the canvas snapshot is the job's source image
    src = [e for e in events if e["type"] == "source_image"]
    assert len(src) == 1
    assert job.source_png is not None
    # the source event references the canvas image entity
    assert src[0]["image_id"] == canvas.current_id()


def test_inpaint_retains_canvas_until_thumb_selected():
    # inpaint jobs do not auto-set either: the canvas keeps the displayed
    # image (published as the source thumbnail) until the user selects a
    # result
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    base = np.full((8, 8, 3), 5, dtype=np.uint8)
    canvas.import_image(_png(base))
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3:5, 3:5] = 255
    plan = InpaintPlan("inpaint_masked", base, mask, bbox=(0, 0, 8, 8))
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1, plan=plan)
    _wait_done(q, job.id)
    # canvas untouched, no auto-set
    np.testing.assert_array_equal(canvas.image, base)
    assert not any(e["type"] == "canvas_set" for e in events)
    # the source thumbnail is published, so the user can go back
    next(e for e in events if e["type"] == "source_image")


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


def test_image_events_carry_image_id():
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 2)
    _wait_done(q, job.id)
    img_events = [e for e in events if e["type"] == "image"]
    assert len(img_events) == 2
    # each result is registered as a canvas entity referenced by the event;
    # key on the event's own index, NOT its position in the list -- the
    # queue runs one worker per GPU, so a batch's tasks complete (and emit
    # events) in racy order
    assert {e["index"] for e in img_events} == {0, 1}
    for e in img_events:
        assert e["image_id"] == f"job:{job.id}:{e['index']}"
        assert e["image_id"] in canvas._images


def test_finished_jobs_dropped_on_next_submit():
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events)
    j1 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, j1.id)
    assert j1.id in q.jobs()  # present until the next submit
    j2 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=2), 1)
    _wait_done(q, j2.id)
    # the finished first job was dropped when the second was submitted;
    # the in-flight/finished-but-newest job stays
    assert j1.id not in q.jobs()
    assert j2.id in q.jobs()


def test_wildcards_resolved_per_seed_and_stored_in_metadata(tmp_path):
    # with a wildcards_dir, each task's prompt is resolved with its own seed;
    # the resolved text (not the raw prompt) is what lands in the PNG metadata
    (tmp_path / "mood.txt").write_text("sunny\nstormy\n", encoding="utf-8")
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events, wildcards_dir=tmp_path)
    job = q.submit(
        "m",
        GenerateRequest(prompt="a __mood__ day", width=8, height=8, steps=1, seed=10),
        2,
    )
    _wait_done(q, job.id)
    prompts = []
    for i in (0, 1):
        meta = metadata.read_metadata(job.results[i])
        assert meta["prompt"].startswith("a ") and meta["prompt"].endswith(" day")
        assert meta["prompt"].split()[1] in {"sunny", "stormy"}
        assert "__mood__" not in meta["prompt"]  # fully resolved, not raw
        prompts.append(meta["prompt"])
    # the two tasks used different seeds -> they may (and across many runs do)
    # resolve to different lines; at minimum both are valid resolutions
    assert all(p in ("a sunny day", "a stormy day") for p in prompts)


def test_wildcard_error_settles_task_with_error_event(tmp_path):
    # a per-task WildcardError (shouldn't normally reach the worker, since it
    # is validated at submit) is surfaced as an error event, not a crash
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events, wildcards_dir=tmp_path)
    job = q.submit(
        "m",
        GenerateRequest(prompt="a __ghost__ day", width=8, height=8, steps=1, seed=1),
        1,
    )
    # an error settles the task (finished) without producing a result, so
    # wait on done_emitted rather than completed==batch_size
    deadline = time.time() + 10
    while time.time() < deadline and not job.done_emitted:
        time.sleep(0.02)
    assert job.done_emitted
    assert any(e["type"] == "error" and "not found" in e["message"] for e in events)
    assert job.results == {}


def test_live_image_ids_contains_results_and_source():
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    canvas.import_image(_png(np.full((8, 8, 3), 7, np.uint8)))
    src_id = canvas.current_id()
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, job.id)
    live = q.live_image_ids()
    assert src_id in live  # the source canvas image is referenced
    assert f"job:{job.id}:0" in live  # the result is referenced


# --- multi-GPU distribution ------------------------------------------------

class ConcurrentBackend(MockBackend):
    """Records peak concurrency so "ran in parallel" is deterministic."""

    def __init__(self):
        super().__init__(steps_delay=0.3)
        self._lk = threading.Lock()
        self._active = 0
        self.max_active = 0

    def generate(self, request, progress=None):
        with self._lk:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            return super().generate(request, progress)
        finally:
            with self._lk:
                self._active -= 1


def test_single_image_uses_primary():
    q, _ = _queue(MockBackend(), events=[], gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, job.id)
    assert q.registry.uses == [(0, 1)]  # went to primary, not the other


def test_two_images_distribute_across_gpus():
    q, _ = _queue(MockBackend(steps_delay=0.05), events=[], gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=1), 2)
    _wait_done(q, job.id)
    mapping = {seed: gpu for gpu, seed in q.registry.uses}
    assert mapping[1] == 0  # first task -> primary
    assert mapping[2] == 1  # second task -> the other GPU


def test_three_images_run_in_parallel_and_refill():
    backend = ConcurrentBackend()
    q, _ = _queue(backend, events=[], gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=100), 3)
    _wait_done(q, job.id)
    mapping = {seed: gpu for gpu, seed in q.registry.uses}
    assert mapping[100] == 0          # first -> primary
    assert mapping[101] == 1          # second -> the other GPU
    assert mapping[102] in (0, 1)     # third -> whichever freed first
    assert set(mapping.values()) == {0, 1}
    assert backend.max_active == 2    # two GPUs truly generated at once


def test_primary_wins_tie_when_both_free():
    q, _ = _queue(MockBackend(), events=[], gpus=[0, 1])
    j1 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, j1.id)
    j2 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=2), 1)
    _wait_done(q, j2.id)
    mapping = {seed: gpu for gpu, seed in q.registry.uses}
    assert mapping[1] == 0
    assert mapping[2] == 0           # both GPUs free again -> primary


def test_interrupt_cancels_only_that_job_gpus():
    backend = MockBackend(steps_delay=0.2)
    events = []
    q, _ = _queue(backend, events, gpus=[0, 1])
    # batch 2 occupies BOTH GPUs
    j1 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=20, seed=1), 2)
    deadline = time.time() + 5
    while time.time() < deadline and \
            sum(1 for e in events if e["type"] == "progress") < 2:
        time.sleep(0.01)
    # a second job goes pending behind the busy GPUs
    j2 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=50), 1)
    time.sleep(0.1)  # let j1 start on both GPUs
    q.interrupt(j1.id)
    time.sleep(0.5)
    assert j1.interrupted
    assert j1.results == {}
    # exactly the two GPUs running j1 were cancelled
    assert sorted(q.registry.cancels) == [0, 1]
    # the other job is unaffected and still completes
    _wait_done(q, j2.id)
    assert set(j2.results) == {0}


def test_model_switch_mid_queue_uses_each_model():
    backend = MockBackend()
    events = []
    q, _ = _queue(backend, events, gpus=[0, 1])
    j1 = q.submit("model-a", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, j1.id)
    j2 = q.submit("model-b", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=2), 1)
    _wait_done(q, j2.id)
    assert q.registry.model_uses == [(0, "model-a"), (0, "model-b")]


def test_lone_image_uses_other_gpu_when_primary_busy():
    backend = MockBackend(steps_delay=0.15)
    events = []
    q, _ = _queue(backend, events, gpus=[0, 1])
    j1 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=20, seed=1), 1)
    deadline = time.time() + 5
    while time.time() < deadline and not any(
        e["type"] == "progress" and e["job"] == j1.id for e in events
    ):
        time.sleep(0.01)
    # primary is now busy; a *new* lone image must not stall for it
    j2 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=100), 1)
    _wait_done(q, j2.id)
    mapping = {seed: gpu for gpu, seed in q.registry.uses}
    assert mapping[1] == 0
    assert mapping[100] == 1
