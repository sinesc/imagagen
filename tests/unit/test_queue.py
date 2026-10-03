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

    def unload_all(self):
        # mirrors ModelRegistry.unload_all (park to RAM); the fake records it
        # so tests can assert the queue parks models when it goes idle
        self.unloads = getattr(self, "unloads", 0) + 1


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
    # canvas_set emitted exactly once (first image only)
    cs = [e for e in events if e["type"] == "canvas_set"]
    assert len(cs) == 1
    assert cs[0]["job"] == job.id


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


def test_source_event_carries_seed_of_generated_source():
    # a second generation on a canvas showing an earlier result: the
    # source thumbnail is that result entity, whose parameters chunk
    # records the seed -> the seed-from-image button can use it
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    j1 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=12345), 1)
    _wait_done(q, j1.id)
    assert canvas.current_id() == f"job:{j1.id}:0"  # first result auto-set
    events.clear()
    j2 = q.submit("m", GenerateRequest(prompt="y", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, j2.id)
    src = [e for e in events if e["type"] == "source_image"]
    assert len(src) == 1
    assert src[0]["image_id"] == f"job:{j1.id}:0"
    assert src[0]["seed"] == 12345
    assert j2.source_seed == 12345


def test_source_event_seed_null_for_plain_upload():
    # an imported image without a parameters chunk has no seed: the
    # seed-from-image button stays inert for it
    backend = MockBackend()
    events = []
    q, canvas = _queue(backend, events)
    canvas.import_image(_png(np.full((8, 8, 3), 5, dtype=np.uint8)))
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, job.id)
    src = [e for e in events if e["type"] == "source_image"]
    assert len(src) == 1
    assert src[0]["seed"] is None


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
    plan = InpaintPlan("inpaint_masked", requested_size=(8, 8),
                       bbox=(0, 0, 8, 8), work_image=base, work_mask=mask)
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


def test_generation_error_is_logged_to_console(caplog):
    # A backend/generation exception (OOM, LoRA load failure, ...) must
    # reach the server console, not only the WebSocket: before this the
    # except block only emitted a WS error event, so a batch settling
    # N-1/B with no traceback in the terminal hid its cause.
    import logging

    class FailingBackend(MockBackend):
        def generate(self, request, progress=None):
            raise ValueError("boom: LoRA load failed")

    events = []
    with caplog.at_level(logging.ERROR, logger="imagagen.queue"):
        q, _ = _queue(FailingBackend(), events)
        job = q.submit(
            "m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 2)
        deadline = time.time() + 10
        while time.time() < deadline and not job.done_emitted:
            time.sleep(0.02)
    # both tasks settled via the error path (no results)
    assert job.done_emitted
    assert job.results == {}
    # the WS error events are still emitted (one per failed task)
    errs = [e for e in events if e["type"] == "error"]
    assert len(errs) == 2
    assert all("ValueError" in e["message"] and "boom" in e["message"]
               for e in errs)
    # ...and the failure also landed in the server console with context
    logged = [r for r in caplog.records
              if "generation failed" in r.getMessage()]
    assert logged
    assert any("ValueError" not in r.getMessage()  # message carries str(exc)
               and "boom" in r.getMessage()
               and "model=Fake Model" in r.getMessage()
               for r in logged)
    # a traceback is attached (exc_info), not just a one-line message
    assert all(r.exc_info for r in logged)
    assert logged[0].exc_info[0].__name__ == "ValueError"


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


def test_first_image_warms_primary_then_batch_continues():
    # Cold start: the first image of a model runs ALONE on the primary GPU
    # (warm-up gate); only after it fully completes do further images flow,
    # and subsequent batches distribute across both GPUs in parallel.
    backend = MockBackend(steps_delay=0.05)
    q, _ = _queue(backend, events=[], gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=1), 2)
    _wait_done(q, job.id)
    mapping = {seed: gpu for gpu, seed in q.registry.uses}
    assert mapping[1] == 0   # first (cold) image: primary only
    assert mapping[2] == 1   # after warm-up it runs on the other GPU
    # a subsequent batch now spreads across both GPUs
    job2 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=10), 2)
    _wait_done(q, job2.id)
    mapping2 = {seed: gpu for gpu, seed in q.registry.uses if seed >= 10}
    assert mapping2[10] == 0
    assert mapping2[11] == 1


def test_cold_second_image_waits_for_primary():
    # The second image of a cold batch must NOT start on the secondary GPU
    # before the first image has fully generated + decoded.
    backend = MockBackend(steps_delay=0.15)
    events = []
    q, _ = _queue(backend, events, gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=20, seed=1), 2)
    deadline = time.time() + 5
    while time.time() < deadline and not any(
        e["type"] == "progress" and e["job"] == job.id and e["index"] == 0
        for e in events
    ):
        time.sleep(0.01)
    # first image is in flight (steps=20 @ 0.15s ~ 3s); give the dispatch
    # time to (incorrectly) start image 2 early, then assert it has not
    time.sleep(0.5)
    assert q.registry.uses == [(0, 1)]
    _wait_done(q, job.id)
    # image 2 started only after image 1 fully completed, on the other GPU
    assert q.registry.uses == [(0, 1), (1, 2)]


def test_three_images_run_in_parallel_and_refill():
    backend = ConcurrentBackend()
    q, _ = _queue(backend, events=[], gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=100), 3)
    _wait_done(q, job.id)
    mapping = {seed: gpu for gpu, seed in q.registry.uses}
    assert mapping[100] == 0     # first (cold) image -> primary
    assert mapping[101] == 1     # after warm-up, the other GPU
    assert mapping[102] == 0     # primary picks up the remaining image
    assert set(mapping.values()) == {0, 1}
    assert backend.max_active == 2    # two GPUs truly generated at once


def test_progress_aggregates_across_batch_on_multiple_gpus():
    # When a batch runs across GPUs in parallel, the progress bar must track
    # the WHOLE batch (steps * batch_size), not bounce between each GPU's
    # individual step counter.
    backend = MockBackend(steps_delay=0.02)
    events = []
    q, _ = _queue(backend, events, gpus=[0, 1])
    # warm the model so the whole batch can run in parallel on both GPUs
    warm = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=0), 1)
    _wait_done(q, warm.id)
    steps, batch = 4, 4
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=steps, seed=10), batch)
    _wait_done(q, job.id)
    progs = [e for e in events if e["type"] == "progress" and e["job"] == job.id]
    assert progs, "expected progress events for the batch job"
    total = steps * batch
    # the denominator is the whole batch, not a single image's steps
    assert all(e["message"].endswith(f"/ {total}") for e in progs)
    fracs = [e["fraction"] for e in progs]
    # the bar only ever moves forward (no bouncing between GPU progress)
    assert fracs == sorted(fracs), f"progress not monotonic: {fracs}"
    # the batch reaches 100% once every image has finished all its steps
    assert progs[-1]["fraction"] == 1.0
    assert progs[-1]["message"] == f"step {total} / {total}"


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
    # warm-up image so j1's batch can immediately occupy BOTH GPUs
    j0 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=0), 1)
    _wait_done(q, j0.id)
    # batch 2 occupies BOTH GPUs
    j1 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=20, seed=1), 2)
    deadline = time.time() + 5
    while time.time() < deadline and \
            sum(1 for e in events if e["type"] == "progress" and e["job"] == j1.id) < 2:
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
    # warm-up on the primary (cold models are gated to the primary first)
    j0 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=0), 1)
    _wait_done(q, j0.id)
    j1 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=20, seed=1), 1)
    deadline = time.time() + 5
    while time.time() < deadline and not any(
        e["type"] == "progress" and e["job"] == j1.id for e in events
    ):
        time.sleep(0.01)
    # primary is now busy with a warm-model job; a *new* lone image must
    # not stall for it
    j2 = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=100), 1)
    _wait_done(q, j2.id)
    mapping = {seed: gpu for gpu, seed in q.registry.uses}
    assert mapping[1] == 0
    assert mapping[100] == 1
