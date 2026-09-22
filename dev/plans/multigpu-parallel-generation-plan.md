# Multi-GPU Parallel Generation Plan

## Goal

Distribute image-generation tasks across all available (non-ignored) GPUs in parallel, preferring the primary GPU. A batch of N images on 2 GPUs runs image 1 on primary, image 2 on the other GPU, and image 3 on whichever GPU frees first (primary preferred if both are free).

## Current State

- `ModelRegistry` holds **one** backend on **one** device (`self.gpu`). All generation is serialized on that single GPU.
- `GenerationQueue.start()` spawns N worker threads (N = #visible GPUs) but they all hit the **same** backend → no real parallelism.
- `SdxlBackend` already has a per-backend `_lock` (one pipeline = one thread).
- `gpus.py` has `available_gpus(ignored)`, `select_gpu(primary, ignored)`, `gpu_info(...)`.

## Interpretation of the requested logic

The two rules collapse into one scheduling policy, which the example confirms:

> **Always prefer the primary GPU *when it is free*; otherwise run on any other free GPU. A GPU that just freed takes the next queued image (primary wins ties).**

- 1 image queued, primary free → primary (satisfies "use primary").
- 1 image queued, primary busy (prior job still running) → the other free GPU. **We do not stall the lone image waiting for the primary** — "prefer primary when available" implies using a free alternative when it is not. *If a strict "single image must wait for primary" is required, that is a one-line change to the dispatcher (see decision D5); confirm this is not wanted.*
- 3 images / 2 GPUs → img1→primary, img2→other, img3→whichever frees first (primary preferred). Exactly the stated example.

## Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| One `SdxlBackend` instance **per GPU** (each loads its own model copy) | Required for true parallelism; ~7 GiB per GPU, fits 16 GiB cards |
| Per-GPU `threading.RLock` in the registry serializes load/unload/generate for that GPU | Prevents concurrent `.to()` on the same module (SIGSEGV invariant) |
| One **worker thread per GPU** + a **dispatcher** that assigns pending tasks to free GPUs (primary first) | Naturally gives "prefer primary, spread across free GPUs" semantics |
| `gpu_order(primary, ignored)` returns usable GPUs ordered primary-first | Single source of truth for the pool; excludes GPUs that can't init a CUDA context |
| `registry.run(model_id, gpu, req, progress)` returns `(result, backend)` atomically under the GPU lock | Eliminates race between `backend_for` and `generate` (model swap mid-call) |
| No new `imagagen.ini` keys | `primary-gpu` + `ignore-gpus` already express the full policy |
| D5 | Dispatcher picks `free.pop(0)` from a primary-ordered `free` list | Gives primary-wins-ties; the "wait for primary" variant is a one-line change here |

`registry.backend_for(model_id, gpu=None)` is **kept** (not removed) for `/api/models/select` pre-warm, `from_config`, and dev scripts. The queue itself only uses `registry.get()` + `registry.run()`.

## Files to Change

### 1. `src/imagagen/gpus.py`

Add two functions (keep existing `available_gpus`, `select_gpu`, `gpu_info` unchanged):

```python
def usable_gpus(ignored: frozenset[int] = frozenset()) -> list[int]:
    """Visible, non-ignored GPUs that can init a CUDA context."""
    # For each i in available_gpus(ignored), probe torch.cuda.mem_get_info(i).
    # Include only indices where the probe succeeds.

def gpu_order(primary: int | None = None,
              ignored: frozenset[int] = frozenset()) -> list[int]:
    """Ordered list of usable GPUs: primary first, then the rest by index."""
    usable = usable_gpus(ignored)
    if not usable:
        return []
    p = select_gpu(primary, ignored)
    if p not in usable:
        p = max(usable, key=lambda g: _probe_free(g))  # most-free fallback
    return [p] + [g for g in usable if g != p]
```

Add a small helper `_probe_free(gpu) -> int` that returns free bytes (or 0 on failure).

Update `select_gpu` to use `usable_gpus` instead of `available_gpus` for the explicit-primary check, so an explicit primary that is fully occupied returns `None` (clear error) instead of a GPU that will OOM.

### 2. `src/imagagen/models.py` — ModelRegistry per-GPU backends

Replace the single `_backend`/`_current_id`/`gpu`/`_resolve_device` with:

```python
class _GpuState:
    __slots__ = ('gpu', 'lock', 'backend', 'model_id')
    def __init__(self, gpu: int):
        self.gpu = gpu
        self.lock = threading.RLock()
        self.backend = None
        self.model_id = None

class ModelRegistry:
    def __init__(self, config, emit=None, gpus: list[int] | None = None):
        # ... existing discover() ...
        if gpus is None:
            gpus = gpu_order(config.primary_gpu, config.ignore_gpus)
        self.gpus: list[int] = gpus
        self.primary: int | None = gpus[0] if gpus else None
        self._gpu_states: dict[int, _GpuState] = {}
        self._meta_lock = threading.Lock()
        self._current_id: str | None = None

    def _get_gpu_state(self, gpu: int) -> _GpuState:
        with self._meta_lock:
            return self._gpu_states.setdefault(gpu, _GpuState(gpu))

    def backend_for(self, model_id: str, gpu: int | None = None) -> ModelBackend:
        """Ensure model is loaded on the given GPU (or primary). Returns backend."""
        if gpu is None: gpu = self.primary
        state = self._get_gpu_state(gpu)
        with state.lock:
            if state.model_id == model_id and state.backend is not None:
                return state.backend
            if state.backend is not None:
                state.backend.unload()
                state.backend = None; state.model_id = None
            info = self.get(model_id)
            if info is None: raise ValueError(f"unknown model: {model_id}")
            self._current_id = model_id
            self.emit({"type":"status","message":f"loading {info.name} on GPU {gpu} …"})
            import torch
            from .sdxl_backend import SdxlBackend
            state.backend = SdxlBackend(info, torch.device(f"cuda:{gpu}"), config=self.config)
            state.model_id = model_id
            self.emit({"type":"status","message":f"loaded {info.name} on GPU {gpu}"})
            return state.backend

    def run(self, model_id: str, gpu: int, request, progress=None):
        """Atomically ensure model is loaded on `gpu` and run one generation.
        Returns (GenerateResult, ModelBackend)."""
        state = self._get_gpu_state(gpu)
        with state.lock:
            if state.model_id != model_id or state.backend is None:
                if state.backend is not None:
                    state.backend.unload()
                    state.backend = None; state.model_id = None
                info = self.get(model_id)
                if info is None: raise ValueError(f"unknown model: {model_id}")
                self._current_id = model_id
                self.emit({"type":"status","message":f"loading {info.name} on GPU {gpu} …"})
                import torch
                from .sdxl_backend import SdxlBackend
                state.backend = SdxlBackend(info, torch.device(f"cuda:{gpu}"), config=self.config)
                state.model_id = model_id
                self.emit({"type":"status","message":f"loaded {info.name} on GPU {gpu}"})
            result = state.backend.generate(request, progress)
            return result, state.backend

    def cancel(self, gpu: int | None = None):
        """Cancel in-flight generation on a specific GPU (or all GPUs if None)."""
        targets = [self._gpu_states[g] for g in (
            [gpu] if gpu is not None else list(self._gpu_states)
        ) if g in self._gpu_states]
        for s in targets:
            if s.backend is not None:
                s.backend.cancel()

    # Keep for compat
    def current_id(self): return self._current_id
    def current_backend(self):
        if self.primary is not None and self.primary in self._gpu_states:
            return self._gpu_states[self.primary].backend
        return None
```

Remove: `self.gpu`, `_resolve_device()`, old single `_backend`/`_lock`/`_unload_locked`.

### 3. `src/imagagen/queue.py` — Per-GPU workers + dispatcher

Replace the "N workers pull from shared list" model with **one worker per GPU** and a central **dispatcher**:

```python
class GenerationQueue:
    def __init__(self, registry, canvas, emit, wildcards_dir=None):
        # ... same fields ...
        self._gpus: list[int] = []
        self._pending: list[_Task] = []
        self._gpu_queues: dict[int, deque] = {}   # assigned, waiting
        self._gpu_running: dict[int, _Task] = {}   # actively running
        self._cond = threading.Condition()
        self._workers: dict[int, threading.Thread] = {}
        self._stop = False
        self._job_lock = threading.Lock()

    def start(self, gpus: list[int] | None = None):
        if gpus is None:
            gpus = getattr(self.registry, "gpus", None)
            if not gpus:
                gpus = [0]  # CPU mock fallback
        self._gpus = list(gpus)
        self._gpu_queues = {g: deque() for g in self._gpus}
        self._gpu_running = {}
        self._stop = False
        for g in self._gpus:
            t = threading.Thread(target=self._worker_loop, args=(g,),
                                 name=f"gen-worker-{g}", daemon=True)
            self._workers[g] = t
            t.start()

    def _worker_loop(self, gpu: int):
        while not self._stop:
            with self._cond:
                while not self._gpu_queues[gpu] and not self._stop:
                    self._cond.wait(timeout=0.5)
                if self._stop: return
                task = self._gpu_queues[gpu].popleft()
                self._gpu_running[gpu] = task
            try:
                self._run_task(task, gpu)
            except Exception:
                log.exception("worker gpu=%d crashed", gpu)
            finally:
                with self._cond:
                    self._gpu_running.pop(gpu, None)
                    self._dispatch()

    def _dispatch(self):
        """Assign pending tasks to free GPUs (primary first). Caller holds _cond.

        A GPU is *free* iff it is not running a task AND its queue is empty.
        Pruning mutates the deques **in place** (`q[:] = ...`) so a worker
        holding a reference never reads a stale deque object.
        """
        self._pending = [t for t in self._pending
                         if (j := self.job(t.job_id)) is not None and not j.interrupted]
        for g in self._gpus:
            q = self._gpu_queues[g]
            q[:] = [t for t in q
                    if (j := self.job(t.job_id)) is not None and not j.interrupted]
        free = [g for g in self._gpus
                if g not in self._gpu_running and not self._gpu_queues[g]]
        while self._pending and free:
            task = self._pending.pop(0)
            gpu = free.pop(0)          # primary first (self._gpus order)
            self._gpu_queues[gpu].append(task)
        self._cond.notify_all()

    def submit(self, model_id, request, batch_size, plan=None,
               meta_prompt=None, meta_negative=None, meta_styles=None):
        # ... existing job creation, source snapshot, canvas.anchor ...
        with self._cond:
            for i in range(batch_size):
                self._pending.append(_Task(job.id, i, base_seed + i))
            self._dispatch()
        # ... emit job_started, source_image ...

    def interrupt(self, job_id):
        job = self.job(job_id)
        if job is None or job.interrupted: return
        job.interrupted = True
        cancel_gpus = set()
        with self._cond:
            self._pending = [t for t in self._pending if t.job_id != job_id]
            for g in self._gpus:
                q = self._gpu_queues[g]
                q[:] = [t for t in q if t.job_id != job_id]
                rt = self._gpu_running.get(g)
                if rt is not None and rt.job_id == job_id:
                    cancel_gpus.add(g)
            self._cond.notify_all()
        # cancel OUTSIDE the cond lock: it touches backends, not queue state
        for g in cancel_gpus:
            self.registry.cancel(g)
        self.emit({"type": "job_interrupted", "job": job_id})

    def _run_task(self, task: _Task, gpu: int):
        # Called OUTSIDE self._cond (the worker released it after popping).
        # registry.run() holds the per-GPU lock for the whole generation, so
        # model-swap (select) on this GPU blocks until we finish — no race.
        job = self.job(task.job_id)
        if job is None or job.interrupted:
            return
        info = self.registry.get(job.model_id)
        d = job.request.model_dump(exclude={"seed"})
        d.update(seed=task.seed, model_id=job.model_id,
                 model_name=info.name if info else job.model_id)
        req = GenerateRequest.model_validate(d)
        meta_prompt, meta_negative = job.meta_prompt, job.meta_negative
        # Wildcard resolution (unchanged): per-seed pick; a structural error
        # is surfaced + the task settled, then we return before touching a GPU.
        if self._wildcards_dir is not None and (
                wildcards.has_wildcards(req.prompt)
                or wildcards.has_wildcards(req.negative_prompt)):
            try:
                prompt = wildcards.resolve(req.prompt, self._wildcards_dir, task.seed)
                negative = wildcards.resolve(req.negative_prompt, self._wildcards_dir, task.seed)
            except wildcards.WildcardError as e:
                log.error("job %s task %d: %s", job.id, task.index, e)
                self.emit({"type": "error", "job": job.id, "message": str(e)})
                self._settle(job, task.index)
                return
            req = req.model_copy(update={"prompt": prompt, "negative_prompt": negative})
            meta_prompt, meta_negative = prompt, negative

        def progress(fraction, message):
            if not job.interrupted:
                self.emit({"type": "progress", "job": job.id, "index": task.index,
                           "fraction": fraction, "message": message})

        try:
            result, backend = self.registry.run(job.model_id, gpu, req, progress)
        except InterruptedError:
            if not job.interrupted:
                self.emit({"type": "error", "job": job.id, "message": "generation interrupted"})
            return
        except Exception as e:  # model load (OOM) or backend error
            self.emit({"type": "error", "job": job.id, "message": f"{type(e).__name__}: {e}"})
            self._settle(job, task.index)
            return

        if job.interrupted:
            return
        # ... postprocess (job.plan), _result_to_png(backend, ...), canvas.add_result,
        #     emit "image", first-image auto-set, _settle — all unchanged ...
```

Key invariants:
- **One task per GPU at a time** (enforced by the `_gpu_running` dict; a GPU is never handed a second task while one is in flight).
- The per-backend `_lock` in `SdxlBackend` remains as a safety net for direct (non-queue) callers (dev scripts call `backend.generate()` directly).
- `_run_task` runs with **no** queue cond-lock held; `registry.run()` takes the per-GPU lock. Lock order is always `cond → (release) → gpu-lock`, never nested the other way, so no deadlock.

### 4. `src/imagagen/app.py`

- **`_MockRegistry`**: add `gpus=[0]`, `primary=0`, `run(model_id, gpu, request, progress)`, `cancel(gpu=None)`, `get(model_id)` methods. Update `backend_for` signature to `(self, model_id, gpu=None)`.
- **`create_app`**: remove `gpu=select_gpu(...)` from `ModelRegistry(...)` constructor call. The registry computes its own pool from config.
- **`/api/models/select`**: `registry.backend_for(entry.model_id)` still works (gpu=None → primary).
- **`/api/health`**: report `"primary": registry.primary` and `"pool": registry.gpus` in addition to the existing `gpu`/`gpus` fields.

### 5. `src/imagagen/sdxl_backend.py`

- `from_config(config)`: update to `registry.backend_for(registry.models[0].id)` (gpu=None → primary). No other changes needed.

### 6. `dev/*.py` scripts

Update constructor calls: `ModelRegistry(cfg, gpu=1)` → `ModelRegistry(cfg, gpus=[1])`.
Update `backend_for(id)` calls — they still work (gpu=None → primary=1).

Affected files: `soft_inpaint_check.py`, `smoke_sdxl_inpaint.py`, `inpaint_freckles_check.py`, `gen_refs.py`, `smoke_sdxl_lora.py`.

### 7. Tests

#### `tests/unit/test_gpus.py` — extend

Add tests for `gpu_order`:
- `gpu_order(None, frozenset())` with 4 GPUs → [most_free, next, ...]
- `gpu_order(2, frozenset())` → [2, 0, 1, 3] (2 first, rest by index)
- `gpu_order(None, frozenset({0}))` excludes GPU 0
- `gpu_order(0, frozenset())` where GPU 0 is "unusable" (mem_get_info raises) → falls back to most-free usable
- `gpu_order(None, frozenset())` with no CUDA → []

Add a `_mock_cuda_unusable` helper where `mem_get_info` raises for specific indices.

#### `tests/unit/test_queue.py` — update + extend

- Update `FakeRegistry` to the new interface: `gpus`, `primary`, `run(model_id, gpu, req, progress)`, `cancel(gpu)`, `get(model_id)`. Track `(gpu, seed)` calls in `self.uses`.
- Update `_queue()` helper to accept `gpus` param → passes to `q.start(gpus=gpus)`.
- Existing tests: should pass unchanged (single gpu [0]).

**New multi-GPU distribution tests** (concrete):

```python
class FakeRegistry:                       # replaces the old single-gpu FakeRegistry
    def __init__(self, backend, gpus=(0,)):
        self._b = backend
        self.gpus = list(gpus)
        self.primary = self.gpus[0] if self.gpus else None
        self._id = None
        self.uses = []                     # (gpu, seed) in completion order
        self._lk = threading.Lock()

    def list(self): return []
    def models(self): return []
    def get(self, model_id):
        import types; return types.SimpleNamespace(name="Fake Model")
    def current_id(self): return self._id
    def current_backend(self): return self._b
    def backend_for(self, model_id, gpu=None):
        self._id = model_id; return self._b
    def run(self, model_id, gpu=None, request=None, progress=None):
        with self._lk:
            self.uses.append((gpu, request.seed))
        return self._b.generate(request, progress), self._b
    def cancel(self, gpu=None): self._b.cancel()

def _queue(backend, events, wildcards_dir=None, gpus=None):
    canvas = Canvas()
    q = GenerationQueue(registry=FakeRegistry(backend, gpus or (0,)),
                        canvas=canvas, emit=events.append,
                        wildcards_dir=wildcards_dir)
    q.start(gpus=gpus)                     # None -> FakeRegistry.gpus -> [0]
    return q, canvas

# -- the distribution tests -------------------------------------------------
# A backend that records peak concurrency, so the "ran in parallel" check is
# deterministic (no wall-clock flakiness).
class ConcurrentBackend(MockBackend):
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
    assert q.registry.uses == [(0, 1)]        # went to primary, not the other

def test_two_images_distribute_across_gpus():
    q, _ = _queue(MockBackend(steps_delay=0.05), events=[], gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=1), 2)
    _wait_done(q, job.id)
    assert sorted(g for g, _ in q.registry.uses) == [0, 1]   # one on each

def test_three_images_run_in_parallel_and_refill():
    backend = ConcurrentBackend()
    q, _ = _queue(backend, events=[], gpus=[0, 1])
    job = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=100), 3)
    _wait_done(q, job.id)
    gpus_used = [g for g, _ in q.registry.uses]
    assert gpus_used[0] == 0                 # first -> primary
    assert len(set(gpus_used[:2])) == 2      # first two on different GPUs
    assert backend.max_active == 2           # two GPUs truly generated at once
    assert set(gpus_used) == {0, 1}          # third landed on a freed GPU

def test_primary_wins_tie_when_both_free():
    q, _ = _queue(MockBackend(), events=[], gpus=[0, 1])
    j = q.submit("m", GenerateRequest(prompt="x", width=8, height=8, steps=1, seed=1), 1)
    _wait_done(q, j.id)
    assert q.registry.uses[-1][0] == 0       # both free -> primary
```

#### `tests/integration/test_api.py` — no changes needed

Uses MockBackend via `_MockRegistry` (single gpu [0]). Behavior unchanged.

#### New: `dev/multigpu_smoke.py`

End-to-end real-GPU smoke test (both GPUs free):
1. Start the real server (no `--mock`).
2. Wait for health.
3. Submit a batch of 3 images (512×512, 10 steps) via `/api/generate`.
4. Listen on WebSocket; record which GPU each image ran on (from server logs or a new `/api/gpu-status` endpoint — simplest: just measure that total time ≈ 2× single-image time, not 3×).
5. Poll `nvidia-smi` during generation to confirm both GPUs show active VRAM usage.
6. Assert all 3 images completed.

### 8. `user/imagagen.ini` documentation

Update the comment block to explain the multi-GPU behavior:
```ini
# With multiple non-ignored GPUs, generation tasks are distributed across
# all of them in parallel. The primary GPU (set by primary-gpu) is preferred:
# a lone image goes to primary; a batch spreads across GPUs, refilling
# whichever GPU frees first (primary wins ties).
```

## Implementation Order

1. **`gpus.py`**: Add `usable_gpus`, `gpu_order`, `_probe_free`. Update `select_gpu` to use `usable_gpus`. Run `test_gpus.py`.
2. **`models.py`**: Rewrite `ModelRegistry` with per-GPU state. Run `test_model_discovery.py`.
3. **`queue.py`**: Rewrite worker/dispatch logic. Update `_run_task` signature. Run `test_queue.py` (existing tests with updated FakeRegistry).
4. **`app.py`**: Update `_MockRegistry`, `create_app`, health endpoint. Run integration tests.
5. **`sdxl_backend.py`**: Update `from_config`. 
6. **`dev/*.py`**: Update constructor calls.
7. **New tests**: `test_gpus.py` additions, `test_queue.py` multi-GPU tests.
8. **`dev/multigpu_smoke.py`**: Real 2-GPU end-to-end verification.
9. **`user/imagagen.ini`**: Update comments.

## Verification Checklist

- [ ] `uv run --extra dev pytest tests/unit/test_gpus.py -v`
- [ ] `uv run --extra dev pytest tests/unit/test_queue.py -v`
- [ ] `uv run --extra dev pytest tests/integration/test_api.py -v`
- [ ] `uv run --extra dev pytest tests/unit/test_model_discovery.py -v`
- [ ] `uv run --extra dev pytest tests/unit/test_config.py -v`
- [ ] Real GPU: `dev/multigpu_smoke.py` — 3 images on 2 GPUs, both show VRAM activity, total time ≈ 2× single
- [ ] Real GPU: single image → only primary GPU active
- [ ] Interrupt during multi-GPU batch → only the interrupted job's GPUs cancelled
- [ ] Model switch mid-queue → no crash, pending tasks use new model on correct GPU

## Risks / Edge Cases

| Case | Handling |
|------|----------|
| Primary GPU occupied (no CUDA context) | `gpu_order` excludes it via probe; falls back to most-free usable |
| All GPUs occupied | `gpu_order` returns [] → queue falls back to [0] → first task errors with clear message |
| OOM during generation on one GPU | Task settles with error event; other GPUs continue. GPU remains in pool (operator can add to ignore-gpus) |
| Model switch mid-batch | Per-GPU lazy reload; tasks already running finish with old model, new tasks get new model |
| Mock/CPU mode (no CUDA) | `gpu_order` returns [] → fallback [0] → single worker with MockBackend |
| Two users submit simultaneously | `_cond` lock serializes dispatch; each task gets exactly one GPU |
