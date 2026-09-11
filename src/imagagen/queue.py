"""Generation queue.

One worker thread per visible GPU; a worker picks any pending task (dynamic
assignment — faster GPUs naturally drain more work). Only one model may be in
VRAM, so loading/unloading is serialized by the registry's model lock.

A "job" is a user generation request expanded into N single-image tasks
(batch). Tasks of a job carry ascending seeds from the job's base seed.
Interrupting a job drops its pending tasks and cancels any in-flight image
(partial images are never emitted).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from .backends import GenerateRequest, ModelBackend
from .gpus import available_gpus


@dataclass
class Job:
    id: int
    model_id: str
    request: GenerateRequest
    batch_size: int
    base_seed: int
    interrupted: bool = False
    results: dict[int, bytes] = field(default_factory=dict)  # index -> PNG bytes
    finished: set[int] = field(default_factory=set)  # indices settled (ok or error)
    first_emitted: bool = False
    done_emitted: bool = False

    @property
    def completed(self) -> int:
        return len(self.results)


@dataclass
class _Task:
    job_id: int
    index: int
    seed: int


class GenerationQueue:
    def __init__(self, registry, canvas, emit: Callable[[dict], None]) -> None:
        self.registry = registry
        self.canvas = canvas
        self.emit = emit
        self._jobs: dict[int, Job] = {}
        self._pending: list[_Task] = []
        self._pending_lock = threading.Lock()
        self._cond = threading.Condition(self._pending_lock)
        self._next_job_id = 1
        self._workers: list[threading.Thread] = []
        self._stop = False
        self._job_lock = threading.Lock()

    def start(self, workers: int | None = None) -> None:
        if workers is None:
            # one worker per visible, non-ignored GPU (dynamic assignment;
            # generation on a single backend is serialized by its lock)
            ignored = getattr(
                getattr(self.registry, "config", None), "ignore_gpus", frozenset()
            )
            workers = max(1, len(available_gpus(ignored)))
        self._workers = [
            threading.Thread(target=self._worker_loop, name=f"gen-worker-{i}", daemon=True)
            for i in range(workers)
        ]
        for t in self._workers:
            t.start()

    def stop(self) -> None:
        self._stop = True
        with self._cond:
            self._cond.notify_all()

    def jobs(self) -> dict[int, Job]:
        return dict(self._jobs)

    def job(self, job_id: int) -> Job | None:
        with self._job_lock:
            return self._jobs.get(job_id)

    def submit(self, model_id: str, request: GenerateRequest, batch_size: int) -> Job:
        import random

        base_seed = random.randrange(2**31) if request.seed == -1 else request.seed
        job = Job(id=self._next_job_id, model_id=model_id, request=request,
                  batch_size=batch_size, base_seed=base_seed)
        with self._job_lock:
            self._jobs[self._next_job_id] = job
            self._next_job_id += 1
        with self._cond:
            for i in range(batch_size):
                self._pending.append(_Task(job.id, i, base_seed + i))
            self._cond.notify_all()
        self.emit({"type": "job_started", "job": job.id, "batch_size": batch_size,
                   "base_seed": base_seed})
        return job

    def interrupt(self, job_id: int) -> None:
        job = self.job(job_id)
        if job is None or job.interrupted:
            return
        job.interrupted = True
        with self._cond:
            self._pending = [t for t in self._pending if t.job_id != job_id]
            self._cond.notify_all()
        backend = self.registry.current_backend()
        if backend is not None:
            backend.cancel()
        self.emit({"type": "job_interrupted", "job": job_id})

    # --- internals ---
    def _worker_loop(self) -> None:
        while not self._stop:
            task = self._take_task()
            if task is None:
                return
            self._run_task(task)

    def _take_task(self) -> _Task | None:
        with self._cond:
            while not self._stop:
                for i, t in enumerate(self._pending):
                    job = self.job(t.job_id)
                    if job is None or job.interrupted:
                        self._pending.pop(i)
                        continue
                    self._pending.pop(i)
                    return t
                try:
                    self._cond.wait(timeout=0.5)
                except TimeoutError:
                    continue
            return None

    def _run_task(self, task: _Task) -> None:
        job = self.job(task.job_id)
        if job is None or job.interrupted:
            return
        try:
            # model loading can fail (e.g. OOM) -- surface it like a gen error
            backend: ModelBackend = self.registry.backend_for(job.model_id)
        except Exception as e:
            self.emit({"type": "error", "job": job.id,
                       "message": f"{type(e).__name__}: {e}"})
            self._settle(job, task.index)
            return
        d = job.request.model_dump(exclude={"seed"})
        d.update(seed=task.seed, model_id=backend.model_id, model_name=backend.name)
        req = GenerateRequest.model_validate(d)

        def progress(fraction: float, message: str) -> None:
            if not job.interrupted:
                self.emit({"type": "progress", "job": job.id, "index": task.index,
                           "fraction": fraction, "message": message})

        try:
            result = backend.generate(req, progress=progress)
        except InterruptedError:
            if not job.interrupted:
                self.emit({"type": "error", "job": job.id,
                           "message": "generation interrupted"})
            return
        except Exception as e:  # surface backend errors to the UI
            self.emit({"type": "error", "job": job.id, "message": f"{type(e).__name__}: {e}"})
            self._settle(job, task.index)
            return

        if job.interrupted:
            return
        png = _result_to_png(backend, req, result)
        job.results[task.index] = png
        self.emit({"type": "image", "job": job.id, "index": task.index,
                   "seed": result.seed, "url": f"/api/thumb/{job.id}/{task.index}.png"})
        if not job.first_emitted:
            job.first_emitted = True
            # spec: first completed image is automatically set in the canvas
            self.canvas.set_from_png_bytes(png)
            self.emit({"type": "canvas_set", "job": job.id, "index": task.index})
        self._settle(job, task.index)

    def _settle(self, job: Job, index: int) -> None:
        """Mark a task index as settled; emit 'done' once when all are settled."""
        with self._cond:
            job.finished.add(index)
            if not job.done_emitted and len(job.finished) >= job.batch_size:
                job.done_emitted = True
                self.emit({"type": "done", "job": job.id})


def _result_to_png(backend: ModelBackend, req: GenerateRequest, result) -> bytes:
    from io import BytesIO

    from PIL import Image

    from . import metadata

    params = metadata.build_parameters(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        steps=req.steps,
        sampler=req.sampler or backend.default_sampler,
        scheduler=req.scheduler or backend.default_scheduler,
        cfg=req.cfg,
        seed=result.seed,
        width=req.width,
        height=req.height,
        model_name=backend.name,
        model_hash=backend.model_hash,
        loras=req.loras,
    )
    img, pnginfo = metadata.attach_metadata(Image.fromarray(result.image), params)
    buf = BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()
