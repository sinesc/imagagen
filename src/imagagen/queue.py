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

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import wildcards
from .backends import GenerateRequest, ModelBackend
from .generation import InpaintPlan
from .gpus import available_gpus

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: int
    model_id: str
    request: GenerateRequest
    batch_size: int
    base_seed: int
    plan: InpaintPlan | None = None  # inpaint modes: maps output to canvas state
    source_png: bytes | None = None  # source canvas snapshot (strip thumbnail)
    source_image_id: str | None = None  # entity id of the source canvas image
    interrupted: bool = False
    results: dict[int, bytes] = field(default_factory=dict)  # index -> PNG bytes
    finished: set[int] = field(default_factory=set)  # indices settled (ok or error)
    first_emitted: bool = False
    done_emitted: bool = False
    # image metadata: the user's RAW prompt (request.prompt is the
    # style-applied one) plus the styles' actual contents, so the PNG
    # round-trips as "same prompt + same style list"
    meta_prompt: str | None = None
    meta_negative: str | None = None
    meta_styles: list[dict] | None = None

    @property
    def completed(self) -> int:
        return len(self.results)


@dataclass
class _Task:
    job_id: int
    index: int
    seed: int


class GenerationQueue:
    def __init__(
        self,
        registry,
        canvas,
        emit: Callable[[dict], None],
        wildcards_dir: Path | None = None,
    ) -> None:
        self.registry = registry
        self.canvas = canvas
        self.emit = emit
        # when set, each task's prompt / negative prompt is wildcard-resolved
        # per seed before generation (None = no wildcard processing)
        self._wildcards_dir = wildcards_dir
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

    def live_image_ids(self) -> set[str]:
        """Entity ids referenced by a live job: every result plus each job's
        source image. The canvas uses this (via ``anchor``) to decide what to
        evict."""
        with self._job_lock:
            return self._live_image_ids_locked()

    def _live_image_ids_locked(self) -> set[str]:
        ids: set[str] = set()
        for j in self._jobs.values():
            if j.source_image_id is not None:
                ids.add(j.source_image_id)
            for idx in j.results:
                ids.add(f"job:{j.id}:{idx}")
        return ids

    def submit(
        self,
        model_id: str,
        request: GenerateRequest,
        batch_size: int,
        plan: InpaintPlan | None = None,
        meta_prompt: str | None = None,
        meta_negative: str | None = None,
        meta_styles: list[dict] | None = None,
    ) -> Job:
        import random

        base_seed = random.randrange(2**31) if request.seed == -1 else request.seed
        job = Job(id=self._next_job_id, model_id=model_id, request=request,
                  batch_size=batch_size, base_seed=base_seed, plan=plan,
                  meta_prompt=meta_prompt, meta_negative=meta_negative,
                  meta_styles=meta_styles)
        job.source_image_id = self.canvas.current_id()
        if job.source_image_id is not None:
            if plan is not None:
                # the inpaint source image, for the UI's source thumbnail
                # (green border): lets the user go back to it when none of
                # the results turn out good
                job.source_png = _plain_png(plan.base_image)
            else:
                # generate: the image currently displayed on the canvas, kept
                # as a source thumbnail (green border) the same way
                job.source_png = _plain_png(self.canvas.image)
        with self._job_lock:
            # finished jobs are no longer referenced: drop them so their
            # result entities can be evicted (in-flight jobs stay)
            for jid in list(self._jobs.keys()):
                if self._jobs[jid].done_emitted and jid != self._next_job_id:
                    del self._jobs[jid]
            self._jobs[self._next_job_id] = job
            self._next_job_id += 1
            live = self._live_image_ids_locked()
        self.canvas.anchor(live)
        # A new generation resets the crop-thumbnail session: the client
        # clears its crop thumbnails when it generates, so pinned crop
        # entities are no longer reachable and can be evicted.
        self.canvas.release_pins()
        self.canvas.evict_stale()
        with self._cond:
            for i in range(batch_size):
                self._pending.append(_Task(job.id, i, base_seed + i))
            self._cond.notify_all()
        self.emit({"type": "job_started", "job": job.id, "batch_size": batch_size,
                   "base_seed": base_seed})
        if job.source_png is not None:
            # right after job_started: the UI shows the source before any
            # result arrives. A source thumbnail exists exactly when the
            # canvas keeps its image (no auto-set); an empty-canvas
            # generate job has none, so the first result auto-sets.
            self.emit({"type": "source_image", "job": job.id,
                       "image_id": job.source_image_id,
                       "url": f"/api/thumb/{job.id}/source.png"})
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

        # Wildcards are resolved per task, using this task's seed: each image
        # of a batch may pick different wildcard lines. The resolved text is
        # what the model sees AND what the image metadata stores (the raw
        # prompt can't be re-rolled to reproduce the pick). A structural
        # error (missing file / recursion) was already rejected at submit
        # time, so this is a defensive guard only.
        meta_prompt = job.meta_prompt
        meta_negative = job.meta_negative
        # only touch the request / metadata when a reference is actually
        # present: without wildcards the (raw + styles) round-trip behaviour
        # is left exactly as before
        if self._wildcards_dir is not None and (
            wildcards.has_wildcards(req.prompt)
            or wildcards.has_wildcards(req.negative_prompt)
        ):
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
        # inpaint modes: the backend output is the raw width x height
        # generation; the plan maps it to canvas state (downscale+blend for
        # "inpaint masked", whole-canvas replacement + new mask for
        # "inpaint not masked"). Thumbnails show the resulting canvas state,
        # so clicking one puts it on the canvas.
        if job.plan is not None:
            out_image = job.plan.postprocess(result.image)
        else:
            out_image = result.image
        png, info = _result_to_png(
            backend, req, result, out_image, job,
            meta_prompt=meta_prompt, meta_negative=meta_negative,
        )
        job.results[task.index] = png
        # register the result as a persistent canvas entity (a new image with
        # an empty draw history); it inherits its mask state (current mask
        # + history) from the job's source image (the green-bordered
        # source thumbnail)
        eid = self.canvas.add_result(job.id, task.index, png, info,
                                     source_id=job.source_image_id)
        self.emit({"type": "image", "job": job.id, "index": task.index,
                   "seed": result.seed, "image_id": eid,
                   "url": f"/api/thumb/{job.id}/{task.index}.png"})
        if not job.first_emitted:
            job.first_emitted = True
            # the first completed image is automatically set in the canvas
            # only when the canvas has no image to keep: inpaint jobs and
            # generate jobs on a non-empty canvas retain the displayed
            # image (selectable via the green-bordered source thumbnail)
            # until the user picks one of the freshly generated thumbnails.
            if job.source_png is None:
                self.canvas.set_current(eid)
                self.emit({"type": "canvas_set", "job": job.id,
                           "index": task.index,
                           "undo_mask": self.canvas.can_undo_mask(),
                           "redo_mask": self.canvas.can_redo_mask(),
                           "undo_draw": self.canvas.can_undo_draw(),
                           "redo_draw": self.canvas.can_redo_draw()})
        self._settle(job, task.index)

    def _settle(self, job: Job, index: int) -> None:
        """Mark a task index as settled; emit 'done' once when all are settled."""
        with self._cond:
            job.finished.add(index)
            if not job.done_emitted and len(job.finished) >= job.batch_size:
                job.done_emitted = True
                self.emit({"type": "done", "job": job.id})


def _plain_png(image: np.ndarray) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def _result_to_png(
    backend: ModelBackend, req: GenerateRequest, result, image: np.ndarray,
    job: Job | None = None, meta_prompt: str | None = None,
    meta_negative: str | None = None,
) -> tuple[bytes, dict]:
    """Return (PNG bytes, text-chunk dict). The dict carries the metadata
    onto the canvas so saves of the canvas keep it too."""
    from io import BytesIO

    from PIL import Image

    from . import metadata

    # width/height are the requested generation size, like A1111/Forge
    # (their inpaint PNGs carry the inpaint box size, not the final canvas
    # size), so importing the PNG restores the generation settings. The
    # prompt/negative prompt default to the user's raw ones (styles are
    # recorded separately, not pre-applied); the per-task overrides carry
    # the fully wildcard-resolved text instead.
    if meta_prompt is None:
        meta_prompt = job.meta_prompt if (job and job.meta_prompt is not None) else req.prompt
    if meta_negative is None:
        meta_negative = (
            job.meta_negative if (job and job.meta_negative is not None)
            else req.negative_prompt
        )
    params = metadata.build_parameters(
        prompt=meta_prompt,
        negative_prompt=meta_negative,
        steps=req.steps,
        sampler=req.sampler or backend.default_sampler,
        scheduler=req.scheduler or backend.default_scheduler,
        cfg=req.cfg,
        seed=result.seed,
        width=req.width,
        height=req.height,
        # A1111/Forge write the strength only for inpaint/img2img jobs
        denoise=req.denoise if job is not None and job.plan is not None else None,
        model_name=backend.name,
        model_hash=backend.model_hash,
        loras=req.loras,
        styles=job.meta_styles if job is not None else None,
        soft_inpainting=req.soft_inpainting,
        sag=req.sag,
    )
    img, pnginfo = metadata.attach_metadata(Image.fromarray(image), params)
    buf = BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    # read the text chunks back from the saved bytes: PngInfo doesn't
    # expose its chunks as a plain dict across Pillow versions
    text = {k: v for k, v in Image.open(buf).info.items() if isinstance(v, str)}
    return buf.getvalue(), text
