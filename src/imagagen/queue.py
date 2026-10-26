"""Generation queue.

One worker thread per GPU in the registry pool. A central dispatcher
assigns pending tasks to *free* GPUs in pool order (primary first, then by
index); a GPU is free when it is not running a task and its per-GPU queue
is empty. Each GPU therefore generates at most one image at a time, but
different GPUs run in parallel. Model loading/unloading is serialized per
GPU by the registry's per-GPU lock.

Warm-up gate: the first image for a model (fresh server, or a model switch)
runs ALONE on the primary GPU. Only once it has been fully generated and
decoded does that model's work join normal parallel dispatch. Starting a
cold process straight into a parallel multi-GPU batch corrupts CUDA state
and produces black images; a completed warm-up generation makes the
subsequent parallel generation reliable. The mark lives per model in
``_warmed_model`` and is set in ``_run_task`` after a successful run.

A "job" is a user generation request expanded into N single-image tasks
(batch). Tasks of a job carry ascending seeds from the job's base seed.
Interrupting a job drops its pending tasks and cancels any in-flight image
on the GPUs that are running that job (partial images are never emitted).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import memdbg, wildcards
from .backends import (
    PHASE_DENOISING,
    PHASE_ENCODING,
    PHASE_REFERENCE,
    PHASE_REFERENCE_DONE,
    PHASE_REFERENCE_LOADING,
    PHASE_TEXT,
    GenerateRequest,
    ModelBackend,
    format_loras,
)
from .generation import InpaintPlan

log = logging.getLogger(__name__)

# Backends report per-task progress as "step <done>/<total>". The queue
# parses it to aggregate completed steps across every task (GPU) of a job,
# so the progress bar tracks the whole batch instead of bouncing between
# each GPU's individual step counter.
_STEP_RE = re.compile(r"step (\d+)/(\d+)")

# The generation-phase status texts the queue renders from the backends'
# phase messages. "<verb> @ <width>x<height>…" uses the mode word of the
# job's plan and the request's W x H, i.e. the size the model generates at
# (for a rotated generate / outpaint job that is the extended working-frame
# size, not the final canvas size -- see InpaintPlan). The trailing
# ellipsis marks the stage as still running (see backends.PHASE_TEXT).
_MODE_WORDS = {
    "generate": "Generating",
    "inpaint_masked": "Inpainting",
    "inpaint_not_masked": "Outpainting",
}


def _mode_word(plan: InpaintPlan | None) -> str:
    """The status verb for a job: "Generating" / "Inpainting" / "Outpainting"."""
    return _MODE_WORDS.get(plan.mode if plan is not None else "generate",
                           "Generating")


@dataclass
class Job:
    id: int
    model_id: str
    request: GenerateRequest
    batch_size: int
    base_seed: int
    plan: InpaintPlan | None = None  # maps output to canvas state; present
                                     # for inpaint modes and rotated generate
                                     # jobs
    source_png: bytes | None = None  # source canvas snapshot (strip thumbnail)
    source_image_id: str | None = None  # entity id of the source canvas image
    source_seed: int | None = None  # seed recorded in the source image's metadata
    interrupted: bool = False
    results: dict[int, bytes] = field(default_factory=dict)  # index -> PNG bytes
    finished: set[int] = field(default_factory=set)  # indices settled (ok or error)
    first_emitted: bool = False
    done_emitted: bool = False
    # monotonic clock at submit: used for the runtime in the completion log
    submitted_at: float = field(default_factory=time.monotonic)
    # image metadata: the user's RAW prompt (request.prompt is the
    # style-applied one) plus the styles' actual contents, so the PNG
    # round-trips as "same prompt + same style list"
    meta_prompt: str | None = None
    meta_negative: str | None = None
    meta_styles: list[dict] | None = None
    # LoRAs the UI showed toggled off: not applied to the request, recorded
    # in the image metadata so an import restores them in the disabled state
    meta_disabled_loras: list[tuple[str, float]] | None = None
    # the "optional activation text" tags that were folded into the prompt
    # (the LoRAs' tags minus the ones the UI switched off); recorded so the
    # PNG says what the image was actually generated with
    meta_activation_tags: list[str] | None = None
    # NAME of the model configuration the job runs on: recorded in the image
    # metadata so importing the PNG re-selects that configuration
    meta_config_name: str | None = None
    # Wildcard-resolved (prompt, negative) per task index: the per-seed
    # picks are made at submit time (submit) so the batch's prompts can be
    # pre-encoded in ONE text-encoder session before any task runs. None =
    # no wildcard processing (tasks use the request's text as-is).
    resolved: dict[int, tuple[str, str]] | None = None
    # Wildcard-resolved (prompt, negative) per task index FOR THE METADATA:
    # the same resolution applied to the RAW prompt, i.e. without the styles
    # and LoRA activation texts baked in (those travel as their own metadata
    # entries). None = no wildcard processing. Re-importing the PNG and
    # generating with the same seed re-resolves the style-applied prompt to
    # the very same picks, so this is both the raw text and reproducible.
    meta_resolved: dict[int, tuple[str, str]] | None = None
    # Tasks of this job may not be dispatched until its prompts are fully
    # pre-encoded (wildcard jobs); set immediately for all other jobs.
    pre_encoded: threading.Event = field(default_factory=threading.Event)
    # Batch-wide step progress (see _STEP_RE): `steps_per_task` is the total
    # number of denoising steps one image needs (learned from the backend's
    # first progress event; identical for every task of the job because they
    # all share the same request). `steps_done` maps a task index to how many
    # of its steps have completed. Together they let the queue emit a single
    # monotonic progress value for the whole batch: done/total where
    # total = steps_per_task * batch_size.
    steps_per_task: int | None = None
    steps_done: dict[int, int] = field(default_factory=dict)

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
        parking_delay: float = 0.0,
    ) -> None:
        self.registry = registry
        self.canvas = canvas
        self.emit = emit
        # when set, each task's prompt / negative prompt is wildcard-resolved
        # per seed before generation (None = no wildcard processing)
        self._wildcards_dir = wildcards_dir
        # Seconds between the queue going idle and parking the loaded models
        # out of VRAM (imagagen.ini transformer-parking-delay): 0 parks
        # immediately, < 0 never parks explicitly.
        self._parking_delay = parking_delay
        # Pending deferred park (threading.Timer); guarded by ``self._cond``
        # like every other queue-state field.
        self._park_timer: threading.Timer | None = None
        self._jobs: dict[int, Job] = {}
        self._gpus: list[int] = []
        self._pending: list[_Task] = []
        self._gpu_queues: dict[int, deque] = {}
        self._gpu_running: dict[int, _Task] = {}
        self._pending_lock = threading.Lock()
        self._cond = threading.Condition(self._pending_lock)
        self._next_job_id = 1
        self._workers: dict[int, threading.Thread] = {}
        self._stop = False
        self._job_lock = threading.Lock()
        # Guards batch-wide step aggregation (steps_per_task / steps_done)
        # so concurrent worker progress updates stay consistent and the
        # emitted progress sequence remains monotonic.
        self._progress_lock = threading.Lock()
        # The model whose first image has fully generated+decoded in this
        # process (the warm-up gate, see module docstring). Tasks of any
        # other model run on the primary GPU only until one of them has
        # completed. Guarded by ``self._cond``.
        self._warmed_model: str | None = None

    def start(self, gpus: list[int] | None = None) -> None:
        if gpus is None:
            # One worker per GPU in the registry's pool. In CPU/mock mode
            # (no usable CUDA devices) fall back to a single fake GPU 0.
            gpus = getattr(self.registry, "gpus", None)
            if not gpus:
                gpus = [0]
        self._gpus = list(gpus)
        self._pending = []
        self._gpu_queues = {g: deque() for g in self._gpus}
        self._gpu_running = {}
        self._stop = False
        for g in self._gpus:
            t = threading.Thread(target=self._worker_loop, args=(g,),
                                 name=f"gen-worker-{g}", daemon=True)
            self._workers[g] = t
            t.start()

    def stop(self) -> None:
        self._stop = True
        with self._cond:
            if self._park_timer is not None:
                self._park_timer.cancel()
                self._park_timer = None
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
        meta_disabled_loras: list[tuple[str, float]] | None = None,
        meta_activation_tags: list[str] | None = None,
        meta_config_name: str | None = None,
    ) -> Job:
        import random

        base_seed = random.randrange(2**31) if request.seed == -1 else request.seed
        # Wildcards: resolve the per-seed prompt picks for EVERY task of
        # the batch up front (a fresh RNG per resolve, exactly the per-task
        # picks _run_task used to make) so the batch's prompts can be
        # pre-encoded in one text-encoder session before any task runs
        # (below) -- otherwise every task of the batch misses the
        # prompt-encoding cache and re-loads / swaps the multi-GiB text
        # encoder for itself.
        resolved = None
        resolve_error: wildcards.WildcardError | None = None
        if self._wildcards_dir is not None and (
            wildcards.has_wildcards(request.prompt)
            or wildcards.has_wildcards(request.negative_prompt)
        ):
            try:
                resolved = {
                    i: (
                        wildcards.resolve(
                            request.prompt, self._wildcards_dir, base_seed + i),
                        wildcards.resolve(
                            request.negative_prompt, self._wildcards_dir,
                            base_seed + i),
                    )
                    for i in range(batch_size)
                }
            except wildcards.WildcardError as e:
                # app.py already rejected the structural errors; a file may
                # still have changed (emptied or deleted) since. Such an
                # error would fail EVERY task of the batch, so reject the
                # whole job up front instead of settling the tasks one by
                # one.
                resolve_error = e
        # Metadata copy of the resolution: the RAW prompt (styles and LoRA
        # activation texts NOT applied) resolved with the same per-task seeds.
        # Metadata is cosmetic, so a failure here just keeps the raw text.
        meta_resolved: dict[int, tuple[str, str]] | None = None
        meta_neg_src = meta_negative if meta_negative is not None else request.negative_prompt
        if resolved is not None and meta_prompt is not None and (
            wildcards.has_wildcards(meta_prompt)
            or wildcards.has_wildcards(meta_neg_src)
        ):
            try:
                meta_resolved = {
                    i: (
                        wildcards.resolve(
                            meta_prompt, self._wildcards_dir, base_seed + i),
                        wildcards.resolve(
                            meta_neg_src, self._wildcards_dir, base_seed + i),
                    )
                    for i in range(batch_size)
                }
            except wildcards.WildcardError:
                meta_resolved = None
        job = Job(id=self._next_job_id, model_id=model_id, request=request,
                  batch_size=batch_size, base_seed=base_seed, plan=plan,
                  meta_prompt=meta_prompt, meta_negative=meta_negative,
                  meta_styles=meta_styles,
                  meta_disabled_loras=meta_disabled_loras,
                  meta_activation_tags=meta_activation_tags,
                  meta_config_name=meta_config_name, resolved=resolved,
                  meta_resolved=meta_resolved)
        job.source_image_id = self.canvas.current_id()
        job.source_seed = self._source_seed(job.source_image_id)
        if job.source_image_id is not None:
            # the image currently displayed on the canvas (unrotated), kept
            # as the source thumbnail (green border) so the user can go back
            # to it when none of the results turn out good
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
        memdbg.stamp("submit", job=job.id, batch=batch_size,
                     model=model_id, plan=plan is not None,
                     wildcards=resolved is not None)
        if plan is not None:
            # InpaintPlan holds full canvas frames (numpy); if a plan outlives
            # its job, those frames leak -> watch for a missing freed line
            memdbg.track(plan, f"InpaintPlan[job{job.id}]")
        # A new generation resets the crop-thumbnail session: the client
        # clears its crop thumbnails when it generates, so pinned crop
        # entities are no longer reachable and can be evicted.
        self.canvas.release_pins()
        self.canvas.evict_stale()
        if resolve_error is not None:
            log.error("job %d: %s", job.id, resolve_error)
            with self._cond:
                job.finished.update(range(job.batch_size))
                job.done_emitted = True
            self.emit({"type": "error", "job": job.id,
                      "message": str(resolve_error)})
            return job
        if resolved is None:
            job.pre_encoded.set()
        with self._cond:
            for i in range(batch_size):
                self._pending.append(_Task(job.id, i, base_seed + i))
            self._dispatch()
        if resolved is not None:
            # encode the batch's prompts in one text-encoder session before
            # any task of it runs (the dispatch gate holds them until the
            # event is set -- see _pre_encode_job)
            threading.Thread(
                target=self._pre_encode_job, args=(job,),
                name=f"pre-encode-job-{job.id}", daemon=True,
            ).start()
        self.emit({"type": "job_started", "job": job.id, "batch_size": batch_size,
                   "base_seed": base_seed})
        # Cold models warm up alone on the primary GPU first (see module
        # docstring), then join the pool; the log says which pool the batch
        # will use and notes the warm-up when the model is cold.
        suffix = ("" if job.model_id == self._warmed_model
                  else f" (model cold: starts on GPU {self._gpus[0]})")
        req_steps = job.request.steps
        steps_desc = (
            f"{req_steps * batch_size} total steps ({req_steps} steps each)"
            if req_steps > 0 else "model default steps"
        )
        log.info(
            "batch queued: job %d: %d images, %s, model: %s, GPUs: %s%s",
            job.id, batch_size, steps_desc, job.model_id,
            ", ".join(str(g) for g in self._gpus), suffix,
        )
        if job.source_png is not None:
            # right after job_started: the UI shows the source before any
            # result arrives. A source thumbnail exists exactly when the
            # canvas keeps its image (no auto-set); an empty-canvas
            # generate job has none, so the first result auto-sets.
            self.emit({"type": "source_image", "job": job.id,
                       "image_id": job.source_image_id,
                       "seed": job.source_seed,
                       "url": f"/api/thumb/{job.id}/source.png"})
        return job

    def _source_seed(self, image_id: str | None) -> int | None:
        """The seed recorded in a canvas entity's ``parameters`` chunk, if
        any: a generated image (or an import carrying A1111-style metadata)
        has one, a plain upload does not."""
        if image_id is None:
            return None
        info = self.canvas.entity_info(image_id)
        params = info.get("parameters") if info else None
        if not params:
            return None
        from . import metadata
        try:
            seed = metadata.parse_parameters(params).get("seed")
        except Exception:
            return None
        return seed if isinstance(seed, int) else None

    def interrupt(self, job_id: int) -> None:
        job = self.job(job_id)
        if job is None or job.interrupted:
            return
        job.interrupted = True
        log.info("batch canceled: job %d (%d/%d images completed)",
                 job.id, job.completed, job.batch_size)
        cancel_gpus = set()
        with self._cond:
            self._pending = [t for t in self._pending if t.job_id != job_id]
            for g in self._gpus:
                q = self._gpu_queues[g]
                filtered = [t for t in q if t.job_id != job_id]
                q.clear()
                q.extend(filtered)
                rt = self._gpu_running.get(g)
                if rt is not None and rt.job_id == job_id:
                    cancel_gpus.add(g)
            self._dispatch()
        # Cancel OUTSIDE the queue lock: it touches backends (not queue
        # state) and must not block while a worker is still in
        # registry.run() for this job.
        for g in cancel_gpus:
            self.registry.cancel(g)
        self.emit({"type": "job_interrupted", "job": job_id})
        # an interrupt can leave the queue fully idle (all tasks were
        # pending): park models to free VRAM in that case
        self._maybe_unload_idle()

    # --- internals ---
    def _worker_loop(self, gpu: int) -> None:
        while not self._stop:
            with self._cond:
                while not self._gpu_queues[gpu] and not self._stop:
                    self._cond.wait(timeout=0.5)
                if self._stop:
                    return
                task = self._gpu_queues[gpu].popleft()
                self._gpu_running[gpu] = task
            try:
                self._run_task(task, gpu)
            except Exception:
                log.exception("generation worker gpu=%d crashed", gpu)
            finally:
                memdbg.stamp("task-end", job=task.job_id, gpu=gpu)
                with self._cond:
                    self._gpu_running.pop(gpu, None)
                    self._dispatch()
                # a finished (or failed) task may have left the queue fully
                # idle: park loaded models in RAM to free VRAM
                self._maybe_unload_idle()

    def _dispatch(self) -> None:
        """Assign pending tasks to free GPUs (primary first).

        The caller must hold ``self._cond``. A GPU is *free* iff it is not
        running a task AND its per-GPU queue is empty. Pruning mutates the
        deques **in place** (``clear`` + ``extend``) so a worker holding a
        reference to a deque never reads a stale object.
        """
        self._pending = [
            t for t in self._pending
            if (j := self.job(t.job_id)) is not None and not j.interrupted
        ]
        for g in self._gpus:
            q = self._gpu_queues[g]
            filtered = [
                t for t in q
                if (j := self.job(t.job_id)) is not None and not j.interrupted
            ]
            q.clear()
            q.extend(filtered)
        free = [
            g for g in self._gpus
            if g not in self._gpu_running and not self._gpu_queues[g]
        ]
        i = 0
        while i < len(self._pending) and free:
            task = self._pending[i]
            job = self.job(task.job_id)
            if job is None or not job.pre_encoded.is_set():
                # gone, or a wildcard batch still being pre-encoded: its
                # tasks wait (the pre-encode thread dispatches again when it
                # finishes); later jobs' tasks can still flow
                i += 1
                continue
            if job.model_id == self._warmed_model:
                gpu = free.pop(0)  # warm: any free GPU, primary first
            else:
                # Cold model: primary only, and only while it is the next
                # free GPU. A cold task is never sent to a secondary GPU,
                # and at most one cold task leaves the pending list per
                # dispatch (the second would still see the primary busy).
                if free and free[0] == self._gpus[0]:
                    gpu = free.pop(0)
                else:
                    i += 1
                    continue
            del self._pending[i]
            self._gpu_queues[gpu].append(task)
        self._cond.notify_all()

    def _run_task(self, task: _Task, gpu: int) -> None:
        # Called with NO queue lock held: the worker released ``self._cond``
        # after popping the task. registry.run() takes the per-GPU lock for
        # the whole generation, so a model swap on this GPU blocks until we
        # finish (other GPUs are unaffected). Lock order is always
        # cond -> (release) -> gpu-lock, never the reverse.
        job = self.job(task.job_id)
        if job is None or job.interrupted:
            return
        info = self.registry.get(job.model_id)
        d = job.request.model_dump(exclude={"seed"})
        d.update(seed=task.seed, model_id=job.model_id,
                 model_name=info.name if info is not None else job.model_id)
        req = GenerateRequest.model_validate(d)
        memdbg.stamp("task-begin", job=task.job_id, gpu=gpu,
                     size=f"{req.width}x{req.height}",
                     init=req.init_image is not None,
                     mask=req.init_mask is not None, batch=job.batch_size,
                     plan=job.plan.mode if job.plan else None)

        # Wildcards: the per-seed prompt picks were resolved at submit time
        # (Job.resolved) -- identical to per-task resolution (a fresh RNG
        # seeded with the task's seed), just made up front so the batch can
        # be pre-encoded in one text-encoder session (see submit /
        # _pre_encode_job). The model sees the style-applied resolved text;
        # the metadata stores the RAW prompt with the same wildcard picks
        # (Job.meta_resolved), so styles/LoRA activations stay un-baked and
        # the PNG round-trips as "same prompt + same style list".
        meta_prompt = job.meta_prompt
        meta_negative = job.meta_negative
        if job.resolved is not None:
            prompt, negative = job.resolved[task.index]
            req = req.model_copy(update={"prompt": prompt,
                                         "negative_prompt": negative})
        if job.meta_resolved is not None:
            meta_prompt = job.meta_resolved[task.index][0]
            if job.meta_negative is not None:
                meta_negative = job.meta_resolved[task.index][1]

        def progress(fraction: float, message: str) -> None:
            if job.interrupted:
                return
            if message in (PHASE_ENCODING, PHASE_DENOISING, PHASE_REFERENCE,
                           PHASE_REFERENCE_LOADING, PHASE_REFERENCE_DONE):
                # A generation-phase marker, not step progress: it carries no
                # fraction (the bar keeps its value), so it goes out as a
                # "phase" event with the user-facing text -- "Encoding
                # prompt…" while the text encoder runs, "Inpainting @
                # 1024x1280…" while the transformer samples, "Describing
                # reference image…" while the VLM captions the reference
                # image. The texts live in backends.PHASE_TEXT: every one
                # starts with a capital letter and ends on an ellipsis while
                # its stage is still running. The sampling text is built
                # here because only the queue knows the mode verb and the
                # size the model generates at.
                text = PHASE_TEXT.get(message)
                if text is None:
                    text = f"{_mode_word(job.plan)} @ {req.width}x{req.height}…"
                self.emit({"type": "phase", "job": job.id,
                           "index": task.index, "message": text})
                return
            m = _STEP_RE.search(message)
            if m is None:
                # unexpected message format: keep the per-task pass-through
                self.emit({"type": "progress", "job": job.id, "index": task.index,
                           "fraction": fraction, "message": message})
                return
            step, per_task = int(m.group(1)), int(m.group(2))
            # Aggregate across the whole batch: sum the completed steps of
            # every task and divide by (steps per task * batch size). The lock
            # is held across the emit as well so that the emitted sequence
            # stays monotonic (a later-emitted event can never report fewer
            # completed steps than an earlier one).
            with self._progress_lock:
                if job.steps_per_task is None:
                    job.steps_per_task = per_task
                job.steps_done[task.index] = max(
                    job.steps_done.get(task.index, 0), step)
                total = job.steps_per_task * job.batch_size
                done = sum(job.steps_done.values())
                self.emit({"type": "progress", "job": job.id,
                           "index": task.index,
                           "fraction": done / total,
                           "message": f"step {done} / {total}"})

        try:
            result, backend = self.registry.run(job.model_id, gpu, req, progress)
            if req.loras and not backend.supports_loras:
                log.warning(
                    "job %d: ignoring %d LoRA(s) -- %s does not support LoRAs",
                    job.id, len(req.loras), backend.name)
        except InterruptedError:
            if not job.interrupted:
                self.emit({"type": "error", "job": job.id,
                           "message": "generation interrupted"})
            return
        except Exception as e:  # model load (OOM) or backend error
            # Also surface the failure in the server console (a batch that
            # settles N-1/B with no traceback in the terminal otherwise hides
            # its cause in the WebSocket stream only). exc_info captures the
            # full traceback; the message carries the exception itself.
            log.exception(
                "job %d image %d: generation failed: %s "
                "(model=%s loras=%s gpu=%d)",
                job.id, task.index, e,
                info.name if info is not None else job.model_id,
                format_loras(req.loras),
                gpu,
            )
            self.emit({"type": "error", "job": job.id,
                       "message": f"{type(e).__name__}: {e}"})
            self._settle(job, task.index)
            return

        uniform = _uniform_color(result.image)
        if uniform is not None:
            r, g, b = uniform
            log.error(
                "job %d image %d: generated image is a uniform color "
                "(%d, %d, %d) -- usually NaN/inf in the solver state or a "
                "VAE decode overflow (check VRAM, the VAE, unet "
                "quantization). seed=%d model=%s loras=%s size=%dx%d "
                "steps=%s cfg=%s sampler=%s scheduler=%s denoise=%s "
                "init_image=%s init_mask=%s soft_inpainting=%s sag=%s "
                "vae_dtype=%s gpu=%d prompt=%r",
                job.id, task.index, r, g, b,
                task.seed,
                info.name if info is not None else job.model_id,
                format_loras(req.loras),
                req.width, req.height,
                result.steps if result.steps is not None else req.steps,
                result.cfg if result.cfg is not None else req.cfg,
                req.sampler or backend.default_sampler,
                req.scheduler or backend.default_scheduler,
                req.denoise,
                req.init_image is not None, req.init_mask is not None,
                req.soft_inpainting is not None,
                req.sag.enabled if req.sag is not None else None,
                getattr(backend, "vae_dtype", "?"),
                gpu, _short(req.prompt),
            )

        # The image is fully generated and decoded: this model is now warm.
        # From here on its tasks may run in parallel on multiple GPUs
        # (releases the warm-up gate in _dispatch). An interrupted or
        # failed first run does NOT warm the model: the next task stays
        # gated to the primary and retries.
        if job.model_id != self._warmed_model:
            with self._cond:
                if job.model_id != self._warmed_model:
                    self._warmed_model = job.model_id
                    self._dispatch()

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
                           "index": task.index})
        self._settle(job, task.index)

    def _pre_encode_job(self, job: Job) -> None:
        """Encode the whole batch's wildcard-resolved prompts in ONE
        text-encoder session before any task of the job runs.

        Each task of a wildcard batch carries a different prompt, so
        without this every task would miss the prompt-encoding cache and
        re-load / swap the multi-GiB text encoder for itself (the encoder
        crossing the bus once per image). Encoded on the primary GPU's
        backend, the shared per-model cache (backends.shared_enc_cache)
        makes every task of the batch a cache hit on every GPU it is
        spread over. registry.pre_encode holds the per-GPU lock across the
        encode, so a model swap can never park the backend mid-encode.
        """
        gpu = self._gpus[0] if self._gpus else None
        try:
            # the encode is the slow part of a wildcard batch (the encoder
            # crosses the bus once for the whole batch here), so announce it
            self.emit({"type": "phase", "job": job.id, "index": None,
                       "message": PHASE_TEXT[PHASE_ENCODING]})
            self.registry.pre_encode(job.model_id, gpu,
                                     list(job.resolved.values()),
                                     reserve=len(job.resolved))
            log.info("job %d: pre-encoded %d prompt(s) in one encoder "
                     "session before generation", job.id, len(job.resolved))
        except Exception as e:
            # the model failed to load (OOM, ...) or the encode crashed: the
            # whole batch would have failed the same way task by task
            log.exception("job %d: pre-encode failed: %s", job.id, e)
            self.emit({"type": "error", "job": job.id,
                       "message": f"{type(e).__name__}: {e}"})
            for i in range(job.batch_size):
                self._settle(job, i)
        finally:
            with self._cond:
                job.pre_encoded.set()
                self._dispatch()
                self._cond.notify_all()
            # an interrupted batch may have left the queue fully idle
            self._maybe_unload_idle()

    def _settle(self, job: Job, index: int) -> None:
        """Mark a task index as settled; emit 'done' once when all are settled."""
        with self._cond:
            job.finished.add(index)
            if not job.done_emitted and len(job.finished) >= job.batch_size:
                job.done_emitted = True
                runtime = time.monotonic() - job.submitted_at
                log.info(
                    "batch completed: job %d (%d/%d images completed, "
                    "%.1fs)",
                    job.id, len(job.results), job.batch_size, runtime,
                )
                blob = sum(len(v) for v in job.results.values()
                           if isinstance(v, (bytes, bytearray)))
                if job.source_png:
                    blob += len(job.source_png)
                memdbg.stamp("job-done", job=job.id,
                             results=len(job.results),
                             png_kb=blob // 1024,
                             secs=round(runtime, 1))
                self.emit({"type": "done", "job": job.id,
                           "message": f"Completed {len(job.results)} images "
                                      f"in {runtime:.1f}s"})

    def _is_idle_locked(self) -> bool:
        """True when no task is pending/queued/running and no job is still
        in flight (a job counts as settled once done or interrupted).
        Caller must hold self._cond."""
        if self._pending or self._gpu_running or any(self._gpu_queues.values()):
            return False
        for job in self._jobs.values():
            if job.resolved is not None and not job.pre_encoded.is_set():
                # a wildcard batch still being pre-encoded: its encode holds
                # a backend on a GPU, so parking models now would race it
                return False
            if not job.done_emitted and not job.interrupted:
                return False
        return True

    def _maybe_unload_idle(self) -> None:
        """Once the queue is fully idle, schedule moving every loaded model
        out of VRAM into RAM so the app does not hold VRAM while not
        generating.

        ``self._parking_delay`` (imagagen.ini transformer-parking-delay)
        controls WHEN the park happens: 0 parks immediately, > 0 defers via
        a one-shot timer that re-checks idleness when it fires (a batch
        that started meanwhile keeps the models resident; the next idle
        transition re-arms), < 0 never parks explicitly (models stay
        resident until something else needs the VRAM).

        Model types that cannot be moved (can_park is False) are left
        in VRAM by the registry.  The device move runs outside the queue
        lock; the per-GPU registry locks keep it from racing a generation.
        """
        park_now = False
        with self._cond:
            idle = self._is_idle_locked()
            if self._park_timer is not None:
                # re-armed: the pending one-shot is replaced below (when
                # still idle) or becomes moot (new work started); MEMDBG
                # counts how often the deferred park gets postponed --
                # with prefetch firing on every prompt edit, a NEVER
                # firing park would mean parked-forever copies piling up
                memdbg.stamp("park-timer", action="cancel", armed_for=f"{self._parking_delay:.0f}s")
                self._park_timer.cancel()
                self._park_timer = None
            if idle:
                delay = self._parking_delay
                if delay > 0:
                    log.info("queue idle: parking models to RAM in %.0fs",
                             delay)
                    timer = threading.Timer(delay, self._park_idle_fired)
                    timer.daemon = True
                    self._park_timer = timer
                    timer.start()
                    memdbg.stamp("park-timer", action="arm", delay=f"{delay:.0f}s")
                elif delay >= 0:
                    park_now = True
        if park_now:
            log.info("queue idle: parking all models to RAM")
            memdbg.stamp("park-timer", action="immediate")
            self._unload_all_guarded()

    def _park_idle_fired(self) -> None:
        """Timer body for ``transformer-parking-delay``: park, but only if
        the queue is STILL idle (new work may have arrived meanwhile)."""
        with self._cond:
            self._park_timer = None
            idle = self._is_idle_locked()
        memdbg.stamp("park-timer", action="fire", idle=idle)
        if not idle:
            # the workers' _maybe_unload_idle re-arms the timer once this
            # work is done
            return
        log.info("queue idle: parking all models to RAM "
                 "(transformer-parking-delay elapsed)")
        self._unload_all_guarded()

    def _unload_all_guarded(self) -> None:
        try:
            self.registry.unload_all()
        except Exception:
            # a parking failure must not kill the worker thread (this
            # runs in the worker's finally); the model stays resident and
            # the next load handles it
            log.exception("parking models to RAM failed; leaving them resident")


def _uniform_color(image: np.ndarray) -> tuple[int, int, int] | None:
    """(r, g, b) when every pixel of `image` is identical, else None.

    A uniform generation is a classic symptom of NaN/inf in the solver
    state or a VAE decode overflow (e.g. the stock VAE on high-magnitude
    latents): the clamp + uint8 cast then lands on one constant value.
    """
    if image.ndim != 3 or image.size == 0:
        return None
    if image.min() != image.max():
        return None
    return tuple(int(c) for c in image[0, 0])


def _short(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text[: limit - 1] + "…" if len(text) > limit else text


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

    # width/height are the USER'S REQUESTED size — the generation size except
    # for rotated generate/outpaint jobs, where it is the final canvas size
    # (the generation ran at the extended rotated size), so importing the
    # PNG restores the generation settings. rotation rides along so an import
    # regenerates the same extended generation. The prompt/negative prompt
    # default to the user's raw ones (styles are recorded separately, not
    # pre-applied); wildcard jobs pass the per-task RAW prompt with only the
    # wildcard picks applied (see _run_task / Job.meta_resolved).
    if meta_prompt is None:
        meta_prompt = job.meta_prompt if (job and job.meta_prompt is not None) else req.prompt
    if meta_negative is None:
        meta_negative = (
            job.meta_negative if (job and job.meta_negative is not None)
            else req.negative_prompt
        )
    p = job.plan if job is not None else None
    params = metadata.build_parameters(
        prompt=meta_prompt,
        negative_prompt=meta_negative,
        steps=result.steps if result.steps is not None else req.steps,
        sampler=req.sampler or backend.default_sampler,
        scheduler=req.scheduler or backend.default_scheduler,
        cfg=result.cfg if result.cfg is not None else req.cfg,
        seed=result.seed,
        width=p.requested_size[0] if p is not None else req.width,
        height=p.requested_size[1] if p is not None else req.height,
        # A1111/Forge write the strength only for inpaint/img2img jobs
        denoise=req.denoise if (p is not None and p.mode != "generate") else None,
        # the source rotation (0/None = absent; a rotated generate job's PNG
        # must carry it so an import regenerates the same extended size)
        rotation=p.rotation or None if p is not None else None,
        model_name=backend.name,
        model_hash=backend.model_hash,
        # the checkpoint family (models/<family>/): an import that cannot
        # find the checkpoint file falls back on this
        model_family=backend.family,
        # the configuration NAME (job.model_id is the checkpoint id): an
        # import re-selects the configuration, not just the checkpoint
        model_config=(job.meta_config_name if job is not None else None),
        loras=req.loras if backend.supports_loras else [],
        # recorded even though they were not applied to req.loras
        disabled_loras=(job.meta_disabled_loras if job is not None
                        and backend.supports_loras else None),
        styles=job.meta_styles if job is not None else None,
        soft_inpainting=req.soft_inpainting,
        sag=req.sag,
        # only Flux records its distilled/classic-CFG mode (the flag is
        # inert on SDXL, so don't add a misleading line there); it is the
        # EFFECTIVE mode (a schnell-class checkpoint silently demotes a
        # distilled request to classic CFG, so not necessarily req.distilled)
        distilled=((result.distilled if result.distilled is not None
                    else req.distilled) if backend.is_flux else None),
        # only Krea 2 records its raw/turbo checkpoint class (inert on the
        # other families); it is the effective mode as reported by the
        # backend
        turbo=((result.turbo if result.turbo is not None
                else req.turbo) if backend.is_krea2 else None),
        # the reference-image caption the backend applied (None = no
        # reference used)
        reference_caption=result.reference_caption,
        reference_sections=result.reference_sections,
        # the optional-activation-text tags that were applied (the tag row;
        # None = nothing recorded, LoRA-less backends record nothing)
        activation_tags=(job.meta_activation_tags if job is not None
                         and backend.supports_loras else None),
    )
    img, pnginfo = metadata.attach_metadata(Image.fromarray(image), params)
    buf = BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    # read the text chunks back from the saved bytes: PngInfo doesn't
    # expose its chunks as a plain dict across Pillow versions
    text = {k: v for k, v in Image.open(buf).info.items() if isinstance(v, str)}
    return buf.getvalue(), text
