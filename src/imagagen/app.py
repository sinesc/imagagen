"""FastAPI application factory + WebSocket event hub."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, field_validator

from . import memdbg, metadata, wildcards
from .backends import (
    PHASE_REFERENCE,
    PHASE_REFERENCE_DONE,
    PHASE_REFERENCE_LOADING,
    GenerateRequest,
    ModelBackend,
)
from .canvas import Canvas, _decode_mask, _decode_png
from .config import Config, ensure_dirs, load_config
from .generation import INPAINT_PADDING, GenerationError, with_inpaint
from .lora_settings import (
    LoraSettings,
    activation_tag_universe,
    all_settings,
    apply_lora_texts,
    save_settings,
)
from .model_configs import (
    ConfigError,
    ConfigStore,
    ModelConfig,
    ModelConfigUpsert,
    match_entry,
    mode_section,
    resolve_settings,
)
from .models import (
    ModelRegistry,
    filter_loras_for_family,
    list_loras,
    resolve_lora_names,
)
from .paths import next_output_path
from .presets import Preset, PresetError, PresetSaveRequest, PresetStore
from .queue import GenerationQueue
from .sag import SelfAttentionGuidance
from .segmentation import CONFIDENCE_THRESHOLD, MAX_FEATHER, SegmentationError, SegmentationService
from .soft_inpaint import SoftInpainting
from .styles import Style, StyleStore, apply_styles, strip_prompt_comments
from .uistate import UiState, UiStateStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def _lifespan(segmentation: SegmentationService):
    async def lifespan(app: FastAPI):
        yield
        # close the cached MediaPipe segmenters before the process exits
        # (otherwise their C dispatcher teardown prints GC noise)
        segmentation.close()

    return lifespan


class EventHub:
    """Thread-safe fan-out of queue events to all connected WebSocket clients.

    Each socket gets its own unbounded asyncio.Queue; `emit` (called from
    worker threads) does a thread-safe `put_nowait` on every attached queue,
    and a per-socket drain task forwards to the wire. This works even when
    the event loop thread is otherwise busy.
    """

    def __init__(self) -> None:
        self._subs: dict[WebSocket, tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = {}
        self._lock = threading.Lock()

    def attach(self, ws: WebSocket, loop: asyncio.AbstractEventLoop) -> None:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subs[ws] = (q, loop)
        loop.create_task(self._drain(ws))

    def detach(self, ws: WebSocket) -> None:
        with self._lock:
            self._subs.pop(ws, None)

    async def _drain(self, ws: WebSocket) -> None:
        sub = self._subs.get(ws)
        if sub is None:
            return
        q, _ = sub
        while True:
            event = await q.get()
            try:
                await ws.send_json(event)
            except Exception:
                self.detach(ws)
                return

    def emit(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs.values())
        for q, loop in subs:
            # Enqueue on the owning (app) loop; the TestClient websocket is
            # driven by a separate portal so this loop is never blocked by it.
            loop.call_soon_threadsafe(q.put_nowait, event)


class _MockRegistry:
    """Registry stand-in for tests / GPU-less runs: single fixed backend.

    Mirrors the (multi-GPU) ``ModelRegistry`` interface closely enough for
    the generation queue and the /api endpoints: one fake GPU 0, one shared
    backend, no real CUDA.
    """

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend
        self.gpus = [0]
        self.primary = 0

    def list(self) -> list[dict]:
        return [{"id": self._backend.model_id, "name": self._backend.name,
                 "family": "sdxl"}]

    def models(self):
        return []

    def get(self, model_id: str):
        import types

        if model_id != self._backend.model_id:
            return None
        # family mirrors ModelInfo (the mock is an SDXL stand-in); endpoints
        # like /api/samplers read it
        return types.SimpleNamespace(name=self._backend.name, family="sdxl")

    def current_id(self) -> str | None:
        return self._backend.model_id

    def current_backend(self) -> ModelBackend:
        return self._backend

    def backend_for(self, model_id: str, gpu: int | None = None) -> ModelBackend:
        return self._backend

    def run(self, model_id: str, gpu: int, request, progress=None):
        return self._backend.generate(request, progress), self._backend

    def pre_encode(self, model_id: str, gpu: int | None, pairs,
                   reserve: int | None = None):
        self._backend.pre_encode(list(pairs), reserve=reserve)

    def prefetch_text_encoder(self, model_id: str, prompt: str,
                              negative: str, force: bool = False) -> str:
        # mirrors ModelRegistry.prefetch_text_encoder for mock mode (the
        # single mock backend stands in for the resident one)
        if model_id != self._backend.model_id:
            return "not-resident"
        if not force and self._backend.encoding_cached(prompt, negative):
            return "cached"
        self._backend.prepare_text_encode()
        return "prepared"

    def unload_all(self) -> None:
        self._backend.unload()

    def cancel(self, gpu: int | None = None) -> None:
        self._backend.cancel()

    def caption(self, image, gpu: int | None = None, progress=None) -> str:
        # mock stand-in for ModelRegistry.caption (no VLM in mock mode);
        # carries the section structure real captions have and reports
        # the phase messages the real captioner emits
        from .backends import MOCK_REFERENCE_CAPTION, report_phase
        report_phase(progress, PHASE_REFERENCE_LOADING)
        report_phase(progress, PHASE_REFERENCE)
        report_phase(progress, PHASE_REFERENCE_DONE)
        return MOCK_REFERENCE_CAPTION


# ---------------- request bodies ----------------

class LoraSettingsBody(BaseModel):
    # name = the discovered LoRA name (fixed by its .safetensors; the UI's
    # name field is readonly)
    name: str
    description: str = ""
    preferred_weight: float = Field(default=0.0, ge=-10, le=10)
    activation_text: str = ""
    negative_text: str = ""
    notes: str = ""


class GenerateBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    # -1 = auto size: the mask bounding box (per dimension), scaled by
    # `oversample` and rounded up to a multiple of 16 (inpaint modes only)
    width: int = Field(default=1024, ge=-1, le=4096)
    height: int = Field(default=1024, ge=-1, le=4096)
    # scales the generation size (user-specified or mask-derived) before
    # rounding up to a multiple of 16 (inpaint modes only)
    oversample: float = Field(default=1.0, ge=0.1, le=4.0)
    # -1 = "use model default": the model configuration's value, then the
    # backend built-in (the default IS the sentinel, so callers that omit
    # the fields get the per-model built-ins, not a hard-coded value)
    steps: int = -1
    cfg: float = -1.0
    seed: int = -1
    batch_size: int = 1
    sampler: str | None = None
    scheduler: str | None = None
    loras: list[tuple[str, float]] = Field(default_factory=list)
    # LoRAs the UI shows toggled OFF: not applied, but recorded in the
    # image metadata (name@weight, in UI row order) so an import restores
    # them in the disabled state
    disabled_loras: list[tuple[str, float]] = Field(default_factory=list)
    # activation-text tags the UI switched OFF (comma-separated tags of the
    # selected LoRAs' "activation text", see lora_settings.split_tags); they
    # are dropped from the prompt, everything else applies
    disabled_activation_tags: list[str] = Field(default_factory=list)
    # ordered list of style names (styles.csv), applied to the prompt /
    # negative prompt before generation
    styles: list[str] = Field(default_factory=list)
    # style names shown toggled OFF: not applied, recorded in the metadata
    # Styles JSON with enabled=false (restored disabled on import)
    disabled_styles: list[str] = Field(default_factory=list)
    model_id: str | None = None
    mode: str = "generate"  # "generate" | "inpaint_masked" | "inpaint_not_masked"
    # the thumbnail checked as "reference image" (canvas entity id): the
    # generation conditions on that image (see GenerateRequest.reference_image)
    reference_image_id: str | None = None
    # which caption sections to fold into the prompt (None = all;
    # [] = apply nothing; see GenerateRequest.reference_sections)
    reference_sections: list[str] | None = None
    denoise: float = Field(1.0, ge=0.0, le=1.0)  # denoising strength, inpaint only
    # context padding around the masked area (canvas px per side);
    # "inpaint_masked" only, capped at the mask's largest extent
    mask_padding: int = Field(default=INPAINT_PADDING, ge=0, le=2048)
    # rotate the source by this many degrees before generating and
    # counter-rotate the result (positive = clockwise); 0 = no rotation.
    # ±45/±135 extend the generate/outpaint generation size
    rotation: int = Field(default=0, ge=-180, le=180)

    @field_validator("rotation")
    @classmethod
    def _rotation_multiple_of_45(cls, v: int) -> int:
        if v % 45:
            raise ValueError("rotation must be a multiple of 45")
        return v


class ThumbBody(BaseModel):
    job: int
    index: int


class SaveBody(BaseModel):
    prompt: str


class SegmentationMaskBody(BaseModel):
    model: str  # file name in models/segmentation/
    classes: list[str] = Field(default_factory=list)  # class labels to union
    # cut on the segmenter confidence (0..1): at/below it a pixel stays
    # transparent, above it it is fully opaque (see segmentation.py)
    threshold: float = Field(CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    # edge feather radius in canvas px (0 = hard edge)
    feather: float = Field(0.0, ge=0.0, le=MAX_FEATHER)


class SelectBody(BaseModel):
    id: str


class PinBody(BaseModel):
    id: str
    pin: bool


class CropBody(BaseModel):
    # the crop rect in canvas px; the server clamps it to the image bounds
    # (a drag that runs past the edge still crops the intersection)
    x: float
    y: float
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class ReferenceCaptionBody(BaseModel):
    image_id: str


def create_app(
    config: Config | None = None,
    backend: ModelBackend | None = None,
    segmentation: SegmentationService | None = None,
) -> FastAPI:
    config = config or load_config()
    ensure_dirs(config)
    # memory-leak tracing (imagagen.ini memory-debug): stamps every big
    # model move with the process RSS and mirrors everything to
    # user/memdbg.log; a no-op when disabled (see .memdbg for details)
    memdbg.configure(config.memory_debug,
                     interval=config.memory_debug_interval,
                     log_path=(config.user_dir / "memdbg.log"
                               if config.memory_debug else None))

    hub = EventHub()
    canvas = Canvas(brushes_dir=config.brushes_dir,
                    default_brushes_dir=config.default_brushes_dir)
    if segmentation is None:
        # models/segmentation/ may be empty; mediapipe itself is imported
        # lazily on first use (see segmentation.py)
        segmentation = SegmentationService(config.segmentation_models_dir)

    if backend is not None:  # test injection (e.g. MockBackend)
        registry = _MockRegistry(backend)
    else:
        # The registry resolves its own GPU pool from the config (primary
        # first, then the remaining usable non-ignored GPUs).
        registry = ModelRegistry(config, emit=hub.emit)
    queue = GenerationQueue(registry=registry, canvas=canvas, emit=hub.emit,
                            wildcards_dir=config.wildcards_dir,
                            parking_delay=config.transformer_parking_delay)
    queue.start()

    app = FastAPI(title="imagagen", lifespan=_lifespan(segmentation))
    app.state.config = config
    app.state.hub = hub
    app.state.canvas = canvas
    app.state.queue = queue
    app.state.registry = registry
    app.state.ui_state = UiStateStore(config.ui_state_path,
                                      persist=config.save_ui_state)
    # model configurations: files in user/model-configs/ + in-memory
    # defaults for unconfigured models (Phase 4)
    app.state.config_store = ConfigStore(config, lambda: registry.list())
    app.state.preset_store = PresetStore(config.presets_dir)
    app.state.style_store = StyleStore(config.styles_csv_path)
    app.state.segmentation = segmentation
    app.state.last_config_id: str | None = None
    memdbg.start(registry=registry, queue=queue)
    # unique per server start: the client uses it to invalidate its
    # finished-job bookkeeping after a restart (job ids restart at 1)
    app.state.server_id = uuid.uuid4().hex

    # --- cache policy ---
    # Local single-user tool: the browser must never serve stale UI state.
    # Before this middleware, responses carried no Cache-Control at all,
    # so Chrome heuristic-cached app.js (and even JSON APIs) for a while
    # after a deploy -- a restarted server then kept "showing" the old UI.
    # Default: revalidate every response (no-cache; StaticFiles' ETag makes
    # unchanged files a cheap 304). Debug mode (imagagen.ini) forbids
    # caching entirely.
    @app.middleware("http")
    async def cache_headers(request: Request, call_next):
        response = await call_next(request)
        if config.debug:
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/health")
    def health() -> dict:
        from .gpus import gpu_info, select_gpu

        selected = select_gpu(config.primary_gpu, config.ignore_gpus)
        primary = getattr(registry, "primary", selected)
        pool = list(getattr(registry, "gpus", [selected] if selected is not None else []))
        return {
            "status": "ok",
            "root": str(config.root),
            "gpu": selected,
            "gpus": gpu_info(config.ignore_gpus, selected),
            "primary": primary,
            "pool": pool,
        }

    @app.get("/api/memdbg")
    def memdbg_report() -> Response:  # sync: threadpool (it walks the heap)
        """On-demand memory census (imagagen.ini memory-debug = true).

        Returns a plain-text report: RSS/swap, every live model copy and
        which device its weights are on, the biggest UNATTRIBUTED CPU
        tensors and who references them, CUDA allocator state and the
        registry/queue/cache bookkeeping. Call it at the moment the system
        monitor shows RAM filling up. Disabled (default) -> a one-line
        notice; nothing is tracked then.
        """
        text = memdbg.census("api")
        return Response(content=text + "\n", media_type="text/plain")

    @app.get("/api/ui-config")
    def ui_config() -> dict:
        """Client-side knobs read from imagagen.ini, delivered to the UI
        at boot instead of hardcoding them in the frontend."""
        return {"mask_overlay_opacity": config.mask_overlay_opacity}

    # --- models & model configurations (Phase 4) ---
    # The model selector lists *configurations*, not raw checkpoints:
    # every file config in user/model-configs/ plus one in-memory default per
    # unconfigured model. Generation resolves the config's defaults
    # (sampler/scheduler/CFG/LoRAs) server-side.
    @app.get("/api/models")
    def list_models() -> dict:
        try:
            entries = app.state.config_store.entries()
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        current = app.state.last_config_id
        ids = {e.id for e in entries}
        if current not in ids and entries:
            current = entries[0].id
        # attach each entry's model family so the UI can decide which
        # family-specific controls to show (e.g. Flux's "distilled model")
        # synchronously -- reading it from an extra /api/samplers fetch would
        # race the model load and hide the control until that fetch settled
        family_of = {m["id"]: m["family"] for m in app.state.registry.list()}
        models = []
        for e in entries:
            d = e.to_dict()
            d["family"] = family_of.get(e.model_id, "sdxl")
            models.append(d)
        return {"models": models, "current": current}

    @app.get("/api/samplers")
    def list_samplers(model_id: str | None = None) -> dict:
        """Samplers/schedulers for a model's family.

        ``model_id`` selects the family so a Flux model offers only its
        Euler sampler (and no schedulers); without it (or for SDXL) the
        SDXL sets are returned, as before.
        """
        family = "sdxl"
        if model_id is not None:
            info = app.state.registry.get(model_id)
            if info is not None:
                family = info.family
        if family == "flux":
            from .backends.flux import SAMPLERS, SCHEDULERS

            return {"samplers": list(SAMPLERS), "schedulers": list(SCHEDULERS)}
        if family == "zimage":
            from .backends.zimage import SAMPLERS, SCHEDULERS

            return {"samplers": list(SAMPLERS), "schedulers": list(SCHEDULERS)}
        if family == "krea2":
            from .backends.krea2 import SAMPLERS, SCHEDULERS

            return {"samplers": list(SAMPLERS), "schedulers": list(SCHEDULERS)}
        from .backends.sdxl import SAMPLERS, SCHEDULERS

        return {"samplers": list(SAMPLERS), "schedulers": list(SCHEDULERS)}

    @app.get("/api/loras")
    def loras(family: str | None = None) -> dict:
        # family filters to loras/<family>/ so the UI only offers LoRAs
        # for the selected model's family (default: all, as before)
        return {"loras": list_loras(config, family=family),
                "settings": all_settings(config)}

    @app.post("/api/loras/settings")
    def lora_settings_save(body: LoraSettingsBody) -> dict:
        # name is fixed by the LoRA file (the UI's name field is readonly)
        try:
            s = LoraSettings.model_validate(
                body.model_dump(exclude={"name"}))
            settings = save_settings(body.name, s, config)
        except KeyError:
            return JSONResponse(
                {"error": f"unknown LoRA: {body.name!r}"}, status_code=404)
        except ValidationError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True, "settings": settings}

    @app.post("/api/prompt-tokens")
    def prompt_tokens(body: dict) -> dict:
        """Live CLIP token count of the prompt field (UI counter).

        The selected styles and LoRA activation texts are applied first
        (after comment lines are dropped), exactly as ``/api/generate``
        does, so the count reflects the full prompt the model will
        actually see (a style or LoRA text can push a short raw prompt
        past one CLIP window).

        ``tokens`` is None when the tokenizer could not be loaded; the UI then
        hides the counter. ``windows`` is how many 75-token CLIP windows the
        prompt occupies (the long-prompt chunked-encoding window size).
        """
        from .tokenize import count_prompt_tokens, prompt_windows

        prompt, _ = apply_styles(
            strip_prompt_comments(body.get("prompt", "")),
            "",
            body.get("styles", []),
            app.state.style_store.get,
        )
        # LoRA activation texts are part of the model prompt, too
        prompt, _ = apply_lora_texts(
            prompt, "", body.get("loras", []), config,
            body.get("disabled_activation_tags"),
        )
        tokens = count_prompt_tokens(prompt)
        if tokens is None:
            return {"tokens": None, "windows": 0}
        return {"tokens": tokens, "windows": prompt_windows(tokens)}

    @app.post("/api/prefetch-text-encoder")
    def prefetch_text_encoder(body: dict) -> dict:  # sync: threadpool
        """Prompt-change prefetch (imagagen.ini prefetch-text-encoder).

        The UI posts the current prompt form shortly after every prompt
        change. When the FINAL prompt (comments stripped, styles and LoRA
        activation texts applied, exactly as /api/generate computes it)
        would miss the model's prompt-encoding cache, the server moves the
        GPU-resident model into its "about to encode" layout right away:
        text encoders to the GPU, transformer parked to RAM when the two
        cannot co-reside -- instead of waiting for the next generation to
        pay the swap. ``action``: ``disabled`` (ini off) | ``cached`` (no
        work needed) | ``prepared`` | ``not-resident`` (lazy loading wins:
        nothing is woken up for a prompt alone) | ``busy`` | ``failed``
        (the layout move raised on every resident GPU -- a backend bug, see
        the server traceback) | ``no-model``.

        Wildcard prompts always count as a miss (the per-seed line picks
        cannot be predicted, so the raw text is never encoded here); the
        layout preparation still applies.
        """
        if not config.prefetch_text_encoder:
            memdbg.stamp("prefetch-api", action="ini-disabled")
            return {"action": "disabled"}
        cfg_id = body.get("model_id") or app.state.last_config_id
        if not cfg_id:
            memdbg.stamp("prefetch-api", action="no-model-id")
            return {"action": "no-model"}
        try:
            entry = app.state.config_store.resolve(cfg_id)
        except ConfigError:
            return {"action": "no-model"}
        prompt, negative = apply_styles(
            strip_prompt_comments(body.get("prompt", "")),
            strip_prompt_comments(body.get("negative_prompt", "")),
            body.get("styles", []),
            app.state.style_store.get,
        )
        prompt, negative = apply_lora_texts(
            prompt, negative, body.get("loras", []), config,
            body.get("disabled_activation_tags"))
        # wildcards: the per-seed resolution always misses the cache, so
        # the raw text must not be checked (nor encoded)
        force = (wildcards.has_wildcards(prompt)
                 or wildcards.has_wildcards(negative))
        action = registry.prefetch_text_encoder(entry.model_id, prompt,
                                                negative, force=force)
        if action not in ("cached", "not-resident"):
            log.info("prompt prefetch: model=%s action=%s", entry.model_id,
                     action)
        return {"action": action}

    @app.post("/api/models/select")
    def select_model(body: dict) -> dict:  # sync: runs in the threadpool
        try:
            entry = app.state.config_store.resolve(body.get("id", ""))
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        # selection only records the current entry; the checkpoint is
        # loaded lazily on the first generation that uses it (and the
        # client-side LoRA family check runs immediately, independently)
        app.state.last_config_id = entry.id
        return {"current": entry.id, "model_id": entry.model_id}

    @app.post("/api/model-configs")
    def upsert_model_config(body: ModelConfigUpsert) -> dict:
        # optional id in the JSON body: the entry being edited. Saving with
        # a *default* entry's id materializes it into a file config
        # (defaults are editable by turning them into real configurations)
        try:
            mc = ModelConfig.model_validate(body.model_dump(exclude={"id"}))
            entry = app.state.config_store.save(mc, id_=body.id)
        except (ConfigError, ValidationError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if body.id is not None and entry.id != body.id:
            # materialized (or renamed) entry: keep the selection on it
            app.state.last_config_id = entry.id
        return entry.to_dict()

    @app.post("/api/model-configs/create")
    def create_model_config(body: dict) -> dict:
        """Create a configuration copying `based_on` (concept.md: create =
        copy of the currently selected configuration, then edit it)."""
        try:
            entry = app.state.config_store.create_from(
                body.get("name", ""), body.get("based_on", "")
            )
        except (ConfigError, ValidationError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        app.state.last_config_id = entry.id
        return entry.to_dict()

    @app.post("/api/model-configs/delete")
    def delete_model_config(body: dict) -> dict:
        # POST with a body (not a path param): ids may contain slashes
        # (default entries are named after the model file)
        try:
            app.state.config_store.delete(body.get("id", ""))
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if app.state.last_config_id == body.get("id"):
            app.state.last_config_id = None
        return {"ok": True}

    # --- presets (Phase 4) ---
    @app.get("/api/presets")
    def presets_list() -> dict:
        return {"presets": [
            {"name": p.name, "group": p.group}
            for p in app.state.preset_store.all()
        ]}

    @app.get("/api/presets/{name}")
    def preset_get(name: str) -> dict:
        p = app.state.preset_store.get(name)
        if p is None:
            return JSONResponse({"error": f"unknown preset: {name!r}"}, status_code=404)
        return {"preset": p.to_dict()}

    @app.post("/api/presets")
    def preset_save(req: PresetSaveRequest) -> dict:
        # old_name identifies the preset being edited: a name change renames
        # it (old file deleted) instead of creating a new preset
        try:
            p = Preset.model_validate(req.model_dump(exclude={"old_name"}))
            id_ = app.state.preset_store.save(p, old_name=req.old_name)
        except (PresetError, ValidationError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True, "id": id_}

    @app.post("/api/presets/delete")
    def preset_delete(body: dict) -> dict:
        name = body.get("name", "")
        if not app.state.preset_store.delete(name):
            return JSONResponse({"error": f"unknown preset: {name!r}"}, status_code=404)
        return {"ok": True}

    # --- styles (user/styles.csv + in-memory "Session" temporaries) ---
    @app.get("/api/styles")
    def styles_list() -> dict:
        return {"styles": app.state.style_store.all()}

    @app.post("/api/styles")
    def style_save(style: Style) -> dict:
        # Style is validated by FastAPI (422); the store upserts by name
        app.state.style_store.save(style)
        return {"ok": True}

    @app.post("/api/styles/delete")
    def style_delete(body: dict) -> dict:
        name = body.get("name", "")
        if not app.state.style_store.delete(name):
            return JSONResponse({"error": f"unknown style: {name!r}"}, status_code=404)
        return {"ok": True}

    @app.post("/api/styles/temp")
    def style_temp(body: dict) -> dict:
        """Register styles from imported image metadata that are missing
        from styles.csv as in-memory temporaries ("Session" optgroup, never
        persisted). Existing file styles win; temporaries are updated."""
        for s in body.get("styles") or []:
            try:
                style = Style.model_validate(s)
            except ValidationError:
                continue
            app.state.style_store.add_temporary(style)
        return {"styles": app.state.style_store.all()}

    # --- generation ---
    @app.post("/api/generate")
    def generate(body: GenerateBody) -> dict:
        # the auto-size sentinel (-1) resolves from the mask bounding box,
        # which only exists for the inpaint modes
        if body.mode == "generate" and (body.width == -1 or body.height == -1):
            return JSONResponse(
                {"error": "auto size (-1) requires an inpaint mode"},
                status_code=400)
        # model_id here is a *configuration* id (Phase 4): files in
        # user/model-configs/ or the in-memory default of an unconfigured model.
        # Default-override resolution (concept.md): user setting -> (only
        # when it is the "use model default" sentinel: cfg == -1, sampler/
        # scheduler null) the config's t2i/i2i value -> backend built-in.
        try:
            cfg_id = body.model_id or app.state.last_config_id
            if cfg_id is None:
                entries = app.state.config_store.entries()
                cfg_id = entries[0].id if entries else None
            if cfg_id is None:
                return {"error": "no model available"}
            entry = app.state.config_store.resolve(cfg_id)
            app.state.last_config_id = entry.id
            sampler, scheduler, cfg, rsteps, loras, sag_enabled = (
                resolve_settings(
                    entry,
                    body.mode,
                    body.cfg,
                    body.sampler,
                    body.scheduler,
                    body.loras,
                    body.steps,
                )
            )
            sag_section = mode_section(entry, body.mode)
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        model_id = entry.model_id
        # A LoRA discovered under ANOTHER family's dir does not match the
        # selected checkpoint: ignore it (logged, and recorded as disabled
        # so the PNG metadata round-trips) instead of failing the job. The
        # UI force-disables such rows on model switch, but a stale UI (or
        # a scripted JSON caller) may still send them.
        info = app.state.registry.get(model_id)
        family = info.family if info is not None else "sdxl"
        loras, foreign_loras = filter_loras_for_family(loras, family, config)
        disabled_loras = list(body.disabled_loras)
        if foreign_loras:
            log.warning(
                "generate: ignoring %d LoRA(s) from another family "
                "(model family is %s): %s",
                len(foreign_loras), family,
                ", ".join(n for n, _ in foreign_loras),
            )
            known = {n for n, _ in disabled_loras}
            disabled_loras += [
                (n, w) for n, w in foreign_loras if n not in known
            ]
        # Styles are applied right before the prompt reaches the model
        # (in list order; {prompt} replaced with the prompt so far).
        # Each enabled LoRA's activation/negative text (from its settings
        # file) is applied the same way (in LoRA list order, after the
        # styles); empty texts are a no-op.
        # The raw (unapplied) prompt and the style *contents* are kept for
        # the image metadata: the PNG must round-trip as "same prompt +
        # same style list", not as the pre-applied text. The LoRAs are
        # likewise recorded by name/weight, not their expanded texts.
        # comments (a # and everything after it on a line) are ignored
        # during processing; meta_prompt/meta_negative below keep the raw
        # text so they remain in the stored metadata (the queue resolves
        # the wildcards in THAT text for the PNG, never the style/LoRA
        # expansions -- see Job.meta_resolved)
        prompt, negative_prompt = apply_styles(
            strip_prompt_comments(body.prompt),
            strip_prompt_comments(body.negative_prompt),
            body.styles,
            app.state.style_store.get,
        )
        prompt, negative_prompt = apply_lora_texts(
            prompt, negative_prompt, loras, config,
            body.disabled_activation_tags,
        )
        # the tags that DID make it into the prompt (the universe of the
        # applied LoRAs minus the ones the UI switched off): recorded in the
        # PNG so it says what the image was generated with
        applied_tags = [
            t for t in activation_tag_universe(loras, config)
            if t not in set(body.disabled_activation_tags)
        ]
        # Wildcards are validated up front (seed-independent structural
        # errors: a missing file or a recursion) so the whole job is rejected
        # with a clear message instead of failing per-image; the actual
        # per-seed line picks happen later, in the queue.
        try:
            wildcards.validate(prompt, config.wildcards_dir)
            wildcards.validate(negative_prompt, config.wildcards_dir)
        except wildcards.WildcardError as e:
            log.error("generate: %s", e)
            return JSONResponse({"error": str(e)}, status_code=400)
        meta_styles = [
            {
                "name": name,
                "prompt": (s := app.state.style_store.get(name)).prompt,
                "negative_prompt": s.negative_prompt,
                "enabled": True,
            }
            for name in body.styles
            if app.state.style_store.get(name) is not None
        ]
        # styles the UI shows toggled off: not applied to the prompt, but
        # recorded (enabled=false) so an import restores them as disabled
        for name in body.disabled_styles:
            s = app.state.style_store.get(name)
            if s is not None:
                meta_styles.append({
                    "name": name,
                    "prompt": s.prompt,
                    "negative_prompt": s.negative_prompt,
                    "enabled": False,
                })
        # the reference image rides on the request exactly like the
        # img2img base image: resolved in-process from the canvas store
        # (the same path as /api/canvas/image.png); JSON callers never
        # carry pixel data
        reference_image = None
        if body.reference_image_id is not None:
            if not canvas.has_entity(body.reference_image_id):
                return JSONResponse(
                    {"error": "reference image is no longer available"},
                    status_code=400)
            reference_image = _decode_png(canvas.entity_png(
                body.reference_image_id))
        req = GenerateRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=reference_image,
            reference_sections=body.reference_sections,
            width=body.width,
            height=body.height,
            # rsteps/cfg are None only when the user used the -1 sentinel
            # and the configuration has no override. The sentinel then rides
            # on the request and the *backend* resolves its built-in default
            # (Flux's built-ins differ from SDXL's, so they cannot be
            # resolved here)
            steps=rsteps if rsteps is not None else -1,
            cfg=cfg if cfg is not None else -1,
            # the distilled (Flux) / turbo (Krea 2) flags are configuration
            # properties (not user form fields): they come straight from the
            # selected entry (inert on the other families)
            distilled=entry.distilled,
            turbo=entry.turbo,
            seed=body.seed,
            sampler=sampler,
            scheduler=scheduler,
            loras=loras,
            # soft inpainting: like SAG, the enabled flag AND the option
            # values come from the model configuration — but i2i ONLY
            # (mode_section returns the i2i section for the inpaint modes,
            # which are the only ones with an init mask; enabled by
            # default). Sent only for the inpaint modes, so generate-mode
            # PNGs carry no soft metadata and the inpaint PNGs carry
            # Forge's "Soft inpainting ..." lines unconditionally, like the
            # extension
            soft_inpainting=(
                SoftInpainting(
                    enabled=sag_section.soft,
                    mask_blend_power=sag_section.soft_mask_blend_power,
                    mask_blend_scale=sag_section.soft_mask_blend_scale,
                    inpaint_detail_preservation=sag_section.soft_inpaint_detail_preservation,
                    composite_mask_influence=sag_section.soft_composite_mask_influence,
                    composite_difference_threshold=sag_section.soft_composite_difference_threshold,
                    composite_difference_contrast=sag_section.soft_composite_difference_contrast,
                )
                if body.mode != "generate" else None
            ),
            # self attention guidance: the enabled flag AND the option
            # values come from the model configuration's per-direction
            # section (SDXL only); the object is always present so the PNG
            # metadata carries the Forge sag_* keys unconditionally, like
            # the extension
            sag=SelfAttentionGuidance(
                enabled=sag_enabled, scale=sag_section.sag_scale,
                blur_sigma=sag_section.sag_blur_sigma,
                threshold=sag_section.sag_threshold),
        )
        try:
            # inpaint modes snapshot the canvas and populate init image/mask;
            # "generate" leaves the request untouched
            plan = with_inpaint(req, body.mode, canvas, body.denoise,
                                rotation=body.rotation,
                                padding=body.mask_padding,
                                oversample=body.oversample)
        except GenerationError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        job = queue.submit(model_id, req, body.batch_size, plan=plan,
                           meta_prompt=body.prompt,
                           meta_negative=body.negative_prompt,
                           meta_styles=meta_styles,
                           # recorded in the metadata but NOT applied
                           meta_disabled_loras=disabled_loras,
                           # the activation-text tags that were applied
                           meta_activation_tags=applied_tags or None,
                           # the configuration NAME rides to the PNG metadata
                           # so an import re-selects it
                           meta_config_name=entry.name)
        return {
            "job_id": job.id,
            "server_id": app.state.server_id,
            # fast failures can settle the job before this response is
            # delivered; the client must not flip the button to "Interrupt"
            # for a job that is already over
            "done": job.done_emitted or job.interrupted,
        }

    @app.post("/api/interrupt")
    def interrupt(body: dict) -> dict:
        job_id = int(body.get("job", -1))
        if job_id >= 0:
            queue.interrupt(job_id)
        else:
            for jid in list(queue.jobs()):
                queue.interrupt(jid)
        return {"ok": True}

    @app.get("/api/queue")
    def queue_jobs() -> dict:
        # Fetched on page load so the UI can rebuild its thumbnail strip
        # after a browser reload: the 'source_image'/'image' WS events
        # emitted before the reload are lost. The server only retains
        # in-flight jobs plus the last finished one (older jobs are
        # dropped on submit, queue.py), so those are the only job-based
        # thumbnails; user-pinned images ("Pin thumbnail") outlive their
        # jobs and are listed separately in `pinned`.
        jobs = []
        for jid in sorted(queue.jobs()):
            job = queue.job(jid)
            jobs.append({
                "id": job.id,
                "source": (
                    {"url": f"/api/thumb/{job.id}/source.png",
                     "image_id": job.source_image_id,
                     "seed": job.source_seed}
                    if job.source_png is not None else None
                ),
                "results": [
                    {"index": idx, "seed": job.base_seed + idx,
                     "url": f"/api/thumb/{job.id}/{idx}.png",
                     "image_id": f"job:{job.id}:{idx}"}
                    for idx in sorted(job.results)
                ],
            })
        return {"jobs": jobs, "current": canvas.current_id(),
                "pinned": sorted(canvas.user_pinned())}

    @app.get("/api/thumb/{job_id}/source.png")
    def thumb_source(job_id: int) -> Response:
        # the source image (canvas snapshot at submit time; inpaint
        # always, generate on a non-empty canvas); registered before the
        # {index}.png route so "source" is not parsed as an index.
        # The live canvas entity wins when it exists: a source the user
        # keeps editing has a fresher cold PNG than the submit snapshot.
        job = queue.job(job_id)
        data = None
        if job is not None and job.source_image_id is not None:
            data = canvas.entity_png(job.source_image_id)
        if data is None and job is not None:
            data = job.source_png
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    @app.get("/api/thumb/{job_id}/{index}.png")
    def thumb(job_id: int, index: int) -> Response:
        job = queue.job(job_id)
        # the canvas entity is authoritative when it exists (edits arrive
        # through /api/canvas/sync and update its cold PNG); the job's own
        # snapshot only covers entities that never became, or ceased to be,
        # canvas images
        data = canvas.entity_png(f"job:{job_id}:{index}")
        if data is None and job is not None:
            data = job.results.get(index)
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    # --- UI state (persisted form inputs, restored on server start) ---
    @app.get("/api/ui-state")
    def get_ui_state() -> dict:
        state = app.state.ui_state.load()
        # The persisted model may no longer exist (configuration deleted or
        # renamed): the UI must always receive a selectable value, so fall
        # back to the first configuration.
        if state.model is not None:
            try:
                entries = app.state.config_store.entries()
            except ConfigError:
                entries = []
            if entries and state.model not in {e.id for e in entries}:
                state.model = entries[0].id
        # the persisted reference thumbnail may no longer exist (its entity
        # was evicted, or the server restarted): drop the dangling id so the
        # UI never restores a checkbox pointing at a dead image
        if state.reference_image_id is not None and not canvas.has_entity(
                state.reference_image_id):
            state.reference_image_id = None
            state.reference_sections = None
        # by_alias: the tool rows use their JSON names ("copy", not the
        # field's "copy_area", which must not shadow BaseModel.copy())
        return {"state": state.model_dump(by_alias=True)}

    @app.post("/api/ui-state")
    async def save_ui_state(request: Request) -> dict:
        # Content-type agnostic on purpose: the UI also sends this via
        # navigator.sendBeacon on tab close, which forces text/plain.
        raw = await request.body()
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        if not isinstance(data, dict):
            return JSONResponse(
                {"error": "invalid state: expected a JSON object"}, status_code=400
            )
        try:
            state = UiState.model_validate(data)
        except ValidationError as e:
            return JSONResponse(
                {"error": f"invalid state: {e.errors()[:3]}"}, status_code=422
            )
        # always keeps the in-memory copy (served on browser reloads);
        # save-ui-state=false only suppresses the disk write
        app.state.ui_state.save(state)
        return {"ok": True}

    # --- canvas ---
    @app.get("/api/canvas.png")
    def get_canvas() -> Response:
        data = canvas.to_png_bytes()
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    @app.get("/api/mask.png")
    def get_mask() -> Response:
        data = canvas.to_mask_png_bytes()
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    @app.post("/api/canvas/sync")
    async def canvas_sync(request: Request) -> dict:
        """Client-authoritative state push: the browser editor owns the live
        canvas and pushes the full state (image + mask PNGs) after edits."""
        if not canvas.has_image():
            return JSONResponse({"error": "canvas is empty"}, status_code=400)
        form = await request.form()
        img_file, mask_file = form.get("image"), form.get("mask")
        if img_file is None or mask_file is None:
            return JSONResponse(
                {"error": "image and mask required"}, status_code=400
            )
        try:
            img = _decode_png(await img_file.read())
            mask = _decode_mask(await mask_file.read())
        except Exception:
            return JSONResponse({"error": "could not read image"}, status_code=400)
        size = canvas.size()
        if size is None:
            return JSONResponse({"error": "canvas is empty"}, status_code=400)
        w, h = size
        if img.shape != (h, w, 3) or mask.shape != (h, w):
            return JSONResponse({"error": "size mismatch"}, status_code=400)
        canvas.replace_state(img, mask)
        return {"ok": True}

    @app.get("/api/brushes")
    def list_brushes() -> dict:
        return {"brushes": canvas.list_brushes()}

    @app.get("/api/brush/{name}.png")
    def brush_png(name: str) -> Response:
        """The brush's greyscale PNG (luminance = alpha) so the client's
        stroke preview stamps the exact brush shape; 404 when unknown."""
        try:
            data = canvas.brush_png(name)
        except KeyError:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    @app.post("/api/canvas/upload")
    async def upload_canvas(request: Request) -> dict:
        data = await request.body()
        # ?replace=<id>: the client re-pointed its upload thumbnail at this
        # new image, so the superseded upload entity must not linger pinned
        replace = request.query_params.get("replace")
        try:
            image_id = canvas.import_image(data, replace_id=replace)
        except Exception:
            return JSONResponse({"error": "could not read image"}, status_code=400)
        # the new entity's id lets the UI add a restorable thumbnail
        return {"ok": True, "canvas": canvas.has_image(), "image_id": image_id}

    @app.post("/api/canvas/import-pin")
    async def import_pin_canvas(request: Request) -> dict:
        """Thumbnail-strip drag&drop: register the PNG as a pinned canvas
        entity (a permanent thumbnail) and leave the displayed image alone.
        Restorable via /api/canvas/select, served by /api/canvas/image.png."""
        data = await request.body()
        try:
            image_id = canvas.import_pinned(data)
        except Exception:
            return JSONResponse({"error": "could not read image"}, status_code=400)
        return {"ok": True, "image_id": image_id}

    @app.post("/api/png-meta")
    async def png_meta(request: Request) -> dict:
        """Parse the PNG's embedded generation metadata (A1111/Forge
        'parameters' chunk or standalone text chunks) plus the image
        dimensions, for drag&drop import in the UI."""
        data = await request.body()
        try:
            img = Image.open(BytesIO(data))
            width, height = img.size
        except Exception:
            return JSONResponse({"error": "could not read image"}, status_code=400)
        meta = metadata.read_metadata(data)
        # A1111/Forge images store the fully expanded prompt and no style
        # list, so the selected styles are resolved out of the expanded
        # prompt the same way Forge does (StyleStore.extract_styles). imagagen
        # images (JSON metadata, "format" key) carry an explicit style list
        # -- or none -- and their RAW prompts must be shown verbatim
        # (extract_styles would normalize/rewrite them), so this only runs
        # for foreign legacy-text metadata. Meta-data save logic is untouched.
        if (meta is not None and "styles" not in meta and meta.get("prompt")
                and "format" not in meta):
            # A1111/Forge bake LoRAs into the prompt text (not meta lines); pull
            # them out and hand them to the normal LoRA selection logic, then
            # resolve the styles out of the remainder the same way Forge does
            # (StyleStore.extract_styles). The prompt shown is the original
            # un-applied, LoRA-free base. Meta-data save logic is untouched.
            loras, no_lora_prompt = metadata.extract_prompt_loras(meta["prompt"])
            if loras:
                # A1111/Forge record LoRAs by name only; when a LoRA lives in
                # a subfolder the verbatim name is not a discovered one and
                # generation would fail. Resolve to the on-disk name (subfolder
                # by unique basename, display-named LoRAs via Forge's hashes).
                resolved = resolve_lora_names(
                    [n for n, _ in loras], config, meta.get("lora_hashes")
                )
                meta.setdefault("loras", []).extend(
                    f"{n}@{w:g}" for n, (_, w) in zip(resolved, loras, strict=True)
                )
            found, prompt, negative = app.state.style_store.extract_styles(
                no_lora_prompt, meta.get("negative_prompt", "")
            )
            meta["prompt"] = prompt
            if found:
                meta["styles"] = [s.to_dict() for s in found]
                if "negative_prompt" in meta:
                    meta["negative_prompt"] = negative
        if meta is not None:
            # The model configuration the image was generated with, resolved
            # to a selectable configuration id (the UI selects meta.config_id).
            # imagagen PNGs name it outright (model_config); A1111/Forge PNGs
            # are scored over checkpoint + LoRAs, so the configuration that
            # also provides the image's LoRAs wins over a bare-checkpoint one.
            cfg_name = meta.get("model_config")
            try:
                entries = app.state.config_store.entries()
            except ConfigError:
                entries = []  # a broken config file must not break the import
            hit = match_entry(
                entries,
                model=meta.get("model"),
                loras=metadata.parse_lora_entries(meta.get("loras")),
                config_name=cfg_name if isinstance(cfg_name, str) else None,
            )
            if hit is not None:
                meta["config_id"] = hit.id
        return {"meta": meta, "width": width, "height": height}

    @app.post("/api/canvas/thumb")
    def set_canvas_from_thumb(body: ThumbBody) -> dict:
        job = queue.job(body.job)
        if job is None or body.index not in job.results:
            return {"error": "no such thumbnail"}
        try:
            canvas.set_current(f"job:{body.job}:{body.index}")
        except ValueError as e:
            return {"error": str(e)}
        return {"ok": True}

    @app.post("/api/canvas/select")
    def select_canvas_image(body: SelectBody) -> dict:
        try:
            canvas.set_current(body.id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True}

    @app.post("/api/canvas/pin")
    def canvas_pin(body: PinBody) -> dict:
        """The UI's per-thumbnail "Pin thumbnail" checkbox: pinned entities
        survive the job drops that every new generation performs, so the
        thumbnail (and its image, restorable via /api/canvas/select) stays
        until it is unchecked."""
        if body.pin:
            try:
                canvas.user_pin(body.id)
            except ValueError as e:
                return JSONResponse({"error": str(e)}, status_code=400)
        else:
            canvas.user_unpin(body.id)
        return {"ok": True}

    @app.post("/api/canvas/crop")
    def canvas_crop(body: CropBody) -> dict:
        if not canvas.has_image():
            return JSONResponse({"error": "canvas is empty"}, status_code=400)
        try:
            prev_id, crop_id = canvas.crop(body.x, body.y, body.width, body.height)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {
            "ok": True,
            "prev_image_id": prev_id,
            "crop_image_id": crop_id,
            "prev_url": f"/api/canvas/image.png?id={prev_id}",
            "crop_url": f"/api/canvas/image.png?id={crop_id}",
        }

    @app.get("/api/canvas/image.png")
    def canvas_image_png(id: str) -> Response:
        """Serve the PNG of any known canvas entity (crop/result thumbnail)."""
        data = canvas.entity_png(id)
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    # --- reference image ---
    @app.post("/api/reference/caption")
    def reference_caption(body: ReferenceCaptionBody) -> dict:
        """Describe a canvas image with the VLM (the text a checked
        "reference image" thumbnail injects into the prompt).

        Runs on the primary GPU's worker path (registry.caption holds the
        per-GPU lock, so it never races a generation; a resident model is
        parked first to free VRAM for the captioning towers). The UI
        fetches this when a thumbnail is checked as reference and shows
        the caption in the checkbox tooltip.
        """
        png = canvas.entity_png(body.image_id)
        if png is None:
            return JSONResponse(
                {"error": "reference image is no longer available"},
                status_code=400)
        try:
            def _preview_progress(_frac, message):
                text = {
                    PHASE_REFERENCE_LOADING: "Loading captioning model",
                    PHASE_REFERENCE: "Describing reference image",
                    PHASE_REFERENCE_DONE: "Reference caption ready",
                }.get(message)
                if text:
                    hub.emit({"type": "phase", "message": text})
            caption = registry.caption(_decode_png(png),
                                       progress=_preview_progress)
        except Exception as e:
            log.exception("reference caption failed")
            return JSONResponse({"error": str(e)}, status_code=500)
        return {"caption": caption}

    # --- person mask generation (Phase 5) ---
    # Segmentation models are .tflite files in models/segmentation/; the
    # class names come from the model itself, so the UI can offer a
    # per-class checkbox list for any multiclass segmenter.
    @app.get("/api/segmentation/models")
    def segmentation_models() -> dict:
        try:
            models = app.state.segmentation.list_models()
        except Exception as e:  # e.g. mediapipe import failure
            return JSONResponse({"error": str(e)}, status_code=400)
        return {
            "models": [
                {"id": m.id, "name": m.name, "labels": list(m.labels)}
                for m in models
            ]
        }

    @app.post("/api/segmentation/mask")
    def segmentation_mask(body: SegmentationMaskBody) -> dict:
        if not canvas.has_image():
            return JSONResponse({"error": "canvas is empty"}, status_code=400)
        image = canvas.image
        assert image is not None
        try:
            seg = app.state.segmentation.generate(
                body.model, body.classes, image,
                threshold=body.threshold, feather=body.feather)
        except SegmentationError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        # union with the existing mask (element-wise max of the alpha);
        # a user who wants a fresh mask clears it first. The mask is at
        # the current image's size, so union in that pixel space directly.
        merged = np.maximum(canvas.mask, seg)
        changed = canvas.set_mask(merged)
        return {
            "ok": True,
            "changed": changed,
            "pixels": int((merged > 0).sum()),
        }

    # --- save ---
    def _origin_prompt(fallback: str) -> str:
        """The prompt recorded in the current canvas image's PNG metadata
        (i.e. the prompt used when the image was generated); falls back to
        the UI's current prompt for images without generation metadata
        (uploads, edits that dropped the chunks)."""
        cid = canvas.current_id()
        info = canvas.entity_info(cid) if cid is not None else None
        if info:
            params = info.get("parameters")
            if params:
                try:
                    meta = metadata.parse_parameters(params)
                except Exception:
                    meta = None
                prompt = (meta or {}).get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    return prompt
            prompt = info.get("prompt")  # standalone-chunk fallback
            if isinstance(prompt, str) and prompt.strip():
                return prompt
        return fallback

    @app.post("/api/save")
    def save(body: SaveBody) -> dict:
        data = canvas.to_png_bytes()
        if data is None:
            return {"error": "canvas is empty"}
        out = next_output_path(config.output_dir, _origin_prompt(body.prompt))
        out.write_bytes(data)
        return {"ok": True, "path": out.name}

    @app.get("/api/output")
    def list_output() -> list[dict]:
        files = sorted(config.output_dir.iterdir()) if config.output_dir.is_dir() else []
        return [{"name": f.name, "size": f.stat().st_size} for f in files if f.is_file()]

    # --- events ---
    @app.websocket("/api/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        hub.attach(websocket, asyncio.get_running_loop())
        try:
            while True:
                await websocket.receive_text()  # keep-alive pings from client
        except WebSocketDisconnect:
            pass
        finally:
            hub.detach(websocket)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
