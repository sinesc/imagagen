"""FastAPI application factory + WebSocket event hub."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from . import metadata, wildcards
from .backends import DEFAULT_CFG, GenerateRequest, ModelBackend
from .canvas import Canvas
from .config import Config, ensure_dirs, load_config
from .generation import INPAINT_PADDING, GenerationError, with_inpaint
from .lora_settings import LoraSettings, all_settings, save_settings
from .model_configs import (
    ConfigError,
    ConfigStore,
    ModelConfig,
    ModelConfigUpsert,
    resolve_settings,
)
from .models import ModelRegistry, list_loras, resolve_lora_names
from .paths import next_output_path
from .presets import Preset, PresetError, PresetStore
from .queue import GenerationQueue
from .sag import SelfAttentionGuidance
from .segmentation import SegmentationError, SegmentationService
from .soft_inpaint import SoftInpainting
from .styles import Style, StyleStore, apply_styles
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
    """Registry stand-in for tests / GPU-less runs: single fixed backend."""

    def __init__(self, backend: ModelBackend) -> None:
        self._backend = backend

    def list(self) -> list[dict]:
        return [{"id": self._backend.model_id, "name": self._backend.name}]

    def models(self):
        return []

    def current_id(self) -> str | None:
        return self._backend.model_id

    def current_backend(self) -> ModelBackend:
        return self._backend

    def backend_for(self, model_id: str) -> ModelBackend:
        return self._backend


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
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int = 25  # -1 = use the model configuration's steps
    cfg: float = 7.0  # -1 = use the model configuration's CFG
    seed: int = -1
    batch_size: int = 1
    sampler: str | None = None
    scheduler: str | None = None
    loras: list[tuple[str, float]] = Field(default_factory=list)
    # ordered list of style names (styles.csv), applied to the prompt /
    # negative prompt before generation
    styles: list[str] = Field(default_factory=list)
    model_id: str | None = None
    mode: str = "generate"  # "generate" | "inpaint_masked" | "inpaint_not_masked"
    denoise: float = Field(1.0, ge=0.0, le=1.0)  # denoising strength, inpaint only
    # soft inpainting (Forge builtin extension); null = off. The UI only
    # sends it for inpaint modes (the backend requires an init mask anyway)
    soft_inpainting: SoftInpainting | None = None
    # self attention guidance (Forge builtin extension); null = off.
    # Works in all modes; needs cfg > 1 (the SDXL backend skips it, like
    # Forge, when there is no unconditional pass)
    sag: SelfAttentionGuidance | None = None
    # context padding around the masked area (canvas px per side);
    # "inpaint_masked" only, capped at the mask's largest extent
    mask_padding: int = Field(default=INPAINT_PADDING, ge=0, le=2048)


class ThumbBody(BaseModel):
    job: int
    index: int


class SaveBody(BaseModel):
    prompt: str


# Max brush slider value in px. Keep in sync with BRUSH_SIZE_MAX in
# static/app.js (covered by test_stroke_bad_values_422).
MAX_BRUSH_SIZE = 300.0
# The client zoom range is 1/512..64 (see viewZoomAt in static/canvas.js).
# In "screen" mode the slider value is in screen px and is sent as
# size / zoom, so canvas-px stamps span 1/64 .. 300*64 px.
MAX_ZOOM = 64.0
MIN_BRUSH_SIZE = 1.0 / MAX_ZOOM
MAX_BRUSH_SIZE_ZOOMED = MAX_BRUSH_SIZE * MAX_ZOOM


class StrokeBody(BaseModel):
    points: list[list[float]] = Field(..., min_length=1)  # canvas-pixel coords
    size: float = Field(10.0, ge=MIN_BRUSH_SIZE, le=MAX_BRUSH_SIZE_ZOOMED)
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    brush: str = "circle"
    erase: bool = False  # mask tool: shift-draw
    color: str = "#ffffff"  # hex, image tool only
    target: str = "image"  # "image" | "mask"

    def hexcolor(self) -> tuple[int, int, int]:
        s = self.color.lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
            raise ValueError(f"bad color: {self.color!r}")
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


class CloneBody(BaseModel):
    # A clone stroke: for each stamp centre C, copy the brush-shaped region
    # sampled at C + offset onto C (see Canvas.clone). offset is in
    # canvas px, integer (the client rounds it so the live preview and the
    # commit agree).
    points: list[list[float]] = Field(..., min_length=1)  # canvas-pixel coords
    offset: list[float] = Field(..., min_length=2, max_length=2)
    size: float = Field(10.0, ge=MIN_BRUSH_SIZE, le=MAX_BRUSH_SIZE_ZOOMED)
    opacity: float = Field(1.0, ge=0.0, le=1.0)
    brush: str = "circle"


class MaskOpBody(BaseModel):
    op: str  # "dilate" | "erode" | "clear"
    px: int = Field(4, ge=0, le=100)


class SegmentationMaskBody(BaseModel):
    model: str  # file name in models/segmentation/
    classes: list[str] = Field(default_factory=list)  # class labels to union


class UndoRedoBody(BaseModel):
    target: str = "draw"  # "mask" or "draw"


class SelectBody(BaseModel):
    id: str


def create_app(
    config: Config | None = None,
    backend: ModelBackend | None = None,
    segmentation: SegmentationService | None = None,
) -> FastAPI:
    config = config or load_config()
    ensure_dirs(config)

    hub = EventHub()
    canvas = Canvas(brushes_dir=config.brushes_dir)
    if segmentation is None:
        # models/segmentation/ may be empty; mediapipe itself is imported
        # lazily on first use (see segmentation.py)
        segmentation = SegmentationService(config.segmentation_models_dir)

    if backend is not None:  # test injection (e.g. MockBackend)
        registry = _MockRegistry(backend)
    else:
        from .gpus import select_gpu

        # Resolve the generation GPU up front (ini primary-gpu, or auto:
        # most free VRAM among non-ignored GPUs); the registry re-checks.
        registry = ModelRegistry(config, emit=hub.emit,
                                 gpu=select_gpu(config.primary_gpu,
                                                config.ignore_gpus))
    queue = GenerationQueue(registry=registry, canvas=canvas, emit=hub.emit,
                            wildcards_dir=config.wildcards_dir)
    queue.start()

    app = FastAPI(title="imagagen", lifespan=_lifespan(segmentation))
    app.state.config = config
    app.state.hub = hub
    app.state.canvas = canvas
    app.state.queue = queue
    app.state.registry = registry
    app.state.ui_state = UiStateStore(config.root / "ui-state.json")
    # model configurations: files in model-configs/ + in-memory defaults
    # for unconfigured models (Phase 4)
    app.state.config_store = ConfigStore(config, lambda: registry.list())
    app.state.preset_store = PresetStore(config.presets_dir)
    app.state.style_store = StyleStore(config.root / "styles.csv")
    app.state.segmentation = segmentation
    app.state.last_config_id: str | None = None

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
        return {
            "status": "ok",
            "root": str(config.root),
            "gpu": selected,
            "gpus": gpu_info(config.ignore_gpus, selected),
        }

    # --- models & model configurations (Phase 4) ---
    # The model selector lists *configurations*, not raw checkpoints:
    # every file config in model-configs/ plus one in-memory default per
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
        return {"models": [e.to_dict() for e in entries], "current": current}

    @app.get("/api/samplers")
    def list_samplers() -> dict:
        from .sdxl_backend import SAMPLERS, SCHEDULERS

        return {"samplers": list(SAMPLERS), "schedulers": list(SCHEDULERS)}

    @app.get("/api/loras")
    def loras() -> dict:
        return {"loras": list_loras(config), "settings": all_settings(config)}

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

        The selected styles are applied first, exactly as ``/api/generate``
        does, so the count reflects the full prompt the model will actually
        see (a style can push a short raw prompt past one CLIP window).

        ``tokens`` is None when the tokenizer could not be loaded; the UI then
        hides the counter. ``windows`` is how many 75-token CLIP windows the
        prompt occupies (the long-prompt chunked-encoding window size).
        """
        from .tokenize import count_prompt_tokens, prompt_windows

        prompt, _ = apply_styles(
            body.get("prompt", ""),
            "",
            body.get("styles", []),
            app.state.style_store.get,
        )
        tokens = count_prompt_tokens(prompt)
        if tokens is None:
            return {"tokens": None, "windows": 0}
        return {"tokens": tokens, "windows": prompt_windows(tokens)}

    @app.post("/api/models/select")
    def select_model(body: dict) -> dict:  # sync: runs in the threadpool
        try:
            entry = app.state.config_store.resolve(body.get("id", ""))
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        registry.backend_for(entry.model_id)  # blocks: loading is synchronous
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
    def preset_save(p: Preset) -> dict:
        try:
            id_ = app.state.preset_store.save(p)
        except (PresetError, ValidationError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True, "id": id_}

    @app.post("/api/presets/delete")
    def preset_delete(body: dict) -> dict:
        name = body.get("name", "")
        if not app.state.preset_store.delete(name):
            return JSONResponse({"error": f"unknown preset: {name!r}"}, status_code=404)
        return {"ok": True}

    # --- styles (styles.csv + in-memory "Session" temporaries) ---
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
        # model_id here is a *configuration* id (Phase 4): files in
        # model-configs/ or the in-memory default of an unconfigured model.
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
            sampler, scheduler, cfg, rsteps, loras = resolve_settings(
                entry,
                body.mode,
                body.cfg,
                body.sampler,
                body.scheduler,
                body.loras,
                body.steps,
            )
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        model_id = entry.model_id
        # Styles are applied right before the prompt reaches the model
        # (in list order; {prompt} replaced with the prompt so far).
        # The raw (unapplied) prompt and the style *contents* are kept for
        # the image metadata: the PNG must round-trip as "same prompt +
        # same style list", not as the pre-applied text.
        prompt, negative_prompt = apply_styles(
            body.prompt, body.negative_prompt, body.styles,
            app.state.style_store.get,
        )
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
            }
            for name in body.styles
            if app.state.style_store.get(name) is not None
        ]
        req = GenerateRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=body.width,
            height=body.height,
            # rsteps is None only when the user used the -1 sentinel and
            # the configuration has no override -> built-in default
            steps=rsteps if rsteps is not None else 25,
            cfg=cfg if cfg is not None else DEFAULT_CFG,
            seed=body.seed,
            sampler=sampler,
            scheduler=scheduler,
            loras=loras,
            soft_inpainting=body.soft_inpainting,
            sag=body.sag,
        )
        try:
            # inpaint modes snapshot the canvas and populate init image/mask;
            # "generate" leaves the request untouched
            plan = with_inpaint(req, body.mode, canvas, body.denoise,
                                padding=body.mask_padding)
        except GenerationError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        job = queue.submit(model_id, req, body.batch_size, plan=plan,
                           meta_prompt=body.prompt,
                           meta_negative=body.negative_prompt,
                           meta_styles=meta_styles)
        return {"job_id": job.id}

    @app.post("/api/interrupt")
    def interrupt(body: dict) -> dict:
        job_id = int(body.get("job", -1))
        if job_id >= 0:
            queue.interrupt(job_id)
        else:
            for jid in list(queue.jobs()):
                queue.interrupt(jid)
        return {"ok": True}

    @app.get("/api/thumb/{job_id}/source.png")
    def thumb_source(job_id: int) -> Response:
        # the source image (canvas snapshot at submit time; inpaint
        # always, generate on a non-empty canvas); registered before the
        # {index}.png route so "source" is not parsed as an index
        job = queue.job(job_id)
        data = job.source_png if job else None
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    @app.get("/api/thumb/{job_id}/{index}.png")
    def thumb(job_id: int, index: int) -> Response:
        job = queue.job(job_id)
        data = job.results.get(index) if job else None
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
        return {"state": state.model_dump()}

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
        if config.save_ui_state:
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

    @app.get("/api/brushes")
    def list_brushes() -> dict:
        return {"brushes": canvas.list_brushes()}

    def canvas_flags() -> dict:
        return {
            "undo_mask": canvas.can_undo_mask(),
            "redo_mask": canvas.can_redo_mask(),
            "undo_draw": canvas.can_undo_draw(),
            "redo_draw": canvas.can_redo_draw(),
        }

    @app.post("/api/canvas/upload")
    async def upload_canvas(request: Request) -> dict:
        data = await request.body()
        try:
            canvas.import_image(data)
        except Exception:
            return JSONResponse({"error": "could not read image"}, status_code=400)
        return {"ok": True, "canvas": canvas.has_image(), **canvas_flags()}

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
        # writes an explicit style list, so this only runs for foreign
        # metadata: it selects the recovered styles and shows the original
        # (unapplied) prompts. Meta-data save logic is untouched.
        if meta is not None and "styles" not in meta and meta.get("prompt"):
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
                    f"{n}@{w:g}" for n, (_, w) in zip(resolved, loras)
                )
            found, prompt, negative = app.state.style_store.extract_styles(
                no_lora_prompt, meta.get("negative_prompt", "")
            )
            meta["prompt"] = prompt
            if found:
                meta["styles"] = [s.to_dict() for s in found]
                if "negative_prompt" in meta:
                    meta["negative_prompt"] = negative
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
        return {"ok": True, **canvas_flags()}

    @app.post("/api/canvas/select")
    def select_canvas_image(body: SelectBody) -> dict:
        try:
            canvas.set_current(body.id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True, **canvas_flags()}

    @app.post("/api/canvas/undo")
    def canvas_undo(body: UndoRedoBody) -> dict:
        if body.target not in ("mask", "draw"):
            return JSONResponse({"error": "unknown target"}, status_code=422)
        canvas.undo("mask" if body.target == "mask" else "image")
        return {"ok": True, **canvas_flags()}

    @app.post("/api/canvas/redo")
    def canvas_redo(body: UndoRedoBody) -> dict:
        if body.target not in ("mask", "draw"):
            return JSONResponse({"error": "unknown target"}, status_code=422)
        canvas.redo("mask" if body.target == "mask" else "image")
        return {"ok": True, **canvas_flags()}

    @app.post("/api/canvas/stroke")
    def canvas_stroke(body: StrokeBody) -> dict:
        try:
            color = body.hexcolor()
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        try:
            canvas.stroke(
                points=[(float(x), float(y)) for x, y in body.points],
                size=body.size,
                opacity=body.opacity,
                brush=body.brush,
                erase=body.erase,
                color=color,
                target=body.target,
            )
        except (ValueError, KeyError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True, **canvas_flags()}

    @app.post("/api/canvas/clone")
    def canvas_clone(body: CloneBody) -> dict:
        try:
            canvas.clone(
                points=[(float(x), float(y)) for x, y in body.points],
                offset=(body.offset[0], body.offset[1]),
                size=body.size,
                opacity=body.opacity,
                brush=body.brush,
            )
        except (ValueError, KeyError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return {"ok": True, **canvas_flags()}

    @app.post("/api/canvas/mask")
    def canvas_mask_op(body: MaskOpBody) -> dict:
        if body.op not in ("dilate", "erode", "clear"):
            return JSONResponse({"error": "unknown op"}, status_code=422)
        if not canvas.has_image():
            return JSONResponse({"error": "canvas is empty"}, status_code=400)
        changed = {
            "dilate": lambda: canvas.mask_dilate(body.px),
            "erode": lambda: canvas.mask_erode(body.px),
            "clear": lambda: canvas.mask_clear(),
        }[body.op]()
        return {"ok": True, "changed": changed, **canvas_flags()}

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
            seg = app.state.segmentation.generate(body.model, body.classes, image)
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
            **canvas_flags(),
        }

    # --- save ---
    @app.post("/api/save")
    def save(body: SaveBody) -> dict:
        data = canvas.to_png_bytes()
        if data is None:
            return {"error": "canvas is empty"}
        out = next_output_path(config.output_dir, body.prompt)
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
