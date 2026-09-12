"""FastAPI application factory + WebSocket event hub."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from .backends import DEFAULT_CFG, GenerateRequest, ModelBackend
from .canvas import Canvas
from .config import Config, ensure_dirs, load_config
from .generation import GenerationError, with_inpaint
from .model_configs import ConfigError, ConfigStore, ModelConfig, ModelConfigUpsert, resolve_settings
from .models import ModelRegistry, list_loras
from .paths import next_output_path
from .presets import Preset, PresetError, PresetStore
from .queue import GenerationQueue
from .uistate import UiState, UiStateStore

STATIC_DIR = Path(__file__).parent / "static"


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

class GenerateBody(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 7.0
    seed: int = -1
    batch_size: int = 1
    sampler: str | None = None
    scheduler: str | None = None
    loras: list[tuple[str, float]] = Field(default_factory=list)
    model_id: str | None = None
    mode: str = "generate"  # "generate" | "inpaint_masked" | "inpaint_not_masked"
    denoise: float = Field(1.0, ge=0.0, le=1.0)  # denoising strength, inpaint only


class ThumbBody(BaseModel):
    job: int
    index: int


class SaveBody(BaseModel):
    prompt: str


class StrokeBody(BaseModel):
    points: list[list[float]] = Field(..., min_length=1)  # canvas-pixel coords
    size: float = Field(10.0, gt=0.05, le=200)
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


class MaskOpBody(BaseModel):
    op: str  # "dilate" | "erode" | "clear"
    px: int = Field(4, ge=0, le=100)


def create_app(config: Config | None = None, backend: ModelBackend | None = None) -> FastAPI:
    config = config or load_config()
    ensure_dirs(config)

    hub = EventHub()
    canvas = Canvas(brushes_dir=config.brushes_dir)

    if backend is not None:  # test injection (e.g. MockBackend)
        registry = _MockRegistry(backend)
    else:
        from .gpus import select_gpu

        # Resolve the generation GPU up front (ini primary-gpu, or auto:
        # most free VRAM among non-ignored GPUs); the registry re-checks.
        registry = ModelRegistry(config, emit=hub.emit,
                                 gpu=select_gpu(config.primary_gpu,
                                                config.ignore_gpus))
    queue = GenerationQueue(registry=registry, canvas=canvas, emit=hub.emit)
    queue.start()

    app = FastAPI(title="imagagen")
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
        return {"loras": list_loras(config)}

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
        return {"presets": app.state.preset_store.names()}

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
            sampler, scheduler, cfg, loras = resolve_settings(
                entry,
                body.mode,
                body.cfg,
                body.sampler,
                body.scheduler,
                body.loras,
            )
        except ConfigError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        model_id = entry.model_id
        req = GenerateRequest(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            width=body.width,
            height=body.height,
            steps=body.steps,
            cfg=cfg if cfg is not None else DEFAULT_CFG,
            seed=body.seed,
            sampler=sampler,
            scheduler=scheduler,
            loras=loras,
        )
        try:
            # inpaint modes snapshot the canvas and populate init image/mask;
            # "generate" leaves the request untouched
            plan = with_inpaint(req, body.mode, canvas, body.denoise)
        except GenerationError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        job = queue.submit(model_id, req, body.batch_size, plan=plan)
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
        # the inpaint source image (canvas snapshot at submit time);
        # registered before the {index}.png route so "source" is not
        # parsed as an index
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
        return {"state": app.state.ui_state.load().model_dump()}

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

    @app.post("/api/canvas/upload")
    async def upload_canvas(request: Request) -> dict:
        data = await request.body()
        canvas.set_from_png_bytes(data)
        return {"ok": True, "canvas": canvas.has_image()}

    @app.post("/api/canvas/thumb")
    def set_canvas_from_thumb(body: ThumbBody) -> dict:
        job = queue.job(body.job)
        data = job.results.get(body.index) if job else None
        if data is None:
            return {"error": "no such thumbnail"}
        canvas.set_from_png_bytes(data)
        return {"ok": True}

    @app.post("/api/canvas/undo")
    def canvas_undo() -> dict:
        canvas.undo()
        return {"ok": True, "undo": canvas.can_undo(), "redo": canvas.can_redo()}

    @app.post("/api/canvas/redo")
    def canvas_redo() -> dict:
        canvas.redo()
        return {"ok": True, "undo": canvas.can_undo(), "redo": canvas.can_redo()}

    @app.post("/api/canvas/clear")
    def canvas_clear() -> dict:
        canvas.clear()
        return {"ok": True}

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
        return {"ok": True, "undo": canvas.can_undo(), "redo": canvas.can_redo()}

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
        return {"ok": True, "changed": changed,
                "undo": canvas.can_undo(), "redo": canvas.can_redo()}

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
