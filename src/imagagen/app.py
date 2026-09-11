"""FastAPI application factory + WebSocket event hub."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .backends import GenerateRequest, ModelBackend
from .canvas import Canvas
from .config import Config, ensure_dirs, load_config
from .models import ModelRegistry
from .paths import next_output_path
from .queue import GenerationQueue

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


class ThumbBody(BaseModel):
    job: int
    index: int


class SaveBody(BaseModel):
    prompt: str


def create_app(config: Config | None = None, backend: ModelBackend | None = None) -> FastAPI:
    config = config or load_config()
    ensure_dirs(config)

    hub = EventHub()
    canvas = Canvas()

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

    # --- models ---
    @app.get("/api/models")
    def list_models() -> dict:
        return {"models": registry.list(), "current": registry.current_id()}

    @app.get("/api/samplers")
    def list_samplers() -> dict:
        from .sdxl_backend import SAMPLERS, SCHEDULERS

        return {"samplers": list(SAMPLERS), "schedulers": list(SCHEDULERS)}

    @app.post("/api/models/select")
    def select_model(body: dict) -> dict:  # sync: runs in the threadpool
        registry.backend_for(body.get("id"))  # blocks: loading is synchronous
        return {"current": registry.current_id()}

    # --- generation ---
    @app.post("/api/generate")
    def generate(body: GenerateBody) -> dict:
        model_id = body.model_id or registry.current_id()
        if model_id is None and getattr(registry, "models", None):
            model_id = registry.models[0].id
        if model_id is None:
            return {"error": "no model available"}
        req = GenerateRequest(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            width=body.width,
            height=body.height,
            steps=body.steps,
            cfg=body.cfg,
            seed=body.seed,
            sampler=body.sampler,
            scheduler=body.scheduler,
            loras=body.loras,
        )
        job = queue.submit(model_id, req, body.batch_size)
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

    @app.get("/api/thumb/{job_id}/{index}.png")
    def thumb(job_id: int, index: int) -> Response:
        job = queue.job(job_id)
        data = job.results.get(index) if job else None
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

    # --- canvas ---
    @app.get("/api/canvas.png")
    def get_canvas() -> Response:
        data = canvas.to_png_bytes()
        if data is None:
            return Response(status_code=404)
        return Response(content=data, media_type="image/png")

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
