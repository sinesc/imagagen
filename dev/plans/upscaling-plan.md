# Upscaling support plan

Add a toolbar button (right of `Paste`) that opens an Upscale dialog: pick an upscaler
model + scale factor + tiling options, Ok starts upscaling the current canvas image into
a NEW canvas entity with its own thumbnail (crop-style flow), Cancel closes the dialog.

Semantics follow stable-diffusion-webui-forge (`scripts/postprocessing_upscale.py`,
`modules/upscaler.py`, `modules/upscaler_utils.py`, `modules/images.py:split_grid/
combine_grid`), model loading follows Forge/ComfyUI via the **spandrel** library.

## Decisions (fixed)

- **spandrel** for upscaler model loading (architecture auto-detection: ESRGAN,
  RealESRGAN, SwinIR, ScuNET, DAT, HAT, ...). Verified resolution in this env:
  `spandrel 0.4.2` pulls `torchvision 0.28.0+cu129` (matches torch 2.13.0+cu129, torch
  untouched) + `einops`. Add to `pyproject.toml` dependencies: `spandrel>=0.4.2` and
  `torchvision` with `torchvision = { index = "pytorch-cu129" }` in `[tool.uv.sources]`.
- Upscaler models directory: **`models/upscalers/`** (recursive scan of `.pth`/`.pt`,
  same `_walk_files` rule as the model registry; id = extensionless relative posix
  path). Built-in pseudo-upscalers **`Lanczos`** and **`Nearest`** (pure PIL, always
  available, ids taken verbatim from Forge; a model file with the same display name is
  never matched because built-in ids resolve first).
- Dialog options (Forge parity for what we keep): upscaler combobox, **scale by**
  slider 1.0–8.0 step 0.25 (default 4), **max side length** number (0 = no limit,
  Forge's `limit_size_by_one_dimention` semantics), **tile size** (0–512, default 192,
  0 = no tiling) and **tile overlap** (0–48, default 8). Scale auto-sets from the
  selected upscaler's name (Forge regex `(\d)[xX]|[xX](\d)` -> scale N, its
  `set_scale_by_when_changing_upscaler` behaviour, always on here).
  NOT in this cut (future work): Upscaler 2 + visibility blend, "Scale to WxH + crop",
  sd_upscale (img2img-based), interrupting a running upscale.
- Execution: **synchronous POST** (like crop), NOT a queue job. Runs on the primary
  GPU while holding that GPU's registry lock, so it serializes against generation
  (the queue worker just blocks on the lock; a Generate submit during an upscale
  queues normally). A process-wide `threading.Lock` in UpscalerManager rejects
  concurrent upscales with HTTP 409. No interrupt in this cut (the client's Interrupt
  button is job-based and untouched).
- Result: a new canvas entity `upscale:{n}`, pinned exactly like a crop (survives
  thumbnail switches, released by the next generation's `release_pins()`), set current
  immediately, selected thumbnail immediately (the pre-upscale image gets a thumbnail
  if it doesn't already have one). Its per-entity mask is the source mask transformed
  to the new size via `canvas._transform_mask` (exact fit, so effectively nearest-style
  proportional scaling). PNG metadata: the SOURCE entity's text chunks are copied
  verbatim (prompt/seed/model round-trip and /api/save naming keep working); no new
  parameters fields (Forge records upscale info in info text only, not in the PNG).

## Backend

### New module `src/imagagen/upscalers.py`

- `UpscalerManager(registry, upscalers_dir)`:
  - `list_upscalers() -> list[dict]`: built-ins first
    (`[{"id": "Lanczos", "name": "Lanczos", "builtin": True}, ...]`) then
    `{"id": rel_id, "name": rel_id, "builtin": False}` for every `.pth`/`.pt` under
    `models/upscalers/` (use `modelregistry._walk_files` — py3.13 `rglob` skips
    symlinked dirs).
  - `run(arr, upscaler_id, scale, max_side, tile, tile_overlap, progress) -> np.ndarray`
    where `arr` is the canvas RGB uint8 HxWx3. Guards: upscale-busy lock (409 upstream),
    absolute cap — error if either output side would exceed 16384 px.
  - Model cache `{id: spandrel descriptor}` kept **on CPU**; per run the descriptor is
    moved to `torch.device(f"cuda:{registry.primary}")` (CPU when `registry.primary is
    None`, e.g. the mock/test server) and back to CPU afterwards. Model creation
    (`ModelLoader().load_from_path(path, prefer_half=False)`) runs under
    `backends._PARAM_INIT_LOCK` (parameter-creation serialization, same rule as
    backends).
- Size loop ported from `Upscaler.upscale` verbatim: `dest_w = int(w*scale)//8*8`,
  apply the model up to 3 passes while `img < dest`, final LANCZOS resize to
  `(dest_w, dest_h)` (this makes a 4x model at scale 2 shrink back to exactly 2x, and
  scale 1 still runs one model pass — same as Forge). `max_side` clamps via Forge's
  `limit_size_by_one_dimention` (recomputes effective scale, like the
  `upscale_mode` switch in `postprocessing_upscale.py`).
- Model pass = port of `upscaler_utils.upscale_with_model` +
  `pil_image_to_torch_bgr`/`torch_bgr_to_pil_image` + `images.split_grid`/`combine_grid`
  (feathered overlap masks). Work on numpy/PIL as Forge does; `tile<=0` runs one
  untiled pass. `progress(done, total_tiles)` -> WS events (below); `Lanczos`/`Nearest`
  upscalers are one `PIL.Image.resize` and emit no progress.

### `models.py` (ModelRegistry)

Add a tiny public accessor:

```python
def gpu_lock(self, gpu: int) -> threading.RLock:
    """Per-GPU lock so external GPU work (upscaling) serializes against
    generation on that GPU."""
    return self._get_gpu_state(gpu).lock
```

(`_get_gpu_state` creates the state entry on demand; only ever called with a
valid CUDA index — the caller skips the lock when `registry.primary is None`.)

### `config.py` / `paths`

- `Config.upscalers_dir` property = `models/upscalers`; `ensure_dirs` creates it.
- Add a line to `models/readme-folder-names.txt` describing `upscalers/` (.pth RealESRGAN
  etc.). No new ini keys (tile size/overlap live in the dialog).

### `canvas.py`

New method mirroring `crop()`:

```python
def import_upscaled(self, png: bytes, info: dict, mask: np.ndarray) -> str:
    """Upscale result: new pinned entity `upscale:{n}`, set current."""
```

uses a new `_upscale_counter`, builds the `CanvasImage(png, info, mask)`,
`_pin_locked(eid)`, `_enter_entity_locked(eid)`, returns the id. (The PREVIOUS entity
is pinned by the caller right after, same as `crop()` pins prev + crop.)

### `app.py`

- Construct `UpscalerManager(registry, config.upscalers_dir)` next to the queue.
- `GET /api/upscalers` -> `{"upscalers": manager.list_upscalers()}`.
- `POST /api/upscale`, pydantic body `UpscaleBody`:
  `upscaler: str`, `scale: float = Field(4.0, ge=1.0, le=8.0)`,
  `max_side_length: int = Field(0, ge=0)`, `tile: int = Field(192, ge=0, le=512)`,
  `tile_overlap: int = Field(8, ge=0, le=48)` — `def` endpoint (threadpool, sync).
  Flow: 400 when canvas empty (`not canvas.has_image()`); read `canvas.image` +
  `canvas.mask` + source info (`canvas.entity_info(canvas.current_id())`); emit
  `{"type":"phase","message":"Upscaling @ WxH"}` (final dest size) then per-tile
  `{"type":"progress","fraction":done/total,"message":"tile i/n"}` via `hub.emit`
  (exactly how the segmentation endpoint already emits job-less events; the
  existing WS client branches for phase/progress ignore `job`, so no client
  change is needed for progress); take `gpu_lock(primary)` (skip on CPU-only
  server) around `manager.run(...)`; on model-load failure return 400 with the message
  (also `log.exception`).
  Build the result PNG from the returned array and copy the source text chunks:

```python
img = Image.fromarray(out)
pnginfo = PngInfo()
for k, v in (src_info or {}).items():
    pnginfo.add_text(k, v)
buf = BytesIO(); img.save(buf, format="PNG", pnginfo=pnginfo)
```

  Mask for the new entity: `canvas._transform_mask` equivalent via a small canvas-side
  helper (or expose `transform_mask(mask, W, H)` public wrapper in canvas.py).
  Then `prev_id = canvas.current_id()`, `eid = canvas.import_upscaled(...)`,
  `canvas.pin(prev_id)` (crop-style; do it inside import path or right after, mirroring
  `crop()`'s pin handling). Return
  `{"ok": True, "prev_image_id", "upscale_image_id", "prev_url", "upscale_url",
  "width", "height"}` (urls = `/api/canvas/image.png?id=...`, same shape as
  `/api/canvas/crop`). Errors mid-run (OOM etc.) return 400/500 with the message; no
  entity is created.

## Frontend (`static/index.html`, `app.js`, `style.css`)

- Toolbar button directly after `#tool-paste` (before the following `<span class="sep">`):
  `<button id="btn-upscale" class="mini" title="..."><u>U</u>pscale</button>` (the
  underlined letter is the hotkey, cf. Past<u>e</u>) with hotkey
  `U` (add to the plain-letter chain: `else if ((e.key === "u" || e.key === "U") && $("dialog").hidden) $("btn-upscale").onclick();`).
  It is a dialog launcher (like `#seg-run`), NOT a canvas tool — do not add it to
  `TOOLS`/uistate ToolState.
- `openUpscaleDialog()` (app.js, modeled on `openSegDialog`): fetch `GET /api/upscalers`
  on open; body rows (labels follow existing dialog style, capitalized; use
  `makeCombobox` for the upscaler list — options `{value, text, group}` with groups
  "Built-in" and "Models"):
  - Upscaler combobox; on change, auto-set the scale slider from the name regex
    `/(\d)[xX]|[xX](\d)/` when it matches.
  - Scale slider 1–8 step 0.25 default 4 with `<b>` readout (brush-size-slab pattern).
  - "Max side length" number input default 0 (0 = no limit); a live
    output-dimensions readout `<b>` next to/below it computes the Forge size rule:
    `dw = clamp(int(w*scale)//8*8, max_side)`-style preview (`Math.floor(w*scale/8)*8`
    then longest-side cap, exactly the final size the server will produce).
  - Advanced row: tile size number (0–512 default 192, title "0 = no tiling"),
    overlap number (0–48 default 8).
  `openDialog("Upscale", body, onOk, "Upscale")` -> Ok/Cancel pair (`openDialog`
  auto-shows Cancel when onOk is set).
- Ok handler: disable nothing globally — POST `/api/upscale` (await the response;
  progress bar fills via WS events; status line shows the phase message); on error
  `dialogError(...)` and `return false` to keep the dialog open; on success
  `addUpscaleThumbs(r)` + `await viewRefresh()`. `forceSync()` BEFORE the POST (the
  server upscales ITS copy, like crop/save). Note: this does NOT call
  `setGenerating(true)` (that is wired to job interrupt); instead re-enable/keep the
  dialog's Ok disabled while the request is in flight, and clear
  `progress-fill`/`progress-label` after the response returns.
- `addUpscaleThumbs(r)` mirrors `addCropThumbs`: push prev thumb
  (`{image_id, url, upscale: true}`) unless it already exists, push new thumb,
  `state.currentThumb = last`, `currentEntityId = r.upscale_image_id`,
  `renderThumbs()`. `renderThumbs()` title branch: `t.upscale ? \`upscaled image${dims}
  (click to restore)\`` (insert next to the `t.crop`/`t.upload` branches).
- Thumbnails: `upscale: true` thumbs are dropped at the next `generate()` exactly like
  crop thumbs (`state.thumbs.filter(t => t.pinned)` already keeps only pinned).

## Tests

- `tests/integration/test_upscale_api.py` (mock server, no GPU needed):
  - `/api/upscalers` lists `Lanczos` + `Nearest` and a fake model file seeded at
    `models/upscalers/realesrgan/RealESRGAN_x4plus.pth` (dummy file: list-only).
  - upscale Lanczos x2 on a 200x150 canvas upload -> response ids/urls,
    `/api/canvas/image.png?id=upscale:0` decodes to 400x300, entity is current,
    prev entity pinned (`/api/queue` "pinned"), source PNG info chunks survive.
  - `max_side_length=256` clamps 200x150 x4 to 256x192 (Forge size rule:
    `int(w*scale)//8*8`).
  - empty canvas -> 400; unknown upscaler id -> 400; `scale` outside 1..8 rejected
    (pydantic 422).
- `tests/ui/test_upscale_ui.py` (mock server): button sits right of `#tool-paste`;
  click opens the dialog with the upscaler combobox listing Lanczos; empty scale
  preview updates on slider change; Ok with Lanczos x2 adds a second thumbnail titled
  "upscaled image ... (click to restore)", canvas becomes 2x, `currentEntityId` is the
  upscale id; Cancel leaves the canvas untouched; U hotkey opens the dialog
  (and does NOT fire while the dialog is open).
- Run the touched test files, not the full suite; keep ruff clean.

## GPU verification (manual, dev/)

`dev/smoke_upscale.py` (follow the dev/ script conventions: insert src/ on sys.path):
download `RealESRGAN_x4plus.pth` (GitHub release, network is available) into a scratch
upscalers dir, load via UpscalerManager on GPU 1, upscale a 512x512 test photo 4x with
tile 192/8 and again with tile 0, print timings + peak VRAM (`torch.cuda.max_memory_allocated`),
check x2 and scale 2.5 output dims follow the `int(w*s)//8*8` rule. Also run one
through a live server with an SDXL model loaded (confirm the upscale and a following
generation both succeed on one GPU and VRAM never spikes).

## Implementation order

1. `uv add spandrel torchvision` + pyproject/uv.sources; verify `import spandrel`.
2. `upscalers.py` (manager, Forge size-rule port, tiling port) + `config.upscalers_dir`
   + `ensure_dirs` + `models.py:gpu_lock`.
3. `canvas.import_upscaled` + public `transform_mask` wrapper.
4. `app.py` endpoints + UpscaleBody.
5. Integration tests (Lanczos path is pure PIL -> fully offline/mockable).
6. Frontend (button, hotkey, dialog, thumbs) + UI tests.
7. GPU smoke with RealESRGAN + live-server multi-use check.
