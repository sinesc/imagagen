# Implementation Plan — Imagagen (AI image generator)

Scope source: `concept.md`. This plan targets a phased build that reaches a minimal
working generator first, then layers on canvas editing, inpainting, batch/queueing,
configuration, and the plugin ecosystem.

## Architecture overview

```
imagagen/
├── concept.md
├── PLAN.md
├── pyproject.toml          # uv-managed project, Python 3.13
├── src/
│   └── imagagen/
│       ├── __init__.py
│       ├── app.py                  # FastAPI app factory, static file serving, WS
│       ├── config.py               # project root, subdirectory paths, app settings
│       ├── paths.py                # models/, loras/, brushes/, presets/, output/,
│       │                           #   model-configs/ discovery & numbering
│       ├── server.py               # uvicorn entrypoint (`python -m imagagen`)
│       ├── queue.py                # generation queue: workers per GPU, dynamic
│       │                           #   assignment, interrupt, progress events
│       ├── gpus.py                 # CUDA device discovery, VRAM bookkeeping
│       ├── models.py               # model registry: detection, default configs,
│       │                           #   config load/save (model-configs/),
│       │                           #   RAM residency (load/unload/swap)
│       ├── generation.py           # the generation pipeline: settings resolution
│       │                           #   (model defaults vs. user overrides),
│       │                           #   txt2img / inpaint-masked / inpaint-not-masked
│       │                           #   geometry (bounding boxes, aspect-preserving
│       │                           #   scale+center), batch seeding, metadata write
│       ├── canvas.py               # pure-Python canvas operations (PIL/numpy):
│       │                           #   brush stamps, mask draw/erase,
│       │                           #   mask dilate/erode, clone, smudge,
│       │                           #   undo/redo stacks (server-authoritative
│       │                           #   canvas image as single source of truth)
│       ├── metadata.py             # PNG text-chunk read/write in the "parameters"
│       │                           #   format Forge/A1111 emit (via PngInfoPlugin)
│       ├── import_export.py        # image import: decode image + settings
│       ├── plugins/
│       │   ├── __init__.py         # plugin registry + entrypoint scanning
│       │   ├── base.py             # ToolPlugin, SamplerPlugin, SchedulerPlugin,
│       │   │                       #   ModelBackendPlugin interfaces
│       │   ├── sdxl_backend.py     # SDXL model backend (diffusers-based), LoRA
│       │   ├── samplers.py         # Euler a, DPM++ 2M, DPM++ 2M SDE
│       │   ├── schedulers.py       # Karras, Polyexponential
│       │   ├── tools/
│       │   │   ├── clone.py
│       │   │   └── smudge.py
│       │   └── segmentation/
│       │       └── person_mask.py  # port of a-person-mask-generator
│       └── static/                 # web UI (served by FastAPI)
│           ├── index.html
│           ├── app.js              # UI wiring: panels, dialogs, WS client
│           ├── canvas.js           # canvas renderer + input (brush/mask/clone/smudge,
│           │                       #   zoom/pan, brush png alpha loading)
│           ├── tools.js            # tool state, per-tool remembered brush/opacity
│           └── style.css
├── tests/
│   ├── unit/                       # canvas ops, paths/numbering, mask geometry,
│   │                               #   scheduler sigma shapes, metadata roundtrip
│   ├── mock_backend.py             # fake "model" producing deterministic images;
│   │                               #   lets us test queue/inpaint/import without GPU
│   └── integration/                # end-to-end API tests using mock backend
├── models/                         # (user data, gitignored) sdxl/, segmentation/
├── loras/                          # sdxl/
├── brushes/                        # greyscale pngs
├── presets/                        # json
├── model-configs/                  # json, one file per named configuration
└── output/
```

Key architectural decisions:

- **Server-authoritative canvas.** The canvas image + mask live server-side (PIL/
  numpy). The browser renders it and sends *strokes* (point lists + brush params),
  not raw pixels. This makes undo/redo, plugin tools (clone/smudge), and generation
  overpaint semantics simple and consistent. A full-canvas snapshot goes into the
  undo stack at stroke end / generation / mask clear (bounded memory: keep
  snapshots, they're just PNG-compressible arrays — cap the stack, e.g. 200).
- **Generation = server job.** The UI never touches the model. The generate button
  POSTs a job; progress + results stream back over WebSocket; images are pushed to
  the browser as thumbnails as each completes (completion order, per spec).
- **Plugins as pure Python classes** discovered by a simple registry (no
  entrypoint machinery in v1 — a fixed package scan + explicit registration is
  enough; the *interfaces* are designed to be extensible).
- **Inference via `diffusers`**, not by forking Forge. Forge's code is a heavily
  patched A1111; we only *reference* it for: sampler/scheduler math
  (`modules/sd_samplers_kdiffusion.py`), mask helpers (`modules/masking.py`),
  PNG info-text format, and the person-mask extension. `k_diffusion` (standalone
  pip package, upstream of Forge's samplers) implements the actual sampler loops,
  so "port from Forge" mostly means wrapping `k_diffusion.sampling` with the right
  sigma solvers + scheduler.

## Directory & data conventions

- Model discovery: any `.safetensors` under `models/<model-type>/` (type = subdir
  name, e.g. `sdxl`) not covered by a config in `model-configs/` gets an in-memory
  default config. Config file format (JSON):
  ```json
  {
    "name": "My SDXL Model + DMD2 Lora",
    "model": "sdxl/whatever.safetensors",
    "t2i":   {"sampler": null, "scheduler": null, "cfg": null,
              "loras": [{"name": "sdxl/dmd2", "weight": 0.8}]},
    "i2i":   {"sampler": null, "scheduler": null, "cfg": null, "loras": []}
  }
  ```
  `null` = no override (sampler/scheduler/CFG) → falls back to the "use model
  default" behavior; LoRA list is *always* prepended to the user's list.
- Presets: `presets/<name>.json`:
  ```json
  {"name": "Big square", "width": 1024, "height": 1024,
   "controls": {"width": true, "height": true, "steps": false, ...}}
  ```
- Output naming: regex `^(\d+)-` over `output/` filenames, max + 1, else 1;
  prompt sanitized to `[a-z0-9_-]`, empty → "image".

## Generation geometry (per concept.md)

- **txt2img ("generate")**: create at the configured image size.
- **inpaint masked**: bbox = minimal rectangle enclosing mask alpha>0, then the
  *aspect-preserving* bounding: enlarge the bbox on the narrower axis (or both,
  preserving the target aspect ratio) so it fits the requested W:H, then add a
  *context padding* ring (`INPAINT_PADDING` = 256px each side, capped at the
  mask's largest extent — the A1111/Forge convention) so the model sees the
  real surroundings and matches their colour/tone (without it the model
  invents its own background and a visible halo outlines the mask). Generate
  at image size, downscale the result to the (padded) bbox, blend with mask
  alpha (the padding ring is discarded, so the canvas outside the mask stays
  byte-identical).
- **inpaint not masked**: scale current canvas AR-preserving so it "touches" the
  inside of the target size; paste centered onto empty target canvas; inpaint the
  whole target with mask areas protected (mask is scaled the same way as the image:
  e.g. 100×100 → 250×200 scales 2× and offsets 25px right); canvas size updates to
  the target size; mask is retained and scaled identically.
- Batch: seeds `seed+i` for i in 0..n-1; seed -1 → random base at job start.
- Interrupt: cancel not-yet-started batch items; running items finish on the next
  step check (check flag per denoising step) and are discarded, never pushed to
  thumbnails. Already-completed images stay in the thumbnail strip.
- Canvas editable during generation; a generation commit snapshots the canvas into
  the undo stack *before* overpainting, so pre-generation edits remain restorable.

## Phases

### Phase 0 — Scaffolding (foundation)
- `pyproject.toml` (uv), package skeleton, `config.py`/`paths.py`, empty static dir,
  `python -m imagagen` runs FastAPI on :8080 serving a placeholder page.
- **Tooling from day one**:
  - `--mock` backend flag / env var: `tests/mock_backend.py` exposes the same
    `ModelBackend` interface, producing deterministic noise/gradients; all
    queue/inpaint/import/UI flows are testable CPU-only (fast, deterministic,
    no VRAM needed for the CI-style suite; the workspace does have GPUs — see
    "Testing strategy" — and real-GPU verification lives in `devtools/`).
  - pytest + httpx AsyncClient integration tests against the API.
  - ruff + a smoke-test script hitting the main endpoints.
- Deliverable: server runs, directory layout created on first start, tests green.

### Phase 1 — MVP: minimal UI + basic generation
- `gpus.py`: enumerate CUDA devices via `torch.cuda`; `queue.py`: one worker per
  GPU, job dequeued to any free worker (dynamic, not modulo); progress callback.
- `models.py` (v1): detect SDXL `.safetensors` under `models/sdxl/`, load via
  diffusers `StableDiffusionXLPipeline` (single active model, `torch.inference_mode`).
- `generation.py` (v1): txt2img only. One hardcoded sampler path (Euler a + Karras
  via `k_diffusion`) so the plugin layer isn't a prerequisite for the MVP.
- `canvas.py` (v1): server holds one PIL image; endpoints `GET /canvas` (PNG),
  `POST /canvas/set`.
- Minimal UI: prompt + negative prompt, width/height, steps, CFG, seed, batch size,
  **Generate/Interrupt** button, canvas `<canvas>` displaying the server image,
  thumbnail strip (click → canvas), **Save** button, settings written into the
  generated PNG via `metadata.py`.
- Undo/redo endpoints (single image snapshot).
- Deliverable: type a prompt → get batched images → pick one → save to `output/`
  with correct numbering. (Verifiable end-to-end with the mock backend; real
  inference runs on this workspace's GPUs — see "Testing strategy".)

### Phase 2 — Unified canvas editing ✅ (commit ad5cb2d)
- ✅ Brush engine: default 10px circle; custom brushes loaded from `brushes/*.png`
  (greyscale luminance → alpha); log-scale size slider 0.1–100.0; opacity slider.
- ✅ Stroke protocol: pointer events → server stamping (Poisson-disk spacing along
  the segment, standard image-editor behavior); per-tool remembered brush +
  opacity (client-side state in tools.js, persisted in localStorage, sent with strokes).
- ✅ **Mask tool**: separate mask buffer; grey overlay compositing in the renderer;
  shift = erase; dilate/erode endpoint (scipy.ndimage binary_dilation/erosion with
  a disk structuring element sized to the px count); clear mask.
- ✅ Zoom (wheel, cursor-anchored) + middle-button pan (pure client-side transform).
- ✅ Undo/redo v2: mask states recorded too; snapshot on stroke end only.
- ✅ Save flow: filename from prompt + `output/` numbering (already existed from
  Phase 1 as `paths.next_output_path` + `sanitize_prompt`, unit-tested in
  tests/unit/test_paths.py).
- ✅ Deliverable: fully editable canvas + mask with unlimited undo/redo, tested
  server-side with mock backend.

Implementation notes:
- The mask is retained across same-size image replacements (e.g. the first
  generated image auto-set into the canvas, or a thumb selection) and reset on
  a size change — per spec "the mask is retained across image generations
  unless the user clears it".
- The optimistic stroke preview mirrors `Canvas.stamp_positions` on the client;
  the server remains authoritative on commit.
- `GET /api/mask.png` is 404 while no canvas is loaded; the UI treats that as
  an empty mask.
- The UI is browser-tested with Playwright: `tests/ui/test_canvas_ui.py`
  (Chromium, mock-backend server in a temp root). These caught two real wiring
  bugs the API-only tests couldn't: the brush tool sent `target: "brush"`
  (server expects `"image"`) and the no-body `api()` helper fell back to GET
  while `/api/canvas/undo`|`redo` are POST-only. Playwright is a dev dependency;
  its Chromium is installed separately (`playwright install --with-deps chromium`).

### Phase 3 — Inpainting & generation modes ✅ (commit 719d341)
- ✅ `generation.py`: inpaint masked + inpaint not masked incl. all bbox/scale/
  center math (pure functions, 15 unit tests); mask protection for not-masked;
  canvas resize + mask rescale on not-masked (100→250×200 case in tests).
- ✅ Backend contract: the backend always generates at the requested
  width × height; `InpaintPlan` (snapshotted at submit time) maps the raw
  output to canvas state afterwards -- "inpaint masked" downscales to the
  aspect-preserving bbox and alpha-blends (canvas size + mask unchanged),
  "inpaint not masked" is the whole new canvas (edge-replicated borders,
  inverted init mask protects the user mask; canvas + mask rescaled). The
  mask passed to the backend marks the *inpaint* region (255 = regenerate),
  so "inpaint not masked" inverts it.
- ✅ SDXL img2img/inpaint in the vendored k-diffusion loop (A1111/Forge
  convention): init latent via deterministic VAE encode, sigma schedule
  truncated at the denoising strength (`t_enc = int(min(ds, 0.999)·steps)`),
  per-step re-noise + x0-lock of the protected region, final lock before
  decode. Mock backend emulates denoise via an init-image blend.
- ✅ UI: mode select (generate / inpaint masked / inpaint not masked) +
  denoise input, disabled unless inpainting; both persisted in ui-state.
  Server-side setup errors ("canvas is empty", "mask is empty") are
  surfaced in the status line (the API helper now extracts `error` from
  non-OK JSON bodies).
- ✅ Mask retained across generations: `Canvas.set_image_keep_mask` keeps
  the mask when the size is unchanged (thumb clicks via the existing
  `set_from_png_bytes` already did).
- ✅ Thumbnails show the post-processed canvas state, so clicking one
  restores exactly what auto-set would have shown.
- Thumbnails during generation: WS `image_done` events arrive in completion
  order; first done image auto-set to canvas (unless user changed canvas
  meanwhile → snapshot for undo, per spec); ←/→ keys switch variant.
- Interrupt semantics (step-level flag).
- Deliverable: inpaint/outpaint/img2img all working, tested with mock backend.

### Phase 4 — Configuration layer ✅ (commit a3a4359)
- ✅ Model configs: `model-configs/*.json` (one file per named configuration,
  filename = sanitized name; sections `t2i`/`i2i`, each with sampler, scheduler,
  CFG, LoRA list; `null` = no override). `ConfigStore` (model_configs.py) merges
  the files with a read-only in-memory *default* entry for every model no file
  covers, so the selector is never empty. Create = copy of the selected
  configuration (concept.md), via a small name dialog; edit/delete via the config
  dialog. Default entries can't be deleted; name collisions (case-insensitive
  name or sanitized id) are rejected.
- ✅ Default-override resolution, server-side in `/api/generate`
  (`resolve_settings`, unit-tested): the user's setting wins *unless* it is the
  "use model default" sentinel — CFG `-1` (now a valid `GenerateBody.cfg` / ui-state
  value), sampler/scheduler `null` — in which case the configuration's value is
  used; `null` configuration values fall through to the backend built-in
  (`DEFAULT_CFG = 7.0` in backends.py). The configuration's LoRA list is not an
  override: it is always *prepended* to the user's LoRAs. `generate` resolves
  against the `t2i` section, both inpaint modes against `i2i`.
- ✅ LoRAs: discovery via `list_loras` (`loras/<model-type>/*.safetensors|.ckpt`,
  names relative to `loras/`), `GET /api/loras`; UI rows are a dropdown + weight
  (+/−, persisted in ui-state). The SDXL backend applies them with diffusers'
  `load_lora_weights` / `set_adapters` (PEFT `inject_adapter_in_model` under the
  hood — the plan's robust choice, and it also normalizes the common on-disk key
  layouts), one injected adapter per LoRA, per-LoRA weights via `set_adapters`;
  the active set is cached so unchanged generations don't re-inject. `peft>=0.10`
  added to the dependencies. Verified on GPU by `devtools/smoke_sdxl_lora.py`
  (real A1111/sgm-format LoRA: active at 1.0, inert at 0.0, deterministic cache
  hit, merge with a second adapter, clean unload). One diffusers-0.40 /
  transformers-5 skew had to be fixed: the A1111 converter emits
  `text_model.encoder...` te keys but transformers 5 flattened CLIPTextModel, so
  key→module matching came up empty and loading crashed — `SdxlBackend.
  _lora_state_dict` re-prefixes the te keys against the actual model layout.
- ✅ Presets: `presets/*.json` with the spec's per-setting `controls`
  checkboxes; CRUD + apply-on-select (only checked settings are written into the
  form, everything else untouched); last-applied preset + LoRA list
  persisted in ui-state. Manage dialog (new / edit / delete).
- ✅ UI state: `UiState` gained `loras` and `preset`; `cfg`'s floor moved to -1
  (the sentinel). The model selector now lists *configurations* (`id`, `name` —
  with "(default)" suffix for the in-memory entries —, `model_id`, `t2i`, `i2i`),
  and `model_id` in `/api/generate` is a configuration id.
- Implementation notes:
  - Delete endpoints are POST with a body (`/api/model-configs/delete`,
    `/api/presets/delete`) rather than path params: configuration ids may
    contain slashes (default entries are named after the model path, e.g.
    `sdxl/foo.safetensors`).
  - A configuration name sanitizing to an existing id is treated as an in-place
    update of that file (the two are indistinguishable anyway).
  - Resolution is tested end-to-end through the PNG `parameters` metadata the
    queue writes into the generated thumbnails (tests/integration/test_config_api.py)
    — cfg -1 → config value, LoRA merge order, explicit-user-wins, and the i2i
    section for inpaint modes.
  - The create-config flow uses an in-app dialog, not `prompt()` (Playwright
    browser tests cover it; native dialogs are a pain point for a local tool).
- ✅ Deliverable: all settings from the spec wired up, resolution order covered
  by unit + integration + browser tests.

### Phase 5 — Plugin system hardening
- Formalize `SamplerPlugin`/`SchedulerPlugin` interfaces; move Euler a,
  DPM++ 2M, DPM++ 2M SDE + Karras, Polyexponential in; dropdowns populated from
  the registry (with "Use model default" first).
- `ToolPlugin` interface; move **clone** (shift-click source, offset = current −
  start; implemented server-side on stroke) and **smudge** (weighted average drag)
  into plugins.
- **Segmentation**: port `a-person-mask-generator` (yolov8n-cls / person model +
  face-landmark refinement) into `plugins/segmentation/`; button + model dropdown
  over `models/segmentation/`; result written to the mask buffer (undoable).
- Deliverable: registry-driven UI; new sampler/scheduler/tool addable by dropping
  a module; person-mask generation working (CPU fallback OK for testing the
  plumbing with the mock image).

### Phase 6 — GPU/memory management & import
- Model residency: LRU-1-in-RAM — switching model unloads active to RAM (keep
  last), evicts previous RAM model to disk if needed; reload from RAM when
  switching back (torch CPU tensors, no disk round-trip); queue blocks generation
  during load/unload with a "loading model" WS status.
- **Import**: file picker + drag-drop; decodes image → canvas; parses PNG
  text chunks in Forge/A1111 `parameters` format → fills all settings; roundtrip
  test (save → import → identical settings).
- Final pass: `README.md` (setup, model download, CUDA_VISIBLE_DEVICES note),
  full test suite green, manual GPU test checklist.

## Testing strategy

**This workspace has two GPUs** (checked 2025-07: `nvidia-smi`) — index 0: RTX
5070 Ti 16GB, index 1: RTX 4060 Ti 16GB; both comfortably run SDXL. Servers and
devtools select one via the `primary-gpu` ini setting (or
`CUDA_VISIBLE_DEVICES`). Real-GPU verification is therefore part of normal
workflow here, not something deferred to "a GPU machine".

- **Mock backend** is still the workhorse for the automated suite: every
  phase's acceptance criteria are testable via the API + a headless
  browser-less client (httpx + WS) without CUDA — fast and deterministic.
  **But the mock never exercises the torch code paths**: the SDXL inpaint
  crash (3D tensor into `F.interpolate`) passed all 143 tests and only showed
  up in the UI. After any change to `sdxl_backend.py`'s sampling, run
  `devtools/smoke_sdxl_inpaint.py` (and ideally
  `devtools/e2e_inpaint_ui_flow.py`) on a GPU.
- Unit tests: output naming, prompt sanitization, scheduler sigma shapes (known
  values for Karras/Polyexponential at fixed step counts), mask bbox/enlarge math,
  AR-scale+center math (including the 100→250×200 example), dilate/erode,
  clone/smudge on synthetic images, PNG metadata roundtrip, config/preset parsing,
  default-resolution order, undo/redo stack.
- Sampler correctness: compare Euler a / DPM++ 2M outputs against `k_diffusion`
  reference runs on a tiny fake model (CPU) — catches wiring bugs (sigma scaling,
  CFG rescaling, LoRA application points) without an SDXL GPU.
- On the local GPUs: `devtools/` scripts automate the smoke test (real SDXL
  txt2img, inpaint both modes with protected-region assertions, seed
  determinism, sampler/scheduler combos over HTTP+WS, offline cache check,
  pipeline reference diff). Manual: model switch unload/reload, two-GPU queue
  (both cards present — see above).
- UI: keep JS dependency-free so it's eyeball-testable. `tests/ui/` drives a real
  headless Chromium (Playwright) against a live mock-backend server in a temp
  root — stroke/paint, mask+shift-erase+morph, zoom/pan, tool-state persistence
  and custom-brush listing are asserted against the server-side canvas/mask
  pixels (authoritative). The `page` fixture skips gracefully when Playwright's
  Chromium isn't installed (run `playwright install --with-deps chromium` once);
  the rest of the suite runs browser-less (httpx + WS).

## Open questions & recommendations

| Question | Options | Recommendation |
|---|---|---|
| Inference stack | (a) `diffusers` + `k_diffusion`, (b) fork/repurpose Forge, (c) A1111 `sd_samplers` directly | **(a)** `diffusers` (SDXL support, LoRA via PEFT, inpaint pipelines) with `k_diffusion` for sampler loops. Forge's k-diffusion samplers are literally this package; we port the small wrapper layer rather than dragging A1111's model/options/config machinery. |
| Backend framework | FastAPI vs Flask | **FastAPI** — async endpoints + WebSocket are first-class; generation is CPU-offloaded anyway so we block in worker threads, not the event loop. |
| Frontend | vanilla JS + Canvas2D; React; PixiJS | **Vanilla JS + Canvas 2D**, no build step. Requirements (stamps, mask overlay, zoom/pan transform) are all plain 2D canvas; a build pipeline would be overhead. ~1–2k LOC of JS, organized in modules. |
| Python deps | pip / uv | **uv** (Python 3.13-friendly; Forge targets 3.11 but diffusers/torch/k-diffusion are all 3.13-ready as of 2025). |
| LoRA application | diffusers `fuse_loras` vs PEFT `inject_adapter_in_model` | **PEFT inject** (supports multiple LoRAs with per-weight, which is the spec requirement). |
| Mask dilate/erode | scipy.ndimage vs cv2 | **scipy** (smaller dependency than OpenCV; binary_dilation with disk SE is enough). |
| Metadata format | PNG `parameters` text chunk (A1111/Forge) | Use Forge's exact `parameters` chunk layout + standard fields (`Prompt`, `Negative prompt`, `Seed`, `Size`, `Steps`, `Sampler`, `Scheduler`, `CFG scale`, `Model`, `Lora`), so Forge can import our outputs; verified with a roundtrip test. |
| Multi-GPU | thread-per-GPU with `torch.cuda.set_device` per worker | **Thread-per-GPU** (diffusers models aren't shared; each worker owns its device). One model in VRAM total → model switching serializes via a lock; the queue naturally queues behind it. |
| Segmentation model source | the extension uses `ultralytics` YOLO-cls + insightface | Port as-is (`ultralytics` + `insightface` as optional deps); it's self-contained in the extension folder. |
| Canvas sync protocol | stroke-based (chosen) vs pixel-diff vs full-canvas | **Stroke-based** (see Architecture). Brush PNGs sent as data to server on load. |

## Risks / watch-outs

- **SDXL + diffusers vs Forge quality**: sampler parity via k_diffusion should be
  exact (same package); LoRA weights via PEFT differ in plumbing only. Verify
  visually against Forge output in the GPU smoke test.
- **VRAM accounting for model-in-RAM**: a 6.9GB SDXL in CPU RAM per model × 2
  (active + last) is fine; but we must not keep *multiple* in VRAM — enforced by
  the single-loader lock.
- **Inpaint mask protection** in "inpaint not masked": use the mask as an
  `init_image` composite (masked area gets the canvas pixels, unmasked gets the
  noise/blank) per standard SD inpaint convention — confirm exact convention when
  porting from `modules/masking.py`.
- **Undo memory**: full-res snapshots of a 1024² canvas ≈ 3MB RGB; cap stack at
  ~200 entries, document the cap (spec says "unlimited" — revisit if user-visible,
  e.g. store PNG-compressed).
- **Mock-only coverage is not enough for the backend**: real-inference bugs
  (tensor shapes, device placement, dtype, VRAM) only surface when torch code
  actually runs — see "Testing strategy" for the GPU smoke workflow. Keep the
  mock backend interface faithful to the real one to minimize the delta.
