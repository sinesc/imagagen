# imagagen

AI image generator with a unified canvas. See `concept.md` for the full feature
spec and `PLAN.md` for the implementation plan.

## Setup (dev)

```bash
uv sync --extra dev          # creates .venv
.venv/bin/python -m pytest   # run tests (mock backend, no GPU needed)
bash scripts/smoke.sh        # start mock server + hit endpoints
```

## Run

```bash
.venv/bin/python -m imagagen --root <project-root> --port 8080
# --mock forces the CPU mock backend (automatic fallback when no CUDA/models)
```

### Running on the host system

The container's Python lives at container-only paths, so the host uses a
self-contained venv: `.venv3.13` (venv) backed by a full CPython 3.13 install
in `.python313/` (both in the shared workspace, both gitignored). From the
host, run:

```bash
.venv3.13/bin/python -m imagagen --root <project-root> --port 8080
```

Rebuild after dependency changes (from the container):

```bash
UV_PROJECT_ENVIRONMENT=.venv3.13 uv sync \
  --python .python313/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13
```

The rebuild rewrites two files with absolute container paths that only
resolve inside the container. Re-make them relative afterwards (the same
commands originally used to create the venv):

```bash
ln -sfn ../../.python313/cpython-3.13.15-linux-x86_64-gnu/bin/python3.13 .venv3.13/bin/python
sed -i 's|^home = .*|home = ../../.python313/cpython-3.13.15-linux-x86_64-gnu/bin|' .venv3.13/pyvenv.cfg
printf '../../../../src\n' > .venv3.13/lib/python3.13/site-packages/_editable_impl_imagagen.pth
```

Place SDXL weights in `models/sdxl/` and LoRAs in `loras/sdxl/` (subdirectories
are scanned recursively, models/LoRAs are addressed by their relative path
without extension, e.g. `test-dir/JuggernautXL9.0`). Small HuggingFace hub
files fetched on first load are cached in `cache/`. Greyscale brush
PNGs in `brushes/`. Outputs land in `output/`.

**Person mask generation**: drop a MediaPipe image-segmenter `.tflite` into
`models/segmentation/` (e.g. the selfie multiclass model, 16 MB, from
`https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite`).
The toolbar's model dropdown + *segment* button then generate a person-part
mask into the mask buffer (undoable). MediaPipe needs the `libEGL`/`libGLES`
system libraries (Debian/Ubuntu: `apt install libegl1 libgles2`). Named model configurations
live in `model-configs/`, generation-setting presets in `presets/` (both
plain JSON, created/edited from the UI, both safe to edit by hand).

Optional: put an improved VAE (`sdxl_vae.safetensors`, e.g. the ft-MSE
"Improved VAE") in `models/`. The stock SDXL VAE overflows to NaN in fp16 for
high-magnitude latents; the improved VAE is fp16-stable and renders color
better. If it is present, it is swapped in automatically.

### VRAM

SDXL in fp16 is ~6.6 GiB of weights. Generation at 1024² needs roughly
1.5 GiB more. The text encoders are offloaded to CPU after prompt encoding.
On a 16 GB card shared with other GPU work, expect 1024² to be at the ceiling
(512²–768² are comfortable).

### GPU selection

Which GPU the server uses is configured in `imagagen.ini` in the project
root:

```ini
[imagagen]
# "auto" (default): the non-ignored GPU with the most free VRAM
# or an explicit index, e.g. primary-gpu = 1
primary-gpu = auto
# comma-separated indices to exclude from auto-pick (a friendlier
# alternative to CUDA_VISIBLE_DEVICES), e.g. ignore-gpus = 3,4,5,6
ignore-gpus =
```

With `auto` the server re-checks free VRAM on startup and on every
`/api/health` call, so a card that was busy during development is picked
around automatically. The chosen GPU is reported by `GET /api/health`
(`gpu` plus a `selected` flag per entry of `gpus`).

### Model configurations

Every model shows up in the selector as at least one *configuration*:
models with files in `model-configs/` get one entry per file (you can have
several per model, e.g. "Base" and "Base + DMD2 Lora"), unconfigured models
get a read-only in-memory default. A configuration stores, separately for
text-to-image and inpainting, a sampler, a scheduler, a CFG and a LoRA list;
`null` means "no override". Per generation, the form's values win unless a
setting is set to "Use model default" (CFG −1 in the form), in which case the
configuration's value is used, then the backend built-in. The configuration's
LoRAs are always loaded in front of the LoRAs you pick in the form.

### UI state persistence

The web UI's last-used form inputs (model configuration, prompt, negative
prompt, size, steps, CFG, seed, batch, sampler, scheduler, LoRAs, preset) are
saved automatically to `ui-state.json` in the project root as you type and on
every generation. On server start the UI restores them, so the page picks up
where you left off. The file is plain JSON and safe to edit by hand;
a missing or corrupt file simply falls back to the form defaults.

```json
{
  "model": "epicrealismxl11-base",
  "prompt": "a red panda on a mossy log",
  "steps": 28,
  "cfg": 6.5,
  "sampler": "DPM++ 2M",
  "scheduler": "Karras",
  "loras": [["sdxl/dmd2", 0.8]]
}
```

(`model` is a configuration id: the sanitized configuration name, or the
model's relative path for its in-memory default entry.)

This is deliberately separate from `imagagen.ini` (operator
configuration): `ui-state.json` is rewritten constantly by the app, while
the INI is something you edit by hand.
