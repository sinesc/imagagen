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

Place SDXL weights in `models/sdxl/`, LoRAs in `loras/sdxl/`, greyscale brush
PNGs in `brushes/`. Outputs land in `output/`. Named model configurations
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
