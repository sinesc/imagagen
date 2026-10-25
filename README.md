# imagagen

imagagen is a locally-run AI image generator with a **unified canvas**. A
Python backend serves a web UI that runs entirely on your machine:
generate images from text, then paint, edit, inpaint, combine and
upscale them on one canvas — everything stays local.

## Supported platforms/models

- **Currently Linux/NVIDIA only**.
- Supports SDXL, Z-image turbo, Flux dev/schnell, Krea2.

## Features

- **Unified canvas** — several images live on one canvas: inpaint into
  masked regions, paint with customizable brushes, clone and smudge,
  copy/paste with a floating selection, rotate and crop. Everything is
  undoable.
- **Presets and model configurations** — save complete generation
  settings as named presets, and attach per-model configurations
  (sampler, scheduler, CFG and a fixed LoRA set, separately for
  text-to-image and inpainting).
- **Prompt styles & wildcards** — A1111-style reusable prompt templates
  (with a `{prompt}` placeholder and grouping), wildcards, and token
  weights like `(red fox:1.2)`. Every generated image stores its full
  settings in the PNG metadata: drop an image back onto the canvas and
  the UI restores its prompt, styles, LoRAs and seed. Also supports
  reading Forge PNG metadata.
- **LoRA support** — combine several LoRAs per generation with
  individual strengths and enable toggles, plus optional activation-tag
  toggles per LoRA.
- **Mask generation** — one-click subject/person masks from a
  MediaPipe segmentation model; edit the result by hand and undo freely.
- **Upscaling** — RealESRGAN-class upscalers with tiled processing for
  large outputs, cancellable mid-run.
- **Reference images** — check any image on the canvas as a reference: a
  vision model describes it and the description (or just selected
  sections of it) feeds the generation prompt.
- **Multi-GPU and VRAM management** — generation runs in parallel on
  several GPUs; models are parked/swapped automatically, with optional
  quantization to fit small cards.

## Requirements

- Linux (x86_64) and an NVIDIA GPU with a recent driver — the CUDA
  build of PyTorch is installed automatically by `uv`. CPU-only
  machines still start the app with a mock backend, but real generation
  needs a GPU.
- VRAM: SDXL runs comfortably on 8 GB (and on ~6 GB cards with the
  memory-saving options in the ini file). Larger models like Flux want
  12–16 GB. Expect a few GB of downloads for the Python dependencies.

## Installation

```bash
# 1. Get uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/sinesc/imagagen.git
cd imagagen

# 3. Install a self-contained CPython 3.13 into the project folder
#    (imagagen requires Python >= 3.13; keeping the interpreter inside
#    the project makes the setup independent of your distro's Python)
uv python install --install-dir "$PWD/.python313" --no-bin 3.13

# 4. Create the project venv (.venv) on top of it and install the
#    dependencies (a few GB, mostly CUDA-enabled PyTorch — first run only)
uv sync --python .python313/cpython-3.13.*-*/bin/python3.13
```

The launch script `launch.sh` keeps the venv relocatable so you can move
the `imagagen` folder wherever you want.

## Launching

```bash
./launch.sh
```

The first start copies the shipped defaults into `user/` (config,
example styles). The server then listens on port 8080 — open
<http://localhost:8080> in your browser. Extra arguments are passed to
the server, e.g. `./launch.sh --port 9000` or `./launch.sh --host
0.0.0.0` to reach it from another machine on your LAN.

## Configuration file

After the first launch there is `user/imagagen.ini`. Every option is
documented inline in the file; briefly, it controls:

- **GPU selection** — pick the GPU used for generation or exclude
  specific ones (no need for `CUDA_VISIBLE_DEVICES`; by default the
  GPU with the most free VRAM is picked automatically).
- **Memory-saving options** — UNet quantization, keeping text encoders
  in system RAM, forced VAE tiling — trade a little quality/speed to
  fit smaller cards.
- **Speed/residency knobs** — how long a loaded model stays warm in
  VRAM after a generation, and prompt-aware prefetching of text
  encoders so model swaps happen while you type instead of at Generate
  time.
- **Misc/debug** — whether the UI's form state is persisted, and
  verbose logging/memory debugging switches.

The file is plain text and re-read on every start.

## Where to put the weights

All model weights are gitignored; drop them into these folders (they
are scanned recursively and symlinked files/folders are followed):

- `models/<family>/` — generation checkpoints (SDXL, Flux, Z-Image,
  Krea 2), plus segmentation and upscaler models — details in
  [models/README.md](models/README.md)
- `loras/<family>/` — LoRAs, per model family — details in
  [loras/README.md](loras/README.md)
- `infra/` — shared text-encoder / VAE files needed by Flux, Z-Image
  and Krea 2 — details in [infra/README.md](infra/README.md)

The UI lists everything that is found; no registration step is needed.

## Your data

Everything the app writes lives in `user/` — `imagagen.ini`, presets,
model configurations, styles, wildcards, custom brushes and the UI's
saved state. Back up (or sync) that one folder and you keep all your
setup. Saved images go to `output/`; small files auto-fetched from
Hugging Face are cached in `cache/`.

## Development (optional)

```bash
uv sync --extra dev
.venv/bin/python -m pytest   # mock backend, no GPU required
```

Check the `dev` folder for additional information.