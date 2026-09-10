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
PNGs in `brushes/`. Outputs land in `output/`.
