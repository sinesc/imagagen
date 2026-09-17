# devtools

Helper scripts for development sessions (not part of the package). Run from
the project root with the venv interpreter, e.g.:

    .venv/bin/python devtools/gen.py "a red cube on a table"

## Scripts

| script | purpose |
| --- | --- |
| `ascii_view.py <png> [w] [h]` | Render an image as ASCII art + per-region average colors. The main "vision" substitute: shows composition, blank corners, shadows, flat/blank regions. |
| `gen.py "prompt" [--w/--h/--steps/--cfg/--seed/--sampler/--scheduler/--model/--out]` | Generate directly through the backend (no server). Prints timing, VRAM before/after, and a noise-vs-structure check (`structure(16x16)` should be well below `std` for a real image). |
| `seed_check.py` | Regression test for seed determinism: same seed x3 must be byte-identical (default + SDE samplers), different seeds must differ. Exits non-zero on failure. |
| `server_smoke.py [port]` | With a server running: one generation per sampler/scheduler combo over the real HTTP+WebSocket stack. Reports per-combo ok/error. |
| `offline_check.py` | Fresh subprocess with `HF_HUB_OFFLINE=1`: proves `models/hf_cache` is complete (load + generate with zero network). |
| `pipeline_reference.py` | Golden reference: runs the stock diffusers SDXL pipeline and our loop side by side and diffs the latent trajectory step by step (saved to `/tmp/ref_pipeline.png`, `/tmp/ref_mine.png`). Re-run whenever the sampling loop changes — a step-by-step growing diff pinpoints the regression. This is the tool that found the original four sampling-contract bugs. |
| `smoke_sdxl_inpaint.py` | Live SDXL img2img/inpaint check: txt2img baseline, whole-box denoise, and half-protected inpaint (asserts protected region stays ~identical, inpainted region changes). The mock backend never exercises the real mask-lock path, so re-run whenever `sdxl_backend.py` sampling changes. |
| `e2e_inpaint_ui_flow.py` | End-to-end over the real HTTP stack (temp root, GPU from a generated ini): bird generation → painted mask → `inpaint_masked` denoise 0.6, asserting masked-region change, unmasked preservation and mask retention. This is the tool that caught the `F.interpolate` 3D-tensor crash in the SDXL inpaint path. |
| `soft_inpaint_check.py` | Soft-inpainting (Forge builtin extension) A/B on GPU 1: same masked inpaint with the hard lock vs the soft blend + pixel composite (defaults and a bias variant). Asserts the inpaint core changes and the protected core is restored exactly by the composite (the hard run only reaches the VAE round-trip). Saves per-run PNGs + a side-by-side composite with an amplified diff panel. |

## Notes

- GPU scripts load the real model into VRAM; expect ~5 s load + a few s per
  512² generation on the 5070 Ti.
- `models/hf_cache` must be warm for offline/second-model work; a normal
  online generation warms it automatically.
- One-off bug-hunting scripts (bisection, per-step stats) were intentionally
  not kept — their lessons live in the code comments and in
  `pipeline_reference.py`.
