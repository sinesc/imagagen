# Plan: Flux "distilled model" flag in model configurations

## Goal

FLUX.1-dev (our `agFluxQ4KS` GGUF) is guidance-distilled: the "CFG" value is
embedded in the transformer forward as the `guidance` embedding (BFL default
3.5), one pass per step, no negative prompt. A classic-CFG value (e.g. 1.0
or 7.0) silently reinterpreted as guidance produces broken images.

Add a **"distilled model" checkbox to the model configuration** (entry level,
shared by t2i/i2i). No changes to the main generation UI.

- **Checked (default):** CFG value = embedded guidance (current behavior).
  `guidance = cfg if cfg > 0 else 3.5`, one transformer pass per step,
  negative prompt ignored.
- **Unchecked:** classic CFG. The guidance embedding stays at 3.5 (it is part
  of the network and always added to the time embedding, so it cannot be
  removed), and the CFG value runs a real unconditional pass with the
  negative prompt: `v = v_uncond + cfg * (v_cond - v_uncond)` per step,
  `cfg = cfg if cfg > 0 else 1.0`, two transformer passes per step (~2x time).
  Matches ComfyUI/Forge semantics (ComfyUI: guidance knob + cfg knob;
  Forge: "Distilled CFG Scale" slider + CFG slider).

## Capability auto-detection

The transformer's `guidance_in` MLPEmbedder exists only in
guidance-embedding checkpoints (dev). Schnell-class checkpoints lack it
(`guidance_in` = `nn.Identity`). Detect at load:
`supports_distilled = "guidance_in.weight" in sd` (the gguf state dict,
before `load_state_dict`). Effective distilled mode =
`request.distilled and self.supports_distilled`. Log when a checkpoint has
no guidance embedding (distilled requested but impossible).

## Changes

### src/imagagen/model_configs.py
- `ModelConfig.distilled: bool = True` (entry level, NOT in ModeConfig: both
  directions must share the CFG semantics). Pydantic dump/validate round-trips
  it in the JSON files for free.
- `ConfigEntry.distilled: bool`; `to_dict()` includes it.
- `entries()`: file entries pass `c.distilled`; in-memory default entries get
  `distilled=True` (= current behavior for unconfigured models).

### src/imagagen/backends.py
- `GenerateRequest.distilled: bool = True`.
- `ModelBackend.is_flux: bool = False` class attribute (FluxBackend: True) —
  used by the queue to decide whether the metadata line applies.

### src/imagagen/app.py
- `/api/generate`: `GenerateRequest(..., distilled=entry.distilled, ...)`.
- `/api/samplers`: response gains `"family": family` (UI shows the checkbox
  only for flux configurations).

### src/imagagen/flux_backend.py
- `__init__`: `self.supports_distilled` from the state dict keys (above).
- `_generate_impl`:
  - `distilled = request.distilled and self.supports_distilled`.
  - distilled: guidance = `request.cfg if > 0 else 3.5` (unchanged), 1 pass.
  - not distilled: guidance fixed at 3.5, encode `request.negative_prompt`
    once per generation (T5 context + CLIP pooled, same `_encode_prompt`),
    2 passes per step with the CFG combination above; cancel is checked
    before each pass.

### src/imagagen/queue.py + src/imagagen/metadata.py
- `_result_to_png` → `build_parameters(..., distilled=req.distilled if
  backend.is_flux else None)`.
- `build_parameters(distilled: bool | None = None)`: writes
  `Distilled: true|false` (both states, like the SAG flags) on the meta line,
  omitted when None (SDXL). `parse_parameters` already keeps unknown keys
  verbatim (`out["Distilled"] = "true"`), so no parser change; import does NOT
  write the checkbox (it lives in user/model-configs files).

### src/imagagen/static/app.js
- `state.family` set from the `/api/samplers` response in `refreshSamplers`.
- `openConfigDialog`: when `state.family === "flux"`, add a "distilled model"
  checkbox row (between the model row and the first fieldset) initialized from
  `entry.distilled`, with a title explaining both modes; POST body includes
  `distilled` (checkbox when flux, else `entry.distilled` unchanged).

### Tests
- `tests/unit/test_model_configs.py`: `distilled` defaults True (file +
  in-memory default entries), round-trips False through save/reload, present
  in `to_dict`.
- `tests/unit/test_metadata.py`: `build_parameters(distilled=True/False)`
  writes the line, `None` omits it; `parse_parameters` reads it back.

### Verification (real GPU)
`dev/gen.py` gains `--distilled / --no-distilled`:
1. distilled (default): same output as before the change (regression).
2. `--no-distilled --cfg 1.0` with negative prompt: plausible image, no
   uniform/NaN; ~2x per-step time vs distilled.
3. `--no-distilled` with default cfg: equals distilled @ guidance 3.5? (No —
   distinct modes; only sanity-check output quality.)
