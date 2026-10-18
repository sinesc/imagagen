# Transformer parking delay + text-encoder prefetch plan

Self-contained plan (written 2026-02). Two performance features around the
idle-model parking path (`queue._maybe_unload_idle` -> `registry.unload_all`
-> `_park_state_locked` -> `backend.unload()`).

## Feature 1: `transformer-parking-delay` (ini)

Delay the park-to-RAM that currently happens as soon as the queue goes idle.

- `Config.transformer_parking_delay: float = 90.0`; ini key
  `transformer-parking-delay` (seconds). `0` = immediate (old behavior),
  `-1` = never park explicitly (models stay resident; they are still
  unloaded when VRAM is needed, e.g. encode swaps, model switch, captioning).
  Values `< -1` rejected.
- `GenerationQueue.__init__(..., parking_delay: float = 0.0)` (default keeps
  existing tests at current behavior; `create_app` passes the config value).
- `_maybe_unload_idle`: when idle, cancel any pending `threading.Timer` and
  re-arm per delay (`<0` -> none, `0` -> park now, `>0` -> daemon Timer).
  The timer callback re-checks `_is_idle_locked()` under `self._cond` before
  parking, so a job that started meanwhile is safe; the worker `finally` /
  `interrupt` / pre-encode tails re-arm it at the next idle transition.
  `stop()` cancels the timer. Registry parking still takes the per-GPU lock,
  so device moves never race a generation.

## Feature 2: `prefetch-text-encoder` (ini, default false)

On a prompt change (final, style/LoRA-applied prompt) whose encoding would
MISS the per-model `EncCache` (wildcard prompts always count as a miss --
per-seed picks cannot be predicted), immediately move the VRAM layout to the
"about to encode" state instead of waiting for parking-delay: text encoders
to `te_device`, transformer to CPU when they cannot co-reside (fit check,
same rule as `_encode_pairs`). Families where encoder+transformer already
co-reside (SDXL; NF4 Flux) keep the transformer resident -- offloading it
there would only add bus traffic. Only GPU-RESIDENT backends are touched; an
already-parked model stays parked (lazy-load design). Never blocks behind a
running generation (non-blocking lock acquire; busy GPUs are skipped).

- `ModelBackend.encoding_cached(prompt, negative) -> bool` (default True) and
  `ModelBackend.prepare_text_encode()` (default no-op), overridden by:
  - sdxl: `(prompt, negative or "") in cache`; prepare = move CLIPs to
    te_dev (small, UNet stays; encode/generate offload them again later).
  - flux: `(prompt, None) in cache` (classic-CFG negative misses fall back to
    the normal in-generate swap); prepare: if T5/CLIP not on GPU and
    transformer+encoders don't fit -> transformer to CPU, then `_text_to`.
  - zimage / krea2: `prompt in cache`; prepare as flux (single TE).
- Correctness fix required by the new "encoders on GPU, transformer on CPU"
  resident state: in flux/zimage/krea2 `_encode_pairs`/`_encode_prompts`
  add an elif branch (encoder resident but transformer on CPU -> `swapped =
  True` so the epilogue brings it back before sampling), and in the
  `_get_encoding`/`_get_context` CACHE-HIT paths offload resident encoders
  AND restore a CPU-side transformer. Without this, sampling would run with
  the transformer on the CPU.
- `ModelRegistry.prefetch_text_encoder(model_id, prompt, negative, force)`
  -> "not-resident" | "cached" | "prepared" | "busy": snapshot resident
  states matching model_id, skip if `encoding_cached` (unless `force` =
  wildcards present), `state.lock.acquire(blocking=False)` +
  `prepare_text_encode()` per GPU.
- API: `POST /api/prefetch-text-encoder` body `{model_id, prompt,
  negative_prompt, styles, loras}` (same payload shape as /api/prompt-tokens
  plus negative). Applies comments/styles/LoRA texts exactly like
  /api/generate; gated by `config.prefetch_text_encoder`. `_MockRegistry`
  mirrors the method for mock mode.
- Frontend (`static/app.js`): `scheduleTextPrefetch()` (debounced 400 ms,
  fired from `schedulePromptTokens()` which already covers prompt input,
  style changes, PNG import) + a listener on `#negative`. Best-effort POST.

## Docs + tests

- `res/imagagen.ini.default`: document both keys.
- unit: test_config (defaults 90.0/False, custom values, invalid values);
  test_queue (delay defers park; `-1` never parks; delay=0 immediate stays);
  integration: endpoint returns `disabled` by default, `prepared` with
  `prefetch_text_encoder=True` + forced wildcard prompt.
- Verify: `pytest tests/unit tests/integration`, targeted smoke for flux
  encode-swap state if practical; commit.
