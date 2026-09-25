# Plan: Flux `.safetensors` model support (Kijai FP8 + NF4)

## Goal

Load the four `.safetensors` checkpoints in `models/flux/` as selectable Flux
models, alongside the existing `agFluxQ4KS_v21.gguf`:

| file | format | transformer arch |
|---|---|---|
| `flux1DevFP8110GBKijai.safetensors` (11.9 GB) | FP8 (all tensors `F8_E4M3`) | dev (has `guidance_in`) |
| `flux1SchnellFP811GBKijai.safetensors` (11.9 GB) | FP8 (all tensors `F8_E4M3`) | schnell (no `guidance_in`) |
| `flux1DevNF4V2.safetensors` (12.0 GB) | bitsandbytes NF4, no nested stats | dev |
| `flux1SchnellNF4.safetensors` (11.5 GB) | bitsandbytes NF4, nested stats | schnell |

Both stay **quantized on the GPU with per-op bf16 dequantize** (the
`GgufLinear` pattern), so VRAM ≈ 6 GB for the 12 B dev models, not the
12 GB a load-time bf16 dequantize would need. No bitsandbytes *runtime
matmul* — only its dequantize op (bnb `>=0.50.2` is already a declared
dependency).

## File formats (verified against the shipped files + bnb 0.50.2 source)

**FP8 files:** transformer only, 780/776 tensors, **every** tensor
(weights, biases, norm scales) is `F8_E4M3` at the real weight shape, no
scale tensors. Dequantize = exact cast to bf16. Keys are top-level
diffusers names (`double_blocks.0.img_attn.qkv.weight`,
`single_blocks.N.linear1.weight`, `img_in.weight`, …) — the same key
naming as the GGUF file. `guidance_in.*` present only in the dev file.

**NF4 files:** all-in-one pipeline files. Only the transformer section
(`model.diffusion_model.` prefix) is used; the `vae.*` (bf16),
`text_encoders.clip_l.*` (fp16) and `text_encoders.t5xxl.*` (fp8) sections
are ignored — the infra components
(`infra/{clip_l,t5xxl_fp8_e4m3fn,flux_ae}.safetensors`) keep being used.
For every quantized linear there is:

- `<k>.weight` — `U8 [n/2]`, 4-bit packed (2 values per byte);
- `<k>.absmax` — one entry per 64-element block, `F32 [n/64]` (dev V2)
  or `U8 [n/64]` (schnell, nested);
- `<k>.quant_map` — `F32 [16]`, **not used at runtime** (the kernel has
  the NF4 codebook built in);
- `<k>.quant_state.bitsandbytes__nf4` — `U8` blob holding a JSON string:
  `{"quant_type":"nf4","blocksize":64,"dtype":"bfloat16","shape":[out,in]}`
  plus, for the nested files only, `nested_blocksize` (256),
  `nested_dtype`, `nested_offset`;
- nested files only: `<k>.nested_absmax` `F32 [n/64/256]` and
  `<k>.nested_quant_map` `F32 [256]` (the NF4 codebook for the uint8
  absmax indices);
- `<k>.bias` — plain `BF16` (norm scales: `F32` in V2, `BF16` in
  schnell; also plain).

Dequantize is bnb's **native pre-quantized format**: `QuantState`
(`absmax`, `blocksize=64`, `quant_type="nf4"`, `dtype=bf16`, `shape`,
`offset`, and for nested a `state2` QuantState carrying
`nested_absmax`/`nested_quant_map`/`nested_blocksize`) +
`bitsandbytes.functional.dequantize_4bit(packed, quant_state=qs,
blocksize, quant_type)` → bf16. The nested absmax (`dequantize_blockwise`
of the `U8 [n/64]` index array through `state2`, `+ offset`) is handled
inside that call. Verified spikes: CPU decode works for both variants;
CUDA decode works (0.14 ms for a 3072×3072 weight, bit-exact vs CPU).
bnb prints a harmless
`No prebuilt binary for CUDA 12.9, loading CUDA 12.8 instead`.

## Changes

### `src/imagagen/gguf_ops.py` — quantized-linear generalization

1. **`QuantLinear` base class** (extracted from `GgufLinear`, which
   becomes a subclass keeping `Q4_K` dequantize):
   - `__init__(in_features, out_features, bias=True)` — dummy
     `nn.Parameter(torch.empty(1, device=_DEVICE, dtype=_DTYPE))`,
     `weight`/`bias` = `None`, `lora_deltas` list (all as today).
   - `_load_from_state_dict` / `_apply` — as today; `weight` may be a
     `GgufParam`, an `Nf4Param`, or a plain tensor (fp8); `bias` a plain
     tensor. (The `computation_dtype` attribute stays a GgufParam
     detail, accessed via `getattr`.)
   - `set_lora_deltas` / `clear_lora_deltas` — as today.
   - `forward(x)` template: `w = self.weight.to(device=x.device)` (the
     param classes move their aux data with `.to()`), then
     `w = self._dequantize(w).to(device=x.device, dtype=x.dtype)`, then
     `F.linear` + lora-delta rows — exactly the `GgufLinear.forward`
     body with `_dequantize` as the hook.
   - `_dequantize(self, w)`: `GgufLinear` → `dequantize_tensor(w)`;
     `Fp8Linear` → identity (the template's cast does it);
     `Nf4Linear` → build `QuantState` from the param and call
     `dequantize_4bit`.
2. **`Fp8Linear(QuantLinear)`** — weight/bias stored as-is in
   `F8_E4M3`, cast to the input dtype per forward (same math as
   `t5_encoder.Fp8Linear`; that class is left alone).
3. **`Nf4Param(torch.nn.Parameter)`** — like `GgufParam`:
   `.data` = packed `U8 [n/2]`, plus attributes `absmax` (`U8`/`F32`
   `[n/64]`), `nested_absmax`/`nested_quant_map` (nested files only),
   `blocksize`, `quant_type`, `out_dtype`, `real_shape` (`[out, in]`),
   `nested_offset` (or `None`). `.to(*args)` returns a copy with **all**
   tensors moved (same `copy_with_data` mechanism as `GgufParam`), so
   `QuantLinear._apply`/module `.to(device)` keep working unchanged.
   `Nf4Linear.forward` builds the `QuantState` (with `state2` when
   nested) from these fields on `x.device` and calls
   `dequantize_4bit` (no `bnb` import at module top level — import
   inside the function, so a CPU-only test env still imports).
4. **`quant_linear_scope(kind, device, dtype)`** context manager
   (monkeypatches `torch.nn.Linear` to `GgufLinear` / `Fp8Linear` /
   `Nf4Linear` by `kind` in `"gguf" | "fp8" | "nf4"`);
   `gguf_linear_scope` becomes a thin wrapper (still importable).
5. **Loader** (same module):
   - `sniff_flux_checkpoint(path) -> "gguf" | "fp8" | "nf4"`:
     `.gguf` magic → `gguf`; else open with
     `safetensors.safe_open`, key set contains
     `*.quant_state.bitsandbytes__nf4` → `nf4`; a transformer
     `.weight` slice has dtype `F8_E4M3` → `fp8`; anything else →
     raise (plain bf16 flux safetensors need 12 GB bf16, unsupported).
   - `load_flux_state_dict(path, kind) -> dict`: module-path → tensor
     mapping in the GGUF sd convention (e.g.
     `double_blocks.0.img_attn.qkv.weight`).
     - `nf4`: strip the `model.diffusion_model.` prefix, skip
       `vae.`/`text_encoders.` sections; per weight build an
       `Nf4Param` from `weight` + aux tensors (parse the JSON blob for
       `blocksize`/`dtype`/`shape`/`nested_*`; drop `quant_map`);
       biases/scales are plain tensors cast to bf16.
     - `fp8`: skip nothing (no prefix, no other sections); weights and
       biases stay `F8_E4M3`.

### `src/imagagen/flux_backend.py`

- `__init__`: branch on checkpoint kind:
  - `gguf` — unchanged.
  - `fp8` / `nf4` — build `FluxTransformer2DModel(**_TRANSFORMER_ARCH)`
    under `quant_linear_scope(kind, cpu, bf16)`,
    `sd = load_flux_state_dict(self.model_path, kind)`,
    `self.transformer.load_state_dict(sd, strict=False)`, `.to(device)`,
    `eval()`. No bake step (NF4 dequantize builds its state lazily;
    FP8 needs nothing).
  - `supports_distilled = "guidance_in.in_layer.weight" in sd` — same
    detection for all three kinds (dev files have it, schnell files
    don't, GGUF unchanged).
- `_apply_loras`: the `isinstance(module, GgufLinear)` check becomes
  `isinstance(module, QuantLinear)` (and the error message), so LoRAs
  work for every transformer format.

### `src/imagagen/models.py`

- `_FAMILIES`: flux suffixes `(".gguf",)` → `(".gguf", ".safetensors")`
  (+ update the "deliberately left out" comment — the four files now
  surface as models `flux1DevFP8110GBKijai`, `flux1SchnellFP811GBKijai`,
  `flux1DevNF4V2`, `flux1SchnellNF4`).

No API/UI changes: `/api/models` lists the new ids,
`/api/samplers?model_id=` still returns the flux sets, the distilled
checkbox and LoRA list are family-driven and keep working.

## VRAM (16 GB GPU, verified budget)

- NF4 dev: 6.0 GB packed + 0.75 GB absmax (F32) + ~0.2 GB bf16 ≈ 6.9 GB.
- FP8 dev: ≈ 6.1 GB. Schnell ≈ half. Both fit with T5/CLIP parked on
  CPU and the 1024² sampling working set (~13 GB total), like the GGUF
  model (12.7 GB).

## Tests

1. **`tests/unit/test_quant_linear.py`** (new):
   - NF4 round-trip: `bnb.quantize_4bit` a random bf16 weight
     (`blocksize=64`, `compress_statistics`/nested True **and** False),
     `QuantState.as_dict(packed=True)` → reassemble via
     `QuantState.from_dict` → `Nf4Linear` dequantize must be bit-exact
     vs `dequantize_4bit` and within NF4 tolerance of the original.
   - `Nf4Param.to()` moves data + every aux tensor together.
   - `Fp8Linear`: fp8 weight/bias, bf16 input → equals `F.linear` of the
     cast weight; lora deltas add correctly.
   - `sniff_flux_checkpoint`/`load_flux_state_dict` on small synthetic
     safetensors written with `safetensors.torch.save_file` (both
     layouts, incl. the `model.diffusion_model.` prefix, the JSON
     quant-state blob, and a plain-bf16 file that must raise).
2. **`tests/unit/test_model_discovery.py`**: `models/flux/*.safetensors`
   is now discovered (flip the "ignored" fixture assertion).
3. Existing unit suite stays green (esp. `test_flux_lora_apply.py`,
   `test_backends.py` — `GgufLinear` keeps its name and behavior).
4. **GPU smoke `dev/smoke_flux_safetensors.py`** (like
   `dev/smoke_flux_img2img.py`), per file: load through the registry,
   assert `supports_distilled` (dev = True, schnell = False), generate
   one 512²/10-step image, apply a flux LoRA and regenerate, report peak
   VRAM; save PNGs to `dev/smoke_flux_safetensors/` for inspection.
   Cross-check the same seed/prompt on `flux1DevNF4V2` vs
   `flux1DevFP8110GBKijai` (different quantizations ⇒ small differences,
   same scene). The schnell run also exercises the nested
   (absmax-blocksize-256) dequantize path on CUDA.

## Milestones (commit per step)

1. `QuantLinear` base + `Fp8Linear` + `Nf4Param`/`Nf4Linear` + scope +
   loader + unit tests (no backend wiring yet).
2. `flux_backend.py` + `models.py` wiring + smoke script run on all four
   files + peak-VRAM numbers.
