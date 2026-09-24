# Flux LoRA Support Plan

## Context
`FluxBackend.supports_loras = False`. The test asset
`loras/flux/MetalSuitFlux1.0.safetensors` uses standard (unfused) Flux LoRA
key naming (e.g. `transformer.transformer_blocks.0.attn.to_q.lora_A.weight`,
`transformer.single_transformer_blocks.0.attn.to_v.lora_B.weight`; rank 16;
no `.alpha` tensors → scale 1.0). Our transformer is a port of Forge's
`backend/nn/flux.py` (fused `qkv`/`linear1`/`linear2`, GGUF Q4_K weights
dequantized per op in `GgufLinear`), so LoRA keys do not 1:1 match our
modules and a key mapping is required.

## What Forge does (reference: `dev/implementations/stable-diffusion-webui-forge`)
- **Key mapping**: `packages_3rdparty/comfyui_lora_collection/utils.py::flux_to_diffusers`
  maps each standard key to the internal (fused) weight + slice
  `(dim, offset, length)`. `model_lora_keys_unet` registers them under the
  `transformer.` LoRA prefix.
- **Load**: `packages_3rdparty/comfyui_lora_collection/lora.py::load_lora`
  pairs A/B (several naming conventions), reads the per-key scalar
  `<base>.alpha` tensor, builds a patch `("lora", (A, B, alpha, ...))`.
- **Apply**: `backend/patcher/lora.py::LoraLoader.refresh`, two modes:
  - **merged** (default): `w += strength * (alpha/rank) * (B @ A)`; GGUF
    weights are dequantized, merged, then **re-quantized** to Q4_K_S
    (`operations_gguf.quantize_pytorch`).
  - **online**: A/B kept on the layer (`forge_online_loras`), applied on
    every forward in `backend/operations.py::get_weight_and_bias`.

## Design decisions
- **D1 — Runtime additive deltas (Forge's online mode).** We have no Q4_K_S
  *quantizer*, so the merged/re-quantize path is not available without
  porting a quant kernel (and it is lossy). Runtime deltas are
  mathematically identical to merging, lossless, trivially reversible
  (LoRA sets change per generation in our UI), and consistent with the SDXL
  backend's runtime PEFT adapters.
- **D2 — Port Forge's `flux_to_diffusers` mapping table** (verified in the
  reference checkout), adapted to our module paths (no `diffusion_model.`
  prefix; our state-dict keys are `double_blocks.0.img_attn.qkv.weight` etc.).
- **D3 — All-or-nothing apply** under the shared `_PARAM_INIT_LOCK` (SDXL
  LoRA lessons): a failing set must leave no partial state.
- **D4 — Family-filter the LoRA list** (API + UI) so SDXL LoRAs are not
  selectable for Flux models and vice versa.

## Implementation

### 1. Key mapping (new `src/imagagen/flux_lora.py`)
H = 3072, M = 12288. LoRA base key = key minus `transformer.` prefix and
`.lora_{A,B}.weight` suffix (also accept `.lora_up/.lora_down`; strip a
trailing `.0`, so `to_out.0` == `to_out`). Target = our module under
`transformer.` + output-row slice `[r0:r1)`.

**Double blocks** (`transformer_blocks.N`, N = 0..18):
| LoRA base key | Module | rows |
|---|---|---|
| `attn.to_q` / `to_k` / `to_v` | `double_blocks.N.img_attn.qkv` | `[0:H)` / `[H:2H)` / `[2H:3H)` |
| `attn.add_q_proj` / `add_k_proj` / `add_v_proj` | `double_blocks.N.txt_attn.qkv` | `[0:H)` / `[H:2H)` / `[2H:3H)` |
| `attn.to_out` | `double_blocks.N.img_attn.proj` | full |
| `attn.to_add_out` | `double_blocks.N.txt_attn.proj` | full |
| `ff.net.0` (or `ff.net.0.proj`) | `double_blocks.N.img_mlp.0` | full |
| `ff.net.2` | `double_blocks.N.img_mlp.2` | full |
| `ff_context.net.0` (or `.proj`) | `double_blocks.N.txt_mlp.0` | full |
| `ff_context.net.2` | `double_blocks.N.txt_mlp.2` | full |
| `norm1.linear` | `double_blocks.N.img_mod.lin` | full |
| `norm1_context.linear` | `double_blocks.N.txt_mod.lin` | full |

**Single blocks** (`single_transformer_blocks.N`, N = 0..37):
| LoRA base key | Module | rows |
|---|---|---|
| `attn.to_q` / `to_k` / `to_v` | `single_blocks.N.linear1` | `[0:H)` / `[H:2H)` / `[2H:3H)` |
| `attn.proj_mlp` (or `proj_mlp`) | `single_blocks.N.linear1` | `[3H:3H+M)` |
| `proj_out` | `single_blocks.N.linear2` | full |

The single block's `proj_out` is the fused 5H→H output projection (input =
`cat[attn, mlp]`); standard single-block attention is `pre_only` (no
`attn.to_out`), so no input-column slicing is needed anywhere.

**Global** (no block index):
| LoRA base key | Module | note |
|---|---|---|
| `x_embedder` | `img_in` | |
| `context_embedder` | `txt_in` | |
| `time_embedder.linear_1` / `linear_2` | `time_in.in_layer` / `time_in.out_layer` | |
| `vector_embedder.linear_1` / `linear_2` | `vector_in.in_layer` / `vector_in.out_layer` | |
| `guidance_embedder.linear_1` / `linear_2` | `guidance_in.in_layer` / `guidance_in.out_layer` | |
| `norm_out.linear` | `final_layer.adaLN_modulation.1` | swap the two output-row halves of B (diffusers = [scale, shift], ours = [shift, scale]; Forge: `swap_scale_shift`) |
| `proj_out` (block-level, BFL naming) | `final_layer.linear` | |
| `single...modulation` / double `norm1.linear` equivalents | `*.modulation.lin` | (double: `img_mod.lin`/`txt_mod.lin` above) |

Unknown keys (e.g. QK-norm 1-D scales, text-encoder keys, A1111
`lora_unet_*`): collect and raise `ValueError` listing them (fail loudly,
matches our all-or-nothing policy; Forge merely warns).

### 2. Delta application (`src/imagagen/gguf_ops.py`)
`GgufLinear` gains an optional delta list, applied after the base matmul in
`forward` (base weight stays quantized; deltas are small bf16 tensors):

```python
out = torch.nn.functional.linear(x, weight, bias)
if self.lora_deltas:
    for A, B, r0, r1 in self.lora_deltas:   # A (r, H), B (len, r)
        out[..., r0:r1] += (x @ A.t()) @ B.t()
```

- A/B are registered as module parameters (so `transformer.to(device)` moves
  them) and created only under `_PARAM_INIT_LOCK` (see D3).
- At apply time, deltas targeting the same module are concatenated
  (A along dim 0, B along dim 1, adjacent row slices merged) so each module
  costs ONE extra GEMM pair, and `weight * (alpha/rank)` is pre-folded into
  A. MetalSuit → 57 extra GEMM pairs per pass (vs 532 without merging).
- Empty list → zero overhead (one attribute check).

### 3. Backend (`src/imagagen/flux_backend.py`)
- `supports_loras = True`.
- `_lora_path(name)`: `config.loras_dir / "flux" / (name + ".safetensors")`
  (string-append the extension — names contain dots).
- `_parse_lora(path)` (cached by `(path, mtime)`): read safetensors, pair
  A/B, read `<base>.alpha` scalars (default 1.0 → scale = alpha/rank,
  rank = A.shape[0]), map keys per §1, group by module → delta specs
  (module path, A, B, r0, r1). CPU tensors.
- `_apply_loras(loras)` / `_unload_loras()`: all-or-nothing, under
  `self._lock` (called from `_generate_impl` before the sampling loop, as in
  SDXL) and the shared `_PARAM_INIT_LOCK` (move it to `backends.py`, import
  from both backends — no import cycle). Unload clears the actual module
  state, not just the `_active_loras` cache. Commit `_active_loras` only
  after every module is patched; identical set → no-op.
- VRAM: deltas live on the GPU with the model (MetalSuit ≈ 50 MB bf16);
  `unload()`'s `transformer.to("cpu")` moves them too (registered params).

### 4. API & UI (family filter)
- `models.py`: `list_loras(config, family=None)` — filter to
  `loras/<family>/` when given.
- `app.py`: `GET /api/loras?family=...` (default: all, backward compatible).
- `app.js`: `refreshLoras()` mirroring `refreshSamplers()` — read the
  selected model's family, refetch `/api/loras?family=...`, update
  `state.loras`, drop selected rows no longer in the list. Call on init and
  on model change (where `refreshSamplers()` is called).

## Testing
1. **Unit (no GPU)**:
   - `tests/unit/test_flux_lora_mapping.py`: mapping table — representative
     double/single/global keys → expected (module, rows); trailing-`.0`
     normalization; unknown key raises with the key listed.
   - `tests/unit/test_flux_lora_apply.py`: `GgufLinear` with plain (non-
     quantized) float32 weights + synthetic deltas on CPU: forward ==
     manual `base + x@Aᵀ@Bᵀ` (incl. row-sliced case); concatenated multi-
     delta case; empty list == base.
   - Parse `MetalSuitFlux1.0.safetensors` on CPU: 266 A/B pairs, all keys
     mapped, rank 16, scale 1.0.
2. **GPU smoke** `dev/smoke_flux_lora.py` (mirrors `smoke_sdxl_lora.py`):
   same seed 512²/4 steps with and without MetalSuit@1.0 → images differ,
   no NaN/uniform output, no OOM; print with/without runtime delta.
3. **Regression**: `dev/smoke_flux_img2img.py` (LoRA-free path unchanged),
   existing unit suite.

## Risks
- Extra GEMM launch overhead: mitigated by per-module concatenation (57
  pairs); the smoke test measures it (expect ≲ 1 s per 20 steps at 1024²).
- A LoRA with keys we cannot map (QK-norm scales, A1111 naming): loud
  `ValueError` at apply time, job fails with a clear message (WS error
  event), no partial state.
- `norm_out.linear` (adaLN) row-swap: rare key; implemented per Forge's
  `swap_scale_shift`, covered by the mapping unit test.
