# Flux model support

Support the provided FLUX.1-dev GGUF transformer (plus the provided CLIP-L,
T5-XXL and Flux VAE) as a **second model family** alongside SDXL. Generation,
inpaint, LoRA, metadata and the UI all work for Flux models.

## Status (txt2img working)

Implemented and verified end-to-end on the 16 GB GPU:

- `gguf_ops.py` — GGUF Q4_K_S per-op GPU dequantize (kernels ported from the
  reference-vendored copy; `gguf` PyPI only reads the file). Verified against
  an independent dequantize (max abs diff ~3e-4, fp16 rounding).
- `flux_transformer.py` — `FluxTransformer2DModel` port (SDPA, hand-rolled
  patchify/unpatchify, RoPE/`EmbedND`, double+single blocks, `LastLayer`,
  custom RMSNorm/QKNorm/Modulation). Full forward finite bf16 (1024² in ~1.2 s,
  ~7.9 GB peak).
- `t5_encoder.py` — T5-XXL fp8 port (`Fp8Linear` per-op cast, SDPA attention
  with `scale=1.0` + relative-position bias, gated ReLU FF).
- `clip_encoder.py` — bare `CLIPTextModel` (fp16), `pooler_output` = `y` (768).
- `kdiff.py` — `get_sigmas_flux` (flow-matching dynamic time-shift) +
  `sample_euler` (x0-contract flow-matching Euler).
- `flux_backend.py` — `FluxBackend(ModelBackend)`: load/encode/sample/decode,
  T5+CLIP offloaded to CPU after encode, `supports_loras = False`, `model_hash`.
- Wiring: `models.py` (family detection, `.gguf`, `_new_backend` dispatch),
  `config.py` (`ensure_dirs` for `models/{sdxl,flux}` + tokenizer dirs),
  `app.py` (family-aware `/api/samplers`), `backends.py` (`supports_loras`),
  `queue.py` (honest metadata: LoRAs only recorded when the backend applies
  them), `static/app.js` (refetch samplers per selected model's family).

Verification: 512²/1024² generations produce correct on-prompt images; all
unit (299) and integration (124) tests pass; `/api/samplers` returns
`["Euler"]`/no schedulers for flux and the SDXL sets otherwise.

**Key gotchas (see memory):** FLUX.1-dev is **bf16-native** (fp16 overflows);
T5 relative-position bucketing must keep indices in `[0, num_buckets)`
(`max_exact = num_buckets//4`), else a gather index-OOB → `CUBLAS` failure;
run the T5 forward under `inference_mode` (else every `.to(bf16)` weight copy
is retained → ~15 GB peak); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
reclaims load-time pool fragmentation. VRAM: all-weights resident 12.7 GB;
T5 working set ~0.4 GB (inference_mode) so 1024² peaks ~13.1 GB.

## Deferred

- **Flux LoRA** (`loras/flux/`): the diffusers `to_q/to_k/to_v` LoRA pairs
  must be fused into the single fused `qkv` Linear before applying.
- **Other flux checkpoints** on disk (`.safetensors` fp8/nf4, ~12 GB each):
  need a separate fp8/nf4 load path; left out of discovery so they never
  surface as un-loadable entries.
- Inpaint/img2img + soft inpainting for Flux (const `noise_scaling`
  `sigma*noise + (1-sigma)*latent`); SAG is not applicable.
- UI: hide/disable the LoRA section for flux models (currently the LoRA
  dropdown is family-agnostic).

## Assets (already in place)

- `models/flux/agFluxQ4KS_v21.gguf` — ~12B FLUX.1-dev transformer, **GGUF Q4_K_S**
  (the only weight file; not safetensors).
- `infra/clip_l.safetensors` — CLIP-L text encoder (246 MB).
- `infra/t5xxl_fp8_e4m3fn.safetensors` — T5-XXL text encoder, **fp8_e4m3fn** (4.9 GB).
- `infra/flux_ae.safetensors` — Flux VAE (335 MB).
- `loras/flux/MetalSuitFlux1.0.safetensors` — Flux LoRA.
- Reference with working Flux: `dev/implementations/stable-diffusion-webui-forge/`
  (branch `reference-branch`).

## How Flux differs from SDXL

| Aspect | SDXL (today) | Flux (new) |
|---|---|---|
| Backbone | `UNet2DConditionModel` (diffusers) | `FluxTransformer2DModel` (MMDiT: 19 double + 38 single blocks) |
| Checkpoint | `.safetensors` (diffusers single-file) | `.gguf` (Q4_K_S quantized bytes) |
| Prediction | epsilon: `x0 = x - sigma*eps`, `x_in = x/sqrt(sigma²+1)` | **const/flow-matching**: model returns `v`; `x0 = x - sigma*v`; `x_in = x` |
| Sigma schedule | beta schedule (Karras / Exponential / Polyexponential) | **flow-matching**: uniform [0,1] grid + dynamic **time-shift `mu(resolution)`** |
| Sampler | Euler a, DPM++ 2M, DPM++ 2M SDE, LCM | **Euler** (flow-matching Euler, non-ancestral) |
| Text | 2× CLIP-L (77 tok, pooled) | CLIP-L (77 tok, pooled → `y`) **+ T5-XXL (512 tok → `context`)** |
| Latent | 4 ch, scale 0.13025, improved VAE | **16 ch**, `scaling_factor 0.3611`, `shift_factor 0.1159` |
| Conditioning | true CFG (cond/uncond passes) | **dev**: guidance **embedding** (distilled CFG ≈ 3.5), single pass; **schnell**: none |
| LoRA target | `unet` | `transformer` |
| Negative prompt | used | **not used** (ignored) |
| Resolution | snap to 8 | transformer self-pads to even latent; **prefer snap to 16** |

Transformer architecture (FLUX.1-dev; verify exact dims against GGUF tensor
shapes): `hidden_size=3072`, `num_heads=24` (head_dim 128), `axes_dim=[16,56,56]`,
`theta=10000`, `qkv_bias=False`, `depth(double)=19`, `depth_single_blocks=38`,
`in_channels=16` (→64 patchified), `vec_in_dim=768` (CLIP pooled),
`context_in_dim=4096` (T5), `mlp_ratio=4.0`, `guidance_embed=True`.
Reference config: `…/FLUX.1-dev/transformer/config.json`.

## Design decisions

**D1. New `FluxBackend(ModelBackend)`** (new `flux_backend.py`) mirroring
`SdxlBackend`: owns transformer + text encoders + VAE loading, prompt
encoding, flow-matching sampling, VAE decode; implements `generate`, `cancel`,
`unload`.

**D2. Port the reference `IntegratedFluxTransformer2DModel`** (self-contained
nn.Module: RoPE/`EmbedND`, `DoubleStreamBlock`, `SingleStreamBlock`,
`LastLayer`, `MLPEmbedder`, `RMSNorm`) into `flux_transformer.py`. Use standard
PyTorch **SDPA** instead of Forge's `attention_function`. Hand-roll the few
`einops.rearrange/repeat` calls (or add `einops`). `forward(x, timestep,
context, y, guidance)` takes the 4-D latent and returns `v`.

**D3. GGUF per-op GPU dequantize** (Forge style) — required because 12B bf16
≈ 24 GB > 16 GB VRAM. Port into `gguf_ops.py`:
- `ParameterGGUF` (wraps raw GGUF uint8 bytes as a torch tensor; carries
  `gguf_cls`, `real_shape`, `computation_dtype`, `baked`).
- `ForgeOperationsGGUF.Linear` + `using_forge_operations(bnb_dtype="gguf")`
  (monkey-patches `torch.nn.Linear` so the weight stays quantized and is
  dequantized on every forward).
- `dequantize_tensor`, `bake_gguf_model`.
- Q4_K kernels **from the reference-vendored copy**
  (`packages_3rdparty/gguf/quants.py`): `bake`, `bake_inner`,
  `dequantize_blocks_pytorch`, `get_scale_min_pytorch`, helpers
  `quick_split`, `quick_unpack_4bits_u`, `change_4bits_order` (pure torch,
  GPU-resident).
- Load: `gguf.GGUFReader(path)` → `{name: ParameterGGUF(reader.tensor(name))}`
  → build transformer under `using_forge_operations(bnb_dtype="gguf")` →
  `load_state_dict` → `.to(gpu)` → `bake_gguf_model`.
- Memory: ~6.75 GB quantized bytes resident on GPU; per-layer bf16 weight is
  transient.
- Dep: add **`gguf`** (PyPI) for `GGUFReader` + constants. (PyPI `gguf` 0.19
  dequantize is numpy-only; the torch dequantize kernels come from the
  reference-vendored copy.)

**D4. Text encoding** — port reference `ClassicTextProcessingEngine`
(CLIP-L, 77 tok, `return_pooled=True`, `final_layer_norm=True`,
`text_projection=False`) + `T5TextProcessingEngine` (T5-XXL, 512 tok).
Conditioning: `y` = CLIP pooled (768), `context` = T5 last_hidden
(512 × 4096).
- Load CLIP-L from `infra/clip_l.safetensors` (transformers `CLIPTextModel`),
  T5-XXL from `infra/t5xxl_fp8_e4m3fn.safetensors` (transformers
  `T5EncoderModel`, **fp8 storage**); tokenizers/configs from the reference
  bundled HF dir (`FLUX.1-dev/{tokenizer,tokenizer_2,text_encoder,
  text_encoder_2}`).
- **Offload both encoders to CPU** between encode and sampling (like SDXL
  `text_encoder_device=cpu`): T5-XXL is 4.9 GB (fp8).

**D5. Sampling** — add plain **`sample_euler`** (non-ancestral) to `kdiff.py`
(flow-matching Euler: `x += (x - x0)/sigma * (sigma_next - sigma)`; with const
prediction = `x += v*(sigma_next - sigma)`). Flux model wrapper
`model_fn(x, sigma)` returns `x0 = x - sigma*v`.
- Sigma schedule: flow-matching, **not** the beta schedule. Build a
  `PredictionFlux`-style schedule: `sigmas = arange(1..N)/N` (or the 10000-step
  base), apply `calculate_shift(mu)` /
  `FlowMatchEulerDiscreteScheduler.time_shift` with
  `base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15`,
  `mu` from image seq len; subsample to the requested steps; append 0.
- dev: pass `guidance` (= `request.cfg`, default 3.5) to the transformer,
  **single pass** (no cond/uncond). Detect dev vs schnell by presence of
  `guidance_in.*` GGUF tensors / `guidance_embeds`.
- Inpaint/img2img: const `noise_scaling = sigma*noise + (1-sigma)*latent`;
  reuse the existing mask-lock / denoise-truncate logic. **SAG is not
  applicable** (UNet mid-block hook) — skip for Flux. Soft inpainting can
  reuse the x0-blend path (secondary).

**D6. Model family detection** (`models.py`):
- `ModelInfo` gains `family` (`"sdxl" | "flux"`), inferred from the top-level
  dir under `models/`.
- `discover()` also scans `models/flux/` for `.gguf`.
- `_load_backend_locked` dispatches to `SdxlBackend` / `FluxBackend` by family.
- LoRAs: `loras/flux/` is already auto-discovered (dir scan) — no change.

**D7. Samplers endpoint** — `/api/samplers` becomes family-aware: each backend
exposes its sampler list; the endpoint merges (SDXL + Flux). UI dropdown
populated from `/api/samplers` (already does).

**D8. Metadata** — no structural change. Flux images write the same PNG fields
(Steps, Sampler="Euler", CFG, Seed, Size, Model, Lora). `CFG scale` carries the
distilled guidance for dev.

**D9. Config** — `ensure_dirs` adds `models/flux` (+ `loras/flux`). `infra/`
already exists.

## File change list

**New**
- `src/imagagen/flux_backend.py` — `FluxBackend(ModelBackend)`.
- `src/imagagen/flux_transformer.py` — `IntegratedFluxTransformer2DModel` port.
- `src/imagagen/gguf_ops.py` — GGUF load + per-op GPU dequantize.

**Modified**
- `pyproject.toml` — add `gguf` (and `einops` if not hand-rolled).
- `src/imagagen/models.py` — family detection, `.gguf`, backend dispatch.
- `src/imagagen/kdiff.py` — add `sample_euler`.
- `src/imagagen/app.py` — family-aware `/api/samplers`.
- `src/imagagen/config.py` — `ensure_dirs` (+ `models/flux`).
- `src/imagagen/sdxl_backend.py` — `from_config` picks the first model of any
  family (currently hardcodes SDXL discovery order).
- `src/imagagen/tokenize.py` — (optional) Flux token count (T5 512 cap).

## Open items / risks

- **Exact flux sigma subsampling** (how diffusers/Forge picks N steps from the
  shifted schedule) — verify against the reference before locking the
  schedule.
- **T5-XXL fp8**: confirm transformers loads `t5xxl_fp8_e4m3fn.safetensors`
  and runs in fp8 storage with manual cast (reference uses its own
  `IntegratedT5`). bf16 fallback is 9.8 GB — won't co-reside, so the fp8 path
  is the practical one.
- **VRAM fit on 16 GB**: sampling = transformer 6.75 GB (quantized) + VAE +
  activations; encoders 4.9 GB offloaded to CPU. Verify peak on GPU 1.
- dev-vs-schnell detection + default guidance (3.5).
- Confirm transformer dims/`axes_dim`/`theta` against the GGUF tensor shapes
  and the reference bundled config.

## Verification

- Unit: GGUF load (transformer builds, weights bake, forward finite bf16);
  sigma schedule matches the reference for a given resolution/steps; x0 /
  x_in conventions.
- Integration: generate 512×512 (with the provided LoRA) → finite, valid PNG,
  metadata round-trip, LoRA applied; inpaint masked / not-masked.
- Cross-check vs reference (won't be pixel-identical — different
  environment — but structurally valid, same architecture).
- Monitor peak VRAM on GPU 1 (16 GB).
