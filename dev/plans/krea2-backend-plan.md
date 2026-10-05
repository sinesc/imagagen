# Krea 2 backend plan

Add Krea 2 (K2) as the fourth model family alongside SDXL, Flux and Z-Image.
Assets are in place: `models/krea2/krea2_Int8Extended.safetensors` (13.5 GB,
int8-quantized transformer), `infra/qwen_3vl_4b.safetensors` (4.8 GB fp8
Qwen3-VL-4B text encoder), `infra/qwen_image_vae.safetensors` (0.25 GB VAE),
`loras/krea2/rustic-ambience.safetensors`. References: the official
implementation `dev/implementations/krea-2` (inference.py / mmdit.py /
sampling.py / encoder.py / autoencoder.py) and the ComfyUI checkout
(`comfy/ldm/krea2/model.py`, `comfy/text_encoders/krea2.py`,
`comfy/utils.krea2_to_diffusers`, `comfy/latent_formats.Wan21`).
Checkpoint-layout dumps from the earlier look: `dev/scratch/inspect_krea2.py`,
`qkv.py`, `qkv2.py`, `survey.py`.

## Architecture (verified against the shipped state dicts)

**Transformer** (`krea2_Int8Extended.safetensors`, 878 keys, no key prefix):
`blocks.N.*` (28), `first`, `last`, `tmlp`, `tproj`, `txtfusion`, `txtmlp`.
SingleStreamDiT config (reference `inference.py`): features 6144, tdim 256,
txtdim 2560, heads 48, kvheads 12 (head_dim 128), mlp multiplier 4 (dim
16384, SwiGLU), layers 28, patch 2, channels 16, txtlayers 12, txtheads 20,
txtkvheads 20, RoPE theta 1e3 with 3-axis dims [32, 48, 48] (reuse
`flux_transformer.EmbedND`/`apply_rope`).

- Quantization: every `blocks.N` projection is `int8_tensorwise` with
  **per-row** scales (`<key>.weight_scale`, f32 `[out,1]`) plus a
  `<key>.comfy_quant` U8 marker `{"format":"int8_tensorwise","convrot":true,
  "convrot_groupsize":256,"per_row":true}`. Everything else (txtfusion,
  txtmlp, tmlp, tproj, first, last, all norms, `mod.lin` params) is bf16.
- **ConvRot**: the stored int8 weights are Hadamard-rotated along
  in_features in groups of 256. Runtime must rotate the activation by the
  same matrix: `x_rot = x_grouped @ H`, `out = F.linear(x_rot, W_dequant)`
  (H symmetric orthogonal, so the rotations cancel). Port the exact
  construction from `comfy_kitchen/tensor/int8_utils.py` (h4 =
  [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], kron-recursed to 256,
  divided by sqrt(256)); cache H per device/dtype.
- New `Int8Linear(QuantLinear)` in `krea2_transformer.py`: `weight` int8
  `[out,in]` + `weight_scale` f32 `[out,1]` stay resident; per-op dequantize
  `W.to(bf16) * scale` (max transient 16384x6144x2B = 200 MB) and apply the
  Hadamard rotation to the input (in_features 6144/16384/2560 are all
  divisible by 256). Inherit `set_lora_deltas`/`set_lora_bias` from
  `gguf_ops.QuantLinear`. Non-quantized linears (txtfusion etc.) use a small
  bf16 `Linear` that also accepts additive LoRA weight deltas (the sample
  LoRA targets txtfusion attn/ff).
- Block (`comfy/ldm/krea2/model.py` port): AdaLN-single
  `DoubleSharedModulation` (`mod` is a raw Parameter `[6*6144]` added to the
  tvec projection), RMSNorm with the **(1 + scale)** zero-centered
  convention (`F.rms_norm(x.float(), weight=scale+1)`), GQA attention with
  per-head q/k RMSNorm (plain weight multiply, checkpoint stores
  `qknorm.qnorm.scale`/`knorm.scale` [128]), sigmoid output gate
  (`wo(out * sigmoid(gate(x)))`), SwiGLU MLP. TextFusionTransformer: 2
  layerwise blocks run over `(B*12, seq, 2560)`, `projector` Linear(12->1)
  collapses the tap axis, 2 refiner blocks; `txtmlp` = RMSNorm +
  Linear(2560->6144) + GELU(tanh) + Linear(6144->6144). `tmlp` =
  Linear(256->6144) + GELU(tanh) + Linear(6144->6144) over
  `timestep_embedding(t, 256)` (the `flux_transformer` helper, time_factor
  1000 — the reference passes raw t in [0,1] and multiplies inside);
  `tproj` = GELU(tanh) + Linear(6144 -> 6*6144). `last` = RMSNorm +
  SimpleModulation (`mod` Parameter `[2,6144]`) + Linear(6144 -> 64).
- Sequence: text tokens (pos 0,0,0) then image tokens (pos 0, h, w),
  patchified 2x2; `first` Linear(64 -> 6144); output sliced to the image
  tokens and unpatchified. No attention mask (batch 1, no padding).
- **Output convention**: the model returns the flow velocity (ComfyUI FLUX
  type / CONST convention): `x0 = x - sigma * v`, exactly the Z-Image
  reading. Assert empirically in the smoke script (velocity of a noised
  clean latent, the `dev/smoke_zimage.py` pattern).

**Text encoder** (`infra/qwen_3vl_4b.safetensors`, keys `model.*`):
Qwen3-VL-4B text tower — 36 layers, hidden 2560, 32 q / 8 kv heads,
head_dim 128 with per-head q/k RMSNorm (eps 1e-6, plain weight multiply),
RoPE theta **5e6** over the full head dim (text-only input: the interleaved
mRoPE degenerates to 1D positions), intermediate 9728 (SiLU-gated), vocab
151936, no lm_head. Projections are plain `F8_E4M3` with **no** scale
tensors (the `t5_encoder.Fp8Linear` cast-is-dequantize pattern); embedding
and norms bf16. Skip all `model.visual.*` keys (vision tower unused; do not
build it). Hand-roll `krea2_encoder.py` as a Qwen3-VL variant of
`qwen3_encoder.py` (same loader shape: `param.data = t`, fp8 stays fp8).

- Conditioning = the 12-layer tap stack `hidden_states[2,5,8,...,35]` (HF
  numbering, `hidden_states[0]` = embedding, no final norm applied),
  stacked per token and flattened to `(1, seq, 12*2560 = 30720)`; the DiT's
  txtfusion consumes it reshaped to `(B, seq, 12, 2560)`.
- Prompt template: the exact KREA2 conditioning template from
  `comfy/text_encoders/krea2.py` (system line "Describe the image by
  detailing the color, shape, size, texture, quantity, text, spatial
  relationships of the objects and background:" + user prompt + assistant
  opener; copy the string verbatim when implementing). Strip the leading
  system + user-opening tokens: cut at the second chat-start marker
  (include the following "user\n" tokens when present), exactly like
  `Krea2TEModel.encode_token_weights`. No padding (variable length, batch
  1, mask dropped when all-ones).
- Tokenizer: Qwen2 BPE (ComfyUI runs Qwen3-VL on its `qwen25_tokenizer`):
  commit `res/tokenizer_krea2/` (vocab.json, merges.txt,
  tokenizer_config.json) copied from the ComfyUI checkout, loaded via
  `transformers.Qwen2Tokenizer` (the zimage pattern). Never fetch from the
  hub.
- Prompt-level encoding cache: `shared_enc_cache` exactly like Z-Image
  (`_get_context`/`_encode_prompts`/`pre_encode`), including the
  hit-path encoder offload.

**VAE** (`infra/qwen_image_vae.safetensors`, 194 bf16 keys): the Qwen-Image
VAE (Wan-architecture, f8, 16 channels) in ComfyUI Wan naming (`conv1`,
`conv2`, `encoder.downsamples.N.{residual,resample,shortcut,time_conv}`,
`encoder.middle.N.*`, `encoder.head.{0,2}`, mirrored on `decoder.*`,
LayerNormConv `gamma` norms). Remap 1:1 to the diffusers
`AutoencoderKLQwenImage` state dict (also 194 keys): `conv1 ->
encoder.conv_in`, `conv2 -> decoder.conv_in`, `encoder.conv1 ->
decoder.conv_in`, `encoder.head.0/2 -> decoder.norm_out/conv_out`,
`decoder.head.0/2 -> encoder.norm_out/conv_out`... build the full table by
pairing the sorted normalized key lists (verified equal in shape/count) and
commit it in `krea2_vae.py` (small pure module, unit-tested). Load with
`AutoencoderKLQwenImage` built from a committed
`res/qwen_image_vae.config.json` (class defaults already carry the correct
`latents_mean`/`latents_std`, z_dim 16, base 128 — dump the default config
to the res/ file so nothing is fetched); bf16, `enable_slicing()`, Flux's
2-MP tiling threshold (Wan decoder is non-local like the Flux one).
- Latent normalization is **per-channel mean/std**, not scaling/shift:
  solver-space latent `z = (vae_latent - mean) / std`; decode applies
  `vae_latent = z * std + mean` (reference `autoencoder.py`).

## Sampling

Flow matching, plain Euler (`kdiff.sample_euler`, x0 contract, sigmas end
in 0). New `kdiff.get_sigmas_krea2(n, image_seq_len, device, y1=0.5,
y2=1.15)` next to `get_sigmas_flux`: `mu` linear in image-seq-len between
(x1, y1) = (256, 0.5) and (x2, y2) = (6400, 1.15) (the reference's
minres 256 / maxres 1280 px at 16 px per token), then the same exponential
time shift as `get_sigmas_flux` over `linspace(1, 1/n, n)` + append 0.
`SAMPLERS = {"Euler": kdiff.sample_euler}`, `SCHEDULERS = {}`.

- **CFG semantics**: the reference computes `v = cond + g*(cond - uncond)`,
  i.e. standard CFG at `w = g + 1`. Implement the standard form used by all
  imagagen families (`x0 = x0u + cfg*(x0c - x0u)` applied on the x0
  estimates) and express the built-ins in standard terms. cfg <= 1.0 skips
  the uncond pass (and the negative encode).
- "Model default" sentinels, per checkpoint class (a `turbo` flag on the
  model config entry, mirroring Flux's `distilled` flow):
  - raw (default): **28 steps**, standard **CFG 5.5** (reference default
    cfg 4.5), resolution-derived mu;
  - turbo: **8 steps**, **CFG 1.0** (no CFG), mu pinned to **1.15**
    (reference `--mu 1.15` recommendation).
- Size snap to 16 px (8x VAE + 2x2 patchify, as Flux/Z-Image).
- txt2img prior: unit fp32 noise (flow state at sigma 1.0).
- img2img/inpaint: identical to the Z-Image backend (CONST noise scaling
  `x = s0*noise + (1-s0)*init`, `t_enc = int(min(denoise,0.999)*steps)`
  truncation, per-step + final latent lock, 50%-binary lock mask, soft
  inpainting supported). SAG ignored.

## LoRA

`loras/krea2/*.safetensors` (checked: rustic-ambience, 512 keys, bf16,
rank 16, no alpha tensors -> scale 1.0) use the diffusers naming under a
`transformer.` prefix: `transformer_blocks.N.attn.to_q|to_k|to_v|to_gate|
to_out.0.lora_{A,B}.weight`, `transformer_blocks.N.ff.gate|up|down`,
`text_fusion.layerwise_blocks.N.*` / `text_fusion.refiner_blocks.N.*`.
New `krea2_lora.py`: parse + map via the inverse of
`comfy.utils.krea2_to_diffusers` (port its table): `attn.to_q -> attn.wq`,
`to_k -> wk`, `to_v -> wv`, `to_gate -> attn.gate`, `to_out.0 -> attn.wo`,
`ff.gate/up/down -> mlp.gate/up/down`, `transformer_blocks.N -> blocks.N`,
`text_fusion.* -> txtfusion.*`. Runtime additive deltas exactly as Z-Image:
`flux_lora.merge_module_deltas` (one GEMM pair per module),
`Int8Linear.set_lora_deltas` for the quantized blocks, the delta-capable
bf16 Linear for txtfusion targets; `_apply_loras` all-or-nothing, cached on
`_active_loras`, validation before mutation, registration under
`_PARAM_INIT_LOCK`, zero-weight LoRAs skipped.

## VRAM (16 GB cards)

Resident: transformer ~13.5 GiB (int8 + f32 scales + bf16 sections) + VAE
~0.25 GiB. The 4.8 GiB fp8 encoder can **never** co-reside -> always swap:
park the transformer on the CPU while the encoder runs (the Z-Image
`_encode_prompts` swap path, prompt cache makes it once per batch), swap
back after. VAE decode OOM-retry parks the transformer (Flux pattern).
`empty_cache()` at end of `__init__` and after every generate.
`text-encoder-device = cpu` keeps the encoder in RAM permanently.
`can_park = True` (full device moves under `_PARAM_INIT_LOCK`); parked RAM
footprint ~18.5 GB (transformer + encoder + VAE) — fine on the 62 GB host,
note it in the module docstring.

## Wiring

- `models.py`: `("krea2", ".safetensors")` in `_FAMILIES`; lazy-import
  dispatch arm in `_new_backend`.
- `config.py`: `ensure_dirs` adds `models/krea2` (loras/krea2 discovery is
  generic).
- `backends.py`: `is_krea2` flag on `ModelBackend`; `turbo: bool = False`
  on `GenerateRequest` and `turbo: bool | None = None` on `GenerateResult`
  (effective mode, mirroring `distilled`).
- `model_configs.py`: `ModelConfig.turbo: bool = False` (+ to_dict/
  validation/default entry).
- `app.py`: `/api/samplers` krea2 arm; `POST /api/generate` carries
  `turbo=entry.turbo` (config property, same flow as `distilled`).
- `queue.py`: pass `turbo` through to the request; metadata needs no new
  lines (steps/cfg are family-agnostic; effective values come from the
  result).
- `static/app.js`: model-config dialog gets a "turbo (8-step, no CFG)"
  checkbox shown when `entry.family === "krea2"`; samplers/loras refetch
  already keys off family.
- Update `models/krea2/readme-folder-names.txt` context and
  `infra/readme-file-names.txt` (already lists the two infra files).

## Files

- `src/imagagen/krea2_transformer.py` — SingleStreamDiT port (Int8Linear +
  ConvRot Hadamard, bf16 DeltaLinear for txtfusion, RMSNorm (1+scale),
  EmbedND/apply_rope/timestep_embedding from flux_transformer).
- `src/imagagen/krea2_encoder.py` — Qwen3-VL text tower + fp8 linears +
  `load_krea2_encoder(path, device)` + template/prefix-strip + tap stack.
- `src/imagagen/krea2_vae.py` — ComfyUI-Wan -> diffusers key remap table +
  `load_krea2_vae(config)` (AutoencoderKLQwenImage from
  `res/qwen_image_vae.config.json`).
- `src/imagagen/krea2_lora.py` — parse/map (port of krea2_to_diffusers).
- `src/imagagen/krea2_backend.py` — `Krea2Backend(ModelBackend)`,
  `SAMPLERS`/`SCHEDULERS`, `from_config`, `is_krea2 = True`.
- `kdiff.py` — `get_sigmas_krea2`.
- `res/tokenizer_krea2/`, `res/qwen_image_vae.config.json`.
- `dev/smoke_krea2.py`, `tests/unit/test_krea2*.py`,
  `tests/integration/test_krea2_api.py`.

## Verification

1. Tier 1: `get_sigmas_krea2` math (mu interpolation vs reference
   `timesteps()` values), VAE key remap (194 keys, shapes match the
   diffusers state dict), LoRA key mapping (rustic-ambience fixture),
   ConvRot round-trip (port `int8_utils._build_hadamard`/`_rotate_weight`:
   quantize->dequantize->rotate-cancel error small), turbo/raw sentinel
   resolution, family wiring integration tests (MockBackend stands in).
2. Tier 2 (`dev/smoke_krea2.py`, GPU1): velocity-sign assertion; same-seed
   determinism; 512/1024 txt2img sanity (raw 28/5.5 and turbo 8/1.0);
   img2img denoise behavior; masked inpaint protected region (compare
   against a fully-protected run — non-local decoder); LoRA weight-0 ~=
   baseline, high weight shifts, set switching no leak, failing set leaves
   clean state; cancel mid-run; unload/load park round-trip (mem_get_info
   before/after); VRAM after generate ~= resident weights; encoder swap
   logs.
3. Structural parity vs the reference (`dev/implementations/krea-2`):
   schedule values, CFG mapping (standard = reference + 1), template/prefix
   strip, tap indices. No Forge reference exists for this family.

## Commits (incremental, mirroring the Flux/Z-Image rollout)

1. `krea2: transformer (int8+convrot) + Qwen3-VL encoder + VAE (txt2img)`
2. `krea2: img2img/inpaint`
3. `krea2: LoRA support (runtime additive deltas)`
4. `krea2: family wiring (registry, API, UI, turbo flag)` + tests
5. fixes as found during verification
