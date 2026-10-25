# Z-Image (Turbo) backend plan

Add Z-Image as the third model family alongside SDXL and Flux. Assets are in
place: `models/zimage/intorealism_zitV40.safetensors` (6.1 GB, fp8
transformer), `infra/zit_ae.safetensors` (VAE), `infra/zit_encoder.safetensors`
(4.4 GB fp8 text encoder), `loras/zimage/ZIT-shine.safetensors`.
Reference: the ComfyUI checkout in `dev/implementations/ComfyUI` (Z-Image is
implemented there as a `Lumina2`/`NextDiT` model with the Qwen3-4B encoder).

## Architecture (verified against the shipped state dicts)

**Transformer** (`intorealism_zitV40.safetensors`, keys `model.diffusion_model.*`,
all weights `F8_E4M3` with *no* scale tensors → uniform fp8, per-op bf16 cast
like Flux FP8):

- NextDiT / "Simplified MM-DiT": patch 2×2 over 16-ch latents (8× VAE
  downsample → 16-px grid, same snap as Flux).
- dim 3840, n_layers 30, 2 noise_refiner + 2 context_refiner blocks,
  30 heads / 30 kv (head_dim 128), qk_norm (RMSNorm(128), torch default
  eps 1e-6), block norms RMSNorm(3840, eps 1e-5), ffn 3840→10240→3840
  (SiLU-gated, `ffn_dim_multiplier 8/3`, multiple_of 256), no biases except
  the fused qkv has none and adaLN/final/biases per the file.
- `z_image_modulation`: block adaLN = `Linear(256 → 4·3840)` (no SiLU),
  gates applied as `tanh(gate)`; `FinalLayer` = LayerNorm(eps 1e-6, no
  affine) + `SiLU + Linear(256 → 3840)` + Linear(3840 → 64).
- Timestep: `t = 1 − sigma` (flow time flip), `time_scale = 1000`,
  `TimestepEmbedder = Linear(256→1024) SiLU Linear(1024→256)` over the plain
  sinusoidal embedding (max_period 10000, **no** ×1000 factor — that is
  the `time_scale`, not `time_factor`).
- Text: `cap_embedder = RMSNorm(2560, eps 1e-5) + Linear(2560 → 3840)`;
  cap tokens padded with `cap_pad_token` to a multiple of **32**
  (`pad_tokens_multiple`, set because `cap_pad_token` exists), img tokens
  padded with `x_pad_token`.
- RoPE: `EmbedND(head_dim 128, theta 256.0, axes [32, 48, 48])` (NOT 10000/
  [16,56,56] — ComfyUI `model_detection` sets these for dim 3840).
  Cap pos ids: t = arange+1; img pos ids: t = cap_len+1, h/w aranges.
- Context refiner: modulation=False. Noise refiner + main blocks:
  modulation=True.
- No `clip_text_pooled_proj` / `siglip` / `time_text_embed` in this
  checkpoint (omni features absent — build without them).
- **Output convention**: `_forward` returns `−img`. ComfyUI FLOW models run
  under the `CONST` convention (`calculate_denoised = x − sigma·out`,
  `noise_scaling = sigma·noise + (1−sigma)·latent`), i.e. the model output
  is the **velocity**. So `v = −transformer_out` and `x0 = x − sigma·v =
  x + sigma·transformer_out`. The sign (`−img` = v vs = x0 vs = −x0) is
  resolved *empirically* before wiring the sampler: feed a VAE-encoded clean
  latent at flow time t, compare candidate velocities against the known
  `e − x0` (cos similarity). The smoke script asserts it.

**Text encoder** (`zit_encoder.safetensors`): Qwen3-4B, 36 layers, hidden
2560, 32 q / 8 kv heads, head_dim 128, intermediate 9728 (SiLU-gated),
RMSNorm eps 1e-6, RoPE theta 1e6 (full head dim), per-head q/k RMSNorm,
no biases, vocab 151936, final RMSNorm present. Weights: embedding + norms
BF16; every projection F8_E4M3 with per-linear `scale_weight` (fp32) and
`scale_input` (bf16) — the Kijai scaled-fp8 format. Per-op:
`y = F.linear(x·scale_input, W.to(bf16)·scale_weight)` (the
`t5_encoder.Fp8Linear` pattern, plus the two scales).

- Prompt template: the exact `llama_template` string from
  `comfy/text_encoders/z_image.py` (chat-style `user` / `assistant`
  markers around the prompt — read it verbatim from that file when
  implementing), Qwen2 tokenizer, no BOS/EOS, pad id 151643,
  uncapped length.
- Output: the **penultimate** decoder layer's hidden state
  (`hidden_states[-2]`, no final norm — ComfyUI runs
  `Qwen3_4B(layer="hidden", layer_idx=-2, layer_norm_hidden_state=False)`),
  bf16, shape (1, L, 2560).

**VAE**: `zit_ae.safetensors` is **byte-identical to `flux_ae.safetensors`**
(same md5) → load `infra/zit_ae.safetensors` with the existing
`res/flux_ae.config.json` config, bf16, `enable_slicing()`, same scaling
(0.3611) / shift (0.1159), same 2-MP tiling threshold and 512/64 tiles.

## Sampling

Flow matching, single pass (Z-Image-Turbo is an NFE-1 model: no CFG, no
guidance embedding, no negative prompt — `request.cfg`/`distilled` are
ignored; effective cfg 1.0 in `GenerateResult`).

- Sigmas (ComfyUI `simple_scheduler` on the Z-Image flow buffer, which for
  the standard step counts equals the official
  `linspace(1, 1/N, N)` + linear-SNR shift):
  `sigma(t) = 3t / (1 + 2t)` over `t = (1000 − int(i·1000/N))/1000`,
  `i = 0..N−1`, append 0.0 → `kdiff.get_sigmas_zimage(n)` (new, next to
  `get_sigmas_flux`). `SAMPLERS = {"Euler": kdiff.sample_euler}` (x0
  contract, sigmas end in 0), `SCHEDULERS = {}` (internal, like Flux).
- "Model default": **steps 8** (official Turbo default), cfg → 1.0.
- txt2img: unit fp32 prior (flow state at sigma 1.0 is the noise itself).
- img2img/inpaint: `CONST` noise scaling `x = s0·noise + (1−s0)·init`;
  mask lock exactly as Flux (same VAE → harden the sampling mask to the
  50%-level binary, feather only in the pixel composite); denoise
  truncation `t_enc = int(min(denoise, 0.999)·steps)`; per-step + final
  latent lock, soft inpainting supported (same code path as Flux).
- SAG: not supported (ignored, as the contract allows).

## LoRA

`loras/zimage/*.safetensors` use the official Z-Image-Turbo naming:
`diffusion_model.layers.N.attention.to_q|to_k|to_v|to_out.0.lora_{A,B}.weight`,
`diffusion_model.layers.N.feed_forward.w1|w2|w3.lora_{A,B}.weight`,
`diffusion_model.layers.N.adaLN_modulation.0.lora_{A,B}.weight` (rank 32,
bf16, no alpha tensors → scale 1.0, layers 0–29 only). Our transformer
fuses q/k/v into one `qkv` GEMM → row-slice mapping, exactly the Flux
pattern: `to_q → qkv[0:3840)`, `to_k → qkv[3840:7680)`,
`to_v → qkv[7680:11520)`, `to_out.0 → attention.out` (full), w1/w2/w3 →
`feed_forward.w*`, `adaLN_modulation.0` → `adaLN_modulation.0`.
New `zimage_lora.py` (parse + map, `ZImageLoraSpec`) reusing
`flux_lora.merge_module_deltas` and `QuantLinear.set_lora_deltas`
(runtime additive deltas, cached per (name, weight) set, all-or-nothing
under `_PARAM_INIT_LOCK`).

## VRAM (16 GB cards)

Resident: transformer ~6.2 GB (fp8) + Qwen3 ~4.4 GB (fp8) + VAE ~0.67 GB
(bf16) ≈ 11.3 GB. Text encoder co-residency is decided at load by the Flux
`mem_get_info` check; when it doesn't fit, the transformer is swapped to
the CPU for the encode (per-generate swap, Forge-style).
`text-encoder-device = cpu` keeps it in RAM permanently.
`empty_cache()` after every generate; VAE decode OOM-retry parks the
transformer (Flux pattern). `can_park = True` (full CPU moves under
`_PARAM_INIT_LOCK`).

## Files

- `src/imagagen/zimage_transformer.py` — `ZImageTransformer` (NextDiT port:
  fused qkv `QuantLinear` scope like Flux, RMSNorm via `F.rms_norm`, SDPA,
  RoPE/`EmbedND` copied from `flux_transformer`, `z_image_modulation`
  blocks, pad tokens).
- `src/imagagen/qwen3_encoder.py` — Qwen3-4B encoder + scaled-fp8
  `Fp8Linear` + `load_zit_encoder(path, device)`; tokenizer from
  `res/tokenizer_zimage/` (copy of ComfyUI's `qwen25_tokenizer`: merges.txt,
  tokenizer_config.json, vocab.json).
- `src/imagagen/zimage_lora.py` — parse/map as above.
- `src/imagagen/zimage_backend.py` — `ZImageBackend(ModelBackend)`,
  `SAMPLERS`/`SCHEDULERS`, `from_config(config)`, `is_zimage = True`.
- `kdiff.py` — `get_sigmas_zimage`.
- Wiring: `models.py` (`("zimage", ".safetensors")` in `_FAMILIES`, lazy
  dispatch arm), `config.py` (`ensure_dirs`: `models/zimage`,
  `res/tokenizer_zimage`), `app.py` (`/api/samplers` arm), `backends.py`
  (`is_zimage` flag), `queue.py`/`metadata.py` (no new lines — cfg is
  family-agnostic), `static/app.js` (disable the CFG input for
  `entry.family === "zimage"`; samplers/loras refetch already works).
- `dev/smoke_zimage.py` — Tier-2 checks (seed determinism, output-convention
  assertion, img2img/inpaint, LoRA, cancel, park/load, VRAM).
- `tests/unit/test_zimage*.py` — schedule math, LoRA mapping, request
  defaults; integration test for the family wiring.

## Verification

1. Tier 1: new unit + integration tests; full targeted suite green.
2. Tier 2 (`dev/smoke_zimage.py`, GPU): the empirical velocity-sign check;
   same-seed determinism; 512²/1024² txt2img sanity (on-prompt, LoRA shifts
   the image at high weight); img2img denoise behavior; masked inpaint
   protected region (Flux-VAE caveat: compare against a fully-protected
   run); LoRA set switching; cancel; park/load; VRAM after generate ≈
   resident weights.
3. Eyeball against the ComfyUI reference conventions (no ComfyUI runtime
   available offline — its venv has no torch).

## Commits (incremental, mirroring the Flux rollout)

1. `zimage: transformer + Qwen3 encoder + VAE load (txt2img)`
2. `zimage: img2img/inpaint`
3. `zimage: LoRA support (runtime additive deltas)`
4. `zimage: family wiring (registry, API, UI)` + tests
5. fixes as found during verification
