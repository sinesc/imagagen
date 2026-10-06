"""SDXL backend: diffusers pipeline + vendored k-diffusion sampling loop.

Ships four samplers (Euler a, DPM++ 2M, DPM++ 2M SDE, LCM) and three sigma
schedules (Karras, Exponential, Polyexponential); both are chosen from small
registries so Phase 5's plugin system can grow them without touching this
file. LCM follows Forge's LCMCompVisDenoiser: the sigma range is limited to
the 50 original LCM timesteps (19, 39, ..., 999) of the beta schedule, every
sigma is quantized to that grid before reaching the UNet, and the x0 estimate
gets the CompVis rescaling (see _LcmSchedule).

SDXL specifics (matching the k-diffusion / A1111 contract):
- sigma_data = 1.0; the sigma range is the model's actual beta-schedule
  range (SDXL: ~[0.03, ~14.6]), not the clamped k-diffusion [0.1, 10] --
  this is the A1111/Forge "rescaled betas" behavior and is what gives the
  characteristic look (the [0.1, 10] clamp renders flatter and grainier)
- the UNet expects *training* timesteps (0..999); each k-diffusion sigma is
  mapped to the nearest beta-schedule index in log space
  (argmin |log sigma - log beta_sigma_i|) -- A1111/Forge's k-diffusion
  ModelSamplingDiscrete.timestep, and the same as diffusers' _sigma_to_t.
  The ComfyUI-style formula t = sigma/sqrt(sigma^2+1)*999 was tested and
  found to break few-step distilled LoRAs (e.g. DMD2): it shifts the t
  values in the mid-noise range away from the training schedule
- the UNet expects a *unit-variance* input at every noise level, so the
  model input is scaled by 1/sqrt(sigma^2 + 1) (exactly what diffusers'
  EulerDiscreteScheduler.scale_model_input does)
- the UNet predicts EPSILON; the sampler wants an x0 estimate, so the model
  wrap returns x0 = x - sigma * eps (the exact inverse of diffusers' Euler)
- txt2img initial noise is randn * sigma_max (the solver state x is the
  noisy sample, not the unit-variance prior); img2img/inpaint instead start
  from the init latent plus the noise at the truncated start sigma
  (Forge's noise_scaling, max_denoise=False)
- sampling runs in the VAE-scaled latent space; divide by
  vae.config.scaling_factor (0.13025 for SDXL) before decode
- prompts are encoded Forge/A1111-style: split into 75-token windows (77
  incl. BOS/EOS), encode each, and concatenate the hidden states along the
  sequence dim instead of diffusers' hard 77-token truncation, so long prompts
  (e.g. a style stacked on top of a keyword-heavy prompt) lose no content
- token weights (Forge/A1111 "emphasis", token_weights.py): (green:1.2)
  scales the CLIP embeddings of "green"'s tokens by 1.2 (bare (x) = 1.1,
  [x] = 1/1.1, nesting multiplies); the scaling is plain per-token
  multiplication of the window hidden states (Forge's "No norm" emphasis),
  applied to both the prompt and the negative prompt

Precision (matching Forge's kdiff path exactly):
- the solver state x runs in fp32: Forge's model call casts the UNet output
  to fp32, so calculate_denoised (x - sigma * eps) promotes x to fp32 from
  step 1 on, and the whole DPM++ update then stays fp32. Running the
  solver in fp16 (10-bit mantissa) measurably flattens detail/contrast.
- only the UNet itself runs in fp16 (input cast right before the call,
  like Forge's calculate_input + .to(computation_dtype))
- the improved VAE (infra/sdxl_vae.safetensors, ft-MSE) is swapped in when
  present: the stock SDXL VAE baked into the checkpoint overflows to NaN at
  high-magnitude latents (the classic "black image" bug), while the improved
  VAE is fp16-stable and renders color better. All reference images were
  decoded with the improved VAE. The checkpoint VAE is the fallback when the
  file is absent.
- the VAE decodes in bf16 (Forge's default on sm_80+ GPUs): bf16 has the
  same 8-bit mantissa as fp16 but 8x the dynamic range

VRAM options (imagagen.ini, all optional, all off by default):
- unet-quantization = nf4: bitsandbytes NF4 4-bit weights for the UNet.
  Saves ~3.2 GiB (768² peak 6.6 -> 3.4 GiB) at the same speed; the output
  is usable but softer (quantization error diverges from fp16 chaotically).
  (torchao's int8 was rejected: int8_weight_only overflows in fp16
  accumulation -> NaN/black images, and int8 dynamic activations are
  4x slower. bitsandbytes has fused 4-bit kernels.)
- text-encoder-device = cpu: keeps the CLIP encoders in system RAM
  (saves ~1.4 GiB VRAM permanently); CPU fp16 encoding is only ~0.6 s per
  prompt, so the speed cost is ~1 s per generation.
- vae-tiling = on: force tiled VAE decode/encode at every size. Off
  (default) is automatic: tiled above ~4 MP of image (the untiled path
  allocates full-resolution 32-channel fp32 conv buffers and OOMs on a
  16 GB card past ~5 MP; at 3440x1440 tiling is faster, peaks at 7.4 GiB
  instead of 13.3, and differs by <= 1/255 per pixel), untiled below.

LoRAs (Phase 4):
- applied via diffusers' `load_lora_weights` / `set_adapters`, which
  internally does PEFT `inject_adapter_in_model` (the plan's "PEFT is the
  robust choice") plus diffusers' LoRA weight-format translators, so the
  common on-disk layouts (Civitai/Kohya ``lora_down``/``lora_up``, legacy
  A1111 ``lora_unet_*``) all load without a manual key-mapping layer.
- multiple LoRAs at per-LoRA weights: one injected adapter per LoRA, then
  ``set_adapters([tags], [weights])`` scales each.
- transformers-5 te-key fix: diffusers' A1111 conversion emits
  ``text_model.encoder...`` keys (the transformers-4 CLIPTextModel layout),
  but transformers 5 flattened CLIPTextModel, so key-to-module matching
  finds nothing and loading crashes. ``_lora_state_dict`` re-prefixes the
  text-encoder keys against the actual model layout.
- the active (name, weight) set is cached: unchanged between generations
  it is not re-injected; switching sets unloads all adapters first.

Self attention guidance (SAG, Forge builtin extension; see sag.py):
- per step (cfg > 1 only) the unconditional pass' mid-block self-attention
  is captured via forward hooks on the layer's to_q/to_k (the UNet's own
  attention runs untouched, so a generation with SAG installed but not
  applied is bit-identical to one without it);
- the attention's per-key mass (heads-averaged, query-summed, recomputed
  chunked in fp32) forms a mask; the unconditional x0 is Gaussian-blurred
  where the mask is set (the "degraded" x0);
- the degraded x0 is re-noised to the current sigma, denoised with one
  extra unconditional UNet pass, and (degraded - re-denoised) * scale is
  added to the CFG result (x0 space, before the inpaint lock).
- cost: +1 UNet pass per step (~50% more model time); the mask is
  accumulated chunked over the query dim, so unlike the reference (which
  keeps the full (heads, n, n) attention matrix in VRAM, ~6 GB at
  1024x1024) the peak extra VRAM is a few tens of MB.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import warnings

import numpy as np

from .. import kdiff
from ..sag import SagRecorder, compute_sag_mask, create_blur_map, install_sag_hooks
from ..soft_inpaint import adaptive_change_alpha, latent_blend
from . import (
    _PARAM_INIT_LOCK,
    DEFAULT_CFG,
    GenerateRequest,
    GenerateResult,
    ModelBackend,
    format_loras,
    shared_enc_cache,
)

# fp16 pipelines are intentionally offloaded to CPU between generations to
# free VRAM; diffusers warns that fp16 does not compute on CPU -- it does not
# need to (CPU copies are storage only), so silence the warning once.
warnings.filterwarnings(
    "ignore", message="Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device.*"
)

# With two or more LoRAs, one peft adapter per LoRA is injected into the
# same UNet and then blended with set_adapters, so every injection after
# the first finds an existing peft_config. That multi-adapter state is the
# intended design (per-LoRA weights), not a bug.
warnings.filterwarnings(
    "ignore", message=r"Already found a `peft_config` attribute in the model.*"
)

# diffusers' vacuous "should be kept in float32" to(dtype) warning (emitted
# even when the keep-in-fp32 list is the empty []) is suppressed process-
# wide by config.load_config(), because the backend modules are imported
# lazily per family -- a suppression here would be missing in a session that
# only loads the other family.


# bitsandbytes' "No prebuilt binary for CUDA 12.9" notice (its
# nearest-prebuilt fallback is the correct behavior) is filtered process-
# wide by config.load_config(), for the same lazy-import reason as above.


# Every LoRA this project ships is UNet-only (no text-encoder part), so
# diffusers warns "No LoRA keys associated to CLIPTextModel..." at every
# application. It is correct for those files and diffusers itself says the
# warning is safe to ignore when the checkpoint had no text-encoder params.
class _NoLoraKeysFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "No LoRA keys associated to" not in record.getMessage()


logging.getLogger("diffusers.loaders.lora_base").addFilter(_NoLoraKeysFilter())

log = logging.getLogger(__name__)

SIGMA_DATA = 1.0

class _ModelSamplingDiscrete:
    """k-diffusion ModelSamplingDiscrete: t == sigma, x0 == xphi."""

    def __init__(self, fn, sigma_min: float | None = None,
                 sigma_max: float | None = None, sigma_data: float = 1.0):
        self.fn = fn
        self.sigma_data = sigma_data
        self.max_sigma = sigma_max if sigma_max is not None else sigma_data * 10.0
        self.min_sigma = sigma_min if sigma_min is not None else sigma_data * 0.1

    def __call__(self, x, sigma, **extra_args):
        # extra_args (e.g. the noise seed) are consumed by the samplers, not the model
        return self.fn(x, sigma)


SAMPLERS = {
    "Euler a": kdiff.sample_euler_ancestral,
    "DPM++ 2M": kdiff.sample_dpmpp_2m,
    "DPM++ 2M SDE": kdiff.sample_dpmpp_2m_sde,
    "LCM": kdiff.sample_lcm,
}

SCHEDULERS = {
    "Karras": kdiff.get_sigmas_karras,
    "Exponential": kdiff.get_sigmas_exponential,
    "Polyexponential": kdiff.get_sigmas_polyexponential,
}


class _LcmSchedule:
    """Forge's LCMCompVisDenoiser schedule for the LCM sampler.

    LCM-distilled models (e.g. gonzalomoPhotoDMD2XL6.0) are trained on only
    the 50 *original* timesteps of the 1000-step beta schedule (t = 19,
    39, ..., 999), so Forge's LCM path:
    - samples over that 50-slot sigma range [sigma(t=19), sigma(t=999)]
      instead of the full beta-schedule range;
    - maps every sigma to the nearest 50-slot timestep in log space and
      feeds that quantized t to the UNet (the model never sees the other
      950 noise levels);
    - rescales the x0 estimate with CompVis c_skip/c_out coefficients
      (sigma_data = 0.5, scaled timestep t*10 -- a no-op close to identity
      at the LCM timestep magnitudes, replicated for Forge parity).
    """

    N_ORIG = 50          # LCM original timestep count
    SKIP = 1000 // N_ORIG  # index stride between original timesteps

    def __init__(self, alphas_cumprod) -> None:
        # the 50 slots (schedule indices 19, 39, ..., 999); the first
        # (smallest t) carries the smallest sigma, so the order is ascending
        ac = alphas_cumprod[self.SKIP - 1::self.SKIP]
        self.sigmas = ((1.0 - ac) / ac).sqrt()
        self.log_sigmas = self.sigmas.log()
        self.min_sigma = float(self.sigmas[0])
        self.max_sigma = float(self.sigmas[-1])

    def timestep(self, sigma):
        """sigma -> its nearest 50-slot LCM timestep (t = idx * 20 + 19)."""
        log_sigma = sigma.float().log().unsqueeze(-1)
        idx = (log_sigma - self.log_sigmas.unsqueeze(0)).abs().argmin(-1)
        return (idx * self.SKIP + (self.SKIP - 1)).float()

    def x0_scaled(self, x, sigma, eps):
        """Forge LCMCompVisDenoiser.get_scaled_out on the standard x0.

        c_out * (x - sigma * eps) + c_skip * x with
        c_out = t' / sqrt(t'^2 + sd^2), c_skip = sd^2 / (t'^2 + sd^2),
        t' = 10 * quantized_t, sd = 0.5.
        """
        t_s = self.timestep(sigma) * 10.0
        c_out = t_s / (t_s ** 2 + 0.25).sqrt()
        c_skip = 0.25 / (t_s ** 2 + 0.25)
        return c_out * (x - sigma * eps) + c_skip * x


def sigma_to_timestep(sigma, log_sigmas):
    """k-diffusion sigma -> nearest beta-schedule training timestep (0..999).

    A1111/Forge k-diffusion ``ModelSamplingDiscrete.timestep`` maps each
    sigma to the index of the closest noise level of the model's beta
    schedule in log space (identical to diffusers' ``_sigma_to_t``). This
    mapping is load-bearing for few-step distilled LoRAs such as DMD2,
    which were trained against these exact timesteps; the ComfyUI-style
    formula ``sigma / sqrt(sigma^2 + 1) * 999`` shifts mid-range t values
    and turns them into mosaic garbage.
    """
    log_sigma = sigma.float().log().unsqueeze(-1)
    dists = (log_sigma - log_sigmas).abs()
    return dists.argmin(-1).float().clamp(0.0, 999.0)


def scale_lora_alphas(sd: dict, alphas: dict | None) -> dict:
    """Pre-scale every LoRA ``down`` weight by ``alpha / rank`` in place.

    diffusers' dict-input ``load_lora_weights`` path drops alpha metadata
    (already-converted keys skip the conversion that extracts alphas), so
    without this the injected adapters default to alpha == rank (scaling
    1.0) and run ``rank / alpha`` times too strong vs the A1111/Forge
    slider convention (``delta = slider * (alpha/rank) * B@A``).

    Handles the key namings produced by the conversion:
    ``....lora_A.weight`` / ``....lora.down.weight`` for regular layers, and
    the attention-processor naming diffusers 0.40 uses for transformer-block
    attention LoRAs, where the alpha key is ``<down_key>.alpha`` and the
    state-dict key *is* the down weight (no extra suffix). Without matching
    those, the layers silently keep unscaled weights and run ``rank / alpha``
    times too strong (DMD2: 8x).
    """
    for key, alpha in (alphas or {}).items():
        base = key[: -len(".alpha")]
        for down_key in (base + ".lora_A.weight", base + ".lora.down.weight", base):
            w = sd.get(down_key)
            if w is None or not w.shape[0]:
                continue
            sd[down_key] = w * (float(alpha) / w.shape[0])
            break
    return sd


def _quantize_unet_nf4(unet, device) -> int:
    """Replace every nn.Linear in ``unet`` with a bitsandbytes NF4 layer.

    Post-load quantization: diffusers 0.40 cannot load a raw fp16
    checkpoint through its quantization framework via from_single_file
    (PipelineQuantizationConfig hits a quant_method AttributeError; the
    legacy BitsAndBytesConfig expects pre-quantized state dicts). So we
    load fp16 and convert in place -- the same trick A1111/Forge use.
    The Params4bit constructor quantizes fp16 weights on the fly (the
    exact recipe diffusers' bnb_quantizer uses for unquantized tensors).

    Returns the number of converted layers.
    """
    import bitsandbytes as bnb
    import torch
    from bitsandbytes import nn as bnn

    def to_nf4(linear):
        l4 = bnn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=torch.float16,
            quant_type="nf4",
            device=device,
        )
        # quantization parameters taken from the freshly built Params4bit
        # template (quant_type, blocksize, storage, compress_statistics)
        tpl = {
            k: v
            for k, v in l4.weight.__dict__.items()
            if k not in ("data", "requires_grad")
        }
        l4.weight = bnb.nn.Params4bit(
            linear.weight.detach().cpu(), requires_grad=False, **tpl
        ).to(device)
        if linear.bias is not None:
            l4.bias.data = linear.bias.detach().to(device)
        return l4

    n = 0
    for name, module in list(unet.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        parent = unet
        *prefix, last = name.split(".")
        for p in prefix:
            parent = getattr(parent, p)
        setattr(parent, last, to_nf4(module))
        n += 1
    return n


def _model_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            if f.tell() >= 100 * 2**20:
                break
    return h.hexdigest()[:16]


class SdxlBackend(ModelBackend):
    can_park = True  # the pipeline moves between GPU and CPU losslessly

    def __init__(self, info, device, config=None) -> None:
        import os
        import warnings

        import torch
        from diffusers import StableDiffusionXLPipeline

        # diffusers' "Loading pipeline components..." tqdm bar: this is a
        # single synchronous load, not an interactive terminal, so the bar
        # only interleaves with the server log (generation progress goes
        # through the progress callback instead). Affects the whole
        # process, which is fine: nothing else in imagagen uses tqdm.
        from diffusers.utils.logging import disable_progress_bar

        disable_progress_bar()

        # Backstop in case load_config() was not used: keep any hub files
        # from_single_file fetches in the persistent project-local cache.
        if config is not None:
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(config.hf_cache_dir))

        self.model_id = info.id
        self.name = info.name
        self.model_hash = _model_hash(str(info.path))
        self.device = device  # chosen by ModelRegistry (ini/auto-pick)
        # Serializes generate()/unload(): the pipeline is not thread-safe.
        # The queue runs one worker per GPU and workers can pick tasks for
        # the same backend; concurrent .to()/forward on the same module
        # corrupts CUDA state and segfaults the whole process.
        self._lock = threading.Lock()
        self._cancel = False
        self._step = 0
        self._steps_total = 1
        self.config = config  # project Config (loras dir, VRAM options)
        self._active_loras: tuple[tuple[str, float], ...] | None = None
        # (prompt, negative) -> the encoded embeddings on the CPU (few MB
        # each, LRU); SHARED across this model's per-GPU instances so a
        # pre-encoded wildcard batch (queue.py) is a hit on every GPU; see
        # the encoding block in _generate_impl
        self._enc_cache = shared_enc_cache(info.family, info.id)

        # Model init creates torch parameters (unet + transformers text
        # encoders + VAE). peft/transformers monkey-patch GLOBAL torch state
        # during that init (see _PARAM_INIT_LOCK), so serialize it with LoRA
        # apply -- a model load on one GPU must never interleave with a LoRA
        # injection (or another model load) on another GPU.
        with _PARAM_INIT_LOCK:
            with warnings.catch_warnings():
                # loading a partially-cached pipeline emits dtype-casting and
                # image-processor warnings that are harmless for fp16 use
                warnings.simplefilter("ignore")
                self.pipe = StableDiffusionXLPipeline.from_single_file(
                    str(info.path), torch_dtype=torch.float16
                )
            # Swap in the improved fp16-stable VAE (ft-MSE) if present: the
            # stock SDXL VAE baked into the checkpoint overflows to NaN at
            # high-magnitude latents (uniformly black images). Prefer the
            # project's infra/ dir (home for encoder/decoder weights),
            # fall back to next to the model's grandparent dir (the
            # pre-Config-location behavior).
            vae_path = None
            if config is not None:
                candidate = config.infra_dir / "sdxl_vae.safetensors"
                if candidate.is_file():
                    vae_path = candidate
            if vae_path is None:
                candidate = info.path.parent.parent / "sdxl_vae.safetensors"
                if candidate.is_file():
                    vae_path = candidate
            if vae_path is not None:
                from diffusers import AutoencoderKL

                # from_single_file would otherwise download the VAE's
                # architecture config (vae/config.json) from the HF hub: the
                # checkpoint has no embedded diffusers metadata, so diffusers
                # infers a stock v1 VAE and fetches that repo's config. Point
                # it at the identical config committed under res/ instead, so
                # the swap is fully offline (weights still come from the file).
                # vae_path is <root>/infra/sdxl_vae.safetensors, so res/ is
                # its grandparent dir.
                vae_cfg = vae_path.parent.parent / "res" / "sdxl_vae.config.json"
                self.pipe.vae = AutoencoderKL.from_single_file(
                    str(vae_path), config=str(vae_cfg), torch_dtype=torch.float16
                )
                # The SDXL UNet was trained on 0.13025-scaled latents; the
                # decode/encode factors below read vae.config.scaling_factor.
                # A config lacking the key silently falls back to the SD 1.5
                # default (0.18215) and flattens contrast ~40% (see
                # vae-scaling-bug.md) -- fail loudly if that ever recurs.
                if self.pipe.vae.config.scaling_factor != 0.13025:
                    log.warning(
                        "improved-VAE config has scaling_factor %s, expected "
                        "0.13025 (SDXL); decode/encode will be off",
                        self.pipe.vae.config.scaling_factor,
                    )
            # move the whole pipeline to the GPU INSIDE the lock: the .to()
            # re-registers every parameter via nn.Module.register_parameter,
            # which a concurrent model load on another GPU may have globally
            # monkey-patched (see _PARAM_INIT_LOCK)
            self.pipe.to(self.device)
        # The VAE decodes in bf16 (Forge's default on sm_80+ GPUs): same
        # 8-bit mantissa as fp16 but 8x the exponent range. The improved
        # VAE is fp16-stable, so bf16 is even safer for it.
        cap = torch.cuda.get_device_capability(self.device)
        self.vae_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float32
        self.pipe.vae.to(dtype=self.vae_dtype)
        # batch=1 slicing keeps VAE decode at batch-1 memory cost
        self.pipe.vae.enable_slicing()
        # SDPA attention (diffusers' default; Forge's default 'pytorch'
        # attention is the same op): O(tokens) memory. Set explicitly --
        # `enable_attention_slicing()` must NOT be used: in diffusers 0.40
        # it installs SlicedAttnProcessor, which slices the *batch* dim,
        # so at batch=1 every self-attention materializes a full heads×T×T
        # score matrix (~14 GiB per layer at a 430x180 latent) and large
        # images OOM in the first downblock.
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.pipe.unet.set_attn_processor(AttnProcessor2_0())

        # --- VRAM options from imagagen.ini (all default off) ---
        self.text_encoder_device = self.device
        if config is not None:
            if config.text_encoder_device == "cpu":
                # keep the CLIP encoders in system RAM permanently (~1.4
                # GiB saved); CPU fp16 encoding is ~0.6 s per prompt
                self.text_encoder_device = "cpu"
                self.pipe.text_encoder.cpu()
                if getattr(self.pipe, "text_encoder_2", None) is not None:
                    self.pipe.text_encoder_2.cpu()
            if config.unet_quantization == "nf4":
                # Quantization replaces UNet linear layers with bnf4bit
                # (creates parameters) -- serialize it like the other
                # parameter-creation steps so it never interleaves with a
                # concurrent LoRA injection / model init on another GPU.
                with _PARAM_INIT_LOCK:
                    _quantize_unet_nf4(self.pipe.unet, self.device)
                torch.cuda.empty_cache()

    def _encode_prompt_chunked(self, prompt: str, device) -> tuple:
        """Encode `prompt` with Forge/A1111-style chunked CLIP encoding.

        diffusers' `encode_prompt` tokenizes with `truncation=True` and a
        77-token cap, silently dropping any tail beyond it. Instead we split
        the prompt's token stream into fixed 75-token windows (77 including
        the BOS/EOS framing tokens, exactly A1111's `PromptChunk` layout),
        encode each window through the encoder, and concatenate the per-window
        hidden states along the sequence dimension. The UNet's cross-attention
        accepts the longer sequence, so no prompt content is lost. The pooled
        output is taken from the first window (A1111's `zs[0].pooled`).

        Token weights (Forge/A1111 "emphasis", see token_weights.py): a
        segment annotated ``(green:1.2)`` has every token it tokenizes to
        scaled by 1.2 after encoding. When any segment carries a weight
        other than 1.0, the prompt is tokenized segment by segment (A1111's
        ``get_mult`` contract -- a word split into several BPE tokens gets
        all of them weighted); a prompt without effective weights keeps the
        exact whole-prompt tokenization of the cleaned text, so weightless
        prompts stay byte-identical to the pre-token-weight path.

        Returns ``(prompt_embeds, pooled)``: CLIP-L and CLIP-G window
        embeddings are concatenated along the feature dim (as diffusers does
        with `torch.concat(..., dim=-1)`), and `pooled` is CLIP-G's first
        window (``None`` when only a single encoder is present).
        """
        import torch

        from ..token_weights import clean_prompt, parse_prompt_weights

        pipe = self.pipe
        chunk_len = 75  # 77 total per window incl. BOS/EOS

        segments = parse_prompt_weights(prompt)
        weighted = any(w != 1.0 for _, w in segments)

        encoders = []
        if pipe.tokenizer is not None:
            encoders.append((pipe.tokenizer, pipe.text_encoder))
        if getattr(pipe, "tokenizer_2", None) is not None:
            encoders.append((pipe.tokenizer_2, pipe.text_encoder_2))

        window_embeds = []
        pooled = None
        for tokenizer, text_encoder in encoders:
            # content tokens without the framing special tokens
            if weighted:
                # one segment at a time, so every token knows its weight
                content, weights = [], []
                for text, w in segments:
                    ids = tokenizer(
                        text, add_special_tokens=False, return_tensors="pt"
                    ).input_ids[0].tolist()
                    content.extend(ids)
                    weights.extend([w] * len(ids))
            else:
                content = tokenizer(
                    clean_prompt(prompt), add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids[0].tolist()
                weights = [1.0] * len(content)
            bos = tokenizer.bos_token_id
            eos = tokenizer.eos_token_id
            pad = tokenizer.pad_token_id
            # empty prompt -> a single empty window ([BOS] + [EOS] + pad),
            # identical to diffusers encoding ""
            windows = (
                [([], [])] if not content
                else [(content[i:i + chunk_len], weights[i:i + chunk_len])
                      for i in range(0, len(content), chunk_len)]
            )
            hidden_parts = []
            for window, window_weights in windows:
                # each window is framed like diffusers: [BOS] + content + [EOS]
                # + pad. The pad token differs per encoder (49407 for CLIP-L,
                # 0 for CLIP-G), so use the tokenizer's real pad_token_id.
                ids = [bos] + window + [eos]
                n_pad = 77 - len(ids)
                ids += [pad] * n_pad
                out = text_encoder(
                    torch.tensor([ids], device=device, dtype=torch.long),
                    output_hidden_states=True,
                )
                # pooled comes from the encoder whose leading output field is
                # 2-D (CLIP-G); capture it from the first window only
                if pooled is None and out[0].ndim == 2:
                    pooled = out[0]
                # clip_skip=None: penultimate transformer layer (diffusers default)
                hidden = out.hidden_states[-2]
                if weighted:
                    # Forge's "No norm" emphasis: scale every token embedding
                    # by its segment's weight (BOS/EOS/pad stay at 1.0)
                    w = torch.tensor(
                        [1.0] + window_weights + [1.0] + [1.0] * n_pad,
                        device=device, dtype=hidden.dtype)
                    hidden = hidden * w.view(1, -1, 1)
                hidden_parts.append(hidden)
            window_embeds.append(torch.cat(hidden_parts, dim=1))

        prompt_embeds = torch.cat(window_embeds, dim=-1)
        return prompt_embeds, pooled

    def _encode_pairs(self, pairs: list[tuple[str, str]],
                      reserve: int | None = None) -> None:
        """Bring the text encoders to ``te_dev`` ONCE, encode every
        (prompt, negative) pair not yet in the shared cache, and park the
        encoders again.

        The ~1.4 GiB of encoders crosses the bus at most once per call,
        however many prompts it encodes: the generation queue pre-encodes
        a whole wildcard batch here (via :meth:`pre_encode`) instead of
        every task of the batch triggering its own encoder move. Each
        unique text is encoded once; every entry is assembled from the
        shared parts (A1111 semantics: an empty negative prompt means
        encoding "").

        ``reserve`` is the number of cache entries the caller expects to be
        resident afterwards: the LRU cap is raised to it so a pre-encoded
        batch is not evicted before its slowest task has run.
        """
        import torch

        pipe = self.pipe
        te_dev = self.text_encoder_device
        keys = list(dict.fromkeys((p, n or "") for p, n in pairs))
        missing = [k for k in keys if self._enc_cache.get(k) is None]
        if not missing:
            return
        if reserve is not None:
            self._enc_cache.bump(reserve)
        with torch.inference_mode():
            pipe.text_encoder.to(te_dev)
            if getattr(pipe, "text_encoder_2", None) is not None:
                pipe.text_encoder_2.to(te_dev)
            # sorted: a stable encode order (the result does not depend on
            # it, but logs and tests stay deterministic)
            texts = sorted({p for p, _ in missing} | {n for _, n in missing})
            parts: dict[str, tuple] = {}
            for t in texts:
                embeds, pooled = self._encode_prompt_chunked(t, te_dev)
                # CPU copies for the cache
                parts[t] = (embeds.cpu(), pooled.cpu())
            for key in keys:
                if self._enc_cache.get(key) is None:
                    p, n = key
                    self._enc_cache[key] = parts[p] + parts[n]
                    self._enc_cache.move_to_end(key)
            while len(self._enc_cache) > self._enc_cache.max:
                self._enc_cache.popitem(last=False)
            if te_dev != "cpu":
                pipe.text_encoder.cpu()
                if getattr(pipe, "text_encoder_2", None) is not None:
                    pipe.text_encoder_2.cpu()
                torch.cuda.empty_cache()

    def pre_encode(self, pairs: list[tuple[str, str]],
                   reserve: int | None = None) -> None:
        """One encoder session for a whole batch (see ModelBackend)."""
        with self._lock:
            self._encode_pairs(list(pairs), reserve)

    def cancel(self) -> None:
        self._cancel = True

    def unload(self) -> None:
        with self._lock:  # never move the pipeline while a generation is running
            # the .to() move re-registers every parameter via
            # nn.Module.register_parameter, which a concurrent model load on
            # another GPU may have globally monkey-patched -- hold
            # _PARAM_INIT_LOCK so the move never interleaves with it. The fp16
            # pipeline is only parked on CPU to free VRAM; no inference runs
            # there, so silence diffusers' dtype/device warning.
            with _PARAM_INIT_LOCK:
                self.pipe.to("cpu", silence_dtype_warnings=True)

    def load(self) -> None:
        """Bring a parked pipeline back to its GPU (inverse of ``unload``)."""
        with self._lock:
            # hold _PARAM_INIT_LOCK across the .to() moves: they re-register
            # every parameter via nn.Module.register_parameter, which a
            # concurrent model load on another GPU may have globally
            # monkey-patched (see _PARAM_INIT_LOCK)
            with _PARAM_INIT_LOCK:
                self.pipe.to(self.device, silence_dtype_warnings=True)
                # ``unload`` moved the whole pipeline to the CPU; restore the
                # configured text-encoder placement (GPU or CPU).
                self.pipe.text_encoder.to(self.text_encoder_device)
                if getattr(self.pipe, "text_encoder_2", None) is not None:
                    self.pipe.text_encoder_2.to(self.text_encoder_device)

    def generate(self, request: GenerateRequest, progress=None) -> GenerateResult:
        with self._lock:
            return self._generate_impl(request, progress)

    def _lora_path(self, name: str) -> str | None:
        """LoRA name (relative to loras/<model-type>/, no extension) -> path.

        The suffix is *appended*, never `Path.with_suffix`: names may
        themselves contain dots ("EnvyZoomSliderXL1.0"), and with_suffix
        would replace the ".0" instead of adding ".safetensors".
        """
        if self.config is None:
            return None
        loras_dir = self.config.loras_dir
        if not loras_dir.is_dir():
            return None
        for family in sorted(d for d in loras_dir.iterdir() if d.is_dir()):
            for suffix in (".safetensors", ".ckpt"):
                p = family / (name + suffix)
                if p.is_file():
                    return str(p)
        return None

    def _lora_state_dict(self, path: str) -> dict:
        """LoRA checkpoint -> diffusers-format state dict, te keys fixed.

        ``lora_state_dict`` does the real work (A1111/Kohya key conversion,
        sgm/LDM block remap, alphas). Its text-encoder keys assume the
        transformers-4 layout (``text_model.encoder...``); on transformers 5
        CLIPTextModel is flattened (``encoder...``) so key-to-module matching
        finds nothing and loading crashes with an empty rank dict. Re-prefix
        the te keys against the actual model layout, then hand the dict to
        ``load_lora_weights`` (a dict is taken as already-converted).

        Also pre-scales ``lora_down`` by ``alpha / rank``: the dict-input load
        path drops the alpha metadata (already-converted keys skip the
        conversion that extracts alphas), so without this the injected
        adapters would default to alpha == rank (scaling 1.0) and run
        ``rank / alpha`` times too strong vs the A1111/Forge slider
        convention (``delta = slider * (alpha/rank) * B@A``).
        """
        sd, alphas, _meta = self.pipe.lora_state_dict(
            str(path),
            unet_config=self.pipe.unet.config,
            return_lora_metadata=True,
        )

        # Must run before the text-encoder rename below: alphas and weights
        # share the same (possibly transformers-4) naming at this point.
        scale_lora_alphas(sd, alphas)

        def has_prefix(te, prefix: str) -> bool:
            return any(n.startswith(prefix) for n, _ in te.named_modules())

        fix_te1 = not has_prefix(self.pipe.text_encoder, "text_model.")
        fix_te2 = not has_prefix(self.pipe.text_encoder_2, "text_model.")
        fixed = {}
        for k, v in sd.items():
            if fix_te1 and k.startswith("text_encoder.text_model."):
                k = k.replace("text_encoder.text_model.", "text_encoder.", 1)
            elif fix_te2 and k.startswith("text_encoder_2.text_model."):
                k = k.replace("text_encoder_2.text_model.", "text_encoder_2.", 1)
            fixed[k] = v
        return fixed

    def _apply_loras(self, loras: list[tuple[str, float]]) -> None:
        """Inject the requested LoRAs into the pipeline (cached per set).

        Must run under ``self._lock`` (it mutates the UNet's modules).

        The apply is all-or-nothing: a set becomes ``_active_loras`` only once
        every adapter is injected AND weighted. If any injection fails partway,
        the adapters already injected are removed again so the pipeline is left
        clean. (Without this, a half-injected set leaves ``imagagen_0`` in the
        UNet while ``_active_loras`` is still ``None``; the next load reuses the
        same adapter name and raises ``Adapter name ... already in use`` instead
        of surfacing the real failure -- and keeps failing until restart.)
        """
        key = tuple((n, float(w)) for n, w in loras)
        if key == self._active_loras:
            return
        # Resolve every path first: a missing file must not leave the
        # pipeline half-unloaded, and (crucially) `_active_loras` is only
        # committed on full success -- a failed set must not poison the
        # cache or the next identical request would skip loading and run
        # silently without the LoRAs.
        paths = []
        for name, _w in key:
            path = self._lora_path(name)
            if path is None:
                raise ValueError(f"LoRA not found: {name!r} (looked in loras/)")
            paths.append(path)
        # The unload + injection below go through diffusers/peft, which
        # monkey-patch GLOBAL torch state for the text-encoder path (see
        # _PARAM_INIT_LOCK). Serialize it with model init so two GPU workers
        # never interleave those patches.
        with _PARAM_INIT_LOCK:
            # Drop whatever is currently applied before rebuilding. Unload
            # based on the *actual* UNet state as well as the cache, so
            # adapters left behind by a previously failed load are cleared
            # too.
            self._unload_loras()
            if not key:
                return
            tags, weights = [], []
            try:
                for i, (name, weight) in enumerate(key):
                    tag = f"imagagen_{i}"
                    # diffusers 0.40 returns None on success (older versions
                    # returned a (failed, succeeded) tuple); keep the check
                    # for both shapes.
                    res = self.pipe.load_lora_weights(
                        self._lora_state_dict(paths[i]), adapter_name=tag
                    )
                    if res is None:
                        pass
                    elif isinstance(res, tuple) and res[0]:
                        raise ValueError(f"failed to load LoRA {name!r}: {res[0]}")
                    elif res:
                        raise ValueError(f"failed to load LoRA {name!r}: {res}")
                    tags.append(tag)
                    weights.append(weight)
                if weights:
                    self.pipe.set_adapters(tags, weights)
            except Exception:
                # A partial injection must not persist: remove the adapters
                # that were already added, then re-raise the real error.
                self._unload_loras()
                raise
            self._active_loras = key
            log.info("LoRA load: %s", format_loras(list(key)))

    def _unload_loras(self) -> None:
        """Remove every applied LoRA adapter from the pipeline.

        Runs under ``self._lock`` and ``_PARAM_INIT_LOCK`` (it is only called
        from ``_apply_loras``). Clears based on the *actual* UNet state
        (``peft_config``) in addition to the ``_active_loras`` cache, so it
        also recovers from a previously failed (partial) load that left
        adapters injected while the cache was reset to ``None``.
        """
        if self._active_loras is None and \
                not getattr(self.pipe.unet, "peft_config", None):
            return  # nothing applied: nothing to unload
        if self._active_loras is not None:
            log.info("LoRA unload: %s", format_loras(list(self._active_loras)))
        else:
            log.warning("LoRA unload: clearing leftover adapters (no active set tracked)")
        self.pipe.unload_lora_weights()
        self._active_loras = None

    def _encode_init(self, pipe, arr: np.ndarray, device):
        """VAE-encode an HxWx3 uint8 image to its latent (the deterministic
        latent_dist.mode(), i.e. A1111's "Full" encode -- same seed gives the
        same result)."""
        import torch

        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device)
        t = t.permute(2, 0, 1).float()[None] / 127.5 - 1.0
        lat = pipe.vae.encode(t.to(dtype=pipe.vae.dtype)).latent_dist
        return lat.mode() * pipe.vae.config.scaling_factor

    def _generate_impl(self, request: GenerateRequest, progress=None) -> GenerateResult:
        import torch

        self._cancel = False
        pipe = self.pipe
        device = self.device
        w = max(8, (request.width // 8) * 8)
        h = max(8, (request.height // 8) * 8)
        lh, lw = h // 8, w // 8

        # VAE tiling: vae-tiling=true in imagagen.ini forces it; otherwise
        # it is automatic for large images. The untiled decode/encode
        # allocates full-resolution 32-channel fp32 conv buffers (~1.6 GiB
        # per MP of image on top of the ~5 GiB of resident weights) and
        # OOMs on a 16 GB card past ~5 MP; measured at 3440x1440 the tiled
        # path is faster (3.6 s vs 4.3 s), peaks at 7.4 GiB instead of
        # 13.3, and differs by <= 1/255 per pixel.
        if self.config is not None:
            pipe.vae.use_tiling = self.config.vae_tiling or w * h > 4_000_000

        self._apply_loras(request.loras)

        with torch.inference_mode():
            # text encoders live on self.text_encoder_device (GPU by
            # default, CPU when imagagen.ini says so); with the default
            # they are parked on CPU between generations to free ~1.4 GiB
            te_dev = self.text_encoder_device
            # Forge-style chunked encoding (see _encode_prompt_chunked): a
            # prompt longer than 75 tokens is split into 75-token windows and
            # each window is encoded, then concatenated -- nothing is dropped.
            # diffusers' encode_prompt would hard-truncate to 77 instead.
            # The encoding is deterministic, so it is cached on the CPU
            # (a few MB, LRU, shared across this model's per-GPU instances):
            # a batch of identical prompts -- or any later repeat of the same
            # prompt -- is encoded once, and a hit skips the encoder moves
            # entirely.
            key = (request.prompt, request.negative_prompt or "")
            entry = self._enc_cache.get(key)
            if entry is None:
                self._encode_pairs([(request.prompt, request.negative_prompt)])
                entry = self._enc_cache[key]
            self._enc_cache.move_to_end(key)
            prompt_embeds, pooled, neg_embeds, neg_pooled = (
                t.to(device) for t in entry)
            if te_dev != "cpu":
                # Free ~1.3 GiB for UNet + VAE activations: the text
                # encoders are only needed for prompt encoding. On a cache
                # hit they are normally already on the CPU (moving a parked
                # module is a no-op).
                pipe.text_encoder.cpu()
                if getattr(pipe, "text_encoder_2", None) is not None:
                    pipe.text_encoder_2.cpu()
                torch.cuda.empty_cache()

            time_ids = torch.tensor([h, w, 0, 0, h, w], device=device).unsqueeze(0)
            unet = pipe.unet
            # the model's beta schedule (from its scheduler config) -- the
            # sigma range is its [first, last] sigma (A1111/Forge
            # "rescaled betas")
            ac = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
            log_sigmas = torch.log(torch.sqrt((1.0 - ac) / ac))

            # A1111/Forge k-diffusion ModelSamplingDiscrete.timestep: the
            # *nearest* beta-schedule sigma index in log space (see
            # sigma_to_timestep for the full rationale).
            # the "model default" sentinel (request.steps/cfg <= 0) resolves
            # to the built-in values (the PNG metadata records these
            # effective values via the result, not the raw request)
            steps = request.steps if request.steps > 0 else 20
            cfg = request.cfg if request.cfg > 0 else DEFAULT_CFG
            # --- sampler / scheduler registries (Phase 5 grows these into plugins) ---
            sampler_name = request.sampler or self.default_sampler
            scheduler_name = request.scheduler or self.default_scheduler
            if sampler_name not in SAMPLERS:
                raise ValueError(f"unknown sampler {sampler_name!r}")
            if scheduler_name not in SCHEDULERS:
                raise ValueError(f"unknown scheduler {scheduler_name!r}")
            sampler_fn = SAMPLERS[sampler_name]
            sigmas_fn = SCHEDULERS[scheduler_name]
            # LCM models are trained on only 50 of the 1000 schedule
            # timesteps; the sigma range and the UNet t input are
            # quantized to that 50-slot grid (see _LcmSchedule)
            lcm = _LcmSchedule(ac) if sampler_name == "LCM" else None
            sigma_to_t = (lambda sigma: lcm.timestep(sigma)) if lcm is not None \
                else (lambda sigma: sigma_to_timestep(sigma, log_sigmas))

            def unet_forward(emb, pool, x, t):
                added = {"time_ids": time_ids, "text_embeds": pool}
                # the UNet runs in fp16; Forge casts the model output back
                # to fp32 before the x0 conversion (that is what keeps the
                # solver state in fp32)
                return unet(
                    x, t, encoder_hidden_states=emb, added_cond_kwargs=added
                ).sample.float()

            # --- self attention guidance (Forge builtin extension) ---
            # the unconditional pass' mid-block self-attention is captured
            # by hooks (installed right before sampling) and the guidance
            # correction is added in model_fn; see sag.py. SAG needs the
            # unconditional pass (cfg > 1) and a latent grid above 4 px
            # (Forge skips tiny latents); otherwise it is simply not
            # applied, like Forge.
            sag = request.sag
            sag_rec = (
                SagRecorder()
                if (sag is not None and sag.enabled
                        and cfg > 1.0 and min(lh, lw) > 4)
                else None
            )

            def model_fn(x, sigma):
                if self._cancel:
                    raise InterruptedError("cancelled")
                self._step += 1
                t = sigma_to_t(sigma)
                xwork = x
                if mask_t is not None:
                    # per-step lock (Forge's cfg_denoiser): before the model
                    # call, the protected region (mask_t == 1) is replaced by
                    # init latent + fresh noise at the current sigma level
                    # (Forge's noise_scaling, max_denoise=False)
                    xnoise = torch.randn_like(x) * sigma
                    xwork = x * (1.0 - mask_t) + (init_latent + xnoise) * mask_t
                # the UNet expects unit-variance input at each noise level
                # (x / sqrt(sigma^2 + 1)); computed in fp32, cast to the
                # UNet's fp16 dtype right before the call (like Forge's
                # calculate_input + .to(computation_dtype))
                x_in = (xwork / (sigma ** 2 + 1.0).sqrt()).to(torch.float16)
                if cfg <= 1.0:
                    # CFG 1 needs no unconditional pass (Forge skips it)
                    eps = unet_forward(prompt_embeds, pooled, x_in, t)
                else:
                    c = unet_forward(prompt_embeds, pooled, x_in, t)
                    if sag_rec is not None:
                        # SAG records the mid-block self-attention of the
                        # unconditional pass only (see sag.py)
                        sag_rec.record = True
                    uc = unet_forward(neg_embeds, neg_pooled, x_in, t)
                    if sag_rec is not None:
                        sag_rec.record = False
                    eps = uc + cfg * (c - uc)
                if progress and self._step <= self._steps_total:
                    progress(self._step / self._steps_total,
                             f"step {self._step}/{self._steps_total}")
                # UNet output is an epsilon prediction -> convert to x0.
                # x is fp32 (solver state), so x0 stays fp32.
                def x0_of(xx, e):
                    r = xx - sigma * e
                    if lcm is not None:
                        r = lcm.x0_scaled(xx, sigma, e)
                    return r
                x0 = x0_of(xwork, eps)
                if sag_rec is not None and sag_rec.q is not None:
                    # SAG (Forge builtin extension): the uncond x0 is
                    # adversarially blurred where the uncond self-attention
                    # is concentrated, re-noised to the current sigma,
                    # denoised with one extra uncond UNet pass, and the
                    # (degraded - re-denoised) difference is added to the
                    # CFG result, scaled (x0 space, before the lock)
                    uncond_pred = x0_of(xwork, uc)
                    mask = compute_sag_mask(
                        sag_rec.q, sag_rec.k, sag_rec.heads, sag.threshold)
                    degraded = create_blur_map(
                        uncond_pred, mask, sag.blur_sigma)
                    degraded_noised = degraded + xwork - uncond_pred
                    sag_eps = unet_forward(
                        neg_embeds, neg_pooled,
                        (degraded_noised / (sigma ** 2 + 1.0).sqrt())
                        .to(torch.float16),
                        t)
                    x0 = x0 + (degraded - x0_of(degraded_noised, sag_eps)) \
                        * sag.scale
                if mask_t is not None:
                    if soft is not None:
                        # soft inpainting (Forge builtin extension): instead
                        # of the hard lock, the x0 estimate is blended with
                        # the init latent; the blend weight follows the mask
                        # opacity and the current sigma (power -> when it
                        # happens, scale -> how strongly), and the vector
                        # magnitudes are interpolated under the detail
                        # preservation exponent
                        t = torch.pow(
                            nmask4,
                            (sigma ** soft.mask_blend_power) * soft.mask_blend_scale,
                        )
                        x0 = latent_blend(soft, init_latent, x0, t)
                    else:
                        # hard lock: the x0 estimate is locked back to the
                        # init latent (protected content decodes exactly)
                        x0 = x0 * (1.0 - mask_t) + init_latent * mask_t
                return x0

            self._step = 0

            # A1111/Forge "rescaled betas" (their default): sample over the
            # model's real beta-schedule range instead of the clamped
            # k-diffusion [0.1, 10]. For SDXL that is ~[0.03, ~14.6]: start
            # from the true terminal noise level and refine down to almost
            # zero residual noise. The [0.1, 10] clamp renders visibly
            # flatter/grainier (see dev/softness_check.py). The LCM
            # sampler instead uses its 50-slot range (see _LcmSchedule).
            if lcm is not None:
                sigma_min, sigma_max = lcm.min_sigma, lcm.max_sigma
            else:
                beta_sigmas = torch.exp(log_sigmas)
                sigma_min = float(beta_sigmas[0])
                sigma_max = float(beta_sigmas[-1])
            model_wrap = _ModelSamplingDiscrete(model_fn, sigma_min, sigma_max)
            sigmas = sigmas_fn(
                steps, model_wrap.min_sigma, model_wrap.max_sigma, device=device
            )
            # seeded initial noise: same seed -> same image (A1111-compatible).
            # fp32 unit prior: the solver state must not quantize (Forge's path
            # promotes to fp32 at step 1). For txt2img it is scaled up to the max
            # noise level right before sampling (the solver state is the noisy
            # sample itself); img2img/inpaint build their own noise-scaled state.
            gen = torch.Generator(device=device).manual_seed(request.seed)
            noise = torch.randn((1, 4, lh, lw), device=device, dtype=torch.float32,
                                generator=gen)

            # --- img2img / inpainting (A1111/Forge kdiff convention) ---
            # With an init image: start from the init latent instead of pure
            # noise, truncate the sigma schedule at the denoising strength,
            # and (with an init mask) lock the protected region (mask 0) to
            # the init content at every step. init_mask marks the *inpaint*
            # region, so protected = 1 - mask.
            init_latent = None
            mask_t = None
            nmask4 = None  # (4, lh, lw): the inpaint region (1) as seen by
                           # soft inpainting (all four channels carry it)
            soft = request.soft_inpainting
            if soft is not None and not soft.enabled:
                soft = None  # a disabled object behaves exactly like absent
            sigma_sched = sigmas
            if request.init_image is not None:
                init_latent = self._encode_init(pipe, request.init_image, device)
                # A1111: t_enc = int(min(strength, 0.999) * steps); the
                # schedule starts at sigmas[steps - 1 - t_enc] (strength 1.0
                # -> the full schedule)
                t_enc = int(min(request.denoise, 0.999) * steps)
                sigma_sched = sigmas[steps - 1 - t_enc :]
                # Forge's noise_scaling at the truncated start sigma
                # (max_denoise=False): init latent + fresh noise at that
                # sigma level. The noise amplitude IS the denoise strength
                # (denoise 0.55, 12 steps -> sigma0 0.88 -> the model sees
                # ~0.75*init + ~0.66*noise in the region at step 1); the
                # state must not be "almost clean init" there or the
                # effective strength collapses to ~0 and the region comes
                # out as a smoothed version of the init image
                noise = init_latent + noise * sigma_sched[0]
                if request.init_mask is not None:
                    # (1,1,H,W) -> bilinear to the latent grid; this shape
                    # broadcasts against the (1,4,lh,lw) solver tensors
                    m = torch.from_numpy(
                        np.ascontiguousarray(request.init_mask)
                    ).to(device=device).float().div(255.0)[None, None]
                    m = torch.nn.functional.interpolate(m, size=(lh, lw), mode="bilinear")
                    mask_t = (1.0 - m)
                    # A1111's init composite: only the inpaint region starts
                    # re-noised; the protected region starts at the clean
                    # init latent (it is locked there at every step anyway)
                    noise = noise * (1.0 - mask_t) + init_latent * mask_t
                    if soft is not None:
                        nmask4 = (1.0 - mask_t[0, 0]).unsqueeze(0).expand(4, lh, lw)
            self._steps_total = max(1, len(sigma_sched) - 1)
            # txt2img: the solver state is the noisy sample itself, so scale the
            # unit prior to the max noise level. The UNet divides by sqrt(sigma^2+1)
            # to reach the unit-variance input it was trained on; if the state stays
            # at unit scale that input is off-distribution (far too small) at every
            # step and the pure-ODE samplers collapse to a flat image (regression
            # from 719d341, which dropped this scaling when adding img2img). The
            # img2img/inpaint path already produced a noise-scaled state (blended at
            # the truncated start sigma), so it is left untouched.
            if request.init_image is None:
                noise = noise * model_wrap.max_sigma
            # the seed is threaded into the sampler so SDE/ancestral samplers
            # derive their per-step noise deterministically from it
            sag_handles = (
                install_sag_hooks(unet, sag_rec) if sag_rec is not None else []
            )
            try:
                samples = sampler_fn(model_wrap, noise, sigma_sched,
                                     extra_args={"seed": request.seed})
            finally:
                # NB: don't shadow `h` (the image height) with the loop var
                for handle in sag_handles:
                    handle.remove()
            if init_latent is not None and mask_t is not None and soft is None:
                # final lock: the protected region is exactly the init latent
                # (skipped with soft inpainting: the protected content is
                # soft-blended at every step and restored by the pixel
                # composite below instead)
                samples = samples * (1.0 - mask_t) + init_latent * mask_t

            # sampling ran in VAE-scaled latent space -> unscale (fp32) and
            # cast to the VAE dtype before decode (like Forge's decode path)
            latents = (samples / pipe.vae.config.scaling_factor).to(self.vae_dtype)
            image = pipe.vae.decode(latents, return_dict=False)[0].float()
            image = (image / 2 + 0.5).clamp(0, 1)
            arr = (image.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)

        if soft is not None and mask_t is not None:
            # Forge's post_sample pixel composite: composite the original
            # image back over the generation wherever the latents did not
            # change much (locally percentile-normalised distance,
            # threshold scaled by the original mask, blurred at image scale)
            c = adaptive_change_alpha(soft, nmask4, init_latent, samples, w, h)
            base = np.ascontiguousarray(request.init_image)
            if base.shape[:2] != (h, w):
                from PIL import Image

                base = np.asarray(
                    Image.fromarray(base).resize((w, h), Image.LANCZOS)
                )
            arr = arr.astype(np.float32) * c[..., None] \
                + base.astype(np.float32) * (1.0 - c[..., None])
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return GenerateResult(
            image=arr, seed=request.seed, steps=steps, cfg=cfg)


def from_config(config) -> ModelBackend | None:
    """Load the first discovered SDXL model, or None when nothing usable exists."""
    from ..models import ModelRegistry

    registry = ModelRegistry(config)
    if not registry.models or not registry.gpus:
        return None
    return registry.backend_for(registry.models[0].id)
