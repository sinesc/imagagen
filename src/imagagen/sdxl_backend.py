"""SDXL backend: diffusers pipeline + vendored k-diffusion sampling loop.

Phase 1 ships three samplers (Euler a, DPM++ 2M, DPM++ 2M SDE) and three
sigma schedules (Karras, Exponential, Polyexponential); both are chosen
from small registries so Phase 5's plugin system can grow them without
touching this file.

SDXL specifics (matching the k-diffusion / A1111 contract):
- sigma_data = 1.0; the sigma range is the model's actual beta-schedule
  range (SDXL: ~[0.03, ~14.6]), not the clamped k-diffusion [0.1, 10] --
  this is the A1111/Forge "rescaled betas" behavior and is what gives the
  characteristic look (the [0.1, 10] clamp renders flatter and grainier)
- the UNet expects *training* timesteps (0..999), so each k-diffusion sigma
  is mapped to the nearest timestep of the model's beta schedule
  (t = argmin |log(sigma) - log_sigmas|), as diffusers' _sigma_to_t does
- the UNet expects a *unit-variance* input at every noise level, so the
  model input is scaled by 1/sqrt(sigma^2 + 1) (exactly what diffusers'
  EulerDiscreteScheduler.scale_model_input does)
- the UNet predicts EPSILON; the sampler wants an x0 estimate, so the model
  wrap returns x0 = x - sigma * eps (the exact inverse of diffusers' Euler)
- initial noise is randn * sigma_max (the state x is the noisy sample, not
  the unit-variance prior)
- sampling runs in the VAE-scaled latent space; divide by
  vae.config.scaling_factor (0.13025 for SDXL) before decode

Precision (matching Forge's kdiff path exactly):
- the solver state x runs in fp32: Forge's model call casts the UNet output
  to fp32, so calculate_denoised (x - sigma * eps) promotes x to fp32 from
  step 1 on, and the whole DPM++ update then stays fp32. Running the
  solver in fp16 (10-bit mantissa) measurably flattens detail/contrast.
- only the UNet itself runs in fp16 (input cast right before the call,
  like Forge's calculate_input + .to(computation_dtype))
- the VAE decodes in bf16 (Forge's default on sm_80+ GPUs): bf16 has the
  same 8-bit mantissa as fp16 but 8x the dynamic range, which is also why
  Forge does not hit the fp16 stock-VAE overflow/NaN that used to make us
  swap in the improved VAE

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
- vae-tiling = on: tiled VAE decode, lower peak at 1024².
"""

from __future__ import annotations

import hashlib
import threading
import warnings

import numpy as np

from . import kdiff
from .backends import GenerateRequest, GenerateResult, ModelBackend

# fp16 pipelines are intentionally offloaded to CPU between generations to
# free VRAM; diffusers warns that fp16 does not compute on CPU -- it does not
# need to (CPU copies are storage only), so silence the warning once.
warnings.filterwarnings(
    "ignore", message="Pipelines loaded with `dtype=torch.float16` cannot run with `cpu` device.*"
)

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
}
SCHEDULERS = {
    "Karras": kdiff.get_sigmas_karras,
    "Exponential": kdiff.get_sigmas_exponential,
    "Polyexponential": kdiff.get_sigmas_polyexponential,
}


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
    def __init__(self, info, device, config=None) -> None:
        import os
        import warnings

        import torch
        from diffusers import StableDiffusionXLPipeline

        # Backstop in case load_config() was not used: keep any hub files
        # from_single_file fetches in the persistent project-local cache.
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(info.path.parent.parent / "hf_cache"))

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

        with warnings.catch_warnings():
            # loading a partially-cached pipeline emits dtype-casting and
            # image-processor warnings that are harmless for fp16 use
            warnings.simplefilter("ignore")
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                str(info.path), torch_dtype=torch.float16
            )
        self.pipe.to(self.device)
        # Forge's default VAE dtype on sm_80+ is bf16: same 8-bit mantissa
        # as fp16 but 8x the exponent range, so the stock VAE no longer
        # overflows (the fp16 NaN that used to force a VAE swap is gone).
        cap = torch.cuda.get_device_capability(self.device)
        self.vae_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float32
        self.pipe.vae.to(dtype=self.vae_dtype)
        # batch=1 slicing keeps VAE decode at batch-1 memory cost
        self.pipe.vae.enable_slicing()
        # sliced attention: no quality change, lower peak VRAM at 1024²
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass  # not supported by this UNet variant; fall back to full attention

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
            if config.vae_tiling:
                self.pipe.vae.enable_tiling()
            if config.unet_quantization == "nf4":
                _quantize_unet_nf4(self.pipe.unet, self.device)
                torch.cuda.empty_cache()

    def _encoder_kwargs(self, device) -> dict:
        return {
            "device": device,
            "num_images_per_prompt": 1,
        }

    def cancel(self) -> None:
        self._cancel = True

    def unload(self) -> None:
        with self._lock:  # never move the pipeline while a generation is running
            self.pipe.to("cpu")

    def generate(self, request: GenerateRequest, progress=None) -> GenerateResult:
        with self._lock:
            return self._generate_impl(request, progress)

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

        with torch.inference_mode():
            # text encoders live on self.text_encoder_device (GPU by
            # default, CPU when imagagen.ini says so); with the default
            # they are parked on CPU between generations to free ~1.4 GiB
            te_dev = self.text_encoder_device
            pipe.text_encoder.to(te_dev)
            if getattr(pipe, "text_encoder_2", None) is not None:
                pipe.text_encoder_2.to(te_dev)
            ek = self._encoder_kwargs(te_dev)
            # encode_prompt -> (prompt_embeds, neg_embeds, pooled, neg_pooled)
            prompt_embeds, _, pooled, _ = pipe.encode_prompt(prompt=request.prompt, **ek)
            # A1111 semantics: an empty negative prompt means encoding ""
            _, neg_embeds, _, neg_pooled = pipe.encode_prompt(
                prompt=request.negative_prompt or "", **ek
            )
            if te_dev == "cpu":
                # embeddings to the GPU; the encoders stay in system RAM
                prompt_embeds = prompt_embeds.to(device)
                neg_embeds = neg_embeds.to(device)
                pooled = pooled.to(device)
                neg_pooled = neg_pooled.to(device)
            else:
                # Free ~1.3 GiB for UNet + VAE activations: the text
                # encoders are only needed for prompt encoding.
                pipe.text_encoder.cpu()
                if getattr(pipe, "text_encoder_2", None) is not None:
                    pipe.text_encoder_2.cpu()
                torch.cuda.empty_cache()

            time_ids = torch.tensor([h, w, 0, 0, h, w], device=device).unsqueeze(0)
            unet = pipe.unet
            # training sigmas of the model's beta schedule (from its scheduler
            # config) -- used to map a k-diffusion sigma to the nearest
            # training timestep (0..999), exactly like diffusers' _sigma_to_t
            ac = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
            log_sigmas = torch.log(torch.sqrt((1.0 - ac) / ac))

            def sigma_to_t(sigma):
                return (torch.log(sigma.float()) - log_sigmas).abs().argmin(-1)

            def unet_forward(emb, pool, x, t):
                added = {"time_ids": time_ids, "text_embeds": pool}
                # the UNet runs in fp16; Forge casts the model output back
                # to fp32 before the x0 conversion (that is what keeps the
                # solver state in fp32)
                return unet(
                    x, t, encoder_hidden_states=emb, added_cond_kwargs=added
                ).sample.float()

            def model_fn(x, sigma):
                if self._cancel:
                    raise InterruptedError("cancelled")
                self._step += 1
                t = sigma_to_t(sigma)
                xwork = x
                if mask_t is not None:
                    # per-step lock (Forge's cfg_denoiser): before the model
                    # call, the protected region (mask_t == 1) is replaced by
                    # fresh init-latent + noise at the current sigma level
                    xsigma = init_latent * (sigma**2 + 1.0).sqrt()
                    xnoise = torch.randn_like(x) * sigma
                    frac = ((sigma - model_wrap.min_sigma)
                            / (model_wrap.max_sigma - model_wrap.min_sigma)).clamp(0, 1)
                    xwork = x * (1.0 - mask_t) \
                        + (xsigma + (xnoise - xsigma) * frac) * mask_t
                # the UNet expects unit-variance input at each noise level
                # (x / sqrt(sigma^2 + 1)); computed in fp32, cast to the
                # UNet's fp16 dtype right before the call (like Forge's
                # calculate_input + .to(computation_dtype))
                x_in = (xwork / (sigma ** 2 + 1.0).sqrt()).to(torch.float16)
                c = unet_forward(prompt_embeds, pooled, x_in, t)
                uc = unet_forward(neg_embeds, neg_pooled, x_in, t)
                if progress and self._step <= self._steps_total:
                    progress(self._step / self._steps_total,
                             f"step {self._step}/{self._steps_total}")
                # UNet output is an epsilon prediction -> convert to x0.
                # x is fp32 (solver state), so x0 stays fp32.
                eps = uc + request.cfg * (c - uc)
                x0 = xwork - sigma * eps
                if mask_t is not None:
                    # and the x0 estimate is locked back to the init latent
                    # (protected content decodes exactly)
                    x0 = x0 * (1.0 - mask_t) + init_latent * mask_t
                return x0

            self._step = 0

            # --- sampler / scheduler registries (Phase 5 grows these into plugins) ---
            sampler_name = request.sampler or self.default_sampler
            scheduler_name = request.scheduler or self.default_scheduler
            if sampler_name not in SAMPLERS:
                raise ValueError(f"unknown sampler {sampler_name!r}")
            if scheduler_name not in SCHEDULERS:
                raise ValueError(f"unknown scheduler {scheduler_name!r}")
            sampler_fn = SAMPLERS[sampler_name]
            sigmas_fn = SCHEDULERS[scheduler_name]

            # A1111/Forge "rescaled betas" (their default): sample over the
            # model's real beta-schedule range instead of the clamped
            # k-diffusion [0.1, 10]. For SDXL that is ~[0.03, ~14.6]: start
            # from the true terminal noise level and refine down to almost
            # zero residual noise. The [0.1, 10] clamp renders visibly
            # flatter/grainier (see devtools/softness_check.py).
            beta_sigmas = torch.exp(log_sigmas)
            sigma_min = float(beta_sigmas[0])
            sigma_max = float(beta_sigmas[-1])
            model_wrap = _ModelSamplingDiscrete(model_fn, sigma_min, sigma_max)
            sigmas = sigmas_fn(
                request.steps, model_wrap.min_sigma, model_wrap.max_sigma, device=device
            )
            # seeded initial noise: same seed -> same image (A1111-compatible).
            # k-diffusion's state is the noisy sample itself, so scale the
            # unit prior to the max noise level. fp32: the solver state
            # must not quantize (Forge's path promotes to fp32 at step 1).
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
            sigma_sched = sigmas
            if request.init_image is not None:
                init_latent = self._encode_init(pipe, request.init_image, device)
                # A1111: t_enc = int(min(strength, 0.999) * steps); the
                # schedule starts at sigmas[steps - 1 - t_enc] (strength 1.0
                # -> the full schedule)
                t_enc = int(min(request.denoise, 0.999) * request.steps)
                sigma_sched = sigmas[request.steps - 1 - t_enc :]
                # noise_scaling(sigma0, noise, init): the init latent scaled
                # to the start sigma, blended with fresh noise according to
                # where sigma0 sits in [sigma_min, sigma_max]
                sigma0 = sigma_sched[0]
                xsigma = init_latent * (sigma0**2 + 1.0).sqrt()
                xnoise = noise * sigma0
                frac = ((sigma0 - model_wrap.min_sigma)
                        / (model_wrap.max_sigma - model_wrap.min_sigma)).clamp(0, 1)
                noise = xsigma + (xnoise - xsigma) * frac
                if request.init_mask is not None:
                    m = torch.from_numpy(
                        np.ascontiguousarray(request.init_mask)
                    ).to(device=device).float().div(255.0).unsqueeze(0)
                    m = torch.nn.functional.interpolate(m, size=(lh, lw), mode="bilinear")
                    mask_t = (1.0 - m).expand(4, lh, lw).contiguous()
                    # A1111's init composite: only the inpaint region starts
                    # re-noised; the protected region starts at the clean
                    # init latent (it is locked there at every step anyway)
                    noise = noise * (1.0 - mask_t) + init_latent * mask_t
            self._steps_total = max(1, len(sigma_sched) - 1)
            # the seed is threaded into the sampler so SDE/ancestral samplers
            # derive their per-step noise deterministically from it
            samples = sampler_fn(model_wrap, noise, sigma_sched,
                                 extra_args={"seed": request.seed})
            if init_latent is not None and mask_t is not None:
                # final lock: the protected region is exactly the init latent
                samples = samples * (1.0 - mask_t) + init_latent * mask_t

            # sampling ran in VAE-scaled latent space -> unscale (fp32) and
            # cast to the VAE dtype before decode (like Forge's decode path)
            latents = (samples / pipe.vae.config.scaling_factor).to(self.vae_dtype)
            image = pipe.vae.decode(latents, return_dict=False)[0].float()
            image = (image / 2 + 0.5).clamp(0, 1)
            arr = (image.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)

        return GenerateResult(image=arr, seed=request.seed)


def from_config(config) -> ModelBackend | None:
    """Load the first discovered SDXL model, or None when nothing usable exists."""
    from .gpus import available_gpus
    from .models import ModelRegistry

    if not available_gpus():
        return None
    registry = ModelRegistry(config)
    if not registry.models:
        return None
    return registry.backend_for(registry.models[0].id)
