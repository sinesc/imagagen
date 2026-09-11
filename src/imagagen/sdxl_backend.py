"""SDXL backend: diffusers pipeline + vendored k-diffusion sampling loop.

Phase 1 ships three samplers (Euler a, DPM++ 2M, DPM++ 2M SDE) and three
sigma schedules (Karras, Exponential, Polyexponential); both are chosen
from small registries so Phase 5's plugin system can grow them without
touching this file.

SDXL specifics (matching the k-diffusion / A1111 contract):
- sigma_data = 1.0, karras sigma range [0.1, 10]
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

    def __init__(self, fn, sigma_data: float = 1.0):
        self.fn = fn
        self.sigma_data = sigma_data
        self.max_sigma = sigma_data * 10.0
        self.min_sigma = sigma_data * 0.1

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


def _model_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            if f.tell() >= 100 * 2**20:
                break
    return h.hexdigest()[:16]


class SdxlBackend(ModelBackend):
    def __init__(self, info, device) -> None:
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
            # Swap in the improved fp16-stable VAE (ft-MSE) if present: the
            # stock SDXL VAE overflows to NaN in fp16 at high-magnitude latents.
            vae_path = info.path.parent.parent / "sdxl_vae.safetensors"
            if vae_path.is_file():
                from diffusers import AutoencoderKL

                self.pipe.vae = AutoencoderKL.from_pretrained(
                    str(vae_path.parent), torch_dtype=torch.float16
                ) if (vae_path.parent / "config.json").is_file() else \
                    AutoencoderKL.from_single_file(str(vae_path), torch_dtype=torch.float16)
            self.pipe.vae.enable_slicing()
        self.pipe.to(self.device)
        # batch=1 slicing keeps VAE decode at batch-1 memory cost
        self.pipe.vae.enable_slicing()
        # sliced attention: no quality change, lower peak VRAM at 1024²
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass  # not supported by this UNet variant; fall back to full attention

    def _encoder_kwargs(self) -> dict:
        return {
            "device": self.device,
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

    def _generate_impl(self, request: GenerateRequest, progress=None) -> GenerateResult:
        import torch

        self._cancel = False
        pipe = self.pipe
        device = self.device
        w = max(8, (request.width // 8) * 8)
        h = max(8, (request.height // 8) * 8)
        lh, lw = h // 8, w // 8

        with torch.inference_mode():
            # text encoders were offloaded to CPU after the previous encode;
            # bring them back for this prompt, offload again afterwards
            pipe.text_encoder.to(device)
            if getattr(pipe, "text_encoder_2", None) is not None:
                pipe.text_encoder_2.to(device)
            ek = self._encoder_kwargs()
            # encode_prompt -> (prompt_embeds, neg_embeds, pooled, neg_pooled)
            prompt_embeds, _, pooled, _ = pipe.encode_prompt(prompt=request.prompt, **ek)
            # A1111 semantics: an empty negative prompt means encoding ""
            _, neg_embeds, _, neg_pooled = pipe.encode_prompt(
                prompt=request.negative_prompt or "", **ek
            )
            # Free ~1.3 GiB for UNet + VAE activations: the text encoders are
            # only needed for prompt encoding.
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
                return unet(
                    x, t, encoder_hidden_states=emb, added_cond_kwargs=added
                ).sample

            def model_fn(x, sigma):
                if self._cancel:
                    raise InterruptedError("cancelled")
                self._step += 1
                t = sigma_to_t(sigma)
                # diffusers' EulerDiscreteScheduler.scale_model_input: the
                # UNet expects unit-variance input at each noise level
                x_in = x / (sigma ** 2 + 1.0).sqrt()
                c = unet_forward(prompt_embeds, pooled, x_in, t)
                uc = unet_forward(neg_embeds, neg_pooled, x_in, t)
                if progress and self._step <= self._steps_total:
                    progress(self._step / self._steps_total,
                             f"step {self._step}/{self._steps_total}")
                # UNet output is an epsilon prediction -> convert to x0
                eps = uc + request.cfg * (c - uc)
                return x - sigma * eps

            self._steps_total = request.steps
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

            model_wrap = _ModelSamplingDiscrete(model_fn, sigma_data=SIGMA_DATA)
            sigmas = sigmas_fn(
                request.steps, model_wrap.min_sigma, model_wrap.max_sigma, device=device
            )
            # seeded initial noise: same seed -> same image (A1111-compatible).
            # k-diffusion's state is the noisy sample itself, so scale the
            # unit prior to the max noise level (sigma_max = 10).
            gen = torch.Generator(device=device).manual_seed(request.seed)
            noise = torch.randn((1, 4, lh, lw), device=device, dtype=torch.float16,
                                generator=gen) * model_wrap.max_sigma
            # the seed is threaded into the sampler so SDE/ancestral samplers
            # derive their per-step noise deterministically from it
            samples = sampler_fn(model_wrap, noise, sigmas, extra_args={"seed": request.seed})

            # sampling ran in VAE-scaled latent space -> unscale before decode
            latents = samples / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(latents, return_dict=False)[0]
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
