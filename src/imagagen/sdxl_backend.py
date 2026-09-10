"""SDXL backend: diffusers pipeline + vendored k-diffusion sampling loop.

Phase 1 ships one sampler (Euler a) and one scheduler (Karras); both are
chosen from small registries so Phase 5's plugin system can grow them without
touching this file.

SDXL specifics (matching A1111/Forge k-diffusion behavior):
- sigma_data = 1.0, karras sigma range [0.1, 10]
- the UNet timestep is the sigma value directly
- VAE latents are unscaled during sampling; divide by 0.18215 before decode
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np

from . import kdiff
from .backends import GenerateRequest, GenerateResult, ModelBackend

SIGMA_DATA = 1.0
VAE_SCALING = 0.18215


class _ModelSamplingDiscrete:
    """k-diffusion ModelSamplingDiscrete: t == sigma, x0 == xphi."""

    def __init__(self, fn, sigma_data: float = 1.0):
        self.fn = fn
        self.sigma_data = sigma_data
        self.max_sigma = sigma_data * 10.0
        self.min_sigma = sigma_data * 0.1

    def __call__(self, x, sigma):
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
    def __init__(self, info) -> None:
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.model_id = info.id
        self.name = info.name
        self.model_hash = _model_hash(str(info.path))
        self.device = torch.device("cuda:0")
        self._lock = threading.Lock()
        self._cancel = False
        self._step = 0
        self._steps_total = 1

        self.pipe = StableDiffusionXLPipeline.from_single_file(
            str(info.path), torch_dtype=torch.float16
        )
        # Swap in the improved fp16-stable VAE (ft-MSE) if present: the
        # stock SDXL VAE overflows to NaN in fp16 at high-magnitude latents.
        vae_path = info.path.parent.parent / "sdxl_vae.safetensors"
        if vae_path.is_file():
            from diffusers import AutoencoderKL

            self.pipe.vae = AutoencoderKL.from_single_file(
                str(vae_path), torch_dtype=torch.float16
            )
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
        self.pipe.to("cpu")

    def generate(self, request: GenerateRequest, progress=None) -> GenerateResult:
        import torch

        self._cancel = False
        pipe = self.pipe
        device = self.device
        w = max(8, (request.width // 8) * 8)
        h = max(8, (request.height // 8) * 8)
        lh, lw = h // 8, w // 8

        with torch.inference_mode():
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

            def unet_forward(emb, pool, x, t):
                added = {"time_ids": time_ids, "text_embeds": pool}
                return unet(
                    x, t, encoder_hidden_states=emb, added_cond_kwargs=added
                ).sample

            def model_fn(x, sigma):
                if self._cancel:
                    raise InterruptedError("cancelled")
                self._step += 1
                t = sigma.to(torch.float32)
                c = unet_forward(prompt_embeds, pooled, x, t)
                uc = unet_forward(neg_embeds, neg_pooled, x, t)
                if progress and self._step <= self._steps_total:
                    progress(self._step / self._steps_total,
                             f"step {self._step}/{self._steps_total}")
                return uc + request.cfg * (c - uc)

            self._steps_total = request.steps
            self._step = 0

            # --- sampler / scheduler registries (Phase 5 grows these into plugins) ---
            sampler_name = request.sampler or self.default_sampler
            scheduler_name = request.scheduler or self.default_scheduler
            sampler_fn = SAMPLERS[sampler_name]
            sigmas_fn = SCHEDULERS[scheduler_name]

            model_wrap = _ModelSamplingDiscrete(model_fn, sigma_data=SIGMA_DATA)
            sigmas = sigmas_fn(
                request.steps, model_wrap.min_sigma, model_wrap.max_sigma, device=device
            )
            # seeded initial noise: same seed -> same image (A1111-compatible)
            gen = torch.Generator(device=device).manual_seed(request.seed)
            noise = torch.randn((1, 4, lh, lw), device=device, dtype=torch.float16,
                                generator=gen)
            samples = sampler_fn(model_wrap, noise, sigmas)

            latents = samples / VAE_SCALING
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
