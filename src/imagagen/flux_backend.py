"""FLUX.1 backend: GGUF transformer + dual text encoders + flow-matching Euler.

Second model family alongside SDXL (see :mod:`imagagen.sdxl_backend`).

Architecture (FLUX.1-dev, the ``agFluxQ4KS`` GGUF):
- the transformer is a MMDiT (:class:`~imagagen.flux_transformer.
  FluxTransformer2DModel`) whose ``nn.Linear`` weights live quantized (Q4_K)
  on the GPU and are dequantized to bf16 per-op (:mod:`imagagen.gguf_ops`),
  so ~6.8 GB of quantized bytes stay resident instead of ~24 GB of bf16;
- the text condition is a 768-dim CLIP-L pooled vector (``y``) plus a
  (256, 4096) T5-XXL context; CLIP-L runs fp16, T5-XXL keeps its fp8 weights
  on the GPU and casts to bf16 per-op (``t5_encoder.Fp8Linear``);
- sampling is plain flow-matching Euler (``kdiff.sample_euler``) over the
  resolution-dependent shifted sigma schedule (``kdiff.get_sigmas_flux``).

FLUX.1-dev is *guidance-distilled*: the "CFG" value can be embedded in the
forward as the ``guidance`` embedding (BFL default 3.5), so a generation is a
single transformer pass per step and no negative prompt is needed. A model
configuration can also select *classic CFG*: the guidance embedding stays at
3.5 (the layer is part of the network, always added to the time embedding),
and the CFG value runs a second, unconditional pass with the negative
prompt, ``v = v_uncond + cfg * (v_cond - v_uncond)`` (two passes per step,
~2x time) -- the ComfyUI/Forge "distilled CFG" toggle. Whether a checkpoint
actually has the ``guidance_in`` embedding is detected from its state dict
(schnell-class checkpoints lack it and can only run classic CFG).

Precision: the transformer and VAE run in bf16 (FLUX is bf16-native; fp16
overflows), and the solver state ``x`` runs in fp32 (the model output is cast
to fp32 before the ``x0 = x - sigma * v`` conversion), mirroring the SDXL
backend's fp32-solver / low-precision-model split.
"""

from __future__ import annotations

import hashlib
import logging
import threading

import numpy as np
import torch

from . import kdiff
from .backends import GenerateRequest, GenerateResult, ModelBackend
from .clip_encoder import clip_pooled, load_clip_l
from .flux_transformer import FluxTransformer2DModel
from .gguf_ops import bake_gguf_model, gguf_linear_scope, load_gguf_state_dict
from .t5_encoder import load_t5_encoder

def _model_hash(path: str) -> str:
    """md5[:16] of the first 100 MB of the checkpoint (the PNG "Model hash")."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            if f.tell() >= 100 * 2**20:
                break
    return h.hexdigest()[:16]


log = logging.getLogger(__name__)

# FLUX.1-dev transformer architecture (from the shipped GGUF tensor shapes).
_TRANSFORMER_ARCH = dict(
    in_channels=16,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10000,
    qkv_bias=False,
    guidance_embed=True,
)

# FLUX flow-matching dynamic-shift parameters (the reference scheduler config).
_FLOW_SHIFT = dict(base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)

# FLUX.1-dev's embedded guidance default (the "CFG" value when the request
# leaves it at the model default; also the fixed embedded value in classic
# CFG mode).
DEFAULT_GUIDANCE = 3.5
# classic CFG (distilled unchecked): the "model default" CFG value
DEFAULT_CFG = 1.0

SAMPLERS = {"Euler": kdiff.sample_euler}
# Flux has no scheduler concept: the resolution-dependent shifted sigma
# schedule is computed internally (kdiff.get_sigmas_flux).
SCHEDULERS = {}


class FluxBackend(ModelBackend):
    default_sampler = "Euler"
    default_scheduler = "Flow"
    supports_loras = False  # LoRA support is not implemented yet
  # sigmas are computed internally, no registry

    is_flux = True

    def __init__(self, info, device, config=None) -> None:
        self.model_id = info.id
        self.name = info.name
        self.model_hash = _model_hash(str(info.path))
        self.device = device
        self.config = config
        self.model_path = str(info.path)
        # serializes generate()/unload(): the modules are not thread-safe
        self._lock = threading.Lock()
        self._cancel = False
        self._step = 0
        self._steps_total = 1

        from diffusers import AutoencoderKL
        from diffusers.utils.logging import disable_progress_bar
        from transformers import AutoTokenizer, CLIPTokenizer

        disable_progress_bar()

        infra = config.infra_dir
        # --- VAE (bf16) ---
        vae_path = infra / "flux_ae.safetensors"
        vae_cfg = config.root / "res" / "flux_ae.config.json"
        self.vae = AutoencoderKL.from_single_file(
            str(vae_path), config=str(vae_cfg), torch_dtype=torch.bfloat16
        )
        self.vae = self.vae.to(device)
        self.vae.enable_slicing()

        # --- transformer (GGUF Q4_K -> per-op bf16) ---
        # build on CPU (the dummy weights are tiny); the quantized bytes load
        # on CPU and move to the GPU with .to(device).
        with gguf_linear_scope(torch.device("cpu"), torch.bfloat16):
            self.transformer = FluxTransformer2DModel(**_TRANSFORMER_ARCH)
        sd = load_gguf_state_dict(self.model_path)
        # guidance-embedding checkpoints (FLUX.1-dev) carry the guidance_in
        # MLP weights; schnell-class checkpoints don't and can only run
        # classic CFG (distilled requested is silently demoted, like Forge)
        self.supports_distilled = "guidance_in.in_layer.weight" in sd
        if not self.supports_distilled:
            log.info("%s: no guidance embedding -- distilled mode unavailable",
                     self.name)
        self.transformer.load_state_dict(sd, strict=False)
        del sd
        self.transformer.to(device)
        bake_gguf_model(self.transformer)
        self.transformer.eval()

        # --- text encoders ---
        self.clip = load_clip_l(str(infra / "clip_l.safetensors"), device)
        self.t5 = load_t5_encoder(str(infra / "t5xxl_fp8_e4m3fn.safetensors"), device)
        self.clip_tok = CLIPTokenizer.from_pretrained(str(config.res_dir / "tokenizer_clip"))
        self.t5_tok = AutoTokenizer.from_pretrained(str(config.res_dir / "tokenizer_t5"))

        # text encoders live on `te_device` (GPU by default); they are parked
        # on CPU between generations to free VRAM for sampling
        self.te_device = "cpu" if config is not None and config.text_encoder_device == "cpu" else device

        # all weights are resident now; release the transient VRAM the load
        # left behind (bake buffers, from_single_file scratch) so the pool
        # is as tight as the live weights, leaving room for the T5 working
        # set during encode and for sampling activations
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- text ----

    def _encode_prompt(self, prompt: str):
        """Tokenize + encode ``prompt`` -> (context, y).

        ``context`` is the (1, 256, 4096) bf16 T5-XXL hidden state; ``y`` is
        the (1, 768) bf16 CLIP-L pooled vector. The encoders must be on
        ``self.te_device`` (call :meth:`_text_to`).
        """
        device = self.te_device
        # CLIP-L pooled vector y (the 768-dim condition)
        clip_ids = (
            self.clip_tok(
                prompt, padding="max_length", max_length=77, truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
        )
        y = clip_pooled(self.clip, clip_ids).to(torch.bfloat16)
        # T5-XXL context: content tokens, then EOS (1), then pad (0) to 256
        toks = self.t5_tok.encode(prompt, add_special_tokens=False)[:255]
        toks = toks + [1]
        toks = toks + [0] * (256 - len(toks))
        t5_ids = torch.tensor([toks], device=device, dtype=torch.long)
        context = self.t5(t5_ids).to(torch.bfloat16)
        return context, y

    def _text_to(self, device) -> None:
        self.clip.to(device)
        self.t5.to(device)

    def _text_offload(self) -> None:
        self.clip.to("cpu")
        self.t5.to("cpu")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- ctrl ----

    def _encode_init(self, arr: np.ndarray, device, size: tuple[int, int]):
        """VAE-encode an HxWx3 uint8 image to its solver-space latent
        (the deterministic ``latent_dist.mode()`` -- A1111's "Full" encode),
        the inverse of the decode transform ``x / scaling + shift``.

        ``size`` is the snapped ``(w, h)`` generation size. The init image
        arrives at the un-snapped request size, so it is LANCZOS-resized
        when the two differ (the 16-px snap can shrink the request).
        """
        if arr.shape[:2] != (size[1], size[0]):
            from PIL import Image

            arr = np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        t = t.permute(2, 0, 1).float()[None] / 127.5 - 1.0
        lat = self.vae.encode(t.to(self.vae.dtype)).latent_dist.mode().float()
        return (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def cancel(self) -> None:
        self._cancel = True

    def unload(self) -> None:
        with self._lock:
            self.transformer.to("cpu")
            self._text_offload()
            self.vae.to("cpu")
            import gc

            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ----------------------------------------------------------- generate ----

    def generate(self, request: GenerateRequest, progress=None) -> GenerateResult:
        with self._lock:
            return self._generate_impl(request, progress)

    def _generate_impl(self, request: GenerateRequest, progress=None) -> GenerateResult:
        self._cancel = False
        device = self.device
        # Flux is 8x downsample then 2x2 patchify -> snap to 16 px
        w = max(16, (request.width // 16) * 16)
        h = max(16, (request.height // 16) * 16)
        lh, lw = h // 8, w // 8
        image_seq_len = (lh // 2) * (lw // 2)

        # distilled (model configuration default) vs classic CFG: see module
        # docstring. The guidance embedding is ALWAYS active on this
        # architecture; distilled maps the CFG value onto it (one pass),
        # classic CFG keeps it at 3.5 and adds an unconditional pass.
        distilled = request.distilled and self.supports_distilled
        if distilled:
            guidance = request.cfg if request.cfg > 0 else DEFAULT_GUIDANCE
            cfg = None
        else:
            guidance = DEFAULT_GUIDANCE
            cfg = request.cfg if request.cfg > 0 else DEFAULT_CFG
            if not request.distilled:
                log.info("%s: classic CFG %.4g (embedded guidance %.4g)",
                         self.name, cfg, guidance)
        g = torch.full((1,), guidance, device=device, dtype=torch.bfloat16)

        # --- text encode (bring encoders to te_device, then park on CPU) ---
        self._text_to(self.te_device)
        with torch.inference_mode():
            context, y = self._encode_prompt(request.prompt)
            if not distilled:
                context_neg, y_neg = self._encode_prompt(request.negative_prompt)
        if self.te_device != "cpu":
            context = context.to(device)
            y = y.to(device)
            if not distilled:
                context_neg = context_neg.to(device)
                y_neg = y_neg.to(device)
            self._text_offload()

        # --- sigma schedule + noise ---
        steps = max(1, request.steps)
        sigmas = kdiff.get_sigmas_flux(steps, image_seq_len, device, **_FLOW_SHIFT)
        gen = torch.Generator(device=device).manual_seed(request.resolved_seed())
        noise = torch.randn(
            (1, 16, lh, lw), device=device, dtype=torch.float32, generator=gen
        )

        # --- img2img / inpainting (A1111/Forge convention, see sdxl_backend)
        # ---
        # With an init image: start from the init latent instead of pure
        # noise, truncate the sigma schedule at the denoising strength, and
        # (with an init mask) lock the protected region (mask 0) to the init
        # content at every step. init_mask marks the *inpaint* region, so
        # protected = 1 - mask.
        init_latent = None
        mask_t = None
        sigma_sched = sigmas
        if request.init_image is not None:
            init_latent = self._encode_init(request.init_image, device, (w, h))
            # A1111: t_enc = int(min(strength, 0.999) * steps); the schedule
            # starts at sigmas[steps - 1 - t_enc] (strength 1.0 -> full)
            t_enc = int(min(request.denoise, 0.999) * steps)
            sigma_sched = sigmas[steps - 1 - t_enc :]
            # flow-matching const noise_scaling at the truncated start sigma
            # (sigma * noise + (1 - sigma) * latent): the noise amplitude IS
            # the denoise strength
            noise = init_latent + noise * sigma_sched[0]
            if request.init_mask is not None:
                # (1,1,H,W) -> bilinear to the latent grid; this shape
                # broadcasts against the (1,16,lh,lw) solver tensors
                m = torch.from_numpy(
                    np.ascontiguousarray(request.init_mask)
                ).to(device).float().div(255.0)[None, None]
                m = torch.nn.functional.interpolate(
                    m, size=(lh, lw), mode="bilinear")
                mask_t = 1.0 - m
                # only the inpaint region starts re-noised; the protected
                # region starts at the clean init latent
                noise = noise * (1.0 - mask_t) + init_latent * mask_t
        else:
            noise = noise * float(sigmas[0])  # sigmas[0] == 1.0

        transformer = self.transformer

        def model_fn(x, sigma):
            if self._cancel:
                raise InterruptedError("cancelled")
            self._step += 1
            xwork = x
            if mask_t is not None:
                # per-step lock (Forge's cfg_denoiser): before the model
                # call, the protected region (mask_t == 1) is replaced by
                # init latent + fresh noise at the current sigma level
                xwork = x * (1.0 - mask_t) \
                    + (init_latent + torch.randn_like(x) * sigma) * mask_t
            x_bf = xwork.to(torch.bfloat16)
            t = sigma.to(torch.bfloat16)
            v_cond = transformer(x_bf, t, context, y, guidance=g)
            if not distilled:
                if self._cancel:
                    raise InterruptedError("cancelled")
                v_uncond = transformer(
                    x_bf, t, context_neg, y_neg, guidance=g)
                v_cond = v_uncond + cfg * (v_cond - v_uncond)
            v = v_cond.float()
            if progress and self._step <= self._steps_total:
                progress(self._step / self._steps_total,
                         f"step {self._step}/{self._steps_total}")
            x0 = xwork - sigma * v
            if mask_t is not None:
                # hard lock: the protected x0 estimate is locked back to
                # the init latent
                x0 = x0 * (1.0 - mask_t) + init_latent * mask_t
            return x0

        self._step = 0
        self._steps_total = max(1, len(sigma_sched) - 1)

        with torch.inference_mode():
            x = kdiff.sample_euler(model_fn, noise, sigma_sched, extra_args={})
            if init_latent is not None and mask_t is not None:
                # final lock: the protected region is exactly the init
                # latent
                x = x * (1.0 - mask_t) + init_latent * mask_t

            # --- VAE decode (kept under inference_mode so the result has no grad) ---
            latents = (x / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(torch.bfloat16)
            image = self.vae.decode(latents, return_dict=False)[0].float()
            image = (image / 2 + 0.5).clamp(0, 1)
        arr = (image.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        return GenerateResult(image=arr, seed=request.seed)


def from_config(config) -> ModelBackend | None:
    """Load the first discovered Flux model, or None when nothing exists."""
    from .models import ModelRegistry

    registry = ModelRegistry(config)
    flux = [m for m in registry.models if m.family == "flux"]
    if not flux or not registry.gpus:
        return None
    return registry.backend_for(flux[0].id)
