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

FLUX.1-dev is *guidance-distilled*: there is no negative prompt and no
classifier-free-guidance pass. The "CFG" value is embedded in the forward as
the ``guidance`` embedding (default 3.5), so a generation is a single
transformer pass per step.

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
# leaves it at the model default).
DEFAULT_GUIDANCE = 3.5

SAMPLERS = {"Euler": kdiff.sample_euler}
# Flux has no scheduler concept: the resolution-dependent shifted sigma
# schedule is computed internally (kdiff.get_sigmas_flux).
SCHEDULERS = {}


class FluxBackend(ModelBackend):
    default_sampler = "Euler"
    default_scheduler = "Flow"
    supports_loras = False  # LoRA support is not implemented yet
  # sigmas are computed internally, no registry

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

        # guidance-distilled: the "CFG" is the embedded guidance (no uncond)
        guidance = request.cfg if request.cfg > 0 else DEFAULT_GUIDANCE
        g = torch.full((1,), guidance, device=device, dtype=torch.bfloat16)

        # --- text encode (bring encoders to te_device, then park on CPU) ---
        self._text_to(self.te_device)
        with torch.inference_mode():
            context, y = self._encode_prompt(request.prompt)
        if self.te_device != "cpu":
            context = context.to(device)
            y = y.to(device)
            self._text_offload()

        # --- sigma schedule + noise ---
        steps = max(1, request.steps)
        sigmas = kdiff.get_sigmas_flux(steps, image_seq_len, device, **_FLOW_SHIFT)
        gen = torch.Generator(device=device).manual_seed(request.resolved_seed())
        noise = torch.randn(
            (1, 16, lh, lw), device=device, dtype=torch.float32, generator=gen
        )
        noise = noise * float(sigmas[0])  # sigmas[0] == 1.0 (flow-matching)

        transformer = self.transformer

        def model_fn(x, sigma):
            if self._cancel:
                raise InterruptedError("cancelled")
            self._step += 1
            x_bf = x.to(torch.bfloat16)
            t = sigma.to(torch.bfloat16)
            v = transformer(x_bf, t, context, y, guidance=g)
            v = v.float()
            if progress and self._step <= self._steps_total:
                progress(self._step / self._steps_total,
                         f"step {self._step}/{self._steps_total}")
            return x - sigma * v

        self._step = 0
        self._steps_total = len(sigmas) - 1

        with torch.inference_mode():
            x = kdiff.sample_euler(model_fn, noise, sigmas, extra_args={})

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
