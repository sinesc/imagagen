"""FLUX.1 backend: quantized transformer + dual text encoders + flow-matching
Euler.

Second model family alongside SDXL (see :mod:`imagagen.backends.sdxl`).

Architecture (FLUX.1-dev):
- the transformer is a MMDiT (:class:`~imagagen.flux_transformer.
  FluxTransformer2DModel`) whose ``nn.Linear`` weights live quantized on the
  GPU (GGUF Q4_K, Kijai FP8 or bitsandbytes NF4 — sniffed from the file)
  and are dequantized to bf16 per-op (:mod:`imagagen.gguf_ops`), so ~7 GB
  (Q4_K / NF4) or ~11 GB (FP8) of quantized bytes stay resident instead of
  the ~24 GB bf16 would need;
- with a large resident transformer (FP8), the transformer and the text
  encoders cannot share a 16 GB card: the transformer is swapped to the CPU
  while the encoders run on the GPU (and back again afterwards) -- the same
  unet/encoder swap Forge's memory manager performs.
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
from pathlib import Path

import numpy as np
import torch

from .. import kdiff
from ..encoders.clip import clip_pooled, load_clip_l
from ..encoders.t5 import load_t5_encoder
from ..reference import apply_reference, caption_reference
from ..flux_lora import FluxLoraSpec, merge_module_deltas, parse_lora
from ..flux_transformer import FluxTransformer2DModel
from ..gguf_ops import (
    QuantLinear,
    bake_gguf_model,
    load_flux_state_dict,
    quant_linear_scope,
    sniff_flux_checkpoint,
)
from ..soft_inpaint import adaptive_change_alpha, latent_blend
from . import (
    _PARAM_INIT_LOCK,
    PHASE_DENOISING,
    PHASE_ENCODING,
    GenerateRequest,
    GenerateResult,
    ModelBackend,
    format_loras,
    report_phase,
    shared_enc_cache,
)


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
# the "model default" step count, per checkpoint class: schnell-class
# checkpoints (no guidance embedding, 4-step progressive distillation)
# default to BFL's 4 steps; the dev class to 20
DEFAULT_STEPS = 20
DEFAULT_STEPS_SCHNELL = 4

SAMPLERS = {"Euler": kdiff.sample_euler}
# the untiled 16-channel decoder's working set in GiB per megapixel: the
# peak live conv + SiLU activations at the final full-resolution stage
# (measured 3.0 GiB at 1024x1280; it scales linearly with pixel count)
_VAE_DECODE_GIB_PER_MP = 2.5
# margin on top of the estimate: the CUDA context and small per-allocation
# rounding overhead the live working set number does not include
_VAE_DECODE_MARGIN_GIB = 0.5
# Flux has no scheduler concept: the resolution-dependent shifted sigma
# schedule is computed internally (kdiff.get_sigmas_flux).
SCHEDULERS = {}


class FluxBackend(ModelBackend):
    can_park = True  # transformer, VAE and text encoders all move cleanly
    default_sampler = "Euler"
    default_scheduler = "Flow"
    supports_loras = True  # runtime additive deltas (see _apply_loras)
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
        self._active_loras: tuple[tuple[str, float], ...] | None = None
        # path str -> (mtime, parsed specs); LoRA files are immutable per
        # (path, mtime), so a re-apply of the same set skips the re-read
        self._lora_cache: dict[str, tuple[float, list[FluxLoraSpec]]] = {}
        # (prompt, negative-or-None) -> the encoded embeddings on the CPU
        # (few MB each, LRU); SHARED across this model's per-GPU instances
        # so a pre-encoded wildcard batch (queue.py) is a hit on every
        # GPU; see _get_encoding
        self._enc_cache = shared_enc_cache(info.family, info.id)

        from diffusers import AutoencoderKL
        from diffusers.utils.logging import disable_progress_bar
        from transformers import AutoTokenizer, CLIPTokenizer

        disable_progress_bar()

        infra = config.infra_dir
        # --- checkpoint: detect format + read the state dict. This is slow
        #     disk I/O that only BUILDS Parameter objects (it does not register
        #     them in a module or create tensors via the constructors that
        #     peft/accelerate monkey-patch), so it runs OUTSIDE
        #     _PARAM_INIT_LOCK to avoid blocking concurrent parameter work on
        #     another GPU for the whole read.
        kind = sniff_flux_checkpoint(self.model_path)
        sd = load_flux_state_dict(self.model_path, kind)
        # guidance-embedding checkpoints (FLUX.1-dev) carry the guidance_in
        # MLP weights; schnell-class checkpoints don't and can only run
        # classic CFG (distilled requested is silently demoted, like Forge)
        self.supports_distilled = "guidance_in.in_layer.weight" in sd
        if not self.supports_distilled:
            log.info(
                "%s: schnell-class checkpoint (no guidance embedding): "
                "distilled mode unavailable; the model runs classic CFG "
                "(built-in cfg 1.0 needs no unconditional pass)",
                self.name)
        # the architecture follows the checkpoint: without guidance_in the
        # layer would be an unloaded nn.Identity that crashes on forward
        arch = dict(_TRANSFORMER_ARCH, guidance_embed=self.supports_distilled)

        # Every step below CREATES or MOVES module parameters (VAE,
        # transformer, text encoders). diffusers' VAE from_single_file
        # (low_cpu_mem_usage) applies accelerate's GLOBAL register_parameter /
        # tensor-constructor monkey-patch, and the .to() device moves route
        # through nn.Module.register_parameter; a concurrent model load on
        # another GPU would otherwise interleave its patch with ours and
        # corrupt the parameter registration (see _PARAM_INIT_LOCK).
        with _PARAM_INIT_LOCK:
            # --- VAE (bf16) ---
            vae_path = infra / "ae.safetensors"
            vae_cfg = config.root / "res" / "ae.config.json"
            self.vae = AutoencoderKL.from_single_file(
                str(vae_path), config=str(vae_cfg), torch_dtype=torch.bfloat16
            )
            self.vae = self.vae.to(device)
            self.vae.enable_slicing()

            # --- transformer (quantized -> per-op bf16) ---
            # build on CPU (the dummy weights are tiny); the quantized bytes
            # load on CPU and move to the GPU with .to(device).
            with quant_linear_scope(kind, torch.device("cpu"), torch.bfloat16):
                self.transformer = FluxTransformer2DModel(**arch)
            self.transformer.load_state_dict(sd, strict=False)
            del sd
            self.transformer.to(device)
            if kind == "gguf":
                bake_gguf_model(self.transformer)
            self.transformer.eval()

            # --- text encoders ---
            # `te_device` (GPU by default) is where the encoders *run* during a
            # generation. They load on the CPU first and only move to the GPU
            # when they fit next to the transformer (FP8 checkpoints: 11.1 GiB
            # transformer + 4.6 GiB T5-XXL do not share a 16 GiB card);
            # otherwise they stay on the CPU and are brought to the GPU per
            # generation, swapping the transformer out while they run (see
            # _generate_impl, Forge's memory manager performs the same swap).
            # With text_encoder_device=cpu they never leave the CPU.
            self.te_device = "cpu" if config.text_encoder_device == "cpu" else device
            self.clip = load_clip_l(str(infra / "clip_l.safetensors"), "cpu")
            self.t5 = load_t5_encoder(
                str(infra / "t5xxl.safetensors"), "cpu"
            )
            self.clip_tok = CLIPTokenizer.from_pretrained(str(config.res_dir / "tokenizer_clip"))
            self.t5_tok = AutoTokenizer.from_pretrained(str(config.res_dir / "tokenizer_t5"))
            # resident bytes of the text encoders; the fit check below and the
            # per-generate swap decision (_generate_impl) compare this against
            # the free VRAM
            self._encoder_bytes = (
                sum(p.numel() * p.element_size() for p in self.clip.parameters())
                + sum(p.numel() * p.element_size() for p in self.t5.parameters())
            )
            if self.te_device != "cpu" and self._encoder_bytes + (1 << 30) <= torch.cuda.mem_get_info(device)[0]:
                # fits next to the transformer: co-resident, no swaps
                self._text_to(self.te_device)

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

    def _get_encoding(self, prompt: str, negative: str | None,
                      progress=None):
        """(context, y) -- plus (context_neg, y_neg) when ``negative`` is
        not None -- on the model's device.

        Encodings are cached on the CPU (LRU, a few MB each, shared across
        this model's per-GPU instances) keyed by (prompt, negative-or-None):
        the encoding is deterministic, so a batch of identical prompts -- or
        any later repeat of the same prompt -- reuses the first result
        instead of running the ~4.6 GiB T5 encoder again. A cache hit also
        skips the transformer swap that an uncached encode would trigger.

        ``progress`` (when given) receives :data:`PHASE_ENCODING` on a cache
        miss, i.e. whenever the encoder really runs (see
        :meth:`ModelBackend.generate`).
        """
        key = (prompt, negative)
        entry = self._enc_cache.get(key)
        if entry is not None:
            self._enc_cache.move_to_end(key)
            # A hit skips the encode, so the post-encode offload in
            # _encode_pairs never runs: if the co-resident fit check left
            # the T5/CLIP encoders on the GPU, drop them now, or they sit
            # next to the transformer through sampling and starve the VAE
            # decode (see zimage's note)
            if (self.te_device != "cpu"
                    and next(self.t5.parameters()).device == self.device):
                self._text_offload()
            tensors = [t.to(self.device) for t in entry]
            if negative is not None:
                return tensors[0], tensors[1], (tensors[2], tensors[3])
            return tensors[0], tensors[1], None
        report_phase(progress, PHASE_ENCODING)
        self._encode_pairs([key])
        entry = self._enc_cache[key]
        tensors = [t.to(self.device) for t in entry]
        if negative is not None:
            return tensors[0], tensors[1], (tensors[2], tensors[3])
        return tensors[0], tensors[1], None

    def _encode_pairs(self, keys: list[tuple[str, str | None]],
                      reserve: int | None = None) -> None:
        """Bring the T5/CLIP encoders to ``te_device`` ONCE, encode every
        (prompt, negative-or-None) key not yet in the shared cache, and
        park the encoders again.

        The ~4.6 GiB T5 crosses the bus at most once per call, however many
        prompts it encodes: the generation queue pre-encodes a whole
        wildcard batch here (via :meth:`pre_encode`) instead of every task
        of the batch triggering its own transformer swap. With the GPU
        text-encoder default, a large resident transformer (FP8 ~11.1 GiB)
        and the text encoders (T5 ~4.6 GiB + CLIP ~0.2 GiB) may not share a
        16 GB card; Forge's memory manager unloads the unet while the text
        encoders run. We do the same swap when the encoders are parked on
        the CPU for this reason (see __init__): park the transformer on the
        CPU for the encode, and bring it back after the encoders are
        parked again.

        Each unique text (prompt or negative) is encoded once; every key's
        entry is assembled from the shared parts ((prompt, None) entries
        carry only the prompt parts).

        ``reserve`` is the number of cache entries the caller expects to be
        resident afterwards: the LRU cap is raised to it so a pre-encoded
        batch is not evicted before its slowest task has run.
        """
        keys = list(dict.fromkeys(keys))
        missing = [k for k in keys if self._enc_cache.get(k) is None]
        if not missing:
            return
        if reserve is not None:
            self._enc_cache.bump(reserve)
        swapped = False
        if self.te_device != "cpu" and next(self.t5.parameters()).device != self.device:
            free, _ = torch.cuda.mem_get_info(self.device)
            log.info(
                "%s: transformer swapped to the CPU for text encoding of %d "
                "prompt(s) (%.2f GiB encoders, %.2f GiB free)",
                self.name, len(missing), self._encoder_bytes / 2**30, free / 2**30)
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
            swapped = True
        self._text_to(self.te_device)
        with torch.inference_mode():
            # sorted: a stable encode order (the result does not depend on
            # it, but logs and tests stay deterministic)
            texts = sorted({p for p, _ in missing}
                           | {n for _, n in missing if n is not None})
            parts: dict[str, tuple] = {}
            for t in texts:
                context, y = self._encode_prompt(t)
                # CPU copies for the cache (with te_device=cpu the encoders
                # stay in system RAM)
                parts[t] = (context.cpu(), y.cpu())
            for key in keys:
                if self._enc_cache.get(key) is None:
                    p, n = key
                    self._enc_cache[key] = (
                        parts[p] + parts[n] if n is not None else parts[p])
                    self._enc_cache.move_to_end(key)
        while len(self._enc_cache) > self._enc_cache.max:
            self._enc_cache.popitem(last=False)
        self._text_offload()
        if swapped:
            self.transformer.to(self.device)
            torch.cuda.empty_cache()

    def pre_encode(self, pairs: list[tuple[str, str]],
                   reserve: int | None = None) -> None:
        """One encoder session for a whole batch (see ModelBackend).

        For every (prompt, negative) pair an entry is stored under BOTH
        keys a generation could look up: (prompt, negative) and
        (prompt, None) -- which one a generation consumes depends on the
        effective distilled / classic-CFG mode, resolved only when the
        request runs. The variants share the same prompt parts, so
        covering both costs nothing.
        """
        keys: list[tuple[str, str | None]] = []
        for p, n in pairs:
            keys.append((p, None))
            keys.append((p, n))
        with self._lock:
            self._encode_pairs(keys, reserve)

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
        # inference_mode: the VAE's default-grad weights would otherwise
        # build an autograd graph, and the tiny init latent would pin the
        # entire encoder's activation tensors (~6 GiB at 1024 px) for the
        # whole generate call
        with torch.inference_mode():
            lat = self.vae.encode(t.to(self.vae.dtype)).latent_dist.mode().float()
        return (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def cancel(self) -> None:
        self._cancel = True

    def unload(self) -> None:
        with self._lock:
            # the .to() moves re-register every parameter via
            # nn.Module.register_parameter, which a concurrent model load on
            # another GPU may have globally monkey-patched (see
            # _PARAM_INIT_LOCK) -- hold it so the move never interleaves
            with _PARAM_INIT_LOCK:
                self.transformer.to("cpu")
                self._text_offload()
                self.vae.to("cpu")
            import gc

            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def load(self) -> None:
        """Bring a parked model back to its GPU (inverse of ``unload``).

        Transformer and VAE return to the GPU; the text encoders re-evaluate
        the co-residency decision from ``__init__`` so an FP8 card parks them
        on the CPU again while a large transformer sits resident.
        """
        with self._lock:
            # hold _PARAM_INIT_LOCK across the .to() moves: they re-register
            # every parameter via nn.Module.register_parameter, which a
            # concurrent model load on another GPU may have globally
            # monkey-patched (see _PARAM_INIT_LOCK)
            with _PARAM_INIT_LOCK:
                self.transformer.to(self.device)
                self.vae.to(self.device)
                if (
                    self.te_device != "cpu"
                    and self._encoder_bytes + (1 << 30)
                    <= torch.cuda.mem_get_info(self.device)[0]
                ):
                    self._text_to(self.te_device)
            torch.cuda.empty_cache()

    # -------------------------------------------------------------- lora ----

    def _lora_path(self, name: str) -> str | None:
        """LoRA name (no extension) -> path under ``loras/flux/``.

        The suffix is *appended*, never ``Path.with_suffix``: names may
        themselves contain dots ("MetalSuitFlux1.0"), and with_suffix would
        replace the ".0" instead of adding ".safetensors".
        """
        if self.config is None:
            return None
        p = self.config.loras_dir / "flux" / (name + ".safetensors")
        return str(p) if p.is_file() else None

    def _parse_lora(self, path: str) -> list[FluxLoraSpec]:
        """Read + key-map a Flux LoRA (cached by ``(path, mtime)``).

        Raises ``ValueError`` (unknown/unmappable keys, see flux_lora) on
        files this transformer cannot consume.
        """
        cached = self._lora_cache.get(path)
        mtime = Path(path).stat().st_mtime
        if cached is not None and cached[0] == mtime:
            return cached[1]
        specs = parse_lora(path)
        self._lora_cache[path] = (mtime, specs)
        return specs

    def _module_for(self, path: str):
        """Transformer submodule for a delta spec's module path."""
        module = self.transformer
        for part in path.split("."):
            if not hasattr(module, part):
                raise ValueError(f"no transformer module for LoRA path {path!r}")
            module = getattr(module, part)
        return module

    def _merge_lora_deltas(self, loras: tuple[tuple[str, float], ...]):
        """Resolve + map + merge a LoRA set into per-module delta specs.

        Pure: returns ``{module path: [(A, B, r0, r1), ...]}`` (CPU
        tensors; the weight * alpha/rank factor is pre-folded into A, and
        deltas covering adjacent row slices of one module are merged so
        each module costs ONE extra GEMM pair). Raises ``ValueError`` on a
        missing file, an unmappable key, or a shape mismatch -- nothing is
        mutated here, so a failing set leaves no partial state.
        """
        per_module: dict[str, dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]]] = {}
        for name, weight in loras:
            if weight == 0.0:  # zero-weight LoRA: no delta at all
                continue
            path = self._lora_path(name)
            if path is None:
                raise ValueError(f"Flux LoRA not found: {name!r} (looked in loras/flux/)")
            for spec in self._parse_lora(path):
                rank = spec.A.shape[0]
                if spec.B.shape[1] != rank:
                    raise ValueError(
                        f"{name!r}: mismatched LoRA rank for {spec.module} "
                        f"(A {tuple(spec.A.shape)} vs B {tuple(spec.B.shape)})"
                    )
                if spec.B.shape[0] != spec.r1 - spec.r0:
                    raise ValueError(
                        f"{name!r}: {spec.module} LoRA covers {spec.B.shape[0]} "
                        f"rows, expected {spec.r1 - spec.r0}"
                    )
                a = spec.A * (weight * spec.scale)
                b = spec.B
                if spec.swap:
                    # standard [scale, shift] row order -> ours [shift, scale]
                    half = b.shape[0] // 2
                    b = torch.cat([b[half:], b[:half]], dim=0)
                per_module.setdefault(spec.module, {}) \
                    .setdefault((spec.r0, spec.r1), []).append((a, b))
        return merge_module_deltas(per_module)

    def _apply_loras(self, loras: list[tuple[str, float]]) -> None:
        """Attach the requested LoRAs to the transformer (cached per set).

        Must run under ``self._lock`` (called from ``_generate_impl``
        before the sampling loop, as in the SDXL backend).

        Runtime additive deltas (Forge's *online* mode): the fused
        quantized weights (any format) stay quantized, and each affected
        :class:`~imagagen.gguf_ops.QuantLinear` computes
        ``base + (x @ A.t()) @ B.t()`` per forward. That is
        mathematically identical to Forge's merge-and-re-quantize path
        (which is unavailable here: there is no quantizer for the merged
        weights), lossless, and the deltas are registered module
        parameters, so ``transformer.to()`` / ``unload()`` moves them and
        VRAM scales with the deltas (MetalSuit ~50 MB bf16).

        All-or-nothing: a set becomes ``_active_loras`` only once every
        module is patched; :meth:`_merge_lora_deltas` (parsing, mapping,
        validation) runs before ANY module is touched, and parameter
        registration runs under the shared ``_PARAM_INIT_LOCK``.
        """
        key = tuple((n, float(w)) for n, w in loras)
        if key == self._active_loras:
            return  # identical set: no-op
        merged = self._merge_lora_deltas(key)  # may raise: state untouched
        with _PARAM_INIT_LOCK:
            # drop whatever is currently applied, then install the new set
            self._unload_loras()
            for path, deltas in merged.items():
                module = self._module_for(path)
                if not isinstance(module, QuantLinear):
                    raise TypeError(
                        f"{path!r} is not a QuantLinear ({type(module).__name__})"
                    )
                for a, _b, _r0, _r1 in deltas:
                    if a.shape[1] != module.in_features:
                        raise ValueError(
                            f"{path!r}: LoRA input dim {a.shape[1]} != "
                            f"module in_features {module.in_features}"
                        )
                module.set_lora_deltas(deltas)
        self._active_loras = key
        log.info("Flux LoRA load: %s", format_loras(list(key)))

    def _unload_loras(self) -> None:
        """Clear every module's delta state (no cache bookkeeping).

        Runs under ``_PARAM_INIT_LOCK``. Clears the *actual* module state
        (not just ``_active_loras``), so leftover deltas from any earlier
        state are removed too.
        """
        for module in self.transformer.modules():
            if getattr(module, "lora_deltas", None):
                module.clear_lora_deltas()

    # ----------------------------------------------------------- generate ----

    def generate(self, request: GenerateRequest, progress=None) -> GenerateResult:
        with self._lock:
            return self._generate_impl(request, progress)

    def _generate_impl(self, request: GenerateRequest, progress=None) -> GenerateResult:
        self._cancel = False
        device = self.device
        # LoRA deltas must be in place before the sampling loop (as in the
        # SDXL backend); runs under self._lock via generate()
        self._apply_loras(request.loras)
        # Flux is 8x downsample then 2x2 patchify -> snap to 16 px
        w = max(16, (request.width // 16) * 16)
        h = max(16, (request.height // 16) * 16)
        lh, lw = h // 8, w // 8
        image_seq_len = (lh // 2) * (lw // 2)

        # VAE tiling: vae-tiling=true in imagagen.ini forces it; otherwise
        # automatic above 2 MP, where the untiled 16-channel decode working
        # set (~4 GiB/MP on top of the ~7 GiB of resident weights) OOMs a
        # 16 GB card. Below that threshold tiling stays OFF: the flux VAE
        # decoder is strongly non-local (a large receptive field), so unlike
        # the SDXL VAE a tiled decode changes the image, not just the seams
        # (measured max ~45/255 at 512 px tiles, 1024 px image) -- a crash
        # at 2 MP is the floor, quality is not worth trading below it.
        # The small tile size (512 px / 64 latent) is what makes images tile
        # at all: the VAE default (1024 / 128) only kicks in above 1024 px.
        if self.config is not None:
            self.vae.use_tiling = self.config.vae_tiling or w * h > 2_000_000
            if self.vae.use_tiling:
                self.vae.tile_sample_min_size = 512
                self.vae.tile_latent_min_size = 64
                self.vae.tile_overlap_factor = 0.5

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
        # classic CFG at cfg <= 1.0 (the schnell-class built-in) reduces to
        # v = v_cond: the unconditional pass and the negative prompt encode
        # are skipped entirely (the SDXL backend and Forge do the same)
        uncond = (not distilled) and cfg > 1.0
        g = torch.full((1,), guidance, device=device, dtype=torch.bfloat16)

        # --- text encode --- the encoding is deterministic per prompt, so
        # identical prompts reuse the cached result (no encoder run, no
        # transformer swap -- see _get_encoding). The negative prompt is
        # encoded only when the unconditional pass runs (classic CFG with
        # cfg > 1).
        # Reference image (phase 1): the VLM caption rides on the positive
        # prompt BEFORE encoding (the encoding cache keys on the full
        # text). The captioner towers (~5.3 GB) do not fit next to the
        # resident Flux transformer: it is swapped to the CPU for the
        # caption (the same encode-swap the text encoders trigger).
        prompt = request.prompt
        reference_caption = None
        if request.reference_image is not None:
            def _ref_park():
                self.transformer.to("cpu")
                torch.cuda.empty_cache()

            def _ref_restore():
                self.transformer.to(self.device)
                torch.cuda.empty_cache()

            reference_caption = caption_reference(
                request.reference_image, device, self.config,
                park=_ref_park, restore=_ref_restore, progress=progress)
            prompt = apply_reference(prompt, reference_caption)
        encoding = self._get_encoding(
            prompt, request.negative_prompt if uncond else None,
            progress=progress)
        context, y, neg = encoding
        if uncond:
            context_neg, y_neg = neg

        # --- sigma schedule + noise ---
        # the "model default" sentinel (request.steps <= 0) resolves to the
        # built-in for the checkpoint class (schnell class: 4 steps, the
        # BFL default for the 4-step distilled model; dev class: 20)
        steps = request.steps if request.steps > 0 else (
            DEFAULT_STEPS_SCHNELL if not self.supports_distilled
            else DEFAULT_STEPS
        )
        steps = max(1, steps)
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
        nmask = None  # (16, lh, lw): the inpaint region (1), for soft inpaint
        soft = request.soft_inpainting
        if soft is not None and not soft.enabled:
            soft = None  # a disabled object behaves exactly like absent
        sigma_sched = sigmas
        if request.init_image is not None:
            init_latent = self._encode_init(request.init_image, device, (w, h))
            # A1111: t_enc = int(min(strength, 0.999) * steps); the schedule
            # starts at sigmas[steps - 1 - t_enc] (strength 1.0 -> full)
            t_enc = int(min(request.denoise, 0.999) * steps)
            sigma_sched = sigmas[steps - 1 - t_enc :]
            # flow-matching state at the truncated start sigma: the valid
            # state at noise level sigma is sigma*noise + (1-sigma)*latent
            # (Forge's PredictionFlux 'const' noise_scaling). The SDXL-style
            # `latent + sigma*noise` is NOT a valid flow state: the model
            # reads it as a properly noised init image amplified by
            # 1/(1-sigma) (~5x at sigma 0.8) and denoises it into a bright,
            # high-contrast, blurry reconstruction (the "burned" inpaint
            # with a bright halo around the mask)
            s0 = sigma_sched[0]
            noise = s0 * noise + (1.0 - s0) * init_latent
            if request.init_mask is not None:
                # (1,1,H,W) -> bilinear to the latent grid; this shape
                # broadcasts against the (1,16,lh,lw) solver tensors
                m = torch.from_numpy(
                    np.ascontiguousarray(request.init_mask)
                ).to(device).float().div(255.0)[None, None]
                m = torch.nn.functional.interpolate(
                    m, size=(lh, lw), mode="bilinear")
                # Sampling mask = the 50%-level BINARY mask, not the
                # feathered brush: the flux VAE decoder is strongly
                # non-local (receptive field >= 63 px), so it rings a
                # feathered latent blend (the protected/regenerated mix
                # across the brush feather) into a visible dark/bright halo
                # ring along the feather (measured: a ~10-15 px oscillation
                # band centred on the feather). A 1-latent-px step decodes
                # invisibly (the hard-brush case, no halo). The feather is
                # applied only by the final pixel composite
                # (InpaintPlan.postprocess, the original init_mask).
                m_lock = (m > 0.5).float()
                mask_t = 1.0 - m_lock
                # only the inpaint region starts re-noised; the protected
                # region starts at the clean init latent
                noise = noise * (1.0 - mask_t) + init_latent * mask_t
                if soft is not None:
                    # the inpaint weight (1) replicated across all 16
                    # channels, as seen by the soft-inpaint latent blend and
                    # the adaptive pixel composite (hardened like mask_t,
                    # so the blend is per-region, not feathered)
                    nmask = m_lock[0, 0].unsqueeze(0).expand(16, lh, lw)
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
                # fresh noise re-noised to the current sigma level (same
                # 'const' flow scaling as the init state)
                xwork = x * (1.0 - mask_t) \
                    + (sigma * torch.randn_like(x)
                       + (1.0 - sigma) * init_latent) * mask_t
            x_bf = xwork.to(torch.bfloat16)
            t = sigma.to(torch.bfloat16)
            v_cond = transformer(x_bf, t, context, y, guidance=g)
            if uncond:
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
                if soft is not None:
                    # soft inpainting (Forge builtin extension): instead of
                    # the hard lock, the x0 estimate is blended with the
                    # init latent; the blend weight follows the mask opacity
                    # and the current sigma (power -> when it happens,
                    # scale -> how strongly), and the vector magnitudes are
                    # interpolated under the detail-preservation exponent so
                    # the blend does not wash out contrast
                    soft_t = torch.pow(
                        nmask,
                        (sigma ** soft.mask_blend_power)
                        * soft.mask_blend_scale,
                    )
                    x0 = latent_blend(soft, init_latent, x0, soft_t)
                else:
                    # hard lock: the protected x0 estimate is locked back to
                    # the init latent
                    x0 = x0 * (1.0 - mask_t) + init_latent * mask_t
            return x0

        self._step = 0
        self._steps_total = max(1, len(sigma_sched) - 1)

        report_phase(progress, PHASE_DENOISING)
        with torch.inference_mode():
            x = kdiff.sample_euler(model_fn, noise, sigma_sched, extra_args={})
            if init_latent is not None and mask_t is not None and soft is None:
                # final lock: the protected region is exactly the init
                # latent (skipped with soft inpainting: the protected
                # content is soft-blended at every step and restored by
                # the adaptive pixel composite below instead)
                x = x * (1.0 - mask_t) + init_latent * mask_t

            # --- VAE decode (kept under inference_mode so the result has no grad) ---
            latents = (x / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(torch.bfloat16)
            # the untiled decode's working set may not fit next to the large
            # resident transformer (FP8 ~11.1 GiB) on a 16 GiB card; the
            # allocator then OOMs on every large intermediate and retries
            # after freeing cached pool blocks (a wall of CUDACachingAllocator
            # warnings, physical free down to a few MiB). Give the decode the
            # whole pool back first; if it still does not fit, Forge's memory
            # manager unloads the unet for the VAE -- do the same
            parked = False
            if not self.vae.use_tiling:
                needed = _VAE_DECODE_GIB_PER_MP * w * h / 1e6 * (1 << 30) \
                    + _VAE_DECODE_MARGIN_GIB * (1 << 30)
                free, _ = torch.cuda.mem_get_info(device)
                if free < needed:
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info(device)
                    if free < needed:
                        log.info(
                            "%s: parking the transformer for the untiled VAE "
                            "decode (%.2f GiB free, ~%.1f GiB working set)",
                            self.name, free / 2**30,
                            _VAE_DECODE_GIB_PER_MP * w * h / 1e6)
                        self.transformer.to("cpu")
                        torch.cuda.empty_cache()
                        parked = True
            try:
                image = self.vae.decode(latents, return_dict=False)[0].float()
            except torch.cuda.OutOfMemoryError:
                # the estimate above missed this size; same rescue
                if not parked:
                    log.warning(
                        "%s: untiled VAE decode OOMed, retrying with the "
                        "transformer on the CPU", self.name)
                    import gc

                    gc.collect()
                    self.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    parked = True
                image = self.vae.decode(latents, return_dict=False)[0].float()
            if parked:
                self.transformer.to(self.device)
            image = (image / 2 + 0.5).clamp(0, 1)
        arr = (image.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        if soft is not None and mask_t is not None:
            # Forge's post_sample pixel composite: composite the original
            # image back over the generation wherever the latents did not
            # change much (locally percentile-normalised distance,
            # threshold scaled by the original mask, blurred at image scale)
            c = adaptive_change_alpha(soft, nmask, init_latent, x, w, h)
            base = np.ascontiguousarray(request.init_image)
            if base.shape[:2] != (h, w):
                from PIL import Image

                base = np.asarray(
                    Image.fromarray(base).resize((w, h), Image.LANCZOS)
                )
            arr = arr.astype(np.float32) * c[..., None] \
                + base.astype(np.float32) * (1.0 - c[..., None])
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # give the sampling/decode working set back to the driver so the
        # idle footprint is the resident weights (~7 GiB), not the pool's
        # peak
        torch.cuda.empty_cache()
        return GenerateResult(
            image=arr, seed=request.seed,
            steps=steps,
            reference_caption=reference_caption,
            # the effective CFG for the metadata: the embedded guidance in
            # distilled mode, the CFG value in classic mode
            cfg=guidance if distilled else cfg,
            # the EFFECTIVE mode (a schnell-class checkpoint silently
            # demotes a distilled request to classic CFG)
            distilled=distilled)


def from_config(config) -> ModelBackend | None:
    """Load the first discovered Flux model, or None when nothing exists."""
    from ..models import ModelRegistry

    registry = ModelRegistry(config)
    flux = [m for m in registry.models if m.family == "flux"]
    if not flux or not registry.gpus:
        return None
    return registry.backend_for(flux[0].id)
