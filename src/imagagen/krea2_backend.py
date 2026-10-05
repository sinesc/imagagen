"""Krea 2 (K2) backend: int8 SingleStreamDiT + Qwen3-VL-4B encoder +
flow-matching Euler.

Fourth model family alongside SDXL, Flux and Z-Image (see
:mod:`imagagen.zimage_backend` for the closest sibling). Plan:
``dev/plans/krea2-backend-plan.md``; references: the official
implementation in ``dev/implementations/krea-2`` (inference.py /
mmdit.py / sampling.py / encoder.py / autoencoder.py) and the ComfyUI
checkout (``comfy/ldm/krea2/model.py``, ``comfy/text_encoders/krea2.py``).

Architecture:
- the transformer is a 12B-ish SingleStreamDiT
  (:class:`~imagagen.krea2_transformer.Krea2Transformer`) whose ``blocks``
  projections live as ``int8_tensorwise`` on the GPU (per-row fp32 scales,
  ConvRot Hadamard-rotated weights) and dequantize to bf16 per-op --
  ~12.5 GB resident instead of the ~25 GB a bf16 copy would need; the
  txtfusion/txtmlp/tmlp/tproj/first/last sections are plain bf16;
- the text condition is a 12-layer TAP of the Qwen3-VL-4B text tower
  (``hidden_states[2, 5, 8, ..., 35]``), flattened to (1, seq, 12*2560)
  and fused by the DiT's internal ``txtfusion`` adapter; the encoder
  keeps its plain fp8 projections (~4.8 GB resident) and casts to bf16
  per-op (:mod:`imagagen.krea2_encoder`);
- the VAE is the Qwen-Image VAE (Wan architecture, 16 channels, 8x
  downsample; ``infra/qwen_image_vae.safetensors`` remapped to diffusers'
  ``AutoencoderKLQwenImage``, :mod:`imagagen.krea2_vae`); the latent
  normalization is per-channel mean/std (NOT scaling/shift).

Sampling is plain flow-matching Euler (``kdiff.sample_euler``) over the
resolution-dependent mu schedule (``kdiff.get_sigmas_krea2``). The model
returns the flow velocity: ``x0 = x - sigma * v`` (verified empirically,
the ``dev/smoke_krea2.py`` pattern).

Checkpoint classes (the ``turbo`` flag on the model config entry,
mirroring Flux's ``distilled``):
- raw (default): 28 steps, standard CFG 5.5 (the reference's guidance
  4.5 in standard terms: ``v = vu + cfg*(vc - vu)``), resolution-derived
  mu (linear in the image token count between (256, 0.5) and
  (6400, 1.15));
- turbo: 8 steps, no CFG (effective 1.0, no negative encode), mu pinned
  to 1.15 (the reference's ``--mu`` recommendation for the distilled
  checkpoint).
cfg <= 1.0 skips the unconditional pass (and the negative encode),
exactly as Flux classic.

Precision: transformer and VAE in bf16, the solver state ``x`` in fp32
(the model output is cast to fp32 before the ``x0 = x - sigma * v``
conversion), mirroring the Flux/Z-Image backends.

VRAM (16 GB cards): transformer (~12.5 GB) + VAE (~0.25 GB) leaves no
room for the ~4.8 GB encoder, so the encoder NEVER co-resides by default:
the transformer is parked on the CPU while the encoder runs (and back
again afterwards) -- Forge's memory-manager swap, the Z-Image
``_encode_prompts`` path (the prompt cache makes it once per batch, not
per image). ``text-encoder-device = cpu`` keeps the encoder in system RAM
permanently. The parked (RAM) footprint is ~18 GB (transformer +
encoder + VAE) -- comfortable on the 62 GB host, but note that the park
policy bounds ONE parked model per GPU. ``empty_cache()`` after load and
after every generate (the VAE decode OOM-retry also parks the
transformer, the Flux pattern).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import numpy as np
import torch

from . import kdiff
from .backends import (
    _PARAM_INIT_LOCK,
    GenerateRequest,
    GenerateResult,
    ModelBackend,
    format_loras,
    shared_enc_cache,
)
from .flux_lora import FluxLoraSpec, merge_module_deltas
from .gguf_ops import QuantLinear, quant_dtype_scope
from .krea2_encoder import (
    krea2_prompt_template,
    load_krea2_encoder,
    template_prefix_cut,
)
from .krea2_lora import parse_krea2_lora
from .krea2_transformer import DeltaLinear, Int8Linear, Krea2Transformer
from .krea2_vae import denormalize_latent, load_krea2_vae, normalize_latent
from .soft_inpaint import adaptive_change_alpha, latent_blend


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

# the "model default" parameters per checkpoint class:
# raw: the reference's 28-step schedule, standard CFG 5.5 (= reference
# guidance 4.5), resolution-derived mu
DEFAULT_STEPS_RAW = 28
DEFAULT_CFG_RAW = 5.5
# turbo: 8 steps, no CFG, mu pinned to 1.15 (reference --mu 1.15)
DEFAULT_STEPS_TURBO = 8
DEFAULT_CFG_TURBO = 1.0
TURBO_MU = 1.15

SAMPLERS = {"Euler": kdiff.sample_euler}
# Krea 2 has no scheduler concept: the mu schedule is computed internally
# (kdiff.get_sigmas_krea2).
SCHEDULERS = {}


def _load_krea2_state_dict(path: str) -> dict:
    """Read the K2 transformer checkpoint into ``{key: Parameter}``
    (keys relative to the transformer root -- the checkpoint has no
    prefix; the int8 tensors stay int8 and the per-row ``weight_scale``
    tensors ride along; the ``*.comfy_quant`` U8 markers are skipped).

    Slow disk I/O; run OUTSIDE ``_PARAM_INIT_LOCK`` (it only reads
    tensors, it does not register them in a module).
    """
    from safetensors import safe_open

    sd: dict = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".comfy_quant"):
                continue  # quantization metadata marker, no data
            sd[key] = torch.nn.Parameter(f.get_tensor(key), requires_grad=False)
    return sd


class Krea2Backend(ModelBackend):
    can_park = True  # transformer, VAE and encoder all move cleanly
    default_sampler = "Euler"
    default_scheduler = "Flow"
    supports_loras = True  # runtime additive deltas (see _apply_loras)
    is_krea2 = True

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
        # prompt str -> the encoded context on the CPU (few MB each, LRU);
        # SHARED across this model's per-GPU instances so a pre-encoded
        # wildcard batch (queue.py) is a hit on every GPU; see _get_context
        self._enc_cache = shared_enc_cache(info.family, info.id)

        from diffusers.utils.logging import disable_progress_bar
        from transformers import Qwen2Tokenizer

        disable_progress_bar()

        # --- checkpoint: read the transformer state dict. This is slow disk
        #     I/O (it BUILDS tensors on the CPU but does not register them in
        #     a module), so it runs OUTSIDE _PARAM_INIT_LOCK to avoid blocking
        #     concurrent parameter work on another GPU for the whole read.
        sd = _load_krea2_state_dict(self.model_path)

        # Every step below CREATES or MOVES module parameters (VAE,
        # transformer, text encoder). The .to() device moves route through
        # the globally monkey-patched nn.Module.register_parameter (see
        # _PARAM_INIT_LOCK).
        with _PARAM_INIT_LOCK:
            # --- VAE (bf16; Qwen-Image VAE, per-channel mean/std) ---
            self.vae = load_krea2_vae(config, device)
            self.vae.enable_slicing()

            # --- transformer (int8 -> per-op bf16) ---
            # build on CPU: the Int8Linear dummies are seeded from the
            # dtype scope (no nn.Linear replacement -- the transformer
            # mixes Int8Linear block projections with plain bf16 linears);
            # the int8 bytes load on CPU and move to the GPU with .to(device)
            with quant_dtype_scope(torch.device("cpu"), torch.bfloat16):
                self.transformer = Krea2Transformer()
            result = self.transformer.load_state_dict(sd, strict=False)
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            del sd
            if missing or unexpected:
                raise ValueError(
                    f"Krea 2 checkpoint mismatch for {info.name}: "
                    f"missing={missing[:5]} unexpected={unexpected[:5]}")
            # construction left the plain (non-quantized) linears/norms in
            # fp32 while the Int8Linear weights are int8 with bf16 dummies:
            # cast only the parameters outside the Int8Linear submodules so
            # the quantized weights keep their dtype (a module-level .to()
            # would recurse into the Int8Linear children)
            int8_prefixes = {
                name + "." for name, mod in self.transformer.named_modules()
                if isinstance(mod, Int8Linear)}
            for name, p in list(self.transformer.named_parameters()):
                if not any(name.startswith(pref) for pref in int8_prefixes):
                    p.data = p.data.to(torch.bfloat16)
            self.transformer.to(device)
            self.transformer.eval()

            # --- text encoder (Qwen3-VL-4B, plain fp8) ---
            # `te_device` (GPU by default) is where the encoder runs during a
            # generation. It loads on the CPU first and only moves to the
            # GPU when it fits next to the transformer; otherwise it is
            # brought to the GPU per generation, swapping the transformer
            # out while it runs (see _generate_impl). With
            # text_encoder_device=cpu it never leaves the CPU.
            self.te_device = "cpu" if config.text_encoder_device == "cpu" else device
            self.te = load_krea2_encoder(
                str(config.infra_dir / "qwen_3vl_4b.safetensors"), "cpu")
            self.te_tok = Qwen2Tokenizer.from_pretrained(
                str(config.res_dir / "tokenizer_krea2"))
            # resident bytes of the encoder; the fit check below and the
            # per-generate swap decision compare this against the free VRAM
            self._encoder_bytes = sum(
                p.numel() * p.element_size() for p in self.te.parameters()
            )
            if self.te_device != "cpu" and self._encoder_bytes + (1 << 30) \
                    <= torch.cuda.mem_get_info(device)[0]:
                # fits next to the transformer: co-resident, no swaps
                # (never the case on a 16 GB card -- the transformer alone
                # takes ~12.5 GiB)
                self._text_to(self.te_device)

        # all weights are resident now; release the transient VRAM the load
        # left behind so the pool is as tight as the live weights
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- text ----

    def _encode_prompt(self, prompt: str):
        """Tokenize + encode ``prompt`` -> the (1, seq, 12*2560) bf16
        12-tap conditioning (chat-templated, the system + user-opening
        prefix stripped, the reference ``Krea2TEModel.encode_token_
        weights``). The encoder must be on ``self.te_device`` (call
        :meth:`_text_to`)."""
        device = self.te_device
        text = krea2_prompt_template().format(prompt)
        ids = self.te_tok(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        cut = template_prefix_cut(ids[0].tolist())
        ids = ids[:, cut:].to(device)
        # (1, 12, seq, 2560) -> (1, seq, 12*2560): the DiT's txtfusion
        # unpacks the tap axis again
        taps = self.te(ids)
        return taps.permute(0, 2, 1, 3).reshape(1, -1, 12 * 2560).to(
            torch.bfloat16)

    def _get_context(self, prompt: str):
        """The encoded (1, seq, 12*2560) bf16 context for ``prompt`` on the
        model's device.

        Encodings are cached on the CPU (LRU, a few MB each, shared across
        this model's per-GPU instances): the encoding is deterministic, so
        a batch of identical prompts -- or any later repeat of the same
        prompt -- reuses the first encoding instead of running the ~4.8 GiB
        encoder again. A cache hit also skips the transformer swap that an
        uncached encode would trigger.
        """
        cached = self._enc_cache.get(prompt)
        if cached is not None:
            self._enc_cache.move_to_end(prompt)
            # A hit skips the encode, so the post-encode offload in
            # _encode_prompts never runs: if the co-resident fit check
            # (cold load / parked restore) left the encoder on the GPU,
            # drop it now, or its ~4.8 GiB sits next to the transformer
            # through sampling and starves the VAE decode
            if (self.te_device != "cpu"
                    and next(self.te.parameters()).device == self.device):
                self._text_offload()
            return cached.to(self.device)
        self._encode_prompts([prompt])
        return self._enc_cache[prompt].to(self.device)

    def _encode_prompts(self, prompts: list[str],
                        reserve: int | None = None) -> None:
        """Bring the encoder to ``te_device`` ONCE, encode every prompt not
        yet in the shared cache, and park the encoder again.

        The ~4.8 GiB encoder crosses the bus at most once per call, however
        many prompts it encodes: the generation queue pre-encodes a whole
        wildcard batch here (via :meth:`pre_encode`) instead of every task
        of the batch triggering its own transformer swap. With the GPU
        text-encoder default, the resident transformer (~12.5 GiB) and the
        encoder (~4.8 GiB) never fit on a 16 GiB card; Forge's memory
        manager unloads the unet while the text encoders run. We do the
        same swap when the encoder is parked on the CPU for this reason
        (see __init__): park the transformer on the CPU for the encode,
        then bring it back after the encoder is parked again.

        ``reserve`` is the number of cache entries the caller expects to be
        resident afterwards: the LRU cap is raised to it so a pre-encoded
        batch is not evicted before its slowest task has run.
        """
        missing = [p for p in dict.fromkeys(prompts)
                   if self._enc_cache.get(p) is None]
        if not missing:
            return
        if reserve is not None:
            self._enc_cache.bump(reserve)
        swapped = False
        if self.te_device != "cpu" and next(self.te.parameters()).device != self.device:
            free, _ = torch.cuda.mem_get_info(self.device)
            log.info(
                "%s: transformer swapped to the CPU for text encoding of %d "
                "prompt(s) (%.2f GiB encoder, %.2f GiB free)",
                self.name, len(missing), self._encoder_bytes / 2**30,
                free / 2**30)
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
            swapped = True
        self._text_to(self.te_device)
        with torch.inference_mode():
            for p in missing:
                # CPU copy for the cache (with te_device=cpu the encoder
                # stays in system RAM)
                self._enc_cache[p] = self._encode_prompt(p).cpu()
                self._enc_cache.move_to_end(p)
        while len(self._enc_cache) > self._enc_cache.max:
            self._enc_cache.popitem(last=False)
        self._text_offload()
        if swapped:
            self.transformer.to(self.device)
            torch.cuda.empty_cache()

    def pre_encode(self, pairs: list[tuple[str, str]],
                   reserve: int | None = None) -> None:
        """One encoder session for a whole batch (see ModelBackend): the
        K2 classic path encodes the negative prompt as well (CFG), so
        both sides of every pair are encoded."""
        with self._lock:
            self._encode_prompts(
                [p for p, _ in pairs] + [n for _, n in pairs if n], reserve)

    def _text_to(self, device) -> None:
        self.te.to(device)

    def _text_offload(self) -> None:
        self.te.to("cpu")
        torch.cuda.empty_cache()

    def _transformer_out(self) -> bool:
        """Park the transformer on the CPU to free its VRAM for the VAE
        (encode/decode). Returns True when it was swapped and must be
        brought back with :meth:`_transformer_in`."""
        if next(self.transformer.parameters()).device == self.device:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
            return True
        return False

    def _transformer_in(self) -> None:
        self.transformer.to(self.device)

    # ------------------------------------------------------------- ctrl ----

    def _encode_init(self, arr: np.ndarray, device, size: tuple[int, int]):
        """VAE-encode an HxWx3 uint8 image to its solver-space latent
        (the deterministic ``latent_dist.mode()`` -- A1111's "Full" encode),
        normalized to solver space by the VAE's per-channel mean/std:
        ``z = (vae_latent - mean) / std``.

        ``size`` is the snapped ``(w, h)`` generation size; the init image
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
        # entire encoder's activation tensors for the whole generate call
        with torch.inference_mode():
            lat = self.vae.encode(t.to(self.vae.dtype)).latent_dist.mode().float()
        return normalize_latent(self.vae, lat)

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

        Transformer and VAE return to the GPU; the encoder re-evaluates the
        co-residency decision from ``__init__``.
        """
        with self._lock:
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
        """LoRA name (no extension) -> path under ``loras/krea2/``.

        The suffix is *appended*, never ``Path.with_suffix``: names may
        themselves contain dots, and with_suffix would replace the last one
        instead of adding ".safetensors".
        """
        if self.config is None:
            return None
        p = self.config.loras_dir / "krea2" / (name + ".safetensors")
        return str(p) if p.is_file() else None

    def _parse_lora(self, path: str) -> list[FluxLoraSpec]:
        """Read + key-map a Krea 2 LoRA (cached by ``(path, mtime)``).

        Raises ``ValueError`` (unknown/unmappable keys, incomplete pairs,
        see krea2_lora) on files this transformer cannot consume.
        """
        cached = self._lora_cache.get(path)
        mtime = Path(path).stat().st_mtime
        if cached is not None and cached[0] == mtime:
            return cached[1]
        specs = parse_krea2_lora(path)
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

        Pure: returns ``{module path: [(A, B, r0, r1), ...]}`` (CPU tensors;
        the LoRA *weight scale factor is pre-folded into A, and deltas
        covering adjacent row slices of one module are merged so each
        module costs ONE extra GEMM pair). Raises ``ValueError`` on a
        missing file, an unmappable key, or a shape mismatch -- nothing is
        mutated here, so a failing set leaves no partial state.
        """
        per_module: dict[str, dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]]] = {}
        for name, weight in loras:
            if weight == 0.0:  # zero-weight LoRA: no delta at all
                continue
            path = self._lora_path(name)
            if path is None:
                raise ValueError(
                    f"Krea 2 LoRA not found: {name!r} (looked in loras/krea2/)")
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
                per_module.setdefault(spec.module, {}) \
                    .setdefault((spec.r0, spec.r1), []).append((a, spec.B))
        return merge_module_deltas(per_module)

    def _apply_loras(self, loras: list[tuple[str, float]]) -> None:
        """Attach the requested LoRAs to the transformer (cached per set).

        Must run under ``self._lock`` (called from ``_generate_impl``
        before the sampling loop, as in the Z-Image backend).

        Runtime additive deltas: the int8 block weights stay quantized and
        each affected :class:`~imagagen.krea2_transformer.Int8Linear`
        computes ``base + (x_rot @ A.t()) @ B.t()`` per forward on the
        ConvRot-ROTATED input (lossless, reversible; the deltas are
        registered parameters so ``.to()``/``unload()`` moves them and
        VRAM scales with them); the bf16 :class:`DeltaLinear` txtfusion
        targets take the same deltas on the plain input.

        All-or-nothing: a set becomes ``_active_loras`` only once every
        module is patched: :meth:`_merge_lora_deltas` (parsing, mapping,
        validation) runs before ANY module is touched, and every target is
        resolved + shape-checked BEFORE the first mutation, with parameter
        registration under the shared ``_PARAM_INIT_LOCK``.
        """
        key = tuple((n, float(w)) for n, w in loras)
        if key == self._active_loras:
            return  # identical set: no-op
        merged = self._merge_lora_deltas(key)  # may raise
        with _PARAM_INIT_LOCK:
            # resolve + validate every target BEFORE mutating anything
            plan: list[tuple[object, object]] = []
            for path, deltas in merged.items():
                module = self._module_for(path)
                if not isinstance(module, (QuantLinear, DeltaLinear)):
                    raise TypeError(
                        f"{path!r} is not a delta-capable Linear "
                        f"({type(module).__name__})")
                for a, _b, _r0, _r1 in deltas:
                    if a.shape[1] != module.in_features:
                        raise ValueError(
                            f"{path!r}: LoRA input dim {a.shape[1]} != "
                            f"module in_features {module.in_features}"
                        )
                plan.append((module, deltas))
            # drop whatever is currently applied, then install the new set
            self._unload_loras()
            for module, deltas in plan:
                module.set_lora_deltas(deltas)
        self._active_loras = key
        log.info("Krea 2 LoRA load: %s", format_loras(list(key)))

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
        # LoRA deltas must be in place before the sampling loop; runs under
        # self._lock via generate()
        self._apply_loras(request.loras)
        # 8x VAE downsample then 2x2 patchify -> snap to 16 px (as Flux)
        w = max(16, (request.width // 16) * 16)
        h = max(16, (request.height // 16) * 16)
        lh, lw = h // 8, w // 8

        # VAE tiling: vae-tiling=true in imagagen.ini forces it; otherwise
        # automatic above 2 MP. The VAE is the (non-local) Wan decoder, so
        # the threshold and tile sizes are the measured Flux values: below
        # 2 MP tiling changes the image, not just the seams, and a crash
        # above it is the floor.
        if self.config is not None:
            self.vae.use_tiling = self.config.vae_tiling or w * h > 2_000_000
            if self.vae.use_tiling:
                self.vae.tile_sample_min_size = 512
                self.vae.tile_latent_min_size = 64
                self.vae.tile_overlap_factor = 0.5

        # the "model default" sentinels (steps/cfg <= 0) resolve to the
        # checkpoint class's built-ins: raw 28 steps / standard CFG 5.5 /
        # resolution-derived mu, turbo 8 steps / no CFG / pinned mu
        turbo = request.turbo
        steps = request.steps if request.steps > 0 else (
            DEFAULT_STEPS_TURBO if turbo else DEFAULT_STEPS_RAW)
        steps = max(1, steps)
        cfg = request.cfg if request.cfg > 0 else (
            DEFAULT_CFG_TURBO if turbo else DEFAULT_CFG_RAW)
        do_cfg = cfg > 1.0

        # --- text encode --- deterministic per prompt: identical prompts
        # (negative too, on the classic path) reuse the cached encoding
        # (no encoder run, no transformer swap)
        context = self._get_context(request.prompt)
        uncond_context = None
        if do_cfg:
            uncond_context = self._get_context(request.negative_prompt)

        # --- sigma schedule + noise ---
        seq_len = (h // 16) * (w // 16)  # image tokens (the mu variable)
        sigmas = kdiff.get_sigmas_krea2(
            steps, seq_len, device,
            mu=TURBO_MU if turbo else None)
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
            # the VAE encoder's fp32-normalize working set does not fit
            # next to the resident transformer: park it for the encode
            swapped = self._transformer_out()
            try:
                init_latent = self._encode_init(
                    request.init_image, device, (w, h))
            finally:
                if swapped:
                    self._transformer_in()
            # A1111: t_enc = int(min(strength, 0.999) * steps); the schedule
            # starts at sigmas[steps - 1 - t_enc] (strength 1.0 -> full)
            t_enc = int(min(request.denoise, 0.999) * steps)
            sigma_sched = sigmas[steps - 1 - t_enc :]
            # flow-matching state at the truncated start sigma: the valid
            # state at noise level sigma is sigma*noise + (1-sigma)*latent
            # (the same 'const' noise_scaling the Flux backend uses; the
            # SDXL-style `latent + sigma*noise` is not a valid flow state)
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
                # feathered brush: the (Wan) VAE decoder is non-local, so
                # it rings a feathered latent blend into a visible halo
                # along the feather. The feather is applied only by the
                # final pixel composite (InpaintPlan).
                m_lock = (m > 0.5).float()
                mask_t = 1.0 - m_lock
                # only the inpaint region starts re-noised; the protected
                # region starts at the clean init latent
                noise = noise * (1.0 - mask_t) + init_latent * mask_t
                if soft is not None:
                    # the inpaint weight (1) replicated across all 16
                    # channels, hardened like mask_t (per-region blend)
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
            sigma_bf = sigma.to(torch.bfloat16)
            out = transformer(x_bf, sigma_bf, context)
            if do_cfg:
                out_u = transformer(x_bf, sigma_bf, uncond_context)
                # standard CFG in the x0 domain: x0 = x0u + cfg*(x0c - x0u);
                # in velocity form v = vu + cfg*(vc - vu) (the reference's
                # guidance g maps to cfg = g + 1)
                v = out_u.float() + cfg * (out.float() - out_u.float())
            else:
                v = out.float()
            # The transformer returns the flow velocity (reference FLOW /
            # CONST convention): x0 = x - sigma * v. Verified empirically
            # (dev/smoke_krea2.py).
            x0 = xwork - sigma * v
            if mask_t is not None:
                if soft is not None:
                    # soft inpainting (Forge builtin extension): the x0
                    # estimate is blended with the init latent; the blend
                    # weight follows the mask opacity and the current sigma
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
            if progress and self._step <= self._steps_total:
                progress(self._step / self._steps_total,
                         f"step {self._step}/{self._steps_total}")
            return x0

        self._step = 0
        self._steps_total = max(1, len(sigma_sched) - 1)

        # the first forward peaks on the cold allocator and the transformer
        # nearly fills the card: hand it every reserved byte before sampling
        torch.cuda.empty_cache()

        with torch.inference_mode():
            x = kdiff.sample_euler(model_fn, noise, sigma_sched, extra_args={})
            if init_latent is not None and mask_t is not None and soft is None:
                # final lock: the protected region is exactly the init
                # latent (skipped with soft inpainting: the protected
                # content is soft-blended at every step and restored by the
                # adaptive pixel composite below instead)
                x = x * (1.0 - mask_t) + init_latent * mask_t

            # --- VAE decode (kept under inference_mode so the result has no
            #     grad): solver space -> pixels space via the per-channel
            #     mean/std ---
            latents = denormalize_latent(self.vae, x).to(torch.bfloat16)
            # the decode's working set (up to ~4 GiB untiled at 1024px) does
            # not fit next to the resident transformer: park it (Forge's
            # memory manager unloads the unet for the VAE too)
            swapped = self._transformer_out()
            try:
                image = self.vae.decode(latents, return_dict=False)[0].float()
            except torch.cuda.OutOfMemoryError:
                # even the tiled working set did not fit (extreme sizes): a
                # full GC + cache flush leaves only the VAE on the card
                log.warning(
                    "%s: VAE decode OOMed, retrying after a full GC", self.name)
                import gc

                gc.collect()
                torch.cuda.empty_cache()
                image = self.vae.decode(latents, return_dict=False)[0].float()
            finally:
                if swapped:
                    self._transformer_in()
            image = (image / 2 + 0.5).clamp(0, 1)
        arr = (image.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        if soft is not None and mask_t is not None:
            # Forge's post_sample pixel composite: composite the original
            # image back over the generation wherever the latents did not
            # change much
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
        # idle footprint is the resident weights, not the pool's peak
        torch.cuda.empty_cache()
        return GenerateResult(
            image=arr, seed=request.seed,
            steps=steps, cfg=cfg, turbo=turbo)


def from_config(config) -> ModelBackend | None:
    """Load the first discovered Krea 2 model, or None when nothing exists."""
    from .models import ModelRegistry

    registry = ModelRegistry(config)
    krea2 = [m for m in registry.models if m.family == "krea2"]
    if not krea2 or not registry.gpus:
        return None
    return registry.backend_for(krea2[0].id)
