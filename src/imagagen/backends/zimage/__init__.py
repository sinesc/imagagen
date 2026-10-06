"""Z-Image (Turbo) backend: fp8 NextDiT transformer + Qwen3-4B encoder +
flow-matching Euler.

Third model family alongside SDXL and Flux (see :mod:`imagagen.backends.sdxl`,
:mod:`imagagen.backends.flux`). Plan: ``dev/plans/zimage-backend-plan.md``;
reference: the ComfyUI checkout in ``dev/implementations/ComfyUI`` (Z-Image
is a ``Lumina2``/``NextDiT`` model there with a Qwen3-4B text encoder).

Architecture:
- the transformer is a 6.6B NextDiT (:class:`~imagagen.backends.zimage.transformer.
  ZImageTransformer`) whose ``nn.Linear`` weights live as ``F8_E4M3`` on the
  GPU and are dequantized to bf16 per-op (:mod:`imagagen.gguf_ops`
  ``Fp8Linear`` pattern, the same as Flux FP8), so ~6.2 GB of fp8 bytes stay
  resident instead of the ~12.7 GB a bf16 copy would need;
- the text condition is the Qwen3-4B penultimate hidden state (1, L, 2560)
  of the chat-templated prompt; the encoder keeps its scaled-fp8 projections
  (~4.4 GB resident) and casts to bf16 per-op
  (:mod:`imagagen.encoders.qwen3`);
- the VAE is the FLUX AE: ``infra/ae.safetensors`` is byte-identical for
  Z-Image and Flux, and loads with the committed
  ``res/ae.config.json`` (16 channels, 8x downsample, scaling 0.3611 /
  shift 0.1159, same 2-MP tiling threshold as Flux);
- sampling is plain flow-matching Euler (``kdiff.sample_euler``) over the
  shift-3 linear-SNR schedule (``kdiff.get_sigmas_zimage``).

Z-Image-Turbo is an NFE-1 (single-pass) model: there is no guidance
embedding and no CFG, so the request's ``cfg`` / ``distilled`` are ignored
(one transformer pass per step, negative prompt not encoded); the effective
CFG recorded in the PNG metadata is 1.0.

Precision: the transformer and VAE run in bf16, the solver state ``x`` runs
in fp32 (the model output is cast to fp32 before the
``x0 = x - sigma * v`` conversion), mirroring the Flux backend.

VRAM strategy: transformer (~6.2 GB) + encoder (~4.4 GB) + VAE (~0.67 GB)
is ~11.3 GB resident. The encoder co-resides on the GPU only when a
measured byte count fits next to the transformer (``mem_get_info`` at load,
the Flux pattern); otherwise the transformer is swapped to the CPU while
the encoder runs (and back again afterwards) -- Forge's memory-manager
swap. ``text-encoder-device = cpu`` keeps the encoder in system RAM.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import numpy as np
import torch

from ... import kdiff
from ...encoders.qwen3 import load_zit_encoder, zimage_prompt_template
from ...flux_lora import FluxLoraSpec, merge_module_deltas
from ...gguf_ops import QuantLinear, quant_linear_scope
from ...soft_inpaint import adaptive_change_alpha, latent_blend
from .. import (
    _PARAM_INIT_LOCK,
    GenerateRequest,
    GenerateResult,
    ModelBackend,
    format_loras,
    shared_enc_cache,
)
from .lora import ParamDelta, parse_zimage_lora
from .transformer import RMSNorm, ScaledFp8Linear, ZImageTransformer


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

# the "model default" step count: Z-Image-Turbo's official 8-step schedule
DEFAULT_STEPS = 8
# the "model default" CFG: the model is single-pass (no CFG), so the
# effective value is always 1.0
DEFAULT_CFG = 1.0

SAMPLERS = {"Euler": kdiff.sample_euler}
# Z-Image has no scheduler concept: the shifted sigma schedule is computed
# internally (kdiff.get_sigmas_zimage).
SCHEDULERS = {}

#: state-dict prefix of the transformer section in the checkpoint
_SD_PREFIX = "model.diffusion_model."


def _load_zimage_state_dict(path: str) -> tuple[dict, bool]:
    """Read the transformer section of a Z-Image checkpoint into
    ``{module path: tensor}`` (keys relative to the transformer root, fp8
    tensors kept fp8 as Parameters -- the Flux FP8 loading pattern).

    Two on-disk formats are supported:
    - A1111/Forge-style all-in-one files: the transformer keys carry the
      ``model.diffusion_model.`` prefix (``vae.`` / ``text_encoders.``
      siblings are skipped) and hold *plain* fp8 weights;
    - Kijai-style transformer-only files: no key prefix, a ``scaled_fp8``
      marker key, and per-GEMM ``scale_weight`` / ``scale_input`` tensors
      (the scales ride along in the same state dict).

    Returns ``(state_dict, scaled_fp8)``.

    Slow disk I/O; run OUTSIDE ``_PARAM_INIT_LOCK`` (it only reads tensors,
    it does not register them in a module).
    """
    from safetensors import safe_open

    sd: dict = {}
    scaled = False
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        prefixed = any(k.startswith(_SD_PREFIX) for k in keys)
        for key in keys:
            if prefixed:
                if not key.startswith(_SD_PREFIX):
                    continue  # vae. / text_encoders.: the infra components
                name = key[len(_SD_PREFIX):]
            else:
                if key == "scaled_fp8":
                    scaled = True
                    continue  # format marker, no data
                name = key
            sd[name] = torch.nn.Parameter(f.get_tensor(key), requires_grad=False)
        if not prefixed and not scaled:
            # bare transformer keys without the marker: still scaled-fp8 if
            # scale keys are present (the marker is only a convenience)
            scaled = any(k.endswith(("scale_weight", "scale_input")) for k in keys)
    return sd, scaled


class ZImageBackend(ModelBackend):
    can_park = True  # transformer, VAE and encoder all move cleanly
    default_sampler = "Euler"
    default_scheduler = "Flow"
    supports_loras = True  # runtime additive deltas (see _apply_loras)
    is_zimage = True

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

        from diffusers import AutoencoderKL
        from diffusers.utils.logging import disable_progress_bar
        from transformers import Qwen2Tokenizer

        disable_progress_bar()

        infra = config.infra_dir
        # --- checkpoint: read the transformer state dict. This is slow disk
        #     I/O (it BUILDS tensors on the CPU but does not register them in
        #     a module), so it runs OUTSIDE _PARAM_INIT_LOCK to avoid blocking
        #     concurrent parameter work on another GPU for the whole read.
        sd, scaled_fp8 = _load_zimage_state_dict(self.model_path)

        # Every step below CREATES or MOVES module parameters (VAE,
        # transformer, text encoder). diffusers' from_single_file and the
        # .to() device moves route through the globally monkey-patched
        # nn.Module.register_parameter (see _PARAM_INIT_LOCK).
        with _PARAM_INIT_LOCK:
            # --- VAE (bf16): shared with Flux (byte-identical AE) ---
            vae_path = infra / "ae.safetensors"
            # ae.safetensors is the FLUX AE (verified byte-identical to the
            # Z-Image checkpoint's VAE, same md5), so the committed AE
            # config is the correct architecture description
            vae_cfg = config.root / "res" / "ae.config.json"
            self.vae = AutoencoderKL.from_single_file(
                str(vae_path), config=str(vae_cfg), torch_dtype=torch.bfloat16
            )
            self.vae = self.vae.to(device)
            self.vae.enable_slicing()

            # --- transformer (fp8 -> per-op bf16) ---
            # build on CPU (the dummy weights are small); the fp8 bytes load
            # on CPU and move to the GPU with .to(device)
            with quant_linear_scope(
                "fp8", torch.device("cpu"), torch.bfloat16,
                linear_cls=ScaledFp8Linear if scaled_fp8 else None,
            ):
                self.transformer = ZImageTransformer()
            result = self.transformer.load_state_dict(sd, strict=False)
            missing = list(result.missing_keys)
            unexpected = list(result.unexpected_keys)
            del sd
            if missing or unexpected:
                raise ValueError(
                    f"Z-Image checkpoint mismatch for {info.name}: "
                    f"missing={missing[:5]} unexpected={unexpected[:5]}")
            self.transformer.to(device)
            self.transformer.eval()

            # --- text encoder (Qwen3-4B, scaled fp8) ---
            # `te_device` (GPU by default) is where the encoder runs during a
            # generation. It loads on the CPU first and only moves to the
            # GPU when it fits next to the transformer; otherwise it is
            # brought to the GPU per generation, swapping the transformer
            # out while it runs (see _generate_impl, the Flux/Forge pattern).
            # With text_encoder_device=cpu it never leaves the CPU.
            self.te_device = "cpu" if config.text_encoder_device == "cpu" else device
            self.te = load_zit_encoder(str(infra / "qwen_3_4b.safetensors"), "cpu")
            self.te_tok = Qwen2Tokenizer.from_pretrained(
                str(config.res_dir / "tokenizer_zimage"))
            # resident bytes of the encoder; the fit check below and the
            # per-generate swap decision compare this against the free VRAM
            self._encoder_bytes = sum(
                p.numel() * p.element_size() for p in self.te.parameters()
            )
            if self.te_device != "cpu" and self._encoder_bytes + (1 << 30) \
                    <= torch.cuda.mem_get_info(device)[0]:
                # fits next to the transformer: co-resident, no swaps
                self._text_to(self.te_device)

        # all weights are resident now; release the transient VRAM the load
        # left behind so the pool is as tight as the live weights
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- text ----

    def _encode_prompt(self, prompt: str):
        """Tokenize + encode ``prompt`` -> the (1, L, 2560) bf16 Qwen3-4B
        penultimate hidden state (chat-templated, the reference
        Z-Image text-encoder target). The encoder must be on
        ``self.te_device`` (call :meth:`_text_to`)."""
        device = self.te_device
        text = zimage_prompt_template().format(prompt)
        ids = self.te_tok(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)
        return self.te(ids).to(torch.bfloat16)

    def _get_context(self, prompt: str):
        """The encoded (1, L, 2560) bf16 context for ``prompt`` on the
        model's device.

        Encodings are cached on the CPU (LRU, a few MB each, shared across
        this model's per-GPU instances): the encoding is deterministic, so
        a batch of identical prompts -- or any later repeat of the same
        prompt -- reuses the first encoding instead of running the ~4.4 GiB
        encoder again. A cache hit also skips the transformer swap that an
        uncached encode would trigger.
        """
        cached = self._enc_cache.get(prompt)
        if cached is not None:
            self._enc_cache.move_to_end(prompt)
            # A hit skips the encode, so the post-encode offload in
            # _encode_prompts never runs: if the co-resident fit check (cold
            # load / parked restore) left the encoder on the GPU, drop it
            # now, or its ~4.1 GiB sits next to the transformer through
            # sampling and starves the VAE decode (the untiled 1024x1280
            # decode alone wants ~3 GiB)
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

        The ~4.4 GiB encoder crosses the bus at most once per call, however
        many prompts it encodes: the generation queue pre-encodes a whole
        wildcard batch here (via :meth:`pre_encode`) instead of every task
        of the batch triggering its own transformer swap. With the GPU
        text-encoder default, the resident transformer (~6.2 GiB) and the
        encoder (~4.4 GiB) may not leave room for the sampling working set
        on a 16 GiB card; Forge's memory manager unloads the unet while the
        text encoders run. We do the same swap when the encoder is parked
        on the CPU for this reason (see __init__): park the transformer on
        the CPU for the encode, then bring it back after the encoder is
        parked again.

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
        """One encoder session for a whole batch (see ModelBackend). Z-Image
        ignores the negative prompt: only the prompts are encoded."""
        with self._lock:
            self._encode_prompts([prompt for prompt, _ in pairs], reserve)

    def _text_to(self, device) -> None:
        self.te.to(device)

    def _text_offload(self) -> None:
        self.te.to("cpu")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- ctrl ----

    def _encode_init(self, arr: np.ndarray, device, size: tuple[int, int]):
        """VAE-encode an HxWx3 uint8 image to its solver-space latent
        (the deterministic ``latent_dist.mode()`` -- A1111's "Full" encode),
        the inverse of the decode transform ``x / scaling + shift``.

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
        """LoRA name (no extension) -> path under ``loras/zimage/``.

        The suffix is *appended*, never ``Path.with_suffix``: names may
        themselves contain dots, and with_suffix would replace the last one
        instead of adding ".safetensors".
        """
        if self.config is None:
            return None
        p = self.config.loras_dir / "zimage" / (name + ".safetensors")
        return str(p) if p.is_file() else None

    def _parse_lora(self, path: str) -> tuple[list[FluxLoraSpec], list[ParamDelta]]:
        """Read + key-map a Z-Image LoRA (cached by ``(path, mtime)``).

        Returns ``(lora_pairs, param_deltas)``. Raises ``ValueError``
        (unknown/unmappable keys, see zimage_lora) on files this
        transformer cannot consume.
        """
        cached = self._lora_cache.get(path)
        mtime = Path(path).stat().st_mtime
        if cached is not None and cached[0] == mtime:
            return cached[1], cached[2]
        specs, deltas = parse_zimage_lora(path)
        self._lora_cache[path] = (mtime, specs, deltas)
        return specs, deltas

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

        Pure: returns ``({module path: [(A, B, r0, r1), ...]},
        {(module path, attr): delta tensor})`` (CPU tensors; the weight *
        scale factor is pre-folded into A, and deltas covering adjacent row
        slices of one module are merged so each module costs ONE extra
        GEMM pair; the direct deltas are summed per target). Raises
        ``ValueError`` on a missing file, an unmappable key, or a shape
        mismatch -- nothing is mutated here, so a failing set leaves no
        partial state.
        """
        per_module: dict[str, dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]]] = {}
        param: dict[tuple[str, str], torch.Tensor] = {}
        for name, weight in loras:
            if weight == 0.0:  # zero-weight LoRA: no delta at all
                continue
            path = self._lora_path(name)
            if path is None:
                raise ValueError(
                    f"Z-Image LoRA not found: {name!r} (looked in loras/zimage/)")
            specs, deltas = self._parse_lora(path)
            for spec in specs:
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
            for d in deltas:
                key = (d.module, d.attr)
                param[key] = param.get(key, torch.zeros_like(d.delta)) + \
                    d.delta * weight
        return merge_module_deltas(per_module), param

    def _apply_loras(self, loras: list[tuple[str, float]]) -> None:
        """Attach the requested LoRAs to the transformer (cached per set).

        Must run under ``self._lock`` (called from ``_generate_impl``
        before the sampling loop, as in the Flux backend).

        Runtime additive deltas: the fused fp8 weights stay quantized and
        each affected :class:`~imagagen.gguf_ops.QuantLinear` computes
        ``base + (x @ A.t()) @ B.t()`` per forward (lossless, reversible;
        the deltas are registered parameters so ``.to()``/``unload()`` moves
        them and VRAM scales with them). The direct ``.diff``/``.diff_b``
        deltas ride the same way: ``QuantLinear.set_lora_bias`` for biases
        and ``RMSNorm.set_lora_weight`` for the norm scales.

        All-or-nothing: a set becomes ``_active_loras`` only once every
        module is patched; :meth:`_merge_lora_deltas` (parsing, mapping,
        validation) runs before ANY module is touched, and every target is
        resolved + shape-checked BEFORE the first mutation, with parameter
        registration under the shared ``_PARAM_INIT_LOCK``.
        """
        key = tuple((n, float(w)) for n, w in loras)
        if key == self._active_loras:
            return  # identical set: no-op
        merged, param_deltas = self._merge_lora_deltas(key)  # may raise
        with _PARAM_INIT_LOCK:
            # resolve + validate every target BEFORE mutating anything
            plan: list[tuple[object, str, object]] = []
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
                plan.append((module, "deltas", deltas))
            for (path, attr), t in param_deltas.items():
                module = self._module_for(path)
                if attr == "bias":
                    b = getattr(module, "bias", None)
                    if not isinstance(module, QuantLinear) or b is None:
                        raise ValueError(
                            f"{path!r} has no bias for a LoRA bias delta"
                        )
                    if tuple(b.shape) != tuple(t.shape):
                        raise ValueError(
                            f"{path!r}: bias delta {tuple(t.shape)} != "
                            f"bias {tuple(b.shape)}"
                        )
                elif attr == "weight":
                    w = getattr(module, "weight", None)
                    if not isinstance(module, RMSNorm) or w is None:
                        raise ValueError(
                            f"{path!r} is not an RMSNorm (weight-delta target)"
                        )
                    if tuple(w.shape) != tuple(t.shape):
                        raise ValueError(
                            f"{path!r}: weight delta {tuple(t.shape)} != "
                            f"weight {tuple(w.shape)}"
                        )
                else:
                    raise ValueError(
                        f"{path!r}: unknown LoRA delta attribute {attr!r}"
                    )
                plan.append((module, attr, t))
            # drop whatever is currently applied, then install the new set
            self._unload_loras()
            for module, kind, payload in plan:
                if kind == "deltas":
                    module.set_lora_deltas(payload)
                elif kind == "bias":
                    module.set_lora_bias(payload)
                else:
                    module.set_lora_weight(payload)
        self._active_loras = key
        log.info("Z-Image LoRA load: %s", format_loras(list(key)))

    def _unload_loras(self) -> None:
        """Clear every module's delta state (no cache bookkeeping).

        Runs under ``_PARAM_INIT_LOCK``. Clears the *actual* module state
        (not just ``_active_loras``), so leftover deltas from any earlier
        state are removed too.
        """
        for module in self.transformer.modules():
            if getattr(module, "lora_deltas", None):
                module.clear_lora_deltas()  # also clears a bias delta
            if getattr(module, "lora_weight", None) is not None:
                module.clear_lora_weight()

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
        # automatic above 2 MP. The VAE is the (strongly non-local) Flux
        # decoder, so the threshold and tile sizes are the measured Flux
        # values: below 2 MP tiling changes the image, not just the seams,
        # and a crash above it is the floor.
        if self.config is not None:
            self.vae.use_tiling = self.config.vae_tiling or w * h > 2_000_000
            if self.vae.use_tiling:
                self.vae.tile_sample_min_size = 512
                self.vae.tile_latent_min_size = 64
                self.vae.tile_overlap_factor = 0.5

        # the "model default" sentinel (request.steps <= 0) resolves to the
        # built-in 8-step Turbo schedule; cfg is always the effective 1.0
        # (single-pass model, no CFG / guidance embedding)
        steps = request.steps if request.steps > 0 else DEFAULT_STEPS
        steps = max(1, steps)

        # --- text encode --- deterministic per prompt: identical prompts
        # reuse the cached encoding (no encoder run, no transformer swap)
        context = self._get_context(request.prompt)

        # --- sigma schedule + noise ---
        sigmas = kdiff.get_sigmas_zimage(steps, device)
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
                # feathered brush: the (Flux) VAE decoder is strongly
                # non-local, so it rings a feathered latent blend into a
                # visible halo along the feather. The feather is applied
                # only by the final pixel composite (InpaintPlan).
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
            out = transformer(x_bf, sigma.to(torch.bfloat16), context)
            # The transformer returns the raw NextDiT output, which under
            # the reference (ComfyUI FLOW / CONST convention) IS the flow
            # velocity: x0 = x - sigma * out. Verified empirically against
            # the known velocity of a noised clean latent
            # (dev/smoke_zimage.py).
            v = out.float()
            if progress and self._step <= self._steps_total:
                progress(self._step / self._steps_total,
                         f"step {self._step}/{self._steps_total}")
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
            return x0

        self._step = 0
        self._steps_total = max(1, len(sigma_sched) - 1)

        with torch.inference_mode():
            x = kdiff.sample_euler(model_fn, noise, sigma_sched, extra_args={})
            if init_latent is not None and mask_t is not None and soft is None:
                # final lock: the protected region is exactly the init
                # latent (skipped with soft inpainting: the protected
                # content is soft-blended at every step and restored by the
                # adaptive pixel composite below instead)
                x = x * (1.0 - mask_t) + init_latent * mask_t

            # --- VAE decode (kept under inference_mode so the result has no grad) ---
            latents = (x / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(torch.bfloat16)
            try:
                image = self.vae.decode(latents, return_dict=False)[0].float()
            except torch.cuda.OutOfMemoryError:
                # the untiled decode's working set does not fit next to the
                # resident transformer; Forge's memory manager unloads the
                # unet for the VAE -- we do the same
                log.warning(
                    "%s: untiled VAE decode OOMed, retrying with the "
                    "transformer on the CPU", self.name)
                import gc

                gc.collect()
                self.transformer.to("cpu")
                torch.cuda.empty_cache()
                image = self.vae.decode(latents, return_dict=False)[0].float()
                self.transformer.to(self.device)
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
            steps=steps,
            # the single-pass model has no CFG: the effective value is 1.0
            cfg=DEFAULT_CFG)


def from_config(config) -> ModelBackend | None:
    """Load the first discovered Z-Image model, or None when nothing exists."""
    from ...models import ModelRegistry

    registry = ModelRegistry(config)
    zimage = [m for m in registry.models if m.family == "zimage"]
    if not zimage or not registry.gpus:
        return None
    return registry.backend_for(zimage[0].id)
