"""GGUF / FP8 / NF4 loading + on-the-fly GPU dequantization for the Flux transformer.

The 12B FLUX.1 transformer ships in three checkpoint formats, all handled
here:

- **GGUF Q4_K / Q4_0** (``agFluxQ4KS_v21.gguf``) — raw quantized bytes;
  the Q4_K and Q4_0 kernels, :class:`GgufParam` and the ``Linear``
  replacement are ported verbatim from the Forge reference
  (``dev/implementations/stable-diffusion-webui-forge/packages_3rdparty/gguf``
  and ``backend/operations*.py``) so the dequantized values match Forge
  bit-for-bit. The only PyPI dependency used is ``gguf``
  (for ``GGUFReader``); the dequantize kernels are self-contained torch ops.
- **Kijai FP8** (``flux1DevFP8110GBKijai.safetensors`` /
  ``flux1SchnellFP811GBKijai.safetensors``) — transformer only, top-level
  diffusers names (``double_blocks.0.img_attn.qkv.weight``,
  ``single_blocks.N.linear1.weight``, ``img_in.weight``, ...), **every**
  tensor (weights, biases, norm scales) is ``F8_E4M3`` at the real weight
  shape with no scale tensors; dequantize is the exact cast to the input
  dtype (:class:`Fp8Linear` keeps the fp8 bytes resident and casts per-op).
- **bitsandbytes NF4** (``flux1DevNF4V2.safetensors`` /
  ``flux1SchnellNF4.safetensors``) — all-in-one pipeline file; only the
  transformer section (``model.diffusion_model.`` prefix) is used, the
  ``vae.`` / ``text_encoders.`` sections are the infra components and
  stay ignored. Every quantized linear is stored in bnb's *pre-quantized
  format*: 4-bit packed ``<k>.weight`` (uint8, 2 values per byte,
  ``(numel/2, 1)``) plus ``<k>.weight.absmax`` (one entry per 64-element
  block, float32 — or uint8 indices into ``nested_quant_map`` for the
  *nested* files), ``<k>.weight.quant_map`` (the codebook, NOT used at
  runtime: the kernel has the NF4 codebook built in) and a
  ``<k>.weight.quant_state.bitsandbytes__nf4`` uint8 blob holding the JSON
  ``{"quant_type", "blocksize", "dtype", "shape"}`` plus, nested files
  only, ``nested_blocksize`` / ``nested_dtype`` / ``nested_offset`` and the
  ``<k>.weight.nested_absmax`` / ``<k>.weight.nested_quant_map`` tensors.
  Dequantize is bnb's native path: a :class:`Nf4Param` (packed weight +
  aux tensors that move together) plus
  ``bitsandbytes.functional.dequantize_4bit`` with a ``QuantState``
  built from the param fields (:class:`Nf4Linear`). No bitsandbytes
  runtime matmul — only its dequantize op (imported inside the function,
  so a CPU-only test env still imports this module).

All three keep the weights on the GPU in compact form and dequantize each
``nn.Linear`` weight to the input dtype on every forward pass (the Forge
approach), so peak VRAM is just the quantized bytes (~6 GB for the 12B dev
models) plus one layer's transient bf16 weight, not the ~24 GB bf16 would
need.
"""

from __future__ import annotations

import contextlib
import json
import threading

import torch

# Q4_K super-block = 256 elements, 144 quantized bytes.
QK_K = 256
_Q4K_BLOCK_SIZE = 256
_Q4K_TYPE_SIZE = 144
_K_SCALE_SIZE = 12
# Q4_0 block = 32 elements, 18 bytes (2-byte fp16 scale + 16 packed 4-bit values).
_Q40_BLOCK_SIZE = 32
_Q40_TYPE_SIZE = 18


# ------------------------------------------------------------------ 4-bit ----
# From the reference ``packages_3rdparty/gguf/quick_4bits_ops.py``.


def quick_split(x, p):
    return torch.split(x, list(p) + [x.shape[1] - sum(p)], dim=-1)


def change_4bits_order(x):
    y = torch.stack([x & 15, x >> 4], dim=-2).view(x.size(0), -1)
    z = y[:, ::2] | (y[:, 1::2] << 4)
    return z


# lookup-table based 4-bit unpack (the optimized path; falls back to the
# explicit stack when uint16 is unavailable).
def _native_unpack_u(x):
    x = x.view(torch.uint8).view(x.size(0), -1)
    unpacked = torch.stack([x & 15, x >> 4], dim=-1)
    unpacked = unpacked.view(x.size(0), -1)
    return unpacked.view(torch.int32)


_USE_LOOKUP = hasattr(torch, "uint16")
# Per-device cache of the 4-bit lookup table. Kept per device (thread-safe) so
# that concurrent backends on different GPUs each own their own copy instead of
# a single shared table being copied across devices on every dequantize op
# (which serialized the two GPUs and caused the alternating load pattern).
_4BIT_TABLE_U: dict = {}
_4BIT_TABLE_LOCK = threading.Lock()


def _build_table_u() -> torch.Tensor:
    """Build the 256*256 4-bit -> two-nibble lookup table on the CPU (once)."""
    return _native_unpack_u(
        torch.arange(start=0, end=256 * 256, dtype=torch.long).to(torch.uint16)
    )[:, 0]


def _table_for(device) -> torch.Tensor:
    """Return the lookup table resident on ``device`` (built/cached per device)."""
    key = device if isinstance(device, torch.device) else torch.device(device)
    table = _4BIT_TABLE_U.get(key)
    if table is None:
        with _4BIT_TABLE_LOCK:
            table = _4BIT_TABLE_U.get(key)
            if table is None:
                table = _build_table_u().to(key)
                _4BIT_TABLE_U[key] = table
    return table


def quick_unpack_4bits_u(x):
    if not _USE_LOOKUP:
        return torch.stack([x & 15, x >> 4], dim=-1).view(x.size(0), -1)
    table = _table_for(x.device)
    s0 = x.size(0)
    x16 = x.view(torch.uint16)
    y = torch.index_select(input=table, dim=0, index=x16.to(torch.int32).flatten())
    y = y.view(torch.uint8)
    y = y.view(s0, -1)
    return y


# ---------------------------------------------------------------- Q4_K ----
# From the reference ``packages_3rdparty/gguf/quants.py`` (Q4_K).


def q4k_get_scale_min_pytorch(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


def q4k_bake(parameter):
    """Reshape raw bytes into baked blocks + precompute scales (once, on load)."""
    if parameter.baked:
        return
    data = parameter.data
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // _Q4K_TYPE_SIZE
    blocks = rows.reshape((n_blocks, _Q4K_TYPE_SIZE))
    parameter.data = blocks.contiguous()
    _q4k_bake_inner(parameter)
    parameter.baked = True
    return


def _q4k_bake_inner(parameter):
    blocks = parameter.data
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = quick_split(blocks, [2, 2, _K_SCALE_SIZE])
    d = d.view(torch.float16).to(parameter.computation_dtype)
    dmin = dmin.view(torch.float16).to(parameter.computation_dtype)
    sc, m = q4k_get_scale_min_pytorch(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1)).to(parameter.computation_dtype)
    qs = qs.reshape((n_blocks, -1, 1, 32))
    qs = change_4bits_order(qs)
    d = d.view(torch.uint8).reshape((n_blocks, -1))
    dm = dm.view(torch.uint8).reshape((n_blocks, -1))
    qs = qs.view(torch.uint8)
    parameter.data = torch.cat([d, dm, qs], dim=-1).contiguous()
    return


def q4k_dequantize_blocks_pytorch(blocks, parameter) -> torch.Tensor:
    """Dequantize baked blocks to the computation dtype (per forward pass)."""
    n_blocks = blocks.shape[0]
    d, dm, qs = quick_split(blocks, [16, 16])
    d = d.view(parameter.computation_dtype).view((n_blocks, -1, 1))
    dm = dm.view(parameter.computation_dtype).view((n_blocks, -1, 1))
    qs = quick_unpack_4bits_u(qs).view((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def q4k_dequantize_pytorch(x) -> torch.Tensor:
    if not x.baked:
        raise ValueError("GGUF tensor is not baked")
    blocks = q4k_dequantize_blocks_pytorch(x.data, x)
    return blocks.view(x.shape)


# ---------------------------------------------------------------- Q4_0 ----
# From the reference ``packages_3rdparty/gguf/quants.py`` (Q4_0).


def q40_dequantize_pytorch(x) -> torch.Tensor:
    """Dequantize raw Q4_0 bytes to the computation dtype (per forward pass).

    Each 32-element block is 18 bytes: an fp16 scale ``d`` followed by 16
    packed 4-bit values (byte ``j`` holds element ``j`` in the low nibble
    and element ``j + 16`` in the high nibble);
    ``value = d * (nibble - 8)``. Unlike Q4_K this is done directly on the
    raw layout — no bake step is needed.
    """
    raw = x.data.view(torch.uint8)
    n_blocks = raw.numel() // _Q40_TYPE_SIZE
    blocks = raw.view(n_blocks, _Q40_TYPE_SIZE)
    d, qs = quick_split(blocks, [2])
    d = d.view(torch.float16).to(x.computation_dtype)
    vals = torch.cat([qs & 15, qs >> 4], dim=-1)
    return (d * (vals.to(x.computation_dtype) - 8)).view(x.shape)


# ------------------------------------------------------------- Parameter ----


# GGUF qtype enum values (2 == Q4_0, 12 == Q4_K; verified against the shipped files).
_Q4K = 12
_Q40 = 2


class GgufParam(torch.nn.Parameter):
    """An ``nn.Parameter`` wrapping raw GGUF bytes.

    ``.data`` is a 1-D uint8 tensor (the quantized bytes); ``.shape`` reports
    the real weight shape (``out_features, in_features``); ``.to(device)``
    moves the bytes; ``computation_dtype`` is the dtype dequantize outputs.

    Constructed from a ``gguf`` ReaderTensor (``.data`` = raw bytes, ``.shape``
    = GGUF-reversed shape, ``.tensor_type`` = qtype enum) in
    :func:`load_gguf_state_dict`.
    """

    def __new__(cls, tensor=None, requires_grad=False, no_init=False):
        return super().__new__(cls, torch.tensor(tensor.data), requires_grad=requires_grad)

    def __init__(self, tensor=None, requires_grad=False, no_init=False) -> None:
        if no_init:
            return
        self.gguf_type = tensor.tensor_type
        if self.gguf_type not in (0, 1, _Q40, _Q4K):
            raise ValueError(
                f"unsupported GGUF tensor type {self.gguf_type} for {tensor.name}")
        self.real_shape = torch.Size(reversed(list(tensor.shape)))
        self.computation_dtype = torch.float16
        self.baked = False
        return

    @property
    def shape(self):
        return self.real_shape

    def copy_with_data(self, data):
        new = torch.nn.Parameter.__new__(GgufParam, data, requires_grad=False)
        new.gguf_type = self.gguf_type
        new.real_shape = self.real_shape
        new.computation_dtype = self.computation_dtype
        new.baked = self.baked
        return new

    def to(self, *args, **kwargs):
        return self.copy_with_data(self.data.to(*args, **kwargs))

    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self.data, device=device))


class Nf4Param(torch.nn.Parameter):
    """An ``nn.Parameter`` wrapping a bitsandbytes NF4 *pre-quantized*
    weight (the format the ``flux1*NF4*.safetensors`` files ship).

    ``.data`` is the packed 4-bit weight (uint8, 2 values per byte, shape
    ``(numel/2, 1)``); ``.shape`` reports the real weight shape
    (``out_features, in_features``). The aux tensors travel with it:
    ``absmax`` (one entry per 64-element block; float32 — or uint8 indices
    into ``nested_quant_map`` for the nested files), and for nested files
    ``nested_absmax`` / ``nested_quant_map`` plus ``nested_blocksize`` /
    ``nested_dtype`` / ``nested_offset``. ``out_dtype`` is the dtype
    ``dequantize_4bit`` outputs (the JSON ``"dtype"``),
    ``computation_dtype`` the LoRA-delta cast target.

    ``.to(*args)`` moves the packed bytes AND every aux tensor together
    (the ``copy_with_data`` mechanism, as in :class:`GgufParam`), so
    ``QuantLinear._apply`` / module ``.to(device)`` keep working unchanged.
    """

    def __new__(cls, *args, **kwargs):
        data = args[0] if args else None
        return super().__new__(
            cls, data if data is not None else torch.empty(0), requires_grad=False)

    def __init__(self, packed, absmax, blocksize=64, quant_type="nf4",
                 out_dtype=None, real_shape=None, nested_absmax=None,
                 nested_quant_map=None, nested_blocksize=None,
                 nested_dtype=None, nested_offset=None) -> None:
        self.absmax = absmax
        self.blocksize = blocksize
        self.quant_type = quant_type
        self.out_dtype = out_dtype
        self.real_shape = torch.Size(real_shape)
        self.nested_absmax = nested_absmax
        self.nested_quant_map = nested_quant_map
        self.nested_blocksize = nested_blocksize
        self.nested_dtype = nested_dtype
        self.nested_offset = nested_offset
        self.computation_dtype = out_dtype
        return

    @property
    def shape(self):
        return self.real_shape

    def copy_with_data(self, data, absmax=None, nested_absmax=None,
                       nested_quant_map=None):
        new = torch.nn.Parameter.__new__(Nf4Param, data, requires_grad=False)
        new.absmax = absmax if absmax is not None else self.absmax
        new.blocksize = self.blocksize
        new.quant_type = self.quant_type
        new.out_dtype = self.out_dtype
        new.real_shape = self.real_shape
        new.nested_absmax = (nested_absmax if nested_absmax is not None
                             else self.nested_absmax)
        new.nested_quant_map = (nested_quant_map if nested_quant_map is not None
                                else self.nested_quant_map)
        new.nested_blocksize = self.nested_blocksize
        new.nested_dtype = self.nested_dtype
        new.nested_offset = self.nested_offset
        new.computation_dtype = self.computation_dtype
        return new

    def to(self, *args, **kwargs):
        def moved(t):
            return None if t is None else t.to(*args, **kwargs)
        return self.copy_with_data(
            self.data.to(*args, **kwargs), moved(self.absmax),
            moved(self.nested_absmax), moved(self.nested_quant_map))


def dequantize_tensor(tensor):
    if tensor is None:
        return None
    if tensor.gguf_type == _Q4K:
        return q4k_dequantize_pytorch(tensor)
    if tensor.gguf_type == _Q40:
        return q40_dequantize_pytorch(tensor)
    # non-quantized (fp16/fp32 bias): already a normal tensor
    return tensor


def load_gguf_state_dict(path) -> dict:
    """Read a ``.gguf`` file into ``{param_name: GgufParam}``.

    ``param_name`` is the GGUF tensor name (matches the transformer's module
    paths, e.g. ``img_in.weight``, ``double_blocks.0.img_attn.qkv.weight``).
    """
    import gguf

    reader = gguf.GGUFReader(str(path))
    sd = {}
    for tensor in reader.tensors:
        sd[str(tensor.name)] = GgufParam(tensor)
    return sd


def bake_gguf_model(model) -> None:
    """Bake every GGUF parameter of ``model`` (once, right after ``.to()``)."""
    if getattr(model, "gguf_baked", False):
        return
    for p in model.parameters():
        gguf_type = getattr(p, "gguf_type", None)
        if gguf_type == _Q4K:
            q4k_bake(p)
    model.gguf_baked = True
    return model


# -------------------------------------------------------------- Linear ----


def _tensor2parameter(x):
    if isinstance(x, torch.nn.Parameter):
        return x
    return torch.nn.Parameter(x, requires_grad=False)


class QuantLinear(torch.nn.Module):
    """``nn.Linear`` replacement that keeps the weight in a compact format
    and dequantizes it to the input dtype on every forward pass.

    Subclasses pick the weight format through the :meth:`_dequantize` hook:
    :class:`GgufLinear` (GGUF Q4_K bytes), :class:`Fp8Linear` (plain
    ``F8_E4M3`` — the template's cast IS the dequantize), :class:`Nf4Linear`
    (bitsandbytes NF4 4-bit packed). ``weight`` may be a
    :class:`GgufParam`/:class:`Nf4Param` (aux data moves with ``.to()``) or
    a plain tensor (the fp8 case); ``bias`` is a plain tensor (cast to the
    input dtype on first forward, then kept there).

    The runtime LoRA deltas (``lora_deltas``) work for every format: the
    base weight stays quantized and each delta adds its
    ``(rank, in) @ (out, rank)`` product to a slice of the output rows
    (Forge's *online* mode), installed via :meth:`set_lora_deltas` under
    the shared ``_PARAM_INIT_LOCK``.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # seeds the weight device / computation dtype (see
        # quant_linear_scope); _load_from_state_dict reads it and then
        # unregisters it (`del self.dummy`)
        self.dummy = torch.nn.Parameter(torch.empty(1, device=_DEVICE, dtype=_DTYPE))
        self.computation_dtype = _DTYPE
        self.weight = None
        self.bias = None
        self.lora_deltas: list[tuple[torch.nn.Parameter, torch.nn.Parameter, int, int]] = []
        # LoKr (Kronecker) deltas: (w1, w2) pairs whose weight delta is
        # kron(w1, w2); applied as the nested grouped GEMM of ComfyUI's
        # LoKr bypass, never materializing the full product
        self.lora_kron: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
        # optional additive bias delta (the Z-Image LoRA `.diff_b` keys):
        # registered as a parameter ONLY while installed (set_lora_bias);
        # read via getattr(self, "lora_bias", None) -- a plain None here
        # would make register_parameter raise "attribute already exists"

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        if hasattr(self, "dummy"):
            computation_dtype = self.dummy.dtype
            if computation_dtype not in (torch.float16, torch.bfloat16):
                computation_dtype = torch.float16
            if prefix + "weight" in state_dict:
                self.weight = state_dict[prefix + "weight"].to(device=self.dummy.device)
                self.weight.computation_dtype = computation_dtype
            if prefix + "bias" in state_dict:
                self.bias = state_dict[prefix + "bias"].to(device=self.dummy.device)
                self.bias.computation_dtype = computation_dtype
            del self.dummy
        else:
            if prefix + "weight" in state_dict:
                self.weight = state_dict[prefix + "weight"]
            if prefix + "bias" in state_dict:
                self.bias = state_dict[prefix + "bias"]
        return

    def _apply(self, fn, recurse=True):
        for k, p in list(self.named_parameters(recurse=False, remove_duplicate=True)):
            applied = fn(p)
            if isinstance(applied, torch.nn.Parameter):
                # param classes (GgufParam/Nf4Param) return their own type
                setattr(self, k, applied)
            elif isinstance(applied, torch.Tensor):
                # device/dtype converters (`.to()` et al.) return a plain
                # Tensor for a Parameter: re-wrap it so the name stays a
                # REGISTERED parameter. Demoting it to a plain __dict__
                # attribute (the old behavior) left the name in __dict__,
                # so the next register_parameter(name, ...) raised
                # "attribute 'name' already exists" on LoRA re-apply.
                setattr(self, k, torch.nn.Parameter(
                    applied, requires_grad=p.requires_grad))
            else:
                # torch's __setattr__ rejects a non-Parameter for a
                # registered name: unregister first (as `del self.dummy`
                # above), then assign the plain tensor
                delattr(self, k)
                setattr(self, k, applied)
        # plain (non-Parameter) weight/bias tensors (the fp8 case) are not
        # registered parameters, so fn does not reach them
        for name in ("weight", "bias"):
            v = getattr(self, name, None)
            if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter):
                setattr(self, name, fn(v))
        return self

    def _dequantize(self, w):
        raise NotImplementedError

    def forward(self, x):
        if self.bias is not None and self.bias.dtype != x.dtype:
            b = dequantize_tensor(self.bias) if isinstance(self.bias, GgufParam) else self.bias
            self.bias = _tensor2parameter(b.to(x.dtype))
        weight = self.weight
        if weight is not None:
            # the param classes move their aux data with .to(); the
            # dequantized weight is cast to the input dtype (Forge's
            # weights_manual_cast, skip_weight_dtype=False)
            weight = weight.to(device=x.device)
            weight = self._dequantize(weight).to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device) if self.bias is not None else None
        out = torch.nn.functional.linear(x, weight, bias)
        lora_bias = getattr(self, "lora_bias", None)
        if lora_bias is not None:
            # additive bias delta: a registered parameter, so .to(device)/
            # unload moves it with the module (see set_lora_bias)
            out = out + lora_bias.to(dtype=x.dtype)
        if self.lora_deltas:
            # runtime LoRA deltas (Forge's online mode): the base weight
            # stays quantized, each delta adds its (rank, in) @ (out, rank)
            # product to a slice of the output rows. A/B are registered
            # parameters, so .to(device)/unload moves them with the weight
            # and they cost one extra GEMM pair per module.
            for a, b, r0, r1 in self.lora_deltas:
                out[..., r0:r1] += (x @ a.t()) @ b.t()
        for w1, w2 in self.lora_kron:
            # LoKr delta kron(w1, w2) @ x applied as nested GEMMs (the
            # exact ComfyUI LoKr bypass math): group the input axis by
            # w1's column count, apply w2 inside each group, w1 across
            # the groups -- the full Kronecker product is never built.
            # Registered parameters, so .to(device)/unload moves them.
            uq = w1.shape[1]
            hb = torch.nn.functional.linear(
                x.reshape(*x.shape[:-1], uq, -1), w2)
            hc = torch.nn.functional.linear(hb.transpose(-1, -2), w1)
            out += hc.transpose(-1, -2).reshape(*hc.shape[:-2], -1)
        return out

    def set_lora_deltas(self, deltas) -> None:
        """Install runtime LoRA deltas (replaces any previous set).

        ``deltas``: ``(A, B, r0, r1)`` tuples; ``A`` is ``(rank,
        in_features)``, ``B`` is ``(r1 - r0, rank)`` (any device/dtype,
        moved to the weight's). The A/B tensors are registered as
        parameters (named ``lora_A_N`` / ``lora_B_N``) so ``.to(device)``
        moves them with the module. Replaces any previous A/B delta set
        (LoKr factors are untouched). Call under the shared
        ``_PARAM_INIT_LOCK`` (parameter creation is not thread-safe across
        backends, see backends.py).
        """
        self._clear_ab_deltas()
        w = self.weight
        device = w.data.device if w is not None else None
        dtype = (getattr(w, "computation_dtype", None)
                 or self.computation_dtype
                 or (w.dtype if w is not None else None))
        self.lora_deltas = []
        for i, (a, b, r0, r1) in enumerate(deltas):
            if device is not None:
                a = a.to(device=device, dtype=dtype)
                b = b.to(device=device, dtype=dtype)
            pa = torch.nn.Parameter(a.detach(), requires_grad=False)
            pb = torch.nn.Parameter(b.detach(), requires_grad=False)
            self.register_parameter(f"lora_A_{i}", pa)
            self.register_parameter(f"lora_B_{i}", pb)
            self.lora_deltas.append((pa, pb, r0, r1))

    def set_lora_bias(self, delta) -> None:
        """Install an additive bias delta (replaces any previous one).

        ``delta`` is moved to the weight's device/dtype and registered as
        a parameter (named ``lora_bias``) so ``.to(device)`` moves it with
        the module. Call under the shared ``_PARAM_INIT_LOCK``.
        """
        self.clear_lora_bias()
        w = self.weight
        device = w.data.device if w is not None else None
        dtype = (getattr(w, "computation_dtype", None)
                 or self.computation_dtype
                 or (w.dtype if w is not None else None))
        if device is not None:
            delta = delta.to(device=device, dtype=dtype)
        pb = torch.nn.Parameter(delta.detach(), requires_grad=False)
        self.register_parameter("lora_bias", pb)

    def clear_lora_bias(self) -> None:
        """Remove any installed bias delta."""
        if "lora_bias" in self._parameters:
            self.register_parameter("lora_bias", None)

    def _clear_ab_deltas(self) -> None:
        """Remove only the A/B delta set (LoKr factors stay)."""
        self.lora_deltas = []
        for name in [n for n in self._parameters
                     if n.startswith("lora_A_") or n.startswith("lora_B_")]:
            self.register_parameter(name, None)
        for name in [n for n in list(self.__dict__)
                     if n.startswith(("lora_A_", "lora_B_"))]:
            delattr(self, name)

    def _clear_kron_deltas(self) -> None:
        """Remove only the LoKr factor set (A/B deltas stay)."""
        self.lora_kron = []
        for name in [n for n in self._parameters if n.startswith("lora_w")]:
            self.register_parameter(name, None)
        for name in [n for n in list(self.__dict__)
                     if n.startswith(("lora_w1_", "lora_w2_"))]:
            delattr(self, name)

    def clear_lora_deltas(self) -> None:
        """Remove every installed LoRA delta -- A/B set AND the LoKr
        factors (the module goes back to base).

        Also sweeps stray ``lora_A_*`` / ``lora_B_*`` plain-tensor
        attributes (defensive: older versions demoted them to __dict__
        on device moves) so no stale delta storage lingers.
        """
        self._clear_ab_deltas()
        self._clear_kron_deltas()
        for name in [n for n in self._parameters if n.startswith("lora_")]:
            self.register_parameter(name, None)

    def set_lora_kron(self, kron) -> None:
        """Install LoKr (Kronecker) deltas (replaces any previous set).

        ``kron``: ``(w1, w2)`` pairs; the module's weight delta is
        ``kron(w1, w2)`` (``w1`` is ``(out_l, in_l)``, ``w2`` is
        ``(out_k, in_k)``, ``out_l*out_k == out_features``,
        ``in_l*in_k == in_features``; the strength/scale is expected
        pre-folded into ``w1``). Registered as ``lora_w1_N`` /
        ``lora_w2_N`` parameters so ``.to(device)`` moves them with the
        module. Replaces any previous LoKr set (A/B deltas untouched).
        Call under the shared ``_PARAM_INIT_LOCK``.
        """
        self._clear_kron_deltas()
        w = self.weight
        device = w.data.device if w is not None else None
        dtype = (getattr(w, "computation_dtype", None)
                 or self.computation_dtype
                 or (w.dtype if w is not None else None))
        self.lora_kron = []
        for i, (w1, w2) in enumerate(kron):
            if device is not None:
                w1 = w1.to(device=device, dtype=dtype)
                w2 = w2.to(device=device, dtype=dtype)
            pw1 = torch.nn.Parameter(w1.detach(), requires_grad=False)
            pw2 = torch.nn.Parameter(w2.detach(), requires_grad=False)
            self.register_parameter(f"lora_w1_{i}", pw1)
            self.register_parameter(f"lora_w2_{i}", pw2)
            self.lora_kron.append((pw1, pw2))
        self.clear_lora_bias()


class GgufLinear(QuantLinear):
    """``nn.Linear`` replacement that keeps the weight quantized (GGUF
    Q4_K) and dequantizes it on every forward pass."""

    def _dequantize(self, w):
        return dequantize_tensor(w)


class Fp8Linear(QuantLinear):
    """``nn.Linear`` replacement: the ``F8_E4M3`` weight (and bias) stay
    fp8 on-device and are cast to the input dtype per forward (the cast IS
    the dequantize — the file holds no scale tensors)."""

    def _dequantize(self, w):
        return w


class Nf4Linear(QuantLinear):
    """``nn.Linear`` replacement: bitsandbytes NF4 4-bit packed weight
    (:class:`Nf4Param`), dequantized per forward with bnb's native
    pre-quantized format (``dequantize_4bit`` + ``QuantState``; the nested
    files' uint8 absmax indices are decoded through ``state2`` inside it).

    The ``bitsandbytes`` import is inside the function (not at module top
    level) so a CPU-only test env still imports this module.
    """

    def _dequantize(self, w):
        from bitsandbytes import functional as bfn
        from bitsandbytes.nn.modules import QuantState

        state2 = None
        offset = None
        if w.nested_absmax is not None:
            state2 = QuantState(
                absmax=w.nested_absmax, code=w.nested_quant_map,
                blocksize=w.nested_blocksize, dtype=w.nested_dtype)
            offset = torch.tensor(w.nested_offset, device=w.data.device)
        qs = QuantState(
            absmax=w.absmax, blocksize=w.blocksize, quant_type=w.quant_type,
            dtype=w.out_dtype, shape=torch.Size(w.real_shape),
            offset=offset, state2=state2)
        return bfn.dequantize_4bit(w.data, quant_state=qs)


# scope state (set by :func:`quant_linear_scope`)
_DEVICE = None
_DTYPE = None


@contextlib.contextmanager
def quant_linear_scope(kind, device, dtype, linear_cls=None):
    """Build ``nn.Linear`` layers as the quantized variant for ``kind``
    (``"gguf"`` / ``"fp8"`` / ``"nf4"``) inside the block.

    ``device``/``dtype`` seed each layer's dummy param (which fixes the
    weight device and the dequantize computation dtype). ``linear_cls``
    overrides the variant class (e.g. a scaled-fp8 ``Fp8Linear`` subclass).
    """
    global _DEVICE, _DTYPE
    prev_device, prev_dtype = _DEVICE, _DTYPE
    _DEVICE, _DTYPE = device, dtype
    original = torch.nn.Linear
    torch.nn.Linear = linear_cls or {"gguf": GgufLinear, "fp8": Fp8Linear, "nf4": Nf4Linear}[kind]
    try:
        yield
    finally:
        torch.nn.Linear = original
        # restore the previous scope: a leaked _DTYPE would seed every later
        # scope-less QuantLinear with the wrong dummy/computation dtype
        _DEVICE, _DTYPE = prev_device, prev_dtype


@contextlib.contextmanager
def quant_dtype_scope(device, dtype):
    """Seed the QuantLinear scope globals (``_DEVICE``/``_DTYPE``) WITHOUT
    replacing ``nn.Linear``.

    For modules that mix explicit ``QuantLinear`` subclasses with PLAIN
    ``nn.Linear`` layers (e.g. the Krea 2 transformer: the ``Int8Linear``
    block projections alongside the bf16 ``first``/``tmlp``/``txtmlp`` /...
    linears): the subclass constructors read the scope globals for their
    dummy parameter and computation dtype, while the plain layers ignore
    them.
    """
    global _DEVICE, _DTYPE
    prev_device, prev_dtype = _DEVICE, _DTYPE
    _DEVICE, _DTYPE = device, dtype
    try:
        yield
    finally:
        _DEVICE, _DTYPE = prev_device, prev_dtype


@contextlib.contextmanager
def gguf_linear_scope(device, dtype):
    """:func:`quant_linear_scope` for the GGUF (Q4_K) format."""
    with quant_linear_scope("gguf", device, dtype):
        yield


# --------------------------------------------------------------- loaders ----


_GGUF_MAGIC = b"GGUF"
_NF4_PREFIX = "model.diffusion_model."


def sniff_flux_checkpoint(path) -> str:
    """Detect the Flux checkpoint format: ``"gguf"`` / ``"fp8"`` / ``"nf4"``.

    The ``.gguf`` magic wins; else the safetensors key set decides: a
    ``*.quant_state.bitsandbytes__nf4`` key means bitsandbytes NF4
    (all-in-one pipeline file, only the transformer section is used); a
    ``.weight`` slice of dtype ``F8_E4M3`` means Kijai FP8. Anything else
    (e.g. plain bf16 flux weights, which need ~24 GB) is unsupported and
    raises.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == _GGUF_MAGIC:
        return "gguf"
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        keys = [str(k) for k in f.keys()]
        if any(k.endswith(".quant_state.bitsandbytes__nf4") for k in keys):
            return "nf4"
        if any(k.endswith(".weight") and f.get_slice(k).get_dtype() == "F8_E4M3"
               for k in keys):
            return "fp8"
    raise ValueError(
        f"unsupported Flux checkpoint (need .gguf / Kijai FP8 / bnb NF4): {path}")


def load_flux_state_dict(path, kind) -> dict:
    """Read a Flux transformer checkpoint into ``{module path: tensor}``
    (the GGUF state-dict convention: ``img_in.weight``,
    ``double_blocks.0.img_attn.qkv.weight``, ...).

    - ``"gguf"``: Q4_K bytes as :class:`GgufParam`
      (see :func:`load_gguf_state_dict`);
    - ``"nf4"``: bitsandbytes NF4 — only the ``model.diffusion_model.``
      section of the all-in-one pipeline file (``vae.`` /
      ``text_encoders.`` are the infra components and stay ignored);
      quantized linears become :class:`Nf4Param`, biases/scales are plain
      bf16 tensors;
    - ``"fp8"``: Kijai FP8 — transformer only, top-level keys; weights and
      biases stay ``F8_E4M3``.
    """
    if kind == "gguf":
        return load_gguf_state_dict(path)
    from safetensors import safe_open

    sd: dict = {}
    with safe_open(str(path), framework="pt") as f:
        keys = [str(k) for k in f.keys()]
        if kind == "nf4":
            _load_nf4_tensors(f, keys, sd)
        else:
            _load_fp8_tensors(f, keys, sd)
    return sd


def _load_fp8_tensors(f, keys, sd):
    # transformer only, top-level diffusers names; weights and biases stay
    # F8_E4M3 (the norm scales ride along as Parameters too: the default
    # load path casts them to the module dtype, exact for fp8 -> fp32)
    for key in keys:
        sd[key] = torch.nn.Parameter(f.get_tensor(key), requires_grad=False)


def _load_nf4_tensors(f, keys, sd):
    for key in keys:
        if not key.startswith(_NF4_PREFIX):
            continue  # vae. / text_encoders.: the infra components, unused
        name = key[len(_NF4_PREFIX):]
        if name.endswith(".weight"):
            blob = f.get_tensor(key + ".quant_state.bitsandbytes__nf4")
            meta = json.loads(blob.numpy().tobytes().decode())
            out, inn = meta["shape"]
            packed = f.get_tensor(key)
            if packed.dtype != torch.uint8 or packed.numel() != out * inn // 2:
                raise ValueError(
                    f"bad NF4 packed weight {key}: "
                    f"expect uint8 [{out * inn // 2}], "
                    f"got {packed.dtype} {tuple(packed.shape)}")
            # native bnb layout is [(n/2), 1]; accept the transposed one too
            if packed.dim() == 2 and packed.shape[0] == 1:
                packed = packed.t().contiguous().view(-1, 1)
            else:
                packed = packed.contiguous().view(-1, 1)
            nested = "nested_blocksize" in meta
            sd[name] = Nf4Param(
                packed,
                f.get_tensor(key + ".absmax"),
                blocksize=meta["blocksize"],
                quant_type=meta["quant_type"],
                out_dtype=getattr(torch, meta["dtype"]),
                real_shape=(out, inn),
                nested_absmax=(f.get_tensor(key + ".nested_absmax")
                               if nested else None),
                nested_quant_map=(f.get_tensor(key + ".nested_quant_map")
                                  if nested else None),
                nested_blocksize=meta.get("nested_blocksize"),
                nested_dtype=getattr(torch, meta.get("nested_dtype", "float32")),
                nested_offset=meta.get("nested_offset"),
            )
        elif name.endswith(".bias") or name.endswith(".scale"):
            # plain tensors cast to bf16 (norm scales: float32 in the V2
            # file, bf16 in the nested one)
            sd[name] = f.get_tensor(key).to(torch.bfloat16)
        # .weight.{quant_map,quant_state...,absmax,...}: packed into the
        # Nf4Param above; quant_map (the codebook) is dropped — the kernel
        # has the NF4 codebook built in
