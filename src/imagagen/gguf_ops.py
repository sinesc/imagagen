"""GGUF loading + on-the-fly GPU dequantization for the Flux transformer.

The 12B FLUX.1-dev transformer only ships as a GGUF Q4_K_S file, so its
weights cannot be materialized in bf16 at once (that would be ~24 GB). We keep
the raw quantized bytes resident on the GPU and dequantize each ``nn.Linear``
weight on every forward pass (the Forge approach). Peak VRAM is then just the
~6.8 GB of quantized bytes plus one layer's transient bf16 weight.

The Q4_K kernels, ``GgufParam`` and the ``Linear`` replacement are ported
verbatim from the Forge reference
(``dev/implementations/stable-diffusion-webui-forge/packages_3rdparty/gguf``
and ``backend/operations*.py``) so the dequantized values match Forge
bit-for-bit. The only PyPI dependency used is ``gguf`` (for ``GGUFReader``);
the dequantize kernels themselves are self-contained torch ops.
"""

from __future__ import annotations

import contextlib
import torch

# Q4_K super-block = 256 elements, 144 quantized bytes.
QK_K = 256
_Q4K_BLOCK_SIZE = 256
_Q4K_TYPE_SIZE = 144
_K_SCALE_SIZE = 12


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
_4BIT_TABLE_U = None


def _build_table_u():
    global _4BIT_TABLE_U
    if _4BIT_TABLE_U is None:
        _4BIT_TABLE_U = _native_unpack_u(
            torch.arange(start=0, end=256 * 256, dtype=torch.long).to(torch.uint16)
        )[:, 0]
    return _4BIT_TABLE_U


def quick_unpack_4bits_u(x):
    if not _USE_LOOKUP:
        return torch.stack([x & 15, x >> 4], dim=-1).view(x.size(0), -1)
    table = _build_table_u()
    if table.device != x.device:
        table = table.to(device=x.device)
        global _4BIT_TABLE_U
        _4BIT_TABLE_U = table
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


# ------------------------------------------------------------- Parameter ----


# GGUF qtype enum value 12 == Q4_K (verified against the shipped file).
_Q4K = 12


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


def dequantize_tensor(tensor):
    if tensor is None:
        return None
    if tensor.gguf_type == _Q4K:
        return q4k_dequantize_pytorch(tensor)
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


class GgufLinear(torch.nn.Module):
    """``nn.Linear`` replacement that keeps the weight quantized and
    dequantizes it on every forward pass."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dummy = torch.nn.Parameter(torch.empty(1, device=_DEVICE, dtype=_DTYPE))
        self.weight = None
        self.bias = None
        self.lora_deltas: list[tuple[torch.nn.Parameter, torch.nn.Parameter, int, int]] = []

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
        for k, p in self.named_parameters(recurse=False, remove_duplicate=True):
            setattr(self, k, fn(p))
        return self

    def forward(self, x):
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias = _tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))
        weight = self.weight
        if weight is not None:
            # the dequantized weight is cast to the input dtype (Forge's
            # weights_manual_cast, skip_weight_dtype=False); the Q4_K kernel
            # itself promotes to fp32 (uint8 indices), so this cast is what
            # lands the weight in the computation dtype
            weight = weight.to(device=x.device)
            weight = dequantize_tensor(weight).to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device) if self.bias is not None else None
        out = torch.nn.functional.linear(x, weight, bias)
        if self.lora_deltas:
            # runtime LoRA deltas (Forge's online mode): the base weight
            # stays quantized, each delta adds its (rank, in) @ (out, rank)
            # product to a slice of the output rows. A/B are registered
            # parameters, so .to(device)/unload moves them with the weight
            # and they cost one extra GEMM pair per module.
            for a, b, r0, r1 in self.lora_deltas:
                out[..., r0:r1] += (x @ a.t()) @ b.t()
        return out

    def set_lora_deltas(self, deltas) -> None:
        """Install runtime LoRA deltas (replaces any previous set).

        ``deltas``: ``(A, B, r0, r1)`` tuples; ``A`` is ``(rank,
        in_features)``, ``B`` is ``(r1 - r0, rank)`` (any device/dtype,
        moved to the weight's). The A/B tensors are registered as
        parameters (named ``lora_A_N`` / ``lora_B_N``) so ``.to(device)``
        moves them with the module. Call under the shared
        ``_PARAM_INIT_LOCK`` (parameter creation is not thread-safe across
        backends, see backends.py).
        """
        self.clear_lora_deltas()
        w = self.weight
        device = w.data.device if w is not None else None
        dtype = getattr(w, "computation_dtype", None) or (w.dtype if w is not None else None)
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

    def clear_lora_deltas(self) -> None:
        """Remove every installed LoRA delta (the module goes back to base)."""
        self.lora_deltas = []
        for name in [n for n in self._parameters if n.startswith("lora_")]:
            self.register_parameter(name, None)


# scope state (set by :func:`gguf_linear_scope`)
_DEVICE = None
_DTYPE = None


@contextlib.contextmanager
def gguf_linear_scope(device, dtype):
    """Build ``nn.Linear`` layers as :class:`GgufLinear` inside the block.

    ``device``/``dtype`` seed each layer's dummy param (which fixes the weight
    device and the dequantize computation dtype).
    """
    global _DEVICE, _DTYPE
    _DEVICE, _DTYPE = device, dtype
    original = torch.nn.Linear
    torch.nn.Linear = GgufLinear
    try:
        yield
    finally:
        torch.nn.Linear = original
