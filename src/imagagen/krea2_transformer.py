"""Krea 2 (K2) transformer: the SingleStreamDiT, int8 + ConvRot.

Port of the reference ``dev/implementations/krea-2/mmdit.py`` /
``comfy/ldm/krea2/model.py`` with the shipped checkpoint's layout:

- 28 shared single-stream blocks (features 6144, GQA 48/12 heads,
  head_dim 128), AdaLN-single ``DoubleSharedModulation``, RMSNorm with
  the zero-centered ``(1 + scale)`` weight convention, GQA attention
  with per-head q/k RMSNorm + 3-axis RoPE (theta 1e3, axes [32, 48, 48])
  and a sigmoid output gate, SwiGLU MLP (dim 16384, multiplier 4);
- a ``txtfusion`` adapter (2 layerwise blocks over the 12-tap
  encoder stack, a 12->1 ``projector``, 2 refiner blocks) and the
  ``txtmlp`` (RMSNorm + Linear + GELU(tanh) + Linear) that fuse the
  12 x 2560 taps into 6144;
- ``tmlp``/``tproj`` (timestep), ``first`` (patchify embedding),
  ``last`` (RMSNorm + ``SimpleModulation`` + output projection).

The shipped checkpoint (``krea2_Int8Extended.safetensors``) stores
EVERY ``blocks.N`` projection as ``int8_tensorwise`` (per-row fp32
``weight_scale`` plus a ``comfy_quant`` marker declaring
``convrot: true, convrot_groupsize: 256``): the int8 weights are
Hadamard-rotated along the input axis in groups of 256, so the runtime
rotates the ACTIVATION by the same Hadamard matrix before the GEMM
(:class:`Int8Linear`; the construction is ported from
``comfy_kitchen/tensor/int8_utils.py``). The int8 bytes stay resident
(~13.5 GB) and dequantize to bf16 per op (``W.to(bf16) * scale``, the
Forge pattern). Everything else is plain bf16, and every projection
(``first``, the ``blocks.N`` projections, the whole ``txtfusion``
adapter incl. the ``projector``, ``txtmlp`` / ``tmlp`` / ``tproj``,
``last.linear``) is a :class:`DeltaLinear` so ANY LoRA target accepts
the same runtime additive deltas as the int8 modules.

``forward(x, sigma, context)`` takes the (1, 16, h, w) bf16 latent, the
flow time ``sigma`` (1.0 = pure noise ... 0.0 = clean, passed raw to the
sinusoidal embedding with the reference's 1e3 time factor) and the
(1, seq, 12*2560) text conditioning, and returns the RAW model output
-- the flow VELOCITY under the reference's FLOW/CONST convention,
``x0 = x - sigma * v`` (the same reading as Z-Image).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .flux_transformer import timestep_embedding
from .gguf_ops import QuantLinear


# ---------------------------------------------------------- hadamard ----
# Ported from comfy_kitchen/tensor/int8_utils.py (the ConvRot rotation
# the checkpoint's comfy_quant markers declare: group size 256).


_H4 = torch.tensor(
    [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
    dtype=torch.float32,
)

#: (str(device), dtype) -> the Hadamard matrix
_HADAMARD_CACHE: dict[tuple, torch.Tensor] = {}


def _build_hadamard(size: int) -> torch.Tensor:
    """The ``size``-x-``size`` Hadamard matrix (size a power of 4):
    the 4x4 base kron-recursed up to ``size`` and scaled by
    ``1 / sqrt(size)`` so it is orthogonal (``H @ H.T == I``)."""
    h = _H4
    while h.shape[0] < size:
        h = torch.kron(h, _H4)
    return (h / math.sqrt(size)).contiguous()


def _hadamard(group: int, device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    h = _HADAMARD_CACHE.get(key)
    if h is None:
        h = _HADAMARD_CACHE.setdefault(key, _build_hadamard(group).to(
            device=device, dtype=dtype))
    return h


def _rotate(x: torch.Tensor, group: int) -> torch.Tensor:
    """Hadamard-rotate the last axis of ``x`` in groups of ``group``
    (the reference ``_rotate_activation``); any leading batch dims."""
    f = x.shape[-1]
    return (x.view(*x.shape[:-1], f // group, group)
            @ _hadamard(group, x.device, x.dtype)).view(*x.shape)


# ------------------------------------------------------------- rope (k2) ----


def _rope_axis(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """One position axis of the reference ``rope``: (b, n) positions ->
    (b, n, dim/2, 2, 2) adjacent-pair rotation matrices
    ``[[cos, -sin], [sin, cos]]`` with per-axis frequencies
    ``theta ** (-j / dim)`` (j = 0 .. dim/2 - 1) -- NOT the Flux-style
    rotate-half over duplicated frequencies."""
    scale = torch.arange(0, dim, 2, dtype=torch.float64,
                         device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    ang = pos[..., :, None].double() * omega  # (b, n, dim/2)
    c = torch.cos(ang)
    s = torch.sin(ang)
    del ang
    return torch.stack([c, -s, s, c], dim=-1).view(
        *pos.shape, dim // 2, 2, 2).float()


def _rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """The reference ``ropeapply``: ``x`` (b, h, n, d), ``freqs``
    (b, 1, n, d/2, 2, 2); adjacent pairs rotated in fp32, cast back."""
    shp = x.shape
    x_ = x.float().reshape(*shp[:-1], shp[-1] // 2, 1, 2)
    x_ = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return x_.reshape(*shp).to(x.dtype)


class Krea2PositionalEncoding(nn.Module):
    """The reference ``PositionalEncoding``: the head dim is split into the
    three position axes (``axdims``) and each axis gets its own
    adjacent-pair RoPE; concatenated along the (d/2) pair axis."""

    def __init__(self, axdims: list[int], theta: float):
        super().__init__()
        self.axdims = list(axdims)
        self.theta = theta

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """``pos`` (b, n, 3) -> (b, 1, n, sum(d/2), 2, 2)."""
        return torch.cat(
            [_rope_axis(pos[..., i], d, self.theta)
             for i, d in enumerate(self.axdims)],
            dim=-3,
        ).unsqueeze(1)


# --------------------------------------------------------- int8 linears ----


class Int8Linear(QuantLinear):
    """``nn.Linear`` kept as ``int8_tensorwise``: the ``weight`` stays int8
    ``[out, in]`` and the per-row fp32 ``weight_scale`` ``[out, 1]`` rides
    along; the dequantized weight is ``W.to(bf16) * scale`` per op.

    The checkpoint's ``comfy_quant`` marker declares ConvRot (group 256):
    the stored weights are Hadamard-rotated along the input axis, so the
    forward rotates the activation by the same matrix before the GEMM
    (the rotations cancel, the GEMM runs on the rotated input). The
    inherited runtime LoRA deltas are installed on the PLAIN (unrotated)
    input space instead: the LoRAs were trained against the de-rotated
    weights (every trainer -- diffusers / ai-toolkit / kohya, and even
    ComfyUI, which de-rotates ConvRot weights before merging a LoRA diff
    -- operates on plain weights), so :meth:`set_lora_deltas` rotates the
    A matrix by the same group Hadamard and the runtime path computes
    ``(x H) (A H)^T B^T == x A^T B^T`` (H is symmetric, H^2 = I).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 convrot_group: int | None = 256) -> None:
        super().__init__(in_features, out_features, bias)
        self.weight_scale = None  # f32 (out, 1), filled at load
        self._convrot_group = (convrot_group
                               if in_features % convrot_group == 0 else None)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        if prefix + "weight_scale" in state_dict:
            self.weight_scale = state_dict[prefix + "weight_scale"].to(
                device=self.weight.device)
            self.weight_scale.computation_dtype = self.computation_dtype

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse)
        # the per-row scales move with the weight (plain tensor, not a
        # registered parameter)
        if self.weight_scale is not None:
            self.weight_scale = fn(self.weight_scale)
        return out

    def _dequantize(self, w):
        return w.to(self.computation_dtype) * \
            self.weight_scale.to(self.computation_dtype)

    def set_lora_deltas(self, deltas) -> None:
        """Install runtime LoRA deltas, rotating the A matrices into the
        ConvRot space (see the class docstring): the deltas were trained
        on the de-rotated weights, so A becomes ``A @ H`` and the
        rotated-input runtime path reproduces the plain-space delta
        exactly."""
        if self._convrot_group is not None:
            deltas = [(_rotate(a, self._convrot_group), b, r0, r1)
                      for a, b, r0, r1 in deltas]
        super().set_lora_deltas(deltas)

    def forward(self, x):
        if self._convrot_group is not None:
            x = _rotate(x, self._convrot_group)
        return super().forward(x)


class DeltaLinear(nn.Linear):
    """Plain bf16 ``nn.Linear`` that also accepts runtime LoRA weight
    deltas (the same ``set_lora_deltas``/``clear_lora_deltas`` contract
    as :class:`~imagagen.gguf_ops.QuantLinear`): the txtfusion
    projections are the K2 LoRA's non-quantized targets."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.lora_deltas: list[tuple[nn.Parameter, nn.Parameter, int, int]] = []

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        for a, b, r0, r1 in self.lora_deltas:
            out[..., r0:r1] += (x @ a.t()) @ b.t()
        return out

    def set_lora_deltas(self, deltas) -> None:
        """Install runtime LoRA deltas (replaces any previous set).

        ``deltas``: ``(A, B, r0, r1)`` tuples; ``A`` is ``(rank,
        in_features)``, ``B`` is ``(r1 - r0, rank)``. The A/B tensors are
        registered parameters (named ``lora_A_N`` / ``lora_B_N``) so
        ``.to(device)`` moves them with the module. Call under the shared
        ``_PARAM_INIT_LOCK`` (see :meth:`QuantLinear.set_lora_deltas`).
        """
        self.clear_lora_deltas()
        device = self.weight.device
        dtype = self.weight.dtype
        self.lora_deltas = []
        for i, (a, b, r0, r1) in enumerate(deltas):
            pa = nn.Parameter(
                a.detach().to(device=device, dtype=dtype), requires_grad=False)
            pb = nn.Parameter(
                b.detach().to(device=device, dtype=dtype), requires_grad=False)
            self.register_parameter(f"lora_A_{i}", pa)
            self.register_parameter(f"lora_B_{i}", pb)
            self.lora_deltas.append((pa, pb, r0, r1))

    def clear_lora_deltas(self) -> None:
        """Remove every installed LoRA delta (the module goes back to
        base)."""
        self.lora_deltas = []
        for name in [n for n in self._parameters if n.startswith("lora_")]:
            self.register_parameter(name, None)


# ----------------------------------------------------------------- norms ----


class RMSNorm(nn.Module):
    """Affine RMSNorm with the zero-centered ``(1 + scale)`` weight
    convention (checkpoint keys ``<x>.scale``; the reference normalizes
    in fp32 and casts back)."""

    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, dtype=torch.float32))

    def forward(self, x):
        dtype = x.dtype
        w = self.scale.to(device=x.device, dtype=torch.float32) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=w,
                          eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


# -------------------------------------------------------------- modulate ----


class SimpleModulation(nn.Module):
    """``vec + lin`` (``lin`` a ``[2, dim]`` Parameter), chunked into
    scale/shift (state-dict key ``modulation.lin``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim, dtype=torch.float32))

    def forward(self, vec):
        lin = self.lin.to(device=vec.device, dtype=vec.dtype).unsqueeze(0)
        out = vec + lin
        return out.chunk(2, dim=1)


class DoubleSharedModulation(nn.Module):
    """``vec + lin`` (``lin`` a ``[6 * dim]`` Parameter), chunked into the
    6 adaLN values (state-dict key ``mod.lin``)."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim, dtype=torch.float32))

    def forward(self, vec):
        lin = self.lin.to(device=vec.device, dtype=vec.dtype)
        return (vec + lin).chunk(6, dim=-1)


# ------------------------------------------------------------- attention ----


class Attention(nn.Module):
    """GQA multi-head attention with per-head q/k RMSNorm, optional 3-axis
    RoPE and a sigmoid output gate: ``wo(out * sigmoid(gate(x)))``.

    State-dict keys: ``wq`` / ``wk`` / ``wv`` / ``gate`` / ``qknorm`` /
    ``wo`` (no biases in the checkpoint).
    """

    def __init__(self, dim: int, heads: int, linear_cls, bias: bool = False,
                 kvheads: int | None = None):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // heads
        self.wq = linear_cls(dim, self.headdim * self.heads, bias)
        self.wk = linear_cls(dim, self.headdim * self.kvheads, bias)
        self.wv = linear_cls(dim, self.headdim * self.kvheads, bias)
        self.gate = linear_cls(dim, dim, bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = linear_cls(dim, dim, bias)

    def forward(self, x, freqs=None):
        b, l, _ = x.shape
        q = self.wq(x).view(b, l, self.heads, self.headdim).transpose(1, 2)
        k = self.wk(x).view(b, l, self.kvheads, self.headdim).transpose(1, 2)
        v = self.wv(x).view(b, l, self.kvheads, self.headdim).transpose(1, 2)
        gate = self.gate(x)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q = _rope_apply(q, freqs)
            k = _rope_apply(k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, l, self.heads * self.headdim)
        return self.wo(out * F.sigmoid(gate))


class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))`` (state-dict keys ``gate`` / ``up`` /
    ``down``; ``mlpdim = 128 * ceil((2*features/3) * multiplier / 128)``)."""

    def __init__(self, features: int, linear_cls, multiplier: int,
                 bias: bool = False, multiple: int = 128):
        super().__init__()
        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = linear_cls(features, mlpdim, bias)
        self.up = linear_cls(features, mlpdim, bias)
        self.down = linear_cls(mlpdim, features, bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ----------------------------------------------------------------- blocks ----


class SingleStreamBlock(nn.Module):
    """One shared single-stream block: adaLN-single (``DoubleSharedModulation``
    over the ``tproj``-ed timestep) + gated attention + SwiGLU."""

    def __init__(self, features: int, heads: int, linear_cls,
                 multiplier: int, bias: bool = False,
                 kvheads: int | None = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, linear_cls, bias, kvheads)
        self.mlp = SwiGLU(features, linear_cls, multiplier, bias)

    def forward(self, x, vec, freqs):
        prescale, preshift, pregate, postscale, postshift, postgate = \
            self.mod(vec)
        x = x + pregate * self.attn(
            (1 + prescale) * self.prenorm(x) + preshift, freqs)
        x = x + postgate * self.mlp(
            (1 + postscale) * self.postnorm(x) + postshift)
        return x


class TextFusionBlock(nn.Module):
    """Pre/post-norm attention + SwiGLU (no adaLN, no RoPE: the reference
    runs the txtfusion blocks without frequency embeddings)."""

    def __init__(self, features: int, heads: int, linear_cls,
                 multiplier: int, bias: bool = False,
                 kvheads: int | None = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, linear_cls, bias, kvheads)
        self.mlp = SwiGLU(features, linear_cls, multiplier, bias)

    def forward(self, x):
        x = x + self.attn(self.prenorm(x))
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    """Fuses the 12-tap encoder stack: 2 layerwise blocks run independently
    over each tap (``(b*12, seq, 2560)``), the ``projector`` (12 -> 1, no
    bias) collapses the tap axis, then 2 refiner blocks refine the result
    over the full sequence."""

    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int,
                 linear_cls, multiplier: int, bias: bool = False,
                 kvheads: int | None = None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, linear_cls, multiplier, bias,
                             kvheads) for _ in range(2)]
        )
        self.projector = DeltaLinear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList(
            [TextFusionBlock(txt_dim, heads, linear_cls, multiplier, bias,
                             kvheads) for _ in range(2)]
        )

    def forward(self, x):
        # x: (b, seq, n_taps, d)
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x)
        # (b*l, n, d) -> (b, l, d, n): the projector collapses the taps
        # (the reference rearranges "(b l) n d -> b l d n" -- a PERMUTE, not
        # a raw reshape)
        x = self.projector(x.reshape(b, l, n, d).permute(0, 1, 3, 2))
        x = x.squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x)
        return x


class LastLayer(nn.Module):
    """RMSNorm + scale/shift modulate (``SimpleModulation``) + the output
    projection to ``patch * patch * channels`` (state-dict keys ``norm`` /
    ``linear`` / ``modulation``)."""

    def __init__(self, features: int, patch: int, channels: int,
                linear_cls=DeltaLinear):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = linear_cls(features, patch * patch * channels,
                                 bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        return self.linear(x)


# ------------------------------------------------------------------- dit ----


class Krea2Transformer(nn.Module):
    """The Krea 2 SingleStreamDiT (the shipped checkpoint's config):
    features 6144, tdim 256, txtdim 2560, 48 heads / 12 kv heads (head_dim
    128), multiplier 4 (MLP 16384), 28 blocks, patch 2, channels 16,
    12 text taps, RoPE theta 1e3 with 3-axis dims [32, 48, 48].

    The block projections are :class:`Int8Linear` (int8 + ConvRot);
    every other projection (``first``, txtfusion incl. the ``projector``,
    ``txtmlp`` / ``tmlp`` / ``tproj`` linears, ``last.linear``) is a
    :class:`DeltaLinear` (bf16, delta-capable); only the norms and the
    modulation params are plain bf16.

    Construct under ``quant_dtype_scope(cpu, bf16)`` (the Int8Linear
    dummies are seeded from the scope; no ``nn.Linear`` replacement, the
    module mixes Int8Linear with plain bf16 linears), then
    ``load_state_dict`` the checkpoint (minus the ``*.comfy_quant``
    markers).
    """

    def __init__(self, features: int = 6144, tdim: int = 256, txtdim: int = 2560,
                 heads: int = 48, kvheads: int | None = 12,
                 multiplier: int = 4, layers: int = 28, patch: int = 2,
                 channels: int = 16, bias: bool = False, theta: float = 1e3,
                 txtlayers: int = 12, txtheads: int = 20,
                 txtkvheads: int | None = 20) -> None:
        super().__init__()
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        # how the head dim splits across the 3 position axes (reference
        # ``inference.py``: [32, 48, 48] for head_dim 128)
        axes = [headdim - 12 * (headdim // 16),
                6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != head dim {headdim}"
        self.pe_embedder = Krea2PositionalEncoding(axes, theta)
        self.first = DeltaLinear(channels * patch ** 2, features, bias=True)
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(features, heads, Int8Linear, multiplier,
                               bias, kvheads) for _ in range(layers)]
        )
        self.tmlp = nn.Sequential(
            DeltaLinear(tdim, features, bias=True),
            nn.GELU(approximate="tanh"),
            DeltaLinear(features, features, bias=True),
        )
        self.txtfusion = TextFusionTransformer(
            txtlayers, txtdim, txtheads, DeltaLinear, multiplier, bias,
            txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            DeltaLinear(txtdim, features, bias=True),
            nn.GELU(approximate="tanh"),
            DeltaLinear(features, features, bias=True),
        )
        self.last = LastLayer(features, patch, channels, DeltaLinear)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            DeltaLinear(features, features * 6, bias=True),
        )

    def forward(self, x, sigma, context):
        """``x``: (1, 16, h, w) bf16; ``sigma``: (1,) flow time (1 = noise);
        ``context``: (1, seq, txtlayers * txtdim). Returns the raw model
        output -- the flow velocity -- as (1, 16, h, w)."""
        bs, c, h, w = x.shape
        p = self.patch
        # latent dims are multiples of the patch (the 16-px snap); pad
        # defensively like Z-Image
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2] // p, x.shape[-1] // p

        # the reference patchifies as ``b c (h ph) (w pw) -> b (h w) (c ph
        # pw)``: the feature order is (c, ph, pw)
        img = x.view(bs, c, hp, p, wp, p).permute(0, 2, 4, 1, 3, 5) \
            .flatten(1, 2).flatten(2)
        img = self.first(img)
        imglen = img.shape[1]

        # the reference passes the RAW flow time and multiplies by the
        # 1e3 time factor inside the sinusoidal embedding (temb)
        t = self.tmlp(timestep_embedding(sigma, self.tdim).unsqueeze(1).to(
            img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(
            context.reshape(bs, -1, self.txtlayers, self.txtdim))
        context = self.txtmlp(context)
        txtlen = context.shape[1]

        combined = torch.cat([context, img], dim=1)
        # positions: text at (0, 0, 0), image at (0, h, w); batch 1 with no
        # padding means no attention mask (the reference pads to 256 and
        # masks the padding out, which is equivalent to skipping both)
        device = x.device
        imgpos = torch.zeros(bs, imglen, 3, device=device,
                             dtype=torch.float32)
        imgpos[:, :, 1] = torch.arange(hp, device=device,
                                      dtype=torch.float32).view(-1, 1) \
            .expand(hp, wp).reshape(1, -1)
        imgpos[:, :, 2] = torch.arange(wp, device=device,
                                      dtype=torch.float32).view(1, -1) \
            .expand(hp, wp).reshape(1, -1)
        pos = torch.cat([torch.zeros(bs, txtlen, 3, device=device,
                                    dtype=torch.float32), imgpos], dim=1)
        freqs = self.pe_embedder(pos)
        del pos, imgpos

        for block in self.blocks:
            combined = block(combined, tvec, freqs)
        final = self.last(combined, t)
        del combined, tvec, freqs

        out = final[:, txtlen:txtlen + imglen, :]
        # the inverse of the patchify: features (c, ph, pw)
        return (out.view(bs, hp, wp, c, p, p).permute(0, 3, 1, 4, 2, 5)
                .reshape(bs, c, hp * p, wp * p))[:, :, :h, :w]
