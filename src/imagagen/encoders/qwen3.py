"""Qwen3-4B text encoder for Z-Image, ported from the ComfyUI reference
(``comfy/text_encoders/llama.py::Qwen3_4B`` + ``z_image.py``).

The shipped ``qwen_3_4b.safetensors`` (Kijai scaled-fp8 layout) stores the
embedding and all norms in bf16, and every attention/MLP projection as
``float8_e4m3fn`` with a per-linear ``scale_weight`` (fp32) and
``scale_input`` (bf16). The fp8 weights stay fp8 on the GPU and are cast to
bf16 per-op in :class:`ScaledFp8Linear` (the ``t5_encoder.Fp8Linear``
pattern), so the resident size stays ~4.4 GB instead of the ~8 GB a bf16
copy would need.

The encoder runs a CAUSAL decoder stack (GQA 32/8 heads, head_dim 128,
per-head q/k RMSNorm, RoPE theta 1e6, SiLU-gated MLP, no biases) and
returns the PENULTIMATE decoder layer's hidden state -- the layer the
reference Z-Image text-encoder target uses
(``Qwen3_4BModel(layer="hidden", layer_idx=-2, layer_norm_hidden_state=False)``
== HF ``hidden_states[-2]``, no final norm).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .zimage_prompt import ZIMAGE_PROMPT_TEMPLATE

COMPUTE_DTYPE = torch.bfloat16


class ScaledFp8Linear(nn.Linear):
    """``nn.Linear`` with a scaled-fp8 weight: the dequantized weight is
    ``W_fp8 * scale_weight`` (per-tensor, fp32 in the checkpoint).

    The checkpoint's ``scale_input`` is the *activation* quantization step
    (input_absmax / 448): the reference quantizes the activation to fp8 as
    ``q = x / scale_input`` and multiplies the GEMM result by
    ``scale_input * scale_weight``, so the activation scale CANCELS and never
    touches the dequantized weight. We keep the activation in bf16 (strictly
    more accurate than the reference's fp8 round-trip), so ``scale_input``
    must not be applied at all -- multiplying the input by it shrinks every
    layer output by an arbitrary per-layer factor (0.003 ... 0.86 in this
    file) and turns the conditioning into meaningless noise (verified with
    dev/check_qwen_lm.py: only the weight-scale-only reading makes the tied
    embedding head predict "Paris" after "The capital of France is"). The
    scale is still loaded, so a future true-fp8-activation path can use it.
    """

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        self.scale_weight = nn.Parameter(torch.empty(1), requires_grad=False)
        self.scale_input = nn.Parameter(torch.empty(1), requires_grad=False)

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.scale_weight.to(x.dtype)
        return F.linear(x, w)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.weight.dtype != x.dtype:
            self.weight = nn.Parameter(self.weight.to(dtype=x.dtype))
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


def _apply_rotary(x, cos, sin):
    """Standard rotate-half RoPE. ``x``: (b, s, h, d); ``cos``/``sin``:
    (s, d) (duplicated halves)."""
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


class Qwen3Attention(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = ScaledFp8Linear(hidden, n_heads * head_dim)
        self.k_proj = ScaledFp8Linear(hidden, n_kv_heads * head_dim)
        self.v_proj = ScaledFp8Linear(hidden, n_kv_heads * head_dim)
        self.o_proj = ScaledFp8Linear(n_heads * head_dim, hidden)
        self.q_norm = Qwen3RMSNorm(head_dim)
        self.k_norm = Qwen3RMSNorm(head_dim)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        hd = self.head_dim
        xq = self.q_norm(self.q_proj(x).view(b, s, self.n_heads, hd))
        xk = self.k_norm(self.k_proj(x).view(b, s, self.n_kv_heads, hd))
        xv = self.v_proj(x).view(b, s, self.n_kv_heads, hd)

        xq = _apply_rotary(xq, cos, sin)
        xk = _apply_rotary(xk, cos, sin)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)
        # CAUSAL: this is a decoder-only LLM and the reference (comfy
        # llama.py) always applies the causal mask for seq_len > 1; the
        # DiT was trained on the causal conditioning, so bidirectional
        # attention here silently destroys prompt adherence.
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * hd)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = ScaledFp8Linear(hidden, intermediate)
        self.up_proj = ScaledFp8Linear(hidden, intermediate)
        self.down_proj = ScaledFp8Linear(intermediate, hidden)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads, head_dim, intermediate):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden)
        self.self_attn = Qwen3Attention(hidden, n_heads, n_kv_heads, head_dim)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden)
        self.mlp = Qwen3MLP(hidden, intermediate)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    """Qwen3-4B decoder stack (no lm_head: tied weights, not needed for
    encoding). ``forward`` returns the penultimate layer's hidden state."""

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2560,
        intermediate_size=9728,
        num_layers=36,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        rope_theta=1_000_000.0,
    ):
        super().__init__()
        self.rope_theta = rope_theta
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(hidden_size, n_heads, n_kv_heads, head_dim,
                                  intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(hidden_size)

    def _rope(self, seq_len, device, dtype):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.layers[0].self_attn.head_dim, 2,
                             device=device, dtype=torch.float32)
                / self.layers[0].self_attn.head_dim)
        )
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        # (1, s, 1, d) so it broadcasts over (b, s, heads, d)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, :, None]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, :, None]
        return cos.to(dtype), sin.to(dtype)

    def forward(self, input_ids):
        x = F.embedding(input_ids, self.embed_tokens.weight.to(COMPUTE_DTYPE))
        cos, sin = self._rope(x.shape[1], x.device, COMPUTE_DTYPE)
        # the reference (comfy ZImageTE: layer="hidden", layer_idx=-2) asks
        # the text model for `intermediate_output = num_layers - 2`, which
        # comfy captures AFTER that layer ran -> the output of layer index 34
        # of 36, i.e. the penultimate layer's output with NO final norm
        # (HF equivalent: hidden_states[-2] of the 37-entry list). Verified
        # layer-by-layer against transformers' Qwen3Model
        # (dev/check_zimage_encoder.py).
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if i == len(self.layers) - 2:
                return x
        raise AssertionError("unreachable")


def load_zit_encoder(path, device):
    """Load ``qwen_3_4b.safetensors`` (scaled fp8) into a :class:`Qwen3Model`
    on ``device``."""
    import safetensors.torch as st

    file = st.load_file(str(path))
    model = Qwen3Model()
    params = dict(model.named_parameters())
    used = set()
    for key, t in file.items():
        if key == "scaled_fp8":
            continue  # format marker, no data
        k = key[len("model."):] if key.startswith("model.") else key
        if k not in params:
            raise ValueError(f"unexpected Z-Image encoder key: {key!r}")
        # ``param.data = t`` swaps storage + dtype (fp8 stays fp8)
        params[k].data = t
        used.add(k)
    missing = [k for k in params if k not in used]
    if missing:
        raise ValueError(f"Z-Image encoder file is missing keys: {missing}")
    model.to(device)
    return model


def zimage_prompt_template() -> str:
    """The Z-Image chat prompt template (local copy in
    :mod:`imagagen.encoders.zimage_prompt` of the ComfyUI reference's
    ``llama_template``)."""
    return ZIMAGE_PROMPT_TEMPLATE
