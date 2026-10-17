"""T5-XXL text encoder for FLUX, ported from the Forge reference
(``backend/nn/t5.py``).

The shipped ``t5xxl.safetensors`` stores the weights in
``float8_e4m3fn`` (~4.9 GB). They are kept in fp8 on the GPU and cast to the
compute dtype (bf16) per-op in :class:`Fp8Linear`, so the resident size stays
~4.9 GB instead of the ~9.8 GB a bf16 copy would need.

Module names mirror the safetensors keys exactly (``encoder.block.N.layer.M.
SelfAttention.q.weight``, ``...DenseReluDense.wi_0.weight``, ``shared.weight``),
so the state dict loads directly. The unused ``encoder.embed_tokens.weight``
(duplicate of ``shared``) is ignored.

Attention uses ``torch.nn.functional.scaled_dot_product_attention`` with
``scale=1.0`` (T5's self-attention applies no ``1/sqrt(d_k)`` scaling — the
reference folds ``sqrt(d_k)`` into ``k`` and divides it back out, netting to
scale 1), with the shared bidirectional relative-position bias as the mask.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

COMPUTE_DTYPE = torch.bfloat16


class Fp8Linear(nn.Linear):
    """``nn.Linear`` whose fp8 weight is cast to the input dtype per forward."""

    def forward(self, x):
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(x.dtype) * x


class T5DenseGatedActDense(nn.Module):
    def __init__(self, model_dim, ff_dim):
        super().__init__()
        self.wi_0 = Fp8Linear(model_dim, ff_dim, bias=False)
        self.wi_1 = Fp8Linear(model_dim, ff_dim, bias=False)
        self.wo = Fp8Linear(ff_dim, model_dim, bias=False)

    def forward(self, x):
        hidden_relu = F.relu(self.wi_0(x))
        hidden_linear = self.wi_1(x)
        x = hidden_relu * hidden_linear
        return self.wo(x)


class T5LayerFF(nn.Module):
    def __init__(self, model_dim, ff_dim):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(model_dim, ff_dim)
        self.layer_norm = T5LayerNorm(model_dim)

    def forward(self, x):
        forwarded_states = self.layer_norm(x)
        forwarded_states = self.DenseReluDense(forwarded_states)
        x += forwarded_states
        return x


class T5Attention(nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, has_relative_bias):
        super().__init__()
        self.q = Fp8Linear(model_dim, inner_dim, bias=False)
        self.k = Fp8Linear(model_dim, inner_dim, bias=False)
        self.v = Fp8Linear(model_dim, inner_dim, bias=False)
        self.o = Fp8Linear(inner_dim, model_dim, bias=False)
        self.num_heads = num_heads
        self.d_head = inner_dim // num_heads
        self.relative_attention_bias = None
        if has_relative_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.num_heads
            )

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets_half = num_buckets // 2
        relative_buckets = relative_buckets + (
            (relative_position > 0).to(torch.long) * num_buckets_half
        )
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets_half // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets_half - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets_half - 1),
        )
        relative_buckets = relative_buckets + torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device, dtype):
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket).to(dtype)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, x, past_bias=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # block 0 computes the shared relative bias; later blocks reuse it
        if past_bias is None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], x.device, x.dtype)

        b, s, inner = q.shape
        h, d = self.num_heads, self.d_head
        q = q.view(b, s, h, d).transpose(1, 2)
        k = k.view(b, s, h, d).transpose(1, 2)
        v = v.view(b, s, h, d).transpose(1, 2)
        # T5 self-attention: no 1/sqrt(d_k) scaling (scale=1.0), relative bias added.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=past_bias, scale=1.0)
        out = out.transpose(1, 2).reshape(b, s, inner)
        return self.o(out), past_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(self, model_dim, inner_dim, num_heads, has_relative_bias):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, has_relative_bias)
        self.layer_norm = T5LayerNorm(model_dim)

    def forward(self, x, past_bias=None):
        output, past_bias = self.SelfAttention(self.layer_norm(x), past_bias=past_bias)
        x += output
        return x, past_bias


class T5Block(nn.Module):
    def __init__(self, model_dim, inner_dim, ff_dim, num_heads, has_relative_bias):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(model_dim, inner_dim, num_heads, has_relative_bias))
        self.layer.append(T5LayerFF(model_dim, ff_dim))

    def forward(self, x, past_bias=None):
        x, past_bias = self.layer[0](x, past_bias)
        x = self.layer[-1](x)
        return x, past_bias


class T5Stack(nn.Module):
    def __init__(self, num_layers, model_dim, inner_dim, ff_dim, num_heads):
        super().__init__()
        self.block = nn.ModuleList(
            [
                T5Block(model_dim, inner_dim, ff_dim, num_heads, has_relative_bias=(i == 0))
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(model_dim)

    def forward(self, x):
        past_bias = None
        for blk in self.block:
            x, past_bias = blk(x, past_bias)
        x = self.final_layer_norm(x)
        return x


class T5(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, vocab_size, inner_dim=None):
        super().__init__()
        inner_dim = inner_dim or d_model
        self.encoder = T5Stack(num_layers, d_model, inner_dim, d_ff, num_heads)
        self.shared = nn.Embedding(vocab_size, d_model)
        self.compute_dtype = COMPUTE_DTYPE

    def forward(self, input_ids):
        w = self.shared.weight.to(self.compute_dtype)
        x = F.embedding(input_ids, w)
        x = torch.nan_to_num(x)
        return self.encoder(x)


def T5XXL(
    num_layers=24,
    d_model=4096,
    d_ff=10240,
    num_heads=64,
    vocab_size=32128,
):
    return T5(
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        vocab_size=vocab_size,
    )


def load_t5_encoder(path, device):
    """Load ``t5xxl.safetensors`` (fp8) into a :class:`T5` on ``device``."""
    import safetensors.torch as st

    file = st.load_file(path)
    model = T5XXL()
    # assign fp8 tensors directly (no upcast): ``param.data = t`` swaps storage+dtype
    params = dict(model.named_parameters())
    for name, p in params.items():
        if name in file:
            p.data = file[name]
    model.to(device)
    return model
