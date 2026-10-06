"""Krea 2 text encoder: the Qwen3-VL-4B text tower, 12-layer tap.

K2 conditions on a STACK of hidden states from 12 layers of Qwen3-VL-4B
(the reference taps ``hidden_states[2, 5, 8, ..., 35]`` in HF numbering,
``hidden_states[0]`` = the embedding, no final norm applied).

The shipped ``infra/qwen_3vl_4b.safetensors`` stores the embedding and all
norms in bf16 and every attention/MLP projection as plain ``F8_E4M3``
with NO scale tensors (unlike the Z-Image ``qwen_3_4b.safetensors``): the
fp8 weights stay fp8 on the GPU and are cast to bf16 per op in
:class:`Fp8Linear` (the ``t5_encoder.Fp8Linear`` cast-is-dequantize
pattern), so the resident size stays ~4.8 GB instead of the ~9.6 GB a
bf16 copy would need.

The text tower is a CAUSAL decoder (GQA 32/8 heads, head_dim 128,
per-head q/k RMSNorm eps 1e-6, RoPE theta 5e6 over the full head dim --
with text-only input the Qwen3-VL interleaved mRoPE degenerates to plain
1D positions, SiLU-gated MLP, no biases, vocab 151936, no lm_head). The
checkpoint's ``model.visual.*`` (the vision tower) and ``model.norm``
(the final norm, unused: the taps are raw layer outputs) are skipped at
load.

Prompt handling (the reference ``Krea2TEModel.encode_token_weights``):
the prompt is chat-templated with :func:`krea2_prompt_template`, the
leading system + user-opening prefix is stripped (cut at the SECOND
im_start token, including the following ``user\\n`` tokens when present),
and the 12-tap stack is flattened to ``(1, seq, 12 * 2560)``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

COMPUTE_DTYPE = torch.bfloat16

#: HF hidden-states indices of the 12 taps (no offset; hidden_states[0]
#: is the embedding)
KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

#: the chat special tokens for the prefix strip: the im_start marker,
#: "user", "\n"
_IM_START, _USER, _NEWLINE = 151644, 872, 198


class Fp8Linear(nn.Linear):
    """``nn.Linear`` with a plain ``F8_E4M3`` weight: the cast IS the
    dequantize (the checkpoint stores no scale tensors)."""

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Qwen3RMSNorm(nn.Module):
    """Plain affine RMSNorm (checkpoint keys ``*.weight``; bf16)."""

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
        self.q_proj = Fp8Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = Fp8Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.v_proj = Fp8Linear(hidden, n_kv_heads * head_dim, bias=False)
        self.o_proj = Fp8Linear(n_heads * head_dim, hidden, bias=False)
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
        # CAUSAL: this is a decoder-only LLM; the reference always applies
        # the causal mask and the DiT was trained on the causal
        # conditioning, so bidirectional attention here silently destroys
        # prompt adherence.
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * hd)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = Fp8Linear(hidden, intermediate, bias=False)
        self.up_proj = Fp8Linear(hidden, intermediate, bias=False)
        self.down_proj = Fp8Linear(intermediate, hidden, bias=False)

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


class Krea2Qwen3VLModel(nn.Module):
    """The Qwen3-VL-4B TEXT tower (no vision tower, no lm_head: both are
    unused for image conditioning). ``forward`` returns the 12-tap stack
    ``(b, 12, s, 2560)`` of the tapped layers' hidden states (HF
    ``hidden_states[2, 5, 8, ..., 35]``, no final norm)."""

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2560,
        intermediate_size=9728,
        num_layers=36,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        rope_theta=5_000_000.0,
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
        taps: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            # layer i (0-based) outputs HF hidden_states[i + 1]
            if i + 1 in KREA2_TAP_LAYERS:
                taps.append(x)
        return torch.stack(taps, dim=1)  # (b, 12, s, hidden)


def load_krea2_encoder(path, device):
    """Load ``qwen_3vl_4b.safetensors`` (plain fp8 projections, bf16
    embedding/norms) into a :class:`Krea2Qwen3VLModel` on ``device``.

    The ``model.visual.*`` keys (the unused vision tower) and
    ``model.norm.weight`` (the final norm -- the taps are raw layer
    outputs, no final norm) are skipped.
    """
    import safetensors.torch as st

    file = st.load_file(str(path))
    model = Krea2Qwen3VLModel()
    params = dict(model.named_parameters())
    used = set()
    for key, t in file.items():
        if key.startswith("model.visual."):
            continue  # the vision tower: unused for image conditioning
        k = key[len("model."):] if key.startswith("model.") else key
        if k == "norm.weight":
            continue  # final norm: unused (raw tap outputs, no final norm)
        if k not in params:
            raise ValueError(f"unexpected Krea 2 encoder key: {key!r}")
        # ``param.data = t`` swaps storage + dtype (fp8 stays fp8)
        params[k].data = t
        used.add(k)
    missing = [k for k in params if k not in used]
    if missing:
        raise ValueError(f"Krea 2 encoder file is missing keys: {missing}")
    model.to(device)
    return model


def template_prefix_cut(token_ids: list[int]) -> int:
    """The number of leading tokens to strip from a chat-templated
    prompt (the reference ``Krea2TEModel.encode_token_weights``): the cut
    sits at the SECOND im_start token, extended past the following
    ``user`` + newline tokens when present."""
    template_end = None
    count = 0
    for i, tok in enumerate(token_ids):
        if tok == _IM_START and count < 2:
            template_end = i
            count += 1
    if template_end is not None and len(token_ids) > template_end + 3 \
            and token_ids[template_end + 1] == _USER \
            and token_ids[template_end + 2] == _NEWLINE:
        template_end += 3
    return template_end if template_end is not None else 0


def krea2_prompt_template() -> str:
    """The Krea 2 chat prompt template (local copy in
    :mod:`imagagen.encoders.krea2_prompt` of the ComfyUI reference's
    ``KREA2_TEMPLATE``)."""
    from .krea2_prompt import KREA2_PROMPT_TEMPLATE

    return KREA2_PROMPT_TEMPLATE
