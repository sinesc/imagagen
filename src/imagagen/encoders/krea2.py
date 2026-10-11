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

from contextlib import contextmanager

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

#: Qwen3-VL interleaved MRoPE section (rope_dims in the reference's
#: Qwen3VL_8BConfig; head_dim 128 = 24 + 20 + 20 doubled to 64 freqs)
_MROPE_SECTION = (24, 20, 20)

#: end-of-text (im_end) and im_start: the greedy caption stop tokens
_END_OF_TEXT = 151645
#: the image-pad token the vision features are scattered into
IMAGE_PAD_ID = 151655


class Fp8Linear(nn.Linear):
    """``nn.Linear`` with a plain ``F8_E4M3`` weight: the cast IS the
    dequantize (the checkpoint stores no scale tensors). ``_decode_cache``
    (set by :func:`decode_weight_cache`) holds a pre-dequantized copy for
    the caption decode phase: re-casting the full tower per op costs
    ~250 casts per generated token."""

    _decode_cache = None

    def forward(self, x):
        w = self._decode_cache
        if w is not None:
            return F.linear(x, w)
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

    def forward(self, x, cos, sin, past_kv=None):
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
        if past_kv is None:
            # prefill: preallocate a buffer with room for the generated
            # tokens (growing the cache by torch.cat per step copies the
            # whole prefix through 36 layers x 128 steps = tens of s)
            cap = s + 768
            kbuf = xk.new_zeros(b, self.n_kv_heads, cap, hd)
            vbuf = xv.new_zeros(b, self.n_kv_heads, cap, hd)
            kbuf[:, :, :s] = xk
            vbuf[:, :, :s] = xv
            kv = (kbuf, vbuf, s)
        else:
            # decode step: the new token attends to the cached prefix
            kbuf, vbuf, n = past_kv
            if n + s > kbuf.shape[2]:
                kbuf = torch.cat([kbuf, kbuf.new_zeros(
                    b, self.n_kv_heads, kbuf.shape[2], hd)], dim=2)
                vbuf = torch.cat([vbuf, vbuf.new_zeros(
                    b, self.n_kv_heads, vbuf.shape[2], hd)], dim=2)
            kbuf[:, :, n:n + s] = xk
            vbuf[:, :, n:n + s] = xv
            kv = (kbuf, vbuf, n + s)
            xk = kbuf[:, :, :n + s]
            xv = vbuf[:, :, :n + s]
        n_rep = self.n_heads // self.n_kv_heads
        if s == 1 and n_rep > 1:
            # cached decode: broadcast k/v over the GQA group instead of
            # materializing the repeated cache every step
            bk = b * self.n_kv_heads
            n = xk.shape[2]
            qg = xq.reshape(bk, n_rep, 1, hd)
            kg = xk.reshape(bk, 1, n, hd)
            vg = xv.reshape(bk, 1, n, hd)
            out = F.scaled_dot_product_attention(qg, kg, vg)
            out = out.reshape(b, self.n_heads, 1, hd)
        else:
            if n_rep > 1:
                xk = xk.repeat_interleave(n_rep, dim=1)
                xv = xv.repeat_interleave(n_rep, dim=1)
            # CAUSAL: this is a decoder-only LLM; the reference always
            # applies the causal mask and the DiT was trained on the
            # causal conditioning, so bidirectional attention here
            # silently destroys prompt adherence. During a cached decode
            # step (s == 1) the single query attends to the whole cache,
            # so causality is already satisfied by construction.
            out = F.scaled_dot_product_attention(xq, xk, xv,
                                                 is_causal=(s > 1))
        out = out.transpose(1, 2).reshape(b, s, self.n_heads * hd)
        return self.o_proj(out), kv


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

    def forward(self, x, cos, sin, past_kv=None):
        attn, kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + attn
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, kv


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
        # the final norm: unused by the 12-tap conditioning (raw layer
        # outputs), loaded + used by the reference-captioning path
        # (forward_caption / tied-lm_head logits)
        self.norm = Qwen3RMSNorm(hidden_size)

    def _rope(self, seq_len, device, dtype, position_ids=None):
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.layers[0].self_attn.head_dim, 2,
                             device=device, dtype=torch.float32)
                / self.layers[0].self_attn.head_dim)
        )
        if position_ids is None:
            pos = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(pos, inv_freq)
            # (1, s, 1, d) so it broadcasts over (b, s, heads, d)
            cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, :, None]
            sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, :, None]
            return cos.to(dtype), sin.to(dtype)
        # Qwen3-VL interleaved MRoPE: T-freqs by default, the H/W position
        # rows replace every 3rd dim in their section (the reference's
        # precompute_freqs_cis(interleaved_mrope=True) / HF's
        # recomposition_frequencies). position_ids: (3, seq)
        inv = inv_freq[None, None, :, None].expand(3, -1, -1, 1)
        pos = position_ids[:, None, None, :].float()  # (3, 1, 1, s)
        freqs = (inv @ pos).transpose(2, 3)  # (3, 1, s, head_dim/2)
        f = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = _MROPE_SECTION[axis] * 3
            idx = slice(offset, length, 3)
            f[..., idx] = freqs[axis][..., idx]
        emb = torch.cat((f, f), dim=-1)  # (1, s, head_dim)
        cos = emb.cos().squeeze(0)[None, :, None]
        sin = emb.sin().squeeze(0)[None, :, None]
        return cos.to(dtype), sin.to(dtype)

    def forward(self, input_ids):
        x = F.embedding(input_ids, self.embed_tokens.weight.to(COMPUTE_DTYPE))
        cos, sin = self._rope(x.shape[1], x.device, COMPUTE_DTYPE)
        taps: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x, _ = layer(x, cos, sin)
            # layer i (0-based) outputs HF hidden_states[i + 1]
            if i + 1 in KREA2_TAP_LAYERS:
                taps.append(x)
        return torch.stack(taps, dim=1)  # (b, 12, s, hidden)

    def forward_caption(self, input_ids, visual_embeds=None,
                        deepstack_embeds=None, position_ids=None,
                        kv_cache=None):
        """One step of the caption path: the image-pad token positions
        carry the vision-tower features instead of their word embedding,
        the DeepStack features are ADDED to those positions after
        decoder layers 0/1/2 (the HF ``_deepstack_process`` convention),
        and the logits come from the final norm + the TIED lm_head (the
        4B ties word embeddings -- ``logits = norm(h[-1]) @
        embed_tokens.T``, the reference's ``BaseQwen3.logits``).

        KV-cache decoding: call once with the full prompt stream
        (``visual_embeds``/``deepstack_embeds`` set, ``kv_cache=None``),
        then per generated token with ``input_ids`` = the single new
        token, ``visual_embeds=None`` and the cache returned by the
        previous call. Without a cache the loop would re-run the whole
        ~1300-token prefill for every generated token (~3 min for 128
        tokens on a 4060 Ti; with the cache it is one prefill + N tiny
        decodes).

        ``visual_embeds``: (n_pad, 2560) rows in stream order; the
        DeepStack list has one (n_pad, 2560) entry per injection layer.
        ``position_ids``: (3, seq) MRoPE ids (None = plain 1D positions).
        Returns ``(logits, cache)``: the (1, vocab) logits of the last
        position and the per-layer KV cache for the next step.
        """
        x = F.embedding(input_ids, self._embed_weight())
        if visual_embeds is not None:
            vis_mask = input_ids == IMAGE_PAD_ID
            x = x.clone()
            x[vis_mask] = visual_embeds.to(x.dtype)
        else:
            vis_mask = None
        cos, sin = self._rope(x.shape[1], x.device, COMPUTE_DTYPE,
                              position_ids)
        new_cache: list = []
        for i, layer in enumerate(self.layers):
            x, kv = layer(x, cos, sin,
                          None if kv_cache is None else kv_cache[i])
            new_cache.append(kv)
            if vis_mask is not None and deepstack_embeds \
                    and i < len(deepstack_embeds):
                ds = deepstack_embeds[i].to(x.dtype)
                x = x.clone()
                x[vis_mask] = x[vis_mask] + ds
        x = self.norm(x)
        h = x[:, -1].to(COMPUTE_DTYPE)
        return h @ self._embed_weight().T, new_cache

    def _embed_weight(self):
        cached = getattr(self, "_embed_cache", None)
        return cached if cached is not None \
            else self.embed_tokens.weight.to(COMPUTE_DTYPE)


@contextmanager
def decode_weight_cache(model, enabled=True):
    """Context manager: temporarily dequantize every Fp8Linear weight and
    the (tied) embedding table into ``COMPUTE_DTYPE`` copies, so the
    caption prefill/decode phase does not re-cast the full tower on
    every op (~250 casts per generated token, ~10 s for 128 tokens on a
    4060 Ti). Costs one extra full fp32 copy of the tower (~6.5 GB), so
    callers must pass ``enabled`` based on free VRAM (a Krea 2 backend
    keeps its 12 GB transformer resident -- the cache must not be built
    there). Nesting-safe use is not supported (one active cache per
    tower)."""
    cached = []
    try:
        if enabled:
            for m in model.modules():
                if isinstance(m, Fp8Linear):
                    m._decode_cache = m.weight.to(COMPUTE_DTYPE)
                    cached.append(m)
            model._embed_cache = model.embed_tokens.weight.to(COMPUTE_DTYPE)
        yield
    finally:
        if enabled:
            for m in cached:
                m._decode_cache = None
            model._embed_cache = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def load_krea2_encoder(path, device, load_norm=False):
    """Load ``qwen_3vl_4b.safetensors`` (plain fp8 projections, bf16
    embedding/norms) into a :class:`Krea2Qwen3VLModel` on ``device``.

    The ``model.visual.*`` keys (the vision tower -- a separate module,
    see :mod:`imagagen.encoders.qwen3vl_vision`) are always skipped.
    ``model.norm.weight`` (the final norm) is skipped by default: the
    12-tap conditioning uses raw layer outputs. ``load_norm=True``
    loads it for the reference-captioning path (the tied-lm_head logits
    run through it).
    """
    import safetensors.torch as st

    file = st.load_file(str(path))
    model = Krea2Qwen3VLModel()
    params = dict(model.named_parameters())
    used = set()
    for key, t in file.items():
        if key.startswith("model.visual."):
            continue  # the vision tower: a separate module
        k = key[len("model."):] if key.startswith("model.") else key
        if k == "norm.weight" and not load_norm:
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
