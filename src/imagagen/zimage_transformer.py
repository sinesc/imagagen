"""Z-Image-Turbo transformer (NextDiT), ported from the ComfyUI reference
(``dev/implementations/ComfyUI``: ``comfy/ldm/lumina/model.py`` +
``comfy/model_detection.py``).

Self-contained re-implementation of the ``NextDiT`` model with the
Z-Image-specific detection config (dim 3840):
- 2 context-refiner + 2 noise-refiner blocks, then 30 joint blocks;
- fused ``qkv`` GEMM (30 heads, no GQA), per-head q/k RMSNorm, RoPE
  (``EmbedND`` theta 256, axes [32, 48, 48]);
- ``z_image_modulation``: the block adaLN is a plain ``Linear(256 -> 4*dim)``
  (no SiLU) and the residual gates are ``tanh``-gated; the final layer keeps
  the SiLU + ``Linear(256 -> dim)`` adaLN and a scale-only modulate;
- text tokens (Qwen3-4B hidden states, dim 2560) are projected through
  ``cap_embedder`` (RMSNorm + Linear) and, together with the 2x2-patchified
  16-channel latents, padded with learned pad tokens to a multiple of 32;
- the timestep is the flow time flipped and scaled:
  ``t_embedder(sinusoidal((1 - sigma) * 1000))``.

The weights ship as ``F8_E4M3`` (no scale tensors) and are kept fp8 on the
GPU with a per-op bf16 cast, via :mod:`imagagen.gguf_ops`'
:class:`QuantLinear` (the Flux FP8 pattern).

``forward(x, sigma, context)`` takes the (1, 16, h, w) latent, the flow time
``sigma`` (1.0 = pure noise ... 0.0 = clean) and the (1, L, 2560) text
context, and returns ``-img`` -- the raw final-layer output, negated,
EXACTLY as the reference ``NextDiT._forward`` does. The backend converts it
to the k-diffusion x0 contract (see :mod:`imagagen.zimage_backend`).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# ---------------------------------------------------------------- rope ----


def rope(pos, dim, theta):
    if pos.device.type in ("mps", "xpu"):
        scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    else:
        scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)

    out = pos.unsqueeze(-1) * omega.unsqueeze(0)

    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    del cos_out, sin_out

    b, n, d, _ = out.shape
    out = out.view(b, n, d, 2, 2)

    return out.float()


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    del xq_, xk_
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        del ids, n_axes
        return emb.unsqueeze(1)


def timestep_embedding(t, dim, max_period=10000):
    """Plain sinusoidal embedding (ComfyUI ``diffusionmodules.util`` version:
    no ``time_factor`` -- the Z-Image ``time_scale`` plays that role upstream)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    del freqs
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    del args
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


# --------------------------------------------------------------- norms ----


class RMSNorm(nn.Module):
    """Affine RMSNorm with an explicit ``eps`` (``F.rms_norm`` kernel)."""

    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.weight.dtype != x.dtype:
            self.weight = nn.Parameter(self.weight.to(dtype=x.dtype))
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)


# --------------------------------------------------------------- model ----


class TimestepEmbedder(nn.Module):
    """``Linear(256 -> hidden) SiLU Linear(hidden -> 256)`` over the plain
    sinusoidal embedding (state-dict keys ``t_embedder.mlp.0`` / ``.2``)."""

    def __init__(self, hidden_size, output_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size, bias=True),
        )

    def forward(self, t):
        return self.mlp(timestep_embedding(t, 256))


class FeedForward(nn.Module):
    """SiLU-gated MLP (``w2(silu(w1(x)) * w3(x))``, no biases)."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class JointAttention(nn.Module):
    """Fused-qkv multi-head attention with per-head q/k RMSNorm + RoPE.

    State-dict keys: ``attention.qkv`` / ``attention.out`` /
    ``attention.q_norm`` / ``attention.k_norm``.
    """

    def __init__(self, dim, n_heads, n_kv_heads, norm_eps):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, (n_heads + 2 * n_kv_heads) * self.head_dim,
                             bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, norm_eps)
        self.k_norm = RMSNorm(self.head_dim, norm_eps)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.qkv(x).split(
            [self.n_heads * self.head_dim,
             self.n_kv_heads * self.head_dim,
             self.n_kv_heads * self.head_dim],
            dim=-1,
        )
        hd = self.head_dim
        xq = self.q_norm(xq.view(bsz, seqlen, self.n_heads, hd))
        xk = self.k_norm(xk.view(bsz, seqlen, self.n_kv_heads, hd))
        xv = xv.view(bsz, seqlen, self.n_kv_heads, hd)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        xq, xk = apply_rope(xq, xk, freqs_cis)

        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(xq, xk, xv)
        out = out.transpose(1, 2).reshape(bsz, seqlen, self.n_heads * hd)
        return self.out(out)


class JointTransformerBlock(nn.Module):
    """One NextDiT block.

    ``modulation=True`` (noise refiner + main blocks): the adaLN
    ``Linear(256 -> 4*dim)`` (no SiLU, z_image_modulation) emits
    scale/gate pairs for the attention and the FFN, and both residual
    branches are ``tanh``-gated. ``modulation=False`` (context refiner):
    plain pre-norm residuals, no adaLN weights.
    """

    def __init__(self, dim, n_heads, n_kv_heads, ffn_hidden, norm_eps,
                 modulation):
        super().__init__()
        self.modulation = modulation
        self.attention = JointAttention(dim, n_heads, n_kv_heads, 1e-6)
        self.feed_forward = FeedForward(dim, ffn_hidden)
        self.attention_norm1 = RMSNorm(dim, norm_eps)
        self.ffn_norm1 = RMSNorm(dim, norm_eps)
        self.attention_norm2 = RMSNorm(dim, norm_eps)
        self.ffn_norm2 = RMSNorm(dim, norm_eps)
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, 256), 4 * dim, bias=True),
            )

    def forward(self, x, freqs_cis, adaln_input=None):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            attn = self.attention(
                self.attention_norm1(x) * (1 + scale_msa.unsqueeze(1)),
                freqs_cis,
            )
            x = x + gate_msa.tanh().unsqueeze(1) * self.attention_norm2(attn)
            ff = self.feed_forward(
                self.ffn_norm1(x) * (1 + scale_mlp.unsqueeze(1)),
            )
            x = x + gate_mlp.tanh().unsqueeze(1) * self.ffn_norm2(ff)
        else:
            attn = self.attention(self.attention_norm1(x), freqs_cis)
            x = x + self.attention_norm2(attn)
            ff = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ff)
        return x


class FinalLayer(nn.Module):
    """LayerNorm(eps 1e-6, no affine) + scale-only modulate from
    ``SiLU + Linear(256 -> dim)`` + the output projection (``-> 64``)."""

    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels,
                                bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 256), dim, bias=True),
        )

    def forward(self, x, t_emb):
        scale = self.adaLN_modulation(t_emb)
        x = x * (1 + scale.unsqueeze(1))
        return self.linear(x)


class ZImageTransformer(nn.Module):
    """The Z-Image-Turbo transformer (NextDiT, dim 3840 config).

    Constructor arguments follow the reference architecture; the defaults
    ARE the shipped checkpoint's detection config, so
    ``ZImageTransformer()`` matches ``intorealism_zitV40.safetensors``.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        ffn_hidden: int = 10240,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 2560,
        axes_dims: list[int] = (32, 48, 48),
        rope_theta: float = 256.0,
        pad_tokens_multiple: int = 32,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim
        self.pad_tokens_multiple = pad_tokens_multiple
        self.time_scale = time_scale

        if dim % n_heads != 0:
            raise ValueError(
                f"dim {dim} must be divisible by n_heads {n_heads}")
        pe_dim = dim // n_heads
        if sum(axes_dims) != pe_dim:
            raise ValueError(f"axes_dims {axes_dims} != head dim {pe_dim}")

        self.x_embedder = nn.Linear(
            patch_size * patch_size * in_channels, dim, bias=True)
        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(dim, n_heads, n_kv_heads, ffn_hidden,
                                      norm_eps, modulation=True)
                for _ in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(dim, n_heads, n_kv_heads, ffn_hidden,
                                      norm_eps, modulation=False)
                for _ in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, 1024), output_size=256)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )
        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(dim, n_heads, n_kv_heads, ffn_hidden,
                                      norm_eps, modulation=True)
                for _ in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)
        self.x_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.rope_embedder = EmbedND(dim=pe_dim, theta=rope_theta,
                                     axes_dim=axes_dims)

    # ------------------------------------------------------------ embed ----

    def _pad_to_multiple(self, feats, pad_token, multiple):
        """Append learned pad tokens until the sequence is a multiple of
        ``multiple`` (reference ``pad_zimage``)."""
        pad_extra = (-feats.shape[1]) % multiple
        if pad_extra == 0:
            return feats, 0
        pad = pad_token.to(device=feats.device, dtype=feats.dtype) \
            .unsqueeze(0).repeat(feats.shape[0], pad_extra, 1)
        return torch.cat([feats, pad], dim=1), pad_extra

    def forward(self, x, sigma, context):
        """``x``: (1, 16, h, w); ``sigma``: (1,) flow time (1 = noise);
        ``context``: (1, L, 2560) text hidden states. Returns the raw
        (negated) model output of shape (1, 16, h, w)."""
        bs, c, h, w = x.shape
        pH = pW = self.patch_size
        # latent dims are multiples of 2 (the 16-px snap); pad defensively
        pad_h = (pH - h % pH) % pH
        pad_w = (pW - w % pW) % pW
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        h_p, w_p = x.shape[-2], x.shape[-1]

        # timestep: flip the flow time and scale (reference: t = 1 - timesteps,
        # t_embedder(t * time_scale))
        t = 1.0 - sigma
        t_emb = self.t_embedder((t * self.time_scale).to(x.dtype))

        # --- text stream: embed, pad to the multiple, position ids 1..n ---
        cap = self.cap_embedder(context)
        cap, _ = self._pad_to_multiple(cap, self.cap_pad_token,
                                       self.pad_tokens_multiple)
        ncap = cap.shape[1]
        cap_pos = torch.zeros(bs, ncap, 3, device=x.device, dtype=torch.float32)
        cap_pos[:, :, 0] = torch.arange(ncap, device=x.device,
                                        dtype=torch.float32) + 1.0
        cap_freqs = self.rope_embedder(cap_pos).movedim(1, 2)

        # --- image stream: patchify, embed, pad, position ids ---
        hp, wp = h_p // pH, w_p // pW
        x_img = x.view(bs, c, hp, pH, wp, pW).permute(0, 2, 4, 3, 5, 1) \
            .flatten(3).flatten(1, 2)
        x_img = self.x_embedder(x_img)
        nimg = x_img.shape[1]
        x_img, _ = self._pad_to_multiple(x_img, self.x_pad_token,
                                         self.pad_tokens_multiple)
        nimg_p = x_img.shape[1]
        x_pos = torch.zeros(bs, nimg_p, 3, device=x.device,
                            dtype=torch.float32)
        x_pos[:, :, 0] = ncap + 1
        x_pos[:, :nimg, 1] = (
            torch.arange(hp, device=x.device, dtype=torch.float32).view(-1, 1)
            .repeat(1, wp).flatten()
        )
        x_pos[:, :nimg, 2] = (
            torch.arange(wp, device=x.device, dtype=torch.float32).view(1, -1)
            .repeat(hp, 1).flatten()
        )
        x_freqs = self.rope_embedder(x_pos).movedim(1, 2)

        # --- refiners (separate streams, as the reference does) ---
        cap_feats = cap
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs)
        x_feats = x_img
        for layer in self.noise_refiner:
            x_feats = layer(x_feats, x_freqs, t_emb)

        # --- joint blocks over [cap, x] ---
        feats = torch.cat([cap_feats, x_feats], dim=1)
        freqs = torch.cat([cap_freqs, x_freqs], dim=1)
        del cap_feats, x_feats, cap_freqs, x_freqs
        for layer in self.layers:
            feats = layer(feats, freqs, t_emb)

        out = self.final_layer(feats, t_emb)
        del feats, freqs, t_emb
        # unpatchify the image tail (drop the text tokens and any padding)
        out = out[:, ncap:, :]
        out = (
            out.view(bs, hp, wp, pH, pW, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(bs, self.out_channels, h_p, w_p)
        )[:, :, :h, :w]
        # reference NextDiT._forward negates the raw output
        return -out
