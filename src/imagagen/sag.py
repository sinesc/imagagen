"""Self attention guidance (SAG).

Port of Forge's builtin extension ``sd_forge_sag`` (itself A1111's
"Self-Attention Guidance"). SAG sharpens the high-detail regions of a
generation (faces, eyes, fine structures) by adversarially degrading the
regions the model's own self-attention focuses on and feeding the
degradation back through the network:

per sampling step (requires CFG > 1, i.e. an unconditional pass):
  1. the unconditional UNet forward's mid-block self-attention query/key
     are captured (forward hooks), so the attention distribution can be
     recomputed afterwards without perturbing the UNet's own numerics;
  2. the attention's per-key column sums (heads-averaged) form a mask of
     "high attention" pixels; the unconditional x0 is Gaussian-blurred
     exactly there, yielding the adversarially "degraded" x0;
  3. the degraded x0 is re-noised back to the current sigma, denoised
     with one extra *unconditional* UNet pass, and the difference
     ``(degraded - re-denoised) * scale`` is added to the CFG result.

The extra pass costs ~50% more UNet time than a plain cfg > 1 generation.

Option names, ranges, defaults, and the ``sag_*`` metadata keys match the
Forge extension exactly (it publishes them through
``extra_generation_params`` under those raw snake_case names).

Porting notes
--------------
The attention lives in the UNet's mid-block, whose grid is 1/4 of the
latent (16x16 = 256 tokens at 512^2, 32x32 = 1024 at 1024^2). The
reference stores the full ``(heads, n, n)`` matrix (a few MB at 1024^2)
and reads it back in a second stage. Here the per-key column sums are
accumulated directly while recomputing the softmax, chunked over the
query dim, so peak extra VRAM stays bounded (a ~128 MB transient at
1024^2; chunking only matters at 4096^2+ latents) and the mask is
ready by the time the step's model call has finished.

The reference computes the attention with its own ``attention_function``
and *replaces* the mid-block attention (so the base numerics change
slightly when SAG is on). Here the mask is recomputed in fp32 from q/k
captured by forward hooks on the *same* mid-block layer the reference
replaces, leaving the UNet's actual attention -- and thus the base
generation -- bit-identical to a run without SAG (a SAG run with
``scale == 0`` is bit-identical to a plain run). Pixels right on the
threshold boundary may differ from the reference's mask by a negligible
epsilon -- see AGENTS.md on reference-image matching.
"""

from __future__ import annotations

import math

import torch
from pydantic import BaseModel, Field


class SelfAttentionGuidance(BaseModel):
    """SAG options; same names, ranges and defaults as the Forge sliders.

    ``enabled`` mirrors the model configuration's per-direction sag
    checkbox (resolved server-side from the selected entry). The request
    carries the object (not ``null``) so the PNG metadata carries
    ``sag_enabled`` in both states -- like the Forge extension, which
    publishes all four parameters unconditionally. The backend treats a
    disabled object as absent.
    """

    enabled: bool = True
    scale: float = Field(0.5, ge=-2.0, le=5.0)
    blur_sigma: float = Field(2.0, ge=0.0, le=10.0)
    threshold: float = Field(1.0, ge=0.0, le=4.0)


# fp32 (chunk, n) sim elements per chunk of compute_sag_mass: the chunked
# accumulation keeps the transient (chunk, n) softmax in ~32 MB fp32. The
# full (heads, n, n) matrix would be ~4 MB at 512^2 / 16 MB at 1024^2
# (mid-block n = 256 / 1024) and 256 MB at a 4096^2 latent -- the chunk
# keeps the cost bounded at arbitrary resolutions
_SAG_SIM_CHUNK_ELEMENTS = 32 * 2**20


class SagRecorder:
    """Per-generation state shared between the attention hooks and the
    sampling loop. ``record`` is set True only around the unconditional
    UNet forward; the hooks capture q/k of the mid-block self-attention
    only while it is set, so the conditional pass and the extra SAG pass
    leave the capture untouched."""

    def __init__(self) -> None:
        self.record = False
        self.q: torch.Tensor | None = None
        self.k: torch.Tensor | None = None
        self.heads: int | None = None


def install_sag_hooks(unet, rec: SagRecorder) -> list:
    """Capture q/k of the unet's mid-block self-attention (the layer the
    Forge extension hooks: ``get_block("mid")``'s ``attentions[0]`` -- in
    diffusers that entry is a ``Transformer2DModel`` wrapper, the actual
    ``Attention`` is ``.transformer_blocks[0].attn1``).
    Returns the hook handles; remove them when generation finishes."""
    mid = unet.mid_block.attentions[0]
    attn = getattr(mid, "attn", None) or mid.transformer_blocks[0].attn1
    rec.heads = attn.heads
    handles = []
    for which in ("q", "k"):
        def hook(module, inputs, output, rec=rec, which=which):
            if rec.record:
                setattr(rec, which, output.detach())
        handles.append(getattr(attn, f"to_{which}").register_forward_hook(hook))
    return handles


def compute_sag_mass(q: torch.Tensor, k: torch.Tensor, heads: int) -> torch.Tensor:
    """Per-key attention mass of the mid-block self-attention (fp32).

    ``mass[i]`` is the heads-averaged, query-summed mass of key i:
    ``(1 / heads) * sum_h sum_q softmax_h(q_h . k_h^T * scale)[q, i]`` --
    the numerator of Forge's ``mask = attn.mean(1).sum(1) > threshold``
    (``attn`` reshaped to ``(b, heads, n, n)``).

    Recomputed from the captured q/k, chunked over the query dim so the
    (chunk, n) softmax blocks stay in a few tens of MB (the reference
    keeps the full (heads, n, n) matrix, ~6 GB at 1024x1024).

    :param q, k: (1, n, dim) query/key outputs of the layer's to_q/to_k
        (fp16, as produced by the fp16 UNet).
    :param heads: number of attention heads (``attn.heads``).
    :returns: (n,) float32 mass over the keys (n == h * w for the
        mid-block grid, which is 1/4 of the latent per side).
    """
    n, dim = q.shape[1], q.shape[2]
    dim_head = dim // heads
    scale = dim_head ** -0.5
    # token-major head grouping (Forge's reshape(b, -1, heads, dim_head))
    qf = q[0].float().view(n, heads, dim_head).transpose(0, 1)
    kf = k[0].float().view(n, heads, dim_head).transpose(0, 1)
    colsum = torch.zeros(n, device=q.device, dtype=torch.float32)
    # one ~32M-element fp32 (chunk, n) sim block per chunk
    chunk = max(1, _SAG_SIM_CHUNK_ELEMENTS // n)
    for h in range(heads):
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            sim = (qf[h, i0:i1] @ kf[h].t()) * scale
            colsum += sim.softmax(dim=-1).sum(dim=0)
    return colsum / heads


def compute_sag_mask(
    q: torch.Tensor, k: torch.Tensor, heads: int, threshold: float,
) -> torch.Tensor:
    """Forge's ``mask = attn.mean(1).sum(1) > threshold`` (see
    :func:`compute_sag_mass` for the mass definition).

    :returns: (n,) bool mask over the keys.
    """
    return compute_sag_mass(q, k, heads) > threshold


def create_blur_map(
    x0: torch.Tensor, mask: torch.Tensor, blur_sigma: float,
) -> torch.Tensor:
    """Forge's ``create_blur_map`` with the per-key mask precomputed.

    The mask lives at the *mid-block* resolution (``n = h * w`` tokens,
    where the mid-block sits at a fixed downscale of the latent), while
    ``x0`` is at full latent resolution. As in the reference, the mask is
    reshaped to its mid-block grid and bilinearly **upsampled** to the
    latent grid, then used to blend the blurred copy with x0.

    :param x0: (1, 4, lh, lw) x0 estimate (fp32).
    :param mask: (n,) bool/float over the mid-block keys, n == h * w.
    :param blur_sigma: the SAG ``blur_sigma`` option (0 = the "degraded"
        copy is x0 itself, i.e. no blur).
    :returns: the adversarially blurred x0: the Gaussian blur where the
        (upsampled) mask is set, x0 elsewhere.
    """
    if blur_sigma <= 0:
        # Forge's documented "0 leaves the image unchanged": at sigma 0
        # its Gaussian kernel divides by zero and turns NaN, so the
        # no-blur path is simply the identity (the whole SAG correction
        # is then zero: the degraded copy re-denoises back to itself)
        return x0
    b, _, lh, lw = x0.shape
    n = mask.numel() // b
    # Forge's ratio trick: recover the mid-block downscale factor s from
    # the latent size (lh, lw) and the mid-block token count n
    # (n == (lh/s) * (lw/s)  =>  s == sqrt(lh * lw / n)); s is a power of
    # two, and 2 ** (s - 1).bit_length() == s
    s = 2 ** (math.ceil(math.sqrt(lh * lw / n)) - 1).bit_length()
    h = math.ceil(lh / s)
    w = math.ceil(lw / s)
    if h * w != n:
        # non-power-of-two latent (shouldn't happen for SDXL); fall back
        # to the nearest integral grid that holds n tokens
        s = max(1, round(math.sqrt(lh * lw / n)))
        h, w = lh // s, lw // s
    m = (
        mask.reshape(b, h, w)
        .unsqueeze(1)
        .to(x0.dtype)
    )
    m = torch.nn.functional.interpolate(m, (lh, lw))
    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=blur_sigma)
    return blurred * m + x0 * (1.0 - m)


def gaussian_blur_2d(img: torch.Tensor, kernel_size: int, sigma: float):
    """Forge's ``gaussian_blur_2d`` (``sd_forge_sag``), verbatim except
    for type hints."""
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()

    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2] * 4

    img = torch.nn.functional.pad(img, padding, mode="reflect")
    img = torch.nn.functional.conv2d(img, kernel2d, groups=img.shape[-3])
    return img
