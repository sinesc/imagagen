"""Flux LoRA key mapping and checkpoint parsing.

Standard (unfused) Flux LoRAs (ComfyUI/SimpleTrainer, Kohya, A1111) store
A/B matrices keyed by BFL's *unfused* module names
(``transformer.transformer_blocks.0.attn.to_q`` and friends), while our
transformer (:mod:`imagagen.flux_transformer`) fuses q/k/v into single
``qkv``/``linear1`` GEMMs. This module maps each standard key to our fused
module plus the slice of its output rows the LoRA covers -- the port of
Forge's ``comfyui_lora_collection.utils.flux_to_diffusers`` (reference
checkout), adapted to our module paths (no ``diffusion_model.`` prefix;
keys relative to the transformer root, e.g. ``double_blocks.0.img_attn.qkv``).

A/B conventions accepted: A1111/Kohya ``.lora_A.weight`` (down projection,
``(rank, in_features)``) + ``.lora_B.weight`` (up projection,
``(out, rank)``), and diffusers-style ``.lora_down.weight`` /
``.lora_up.weight`` (same shapes, A == down). The per-key
``<base>.alpha`` scalar defaults to the rank (scale 1.0) when absent.
``.lora_*.weight`` keys get their trailing ``.0`` stripped
(``attn.to_out.0`` == ``attn.to_out``) before mapping.

Keys that are neither a mappable A/B pair nor an alpha scalar (QK-norm
1-D scales, text-encoder keys, A1111 ``lora_unet_*`` naming, ...) are
collected and raised in one :class:`ValueError` -- the Flux backend's
all-or-nothing apply policy (a failing set must leave no partial state).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

#: transformer hidden size (H in the plan's row arithmetic)
H = 3072
#: single-block MLP width (mlp_ratio 4.0)
M = 12288


@dataclass(frozen=True)
class FluxLoraSpec:
    """One LoRA key, mapped to a fused module.

    ``module``: path relative to the transformer root; the delta is
    applied to its output rows ``[r0:r1)``. ``A`` is ``(rank,
    in_features)``, ``B`` is ``(r1 - r0, rank)``; ``scale`` is
    ``alpha / rank`` (1.0 when the file carries no ``<base>.alpha``).
    ``swap`` swaps B's two row halves on apply (``norm_out.linear``:
    the standard layout is ``[scale, shift]``, ours is ``[shift, scale]``
    -- Forge's ``swap_scale_shift``).

    Krea 2 extensions (see ``backends/krea2/lora.py``; unused by the
    Flux/Z-Image parsers):

    ``kron``: a LoKr factorized delta ``(w1, w2)`` -- the weight delta
    is ``scale * kron(w1, w2)`` (``A`` carries ``w1``, ``B`` is unused);
    the backend applies it as the nested grouped GEMM of ComfyUI's LoKr
    bypass, never materializing the full Kronecker product.

    ``dora_scale``: a DoRA weight-decomposition scale vector (length
    ``in_features``, the INPUT axis -- the Krea 2 releases ship the
    input-axis variant). The effective delta is ``c * (W + d) - W``
    with ``d`` the scaled A/B delta, ``W`` the module's base weight and
    ``c = dora_scale / ||W + d||`` per input column -- the backend
    folds ``c`` into the delta's A columns at apply time (it needs the
    base weight, so it cannot be pre-folded into ``A``). LoKr + DoRA
    on one module is rejected (the column scaling does not factor
    through the Kronecker application).
    """

    module: str
    r0: int
    r1: int
    swap: bool
    A: torch.Tensor
    B: torch.Tensor
    scale: float
    kron: tuple[torch.Tensor, torch.Tensor] | None = None
    dora_scale: torch.Tensor | None = None


_LORA_KEY = re.compile(r"^transformer\.(.+)\.lora_(A|B|down|up)\.weight$")
_ALPHA_KEY = re.compile(r"^transformer\.(.+)\.alpha$")


def _strip_dot0(base: str) -> str:
    return base[:-2] if base.endswith(".0") else base


def lora_key(key: str) -> tuple[str, str] | None:
    """Full tensor name -> ``(base key, 'A'/'B')``, or None.

    Accepts A1111/Kohya ``.lora_A.weight`` / ``.lora_B.weight`` and
    diffusers-style ``.lora_down.weight`` / ``.lora_up.weight``
    (A == the down projection). A trailing ``.0`` on the base is
    stripped (``attn.to_out.0`` == ``attn.to_out``). None for keys that
    are not part of an A/B pair (alphas, ...).
    """
    m = _LORA_KEY.match(key)
    if m is None:
        return None
    return _strip_dot0(m.group(1)), \
        "A" if m.group(2) in ("A", "down") else "B"


def alpha_key(key: str) -> str | None:
    """Full tensor name -> its base key, or None (not an alpha scalar)."""
    m = _ALPHA_KEY.match(key)
    return _strip_dot0(m.group(1)) if m is not None else None


def _map_double(n: int, rest: str) -> tuple[str, int, int, bool]:
    P = f"double_blocks.{n}"
    table = {
        # fused qkv: q/k/v own the first 3H output rows
        "attn.to_q": (f"{P}.img_attn.qkv", 0, H),
        "attn.to_k": (f"{P}.img_attn.qkv", H, 2 * H),
        "attn.to_v": (f"{P}.img_attn.qkv", 2 * H, 3 * H),
        "attn.add_q_proj": (f"{P}.txt_attn.qkv", 0, H),
        "attn.add_k_proj": (f"{P}.txt_attn.qkv", H, 2 * H),
        "attn.add_v_proj": (f"{P}.txt_attn.qkv", 2 * H, 3 * H),
        # pre_only attention: `to_out` is the full-width output projection
        "attn.to_out": (f"{P}.img_attn.proj", 0, H),
        "attn.to_add_out": (f"{P}.txt_attn.proj", 0, H),
        # BFL nn.Linear modules named `...net.0` / `...net.0.proj`
        "ff.net.0": (f"{P}.img_mlp.0", 0, M),
        "ff.net.0.proj": (f"{P}.img_mlp.0", 0, M),
        "ff.net.2": (f"{P}.img_mlp.2", 0, H),
        "ff_context.net.0": (f"{P}.txt_mlp.0", 0, M),
        "ff_context.net.0.proj": (f"{P}.txt_mlp.0", 0, M),
        "ff_context.net.2": (f"{P}.txt_mlp.2", 0, H),
        # double-block modulation: lin maps H -> 6H
        "norm1.linear": (f"{P}.img_mod.lin", 0, 6 * H),
        "norm1_context.linear": (f"{P}.txt_mod.lin", 0, 6 * H),
    }
    if rest not in table:
        raise ValueError(f"unknown Flux LoRA key: transformer_blocks.{n}.{rest!r}")
    module, r0, r1 = table[rest]
    return module, r0, r1, False


def _map_single(n: int, rest: str) -> tuple[str, int, int, bool]:
    P = f"single_blocks.{n}"
    table = {
        # fused linear1: q/k/v own [0:3H), the mlp projection [3H:3H+M)
        "attn.to_q": (f"{P}.linear1", 0, H),
        "attn.to_k": (f"{P}.linear1", H, 2 * H),
        "attn.to_v": (f"{P}.linear1", 2 * H, 3 * H),
        "attn.proj_mlp": (f"{P}.linear1", 3 * H, 3 * H + M),
        "proj_mlp": (f"{P}.linear1", 3 * H, 3 * H + M),
        # fused linear2: H+M -> H (input = cat[attn, mlp])
        "proj_out": (f"{P}.linear2", 0, H),
        # single-block modulation: lin maps H -> 3H
        "norm.linear": (f"{P}.modulation.lin", 0, 3 * H),
    }
    if rest not in table:
        raise ValueError(
            f"unknown Flux LoRA key: single_transformer_blocks.{n}.{rest!r}"
        )
    module, r0, r1 = table[rest]
    return module, r0, r1, False


# global (block-less) keys: (module, r0, r1, swap)
_GLOBAL = {
    "x_embedder": ("img_in", 0, H, False),
    "context_embedder": ("txt_in", 0, H, False),
    "time_embedder.linear_1": ("time_in.in_layer", 0, H, False),
    "time_embedder.linear_2": ("time_in.out_layer", 0, H, False),
    "vector_embedder.linear_1": ("vector_in.in_layer", 0, H, False),
    "vector_embedder.linear_2": ("vector_in.out_layer", 0, H, False),
    "guidance_embedder.linear_1": ("guidance_in.in_layer", 0, H, False),
    "guidance_embedder.linear_2": ("guidance_in.out_layer", 0, H, False),
    # last layer's adaLN: diffusers order [scale, shift], ours [shift,
    # scale] -> swap B's row halves (Forge: swap_scale_shift)
    "norm_out.linear": ("final_layer.adaLN_modulation.1", 0, 2 * H, True),
    # BFL block-level naming: the final output projection (H -> 16)
    "proj_out": ("final_layer.linear", 0, 16, False),
}


def map_base_key(base: str) -> tuple[str, int, int, bool]:
    """Standard Flux LoRA base key -> ``(module, r0, r1, swap)``.

    ``base`` is the LoRA key minus the ``transformer.`` prefix and the
    ``.lora_*.weight`` suffix (trailing ``.0`` already stripped), e.g.
    ``transformer_blocks.0.attn.to_q``. ``module`` is relative to the
    transformer root; the delta targets its output rows ``[r0:r1)``.

    Raises :class:`ValueError` for keys with no weight-matrix target
    (QK-norm scales, text-encoder keys, unknown names, ...).
    """
    m = re.match(r"^(?:transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)$",
                 base)
    if m:
        if base.startswith("single_"):
            return _map_single(int(m.group(1)), m.group(2))
        return _map_double(int(m.group(1)), m.group(2))
    if base in _GLOBAL:
        return _GLOBAL[base]
    raise ValueError(f"unknown Flux LoRA key: {base!r}")


def sniff_lora_format(path: str | Path,
                      *, skip: tuple[str, ...] = ()) -> str | None:
    """Name an unsupported LoRA family from the file's key names, or
    ``None`` when nothing recognizable fires.

    Every LoRA loader in imagagen (SDXL, Flux, Krea 2) runs this on the
    file BEFORE parsing, so a LoKr / LoHa / DoRA / GaLore / GGUF file
    fails with a named diagnosis ("LoKr: needs Kronecker product
    support") instead of a cryptic key-mismatch stack from the library
    loader. Reads only the safetensors header (cheap). ``skip`` lists
    kinds a caller supports (e.g. ``("lokr",)`` on Krea 2).
    """
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            if fh.read(4) == b"GGUF":
                return ("GGUF container: quantized LoRAs need dequantization "
                        "support (not implemented)")
    except OSError:
        return None
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            names = list(f.keys())
    except Exception:
        return None
    kinds = []
    if ("lokr" not in skip
            and any("lokr_w1" in k or "lokr_w2" in k for k in names)):
        kinds.append("LoKr (Kronecker-product LoRA, needs kron factor "
                     "support)")
    if ("loha" not in skip
            and any("loha_w1" in k or "loha_w2" in k for k in names)):
        kinds.append("LoHa (Hadamard-product LoRA, not implemented)")
    if "galore" not in skip and any("galore" in k for k in names):
        kinds.append("GaLore (projection-matrix LoRA, not implemented)")
    if ("dora" not in skip and any(".dora_scale" in k for k in names)):
        kinds.append("DoRA (direction-aware scaling, needs dora support)")
    return ", ".join(kinds) if kinds else None


def parse_lora(path: str | Path) -> list[FluxLoraSpec]:
    """A Flux LoRA ``.safetensors`` file -> its delta specs (CPU tensors).

    Raises :class:`ValueError` listing every key that is neither a
    mappable A/B pair nor an alpha scalar, and for incomplete pairs.
    Unsupported families (LoKr/LoHa/DoRA/GaLore/GGUF) are named first.
    """
    from safetensors.torch import load_file

    if why := sniff_lora_format(path):
        raise ValueError(
            f"unsupported Flux LoRA format in {Path(path).name}: {why}")
    data = load_file(str(path))
    parts: dict[str, dict[str, torch.Tensor | float]] = {}
    unknown: list[str] = []
    for key, t in data.items():
        if key == "__metadata__":
            continue
        pair = lora_key(key)
        if pair is not None:
            base, role = pair
            parts.setdefault(base, {})[role] = t
            continue
        base = alpha_key(key)
        if base is not None:
            parts.setdefault(base, {})["alpha"] = float(t)
            continue
        unknown.append(key)
    if unknown:
        raise ValueError(
            f"unknown Flux LoRA keys in {Path(path).name}: {sorted(set(unknown))}"
        )
    specs = []
    for base, p in parts.items():
        if "A" not in p or "B" not in p:
            raise ValueError(f"incomplete LoRA A/B pair for {base!r}")
        A, B = p["A"], p["B"]
        module, r0, r1, swap = map_base_key(base)
        rank = A.shape[0]
        alpha = p.get("alpha", float(rank))
        specs.append(FluxLoraSpec(module, r0, r1, swap, A, B, alpha / rank))
    return specs


def merge_module_deltas(
    per_module: dict[str, dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]]]
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor, int, int]]]:
    """Merge a module's deltas so it costs ONE extra GEMM pair.

    ``per_module``: module path -> ``(r0, r1)`` slice -> list of
    ``(A, B)`` pairs (A (rank, in), B (r1 - r0, rank); the LoRA weight is
    already folded into A). Deltas on the same slice concatenate exactly
    (A along the rank dim, B along its rank dim); adjacent slices merge
    into one delta, A stacking along the rank dim and B zero-padding the
    other's rank block (row block i only multiplies with its own rank
    rows, so the padded product is exact). Returns module path ->
    ``[(A, B, r0, r1), ...]`` for :meth:`gguf_ops.GgufLinear.set_lora_deltas`.
    """
    merged: dict[str, list[tuple[torch.Tensor, torch.Tensor, int, int]]] = {}
    for module, slices in per_module.items():
        parts = []
        for (r0, r1), pairs in slices.items():
            # every LoRA in the set targeting this (module, slice) is
            # concatenated: A along the rank dim, B along its rank dim
            parts.append((r0, r1,
                          torch.cat([p[0] for p in pairs], dim=0),
                          torch.cat([p[1] for p in pairs], dim=1)))
        parts.sort(key=lambda part: part[0])
        out: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for r0, r1, a, b in parts:
            if out and out[-1][1] == r0:
                pr0, pr1, pa, pb = out.pop()
                ra = pa.shape[0]
                len1 = pr1 - pr0
                bm = torch.zeros(len1 + (r1 - r0), ra + a.shape[0],
                                 dtype=b.dtype)
                bm[:len1, :ra] = pb
                bm[len1:, ra:] = b
                out.append((pr0, r1, torch.cat([pa, a], dim=0), bm))
            else:
                out.append((r0, r1, a, b))
        merged[module] = [(a, b, r0, r1) for r0, r1, a, b in out]
    return merged
