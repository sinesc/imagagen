"""Z-Image LoRA key mapping and checkpoint parsing.

The shipped LoRAs use the official Z-Image-Turbo naming
(``diffusion_model.layers.N.attention.to_q.lora_A.weight`` and friends;
rank 32; bf16; no alpha tensors -> scale 1.0), while our transformer
(:mod:`imagagen.zimage_transformer`) fuses q/k/v into one ``qkv`` GEMM.
Each key is mapped to our fused module plus the slice of its output rows
the LoRA covers (the Flux LoRA pattern: ``to_q``/``to_k``/``to_v`` own the
three thirds of ``qkv``). Deltas are applied as runtime additive merges
(see :mod:`imagagen.zimage_backend` :meth:`_apply_loras`) using
:func:`flux_lora.merge_module_deltas` and ``QuantLinear.set_lora_deltas``.

Only the 30 main blocks are LoRA-able in the shipped format (attention
q/k/v/out, the SiLU-gated FFN w1/w2/w3, and the block adaLN modulation);
refiner layers and the embedders are not part of that format and are
rejected with a :class:`ValueError` (all-or-nothing apply).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from .flux_lora import FluxLoraSpec

#: transformer hidden size (qkv output rows per q/k/v)
DIM = 3840
#: SiLU-gated FFN width (ffn_dim_multiplier 8/3)
FFN = 10240

_LORA_KEY = re.compile(r"^(.+)\.lora_(A|B|down|up)\.weight$")
_ALPHA_KEY = re.compile(r"^(.+)\.alpha$")


def _strip_dot0(base: str) -> str:
    return base[:-2] if base.endswith(".0") else base


def map_base_key(base: str) -> tuple[str, int, int]:
    """Mapped base key (``diffusion_model.`` stripped, ``.lora_*.weight``
    removed, trailing ``.0`` stripped) -> ``(module, r0, r1)``.

    ``module`` is relative to the transformer root; the delta targets the
    fused weight's output rows ``[r0:r1)``. Raises :class:`ValueError` for
    keys with no weight-matrix target.
    """
    m = re.match(r"^layers\.(\d+)\.(.+)$", base)
    if m is None:
        raise ValueError(f"unknown Z-Image LoRA key: {base!r}")
    n, rest = int(m.group(1)), m.group(2)
    P = f"layers.{n}"
    table = {
        # fused qkv: q/k/v own the three output-thirds
        "attention.to_q": (f"{P}.attention.qkv", 0, DIM),
        "attention.to_k": (f"{P}.attention.qkv", DIM, 2 * DIM),
        "attention.to_v": (f"{P}.attention.qkv", 2 * DIM, 3 * DIM),
        "attention.to_out": (f"{P}.attention.out", 0, DIM),
        "feed_forward.w1": (f"{P}.feed_forward.w1", 0, FFN),
        "feed_forward.w2": (f"{P}.feed_forward.w2", 0, DIM),
        "feed_forward.w3": (f"{P}.feed_forward.w3", 0, FFN),
        # block adaLN: Linear(256 -> 4*DIM)
        "adaLN_modulation": (f"{P}.adaLN_modulation.0", 0, 4 * DIM),
    }
    if rest not in table:
        raise ValueError(f"unknown Z-Image LoRA key: layers.{n}.{rest!r}")
    return table[rest]


def parse_zimage_lora(path: str | Path) -> list[FluxLoraSpec]:
    """A Z-Image LoRA ``.safetensors`` file -> its delta specs (CPU tensors).

    A/B conventions accepted: ``.lora_A.weight`` / ``.lora_B.weight`` (the
    official format) and diffusers-style ``.lora_down.weight`` /
    ``.lora_up.weight``. A per-key ``<base>.alpha`` scalar defaults to the
    rank (scale 1.0) when absent. Raises :class:`ValueError` listing every
    unmappable key and for incomplete pairs.
    """
    from safetensors.torch import load_file

    data = load_file(str(path))
    parts: dict[str, dict[str, torch.Tensor | float]] = {}
    unknown: list[str] = []
    for key, t in data.items():
        if key == "__metadata__":
            continue
        k = key[len("diffusion_model."):] if key.startswith("diffusion_model.") else key
        m = _LORA_KEY.match(k)
        if m is not None:
            base = _strip_dot0(m.group(1))
            parts.setdefault(base, {})[
                "A" if m.group(2) in ("A", "down") else "B"
            ] = t
            continue
        m = _ALPHA_KEY.match(k)
        if m is not None:
            parts.setdefault(_strip_dot0(m.group(1)), {})["alpha"] = float(t)
            continue
        unknown.append(key)
    if unknown:
        raise ValueError(
            f"unknown Z-Image LoRA keys in {Path(path).name}: {sorted(set(unknown))}"
        )
    specs = []
    for base, p in parts.items():
        if "A" not in p or "B" not in p:
            raise ValueError(f"incomplete LoRA A/B pair for {base!r}")
        A, B = p["A"], p["B"]
        rank = A.shape[0]
        module, r0, r1 = map_base_key(base)
        alpha = p.get("alpha", float(rank))
        specs.append(FluxLoraSpec(module, r0, r1, False, A, B, alpha / rank))
    return specs
