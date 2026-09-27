"""Z-Image LoRA key mapping and checkpoint parsing.

Two on-disk formats are supported, both keyed under ``diffusion_model.``:

- *official naming* (``diffusion_model.layers.N.attention.to_q.lora_A.weight``
  and friends; rank 32; bf16; no alpha tensors -> scale 1.0): q/k/v are
  *separate* modules, so each of those keys maps to the row slice of the
  fused ``qkv`` GEMM our transformer (:mod:`imagagen.zimage_transformer`)
  uses (the Flux LoRA pattern: ``to_q``/``to_k``/``to_v`` own the three
  thirds of ``qkv``). Only the 30 main blocks are LoRA-able in this format.
- *export naming* (``diffusion_model.layers.N.attention.qkv.lora_down.weight``
  and friends): the module paths are VERBATIM (fused ``qkv``), the pairs
  use diffusers-style ``lora_down``/``lora_up``, and non-LoRA parameters
  ship as raw additive deltas: ``<module>.diff`` (weight delta, the
  RMSNorm scales) and ``<module>.diff_b`` (bias delta). This format LoRAs
  all 30 main blocks plus the context/noise refiners, the
  ``x_embedder``/``cap_embedder``/``t_embedder`` and the ``final_layer``.

LoRA pairs become runtime additive merges (see
:meth:`imagagen.zimage_backend.ZImageBackend._apply_loras` using
:func:`flux_lora.merge_module_deltas` and ``QuantLinear.set_lora_deltas``);
direct deltas are added to the target parameter (``QuantLinear.set_lora_
bias`` / ``RMSNorm.set_lora_weight``). Everything is all-or-nothing: an
unmappable key raises :class:`ValueError` before any module is touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .flux_lora import FluxLoraSpec

#: transformer hidden size (qkv output rows per q/k/v)
DIM = 3840
#: SiLU-gated FFN width (ffn_dim_multiplier 8/3)
FFN = 10240
#: final output projection width (patch 2x2 x 16 channels)
OUT = 64
#: t_embedder hidden width (``min(dim, 1024)``)
THID = 1024

_LORA_KEY = re.compile(r"^(.+)\.lora_(A|B|down|up)\.weight$")
_ALPHA_KEY = re.compile(r"^(.+)\.alpha$")
_DIFF_KEY = re.compile(r"^(.+)\.diff$")
_DIFF_B_KEY = re.compile(r"^(.+)\.diff_b$")


@dataclass(frozen=True)
class ParamDelta:
    """One direct additive delta onto a transformer parameter.

    ``module``: path relative to the transformer root (the LoRA key with
    the ``diffusion_model.`` prefix and the ``.diff``/``.diff_b`` suffix
    removed -- verbatim, no ``.0`` stripping); ``attr`` is ``"weight"``
    or ``"bias"``. ``delta`` has the exact shape of the target parameter
    (validated at apply time).
    """

    module: str
    attr: str
    delta: torch.Tensor


def _strip_dot0(base: str) -> str:
    return base[:-2] if base.endswith(".0") else base


def _map_block(n: int, rest: str, kind: str) -> tuple[str, int, int]:
    P = f"{kind}.{n}"
    table = {
        # official naming: unfused q/k/v own the three output-thirds
        "attention.to_q": (f"{P}.attention.qkv", 0, DIM),
        "attention.to_k": (f"{P}.attention.qkv", DIM, 2 * DIM),
        "attention.to_v": (f"{P}.attention.qkv", 2 * DIM, 3 * DIM),
        # export naming: the fused qkv, whole weight
        "attention.qkv": (f"{P}.attention.qkv", 0, 3 * DIM),
        # `to_out.0` (trailing .0 stripped before mapping) == attention.out
        "attention.out": (f"{P}.attention.out", 0, DIM),
        "attention.to_out": (f"{P}.attention.out", 0, DIM),
        "feed_forward.w1": (f"{P}.feed_forward.w1", 0, FFN),
        "feed_forward.w2": (f"{P}.feed_forward.w2", 0, DIM),
        "feed_forward.w3": (f"{P}.feed_forward.w3", 0, FFN),
        # block adaLN: Linear(256 -> 4*DIM) (the export naming's trailing
        # .0 is stripped before mapping, so both reduce to this entry)
        "adaLN_modulation": (f"{P}.adaLN_modulation.0", 0, 4 * DIM),
    }
    if rest not in table:
        raise ValueError(f"unknown Z-Image LoRA key: {kind}.{n}.{rest!r}")
    return table[rest]


#: global (block-less) LoRA targets (export naming): base key ->
#: (module path, r0, r1); the module paths are verbatim
_GLOBAL = {
    "x_embedder": ("x_embedder", 0, DIM),
    "cap_embedder.1": ("cap_embedder.1", 0, DIM),
    # TimestepEmbedder.mlp: [Linear(256 -> 1024), SiLU, Linear(1024 -> 256)]
    "t_embedder.mlp": ("t_embedder.mlp.0", 0, THID),  # mlp.0, .0 stripped
    "t_embedder.mlp.2": ("t_embedder.mlp.2", 0, 256),
    "final_layer.linear": ("final_layer.linear", 0, OUT),
    "final_layer.adaLN_modulation.1": ("final_layer.adaLN_modulation.1", 0, DIM),
}


def map_base_key(base: str) -> tuple[str, int, int]:
    """Mapped base key -> ``(module, r0, r1)``.

    ``base`` is the LoRA key minus the ``diffusion_model.`` prefix, the
    ``.lora_*.weight`` suffix and a trailing ``.0``. ``module`` is
    relative to the transformer root; the delta targets the fused
    weight's output rows ``[r0:r1)``. Raises :class:`ValueError` for keys
    with no weight-matrix target.
    """
    m = re.match(r"^(layers|noise_refiner|context_refiner)\.(\d+)\.(.+)$", base)
    if m is not None:
        return _map_block(int(m.group(2)), m.group(3), m.group(1))
    if base in _GLOBAL:
        return _GLOBAL[base]
    raise ValueError(f"unknown Z-Image LoRA key: {base!r}")


def parse_zimage_lora(
    path: str | Path,
) -> tuple[list[FluxLoraSpec], list[ParamDelta]]:
    """A Z-Image LoRA ``.safetensors`` file -> its delta specs (CPU tensors).

    Returns ``(lora_pairs, param_deltas)``: the A/B pair specs (see
    :class:`flux_lora.FluxLoraSpec`) and the direct ``.diff``/``.diff_b``
    parameter deltas. A/B conventions accepted: ``.lora_A.weight`` /
    ``.lora_B.weight`` (the official format) and diffusers-style
    ``.lora_down.weight`` / ``.lora_up.weight``. A per-key
    ``<base>.alpha`` scalar defaults to the rank (scale 1.0) when absent.
    Raises :class:`ValueError` listing every unmappable key and for
    incomplete pairs.
    """
    from safetensors.torch import load_file

    data = load_file(str(path))
    parts: dict[str, dict[str, torch.Tensor | float]] = {}
    deltas: list[ParamDelta] = []
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
        # direct parameter deltas (export naming): verbatim module paths
        m = _DIFF_KEY.match(k)
        if m is not None:
            deltas.append(ParamDelta(m.group(1), "weight", t))
            continue
        m = _DIFF_B_KEY.match(k)
        if m is not None:
            deltas.append(ParamDelta(m.group(1), "bias", t))
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
    return specs, deltas
