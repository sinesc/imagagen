"""Krea 2 LoRA key mapping and checkpoint parsing.

K2 LoRAs ship in two formats:

**diffusers naming** (e.g. ``rustic-ambience.safetensors``): a
``transformer.`` prefix:

- ``transformer.transformer_blocks.N.attn.to_q|to_k|to_v|to_gate|
  to_out.0.lora_{A,B}.weight`` and ``transformer.transformer_blocks.N.
  ff.gate|up|down.lora_{A,B}.weight`` (the 28 main blocks);
- ``transformer.text_fusion.layerwise_blocks.N.*`` / ``transformer.
  text_fusion.refiner_blocks.N.*`` (the txtfusion adapter, same
  ``attn.*`` / ``ff.*`` sub-keys) -- these are the LoRA's only
  NON-quantized targets (the bf16 :class:`imagagen.krea2_transformer.
  DeltaLinear` modules);
- rank 16, bf16, no alpha tensors (scale 1.0).

**native/ComfyUI naming** (e.g. ``Real3dKII1.0.safetensors``): a
``diffusion_model.`` prefix whose module names already match ours
(``blocks.N.attn.wq|wk|wv|gate|wo``, ``blocks.N.mlp.gate|up|down``,
``txtfusion.layerwise_blocks.N.*`` / ``txtfusion.refiner_blocks.N.*``).
These are identity-mapped (no renaming).

**ComfyUI flat native naming** (e.g. ``HyperspaceKII1.0.safetensors``):
the same native module names, but flat under a ``lora_unet_`` prefix
with underscores standing in for dots
(``lora_unet_blocks_0_attn_wq.lora_down.weight``). The underscores are
turned back into dots, giving the identical native names (so this
format also covers the non-block modules ``first``, ``last.linear``,
``tmlp.0`` / ``tmlp.2``, ``txtmlp.1`` / ``txtmlp.3``, ``tproj.1`` and
``txtfusion.projector``).

The diffusers mapping (the inverse of ``comfy.utils.krea2_to_diffusers``'
diffusers->native table, adapted to our module paths, which are relative
to the transformer root):

- ``transformer_blocks.N``  -> ``blocks.N``
- ``attn.to_q``/``to_k``/``to_v``/``to_gate`` -> ``attn.wq``/``wk``/``wv``/
  ``gate``; ``attn.to_out.0`` (trailing ``.0`` stripped) -> ``attn.wo``
- ``ff.gate``/``ff.up``/``ff.down`` -> ``mlp.gate``/``mlp.up``/``mlp.down``
- ``text_fusion.layerwise_blocks.N`` / ``text_fusion.refiner_blocks.N``
  -> ``txtfusion.layerwise_blocks.N`` / ``txtfusion.refiner_blocks.N``

The diffusers-format LoRA targets only the 28 blocks + txtfusion
attn/mlp; the flat native format additionally carries the top-level
projections (``first``, ``last.linear``, ``tmlp`` / ``txtmlp`` /
``tproj`` linears, ``txtfusion.projector``) -- every K2 projection is a
:class:`~imagagen.krea2_transformer.DeltaLinear` so all of them accept
runtime deltas.

The A/B pairs become runtime additive deltas exactly as Z-Image/Flux:
:func:`flux_lora.merge_module_deltas` (one GEMM pair per module) and
``Int8Linear.set_lora_deltas`` (ConvRot blocks -- the deltas run on the
ROTATED input, matching the ConvRot weights) / ``DeltaLinear.set_lora_
deltas`` (txtfusion). Everything is all-or-nothing: an unmappable key
raises :class:`ValueError` before any module is touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from .flux_lora import FluxLoraSpec

#: main-block / txtfusion projection widths (rows of each module's weight)
_D = 6144      # blocks: q/gate/out, hidden
_KD = 1536     # blocks: k/v (GQA 12 kv heads x 128)
_M = 16384     # blocks: mlp gate/up
_TD = 2560     # txtfusion: q/k/v/gate/out, hidden
_TM = 6912     # txtfusion: mlp gate/up
_OUT = 64      # last.linear out: patch*patch*channels

_LORA_KEY = re.compile(
    r"^(?:transformer|diffusion_model)\.(.+)\.lora_(A|B|down|up)\.weight$"
)
_LORA_KEY_UNET = re.compile(
    r"^lora_unet_(.+)\.lora_(A|B|down|up)\.weight$"
)
_ALPHA_KEY = re.compile(r"^(?:transformer|diffusion_model)\.(.+)\.alpha$")
_ALPHA_KEY_UNET = re.compile(r"^lora_unet_(.+)\.alpha$")

#: native top-level (non-block) module -> row count of its weight
_NATIVE_TOP = {
    "first": _D,
    "last.linear": _OUT,
    "tmlp.0": _D, "tmlp.2": _D,
    "txtmlp.1": _D, "txtmlp.3": _D,
    "tproj.1": 6 * _D,
    "txtfusion.projector": 1,
}

#: native-format module -> row count of its weight
_NATIVE_ROWS = {
    "blocks": {
        "attn.wq": _D, "attn.wk": _KD, "attn.wv": _KD, "attn.gate": _D,
        "attn.wo": _D, "mlp.gate": _M, "mlp.up": _M, "mlp.down": _D,
    },
    "txtfusion": {
        "attn.wq": _TD, "attn.wk": _TD, "attn.wv": _TD, "attn.gate": _TD,
        "attn.wo": _TD, "mlp.gate": _TM, "mlp.up": _TM, "mlp.down": _TD,
    },
}


def _strip_dot0(base: str) -> str:
    return base[:-2] if base.endswith(".0") else base


def _flat_to_native(name: str) -> str:
    """ComfyUI flat name -> dotted native module name.

    ``layerwise_blocks`` / ``refiner_blocks`` are themselves single tokens
    with an underscore, so they are protected before the remaining
    underscores (the dots) are restored.
    """
    for tok in ("layerwise_blocks", "refiner_blocks"):
        name = name.replace(tok, tok.replace("_", "~"))
    return name.replace("_", ".").replace("~", "_")


def lora_key(key: str) -> tuple[str, str] | None:
    """Full tensor name -> ``(base key, 'A'/'B')``, or None.

    Accepts A1111/Kohya ``.lora_A.weight`` / ``.lora_B.weight`` and
    diffusers-style ``.lora_down.weight`` / ``.lora_up.weight`` (A == the
    down projection). A trailing ``.0`` on the base is stripped
    (``attn.to_out.0`` == ``attn.to_out``). The flat ComfyUI
    ``lora_unet_*`` keys are converted to the dotted native names
    (underscores -> dots)."""
    m = _LORA_KEY.match(key)
    if m is not None:
        base = _strip_dot0(m.group(1))
    else:
        m = _LORA_KEY_UNET.match(key)
        if m is None:
            return None
        base = _flat_to_native(m.group(1))
    return base, "A" if m.group(2) in ("A", "down") else "B"


def alpha_key(key: str) -> str | None:
    """Full tensor name -> its base key, or None (not an alpha scalar)."""
    m = _ALPHA_KEY.match(key)
    if m is not None:
        return _strip_dot0(m.group(1))
    m = _ALPHA_KEY_UNET.match(key)
    return _flat_to_native(m.group(1)) if m is not None else None


def _map_block(base: str, n: int, rest: str, module_prefix: str) -> tuple[str, int, int]:
    P = f"{module_prefix}.{n}"
    if base.startswith("transformer_blocks"):
        table = {
            "attn.to_q": (f"{P}.attn.wq", 0, _D),
            "attn.to_k": (f"{P}.attn.wk", 0, _KD),
            "attn.to_v": (f"{P}.attn.wv", 0, _KD),
            "attn.to_gate": (f"{P}.attn.gate", 0, _D),
            "attn.to_out": (f"{P}.attn.wo", 0, _D),
            "ff.gate": (f"{P}.mlp.gate", 0, _M),
            "ff.up": (f"{P}.mlp.up", 0, _M),
            "ff.down": (f"{P}.mlp.down", 0, _D),
        }
    else:  # text_fusion.* blocks (same sub-keys, txtfusion widths)
        table = {
            "attn.to_q": (f"{P}.attn.wq", 0, _TD),
            "attn.to_k": (f"{P}.attn.wk", 0, _TD),
            "attn.to_v": (f"{P}.attn.wv", 0, _TD),
            "attn.to_gate": (f"{P}.attn.gate", 0, _TD),
            "attn.to_out": (f"{P}.attn.wo", 0, _TD),
            "ff.gate": (f"{P}.mlp.gate", 0, _TM),
            "ff.up": (f"{P}.mlp.up", 0, _TM),
            "ff.down": (f"{P}.mlp.down", 0, _TD),
        }
    if rest not in table:
        raise ValueError(f"unknown Krea 2 LoRA key: {base!r}")
    return table[rest]


def map_base_key(base: str) -> tuple[str, int, int]:
    """Mapped K2 LoRA base key -> ``(module, r0, r1)``.

    ``base`` is the LoRA key minus the ``transformer.`` / ``diffusion_model.``
    prefix and the ``.lora_*.weight`` suffix (trailing ``.0`` already
    stripped). Native-format keys are identity-mapped (their names already
    match our modules). ``module`` is relative to the transformer root; the
    delta covers the whole module weight (``r0 = 0`` in every case). Raises
    :class:`ValueError` for keys with no mappable target.
    """
    if base in _NATIVE_TOP:
        return base, 0, _NATIVE_TOP[base]
    nm = re.match(
        r"^(blocks|txtfusion\.(?:layerwise_blocks|refiner_blocks))"
        r"\.\d+\.(.+)$",
        base,
    )
    if nm is not None:
        part = nm.group(2)
        r1 = _NATIVE_ROWS[nm.group(1).split(".")[0]].get(part)
        if r1 is None:
            raise ValueError(f"unknown Krea 2 LoRA key: {base!r}")
        return base, 0, r1
    m = re.match(
        r"^(transformer_blocks|text_fusion\.(layerwise_blocks|refiner_blocks))"
        r"\.(\d+)\.(.+)$",
        base,
    )
    if m is None:
        raise ValueError(f"unknown Krea 2 LoRA key: {base!r}")
    module_prefix = (
        "blocks" if m.group(1) == "transformer_blocks"
        else f"txtfusion.{m.group(2)}"
    )
    return _map_block(m.group(1), int(m.group(3)), m.group(4), module_prefix)


def parse_krea2_lora(path: str | Path) -> list[FluxLoraSpec]:
    """A Krea 2 LoRA ``.safetensors`` file -> its delta specs (CPU tensors).

    Raises :class:`ValueError` listing every key that is neither a
    mappable A/B pair nor an alpha scalar, and for incomplete pairs.
    """
    from safetensors.torch import load_file

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
            f"unknown Krea 2 LoRA keys in {Path(path).name}: "
            f"{sorted(set(unknown))}"
        )
    specs = []
    for base, p in parts.items():
        if "A" not in p or "B" not in p:
            raise ValueError(f"incomplete LoRA A/B pair for {base!r}")
        A, B = p["A"], p["B"]
        module, r0, r1 = map_base_key(base)
        rank = A.shape[0]
        alpha = p.get("alpha", float(rank))
        specs.append(FluxLoraSpec(module, r0, r1, False, A, B, alpha / rank))
    return specs
