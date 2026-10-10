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
  NON-quantized targets (the bf16 :class:`imagagen.backends.krea2.transformer.
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
:class:`~imagagen.backends.krea2.transformer.DeltaLinear` so all of them accept
runtime deltas.

The A/B pairs become runtime additive deltas exactly as Z-Image/Flux:
:func:`flux_lora.merge_module_deltas` (one GEMM pair per module) and
``Int8Linear.set_lora_deltas`` (ConvRot blocks -- the deltas are trained
on the de-rotated weights, so the A matrices are rotated into the ConvRot
space at install) / ``DeltaLinear.set_lora_deltas`` (txtfusion).
Everything is all-or-nothing: an unmappable key raises :class:`ValueError`
before any module is touched.

**LoKr** (ai-toolkit, e.g. ``BenchesKII1.0.safetensors``): native naming
with ``.lokr_w1`` / ``.lokr_w2`` (+ ``.alpha``) per module -- the weight
delta is the Kronecker product ``kron(w1, w2)`` (w1 is the small
factor, w2 the big one; the file's ``alpha`` is the ai-toolkit
sentinel ``9999220736`` == alpha-as-rank, and ComfyUI's LoKr applies
scale 1.0 when ``lokr_w2`` is stored full, so the alpha is ignored).
These parse to a spec with :attr:`FluxLoraSpec.kron` and are applied
as the nested grouped GEMM of ComfyUI's LoKr bypass
(``Int8Linear/DeltaLinear.set_lora_kron``). Factorized LoKr
(``lokr_w1_a/b``, ``lokr_w2_a/b``), Tucker (``lokr_t2``) and LoHa
(``loha_*``) are rejected with a named diagnosis, as are DoRA scales
whose length matches the input axis (input-axis decomposition).

**DoRA** (e.g. ``HighcutKII1.0.safetensors`` / ``StreamlineKII1.0``,
flat native naming): a standard LoRA pair plus ``.dora_scale``
(``[1, in_features]``, the input-axis variant). The weight-decompose
math (ComfyUI's ``weight_decompose``, input-axis branch) needs the
module's base weight, so the parser only carries the vector
(:attr:`FluxLoraSpec.dora_scale`) and the backend folds the per-column
ratio ``dora_scale / ||W + d||`` into the delta's A columns at apply
time.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from ...flux_lora import FluxLoraSpec

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
_LOKR_KEY = re.compile(
    r"^(?:transformer|diffusion_model)\.(.+)\.(lokr_w1|lokr_w2|lokr_w1_a"
    r"|lokr_w1_b|lokr_w2_a|lokr_w2_b|lokr_t2)$"
)
_LOKR_KEY_UNET = re.compile(
    r"^lora_unet_(.+)\.(lokr_w1|lokr_w2|lokr_w1_a|lokr_w1_b|lokr_w2_a"
    r"|lokr_w2_b|lokr_t2)$"
)
_DORA_KEY = re.compile(
    r"^(?:transformer|diffusion_model)\.(.+)\.dora_scale$"
)
_DORA_KEY_UNET = re.compile(r"^lora_unet_(.+)\.dora_scale$")

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


#: main-block / txtfusion projection input widths (columns of each
#: module's weight) -- LoKr factor-shape validation
_NATIVE_IN = {
    "blocks": {
        "attn.wq": _D, "attn.wk": _D, "attn.wv": _D, "attn.gate": _D,
        "attn.wo": _D, "mlp.gate": _D, "mlp.up": _D, "mlp.down": _M,
    },
    "txtfusion": {
        "attn.wq": _TD, "attn.wk": _TD, "attn.wv": _TD, "attn.gate": _TD,
        "attn.wo": _TD, "mlp.gate": _TD, "mlp.up": _TD, "mlp.down": _TM,
    },
}
_NATIVE_TOP_IN = {
    "first": _D,
    "last.linear": _D,
    "tmlp.0": _D, "tmlp.2": _M,
    "txtmlp.1": _D, "txtmlp.3": _M,
    "tproj.1": 256,
    "txtfusion.projector": _TD,
}


def module_in_features(module: str) -> int:
    """Row count -> column count of a mapped K2 module's weight.

    The inverse of :func:`map_base_key`'s row table (the LoKr factor
    shapes must multiply out to ``(r1 - r0) x in_features`).
    """
    if module in _NATIVE_TOP_IN:
        return _NATIVE_TOP_IN[module]
    nm = re.match(
        r"^(blocks|txtfusion\.(?:layerwise_blocks|refiner_blocks))"
        r"\.\d+\.(.+)$",
        module,
    )
    if nm is not None:
        width = _NATIVE_IN[nm.group(1).split(".")[0]].get(nm.group(2))
        if width is not None:
            return width
    raise ValueError(f"no input width for Krea 2 module {module!r}")


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


def _sniff_unsupported(unknown: list[str]) -> str | None:
    """Name an unsupported LoRA family from its keys (LoHa, GaLore), or
    None when the keys are simply unknown."""
    keys = set(unknown)
    if any(".loha_" in k or k.startswith("loha_") for k in keys):
        return "LoHa (loha_* keys) is not supported for Krea 2"
    if any(".galore_" in k for k in keys):
        return "GaLore LoRAs are not supported for Krea 2"
    return None


def parse_krea2_lora(path: str | Path) -> list[FluxLoraSpec]:
    """A Krea 2 LoRA ``.safetensors`` file -> its delta specs (CPU tensors).

    Raises :class:`ValueError` listing every key that is neither a
    mappable A/B (or LoKr/DoRA) key nor an alpha scalar, and for
    incomplete pairs.
    """
    from safetensors.torch import load_file

    from ...flux_lora import sniff_lora_format

    if why := sniff_lora_format(path, skip=("lokr", "dora")):
        raise ValueError(
            f"unsupported Krea 2 LoRA format in {Path(path).name}: {why}")
    try:
        data = load_file(str(path))
    except Exception as exc:  # safetensors rejects a GGUF magic header
        with open(path, "rb") as fh:
            magic = fh.read(4)
        if magic == b"GGUF":
            raise ValueError(
                f"{Path(path).name}: GGUF-quantized LoRAs are not "
                "supported; use the plain .safetensors release"
            ) from exc
        raise
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
        m = _LOKR_KEY.match(key) or _LOKR_KEY_UNET.match(key)
        if m is not None:
            base = _strip_dot0(_flat_to_native(m.group(1))
                               if key.startswith("lora_unet_")
                               else m.group(1))
            parts.setdefault(base, {})[m.group(2)] = t
            continue
        m = _DORA_KEY.match(key) or _DORA_KEY_UNET.match(key)
        if m is not None:
            base = _strip_dot0(_flat_to_native(m.group(1))
                               if key.startswith("lora_unet_")
                               else m.group(1))
            parts.setdefault(base, {})["dora"] = t
            continue
        unknown.append(key)
    if unknown:
        diagnosis = _sniff_unsupported(unknown)
        if diagnosis is not None:
            raise ValueError(f"{Path(path).name}: {diagnosis}")
        raise ValueError(
            f"unknown Krea 2 LoRA keys in {Path(path).name}: "
            f"{sorted(set(unknown))}"
        )
    specs = []
    for base, p in parts.items():
        module, r0, r1 = map_base_key(base)
        dora = p.get("dora")
        if dora is not None:
            dora = dora.reshape(-1)
            fin = module_in_features(module)
            if dora.numel() != fin:
                raise ValueError(
                    f"{base!r}: DoRA scale covers {dora.numel()} values, "
                    f"expected {fin} (input axis of {module}); output-axis "
                    "DoRA is not supported"
                )
        if "lokr_w1" in p or "lokr_w2" in p:
            if dora is not None:
                raise ValueError(
                    f"{base!r}: LoKr + DoRA on one module is not "
                    "supported (the DoRA column scaling does not factor "
                    "through the Kronecker application)"
                )
            for extra in ("lokr_w1_a", "lokr_w1_b", "lokr_w2_a",
                          "lokr_w2_b", "lokr_t2"):
                if extra in p:
                    raise ValueError(
                        f"{base!r}: factorized/Tucker LoKr "
                        f"({extra}) is not supported (only full-matrix "
                        "lokr_w1 + lokr_w2); train a plain LoRA/DoRA "
                        "for Krea 2"
                    )
            if "lokr_w1" not in p or "lokr_w2" not in p:
                raise ValueError(f"incomplete LoKr pair for {base!r}")
            w1, w2 = p["lokr_w1"], p["lokr_w2"]
            if w1.ndim != 2 or w2.ndim != 2:
                raise ValueError(f"{base!r}: non-2D LoKr factors")
            out, fin = r1 - r0, module_in_features(module)
            if w1.shape[0] * w2.shape[0] != out \
                    or w1.shape[1] * w2.shape[1] != fin:
                raise ValueError(
                    f"{base!r}: LoKr kron shape "
                    f"{w1.shape[0] * w2.shape[0]}x"
                    f"{w1.shape[1] * w2.shape[1]} != module weight "
                    f"{out}x{fin}"
                )
            specs.append(FluxLoraSpec(
                module, r0, r1, False, w1, torch.empty(0), 1.0,
                kron=(w1, w2), dora_scale=dora))
            continue
        if "A" not in p or "B" not in p:
            raise ValueError(f"incomplete LoRA A/B pair for {base!r}")
        A, B = p["A"], p["B"]
        rank = A.shape[0]
        alpha = p.get("alpha", float(rank))
        specs.append(FluxLoraSpec(
            module, r0, r1, False, A, B, alpha / rank, dora_scale=dora))
    return specs
