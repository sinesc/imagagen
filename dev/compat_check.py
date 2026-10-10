"""Header-only checkpoint compatibility check: run this after adding any
file under ``models/`` to answer "does it just work" in seconds.

For every ``.safetensors`` file the script reads ONLY the safetensors
header (no tensor payload), detects the model family from the key
signatures, classifies the quantization layout (metadata / markers /
dtype heuristics), and -- for Krea 2 -- maps every tensor key against a
tiny meta-device clone of the architecture built from the header shapes,
so unmapped keys (wrong family, renamed layers, extra adapters) are
listed explicitly. Flux reuses the backend's own checkpoint sniffer.

Usage:  .venv/bin/python dev/compat_check.py [path ...]
(default: every file under models/)
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends.krea2.transformer import Krea2Transformer  # noqa: E402
from imagagen.gguf_ops import sniff_flux_checkpoint  # noqa: E402

QUANT_FORMATS = {  # metadata/marker "format" strings -> layout name
    "convrot_w4a4": "int4",
    "int4_convrot": "int4",
    "int8_tensorwise": "int8",
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
}


def read_header(path: Path):
    with safe_open(str(path), framework="pt") as f:
        names = list(f.keys())
        shapes = {k: (f.get_slice(k).get_dtype(),
                      tuple(f.get_slice(k).get_shape())) for k in names}
        meta = f.metadata() or {}
    return names, shapes, meta


def detect_family(names: list[str]) -> str | None:
    s = set(names)
    # krea2: SingleStreamDiT -- "txtfusion.<n>." + the single-stream
    # blocks (native "single_transformer_blocks." / de-prefixed
    # "blocks." naming; the int8 release carries bare block keys with
    # the attention gate)
    if any("txtfusion." in k for k in s) and (
            any("single_transformer_blocks." in k for k in s)
            or any("attn.gate" in k for k in s)):
        return "krea2"
    # flux: the double blocks (native "model.diffusion_model."
    # naming and the diffusers "transformer." naming both carry
    # "double_blocks."; no other family uses that name)
    if any("double_blocks." in k for k in s):
        return "flux"
    if "losses" in s \
            or any(k.startswith(("cond_stage_model.", "conditioner.embedders.",
                                 "diffusion_model.")) for k in s):
        return "sdxl"
    if any(k.startswith("cap_embedder.") for k in s) \
            or any(k.startswith("adaLN_modulation.") for k in s):
        return "zimage"
    return None


def classify_krea2(names, shapes, meta) -> tuple[str, str | None]:
    """(layout, problem) mirroring the loader's classifier."""
    formats = QUANT_FORMATS
    file_fmt, layers = "", {}
    if qm := meta.get("_quantization_metadata"):
        try:
            qm = json.loads(qm)
        except Exception:
            return "?", "unreadable _quantization_metadata"
        file_fmt = qm.get("format", "")
        layers = qm.get("layers", {}) or {}
    prefix = "model.diffusion_model."
    if all(k.startswith(prefix) for k in names):
        names = [k[len(prefix):] for k in names]
        shapes = {k[len(prefix):]: v for k, v in shapes.items()
                  if k.startswith(prefix)}
    found = set()
    for name in names:
        if not name.endswith(".weight"):
            continue
        base = name[: -len(".weight")]
        layer = layers.get(base)
        fmt = ""
        if layer is not None:
            fmt = layer.get("format", "")
            if fmt not in formats:
                return "?", f"unsupported layer format {fmt!r} ({base})"
        else:
            dtype = shapes[name][0]
            scale = f"{name}_scale"
            ndim = len(shapes[scale][1]) if scale in shapes else None
            if dtype == "F8_E4M3" and scale in shapes:
                fmt = "float8_e4m3fn"
            elif dtype == "I8" and ndim == 1:
                fmt = "convrot_w4a4"
            elif dtype == "I8" and ndim == 2:
                fmt = "int8_tensorwise"
            elif scale in shapes and file_fmt in formats:
                fmt = file_fmt
        if fmt:
            found.add(formats[fmt])
    if not found:
        # no quantized tensors at all: plain bf16 (unsupported size)
        return "bf16", None
    if len(found) > 1:
        return "+".join(sorted(found)), "mixed quantized layouts"
    return found.pop(), None


def krea2_tiny_map(shapes) -> tuple[set[str], dict]:
    """Module-name set of a tiny meta Krea2Transformer sized from the
    checkpoint header (features/layers/heads read from the shapes)."""
    layers = 1 + max(
        int(k.split("blocks.")[1].split(".")[0])
        for k in shapes if "blocks." in k)
    last = shapes.get("model.diffusion_model.last.linear.weight") \
        or shapes.get("last.linear.weight")
    feats = last[1][-1]
    heads = 48
    an = shapes.get("model.diffusion_model.blocks.0.attn.k_norm.weight") \
        or shapes.get("blocks.0.attn.k_norm.weight")
    kvheads = an[1][0] if an and len(an[1]) == 2 else 12
    with torch.device("meta"):
        m = Krea2Transformer(features=feats, layers=1, heads=heads,
                             kvheads=kvheads, multiplier=1, tdim=8,
                             txtdim=16, txtlayers=1)
    return {n for n, _ in m.named_modules()}, layers


def check_krea2(path: Path, names, shapes, meta) -> list[str]:
    notes = []
    layout, problem = classify_krea2(list(names), shapes, meta)
    notes.append(f"layout={layout}" + (f" [{problem}]" if problem else ""))
    if problem and layout == "?":
        return notes
    prefix = "model.diffusion_model."
    bases = {k[len(prefix):] if k.startswith(prefix) else k
             for k in names}
    bases = {k.rsplit(".", 1)[0] for k in bases
             if k.endswith((".weight", ".bias"))
             or ".comfy_quant" in k or k.endswith("_scale")}
    modules, layers = krea2_tiny_map(shapes)
    # the tiny clone has 1 block / 1 txt layer: compare the block keys of
    # block 0 (all blocks share the structure) plus the global keys
    unmapped = []
    for b in sorted(bases):
        if b.endswith(".comfy_quant") or b.endswith("_scale"):
            continue
        probe = b
        # normalize the block/layer index to the clone's index 0; the
        # txtfusion pattern first (its name CONTAINS "blocks.")
        for pat in ("txtfusion.layerwise_blocks.", "blocks."):
            i = probe.find(pat)
            if i >= 0:
                tail = probe[i + len(pat):]
                idx = tail.split(".")[0]
                if idx.isdigit():
                    probe = probe[:i + len(pat)] + "0." + tail[len(idx) + 1:]
                break
        if probe not in modules:
            unmapped.append(b)
    notes.append(f"keys mapped, unmapped={len(unmapped)}")
    for u in unmapped[:6]:
        notes.append(f"  unmapped: {u}")
    notes.append(f"(tiny clone: 1/{layers} blocks, features from header)")
    return notes


def check_flux(path_str: str, names) -> list[str]:
    notes = []
    try:
        kind = sniff_flux_checkpoint(path_str)
        notes.append(f"layout={kind} (backend sniffer accepts the file)")
    except ValueError as e:
        notes.append(f"layout=REJECTED: {e}")
    return notes


def check_generic(family, names, shapes) -> list[str]:
    tr = [k for k in names if "transformer." in k or "unet." in k
          or "diffusion_model." in k or k.startswith("layers.")]
    dtypes = {}
    for dt, _ in shapes.values():
        dtypes[dt] = dtypes.get(dt, 0) + 1
    quant = " ".join(f"{dt}:{n}" for dt, n in sorted(dtypes.items())
                     if dt in ("F8_E4M3", "F8_E5M2", "I8", "I4", "U8"))
    return [f"{len(names)} tensors ({len(tr)} unet/transformer-section)"
            + (f", quantized dtypes: {quant}" if quant else "")]


if __name__ == "__main__":
    args = sys.argv[1:]
    files = [Path(a).resolve() for a in args] if args else sorted(
        (ROOT / "models").rglob("*.safetensors"))
    for p in files:
        print(f"== {p.relative_to(ROOT)}")
        try:
            names, shapes, meta = read_header(p)
        except Exception as e:
            print(f"   not a safetensors file: {e}")
            continue
        family = detect_family(names)
        if family is None:
            print("   family: ??? (no known key signature)")
            continue
        print(f"   family: {family}")
        if family == "krea2":
            for line in check_krea2(p, names, shapes, meta):
                print(f"   {line}")
        elif family == "flux":
            for line in check_flux(str(p), names):
                print(f"   {line}")
        else:
            for line in check_generic(family, names, shapes):
                print(f"   {line}")
