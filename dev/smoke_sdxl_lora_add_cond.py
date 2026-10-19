"""Live SDXL smoke test for two error.log fixes (GPU).

1. prompt-change prefetch: ``prepare_text_encode`` used to pass diffusers'
   ``silence_dtype_warnings`` kwarg to the *transformers* CLIP encoders, so
   every prefetch after the first generation raised a TypeError. After a
   real generation (which parks the encoders on the CPU to free VRAM for
   sampling) the prefetch must move BOTH encoders back to the GPU.

2. SDXL added-conditioning LoRA keys: A1111/Kohya name that MLP
   ``add_embedding.linear_<N>``, and diffusers' kohya conversion eats the
   layer index, so the weights of both Linears collapsed onto one
   nonexistent ``add.embedding.linear`` key and peft dropped them
   ("Loading adapter weights ... unexpected keys found in the model").
   The fabricated LoRA below carries exactly those two layers in the A1111
   naming; the check asserts no key was dropped and that BOTH Linears got
   the right weights at the right alpha scaling.

3. the img2img/inpaint init image + mask: Pillow >= 12 exports its pixel
   buffer read-only, so ``np.asarray(Image...)`` arrays fed straight into
   ``torch.from_numpy(np.ascontiguousarray(...))`` warned "The given NumPy
   array is not writable" (once per process). The inpaint run below feeds
   such read-only arrays and must produce no such warning.

Usage (from the project root; needs GPU + models/sdxl/*.safetensors):
    .venv/bin/python dev/smoke_sdxl_lora_add_cond.py [gpu-index]
    IMAGAGEN_SMOKE_NO_FIX=1 .venv/bin/python dev/smoke_sdxl_lora_add_cond.py 1
        negative control: runs with the raw-key fix disabled, the peft drop
        from error.log must reproduce
"""
import dataclasses
import logging
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

#: the two Linear layers of the SDXL added-conditioning MLP (diffusers
#: naming), emitted under their A1111/Kohya names (layer index 0 / 2)
ADD_COND_TARGETS = [("add_embedding.linear_1", "0"), ("add_embedding.linear_2", "2")]
#: a normal attention projection, carried alongside: a LoRA made ONLY of the
#: mangled layers fails hard ("Target modules ... not found in the base
#: model") instead of producing error.log's warning. Real files such as
#: VibrantDreamXL1.0 carry hundreds of normal layers, so peft injects those
#: and reports the add-cond weights as unexpected (silently dropped).
ATTN_MOD = "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
ATTN_KEY = ("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0"
            "_attn1_to_q")
RANK = 4
ALPHA = 8.0


def module(unet, path: str):
    m = unet
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def fabricate_add_cond_lora(unet, path: Path, seed: int = 7) -> dict:
    """Write an A1111-format LoRA over the added-conditioning MLP.

    Returns the raw state dict (for the expected-value checks). The keys use
    the A1111 spelling -- ``lora_unet_add_embedding_linear_<0|2>`` -- which
    is the one diffusers' converter mangles.
    """
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for mod_path, idx in ADD_COND_TARGETS:
        m = module(unet, mod_path)
        base = f"lora_unet_add_embedding_linear_{idx}"
        sd[f"{base}.alpha"] = torch.tensor(ALPHA)
        # deterministic per layer so the two layers are distinguishable
        sd[f"{base}.lora_down.weight"] = torch.arange(
            RANK * m.in_features, dtype=torch.float32).reshape(RANK, m.in_features) / 100
        sd[f"{base}.lora_up.weight"] = torch.randn(
            m.out_features, RANK, generator=g)
    m = module(unet, ATTN_MOD)
    sd[f"{ATTN_KEY}.alpha"] = torch.tensor(ALPHA)
    sd[f"{ATTN_KEY}.lora_down.weight"] = torch.randn(
        RANK, m.in_features, generator=g)
    sd[f"{ATTN_KEY}.lora_up.weight"] = torch.randn(
        m.out_features, RANK, generator=g)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))
    return sd


class _CaptureWarnings(logging.Handler):
    """Collect WARNING records from the library loggers that report drops.

    diffusers configures its own ``diffusers`` logger with ``propagate =
    False`` and its own handler (that is why the error.log line appears
    unformatted), so a handler on the root logger alone never sees it.
    """

    LOGGERS = ("", "diffusers", "peft", "transformers")

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def __enter__(self) -> "_CaptureWarnings":
        for name in self.LOGGERS:
            logging.getLogger(name).addHandler(self)
        return self

    def __exit__(self, *exc) -> None:
        for name in self.LOGGERS:
            logging.getLogger(name).removeHandler(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def check_prefetch(backend, gpu: int) -> None:
    print("\n[prefetch] one small generation to park the text encoders ...",
          flush=True)
    backend.generate(GenerateRequest(prompt="a red cube", width=512, height=512,
                                     steps=4, seed=3))
    te1 = next(backend.pipe.text_encoder.parameters()).device
    print(f"[prefetch] after generate, text_encoder is on {te1}", flush=True)
    if te1.type != "cpu":
        print("[prefetch] encoders were not parked; forcing the CPU layout",
              flush=True)
        backend.pipe.text_encoder.cpu()
        backend.pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()

    backend.prepare_text_encode()  # used to raise TypeError
    for name in ("text_encoder", "text_encoder_2"):
        dev = next(getattr(backend.pipe, name).parameters()).device
        print(f"[prefetch] after prepare_text_encode, {name} on {dev}", flush=True)
        assert dev.type == "cuda" and dev.index == gpu, (name, dev)
    print("[prefetch] OK: both encoders moved to the GPU without raising",
          flush=True)

    # negative control against the REAL models: diffusers' kwarg really is
    # invalid for transformers modules (this is what spammed the log)
    try:
        backend.pipe.text_encoder.to(torch.device("cpu"),
                                     silence_dtype_warnings=True)
    except TypeError as exc:
        print(f"[prefetch] negative control ok: {str(exc)[:70]}...", flush=True)
    else:
        raise AssertionError("transformers .to() accepted a diffusers kwarg")


def check_add_cond_lora(backend, raw: dict, *, expect_drop: bool = False) -> None:
    print("\n[add-cond LoRA] applying the fabricated LoRA ...", flush=True)
    cap = _CaptureWarnings()
    failure = None
    with cap:
        try:
            backend._apply_loras([("addcond", 1.0)])
        except Exception as exc:  # noqa: BLE001 - reported below
            failure = exc
    dropped = [m for m in cap.records if "unexpected keys" in m]
    print("[add-cond LoRA] warnings during load:", dropped or "(none)", flush=True)
    if expect_drop:
        # negative control: with the raw-key fix disabled the bug from
        # error.log must reproduce -- either the drop warning (real LoRAs,
        # which also carry normal layers) or a hard "target not found"
        assert dropped or (failure is not None
                           and "not found in the base model" in str(failure)), \
            "expected the unmapped added-conditioning keys to surface"
        print("[add-cond LoRA] negative control reproduced the drop "
              f"({'warning' if dropped else 'hard failure'})", flush=True)
        return
    assert failure is None, failure
    assert not dropped, f"peft dropped LoRA weights: {dropped}"

    factor = ALPHA / RANK  # imagagen pre-scales down by alpha/rank
    unet = backend.pipe.unet
    for mod_path, idx in ADD_COND_TARGETS:
        m = module(unet, mod_path)
        tag = "imagagen_0"
        down_raw = raw[f"lora_unet_add_embedding_linear_{idx}.lora_down.weight"]
        up_raw = raw[f"lora_unet_add_embedding_linear_{idx}.lora_up.weight"]
        down = m.lora_A[tag].weight
        up = m.lora_B[tag].weight
        # the adapter weights live in the pipeline dtype (fp16), so compare
        # with fp16-appropriate tolerances
        d_down = (down.float().cpu() - down_raw * factor).abs().max().item()
        d_up = (up.float().cpu() - up_raw).abs().max().item()
        print(f"[add-cond LoRA] {mod_path}: max |Δdown|={d_down:.4f} "
              f"(expected down = checkpoint * alpha/rank = {factor}), "
              f"max |Δup|={d_up:.4f}, scaling={m.scaling[tag]}", flush=True)
        assert d_down < 0.5, (mod_path, d_down)
        assert d_up < 0.01, (mod_path, d_up)
        assert m.scaling[tag] == 1.0, (mod_path, m.scaling)
        print(f"[add-cond LoRA] {mod_path}: adapter weights match the "
              f"checkpoint (alpha/rank factor {factor}, scaling 1.0)", flush=True)
    # the layers must not have collapsed onto each other
    a = module(unet, ADD_COND_TARGETS[0][0]).lora_A["imagagen_0"].weight
    b = module(unet, ADD_COND_TARGETS[1][0]).lora_A["imagagen_0"].weight
    assert a.shape != b.shape or not torch.allclose(a.float(), b.float())
    print("[add-cond LoRA] OK: both layers applied, none dropped", flush=True)


def check_inpaint_init_arrays(backend, gpu: int, *, expect_warn: bool = False) -> None:
    """Read-only init image/mask must not warn (device_tensor's job)."""
    from PIL import Image

    def readonly(arr):
        # exactly what generation.resize_image hands to the backend
        return np.asarray(Image.fromarray(arr).resize((512, 512), Image.LANCZOS))

    img_src = np.full((512, 512, 3), 40, np.uint8)
    img_src[:, :256] = (200, 60, 60)
    mask_src = np.zeros((512, 512), np.uint8)
    mask_src[:, 256:] = 255  # A1111 convention: 255 = regenerate
    img, mask = readonly(img_src), readonly(mask_src)
    assert not img.flags.writeable and not mask.flags.writeable, \
        "expected Pillow to export read-only buffers"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = backend.generate(GenerateRequest(
            prompt="a blue cube", width=512, height=512, steps=6, seed=5,
            init_image=img, init_mask=mask, denoise=0.6))
    msgs = [str(w.message) for w in caught if "writable" in str(w.message)]
    print(f"[inpaint] {out.image.shape} inpainted from read-only arrays, "
          f"numpy-writable warnings: {msgs or '(none)'}", flush=True)
    assert out.image.shape == (512, 512, 3)
    if expect_warn:
        assert msgs, "expected the read-only-array warning without the fix"
        print("[inpaint] negative control reproduced the warning", flush=True)
        return
    assert not msgs, msgs
    print("[inpaint] OK: no 'NumPy array is not writable' warning", flush=True)


def main(gpu: int) -> None:
    # temp root: models/ is a symlink to the project's (incl. the HF cache);
    # loras/ is a copy so the fabricated LoRA stays out of the project
    tmp = Path(tempfile.mkdtemp(prefix="imagagen_addcond_"))
    (tmp / "models").symlink_to(ROOT / "models")
    shutil.copytree(ROOT / "loras", tmp / "loras", dirs_exist_ok=True)
    cfg = dataclasses.replace(load_config(ROOT), root=tmp)

    reg = ModelRegistry(cfg, gpus=[gpu])
    info = reg.models[0]
    print(f"loading {info.name} on GPU {gpu} ...", flush=True)
    backend = reg.backend_for(info.id)

    raw = fabricate_add_cond_lora(backend.pipe.unet,
                                 cfg.loras_dir / "sdxl" / "addcond.safetensors")
    # IMAGAGEN_SMOKE_NO_FIX=1 turns this into a negative control that must
    # reproduce the error.log behaviour
    no_fix = bool(os.environ.get("IMAGAGEN_SMOKE_NO_FIX"))
    if no_fix:
        from imagagen.backends import sdxl

        sdxl.normalize_add_cond_lora_keys = lambda sd: sd
        # undo the read-only-array fix too (the pre-fix call shape)
        sdxl.device_tensor = lambda arr, device: torch.from_numpy(
            np.ascontiguousarray(arr)).to(device)
        print("negative control: raw-key fix + device_tensor disabled",
              flush=True)
    check_add_cond_lora(backend, raw, expect_drop=no_fix)
    if not no_fix:
        backend._unload_loras()
    check_prefetch(backend, gpu)
    check_inpaint_init_arrays(backend, gpu, expect_warn=no_fix)

    print("\nOK: SDXL prefetch, the added-conditioning LoRA keys and the "
          "read-only init arrays all behave", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
