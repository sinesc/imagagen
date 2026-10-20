"""Project configuration and directory layout."""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Root of the imagagen project plus all derived data directories.

    User data (settings, presets, styles, wildcards, brushes, ...)
    lives in ``<root>/user/`` so it is easy to share/backup as one
    folder; the provided defaults live in ``<root>/res/`` (tracked in
    git, never written by the app).

    GPU selection is read from ``<root>/user/imagagen.ini`` (see
    :func:`_parse_ini`): ``primary_gpu`` is ``None`` for auto-pick
    (most free VRAM) or an explicit index; ``ignore_gpus`` are excluded
    from auto-pick (a friendlier alternative to CUDA_VISIBLE_DEVICES).
    """

    root: Path
    primary_gpu: int | None = None
    ignore_gpus: frozenset[int] = field(default_factory=frozenset)
    # VRAM options (imagagen.ini): trade VRAM for speed/quality
    unet_quantization: str = "off"  # "off" | "nf4"
    text_encoder_device: str = "gpu"  # "gpu" | "cpu"
    vae_tiling: bool = False
    # Seconds the queue waits after going idle before parking the loaded
    # models out of VRAM into RAM (0 = park immediately, -1 = never park
    # explicitly; the models are still unloaded whenever VRAM is needed).
    transformer_parking_delay: float = 90.0
    # On a prompt change whose final prompt would miss the prompt-encoding
    # cache (always true for wildcard prompts), immediately move the GPU to
    # its "about to encode" layout (text encoders resident, transformer
    # parked when they cannot co-reside) instead of waiting for the next
    # generation to pay the swap.
    prefetch_text_encoder: bool = True
    # Memory-leak debugging (see imagagen.memdbg): true stamps every
    # multi-GiB model move (park/load/prefetch) with the process RSS, runs a
    # periodic object census and mirrors it all into user/memdbg.log.
    memory_debug: bool = False
    # Seconds between the sampler's cheap RSS lines (memory-debug only).
    memory_debug_interval: float = 30.0
    # Development aid: no-store on all responses (see cache middleware in
    # app.py); without it responses are revalidated on every request.
    debug: bool = False
    # Persist the UI's form state to <root>/user/ui-state.json on every
    # save (true = default). Disable it to keep the file from being
    # written; an existing file is still loaded on start.
    save_ui_state: bool = True
    # Mask overlay opacity: the alpha multiplier the UI applies when
    # rendering the mask over the canvas (0.9 = the historical default).
    mask_overlay_opacity: float = 0.9

    @property
    def user_dir(self) -> Path:
        """All user data (share/backup this folder)."""
        return self.root / "user"

    @property
    def res_dir(self) -> Path:
        """Provided defaults (tracked in git; never written by the app)."""
        return self.root / "res"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def loras_dir(self) -> Path:
        return self.root / "loras"

    @property
    def infra_dir(self) -> Path:
        """Optional encoder/decoder weights (VAEs, text encoders, ...)."""
        return self.root / "infra"

    @property
    def brushes_dir(self) -> Path:
        """User brushes (grey PNGs), searched after the provided ones."""
        return self.user_dir / "brushes"

    @property
    def default_brushes_dir(self) -> Path:
        """Provided default brushes (res/brushes/), searched first."""
        return self.res_dir / "brushes"

    @property
    def wildcards_dir(self) -> Path:
        return self.user_dir / "wildcards"

    @property
    def presets_dir(self) -> Path:
        return self.user_dir / "presets"

    @property
    def model_configs_dir(self) -> Path:
        return self.user_dir / "model-configs"

    @property
    def styles_csv_path(self) -> Path:
        return self.user_dir / "styles.csv"

    @property
    def ui_state_path(self) -> Path:
        return self.user_dir / "ui-state.json"

    @property
    def ini_path(self) -> Path:
        return self.user_dir / "imagagen.ini"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def segmentation_models_dir(self) -> Path:
        return self.models_dir / "segmentation"

    @property
    def hf_cache_dir(self) -> Path:
        """Persistent HuggingFace hub cache (tokenizers, configs, ...).

        Kept in the project-local ``cache/`` directory so the ~2 MB of hub
        files a pipeline's `from_single_file` still fetches are downloaded
        once and survive container rebuilds (the container's ~/.cache is
        ephemeral).
        """
        return self.root / "cache"


def load_config(root: Path | str | None = None) -> Config:
    """Build a Config from an explicit root, $IMAGEN_ROOT, or the cwd."""
    if root is None:
        root = os.environ.get("IMAGEN_ROOT", Path.cwd())
    root = Path(root).expanduser().resolve()
    (
        primary_gpu,
        ignore_gpus,
        unet_quantization,
        text_encoder_device,
        vae_tiling,
        debug,
        save_ui_state,
        mask_overlay_opacity,
        transformer_parking_delay,
        prefetch_text_encoder,
        memory_debug,
        memory_debug_interval,
    ) = _parse_ini(root / "user" / "imagagen.ini")
    config = Config(
        root=root,
        primary_gpu=primary_gpu,
        ignore_gpus=ignore_gpus,
        unet_quantization=unet_quantization,
        text_encoder_device=text_encoder_device,
        vae_tiling=vae_tiling,
        debug=debug,
        save_ui_state=save_ui_state,
        mask_overlay_opacity=mask_overlay_opacity,
        transformer_parking_delay=transformer_parking_delay,
        prefetch_text_encoder=prefetch_text_encoder,
        memory_debug=memory_debug,
        memory_debug_interval=memory_debug_interval,
    )
    # Must happen before huggingface_hub is imported (it reads these at
    # import time): cache hub files in the project, keep the console quiet.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(config.hf_cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # NOTE: expandable_segments was here (0d4589d) for Flux VRAM headroom but
    # breaks multi-GPU SDXL generation with CUBLAS_STATUS_INTERNAL_ERROR on
    # mixed-arch hosts (sm_120 + sm_89).  Re-add only if/when Flux OOMs on a
    # single GPU and multi-GPU SDXL is not a concern.
    # torchvision-less image-processor fallbacks are exactly what we want
    # (PIL backend); stop transformers from announcing them on first import
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import logging

    # Keep the console clean: these loggers are noisy at import time but
    # harmless for our fp16 GPU-only usage.
    for name in ("huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)

    # diffusers warns on every ModelMixin .to(dtype) that some modules "should
    # be kept in float32" -- even when the keep-in-fp32 list is empty ([]).
    # Both VAEs we load are plain AutoencoderKLs that keep nothing in fp32, and
    # our fp16/bf16 VAE casts are deliberate (Forge parity; from_single_file
    # casts internally, with no opt-out), so the message is pure spam. Filter
    # just this message on the emitting logger (a blanket level would also hide
    # useful modeling_utils warnings like state-dict mismatches).
    class _KeepInFp32Filter(logging.Filter):
        def filter(self, record):
            return "should be kept in float32" not in record.getMessage()

    _modeling_utils = logging.getLogger("diffusers.models.modeling_utils")
    if not any(isinstance(f, _KeepInFp32Filter) for f in _modeling_utils.filters):
        _modeling_utils.addFilter(_KeepInFp32Filter())

    # bitsandbytes has no prebuilt binary for this host's CUDA minor version
    # (12.9); its nearest-prebuilt fallback (12.8) is the correct behavior, so
    # drop the one-line notice rather than pinning BNB_CUDA_VERSION to a
    # version that a future bnb release might not ship. (Attached on the
    # emitting logger: filters on an ancestor do not see propagated records.)
    class _BnbCudaFallbackFilter(logging.Filter):
        def filter(self, record):
            return "No prebuilt binary" not in record.getMessage()

    _bnb = logging.getLogger("bitsandbytes.cextension")
    if not any(isinstance(f, _BnbCudaFallbackFilter) for f in _bnb.filters):
        _bnb.addFilter(_BnbCudaFallbackFilter())
    return config


def _parse_ini(
    path: Path,
) -> tuple[int | None, frozenset[int], str, str, bool, bool, bool, float,
           float, bool, bool, float]:
    """Parse ``<root>/user/imagagen.ini``: GPU selection, VRAM options, debug.

    A missing file yields the defaults
    (auto, none, off, gpu, False, False, True, 0.9, 90.0, True, False, 30.0).
    """
    if not path.is_file():
        return (None, frozenset(), "off", "gpu", False, False, True, 0.9,
                90.0, True, False, 30.0)
    cp = configparser.ConfigParser()
    try:
        cp.read(path, encoding="utf-8")
    except configparser.Error as e:
        raise ValueError(f"cannot parse {path.name}: {e}") from e
    if not cp.has_section("imagagen"):
        return (None, frozenset(), "off", "gpu", False, False, True, 0.9,
                90.0, True, False, 30.0)
    sec = cp["imagagen"]

    raw = sec.get("primary-gpu", "auto").strip().lower()
    primary: int | None
    if raw in ("", "auto"):
        primary = None
    else:
        try:
            primary = int(raw)
        except ValueError:
            raise ValueError(
                f"{path.name}: primary-gpu must be 'auto' or a GPU index, "
                f"got {raw!r}"
            ) from None

    ignored: set[int] = set()
    for part in sec.get("ignore-gpus", "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ignored.add(int(part))
        except ValueError:
            raise ValueError(
                f"{path.name}: ignore-gpus must be comma-separated GPU "
                f"indices, got {part!r}"
            ) from None

    quant = sec.get("unet-quantization", "off").strip().lower()
    if quant not in ("off", "nf4"):
        raise ValueError(
            f"{path.name}: unet-quantization must be 'off' or 'nf4', got {quant!r}"
        )
    te_dev = sec.get("text-encoder-device", "gpu").strip().lower()
    if te_dev not in ("gpu", "cpu"):
        raise ValueError(
            f"{path.name}: text-encoder-device must be 'gpu' or 'cpu', got {te_dev!r}"
        )
    vae_tiling = sec.getboolean("vae-tiling", False)
    try:
        debug = sec.getboolean("debug", False)
    except ValueError:
        raise ValueError(
            f"{path.name}: debug must be true/false, got {sec.get('debug')!r}"
        ) from None
    try:
        save_ui_state = sec.getboolean("save-ui-state", True)
    except ValueError:
        raise ValueError(
            f"{path.name}: save-ui-state must be true/false, "
            f"got {sec.get('save-ui-state')!r}"
        ) from None
    try:
        mask_overlay_opacity = float(sec.get("mask-overlay-opacity", "0.9"))
    except ValueError:
        raise ValueError(
            f"{path.name}: mask-overlay-opacity must be a number, "
            f"got {sec.get('mask-overlay-opacity')!r}"
        ) from None
    if not 0.0 <= mask_overlay_opacity <= 1.0:
        raise ValueError(
            f"{path.name}: mask-overlay-opacity must be in [0, 1], "
            f"got {mask_overlay_opacity!r}"
        )
    try:
        parking_delay = float(sec.get("transformer-parking-delay", "90"))
    except ValueError:
        raise ValueError(
            f"{path.name}: transformer-parking-delay must be a number of "
            f"seconds, -1 (never park) or 0 (park immediately), got "
            f"{sec.get('transformer-parking-delay')!r}"
        ) from None
    if parking_delay < -1.0:
        raise ValueError(
            f"{path.name}: transformer-parking-delay must be -1 (never "
            f"park), 0 (park immediately) or a number of seconds, got "
            f"{parking_delay!r}"
        )
    try:
        prefetch_te = sec.getboolean("prefetch-text-encoder", True)
    except ValueError:
        raise ValueError(
            f"{path.name}: prefetch-text-encoder must be true/false, "
            f"got {sec.get('prefetch-text-encoder')!r}"
        ) from None
    try:
        memory_debug = sec.getboolean("memory-debug", False)
    except ValueError:
        raise ValueError(
            f"{path.name}: memory-debug must be true/false, "
            f"got {sec.get('memory-debug')!r}"
        ) from None
    try:
        memory_debug_interval = float(
            sec.get("memory-debug-interval", "30"))
    except ValueError:
        raise ValueError(
            f"{path.name}: memory-debug-interval must be a number of seconds, "
            f"got {sec.get('memory-debug-interval')!r}"
        ) from None
    if memory_debug_interval < 2.0:
        raise ValueError(
            f"{path.name}: memory-debug-interval must be at least 2 seconds, "
            f"got {memory_debug_interval!r}"
        )
    return (
        primary,
        frozenset(ignored),
        quant,
        te_dev,
        vae_tiling,
        debug,
        save_ui_state,
        mask_overlay_opacity,
        parking_delay,
        prefetch_te,
        memory_debug,
        memory_debug_interval,
    )


def ensure_dirs(config: Config) -> None:
    """Create the data-directory layout if it does not exist yet."""
    for d in (
        config.models_dir,
        config.models_dir / "sdxl",
        config.models_dir / "flux",
        config.models_dir / "zimage",
        config.models_dir / "krea2",
        config.segmentation_models_dir,
        config.loras_dir,
        config.infra_dir,
        config.user_dir,
        config.brushes_dir,
        config.wildcards_dir,
        config.presets_dir,
        config.model_configs_dir,
        config.output_dir,
        config.hf_cache_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
