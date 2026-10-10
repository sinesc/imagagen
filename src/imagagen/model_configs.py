"""Model configurations (user/model-configs/*.json) and default-override resolution.

Every *model* (a .safetensors under models/<type>/) shows up in the model
selector as at least one *configuration*:

- models covered by a file in user/model-configs/ get one entry per file -- the
  user can create several per model (e.g. "My SDXL Model" and "My SDXL Model
  + DMD2 Lora");
- models with no configuration files get a read-only in-memory *default*
  entry (id = model id, all settings null), so they are usable immediately.

Each configuration stores, separately for text-to-image ("t2i") and
image-to-image/inpaint ("i2i"): sampler, scheduler, steps, CFG, a LoRA
list, the self-attention-guidance (SAG) enable flag + options and the
soft-inpainting enable flag + options. Null means "no override" -- the
built-in backend default is used. The LoRA list is special: it is *always*
prepended to the user-selected LoRAs (concept.md), it is not a
default-override. SAG is implemented for the SDXL UNet only (inert for the
other families) and applies to both directions; soft inpainting works for
all four families but is i2i-only (it needs an init mask; the t2i section's
soft keys are inert). Soft inpainting is *enabled by default*: the flag
defaults to true, so new configurations and old configuration files without
the key have it on.

File layout (one JSON per configuration, filename = sanitized name):

    {
      "name": "My SDXL Model + DMD2 Lora",
      "model": "whatever",  # model id: path within models/sdxl, no extension
      "distilled": true,  # Flux only: CFG = embedded guidance (see ModelConfig)
      "turbo": false,  # Krea 2 only: the turbo checkpoint class (see ModelConfig)
      "t2i": {"sampler": null, "scheduler": null, "cfg": null, "sag": false,
              "sag_scale": 0.5, "sag_blur_sigma": 2.0, "sag_threshold": 1.0,
              "soft": true, "soft_mask_blend_power": 1.0,
              "soft_mask_blend_scale": 0.5, "soft_inpaint_detail_preservation": 4.0,
              "soft_composite_mask_influence": 0.0,
              "soft_composite_difference_threshold": 0.5,
              "soft_composite_difference_contrast": 2.0,
              "loras": [{"name": "dmd2", "weight": 0.8}]},  # rel. to loras/<type>/
      "i2i": {"sampler": null, "scheduler": null, "cfg": null, "sag": false,
              "sag_scale": 0.5, "sag_blur_sigma": 2.0, "sag_threshold": 1.0,
              "soft": true, "soft_mask_blend_power": 1.0,
              "soft_mask_blend_scale": 0.5, "soft_inpaint_detail_preservation": 4.0,
              "soft_composite_mask_influence": 0.0,
              "soft_composite_difference_threshold": 0.5,
              "soft_composite_difference_contrast": 2.0,
              "loras": []}
    }
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .paths import sanitize_prompt

# the "no override" sentinels in the *user* settings (form field)
CFG_SENTINEL = -1.0
STEPS_SENTINEL = -1


class LoraRef(BaseModel):
    name: str = Field(min_length=1)  # relative to loras/, e.g. "sdxl/dmd2"
    weight: float = 1.0


class ModeConfig(BaseModel):
    """Defaults for one direction: null = no override (backend built-in)."""

    sampler: str | None = None
    scheduler: str | None = None
    steps: int | None = Field(default=None, ge=1)
    cfg: float | None = Field(default=None, ge=0)
    # self attention guidance (Forge builtin extension) enabled for this
    # direction; SDXL only (inert for the other families). The options
    # carry the Forge slider names/ranges/defaults (the generate request
    # no longer carries them: the configuration is the source of truth)
    sag: bool = False
    sag_scale: float = Field(default=0.5, ge=-2.0, le=5.0)
    sag_blur_sigma: float = Field(default=2.0, ge=0.0, le=10.0)
    sag_threshold: float = Field(default=1.0, ge=0.0, le=4.0)
    # soft inpainting (Forge builtin extension) enabled flag + options.
    # i2i ONLY: soft inpainting needs an init mask, so only the inpaint
    # modes (which read the i2i section) use these keys; the t2i section's
    # soft keys are inert (the UI shows the row in the i2i fieldset only).
    # Implemented for every family (unlike SAG). ENABLED BY DEFAULT: the
    # flag defaults to true, so new configurations and old files without
    # the key have it on. The options carry the Forge slider names/ranges/
    # defaults (like the SAG options: the generate request carries no soft
    # fields, the configuration is the source of truth)
    soft: bool = True
    soft_mask_blend_power: float = Field(default=1.0, ge=0.0, le=8.0)
    soft_mask_blend_scale: float = Field(default=0.5, ge=0.0, le=8.0)
    soft_inpaint_detail_preservation: float = Field(default=4.0, ge=1.0, le=32.0)
    soft_composite_mask_influence: float = Field(default=0.0, ge=0.0, le=1.0)
    soft_composite_difference_threshold: float = Field(default=0.5, ge=0.0, le=8.0)
    soft_composite_difference_contrast: float = Field(default=2.0, ge=0.0, le=8.0)
    loras: list[LoraRef] = Field(default_factory=list)


class ModelConfig(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    model: str = Field(min_length=1)  # model id, e.g. "test-dir/JuggernautXL9.0"
    # Flux only (inert for SDXL): True = the configuration's CFG value is the
    # embedded guidance (distilled, one pass per step, no negative prompt);
    # False = classic CFG with the negative prompt (two passes per step). The
    # semantics apply to both directions, hence entry level, not ModeConfig.
    # Inert for every other family (always False there).
    distilled: bool = True
    # Krea 2 only (inert, always False for other families): True selects the
    # turbo checkpoint class (8 steps, no CFG, mu pinned to 1.15) over the
    # raw class (12 steps, standard CFG 1.5, resolution-derived mu). The
    # user's steps/cfg overrides still apply on top (sentinel -1 resolves to
    # the class's built-in).
    turbo: bool = False
    t2i: ModeConfig = Field(default_factory=ModeConfig)
    i2i: ModeConfig = Field(default_factory=ModeConfig)


class ModelConfigUpsert(ModelConfig):
    """API body for POST /api/model-configs: a ModelConfig plus the optional
    id of the entry being edited (a default entry's id materializes it)."""

    id: str | None = None


@dataclass
class ConfigEntry:
    """What the model selector lists: file-backed (editable) or the
    in-memory default for an unconfigured model."""

    id: str
    name: str
    model_id: str
    t2i: ModeConfig
    i2i: ModeConfig
    is_default: bool
    distilled: bool = True
    turbo: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model_id": self.model_id,
            "distilled": self.distilled,
            "turbo": self.turbo,
            "t2i": self.t2i.model_dump(),
            "i2i": self.i2i.model_dump(),
            "is_default": self.is_default,
        }


class ConfigError(Exception):
    """User-facing model-config error (bad name, missing file, ...)."""


def entry_id(name: str) -> str:
    """File/selector id for a configuration name (lowercase, [a-z0-9-])."""
    return sanitize_prompt(name, max_len=120)


class ConfigStore:
    """Loads/saves user/model-configs/*.json and merges them with the in-memory
    defaults for unconfigured models. Thread-safe (saves are rare)."""

    def __init__(self, config, models: Callable[[], list[dict]]) -> None:
        self.config = config
        self._models = models  # provider: [{"id", "name"}, ...]
        self._lock = threading.RLock()

    # ------------------------------ read ------------------------------
    def _path(self, id_: str) -> Path:
        return self.config.model_configs_dir / f"{id_}.json"

    def load_files(self) -> list[ModelConfig]:
        out: list[ModelConfig] = []
        d = self.config.model_configs_dir
        if not d.is_dir():
            return out
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text("utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise ConfigError(f"cannot read {f.name}: {e}") from e
            try:
                out.append(ModelConfig.model_validate(data))
            except Exception as e:
                raise ConfigError(f"{f.name} is not a valid model config: {e}") from e
        return out

    def entries(self) -> list[ConfigEntry]:
        """All selectable configurations: file configs (sorted by name) plus
        the in-memory default of every model no file config covers."""
        with self._lock:
            files = self.load_files()
            file_ids = [entry_id(c.name) for c in files]
            if len(set(file_ids)) != len(file_ids):
                dupes = {i for i in file_ids if file_ids.count(i) > 1}
                raise ConfigError(f"duplicate configuration name(s): {sorted(dupes)}")
            out = [
                ConfigEntry(id=file_ids[i], name=files[i].name, model_id=files[i].model,
                            distilled=files[i].distilled, turbo=files[i].turbo,
                            t2i=files[i].t2i, i2i=files[i].i2i, is_default=False)
                for i in range(len(files))
            ]
            covered = {c.model for c in files}
            for m in self._models():
                if m["id"] in covered:
                    continue
                # Krea 2 turbo auto-detection: a CivitAI download named
                # "...turbo..." gets the turbo class (8 steps, no CFG,
                # mu 1.15) with zero setup; an explicit ModelConfig entry
                # always overrides (the file branch above wins by the
                # `covered` check)
                turbo = (m.get("family") == "krea2"
                         and "turbo" in m["name"].lower())
                out.append(
                    ConfigEntry(id=m["id"], name=m["name"], model_id=m["id"],
                                distilled=True,  # = current Flux behavior
                                turbo=turbo,
                                t2i=ModeConfig(), i2i=ModeConfig(), is_default=True)
                )
            out.sort(key=lambda e: e.name.lower())
            return out

    def get(self, id_: str) -> ConfigEntry | None:
        for e in self.entries():
            if e.id == id_:
                return e
        return None

    def resolve(self, id_: str) -> ConfigEntry:
        e = self.get(id_)
        if e is None:
            raise ConfigError(f"unknown model configuration: {id_!r}")
        return e

    # ----------------------------- write ------------------------------
    def save(self, mc: ModelConfig, id_: str | None = None) -> ConfigEntry:
        """Create or update a file configuration. ``id_`` defaults to the
        sanitized name; a differing id renames the file (name changed).
        Saving with the id of an in-memory *default* entry materializes it
        into a file configuration under the (possibly new) name; the model
        then becomes covered by the file and the default entry disappears."""
        with self._lock:
            materialized = None
            if id_ is not None and not self._path(id_).is_file():
                e = self.get(id_)
                if e is not None and e.is_default:
                    materialized = id_
                    id_ = None
            id_ = id_ or entry_id(mc.name)
            self.config.model_configs_dir.mkdir(parents=True, exist_ok=True)
            target = self._path(id_)
            if not target.is_file() and self.get(id_) is not None:
                raise ConfigError(f"configuration id already in use: {id_!r}")
            for e in self.entries():
                # on a plain update, skip the entry being updated; on a
                # materialization the target (if a file exists) must still
                # take part in the collision check
                if e.id == id_ and materialized is None:
                    continue
                if e.id == materialized:
                    continue
                if e.name.lower() == mc.name.lower() or e.id == entry_id(mc.name):
                    raise ConfigError(f"configuration name already in use: {mc.name!r}")
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(mc.model_dump(), indent=2, ensure_ascii=False) + "\n",
                           "utf-8")
            tmp.replace(target)
            if id_ != entry_id(mc.name):
                # name changed without the caller tracking the id: normalize
                # the filename to the new name if it is free
                fresh = self._path(entry_id(mc.name))
                if not fresh.is_file():
                    target.replace(fresh)
                    id_ = entry_id(mc.name)
            return self.resolve(id_)

    def create_from(self, name: str, based_on: str) -> ConfigEntry:
        """Create a new configuration copying the settings of ``based_on``
        (concept.md: create = copy of the currently selected configuration)."""
        src = self.resolve(based_on)
        return self.save(
            ModelConfig(name=name, model=src.model_id, distilled=src.distilled,
                        turbo=src.turbo,
                        t2i=src.t2i.model_copy(deep=True), i2i=src.i2i.model_copy(deep=True)),
        )

    def delete(self, id_: str) -> None:
        with self._lock:
            e = self.get(id_)
            if e is None:
                raise ConfigError(f"unknown model configuration: {id_!r}")
            if e.is_default:
                raise ConfigError(
                    f"{e.name!r} is the in-memory default configuration and "
                    "cannot be deleted"
                )
            self._path(id_).unlink()


# ---------------------------------------------------------------- resolve --


def mode_section(entry: ConfigEntry, mode: str) -> ModeConfig:
    """The configuration section for a generation mode: t2i for
    "generate", i2i for the inpaint modes."""
    return entry.t2i if mode == "generate" else entry.i2i


def resolve_settings(
    entry: ConfigEntry,
    mode: str,
    user_cfg: float,
    user_sampler: str | None,
    user_scheduler: str | None,
    user_loras: list[tuple[str, float]] | None = None,
    user_steps: int = STEPS_SENTINEL,
) -> tuple[str | None, str | None, float | None, int | None, list[tuple[str, float]]]:
    """Resolve the effective generation settings.

    Order per setting (concept.md): the user's setting wins unless it is the
    "use model default" sentinel (cfg == -1, steps == -1,
    sampler/scheduler == null), in which case the *configuration's* value is
    used; null configuration values fall through to the backend built-in
    (returned as None / None, the caller substitutes the built-in default).
    The configuration's LoRA list is always *prepended* to the user's LoRAs.
    ``mode`` picks the t2i section for "generate" and the i2i section for
    the inpaint modes.

    Returns ``(sampler, scheduler, cfg, steps, loras, sag)`` with cfg/steps
    possibly None (= backend built-in) and ``sag`` the configuration's
    self-attention-guidance flag for the picked section. The SAG *options*
    are read from the section directly (``mode_section``) -- they have no
    user-level override, so they are not part of the resolution."""
    section = mode_section(entry, mode)
    sampler = user_sampler if user_sampler else section.sampler
    scheduler = user_scheduler if user_scheduler else section.scheduler
    cfg = section.cfg if user_cfg == CFG_SENTINEL else user_cfg
    steps = section.steps if user_steps == STEPS_SENTINEL else user_steps
    loras = [(ref.name, ref.weight) for ref in section.loras] + list(user_loras or [])
    return sampler, scheduler, cfg, steps, loras, section.sag
