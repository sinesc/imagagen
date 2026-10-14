"""Model configurations: file layout, entry merging, CRUD, default-override
resolution order (user -> config -> backend built-in)."""

import json

import pytest
from pydantic import ValidationError

from imagagen.config import Config
from imagagen.model_configs import (
    CFG_SENTINEL,
    STEPS_SENTINEL,
    ConfigError,
    ConfigStore,
    LoraRef,
    ModeConfig,
    ModelConfig,
    entry_id,
    match_entry,
    resolve_settings,
)

MOCKS = [{"id": "sdxl/a.safetensors", "name": "A"},
          {"id": "sdxl/b.safetensors", "name": "B"}]


def store(tmp_path) -> ConfigStore:
    return ConfigStore(Config(root=tmp_path), lambda: MOCKS)


def test_default_entries_for_unconfigured_models(tmp_path):
    s = store(tmp_path)
    entries = s.entries()
    assert [(e.id, e.is_default) for e in entries] == [
        ("sdxl/a.safetensors", True), ("sdxl/b.safetensors", True)]
    assert all(e.t2i == ModeConfig() and e.i2i == ModeConfig() for e in entries)


def test_turbo_auto_detection_for_krea2_defaults(tmp_path):
    """An unconfigured Krea 2 model whose name contains "turbo" gets the
    turbo class (8 steps / no CFG / mu 1.15) with zero setup; other
    families and non-turbo names stay raw; an explicit file config
    overrides the auto-detection."""
    mocks = [{"id": "krea2/krea2_finetuned_turbo", "name": "krea2_finetuned_turbo",
              "family": "krea2"},
             {"id": "krea2/krea2_Int8Extended", "name": "krea2_Int8Extended",
              "family": "krea2"},
             {"id": "flux/FLUX1-turbo", "name": "FLUX1-turbo", "family": "flux"}]
    s = ConfigStore(Config(root=tmp_path), lambda: mocks)
    by_id = {e.id: e for e in s.entries()}
    assert by_id["krea2/krea2_finetuned_turbo"].turbo is True
    assert by_id["krea2/krea2_Int8Extended"].turbo is False
    assert by_id["flux/FLUX1-turbo"].turbo is False  # krea2-only flag
    # the explicit config wins (turbo False on a turbo-named model)
    s.save(ModelConfig(name="Raw run", model="krea2/krea2_finetuned_turbo",
                       turbo=False))
    by_id = {e.id: e for e in s.entries()}
    assert by_id["raw-run"].is_default is False
    assert by_id["raw-run"].turbo is False


def test_file_config_replaces_default_of_its_model(tmp_path):
    s = store(tmp_path)
    s.save(ModelConfig(name="My A tuned", model="sdxl/a.safetensors",
                       t2i=ModeConfig(sampler="DPM++ 2M", cfg=8.0)))
    by_id = {e.id: e for e in s.entries()}
    assert set(by_id) == {"my-a-tuned", "sdxl/b.safetensors"}
    # model A is covered by the file config -> no default entry for it
    assert by_id["my-a-tuned"].is_default is False
    assert by_id["sdxl/b.safetensors"].is_default is True
    e = by_id["my-a-tuned"]
    assert e.t2i.sampler == "DPM++ 2M"
    assert e.t2i.cfg == 8.0
    assert e.i2i.sampler is None


def test_file_layout(tmp_path):
    s = store(tmp_path)
    s.save(ModelConfig(name="My A tuned", model="sdxl/a.safetensors",
                       t2i=ModeConfig(loras=[LoraRef(name="sdxl/dmd2", weight=0.8)])))
    f = tmp_path / "user" / "model-configs" / "my-a-tuned.json"
    assert f.is_file()
    on_disk = json.loads(f.read_text())
    assert on_disk["name"] == "My A tuned"
    assert on_disk["model"] == "sdxl/a.safetensors"
    assert on_disk["t2i"]["loras"] == [{"name": "sdxl/dmd2", "weight": 0.8}]
    assert on_disk["t2i"]["sag"] is False
    assert on_disk["i2i"] == {"sampler": None, "scheduler": None,
                              "steps": None, "cfg": None, "sag": False,
                              "sag_scale": 0.5, "sag_blur_sigma": 2.0,
                              "sag_threshold": 1.0,
                              "soft": True, "soft_mask_blend_power": 1.0,
                              "soft_mask_blend_scale": 0.5,
                              "soft_inpaint_detail_preservation": 4.0,
                              "soft_composite_mask_influence": 0.0,
                              "soft_composite_difference_threshold": 0.5,
                              "soft_composite_difference_contrast": 2.0,
                              "loras": []}


def test_soft_flag_defaults_enabled_and_roundtrips(tmp_path):
    # soft inpainting is ENABLED by default (new configs and old files
    # without the keys); the keys live in both sections (shared model) but
    # only the i2i section is ever read (soft inpainting needs an init
    # mask), the UI shows the row in the i2i fieldset only
    s = store(tmp_path)
    s.save(ModelConfig(name="Plain", model="sdxl/a.safetensors"))
    e = s.resolve("plain")
    assert e.t2i.soft is True and e.i2i.soft is True
    s.save(ModelConfig(name="Soft off", model="sdxl/a.safetensors",
                       i2i=ModeConfig(soft=False, soft_mask_blend_power=3.0)))
    e = s.resolve("soft-off")
    assert e.t2i.soft is True and e.i2i.soft is False
    assert e.i2i.soft_mask_blend_power == 3.0
    assert e.t2i.soft_mask_blend_power == 1.0  # Forge default
    # an old file without any soft keys loads with the defaults (enabled)
    cfg = tmp_path / "user" / "model-configs" / "soft-off.json"
    data = json.loads(cfg.read_text())
    for section in ("t2i", "i2i"):
        data[section] = {k: v for k, v in data[section].items()
                         if not k.startswith("soft")}
    cfg.write_text(json.dumps(data))
    e2 = store(tmp_path).resolve("soft-off")
    assert e2.t2i.soft is True and e2.i2i.soft is True
    assert e2.i2i.soft_mask_blend_power == 1.0
    # outside the Forge slider ranges -> rejected
    with pytest.raises(ValidationError):
        store(tmp_path).save(ModelConfig(
            name="Bad", model="sdxl/a.safetensors",
            i2i=ModeConfig(soft_mask_blend_power=99.0)))


def test_sag_flag_and_options_per_direction(tmp_path):
    # the enable flag and the option values are per direction (t2i/i2i)
    # and round-trip through the configuration file
    s = store(tmp_path)
    s.save(ModelConfig(name="SAG on", model="sdxl/a.safetensors",
                       t2i=ModeConfig(sag=True, sag_scale=1.25,
                                      sag_blur_sigma=5.0)))
    e = s.resolve("sag-on")
    assert e.t2i.sag is True and e.i2i.sag is False
    # t2i carries the saved values, i2i the Forge defaults
    assert (e.t2i.sag_scale, e.t2i.sag_blur_sigma, e.t2i.sag_threshold) == \
        (1.25, 5.0, 1.0)
    assert (e.i2i.sag_scale, e.i2i.sag_blur_sigma, e.i2i.sag_threshold) == \
        (0.5, 2.0, 1.0)
    on_disk = json.loads(
        (tmp_path / "user" / "model-configs" / "sag-on.json").read_text())
    assert on_disk["t2i"]["sag"] is True and on_disk["i2i"]["sag"] is False
    assert on_disk["t2i"]["sag_scale"] == 1.25
    # survives a reload from disk (the store reads the files fresh)
    e2 = store(tmp_path).resolve("sag-on")
    assert e2.t2i.sag is True and e2.i2i.sag is False
    assert e2.t2i.sag_scale == 1.25
    # outside the Forge slider ranges -> rejected
    with pytest.raises(ValidationError):
        store(tmp_path).save(ModelConfig(
            name="Bad", model="sdxl/a.safetensors",
            t2i=ModeConfig(sag=True, sag_scale=5.5)))


def test_distilled_flag_default_and_roundtrip(tmp_path):
    # entry-level flag (not per-mode): defaults True (the historical Flux
    # behavior) for file and in-memory default entries alike
    s = store(tmp_path)
    assert all(e.distilled is True for e in s.entries())
    assert all(
        e.to_dict()["distilled"] is True for e in s.entries())
    s.save(ModelConfig(name="My A", model="sdxl/a.safetensors", distilled=False))
    e = s.resolve("my-a")
    assert e.distilled is False
    # survives a reload from disk (the store reads the files fresh)
    assert store(tmp_path).resolve("my-a").distilled is False
    on_disk = json.loads(
        (tmp_path / "user" / "model-configs" / "my-a.json").read_text())
    assert on_disk["distilled"] is False
    # t2i/i2i share the flag (it is not part of the mode sections)
    assert "distilled" not in on_disk["t2i"] and "distilled" not in on_disk["i2i"]
    # create_from copies it
    e2 = s.create_from("My A copy", "my-a")
    assert e2.distilled is False


def test_save_updates_existing_and_rejects_collisions(tmp_path):
    s = store(tmp_path)
    s.save(ModelConfig(name="Tuned", model="sdxl/a.safetensors"))
    # same id (sanitized name) -> update in place
    s.save(ModelConfig(name="Tuned", model="sdxl/a.safetensors",
                       i2i=ModeConfig(cfg=5.5)))
    assert s.resolve("tuned").i2i.cfg == 5.5
    # a name sanitizing to an existing id is the same file -> in-place
    # update (there is no way to tell them apart anyway)
    s.save(ModelConfig(name="Tuned!", model="sdxl/a.safetensors"))
    assert s.resolve("tuned").name == "Tuned!"
    # a config named after an existing entry (here the default entry of
    # model B) is a collision
    with pytest.raises(ConfigError):
        s.save(ModelConfig(name="B", model="sdxl/a.safetensors"))


def test_save_with_default_id_materializes_it(tmp_path):
    s = store(tmp_path)
    # saving with the in-memory default's id turns it into a file config
    e = s.save(ModelConfig(name="A", model="sdxl/a.safetensors",
                           t2i=ModeConfig(cfg=6.0)),
               id_="sdxl/a.safetensors")
    assert (tmp_path / "user" / "model-configs" / "a.json").is_file()
    by_id = {x.id: x for x in s.entries()}
    # model A is covered now -> the default entry is gone
    assert set(by_id) == {"a", "sdxl/b.safetensors"}
    assert by_id["a"].is_default is False
    assert by_id["a"].t2i.cfg == 6.0
    # editing the materialized config again is a plain file update
    e2 = s.save(ModelConfig(name="A", model="sdxl/a.safetensors",
                            t2i=ModeConfig(cfg=7.5)), id_="a")
    assert e2.t2i.cfg == 7.5


def test_save_with_default_id_can_rename(tmp_path):
    s = store(tmp_path)
    e = s.save(ModelConfig(name="A tuned", model="sdxl/a.safetensors"),
               id_="sdxl/a.safetensors")
    assert e.id == "a-tuned"
    # the old default (name "A") is gone: the model is covered
    assert {x.name for x in s.entries()} == {"A tuned", "B"}


def test_create_from_copies_settings(tmp_path):
    s = store(tmp_path)
    s.save(ModelConfig(name="Base", model="sdxl/a.safetensors",
                       t2i=ModeConfig(sampler="Euler a", cfg=4.0,
                                      loras=[LoraRef(name="sdxl/x", weight=0.5)])))
    e = s.create_from("Base + Lora", "base")
    assert e.id == "base-lora"
    assert e.model_id == "sdxl/a.safetensors"
    assert e.t2i.cfg == 4.0
    # the copy is a copy, not a shared reference
    e2 = s.resolve("base")
    e2.t2i.cfg = 99.0
    assert s.resolve("base-lora").t2i.cfg == 4.0


def test_delete(tmp_path):
    s = store(tmp_path)
    s.save(ModelConfig(name="Base", model="sdxl/a.safetensors"))
    with pytest.raises(ConfigError):
        s.delete("sdxl/a.safetensors")  # in-memory default: not deletable
    s.delete("base")
    assert not (tmp_path / "user" / "model-configs" / "base.json").exists()
    assert {e.id for e in s.entries()} == {"sdxl/a.safetensors", "sdxl/b.safetensors"}
    with pytest.raises(ConfigError):
        s.delete("base")


def test_corrupt_file_is_reported(tmp_path):
    (tmp_path / "user" / "model-configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "user" / "model-configs" / "bad.json").write_text("{nope")
    with pytest.raises(ConfigError, match="bad.json"):
        store(tmp_path).entries()


# ---------------------------------------------------------- resolution ----


def _entry(sampler=None, scheduler=None, cfg=None, loras=(), i2i=None, steps=None,
           sag=False):
    from imagagen.model_configs import ConfigEntry

    i2i = i2i or ModeConfig()
    return ConfigEntry(id="x", name="x", model_id="m",
                       t2i=ModeConfig(sampler=sampler, scheduler=scheduler,
                                      steps=steps, cfg=cfg, sag=sag,
                                      loras=[LoraRef(name=t[0], weight=t[1])
                                              for t in loras]),
                       i2i=i2i, is_default=False)


def test_resolve_user_wins():
    e = _entry(sampler="Euler a", scheduler="Karras", cfg=4.0, loras=[("a", 1)], steps=12)
    sampler, scheduler, cfg, steps, loras, sag = resolve_settings(
        e, "generate", 7.5, "DPM++ 2M", "Polyexponential", [("b", 0.5)], 25)
    assert (sampler, scheduler, cfg, steps) == ("DPM++ 2M", "Polyexponential", 7.5, 25)
    # config loras are prepended to the user's list
    assert loras == [("a", 1.0), ("b", 0.5)]
    assert sag is False


def test_resolve_defaults_fall_to_config():
    e = _entry(sampler="Euler a", scheduler="Karras", cfg=4.0, loras=[("a", 1)], steps=12)
    sampler, scheduler, cfg, steps, loras, sag = resolve_settings(
        e, "generate", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (sampler, scheduler, cfg, steps) == ("Euler a", "Karras", 4.0, 12)
    assert loras == [("a", 1.0)]
    assert sag is False


def test_resolve_null_config_falls_to_backend_builtin():
    e = _entry()
    sampler, scheduler, cfg, steps, loras, sag = resolve_settings(
        e, "generate", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (sampler, scheduler, cfg, steps, loras, sag) == (None, None, None, None, [], False)


def test_resolve_steps_sentinel_falls_to_config():
    # -1 in the main UI -> the configuration's steps; a null config value
    # falls through to the backend built-in (None), same logic as CFG
    e = _entry(steps=18)
    _, _, _, steps, _, _ = resolve_settings(e, "generate", 7.0, None, None, [], STEPS_SENTINEL)
    assert steps == 18
    _, _, _, steps, _, _ = resolve_settings(e, "generate", 7.0, None, None, [], 30)
    assert steps == 30
    e_null = _entry()
    _, _, _, steps, _, _ = resolve_settings(e_null, "generate", 7.0, None, None, [], STEPS_SENTINEL)
    assert steps is None


def test_resolve_mode_selects_section():
    e = _entry(cfg=4.0, steps=10, sag=False,
               i2i=ModeConfig(sampler="DPM++ 2M SDE", cfg=6.0, steps=14, sag=True))
    # inpaint modes use the i2i section (including the SAG flag)
    sampler, _, cfg, steps, _, sag = resolve_settings(e, "inpaint_masked", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (sampler, cfg, steps) == ("DPM++ 2M SDE", 6.0, 14)
    assert sag is True
    _, _, cfg_t2i, steps_t2i, _, sag_t2i = resolve_settings(e, "generate", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (cfg_t2i, steps_t2i) == (4.0, 10)
    assert sag_t2i is False


def test_entry_id_sanitizes():
    assert entry_id("My SDXL Model + DMD2 Lora") == "my-sdxl-model-dmd2-lora"


# ---------------- import matching (PNG metadata -> configuration) ----------


def _import_store(tmp_path):
    """epicrealismXL11.0 with a bare-checkpoint config and a +DMD2@0.7 one,
    plus an unconfigured model (in-memory default entry)."""
    s = ConfigStore(Config(root=tmp_path), lambda: [
        {"id": "sdxl/epicrealismXL11.0", "name": "epicrealismXL11.0"},
        {"id": "sdxl/other", "name": "other"},
    ])
    s.save(ModelConfig(name="epicrealismXL11.0 (plain)", model="sdxl/epicrealismXL11.0"))
    s.save(ModelConfig(
        name="epicrealismXL11.0 (DMD)", model="sdxl/epicrealismXL11.0",
        t2i=ModeConfig(loras=[LoraRef(name="DMD2fp32XL1.0", weight=0.7)]),
        i2i=ModeConfig(loras=[LoraRef(name="DMD2fp32XL1.0", weight=0.7)])))
    return s


def test_match_prefers_the_lora_matching_configuration(tmp_path):
    """A Forge image made with the checkpoint + DMD2@0.7 picks the config
    that provides that LoRA, not the bare-checkpoint config."""
    e = _import_store(tmp_path)
    m = match_entry(e.entries(), model="epicrealismXL11.0",
                    loras=[("DMD2fp32XL1.0", 0.7)])
    assert m is not None and m.name == "epicrealismXL11.0 (DMD)"


def test_match_without_loras_picks_the_bare_configuration(tmp_path):
    m = match_entry(_import_store(tmp_path).entries(), model="epicrealismXL11.0",
                    loras=[])
    assert m is not None and m.name == "epicrealismXL11.0 (plain)"


def test_match_weight_mismatch_falls_back_to_the_bare_configuration(tmp_path):
    # the image used 0.8, the config carries 0.7: the LoRA must stay a UI
    # selection, so the bare-checkpoint config is the better match
    m = match_entry(_import_store(tmp_path).entries(), model="epicrealismXL11.0",
                    loras=[("DMD2fp32XL1.0", 0.8)])
    assert m is not None and m.name == "epicrealismXL11.0 (plain)"


def test_match_partial_lora_overlap(tmp_path):
    # image: DMD2@0.7 + an extra UI LoRA -> the DMD config still covers the
    # most LoRAs, so it wins over the bare one
    m = match_entry(_import_store(tmp_path).entries(), model="epicrealismXL11.0",
                    loras=[("DMD2fp32XL1.0", 0.7), ("extra", 1.0)])
    assert m is not None and m.name == "epicrealismXL11.0 (DMD)"


def test_match_config_name_wins_and_survives_scoring(tmp_path):
    # imagagen PNGs record the configuration name: it is selected verbatim,
    # even when the LoRA list would score the bare config higher
    m = match_entry(_import_store(tmp_path).entries(), model="epicrealismXL11.0",
                    loras=[("DMD2fp32XL1.0", 0.8)],
                    config_name="epicrealismXL11.0 (DMD)")
    assert m is not None and m.name == "epicrealismXL11.0 (DMD)"


def test_match_deleted_config_name_falls_back_to_scoring(tmp_path):
    s = _import_store(tmp_path)
    s.delete(entry_id("epicrealismXL11.0 (DMD)"))
    m = match_entry(s.entries(), model="epicrealismXL11.0",
                    loras=[("DMD2fp32XL1.0", 0.7)],
                    config_name="epicrealismXL11.0 (DMD)")
    assert m is not None and m.name == "epicrealismXL11.0 (plain)"


def test_match_normalizes_foreign_model_names(tmp_path):
    """A1111/Forge write the checkpoint basename (extension stripped, some
    versions append " [hash]"); imagagen writes the model id (a path)."""
    entries = _import_store(tmp_path).entries()
    for name in ("epicrealismXL11.0", "epicrealismXL11.0.safetensors",
                 "epicrealismXL11.0 [c8f1a2b3]", "sdxl/epicrealismXL11.0",
                 "Epicrealismxl11.0"):
        m = match_entry(entries, model=name, loras=[("DMD2fp32XL1.0", 0.7)])
        assert m is not None and m.name == "epicrealismXL11.0 (DMD)", name


def test_match_unknown_model_returns_none(tmp_path):
    assert match_entry(_import_store(tmp_path).entries(), model="nope",
                       loras=[]) is None
    assert match_entry(_import_store(tmp_path).entries(), model=None) is None


def test_match_prefers_file_config_over_default_on_a_tie(tmp_path):
    # only the in-memory default covers the model -> it is the match
    s = ConfigStore(Config(root=tmp_path), lambda: MOCKS)
    m = match_entry(s.entries(), model="A", loras=[])
    assert m is not None and m.id == "sdxl/a.safetensors" and m.is_default
