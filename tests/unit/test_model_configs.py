"""Model configurations: file layout, entry merging, CRUD, default-override
resolution order (user -> config -> backend built-in)."""

import json

import pytest

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
    assert on_disk["i2i"] == {"sampler": None, "scheduler": None,
                              "steps": None, "cfg": None, "loras": []}


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


def _entry(sampler=None, scheduler=None, cfg=None, loras=(), i2i=None, steps=None):
    from imagagen.model_configs import ConfigEntry

    i2i = i2i or ModeConfig()
    return ConfigEntry(id="x", name="x", model_id="m",
                       t2i=ModeConfig(sampler=sampler, scheduler=scheduler,
                                      steps=steps, cfg=cfg,
                                      loras=[LoraRef(name=t[0], weight=t[1])
                                              for t in loras]),
                       i2i=i2i, is_default=False)


def test_resolve_user_wins():
    e = _entry(sampler="Euler a", scheduler="Karras", cfg=4.0, loras=[("a", 1)], steps=12)
    sampler, scheduler, cfg, steps, loras = resolve_settings(
        e, "generate", 7.5, "DPM++ 2M", "Polyexponential", [("b", 0.5)], 25)
    assert (sampler, scheduler, cfg, steps) == ("DPM++ 2M", "Polyexponential", 7.5, 25)
    # config loras are prepended to the user's list
    assert loras == [("a", 1.0), ("b", 0.5)]


def test_resolve_defaults_fall_to_config():
    e = _entry(sampler="Euler a", scheduler="Karras", cfg=4.0, loras=[("a", 1)], steps=12)
    sampler, scheduler, cfg, steps, loras = resolve_settings(
        e, "generate", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (sampler, scheduler, cfg, steps) == ("Euler a", "Karras", 4.0, 12)
    assert loras == [("a", 1.0)]


def test_resolve_null_config_falls_to_backend_builtin():
    e = _entry()
    sampler, scheduler, cfg, steps, loras = resolve_settings(
        e, "generate", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (sampler, scheduler, cfg, steps, loras) == (None, None, None, None, [])


def test_resolve_steps_sentinel_falls_to_config():
    # -1 in the main UI -> the configuration's steps; a null config value
    # falls through to the backend built-in (None), same logic as CFG
    e = _entry(steps=18)
    _, _, _, steps, _ = resolve_settings(e, "generate", 7.0, None, None, [], STEPS_SENTINEL)
    assert steps == 18
    _, _, _, steps, _ = resolve_settings(e, "generate", 7.0, None, None, [], 30)
    assert steps == 30
    e_null = _entry()
    _, _, _, steps, _ = resolve_settings(e_null, "generate", 7.0, None, None, [], STEPS_SENTINEL)
    assert steps is None


def test_resolve_mode_selects_section():
    e = _entry(cfg=4.0, steps=10, i2i=ModeConfig(sampler="DPM++ 2M SDE", cfg=6.0, steps=14))
    # inpaint modes use the i2i section
    sampler, _, cfg, steps, _ = resolve_settings(e, "inpaint_masked", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (sampler, cfg, steps) == ("DPM++ 2M SDE", 6.0, 14)
    _, _, cfg_t2i, steps_t2i, _ = resolve_settings(e, "generate", CFG_SENTINEL, None, None, [], STEPS_SENTINEL)
    assert (cfg_t2i, steps_t2i) == (4.0, 10)


def test_entry_id_sanitizes():
    assert entry_id("My SDXL Model + DMD2 Lora") == "my-sdxl-model-dmd2-lora"
