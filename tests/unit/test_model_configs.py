"""Model configurations: file layout, entry merging, CRUD, default-override
resolution order (user -> config -> backend built-in)."""

import json

import pytest

from imagagen.config import Config
from imagagen.model_configs import (
    CFG_SENTINEL,
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
    f = tmp_path / "model-configs" / "my-a-tuned.json"
    assert f.is_file()
    on_disk = json.loads(f.read_text())
    assert on_disk["name"] == "My A tuned"
    assert on_disk["model"] == "sdxl/a.safetensors"
    assert on_disk["t2i"]["loras"] == [{"name": "sdxl/dmd2", "weight": 0.8}]
    assert on_disk["i2i"] == {"sampler": None, "scheduler": None,
                              "cfg": None, "loras": []}


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
    assert (tmp_path / "model-configs" / "a.json").is_file()
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
    assert not (tmp_path / "model-configs" / "base.json").exists()
    assert {e.id for e in s.entries()} == {"sdxl/a.safetensors", "sdxl/b.safetensors"}
    with pytest.raises(ConfigError):
        s.delete("base")


def test_corrupt_file_is_reported(tmp_path):
    (tmp_path / "model-configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "model-configs" / "bad.json").write_text("{nope")
    with pytest.raises(ConfigError, match="bad.json"):
        store(tmp_path).entries()


# ---------------------------------------------------------- resolution ----


def _entry(sampler=None, scheduler=None, cfg=None, loras=(), i2i=None):
    from imagagen.model_configs import ConfigEntry

    i2i = i2i or ModeConfig()
    return ConfigEntry(id="x", name="x", model_id="m",
                       t2i=ModeConfig(sampler=sampler, scheduler=scheduler,
                                      cfg=cfg,
                                      loras=[LoraRef(name=t[0], weight=t[1])
                                              for t in loras]),
                       i2i=i2i, is_default=False)


def test_resolve_user_wins():
    e = _entry(sampler="Euler a", scheduler="Karras", cfg=4.0, loras=[("a", 1)])
    sampler, scheduler, cfg, loras = resolve_settings(
        e, "generate", 7.5, "DPM++ 2M", "Polyexponential", [("b", 0.5)])
    assert (sampler, scheduler, cfg) == ("DPM++ 2M", "Polyexponential", 7.5)
    # config loras are prepended to the user's list
    assert loras == [("a", 1.0), ("b", 0.5)]


def test_resolve_defaults_fall_to_config():
    e = _entry(sampler="Euler a", scheduler="Karras", cfg=4.0, loras=[("a", 1)])
    sampler, scheduler, cfg, loras = resolve_settings(
        e, "generate", CFG_SENTINEL, None, None, [])
    assert (sampler, scheduler, cfg) == ("Euler a", "Karras", 4.0)
    assert loras == [("a", 1.0)]


def test_resolve_null_config_falls_to_backend_builtin():
    e = _entry()
    sampler, scheduler, cfg, loras = resolve_settings(
        e, "generate", CFG_SENTINEL, None, None)
    assert (sampler, scheduler, cfg, loras) == (None, None, None, [])


def test_resolve_mode_selects_section():
    e = _entry(cfg=4.0, i2i=ModeConfig(sampler="DPM++ 2M SDE", cfg=6.0))
    # inpaint modes use the i2i section
    sampler, _, cfg, _ = resolve_settings(e, "inpaint_masked", CFG_SENTINEL, None, None)
    assert (sampler, cfg) == ("DPM++ 2M SDE", 6.0)
    _, _, cfg_t2i, _ = resolve_settings(e, "generate", CFG_SENTINEL, None, None)
    assert cfg_t2i == 4.0


def test_entry_id_sanitizes():
    assert entry_id("My SDXL Model + DMD2 Lora") == "my-sdxl-model-dmd2-lora"
