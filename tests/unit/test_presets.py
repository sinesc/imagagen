"""Setting presets: file layout, CRUD, controls filtering."""

import json

import pytest

from imagagen.presets import PRESET_FIELDS, Preset, PresetStore


def test_save_list_get_delete(tmp_path):
    s = PresetStore(tmp_path / "presets")
    assert s.names() == []
    p = Preset(name="Big square", width=1024, height=1024,
               controls={"width": True, "height": True, "steps": False})
    id_ = s.save(p)
    assert id_ == "big-square"
    assert (tmp_path / "presets" / "big-square.json").is_file()
    assert s.names() == ["Big square"]
    got = s.get("big-square")
    assert got.name == "Big square"
    assert s.delete("Big square")
    assert s.names() == []
    assert not s.delete("Big square")


def test_file_layout(tmp_path):
    s = PresetStore(tmp_path / "presets")
    s.save(Preset(name="Big square", width=1024, height=1024, steps=30,
                  loras=[("sdxl/dmd2", 0.8)],
                  controls={"width": True, "height": True, "steps": True,
                            "loras": True}))
    on_disk = json.loads((tmp_path / "presets" / "big-square.json").read_text())
    assert on_disk["name"] == "Big square"
    assert on_disk["width"] == 1024
    assert on_disk["steps"] == 30
    assert on_disk["loras"] == [["sdxl/dmd2", 0.8]]
    assert on_disk["cfg"] is None  # unset settings stay null
    assert on_disk["controls"]["width"] is True
    assert on_disk["controls"]["seed"] is False  # normalized to full field list


def test_sampler_scheduler_empty_string_roundtrip(tmp_path):
    """sampler/scheduler "" means "use model default" (the main UI's empty
    option) and must survive the round trip, unlike null (unset)."""
    s = PresetStore(tmp_path / "presets")
    s.save(Preset(name="Reset", sampler="", scheduler="", mode=None,
                  controls={"sampler": True, "scheduler": True}))
    got = s.get("reset")
    assert got.sampler == ""
    assert got.scheduler == ""
    assert got.controlled() == {"sampler": "", "scheduler": ""}
    with pytest.raises(ValueError):
        Preset(name="x", mode="")  # mode has no empty state


def test_get_by_display_name_or_id(tmp_path):
    s = PresetStore(tmp_path / "presets")
    s.save(Preset(name="Set defaults", steps=-1, cfg=-1.0))
    assert s.get("set-defaults") is not None   # sanitized id
    assert s.get("Set defaults") is not None   # display name
    assert s.get("set-defaults") == s.get("Set defaults")
    assert s.get("missing") is None


def test_controlled_returns_only_checked_settings(tmp_path):
    s = PresetStore(tmp_path / "presets")
    p = Preset(name="Size only", width=512, height=768, steps=99,
               controls={"width": True, "height": True})
    s.save(p)
    got = s.get("size-only")
    assert got.controlled() == {"width": 512, "height": 768}


def test_group_field(tmp_path):
    s = PresetStore(tmp_path / "presets")
    s.save(Preset(name="A", group="Touchup"))
    s.save(Preset(name="B"))  # ungrouped
    # the list API payload: names keep file order, group null when unset
    assert [(p.name, p.group) for p in s.all()] == [("A", "Touchup"), ("B", None)]
    got = s.get("a")
    assert got.group == "Touchup"
    on_disk = json.loads((tmp_path / "presets" / "a.json").read_text())
    assert on_disk["group"] == "Touchup"
    # "" is normalized to None (the UI sends null, be lenient)
    assert Preset(name="x", group="").group is None
    assert Preset(name="x", group=None).group is None
    with pytest.raises(ValueError):
        Preset(name="x", group="y" * 121)


def test_validation(tmp_path):
    s = PresetStore(tmp_path / "presets")
    # pydantic ValidationError is a ValueError subclass
    with pytest.raises(ValueError):
        s.save(Preset(name="", width=1024))
    with pytest.raises(ValueError):
        s.save(Preset(name="x", width=9999))
    with pytest.raises(ValueError):
        s.save(Preset(name="x", mode="bogus"))


def test_model_default_sentinels_allowed(tmp_path):
    """steps/cfg of -1 mean "use model default" (same as the form), so a
    preset must be able to carry them."""
    s = PresetStore(tmp_path / "presets")
    p = Preset(name="Set defaults", steps=-1, cfg=-1.0,
               controls={"steps": True, "cfg": True})
    s.save(p)
    got = s.get("set-defaults")
    assert got.controlled() == {"steps": -1, "cfg": -1.0}
    with pytest.raises(ValueError):
        Preset(name="x", steps=-2)
    with pytest.raises(ValueError):
        Preset(name="x", cfg=-1.5)


def test_unknown_keys_ignored_on_save(tmp_path):
    s = PresetStore(tmp_path / "presets")
    p = Preset.model_validate({"name": "p", "width": 64,
                               "controls": {"width": True, "bogus": True}})
    s.save(p)
    on_disk = json.loads((tmp_path / "presets" / "p.json").read_text())
    assert "bogus" not in on_disk["controls"]
    assert set(PRESET_FIELDS) >= set(on_disk["controls"])
