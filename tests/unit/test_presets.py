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


def test_controlled_returns_only_checked_settings(tmp_path):
    s = PresetStore(tmp_path / "presets")
    p = Preset(name="Size only", width=512, height=768, steps=99,
               controls={"width": True, "height": True})
    s.save(p)
    got = s.get("size-only")
    assert got.controlled() == {"width": 512, "height": 768}


def test_validation(tmp_path):
    s = PresetStore(tmp_path / "presets")
    # pydantic ValidationError is a ValueError subclass
    with pytest.raises(ValueError):
        s.save(Preset(name="", width=1024))
    with pytest.raises(ValueError):
        s.save(Preset(name="x", width=9999))
    with pytest.raises(ValueError):
        s.save(Preset(name="x", mode="bogus"))


def test_unknown_keys_ignored_on_save(tmp_path):
    s = PresetStore(tmp_path / "presets")
    p = Preset.model_validate({"name": "p", "width": 64,
                               "controls": {"width": True, "bogus": True}})
    s.save(p)
    on_disk = json.loads((tmp_path / "presets" / "p.json").read_text())
    assert "bogus" not in on_disk["controls"]
    assert set(PRESET_FIELDS) >= set(on_disk["controls"])
