"""Recursive model/LoRA discovery (subdirectories) and id formats."""

from imagagen.config import Config
from imagagen.models import ModelRegistry, list_loras


def _touch(root, *parts):
    p = root.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def test_checkpoint_discovery_is_recursive(tmp_path):
    _touch(tmp_path, "models/sdxl/top.safetensors")
    _touch(tmp_path, "models/sdxl/test-dir/JuggernautXL9.0.safetensors")
    _touch(tmp_path, "models/sdxl/test-dir/nested/deeper.safetensors")
    _touch(tmp_path, "models/sdxl/test-dir/notes.txt")  # ignored
    _touch(tmp_path, "models/sdxl_vae.safetensors")  # not part of the family

    reg = ModelRegistry(Config(root=tmp_path))
    assert [m.id for m in reg.models] == [
        "test-dir/JuggernautXL9.0",
        "test-dir/nested/deeper",
        "top",
    ]
    # name == id: nested models are distinguished by their path segments
    assert [m.name for m in reg.models] == [m.id for m in reg.models]
    assert reg.get("test-dir/JuggernautXL9.0").path == (
        tmp_path / "models/sdxl/test-dir/JuggernautXL9.0.safetensors"
    )


def test_lora_discovery_is_recursive(tmp_path):
    _touch(tmp_path, "loras/sdxl/DMD2fp32XL1.0.safetensors")
    _touch(tmp_path, "loras/sdxl/test-dir/FuturisticXL.safetensors")
    _touch(tmp_path, "loras/sdxl/test-dir/old.ckpt")

    assert list_loras(Config(root=tmp_path)) == [
        "DMD2fp32XL1.0",
        "test-dir/FuturisticXL",
        "test-dir/old",
    ]


def test_missing_dirs_yield_empty_lists(tmp_path):
    assert ModelRegistry(Config(root=tmp_path)).models == []
    assert list_loras(Config(root=tmp_path)) == []
