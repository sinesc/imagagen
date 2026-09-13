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


def test_symlinked_checkpoint_dir_is_followed(tmp_path):
    """regression: pathlib's ** skips symlinked dirs since Python 3.13."""
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    (real / "nested" / "Linked.safetensors").touch()
    (tmp_path / "models" / "sdxl").mkdir(parents=True)
    (tmp_path / "models" / "sdxl" / "linked").symlink_to(real)

    reg = ModelRegistry(Config(root=tmp_path))
    assert [m.id for m in reg.models] == ["linked/nested/Linked"]


def test_symlinked_lora_family_and_subdir_are_followed(tmp_path):
    (tmp_path / "real-loras").mkdir()
    (tmp_path / "real-loras" / "A.safetensors").touch()
    (tmp_path / "loras").mkdir()
    (tmp_path / "loras" / "sdxl").symlink_to(tmp_path / "real-loras")
    (tmp_path / "loras" / "extra").mkdir()
    (tmp_path / "extra-loras").mkdir()
    (tmp_path / "extra-loras" / "B.ckpt").touch()
    (tmp_path / "loras" / "extra" / "linked").symlink_to(tmp_path / "extra-loras")

    # family dirs are listed alphabetically: "extra" before "sdxl"
    assert list_loras(Config(root=tmp_path)) == ["linked/B", "A"]


def test_symlink_cycle_does_not_loop_forever(tmp_path):
    (tmp_path / "models" / "sdxl" / "loop").mkdir(parents=True)
    (tmp_path / "models" / "sdxl" / "loop" / "Model.safetensors").touch()
    (tmp_path / "models" / "sdxl" / "loop" / "back").symlink_to(
        tmp_path / "models" / "sdxl")

    reg = ModelRegistry(Config(root=tmp_path))
    assert [m.id for m in reg.models] == ["loop/Model"]


def test_broken_symlink_is_ignored(tmp_path):
    (tmp_path / "models" / "sdxl").mkdir(parents=True)
    (tmp_path / "models" / "sdxl" / "ok.safetensors").touch()
    (tmp_path / "models" / "sdxl" / "gone").symlink_to(tmp_path / "nope")

    reg = ModelRegistry(Config(root=tmp_path))
    assert [m.id for m in reg.models] == ["ok"]


def test_missing_dirs_yield_empty_lists(tmp_path):
    assert ModelRegistry(Config(root=tmp_path)).models == []
    assert list_loras(Config(root=tmp_path)) == []
