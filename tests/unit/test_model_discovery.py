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


def test_flux_family_discovery(tmp_path):
    # family is the directory name; .gguf and .safetensors (Kijai FP8 /
    # bnb NF4 checkpoints) are all loadable for flux
    _touch(tmp_path, "models/flux/MyFlux.gguf")
    _touch(tmp_path, "models/flux/MyFlux2.gguf")
    _touch(tmp_path, "models/flux/DevNF4.safetensors")
    _touch(tmp_path, "models/sdxl/Epic.safetensors")

    reg = ModelRegistry(Config(root=tmp_path))
    by_id = {m.id: m for m in reg.models}
    assert set(by_id) == {"MyFlux", "MyFlux2", "DevNF4", "Epic"}
    assert by_id["MyFlux"].family == "flux"
    assert by_id["MyFlux2"].family == "flux"
    assert by_id["DevNF4"].family == "flux"
    assert by_id["Epic"].family == "sdxl"


def test_missing_dirs_yield_empty_lists(tmp_path):
    assert ModelRegistry(Config(root=tmp_path)).models == []
    assert list_loras(Config(root=tmp_path)) == []


def _lora_tree(root):
    (root / "loras" / "sdxl" / "sub").mkdir(parents=True)
    (root / "loras" / "sdxl" / "sub2").mkdir(parents=True)
    (root / "loras" / "sdxl" / "Top.safetensors").touch()
    (root / "loras" / "sdxl" / "sub" / "Deep.safetensors").touch()
    (root / "loras" / "sdxl" / "sub" / "Twin.safetensors").touch()
    (root / "loras" / "sdxl" / "sub2" / "Twin.safetensors").touch()
    # plain file whose prompt-token (display) name differs from the basename
    (root / "loras" / "sdxl" / "Xylora.safetensors").touch()
    return Config(root=root)


def test_resolve_duplicate_basename_symlinked_folder(tmp_path):
    """A duplicate in a symlinked folder is discovered too; the first
    candidate in registry order wins (here the real file, which sorts
    before the link)."""
    from imagagen.models import resolve_lora_names

    (tmp_path / "loras" / "sdxl" / "real").mkdir(parents=True)
    (tmp_path / "loras" / "sdxl" / "real" / "Twin.safetensors").touch()
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "Twin.safetensors").touch()
    (tmp_path / "loras" / "sdxl" / "zz-link").symlink_to(tmp_path / "extra")

    assert resolve_lora_names(["Twin"], Config(root=tmp_path)) == ["real/Twin"]


def test_resolve_lora_names(tmp_path):
    from imagagen.metadata import forge_lora_hash
    from imagagen.models import resolve_lora_names

    cfg = _lora_tree(tmp_path)
    # exact match stays untouched
    assert resolve_lora_names(["Top"], cfg) == ["Top"]
    assert resolve_lora_names(["sub/Deep"], cfg) == ["sub/Deep"]
    # unique basename: the subfolder path is found
    assert resolve_lora_names(["Deep"], cfg) == ["sub/Deep"]
    # duplicate basename, no hashes: first in registry (sorted discovery)
    # order is picked, so generation still works
    assert resolve_lora_names(["Twin"], cfg) == ["sub/Twin"]
    # identical basenames in two subfolders hash identically -> still
    # ambiguous -> first in registry order
    h_twin = forge_lora_hash("Twin.safetensors")
    assert resolve_lora_names(["Twin"], cfg, {"Twin": h_twin}) == ["sub/Twin"]
    # different extensions = different basenames = different hashes:
    # the hash picks the right one
    (tmp_path / "loras" / "sdxl" / "sub2" / "Twin.ckpt").touch()
    h_ckpt = forge_lora_hash("Twin.ckpt")
    assert resolve_lora_names(
        ["Twin"], cfg, {"Twin": h_ckpt}
    ) == ["sub2/Twin"]
    # display name: token differs from the basename, but the Forge hash
    # (sha256 of the basename) links them
    h_xy = forge_lora_hash("Xylora.safetensors")
    assert resolve_lora_names(
        ["Xy Lora"], cfg, {"Xy Lora": h_xy}
    ) == ["Xylora"]
    # Forge strips ":"/"," from hash keys: the stripped form still matches
    assert resolve_lora_names(
        ["Xy:Lora"], cfg, {"XyLora": h_xy}
    ) == ["Xylora"]
    # renamed file: the stored hash is of the OLD basename, so it matches
    # nothing locally and the name is left as-is
    h_old = forge_lora_hash("oldname.safetensors")
    assert resolve_lora_names(
        ["oldname"], cfg, {"oldname": h_old}
    ) == ["oldname"]
    # unknown name, no usable hash: unchanged
    assert resolve_lora_names(["Mystery"], cfg, {"Mystery": "0" * 12}) == ["Mystery"]
