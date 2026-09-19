"""Unit tests for prompt wildcard processing."""

from pathlib import Path

import pytest

from imagagen import wildcards


@pytest.fixture
def wcdir(tmp_path: Path) -> Path:
    """A small wildcard tree (references are relative to the referencing
    file's own folder, and may only reach same folder or subfolders):
        color.txt            -> one colour per line
        nested.txt           -> a __color__ __places/rural__
                                (root: same-folder + a subfolder file)
        places/rural.txt     -> a quiet __place__   (same folder as itself)
        places/place.txt     -> a field / a lake
        bad.txt              -> references __bad__ (self recursion)
        empty.txt            -> only blank lines
    """
    (tmp_path / "places").mkdir()
    (tmp_path / "color.txt").write_text("red\ngreen\nblue\n", encoding="utf-8")
    (tmp_path / "nested.txt").write_text(
        "a __color__ __places/rural__", encoding="utf-8")
    (tmp_path / "places" / "rural.txt").write_text("a quiet __place__", encoding="utf-8")
    (tmp_path / "places" / "place.txt").write_text("a field\n\na lake\n", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("loop __bad__ here", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("\n   \n\n", encoding="utf-8")
    return tmp_path


def test_plain_wildcard_picks_a_line(wcdir):
    out = wildcards.resolve("a __color__ car", wcdir, seed=1)
    assert out.startswith("a ") and out.endswith(" car")
    assert out.split()[1] in {"red", "green", "blue"}


def test_deterministic_for_seed(wcdir):
    a = wildcards.resolve("x __color__ y", wcdir, seed=42)
    b = wildcards.resolve("x __color__ y", wcdir, seed=42)
    assert a == b


def test_different_seed_can_differ(wcdir):
    # 3 lines, many seeds; not every seed differs but the space must be
    # reachable (a single fixed line for every seed would be a bug)
    seen = {wildcards.resolve("__color__", wcdir, seed=s) for s in range(200)}
    assert len(seen) > 1


def test_multiple_references_roll_independently(wcdir):
    # the same wildcard twice can yield two different lines
    out = wildcards.resolve("__color__ __color__", wcdir, seed=3)
    first, second = out.split()
    assert first in {"red", "green", "blue"}
    assert second in {"red", "green", "blue"}


def test_nested_relative_to_referencing_folder(wcdir):
    # rural.txt (in places/) references __place__, which lives beside it in
    # the same folder -- the reference is resolved relative to rural.txt
    out = wildcards.resolve("__places/rural__", wcdir, seed=7)
    assert out.startswith("a quiet ")
    assert out.endswith((" field", " lake"))


def test_root_references_same_folder_and_subfolder(wcdir):
    # nested.txt (at the root) references __color__ (same folder) and
    # __places/rural__ (a subfolder), which in turn resolves __place__
    out = wildcards.resolve("__nested__", wcdir, seed=1)
    assert out.startswith("a ")
    colour = out.split()[1]
    assert colour in {"red", "green", "blue"}
    assert " quiet " in out
    assert out.endswith((" field", " lake"))


def test_parent_folder_reference_is_not_found(wcdir):
    # a file in a subfolder cannot reach back up to a same-named root file:
    # places/ has no color.txt, so a color reference from places/ is missing
    (wcdir / "places" / "needscolor.txt").write_text("__color__", encoding="utf-8")
    with pytest.raises(wildcards.WildcardError, match="not found"):
        wildcards.resolve("__places/needscolor__", wcdir, seed=1)


def test_missing_wildcard_raises(wcdir):
    with pytest.raises(wildcards.WildcardError, match="not found"):
        wildcards.resolve("a __nope__ b", wcdir, seed=1)


def test_self_recursion_raises(wcdir):
    with pytest.raises(wildcards.WildcardError, match="recursion"):
        wildcards.resolve("__bad__", wcdir, seed=1)


def test_empty_file_raises(wcdir):
    with pytest.raises(wildcards.WildcardError, match="no lines"):
        wildcards.resolve("__empty__", wcdir, seed=1)


def test_parent_folder_not_addressable(wcdir):
    # parent traversal is not expressible: '.' is not a valid name char, so a
    # would-be '../..' reference is left as literal text (never a lookup)
    out = wildcards.resolve("__..__", wcdir, seed=1)
    assert out == "__..__"


def test_absolute_path_not_matched(wcdir):
    # a leading slash is not a valid wildcard name, so it stays literal
    out = wildcards.resolve("__/etc/passwd__", wcdir, seed=1)
    assert out == "__/etc/passwd__"


def test_no_wildcard_text_is_unchanged(wcdir):
    assert wildcards.resolve("just a prompt", wcdir, seed=1) == "just a prompt"
    assert wildcards.has_wildcards("just a prompt") is False
    assert wildcards.has_wildcards("__x__") is True


def test_validate_passes_for_valid_and_raises_for_invalid(wcdir):
    wildcards.validate("__color__ and __places/rural__", wcdir)  # no raise
    with pytest.raises(wildcards.WildcardError):
        wildcards.validate("__nope__", wcdir)
    with pytest.raises(wildcards.WildcardError):
        wildcards.validate("__bad__", wcdir)


def test_validate_noop_without_wildcards(wcdir):
    # no tokens -> trivially valid (no file reads)
    wildcards.validate("plain text", wcdir)
