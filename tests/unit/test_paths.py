
from imagagen.paths import next_output_number, next_output_path, sanitize_prompt


def test_sanitize_prompt():
    assert sanitize_prompt("Photo of a Turtle!") == "photo-of-a-turtle"
    assert sanitize_prompt("   ") == "image"
    assert sanitize_prompt("a  b\n\n  c") == "a-b-c"
    assert len(sanitize_prompt("x" * 500)) <= 80


def test_next_output_number_empty(tmp_path):
    assert next_output_number(tmp_path) == 1


def test_next_output_number_missing_dir(tmp_path):
    assert next_output_number(tmp_path / "nope") == 1


def test_next_output_number_uses_max(tmp_path):
    (tmp_path / "5-foo.png").touch()
    (tmp_path / "123-photo-of-a-turtle.png").touch()
    (tmp_path / "no-number.png").touch()
    (tmp_path / "0-zero.png").touch()
    assert next_output_number(tmp_path) == 124


def test_next_output_path(tmp_path):
    (tmp_path / "123-photo-of-a-turtle.png").touch()
    p = next_output_path(tmp_path, "Photo of a frog")
    assert p.name == "124-photo-of-a-frog.png"
    assert p.parent == tmp_path


def test_subdirectories_of_output_ignored(tmp_path):
    (tmp_path / "subdir/999-x.png").parent.mkdir()
    (tmp_path / "subdir/999-x.png").touch()
    assert next_output_number(tmp_path) == 1
