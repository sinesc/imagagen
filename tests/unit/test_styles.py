"""Unit tests for styles (styles.csv store, {prompt} application, metadata)."""

import csv
from pathlib import Path

import pytest
from PIL import Image

from imagagen import metadata
from imagagen.styles import TEMP_GROUP, Style, StyleStore, apply_styles


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    return tmp_path / "styles.csv"


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", "utf-8")


def test_load_and_all(csv_path):
    write_csv(
        csv_path,
        [
            "name,prompt,negative_prompt,optgroup",
            'Cinematic,"cinematic film still {prompt} . moody",,"Photo Styles"',
            "Plain,just plain text,,",
        ],
    )
    store = StyleStore(csv_path)
    all_ = store.all()
    assert [s["name"] for s in all_] == ["Cinematic", "Plain"]
    cin = [s for s in all_ if s["name"] == "Cinematic"][0]
    assert cin["prompt"].startswith("cinematic film still {prompt}")
    assert cin["optgroup"] == "Photo Styles"
    assert cin["temporary"] is False
    assert store.get("Plain").prompt == "just plain text"
    assert store.get("Nope") is None


def test_load_missing_and_bom_file(tmp_path, csv_path):
    assert StyleStore(csv_path).all() == []
    # editors often insert a UTF-8 BOM; the header must still be read
    csv_path.write_bytes(b"\xef\xbb\xbf" + b"name,prompt,negative_prompt,optgroup\n"
 + b"X,hello,,\n")
    store = StyleStore(csv_path)
    assert [s["name"] for s in store.all()] == ["X"]


def test_malformed_rows_are_skipped(csv_path):
    write_csv(
        csv_path,
        [
            "name,prompt,negative_prompt,optgroup",
            ',no name,,',          # empty name -> skipped
            "Good,fine,,",
        ],
    )
    store = StyleStore(csv_path)
    assert [s["name"] for s in store.all()] == ["Good"]


def test_save_upsert_delete_and_roundtrip(csv_path):
    store = StyleStore(csv_path)
    store.save(Style(name="A", prompt="a {prompt}", optgroup="G"))
    store.save(Style(name="B", prompt="b", negative_prompt="bad"))
    # upsert: same name replaces in place
    store.save(Style(name="A", prompt="a2", negative_prompt="neg-a"))
    names = [s["name"] for s in store.all()]
    assert names == ["A", "B"]
    assert store.get("A").prompt == "a2"
    assert store.get("A").negative_prompt == "neg-a"

    # the CSV on disk survives a reload (and a fresh store)
    store2 = StyleStore(csv_path)
    assert [s["name"] for s in store2.all()] == ["A", "B"]
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["prompt"] == "a2"

    assert store.delete("B") is True
    assert store.delete("B") is False
    assert [s["name"] for s in store.all()] == ["A"]


def test_temporary_styles(tmp_path, csv_path):
    store = StyleStore(csv_path)
    store.save(Style(name="Real", prompt="real"))
    store.add_temporary(Style(name="Ghost", prompt="ghost {prompt}"))
    all_ = {s["name"]: s for s in store.all()}
    assert all_["Ghost"]["temporary"] is True
    assert all_["Ghost"]["optgroup"] == TEMP_GROUP
    assert all_["Real"]["temporary"] is False
    # lookup finds the temporary
    assert store.get("Ghost").prompt == "ghost {prompt}"
    # a temporary is never persisted
    assert "Ghost" not in csv_path.read_text("utf-8")
    # a temporary never shadows a same-named real style, and saving a real
    # style with a temporary's name drops the temporary
    store.add_temporary(Style(name="Real", prompt="other"))
    assert store.get("Real").prompt == "real"
    store.save(Style(name="Ghost", prompt="ghost now real"))
    all_ = {s["name"]: s for s in store.all()}
    assert all_ == {"Real": {"name": "Real", "prompt": "real",
                             "negative_prompt": "", "optgroup": "",
                             "temporary": False},
                   "Ghost": {"name": "Ghost", "prompt": "ghost now real",
                             "negative_prompt": "", "optgroup": "",
                             "temporary": False}}


def _lookup(store: StyleStore):
    return store.get


def test_apply_styles_replaces_prompt_placeholder(tmp_path):
    store = StyleStore(tmp_path / "s.csv")
    store.save(Style(name="Film", prompt="analog film photo {prompt} . grainy"))
    p, n = apply_styles("a frog", "", ["Film"], _lookup(store))
    assert p == "analog film photo a frog . grainy"
    assert n == ""


def test_apply_styles_append_without_placeholder(tmp_path):
    store = StyleStore(tmp_path / "s.csv")
    store.save(Style(name="BG", prompt="in the dunes, white sand",
                     negative_prompt="clutter"))
    p, n = apply_styles("a frog", "ugly", ["BG"], _lookup(store))
    assert p == "a frog, in the dunes, white sand"
    assert n == "ugly, clutter"


def test_apply_styles_chained_order(tmp_path):
    store = StyleStore(tmp_path / "s.csv")
    store.save(Style(name="A", prompt="A[{prompt}]"))
    store.save(Style(name="B", prompt="B[{prompt}]"))
    p, _ = apply_styles("x", "", ["A", "B"], _lookup(store))
    assert p == "B[A[x]]"
    # reversed order
    p, _ = apply_styles("x", "", ["B", "A"], _lookup(store))
    assert p == "A[B[x]]"


def test_apply_styles_unknown_and_empty(tmp_path):
    store = StyleStore(tmp_path / "s.csv")
    store.save(Style(name="E", prompt=""))  # empty style: no-op
    p, n = apply_styles("x", "y", ["Missing", "E"], _lookup(store))
    assert (p, n) == ("x", "y")
    # empty current prompt + no placeholder -> bare style text
    p, _ = apply_styles("", "", ["A"], lambda n: Style(name=n, prompt="A text"))
    assert p == "A text"


# ---------------- metadata round-trip ----------------

def test_metadata_styles_roundtrip():
    styles = [
        {"name": "Cinematic", "prompt": "cinematic film still {prompt} . moody",
         "negative_prompt": ""},
        {"name": "Realism Negatives", "prompt": "",
         "negative_prompt": "digital art, painting, cartoon"},
    ]
    text = metadata.build_parameters(
        prompt="raw user prompt",
        negative_prompt="ugly",
        steps=25, sampler="dpmpp_2m", scheduler="karras", cfg=7.0, seed=1,
        width=1024, height=1024, model_name="m", model_hash="h", styles=styles,
    )
    # the parameters text still records the RAW prompt
    assert text.splitlines()[0] == "raw user prompt"
    # Styles line is AFTER the meta line (A1111/Forge ignore it)
    lines = text.splitlines()
    meta_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Steps:"))
    styles_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Styles:"))
    assert styles_idx > meta_idx

    out = metadata.parse_parameters(text)
    assert out["prompt"] == "raw user prompt"
    assert out["negative_prompt"] == "ugly"
    assert out["steps"] == 25
    assert out["styles"] == styles


def test_metadata_no_styles_unchanged():
    text = metadata.build_parameters(
        prompt="p", negative_prompt="n", steps=3, sampler="s", scheduler="sc",
        cfg=1.0, seed=0, width=64, height=64, model_name="m", model_hash="h",
    )
    assert "Styles:" not in text
    assert "styles" not in metadata.parse_parameters(text)


def test_metadata_bad_styles_line_is_ignored():
    text = "p\nNegative prompt: n\nSteps: 3, Version: v\nStyles: not-json"
    out = metadata.parse_parameters(text)
    assert "styles" not in out


def test_metadata_prompt_with_newlines_survives(tmp_path):
    # a multi-line raw prompt must not swallow the Styles line
    text = metadata.build_parameters(
        prompt="line1\nline2", negative_prompt="", steps=3, sampler="s",
        scheduler="sc", cfg=1.0, seed=0, width=64, height=64, model_name="m",
        model_hash="h", styles=[{"name": "X", "prompt": "x", "negative_prompt": ""}],
    )
    out = metadata.parse_parameters(text)
    assert out["prompt"] == "line1\nline2"
    assert out["styles"][0]["name"] == "X"


def test_styles_survive_png_write(tmp_path):
    # end-to-end chunk check: attach_metadata -> PNG bytes -> read_metadata
    img = Image.new("RGB", (8, 8), "red")
    text = metadata.build_parameters(
        prompt="raw", negative_prompt="bad things", steps=3, sampler="s",
        scheduler="sc", cfg=1.0, seed=0, width=8, height=8, model_name="m",
        model_hash="h",
        styles=[{"name": "S", "prompt": "s {prompt}", "negative_prompt": ""}],
    )
    img, info = metadata.attach_metadata(img, text)
    img.save(tmp_path / "x.png", pnginfo=info)
    out = metadata.read_metadata((tmp_path / "x.png").read_bytes())
    assert out["prompt"] == "raw"
    assert out["styles"] == [{"name": "S", "prompt": "s {prompt}",
                              "negative_prompt": ""}]
