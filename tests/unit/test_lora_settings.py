"""Tests for apply_lora_texts (LoRA activation/negative text in the prompt)
and the activation-text tag split the UI toggles work on."""

import json
from pathlib import Path

from imagagen.config import Config
from imagagen.lora_settings import (
    activation_tag_universe,
    apply_lora_texts,
    split_tags,
)


def make_config(tmp_path: Path) -> Config:
    config = Config(root=tmp_path)
    loras = config.loras_dir / "sdxl"
    loras.mkdir(parents=True)
    return config


def add_lora(config: Config, name: str, **fields) -> None:
    f = config.loras_dir / "sdxl" / f"{name}.safetensors"
    f.write_bytes(b"")
    if fields:
        f.with_suffix(".json").write_text(
            json.dumps(fields, indent=4) + "\n", encoding="utf-8"
        )


def test_append_without_placeholder(tmp_path):
    config = make_config(tmp_path)
    add_lora(
        config, "a",
        **{"activation text": "blue eyes", "negative text": "blurry"},
    )
    prompt, negative = apply_lora_texts(
        "a cat", "ugly", [("a", 1.0)], config
    )
    assert prompt == "a cat, blue eyes"
    assert negative == "ugly, blurry"


def test_placeholder_wraps_prompt(tmp_path):
    config = make_config(tmp_path)
    add_lora(
        config, "b",
        **{"activation text": "photo of {prompt}, 8k",
           "negative text": "{prompt} is not blurry"},
    )
    prompt, negative = apply_lora_texts(
        "a cat", "blurry", [("b", 0.5)], config
    )
    assert prompt == "photo of a cat, 8k"
    assert negative == "blurry is not blurry"


def test_empty_texts_are_noop(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "c")  # no settings file at all
    add_lora(config, "d", **{"activation text": "", "negative text": ""})
    prompt, negative = apply_lora_texts(
        "a cat", "ugly", [("c", 1.0), ("d", 0.8)], config
    )
    assert prompt == "a cat"
    assert negative == "ugly"


def test_unknown_names_are_skipped(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "e", **{"activation text": "sparkles"})
    prompt, negative = apply_lora_texts(
        "a cat", "", [("missing", 1.0), ("e", 0.3)], config
    )
    assert prompt == "a cat, sparkles"
    assert negative == ""


def test_multiple_loras_in_list_order(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "f", **{"activation text": "red hair"})
    add_lora(config, "g", **{"activation text": "green eyes"})
    prompt, _ = apply_lora_texts(
        "a cat", "", [("f", 1.0), ("g", 0.5), ("f", 0.5)], config
    )
    assert prompt == "a cat, red hair, green eyes, red hair"


def test_placeholder_chains_across_loras(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "h", **{"activation text": "[{prompt}]"})
    add_lora(config, "i", **{"activation text": "({prompt})"})
    prompt, _ = apply_lora_texts("a cat", "", [("h", 1.0), ("i", 1.0)], config)
    assert prompt == "([a cat])"


def test_split_tags():
    assert split_tags("blue eyes, 8k ,, photo of {prompt}") == [
        "blue eyes", "8k", "photo of {prompt}"
    ]
    assert split_tags("") == []


def test_activation_tag_universe(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "j", **{"activation text": "red hair, green eyes"})
    add_lora(config, "k", **{"activation text": " green eyes , 8k"})
    add_lora(config, "l")  # no settings file -> no tags
    # LoRA list order, deduped across LoRAs (a repeated tag has one toggle)
    assert activation_tag_universe(
        [("j", 1.0), ("k", 0.5), ("l", 1.0), ("missing", 1.0)], config
    ) == ["red hair", "green eyes", "8k"]


def test_disabled_tags_drop_only_those_tags(tmp_path):
    config = make_config(tmp_path)
    add_lora(
        config, "m",
        **{"activation text": "red hair, green eyes, 8k",
           "negative text": "blurry, low res"},
    )
    prompt, negative = apply_lora_texts(
        "a cat", "", [("m", 1.0)], config, ["green eyes"]
    )
    assert prompt == "a cat, red hair, 8k"
    # the negative text is NOT split into tags: it applies as a whole
    assert negative == "blurry, low res"


def test_disabled_tags_are_per_lora_and_keep_the_order(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "n", **{"activation text": "one, two"})
    add_lora(config, "o", **{"activation text": "three, four"})
    prompt, _ = apply_lora_texts(
        "a cat", "", [("n", 1.0), ("o", 1.0)], config, ["two", "three"]
    )
    assert prompt == "a cat, one, four"


def test_all_tags_disabled_applies_no_activation_text(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "p", **{"activation text": "x, y",
                             "negative text": "bad"})
    prompt, negative = apply_lora_texts(
        "a cat", "", [("p", 1.0)], config, ["x", "y"]
    )
    assert prompt == "a cat"
    assert negative == "bad"  # the negative text is not affected


def test_disabled_tags_unknown_ones_are_inert(tmp_path):
    config = make_config(tmp_path)
    add_lora(config, "q", **{"activation text": "only tag"})
    prompt, _ = apply_lora_texts(
        "a cat", "", [("q", 1.0)], config, ["a tag of a removed lora"]
    )
    assert prompt == "a cat, only tag"
