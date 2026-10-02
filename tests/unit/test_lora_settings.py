"""Tests for apply_lora_texts (LoRA activation/negative text in the prompt)."""

import json
from pathlib import Path

from imagagen.config import Config
from imagagen.lora_settings import apply_lora_texts


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
