"""Prompt token counting (for the UI's live token counter).

The counter must reflect exactly what the SDXL backend feeds to CLIP, so it
uses the same BPE vocabulary the pipeline's CLIP-L encoder uses (the two SDXL
tokenizers share identical vocab/merges; only their special/pad token *config*
differs, which does not affect the content token count). Counts are computed
without the BOS/EOS framing tokens -- the same content tokens
``SdxlBackend._encode_prompt_chunked`` splits into 75-token windows. Token
weights (``(green:1.2)``) are stripped before counting and, when any weight
is in effect, the prompt is tokenized segment by segment, exactly like the
backend's weighted encoding path.
"""

from __future__ import annotations

import threading

# A1111/Forge prompt-window size (tokens per CLIP window, excl. BOS/EOS).
PROMPT_WINDOW = 75

_TOKENIZER_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

_lock = threading.Lock()
_tokenizer = None  # lazily loaded, cached for the process lifetime
_load_failed = False  # don't retry a failed load on every keystroke


def _load_tokenizer():
    from transformers import CLIPTokenizer

    # Offline first (the files live in the project-local HF cache after any
    # model load); fall back to a hub fetch if the cache is empty.
    for local_only in (True, False):
        try:
            return CLIPTokenizer.from_pretrained(
                _TOKENIZER_REPO,
                subfolder="tokenizer",
                use_fast=False,
                local_files_only=local_only,
            )
        except Exception:
            continue
    raise RuntimeError("could not load the CLIP tokenizer")


def count_prompt_tokens(prompt: str) -> int | None:
    """Number of content tokens in ``prompt`` (excl. BOS/EOS), or None if the
    tokenizer could not be loaded (the UI then hides the counter).

    Token weights ``(green:1.2)`` are counted the way the backend feeds
    them: the annotations are stripped, and when any weight differs from
    1.0 the prompt is tokenized segment by segment (the backend's weighted
    path); otherwise the cleaned prompt is tokenized whole."""
    global _tokenizer, _load_failed
    if _tokenizer is None and not _load_failed:
        with _lock:
            if _tokenizer is None and not _load_failed:
                try:
                    _tokenizer = _load_tokenizer()
                except Exception:
                    _load_failed = True
    if _tokenizer is None:
        return None
    from .token_weights import clean_prompt, parse_prompt_weights

    segments = parse_prompt_weights(prompt)
    if any(w != 1.0 for _, w in segments):
        return sum(
            len(_tokenizer(text, add_special_tokens=False).input_ids)
            for text, _ in segments if text
        )
    return len(_tokenizer(clean_prompt(prompt), add_special_tokens=False).input_ids)


def prompt_windows(tokens: int) -> int:
    """How many 75-token windows a prompt of ``tokens`` content tokens needs
    (0 tokens -> 0)."""
    if tokens <= 0:
        return 0
    return (tokens + PROMPT_WINDOW - 1) // PROMPT_WINDOW
