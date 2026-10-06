"""Tests for the Krea 2 chat-template prefix cut
(imagagen.encoders.krea2.template_prefix_cut).

The reference encodes the fully chat-templated prompt and feeds the DiT
only the tokens from the SECOND im_start on (the user-turn opening),
skipping the following ``user`` + newline tokens when present. The
template text itself (``imagagen.encoders.krea2_prompt.KREA2_PROMPT_TEMPLATE``) is
never printed by these tests -- it carries special tokens; only token-id
arithmetic on its tokenization is checked.
"""

from pathlib import Path

import pytest

from imagagen.encoders.krea2 import krea2_prompt_template, template_prefix_cut
from transformers import Qwen2Tokenizer

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = ROOT / "res" / "tokenizer_krea2"

IM_START = 151644
USER = 872
NEWLINE = 198


@pytest.fixture(scope="module")
def tok():
    return Qwen2Tokenizer.from_pretrained(str(TOKENIZER_DIR))


def _ids(tok, prompt: str) -> list[int]:
    text = krea2_prompt_template().format(prompt)
    return list(tok(text, add_special_tokens=False).input_ids)


def _expected_cut(ids: list[int]) -> int:
    """The reference rule, restated independently: the second im_start,
    extended past a following user + newline pair."""
    pos, count = None, 0
    for i, t in enumerate(ids):
        if t == IM_START and count < 2:
            pos, count = i, count + 1
    if pos is not None and pos + 3 < len(ids) \
            and ids[pos + 1] == USER and ids[pos + 2] == NEWLINE:
        pos += 3
    return pos


class TestTemplatePrefixCut:
    def test_cut_matches_the_reference_rule(self, tok):
        for prompt in ("a zebra on the moon", "x",
                       "one two three four five six seven eight"):
            ids = _ids(tok, prompt)
            assert template_prefix_cut(ids) == _expected_cut(ids)

    def test_cut_is_at_the_second_im_start(self, tok):
        ids = _ids(tok, "a zebra on the moon")
        im_starts = [i for i, t in enumerate(ids) if t == IM_START]
        assert len(im_starts) >= 2  # system + user opening at minimum
        cut = template_prefix_cut(ids)
        # the cut is the second im_start, or three past it (user + \n)
        second = im_starts[1]
        assert cut in (second, second + 3)

    def test_remaining_tokens_start_with_the_prompt(self, tok):
        prompt = "a zebra on the moon"
        ids = _ids(tok, prompt)
        cut = template_prefix_cut(ids)
        decoded = tok.decode(ids[cut:])
        assert decoded.startswith(prompt)

    def test_prompt_reaches_the_encoder_verbatim(self, tok):
        # whatever the cut, the prompt's own tokenization is a contiguous
        # prefix of the remaining sequence
        prompt = "a zebra on the moon"
        p_ids = list(tok(prompt, add_special_tokens=False).input_ids)
        ids = _ids(tok, prompt)
        cut = template_prefix_cut(ids)
        assert ids[cut:cut + len(p_ids)] == p_ids

    def test_no_im_start_yields_zero(self):
        assert template_prefix_cut([1, 2, 3]) == 0
        # a lone im_start: template_end stays at its index (0 here) -- the
        # cut is only *meaningful* from the second occurrence on
        assert template_prefix_cut([IM_START]) == 0

    def test_single_im_start_no_extension(self):
        # one im_start only, no following user + newline: cut at it
        assert template_prefix_cut([0, IM_START, 1, 2]) == 1
        # followed by user + newline: extended
        assert template_prefix_cut([0, IM_START, USER, NEWLINE, 1, 2]) == 4
