"""Forge/A1111-style per-token prompt weights ("emphasis").

The SDXL backend scales the CLIP token embeddings of weighted prompt
segments by their weight after encoding (Forge's "No norm" emphasis --
plain per-token scaling, which Forge notes works better for SDXL than the
mean-preserving "Original" variant).

Grammar (ported from Forge's ``parse_prompt_attention``):
- ``(green:1.2)``  -- weight 1.2
- ``(green)``      -- weight 1.1 (the default emphasis)
- ``[green]``      -- weight 1/1.1
- nested groups multiply: ``(a (b:1.2) c:0.8)`` -> a:0.8, b:0.96, c:0.8
- the colon-weight must sit directly before the closing paren:
  ``(a:1.2 :2.0)`` is text ``a:1.2 `` at weight 2.0
- an unbalanced ``(`` at the end of the prompt still applies its weight to
  the remainder (A1111's behavior); a stray ``)`` or ``]`` is literal text
- ``\\(``, ``\\)``, ``\\[``, ``\\]``, ``\\\\`` escape a literal character

Every token a segment tokenizes to inherits the segment's weight (a word
split into several BPE tokens gets all of them weighted); whitespace and
punctuation outside the groups stay in the neighbouring plain segments.
"""

from __future__ import annotations

import re

DEFAULT_EMPHASIS = 1.1

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)


def parse_prompt_weights(text: str) -> list[tuple[str, float]]:
    """Parse ``text`` into ``(segment, weight)`` pairs.

    Joining the segment texts reproduces the prompt with the weight
    annotations removed. Adjacent segments sharing a weight are merged
    into one, so a prompt without any annotation syntax is a single
    ``(text, 1.0)`` segment.
    """
    res: list[list] = []
    round_brackets = []
    square_brackets = []

    def multiply_range(start_position: int, multiplier: float) -> None:
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        tok = m.group(0)
        weight = m.group(1)

        if tok.startswith('\\'):
            res.append([tok[1:], 1.0])
        elif tok == '(':
            round_brackets.append(len(res))
        elif tok == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif tok == ')' and round_brackets:
            multiply_range(round_brackets.pop(), DEFAULT_EMPHASIS)
        elif tok == ']' and square_brackets:
            multiply_range(square_brackets.pop(), 1.0 / DEFAULT_EMPHASIS)
        else:
            res.append([tok, 1.0])

    # unbalanced openers still apply to the remainder (A1111 behavior)
    for pos in round_brackets:
        multiply_range(pos, DEFAULT_EMPHASIS)
    for pos in square_brackets:
        multiply_range(pos, 1.0 / DEFAULT_EMPHASIS)

    if not res:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return [(t, float(w)) for t, w in res]


def clean_prompt(text: str) -> str:
    """The prompt with all weight annotations removed."""
    return "".join(t for t, _ in parse_prompt_weights(text))


def has_token_weights(text: str) -> bool:
    """True when any parsed segment carries a weight other than 1.0."""
    return any(w != 1.0 for _, w in parse_prompt_weights(text))
