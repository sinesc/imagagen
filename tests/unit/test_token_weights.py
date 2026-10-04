"""Unit tests for Forge/A1111-style token-weight prompt parsing."""

import pytest

from imagagen import token_weights
from imagagen.token_weights import clean_prompt, has_token_weights, parse_prompt_weights


def test_plain_prompt_is_one_segment():
    assert parse_prompt_weights("a red panda") == [("a red panda", 1.0)]
    assert parse_prompt_weights("") == [("", 1.0)]


def test_basic_weight():
    assert parse_prompt_weights("little (green:1.2) men") == [
        ("little ", 1.0), ("green", 1.2), (" men", 1.0)]


def test_default_emphasis():
    # (x) -> 1.1, [x] -> 1/1.1
    assert parse_prompt_weights("(x)") == [("x", 1.1)]
    assert parse_prompt_weights("[x]") == [("x", 1.0 / 1.1)]
    # (x:1.1) is the same as (x)
    assert parse_prompt_weights("(x:1.1)") == parse_prompt_weights("(x)")


def test_nested_weights_multiply():
    assert parse_prompt_weights("(a (b:1.2) c:0.8)") == [
        ("a ", 0.8), ("b", 0.96), (" c", 0.8)]
    assert parse_prompt_weights("((a:1.2):3)") == [("a", 3.5999999999999996)]


def test_weight_must_precede_closing_paren():
    # only a colon-number directly before ) is a weight
    assert parse_prompt_weights("(a:1.2 :2.0)") == [("a:1.2 ", 2.0)]
    # outside parens the syntax is literal text
    assert parse_prompt_weights("a:1.2)") == [("a:1.2)", 1.0)]
    assert parse_prompt_weights(":1.2) text") == [(":1.2) text", 1.0)]
    # whitespace around the weight is fine
    assert parse_prompt_weights("(green: 1.2)") == [("green", 1.2)]
    assert parse_prompt_weights("(green:1.2 )") == [("green", 1.2)]


def test_unbalanced_parens_follow_a1111():
    # an unclosed ( still emphasizes the remainder
    assert parse_prompt_weights("(unbalanced") == [("unbalanced", 1.1)]
    # a stray ) or ] is literal text
    assert parse_prompt_weights("a (b) c) d") == [("a ", 1.0), ("b", 1.1), (" c) d", 1.0)]


def test_escapes():
    assert parse_prompt_weights("\\(literal\\]") == [("(literal]", 1.0)]
    assert parse_prompt_weights("a\\(b") == [("a(b", 1.0)]


def test_zero_and_negative_weights():
    assert parse_prompt_weights("(x:0.0)") == [("x", 0.0)]
    assert parse_prompt_weights("(x:-0.5)") == [("x", -0.5)]


def test_adjacent_equal_segments_merge():
    # (unnecessary)(parens) -> one segment at 1.1
    assert parse_prompt_weights("(unnecessary)(parens)") == [("unnecessaryparens", 1.1)]


def test_deep_nesting_matches_forge_docstring():
    assert parse_prompt_weights("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).") == [
        ("a ", 1.0),
        ("house", 1.5730000000000004),
        (" ", 1.1),
        ("on", 1.0),
        (" a ", 1.1),
        ("hill", 0.55),
        (", sun, ", 1.1),
        ("sky", 1.4641000000000006),
        (".", 1.1)]


def test_clean_prompt():
    assert clean_prompt("little (green:1.2) men") == "little green men"
    assert clean_prompt("\\(x)") == "(x)"
    assert clean_prompt("plain") == "plain"


def test_has_token_weights():
    assert has_token_weights("little (green:1.2) men")
    assert has_token_weights("(green)")
    assert not has_token_weights("little green men")
    # an explicit weight of 1.0 is not a weight
    assert not has_token_weights("(green:1.0)")
