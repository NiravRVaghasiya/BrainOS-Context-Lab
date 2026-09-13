"""Token counting strategies used for context accounting."""

from __future__ import annotations

import pytest

from brain.tokenizers import (
    TokenizerUnavailableError,
    char_ratio_counter,
    estimate_tokens,
    provider_counter,
    resolve_counter,
    tiktoken_counter,
    truncate_to_tokens,
    whitespace_counter,
)


def test_estimate_tokens_is_conservative_and_never_free() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 400) == 100


def test_char_ratio_counter_supports_other_model_families() -> None:
    counter = char_ratio_counter(3.0)
    assert counter("abcdef") == 2
    assert counter("") == 0
    with pytest.raises(ValueError):
        char_ratio_counter(0)


def test_whitespace_counter_is_a_word_level_proxy() -> None:
    assert whitespace_counter("") == 0
    assert whitespace_counter("one two  three") == 3


def test_truncate_to_tokens_respects_the_budget() -> None:
    text = "The production database is PostgreSQL 16 and it is replicated."
    trimmed, truncated = truncate_to_tokens(text, 8, estimate_tokens)
    assert truncated is True
    assert estimate_tokens(trimmed) <= 8
    assert trimmed.endswith("…")
    assert trimmed.startswith("The production")


def test_truncate_to_tokens_is_a_noop_when_the_text_fits() -> None:
    text = "short"
    trimmed, truncated = truncate_to_tokens(text, 100, estimate_tokens)
    assert (trimmed, truncated) == (text, False)


def test_truncate_to_tokens_handles_a_zero_budget() -> None:
    assert truncate_to_tokens("anything", 0, estimate_tokens) == ("", True)


def test_truncation_is_stable_for_a_word_level_counter() -> None:
    text = " ".join(f"token{index}" for index in range(50))
    trimmed, truncated = truncate_to_tokens(text, 10, whitespace_counter)
    assert truncated is True
    assert whitespace_counter(trimmed) <= 10


def test_resolve_counter_prefers_the_first_usable_option() -> None:
    assert resolve_counter(None, whitespace_counter) is whitespace_counter
    assert resolve_counter(None, None) is estimate_tokens


def test_provider_counter_uses_a_provider_hook() -> None:
    class CountingProvider:
        def count_tokens(self, text: str) -> int:
            return len(text)

    counter = provider_counter(CountingProvider())
    assert counter is not None
    assert counter("abcd") == 4
    assert counter("") == 0


def test_provider_counter_falls_back_when_the_hook_fails() -> None:
    class BrokenProvider:
        def count_tokens(self, text: str) -> int:
            raise RuntimeError("tokenizer exploded")

    counter = provider_counter(BrokenProvider())
    assert counter is not None
    assert counter("abcd") == estimate_tokens("abcd")


def test_provider_counter_is_none_without_a_hook() -> None:
    assert provider_counter(object()) is None


def test_tiktoken_counter_reports_a_missing_dependency() -> None:
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        with pytest.raises(TokenizerUnavailableError):
            tiktoken_counter("gpt-4o-mini")
    else:  # pragma: no cover - depends on the optional dependency
        counter = tiktoken_counter("gpt-4o-mini")
        assert counter("hello world") > 0
        with pytest.raises(TokenizerUnavailableError):
            tiktoken_counter("definitely-not-a-real-model")
