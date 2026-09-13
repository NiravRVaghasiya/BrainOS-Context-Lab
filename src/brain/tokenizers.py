"""Token counting strategies used by the context construction engine.

Context accounting is only meaningful if the counter is known and stable for a
whole evaluation run. This module keeps every counting strategy in one place so
the context builder, the evaluation runner, and the UI all report the same
numbers.

Three counters are supported:

``estimate_tokens``
    Dependency-free conservative estimate (``ceil(len(text) / 4)``). It is the
    default so the package keeps zero mandatory third-party dependencies.
``tiktoken_counter``
    Exact counting for OpenAI models when the optional ``tiktoken`` package is
    installed. Requested through the ``providers``/``evaluation`` environments,
    never imported unconditionally.
``provider_counter``
    Uses a provider adapter's own ``count_tokens`` when it exposes one, so a
    local or compatible endpoint can supply model-specific counting.

Counters must be monotonic in text length; :func:`truncate_to_tokens` relies on
that to trim text without exceeding a budget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

TokenCounter = Callable[[str], int]

DEFAULT_CHARS_PER_TOKEN = 4.0


class TokenizerUnavailableError(RuntimeError):
    """Raised when an exact tokenizer was requested but is not installed."""


def estimate_tokens(text: str) -> int:
    """Return a conservative, dependency-free token estimate.

    Empty text costs zero tokens. Any non-empty text costs at least one token so
    a budget can never silently treat a message as free.
    """

    if not text:
        return 0
    return max(1, -(-len(text) // int(DEFAULT_CHARS_PER_TOKEN)))


def char_ratio_counter(chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> TokenCounter:
    """Build an estimator for a model family with a known characters-per-token ratio."""

    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be greater than zero.")
    ratio = int(chars_per_token) if float(chars_per_token).is_integer() else None

    def count(text: str) -> int:
        if not text:
            return 0
        if ratio is not None:
            return max(1, -(-len(text) // ratio))
        return max(1, int(-(-len(text) // chars_per_token)))

    return count


def whitespace_counter(text: str) -> int:
    """Count whitespace-separated words.

    This is an upper-bound proxy for models whose tokenizer is close to
    word-level, and a useful cross-check on the character estimator.
    """

    if not text:
        return 0
    return max(1, len(text.split()))


def tiktoken_counter(model: str | None = None, *, encoding_name: str | None = None) -> TokenCounter:
    """Return an exact counter backed by ``tiktoken``.

    Raises :class:`TokenizerUnavailableError` when the optional dependency is
    missing or the model/encoding is unknown, so callers can fall back to
    :func:`estimate_tokens` explicitly instead of silently mixing counters.
    """

    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise TokenizerUnavailableError(
            "Exact token counting requires tiktoken. Install it with "
            "`pip install tiktoken` or use estimate_tokens()."
        ) from exc
    try:
        encoding = (
            tiktoken.get_encoding(encoding_name)
            if encoding_name
            else tiktoken.encoding_for_model(model or "gpt-4o-mini")
        )
    except (KeyError, ValueError) as exc:
        raise TokenizerUnavailableError(f"No tiktoken encoding for {model!r}.") from exc

    def count(text: str) -> int:
        if not text:
            return 0
        return len(encoding.encode(text, disallowed_special=()))

    return count


def provider_counter(provider: Any) -> TokenCounter | None:
    """Return the provider's own counter when it exposes a usable one."""

    counter = getattr(provider, "count_tokens", None)
    if not callable(counter):
        return None

    def count(text: str) -> int:
        if not text:
            return 0
        try:
            value = int(counter(text))
        except Exception:
            return estimate_tokens(text)
        return max(0, value)

    return count


def resolve_counter(*candidates: TokenCounter | None) -> TokenCounter:
    """Return the first usable counter, defaulting to :func:`estimate_tokens`."""

    for candidate in candidates:
        if callable(candidate):
            return candidate
    return estimate_tokens


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    counter: TokenCounter = estimate_tokens,
    *,
    marker: str = " …",
) -> tuple[str, bool]:
    """Trim ``text`` so it fits ``max_tokens``, returning the text and whether it was cut.

    A binary search over character prefixes keeps this cheap for long histories
    while remaining correct for any monotonic counter. The marker is included in
    the measured budget so the result never exceeds the requested size.
    """

    if max_tokens <= 0:
        return "", bool(text)
    if counter(text) <= max_tokens:
        return text, False

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if counter(text[:middle] + marker) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + marker, True


__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "TokenCounter",
    "TokenizerUnavailableError",
    "char_ratio_counter",
    "estimate_tokens",
    "provider_counter",
    "resolve_counter",
    "tiktoken_counter",
    "truncate_to_tokens",
    "whitespace_counter",
]
