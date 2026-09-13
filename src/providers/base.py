"""Provider-neutral interfaces and secret-safe configuration primitives.

The application talks to an LLM through this module instead of importing a
provider SDK.  Provider implementations should return :class:`ProviderResponse`
objects and raise :class:`ProviderError` subclasses; neither SDK exceptions nor
SDK response objects should cross the application boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol

_SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "client_secret",
        "refresh_token",
    }
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_KEY_VALUE_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|authorization|password|secret|token|access[_-]?token|"
    r"auth[_-]?token|client[_-]?secret|refresh[_-]?token)\b\s*[=:]\s*)"
    r"[^\s,;]+"
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{3,}\b")


class ProviderError(RuntimeError):
    """Base class for errors exposed by a provider adapter.

    The message is sanitized before construction by the adapter.  Keeping a
    dedicated exception type also lets callers display a safe error without
    inspecting a provider SDK exception or its request internals.
    """


class ProviderConfigurationError(ProviderError, ValueError):
    """The active provider configuration cannot be used."""


class ProviderResponseError(ProviderError):
    """The provider returned a response that cannot be normalized."""


class UnsupportedProviderError(ProviderConfigurationError):
    """The provider factory does not know the requested provider name."""


@dataclass(frozen=True)
class ProviderConfig:
    """Settings for one active provider session.

    ``api_key`` is intentionally excluded from ``repr``.  It is held in
    process memory only by the active session and is never included in
    :meth:`safe_dict`, provider errors, or normalized response metadata.
    """

    provider: str = "openai"
    model: str = ""
    api_key: str = field(default="", repr=False)
    base_url: str | None = field(default=None, repr=False)
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ProviderConfigurationError("A provider name is required.")
        if not isinstance(self.temperature, (int, float)) or not isfinite(self.temperature):
            raise ProviderConfigurationError("Temperature must be a finite number.")
        if self.temperature < 0:
            raise ProviderConfigurationError("Temperature cannot be negative.")
        if self.max_tokens is not None and (
            not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise ProviderConfigurationError("max_tokens must be a positive integer or None.")
        if not isinstance(self.timeout_seconds, (int, float)) or not isfinite(self.timeout_seconds):
            raise ProviderConfigurationError("timeout_seconds must be a finite number.")
        if self.timeout_seconds <= 0:
            raise ProviderConfigurationError("timeout_seconds must be greater than zero.")

    def safe_dict(self) -> dict[str, Any]:
        """Return configuration suitable for UI diagnostics or evaluation metadata.

        The API key is deliberately absent rather than masked.  Omitting it
        prevents a masked value from being mistaken for a credential that can
        be restored or used later.
        """

        safe_base_url = (
            redact_text(self.base_url, secrets=(self.api_key,))
            if self.base_url
            else None
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": safe_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized provider output used by the application service layer."""

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMProvider(Protocol):
    """Provider contract required by chat and evaluation modes."""

    def list_models(self) -> list[str]:
        """Return models available to the configured credentials."""

    def validate_credentials(self) -> bool:
        """Validate the active credentials without exposing them."""

    def generate(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> ProviderResponse:
        """Generate one response from model-ready messages."""


def redact_text(text: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Remove credentials and common credential-shaped values from text.

    This function is intentionally conservative: it is used on exception
    messages and response metadata, never on user prompts.  The configured
    secret values are replaced first, then common bearer/key formats are
    scrubbed for cases where an SDK exception contains a copy of a credential
    in a slightly transformed form.
    """

    result = str(text)
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        result = result.replace(secret, "[redacted]")
    result = _BEARER_RE.sub(r"\1[redacted]", result)
    result = _KEY_VALUE_RE.sub(r"\1[redacted]", result)
    result = _OPENAI_KEY_RE.sub("[redacted]", result)
    return result


def safe_error_message(exc: BaseException, *, secrets: tuple[str, ...] = ()) -> str:
    """Return a non-empty, credential-redacted provider error message."""

    message = redact_text(str(exc), secrets=secrets).strip()
    return message or "Provider request failed."


def sanitize_value(value: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    """Convert response metadata to safe JSON-like values.

    Secret-named mapping fields are omitted from diagnostic output entirely.
    Other strings are scrubbed using :func:`redact_text`, and arbitrary SDK
    objects are represented by their public mapping when possible.
    """

    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower().replace("-", "_") in _SECRET_FIELD_NAMES:
                continue
            safe[key_text] = sanitize_value(item, secrets=secrets)
        return safe
    if isinstance(value, (list, tuple)):
        return [sanitize_value(item, secrets=secrets) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return redact_text(value, secrets=secrets) if isinstance(value, str) else value
    for method_name in ("model_dump", "dict"):
        dump = getattr(value, method_name, None)
        if callable(dump):
            try:
                return sanitize_value(dump(), secrets=secrets)
            except Exception:
                break
    if hasattr(value, "__dict__"):
        return sanitize_value(vars(value), secrets=secrets)
    return redact_text(str(value), secrets=secrets)


def _raise_redacted(exc: Exception, *, secrets: tuple[str, ...] = ()) -> None:
    """Raise a public provider error without leaking SDK exception details.

    This helper remains module-level for compatibility with the original
    scaffold while centralizing the Phase 1 secret-redaction behavior.
    """

    raise ProviderError(safe_error_message(exc, secrets=secrets)) from None
