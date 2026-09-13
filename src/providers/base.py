"""Provider-neutral LLM interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderConfig:
    """Provider request settings; API keys must remain process-local."""

    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout_seconds: float = 60.0


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
