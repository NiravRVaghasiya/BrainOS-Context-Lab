"""Provider adapters for OpenAI and OpenAI-compatible APIs."""

from __future__ import annotations

from .base import (
    LLMProvider,
    ProviderConfig,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponse,
    ProviderResponseError,
    UnsupportedProviderError,
)
from .openai import OpenAIProvider
from .openai_compatible import OpenAICompatibleProvider


def create_provider(config: ProviderConfig) -> LLMProvider:
    """Build the adapter selected by a session's provider configuration.

    The factory is deliberately small: adding a provider should require a new
    adapter and one explicit registry entry, not changes to BrainOS, context
    construction, or evaluation code.
    """

    name = config.provider.strip().lower().replace("_", "-")
    if name == "openai":
        return OpenAIProvider(config)
    if name in {"openai-compatible", "compatible"}:
        return OpenAICompatibleProvider(config)
    raise UnsupportedProviderError(f"Unsupported provider: {config.provider}")


__all__ = [
    "LLMProvider",
    "ProviderConfig",
    "ProviderConfigurationError",
    "ProviderError",
    "ProviderResponse",
    "ProviderResponseError",
    "UnsupportedProviderError",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "create_provider",
]
