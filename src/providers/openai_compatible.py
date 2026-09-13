"""Adapter for providers exposing an OpenAI-compatible API."""

from __future__ import annotations

from .base import ProviderConfig
from .openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """OpenAI SDK adapter configured with a caller-supplied base URL."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            raise ValueError("An OpenAI-compatible provider requires a base_url.")
        super().__init__(config)
