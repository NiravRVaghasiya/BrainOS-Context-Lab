"""Adapter for providers exposing an OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

from .base import ProviderConfig
from .openai import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """OpenAI SDK adapter configured with a caller-supplied base URL.

    This supports hosted gateways and local servers that implement the
    ``/chat/completions`` and ``/models`` OpenAI API shapes.  The endpoint is
    supplied per active session and is never inferred from the browser.
    """

    def __init__(self, config: ProviderConfig, *, client: Any | None = None) -> None:
        if not isinstance(config.base_url, str) or not config.base_url.strip():
            raise ValueError("An OpenAI-compatible provider requires a base_url.")
        super().__init__(config, client=client)
