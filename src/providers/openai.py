"""OpenAI provider adapter skeleton.

The SDK import is lazy so importing the package never requires credentials or
network access. Request/response normalization will be completed in Phase 1.
"""

from __future__ import annotations

from typing import Any

from .base import ProviderConfig, ProviderResponse


class OpenAIProvider:
    """Provider implementation backed by the official OpenAI Python SDK."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: Any | None = None

    def _client_or_create(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency optional
                raise RuntimeError(
                    "OpenAI support requires the optional `providers` dependency."
                ) from exc
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def list_models(self) -> list[str]:
        response = self._client_or_create().models.list()
        return [str(item.id) for item in response.data]

    def validate_credentials(self) -> bool:
        try:
            self._client_or_create().models.list()
        except Exception as exc:
            _raise_redacted(exc)
        return True

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        request = {
            "model": kwargs.pop("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
        }
        if self.config.max_tokens is not None:
            request["max_tokens"] = self.config.max_tokens
        request.update(kwargs)
        try:
            response = self._client_or_create().chat.completions.create(**request)
        except Exception as exc:
            _raise_redacted(exc)
        choice = response.choices[0]
        usage = _usage_dict(getattr(response, "usage", None))
        return ProviderResponse(
            text=str(choice.message.content or ""),
            model=str(getattr(response, "model", request["model"])),
            usage=usage,
        )


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        name: int(value)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if (value := getattr(usage, name, None)) is not None
    }


def _raise_redacted(exc: Exception) -> None:
    """Raise a provider error without echoing possible credential text."""

    message = str(exc)
    for marker in ("sk-", "Bearer ", "api_key", "API_KEY"):
        if marker in message:
            message = "Provider request failed; credentials were redacted."
            break
    raise RuntimeError(message) from None
