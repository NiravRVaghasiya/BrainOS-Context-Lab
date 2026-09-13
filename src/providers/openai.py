"""OpenAI and OpenAI-compatible chat-completions provider adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .base import (
    ProviderConfig,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponse,
    ProviderResponseError,
    _raise_redacted,
    safe_error_message,
    sanitize_value,
)


class OpenAIProvider:
    """Provider implementation backed by the official OpenAI Python SDK.

    The SDK is imported and the client is created lazily.  Importing the
    application therefore does not require the optional SDK, make a network
    request, or read a credential.  A client can be injected for deterministic
    tests and for deployments that wrap the SDK.
    """

    def __init__(self, config: ProviderConfig, *, client: Any | None = None) -> None:
        self.config = config
        self._client: Any | None = client

    def _client_or_create(self) -> Any:
        self._require_api_key()
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError:  # pragma: no cover - dependency optional
            raise ProviderError(
                "OpenAI support requires the optional `providers` dependency."
            ) from None

        client_options: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
        }
        if self.config.base_url:
            client_options["base_url"] = self.config.base_url
        try:
            self._client = OpenAI(**client_options)
        except Exception as exc:  # pragma: no cover - exercised with injected clients
            self._raise_request_error(exc)
        return self._client

    def list_models(self) -> list[str]:
        """List model identifiers available to the active credential."""

        try:
            response = self._client_or_create().models.list()
            data = _get(response, "data", [])
            if data is None:
                return []
            return [model_id for item in data if (model_id := _model_id(item))]
        except ProviderError:
            raise
        except Exception as exc:
            self._raise_request_error(exc)
        return []  # pragma: no cover - _raise_request_error always raises

    def validate_credentials(self) -> bool:
        """Perform a low-cost authenticated model-list request."""

        self.list_models()
        return True

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        """Generate a response and normalize SDK-specific response objects.

        ``model``, ``temperature``, and ``max_tokens`` may be overridden for a
        single request.  Credentials cannot be supplied through ``kwargs``;
        they belong only in the session's :class:`ProviderConfig`.
        """

        request_messages = _copy_messages(messages)
        request = self._build_request(request_messages, kwargs)
        try:
            response = self._client_or_create().chat.completions.create(**request)
        except ProviderError:
            raise
        except Exception as exc:
            self._raise_request_error(exc)

        try:
            choice = _first_choice(response)
            message = _get(choice, "message", {})
            text = _content_text(_get(message, "content", ""))
            model = str(_get(response, "model", request["model"]) or request["model"])
            usage = _usage_dict(_get(response, "usage", None))
            raw = sanitize_value(response, secrets=(self.config.api_key,))
            raw_dict = raw if isinstance(raw, dict) else {}
            return ProviderResponse(text=text, model=model, usage=usage, raw=raw_dict)
        except ProviderResponseError:
            raise
        except Exception:
            raise ProviderResponseError(
                "Provider returned an invalid chat-completion response."
            ) from None

    def _build_request(
        self, messages: list[dict[str, str]], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        forbidden = {"api_key", "authorization", "headers"} & set(kwargs)
        if forbidden:
            raise ProviderConfigurationError(
                "Credential overrides are not accepted by the provider interface."
            )

        request: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
        }
        max_tokens = kwargs.pop("max_tokens", self.config.max_tokens)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        request.update(kwargs)

        model = request["model"]
        if not isinstance(model, str) or not model.strip():
            raise ProviderConfigurationError("A model identifier is required for generation.")
        return request

    def _require_api_key(self) -> None:
        if not isinstance(self.config.api_key, str) or not self.config.api_key.strip():
            raise ProviderConfigurationError(
                "An API key is required for the active provider session."
            )

    def _raise_request_error(self, exc: Exception) -> None:
        """Raise a safe error while preserving no SDK exception text."""

        message = safe_error_message(exc, secrets=(self.config.api_key,))
        raise ProviderError(message) from None


def _get(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a test/dict response."""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _model_id(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    model_id = _get(item, "id", "")
    return str(model_id).strip() if model_id is not None else ""


def _copy_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Validate and copy messages without putting message data in errors."""

    if not isinstance(messages, list):
        raise ProviderConfigurationError("messages must be a list of role/content mappings.")
    copied: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ProviderConfigurationError("Each message must be a role/content mapping.")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str):
            raise ProviderConfigurationError(
                "Each message must contain a string role and string content."
            )
        copied.append({"role": role, "content": content})
    return copied


def _first_choice(response: Any) -> Any:
    choices = _get(response, "choices", None)
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ProviderResponseError("Provider returned no chat-completion choices.")
    return choices[0]


def _content_text(content: Any) -> str:
    """Normalize string and modern content-part response formats."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for part in content:
            text = _get(part, "text", None)
            if isinstance(text, Mapping):
                text = text.get("value", "")
            if text is not None:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    output: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _get(usage, name, None)
        if value is None:
            continue
        try:
            output[name] = int(value)
        except (TypeError, ValueError):
            continue
    return output


__all__ = ["OpenAIProvider", "_raise_redacted"]
