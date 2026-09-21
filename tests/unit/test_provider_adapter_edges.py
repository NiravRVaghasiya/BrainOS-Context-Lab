"""Phase 18: the provider adapter's edges.

``tests/unit/test_providers.py`` pins the adapter's happy path — list, validate,
generate, override, redact. These tests pin the branches that decide what a user
sees when the optional SDK is missing, the credential is blank or rejected by a
gateway, the vendor answers with a shape the adapter did not expect, or a caller
tries to smuggle a credential through ``generate()``.

Every assertion about an error message is also an assertion about redaction: a
vendor SDK's own text can quote the credential it rejected, and that text must
never survive into the message the UI renders.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderConfig,
    ProviderConfigurationError,
    ProviderError,
    ProviderResponseError,
)

KEY = "sk-edge-SENTINEL-1234567890"


def _config(**overrides: Any) -> ProviderConfig:
    defaults: dict[str, Any] = {"model": "edge-model", "api_key": KEY}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class RecordingCompletions:
    """``client.chat.completions`` double that records what it was asked."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class RecordingClient:
    """SDK client double whose model list and completion response are scripted."""

    def __init__(
        self,
        *,
        data: Any = (),
        list_error: Exception | None = None,
        response: Any = None,
    ) -> None:
        self.data = data
        self.list_error = list_error
        self.model_list_calls = 0
        self.models = SimpleNamespace(list=self._list_models)
        self.completions = RecordingCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)

    def _list_models(self) -> Any:
        self.model_list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return SimpleNamespace(data=self.data)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.completions.requests


def _completion(**overrides: Any) -> SimpleNamespace:
    """A chat-completion response shaped like the SDK's, before normalization."""

    fields: dict[str, Any] = {
        "model": "edge-model",
        "choices": [
            SimpleNamespace(message=SimpleNamespace(content="Hello from the edge."))
        ],
        "usage": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _prompt() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Hello"}]


# --------------------------------------------------------------------------- #
# Credentials, the optional SDK, and client construction
# --------------------------------------------------------------------------- #


def test_a_blank_credential_is_refused_before_the_injected_client_is_used() -> None:
    client = RecordingClient()
    provider = OpenAIProvider(_config(api_key="   "), client=client)

    with pytest.raises(ProviderConfigurationError, match="API key"):
        provider.list_models()
    with pytest.raises(ProviderConfigurationError, match="API key"):
        provider.generate(_prompt())

    assert client.model_list_calls == 0
    assert client.requests == []


def test_a_missing_optional_sdk_names_the_extra_that_provides_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``None`` in ``sys.modules`` makes ``import openai`` raise ImportError,
    # which is what a deployment without the optional extra looks like.
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(ProviderError, match="providers"):
        OpenAIProvider(_config()).list_models()


def test_the_session_configuration_builds_the_client_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []

    class FakeOpenAI:
        def __init__(self, **options: Any) -> None:
            built.append(options)
            self.models = SimpleNamespace(
                list=lambda: SimpleNamespace(data=[SimpleNamespace(id="edge-model")])
            )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    provider = OpenAIProvider(
        _config(base_url="https://gateway.example/v1", timeout_seconds=7.5)
    )

    assert provider.list_models() == ["edge-model"]
    assert provider.list_models() == ["edge-model"]
    assert built == [
        {
            "api_key": KEY,
            "timeout": 7.5,
            "base_url": "https://gateway.example/v1",
        }
    ]


def test_a_client_that_cannot_be_constructed_reports_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingOpenAI:
        def __init__(self, **options: Any) -> None:
            raise RuntimeError(f"401 Unauthorized: invalid key {KEY}")

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=ExplodingOpenAI))

    with pytest.raises(ProviderError) as raised:
        OpenAIProvider(_config()).list_models()

    assert KEY not in str(raised.value)
    assert "[redacted]" in str(raised.value)


def test_an_injected_client_never_touches_the_optional_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam exists so a deployment (and this suite) can run without the SDK."""

    monkeypatch.setitem(sys.modules, "openai", None)
    client = RecordingClient(data=[{"id": "edge-model"}])

    assert OpenAIProvider(_config(), client=client).list_models() == ["edge-model"]


def test_a_compatible_provider_sends_the_session_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[dict[str, Any]] = []

    class FakeOpenAI:
        def __init__(self, **options: Any) -> None:
            built.append(options)
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    config = _config(provider="openai-compatible", base_url="http://localhost:8000/v1")

    OpenAICompatibleProvider(config).list_models()

    assert built[0]["base_url"] == "http://localhost:8000/v1"
    assert built[0]["api_key"] == KEY


# --------------------------------------------------------------------------- #
# Model listing
# --------------------------------------------------------------------------- #


def test_the_model_list_drops_entries_without_a_usable_identifier() -> None:
    client = RecordingClient(
        data=[
            SimpleNamespace(id="edge-model"),
            " tall-model ",
            {"id": None},
            SimpleNamespace(id="   "),
            "  ",
            {"id": "third-model"},
        ]
    )

    assert OpenAIProvider(_config(), client=client).list_models() == [
        "edge-model",
        "tall-model",
        "third-model",
    ]


def test_a_models_response_with_a_null_list_is_an_empty_list_not_an_error() -> None:
    client = RecordingClient(data=None)

    assert OpenAIProvider(_config(), client=client).list_models() == []


def test_validate_credentials_asks_the_models_endpoint_exactly_once() -> None:
    client = RecordingClient(data=[{"id": "edge-model"}])

    assert OpenAIProvider(_config(), client=client).validate_credentials() is True
    assert client.model_list_calls == 1


def test_a_gateway_error_never_survives_into_the_raised_message() -> None:
    client = RecordingClient(list_error=RuntimeError(f"quota exceeded for api_key={KEY}"))

    with pytest.raises(ProviderError) as raised:
        OpenAIProvider(_config(), client=client).validate_credentials()

    assert KEY not in str(raised.value)
    assert "[redacted]" in str(raised.value)


# --------------------------------------------------------------------------- #
# The request the adapter builds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kwarg", ["api_key", "authorization", "headers"])
def test_a_credential_cannot_be_passed_through_generate(kwarg: str) -> None:
    client = RecordingClient(response=_completion())
    provider = OpenAIProvider(_config(), client=client)

    with pytest.raises(ProviderConfigurationError, match="Credential overrides"):
        provider.generate(_prompt(), **{kwarg: KEY})

    assert client.requests == []


@pytest.mark.parametrize(
    "messages",
    [
        "not a list",
        ["not a mapping"],
        [{"role": "user"}],
        [{"role": "user", "content": None}],
        [{"role": "", "content": "hello"}],
        [{"role": 7, "content": "hello"}],
    ],
)
def test_messages_that_would_reach_the_api_are_refused(messages: Any) -> None:
    client = RecordingClient(response=_completion())

    with pytest.raises(ProviderConfigurationError):
        OpenAIProvider(_config(), client=client).generate(messages)

    assert client.requests == []


@pytest.mark.parametrize("model", ["", "   "])
def test_a_blank_model_identifier_is_refused(model: str) -> None:
    client = RecordingClient(response=_completion())
    provider = OpenAIProvider(_config(), client=client)

    with pytest.raises(ProviderConfigurationError, match="model identifier"):
        provider.generate(_prompt(), model=model)

    assert client.requests == []


def test_max_tokens_is_omitted_when_unset_and_extra_kwargs_pass_through() -> None:
    client = RecordingClient(response=_completion())
    provider = OpenAIProvider(_config(max_tokens=None), client=client)

    provider.generate(_prompt(), top_p=0.5)

    request = client.requests[0]
    assert "max_tokens" not in request
    assert request["top_p"] == 0.5
    assert request["temperature"] == 0.2
    assert request["messages"] == _prompt()


# --------------------------------------------------------------------------- #
# Normalizing the response
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("choices", [[], None, "nope"])
def test_a_response_without_a_usable_choice_is_a_response_error(choices: Any) -> None:
    client = RecordingClient(response=_completion(choices=choices))

    with pytest.raises(ProviderResponseError):
        OpenAIProvider(_config(), client=client).generate(_prompt())


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (None, ""),
        ("plain text", "plain text"),
        (
            [
                {"text": {"value": "a"}},
                {"text": "b"},
                SimpleNamespace(text="c"),
                {"text": None},
            ],
            "abc",
        ),
        (123, "123"),
    ],
)
def test_content_is_normalized_to_text(content: Any, expected: str) -> None:
    response = _completion(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )
    client = RecordingClient(response=response)

    assert OpenAIProvider(_config(), client=client).generate(_prompt()).text == expected


def test_usage_keeps_only_counts_that_are_integers() -> None:
    usage = SimpleNamespace(prompt_tokens="12", completion_tokens="n/a", total_tokens=None)
    client = RecordingClient(response=_completion(usage=usage))

    result = OpenAIProvider(_config(), client=client).generate(_prompt())

    assert result.usage == {"prompt_tokens": 12}


def test_a_response_without_usage_reports_none() -> None:
    client = RecordingClient(response=_completion(usage=None))

    assert OpenAIProvider(_config(), client=client).generate(_prompt()).usage == {}


def test_a_response_that_cannot_be_normalized_is_reported_as_a_response_error() -> None:
    response = _completion()
    response.choices = [SimpleNamespace(no_message_attribute=True)]

    class Unserializable:
        __slots__ = ()

        def __str__(self) -> str:
            raise RuntimeError("cannot render")

    response.choices[0].message = Unserializable()
    client = RecordingClient(response=response)

    with pytest.raises(ProviderResponseError):
        OpenAIProvider(_config(), client=client).generate(_prompt())


def test_raw_metadata_drops_secret_named_fields_and_redacts_the_key() -> None:
    response = _completion(usage=SimpleNamespace(total_tokens=3))
    response.api_key = KEY
    response.headers = {"authorization": f"Bearer {KEY}"}
    response.metadata = {"note": f"the key was {KEY}", "safe": "kept"}

    result = OpenAIProvider(_config(), client=RecordingClient(response=response)).generate(
        _prompt()
    )

    assert "api_key" not in result.raw
    assert result.raw["headers"] == {}
    assert result.raw["metadata"] == {"note": "the key was [redacted]", "safe": "kept"}
    assert KEY not in json.dumps(result.raw)


def test_a_sdk_shape_without_a_public_mapping_still_yields_safe_raw_metadata() -> None:
    class SlottedResponse:
        __slots__ = ("model", "choices", "usage")

        def __init__(self) -> None:
            self.model = "edge-model"
            self.choices = [
                SimpleNamespace(message=SimpleNamespace(content="slotted"))
            ]
            self.usage = None

    client = RecordingClient(response=SlottedResponse())

    result = OpenAIProvider(_config(), client=client).generate(_prompt())

    assert result.text == "slotted"
    assert KEY not in json.dumps(result.raw)
