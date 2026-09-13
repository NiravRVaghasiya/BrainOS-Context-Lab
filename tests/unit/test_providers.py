from types import SimpleNamespace

import pytest

from providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderConfig,
    ProviderConfigurationError,
    ProviderError,
    UnsupportedProviderError,
    create_provider,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            model="fake-model",
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Hello from the fake model."))
            ],
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=5, total_tokens=13),
        )


class FakeClient:
    def __init__(self) -> None:
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id="fake-model"), {"id": "second-model"}]
            )
        )
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_provider_lists_models_and_normalizes_generation() -> None:
    client = FakeClient()
    provider = OpenAIProvider(
        ProviderConfig(model="fake-model", api_key="session-secret"), client=client
    )

    assert provider.list_models() == ["fake-model", "second-model"]
    assert provider.validate_credentials() is True
    result = provider.generate([{"role": "user", "content": "Hello"}])

    assert result.text == "Hello from the fake model."
    assert result.model == "fake-model"
    assert result.usage == {
        "prompt_tokens": 8,
        "completion_tokens": 5,
        "total_tokens": 13,
    }
    assert client.completions.request == {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
    }


def test_request_overrides_do_not_mutate_config() -> None:
    client = FakeClient()
    config = ProviderConfig(
        model="configured-model", api_key="session-secret", max_tokens=100
    )
    provider = OpenAIProvider(config, client=client)

    provider.generate(
        [{"role": "user", "content": "Hello"}],
        model="one-off-model",
        temperature=0.0,
        max_tokens=20,
    )

    assert client.completions.request["model"] == "one-off-model"
    assert client.completions.request["temperature"] == 0.0
    assert client.completions.request["max_tokens"] == 20
    assert config.model == "configured-model"
    assert config.max_tokens == 100


def test_compatible_provider_requires_endpoint_and_factory_selects_it() -> None:
    config = ProviderConfig(
        provider="openai-compatible",
        model="local-model",
        api_key="session-secret",
        base_url="http://localhost:8000/v1",
    )

    provider = create_provider(config)
    assert isinstance(provider, OpenAICompatibleProvider)

    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleProvider(ProviderConfig(model="local-model", api_key="session-secret"))


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(UnsupportedProviderError):
        create_provider(ProviderConfig(provider="unknown", api_key="session-secret"))


def test_provider_errors_redact_api_keys() -> None:
    class FailingModels:
        def list(self):
            raise RuntimeError("request failed for session-secret")

    client = SimpleNamespace(models=FailingModels())
    provider = OpenAIProvider(
        ProviderConfig(model="fake-model", api_key="session-secret"), client=client
    )

    with pytest.raises(ProviderError) as raised:
        provider.list_models()

    assert "session-secret" not in str(raised.value)
    assert "[redacted]" in str(raised.value)


def test_secret_is_not_in_config_diagnostics_or_repr() -> None:
    config = ProviderConfig(
        model="fake-model",
        api_key="session-secret",
        base_url="https://gateway.example/v1?api_key=session-secret",
    )

    assert "session-secret" not in repr(config)
    assert "api_key" not in config.safe_dict()
    assert "session-secret" not in str(config.safe_dict())


def test_response_metadata_drops_secret_fields() -> None:
    response = {
        "api_key": "session-secret",
        "model": "fake-model",
        "choices": [{"message": {"content": "safe response"}}],
        "usage": {"total_tokens": 2},
    }
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **request: response)
        )
    )
    provider = OpenAIProvider(
        ProviderConfig(model="fake-model", api_key="session-secret"), client=client
    )

    result = provider.generate([{"role": "user", "content": "Hello"}])

    assert result.text == "safe response"
    assert "api_key" not in result.raw
    assert "session-secret" not in str(result.raw)


def test_provider_requires_model_and_messages_shape() -> None:
    provider = OpenAIProvider(
        ProviderConfig(api_key="session-secret"), client=FakeClient()
    )

    with pytest.raises(ProviderConfigurationError, match="model"):
        provider.generate([])

    provider = OpenAIProvider(
        ProviderConfig(model="fake-model", api_key="session-secret"), client=FakeClient()
    )
    with pytest.raises(ProviderConfigurationError, match="messages"):
        provider.generate("not-a-list")  # type: ignore[arg-type]
