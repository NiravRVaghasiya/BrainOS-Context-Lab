"""Phase 9 generation path: deterministic, credential-safe model calls.

Every test here runs without a provider SDK and without a network call. The
properties under test are the ones a controlled experiment depends on: identical
sampling parameters, recorded latency, reported usage, credential-free results,
and failures that come back as data instead of exceptions.
"""

from __future__ import annotations

import json

import pytest

from evaluation.generation import (
    GenerationSettings,
    MissingCredentialError,
    ModelSpec,
    api_key_from_environment,
    budgeted_generator,
    build_provider,
    count_messages,
    generate_answer,
)
from evaluation.limits import BudgetExceeded, RunBudget, RunLimits
from tests.fakes import RecordingProvider

KEY = "sk-generation-SECRET-1234"
MESSAGES = [
    {"role": "system", "content": "You are a careful assistant."},
    {"role": "user", "content": "What production database does Project Atlas use?"},
]


def _clock(*ticks: float):
    values = iter(ticks)
    return lambda: next(values)


# --------------------------------------------------------------------------- #
# Settings and model specification
# --------------------------------------------------------------------------- #


def test_fingerprint_ignores_transport_settings_and_tracks_sampling() -> None:
    settings = GenerationSettings(model="gpt-4o-mini", temperature=0.2, max_tokens=64)
    slower = GenerationSettings(
        model="gpt-4o-mini", temperature=0.2, max_tokens=64, timeout_seconds=999.0
    )
    warmer = GenerationSettings(model="gpt-4o-mini", temperature=0.7, max_tokens=64)

    assert settings.fingerprint() == slower.fingerprint()
    assert settings.fingerprint() != warmer.fingerprint()


def test_request_kwargs_only_include_set_parameters() -> None:
    bare = GenerationSettings(model="m")
    full = GenerationSettings(model="m", temperature=0.0, max_tokens=8, top_p=0.9, seed=7)

    assert bare.request_kwargs() == {"model": "m", "temperature": 0.0}
    assert full.request_kwargs()["seed"] == 7
    assert full.request_kwargs()["max_tokens"] == 8


def test_generation_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        GenerationSettings(model="")
    with pytest.raises(ValueError):
        GenerationSettings(model="m", temperature=-1)
    with pytest.raises(ValueError):
        GenerationSettings(model="m", max_tokens=0)
    with pytest.raises(ValueError):
        GenerationSettings(model="m", top_p=1.5)


def test_model_spec_never_carries_a_credential() -> None:
    spec = ModelSpec(model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")

    payload = spec.to_dict()

    assert payload["api_key_env"] == "OPENAI_API_KEY"
    assert "api_key" not in payload
    assert KEY not in json.dumps(payload)
    # A value passed as a credential is a constructor error, not a silent field.
    with pytest.raises(TypeError):
        ModelSpec(model="gpt-4o-mini", api_key=KEY)  # type: ignore[call-arg]


def test_model_spec_applies_run_caps_without_changing_sampling() -> None:
    spec = ModelSpec(model="gpt-4o-mini", temperature=0.3, max_tokens=4096, timeout_seconds=90.0)

    capped = spec.with_limits(RunLimits(max_output_tokens=512, timeout_seconds=30.0))

    assert (capped.max_tokens, capped.timeout_seconds) == (512, 30.0)
    # Everything the model's distribution depends on is untouched: the cap is a
    # cost guard applied once, before the first request, not a per-mode knob.
    assert capped.temperature == spec.temperature
    assert capped.model == spec.model
    assert capped.name == spec.name
    # An output cap *is* a request parameter, so it moves the fingerprint; what
    # keeps a run controlled is that the cap is applied once, before the first
    # request, rather than per mode. Uncapped application changes nothing.
    assert capped.generation_settings().max_tokens == 512
    uncapped = spec.with_limits(RunLimits())
    assert (
        uncapped.generation_settings().fingerprint()
        == spec.generation_settings().fingerprint()
    )


def test_provider_config_from_spec_keeps_the_key_out_of_diagnostics() -> None:
    spec = ModelSpec(model="gpt-4o-mini", base_url="https://example.test/v1")

    config = spec.provider_config(KEY)

    assert config.api_key == KEY
    assert KEY not in repr(config)
    assert KEY not in json.dumps(config.safe_dict())
    assert config.safe_dict()["model"] == "gpt-4o-mini"


def test_api_key_from_environment_reads_the_named_variable() -> None:
    spec = ModelSpec(model="m", api_key_env="LAB_TEST_KEY")

    assert api_key_from_environment(spec, {"LAB_TEST_KEY": KEY}) == KEY


def test_api_key_from_environment_never_echoes_a_value() -> None:
    spec = ModelSpec(model="m", api_key_env="LAB_TEST_KEY")

    for environ in ({}, {"LAB_TEST_KEY": "   "}):
        with pytest.raises(MissingCredentialError) as error:
            api_key_from_environment(spec, environ)
        message = str(error.value)
        assert "LAB_TEST_KEY" in message
        assert KEY not in message


def test_build_provider_hands_the_adapter_its_configuration() -> None:
    captured: dict[str, object] = {}

    def factory(config):  # type: ignore[no-untyped-def]
        captured["config"] = config
        return "provider-object"

    spec = ModelSpec(provider="openai", model="gpt-4o-mini")
    provider = build_provider(spec, api_key=KEY, provider_factory=factory)

    assert provider == "provider-object"
    config = captured["config"]
    assert config.api_key == KEY  # type: ignore[attr-defined]
    assert KEY not in repr(config)


# --------------------------------------------------------------------------- #
# generate_answer
# --------------------------------------------------------------------------- #


def test_generate_answer_records_latency_usage_and_settings() -> None:
    provider = RecordingProvider(text="MongoDB 7", report_model="gpt-4o-mini")
    settings = GenerationSettings(model="gpt-4o-mini", max_tokens=32)

    result = generate_answer(
        provider, MESSAGES, settings, clock=_clock(100.0, 100.25)
    )

    assert result.ok is True
    assert result.text == "MongoDB 7"
    assert result.model == "gpt-4o-mini"
    assert result.latency_ms == 250.0
    assert result.usage["total_tokens"] == 15
    assert result.settings_fingerprint == settings.fingerprint()
    assert result.attempts == 1
    # The request carries exactly the constant sampling parameters.
    assert provider.settings_seen()[0]["model"] == "gpt-4o-mini"
    assert provider.settings_seen()[0]["max_tokens"] == 32


def test_generate_answer_reports_a_provider_error_as_data() -> None:
    provider = RecordingProvider(fail=f"401 unauthorized for key {KEY}")

    result = generate_answer(
        provider,
        MESSAGES,
        GenerationSettings(model="m"),
        secrets=(KEY,),
        clock=_clock(0.0, 0.5),
    )

    assert result.ok is False
    assert result.attempts == 1
    assert result.latency_ms == 500.0
    assert KEY not in (result.error or "")
    assert "[redacted]" in (result.error or "")


def test_generate_answer_treats_an_empty_completion_as_a_failure() -> None:
    provider = RecordingProvider(text="   ")

    result = generate_answer(provider, MESSAGES, GenerationSettings(model="m"))

    assert result.ok is False
    assert "empty completion" in (result.error or "")


def test_generate_answer_retries_only_when_asked() -> None:
    provider = RecordingProvider(fail=f"boom {KEY}")

    single = generate_answer(
        provider, MESSAGES, GenerationSettings(model="m"), secrets=(KEY,),
        clock=_clock(0.0, 1.0),
    )
    assert provider.request_count == 1
    assert single.attempts == 1

    retried = generate_answer(
        provider,
        MESSAGES,
        GenerationSettings(model="m"),
        secrets=(KEY,),
        max_attempts=3,
        retry_delay_seconds=0.0,
        sleep=lambda _seconds: None,
        clock=_clock(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    )
    assert provider.request_count == 4
    assert retried.attempts == 3
    assert retried.latency_ms == 1000.0


def test_count_messages_charges_content_and_message_overhead() -> None:
    assert count_messages(MESSAGES, counter=lambda text: 10) == 2 * (10 + 4)
    assert count_messages([], counter=lambda text: 10) == 0


# --------------------------------------------------------------------------- #
# budgeted_generator
# --------------------------------------------------------------------------- #


def test_budgeted_generator_charges_reported_usage() -> None:
    provider = RecordingProvider(text="ok")
    budget = RunBudget(RunLimits(max_requests=5), counter=lambda text: 100)
    generate = budgeted_generator(provider, GenerationSettings(model="m"), budget)

    result = generate(MESSAGES)

    assert result.ok is True
    assert result.prompt_token_count == 2 * (100 + 4)
    assert budget.requests == 1
    assert budget.input_tokens == 12  # reported usage wins over the counter
    assert budget.output_tokens == 3


def test_budgeted_generator_skips_an_oversized_prompt_without_calling_the_provider() -> None:
    provider = RecordingProvider()
    budget = RunBudget(RunLimits(max_input_tokens=5), counter=lambda text: 100)
    generate = budgeted_generator(provider, GenerationSettings(model="m"), budget)

    result = generate(MESSAGES)

    assert result.skipped is True
    assert "prompt_over_input_limit" in (result.error or "")
    assert result.ok is False
    assert provider.request_count == 0
    assert budget.requests == 0
    assert budget.skips == 1
    assert budget.snapshot()["skips"] == 1


def test_budgeted_generator_raises_when_the_run_allowance_is_gone() -> None:
    provider = RecordingProvider()
    budget = RunBudget(RunLimits(max_requests=0))
    generate = budgeted_generator(provider, GenerationSettings(model="m"), budget)

    with pytest.raises(BudgetExceeded):
        generate(MESSAGES)

    assert provider.request_count == 0


def test_budgeted_generator_counts_a_failure_against_the_failure_allowance() -> None:
    provider = RecordingProvider(fail=f"bad key {KEY}")
    budget = RunBudget(RunLimits(max_generation_failures=1))
    generate = budgeted_generator(
        provider, GenerationSettings(model="m"), budget, secrets=(KEY,)
    )

    first = generate(MESSAGES)

    assert first.ok is False
    assert budget.failures == 1
    assert KEY not in json.dumps(first.to_dict())
    with pytest.raises(BudgetExceeded):
        generate(MESSAGES)


def test_generation_result_carries_no_prompt_text() -> None:
    provider = RecordingProvider(text="ok")
    budget = RunBudget(RunLimits(), counter=lambda text: 7)
    generate = budgeted_generator(provider, GenerationSettings(model="m"), budget)

    payload = json.dumps(generate(MESSAGES).to_dict())

    assert "careful assistant" not in payload
    assert "Project Atlas" not in payload
