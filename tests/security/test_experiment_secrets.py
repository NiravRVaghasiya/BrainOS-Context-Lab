"""Phase 9 security: the generation path must not widen the credential surface.

Phase 9 adds the first route in this repository that holds a real provider key
*while* producing an artifact. So the tests here are about what can leave: a
provider error, a run artifact, a per-mode run file, a model specification, or a
command line. None of them may contain a key value, and the CLI must not even
accept one as an argument.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.datasets import load_jsonl
from evaluation.experiment import (
    ExperimentPlan,
    build_parser,
    main,
    run_controlled_experiment,
)
from evaluation.generation import (
    GenerationSettings,
    MissingCredentialError,
    ModelSpec,
    api_key_from_environment,
    generate_answer,
)
from evaluation.limits import RunLimits
from tests.fakes import RecordingProvider

KEY = "sk-experiment-SECRET-9999"
DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)[:1]
SPEC = ModelSpec(model="fake-model")


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        name="security", modes=("brainos",), limits=RunLimits(max_requests=100)
    )


def test_provider_error_carrying_the_key_never_reaches_the_artifact() -> None:
    provider = RecordingProvider(fail=f"401 unauthorized: Bearer {KEY} (api_key={KEY})")

    run = run_controlled_experiment(
        _plan(), TASKS, SPEC, provider, secrets=(KEY,), clock=lambda: 1.0
    )

    rendered = json.dumps(run.to_dict())
    assert KEY not in rendered
    assert "[redacted]" in rendered
    # The failure is still visible as a failure: redaction must not hide it.
    assert run.mode_result("brainos").generation_usage["failures"] == 1
    assert run.mode_result("brainos").aggregate_metrics["graded_answer_count"] == 0


def test_no_artifact_surface_carries_a_credential() -> None:
    provider = RecordingProvider(text="MongoDB 7")

    run = run_controlled_experiment(_plan(), TASKS, SPEC, provider, clock=lambda: 0.5)

    surfaces = [
        json.dumps(run.to_dict()),
        json.dumps(run.to_dict()["model"]),
        json.dumps(run.budget),
        json.dumps(run.mode_result("brainos").to_run_dict()),
        json.dumps(SPEC.provider_config(KEY).safe_dict()),
    ]
    for surface in surfaces:
        assert KEY not in surface
        assert "sk-" not in surface
    # The spec records where a key *would* come from; it never records one.
    assert run.to_dict()["model"]["api_key_env"] == "OPENAI_API_KEY"


def test_generation_settings_and_results_hold_no_credential() -> None:
    provider = RecordingProvider(text="answer")

    result = generate_answer(
        provider,
        [{"role": "user", "content": "hello"}],
        GenerationSettings(model="fake-model"),
        secrets=(KEY,),
        clock=lambda: 0.0,
    )

    assert KEY not in json.dumps(result.to_dict())
    assert KEY not in json.dumps(GenerationSettings(model="m").to_dict())


def test_missing_credential_message_names_the_variable_not_a_value() -> None:
    spec = ModelSpec(model="fake-model", api_key_env="LAB_MISSING_KEY")

    with pytest.raises(MissingCredentialError) as error:
        api_key_from_environment(spec, {})

    assert "LAB_MISSING_KEY" in str(error.value)
    assert KEY not in str(error.value)


def test_cli_does_not_accept_a_credential_argument() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--api-key", KEY, "--output", "out.json"])

    # The only credential flag names an environment variable.
    options = {action.dest for action in parser._actions}
    assert "api_key" not in options
    assert "api_key_env" in options


def test_cli_reports_a_missing_credential_without_echoing_a_value(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--model",
                "gpt-4o-mini",
                "--dataset",
                str(DATASET),
                "--output",
                str(tmp_path / "out.json"),
                "--quiet",
            ]
        )

    assert "OPENAI_API_KEY" in str(error.value)
    assert KEY not in str(error.value)
