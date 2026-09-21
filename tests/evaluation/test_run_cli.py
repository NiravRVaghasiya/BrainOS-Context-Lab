"""Phase 9 additions to the single-mode run CLI (``python -m evaluation.run``).

The retrieval-only path is covered by the Phase 6/7 tests. What matters here is
that ``--generate`` reaches the provider through the same prompt the experiment
uses, that the credential never comes from the command line, and that a
misconfigured provider fails before any request is sent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import run as run_cli
from tests.fakes import RecordingProvider

DATASET = Path("benchmarks/context_rot/dataset.jsonl")


def test_run_parser_does_not_accept_an_abbreviated_credential_flag() -> None:
    parser = run_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--api-key", "sk-secret", "--mode", "brainos", "--output", "o.json"])

    assert "api_key_env" in {action.dest for action in parser._actions}


@pytest.mark.requires_runtime
def test_an_ungraded_run_prints_no_accuracy(tmp_path, capsys) -> None:
    output = tmp_path / "ungraded.json"

    code = run_cli.main(["--mode", "brainos", "--limit", "1", "--output", str(output)])
    summary = capsys.readouterr().out

    assert code == 0
    assert "accuracy=—" in summary
    assert "accuracy=0.000" not in summary
    assert "No answers were graded" in summary


def test_generate_requires_a_model(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        run_cli.main(["--mode", "brainos", "--generate", "--output", str(tmp_path / "o.json")])

    assert "--model" in str(error.value)


def test_an_incompatible_endpoint_fails_before_any_request(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    with pytest.raises(SystemExit) as error:
        run_cli.main(
            [
                "--mode",
                "brainos",
                "--generate",
                "--provider",
                "openai-compatible",
                "--model",
                "stub-model",
                "--dataset",
                str(DATASET),
                "--limit",
                "1",
                "--output",
                str(tmp_path / "o.json"),
            ]
        )

    assert "base_url" in str(error.value)
    assert not (tmp_path / "o.json").exists()


@pytest.mark.requires_runtime
def test_generated_run_grades_its_own_answers_and_records_usage(
    tmp_path, monkeypatch
) -> None:
    provider = RecordingProvider(text="MongoDB 7")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(run_cli, "build_provider", lambda *_a, **_k: provider)
    output = tmp_path / "generated.json"

    code = run_cli.main(
        [
            "--mode",
            "brainos",
            "--generate",
            "--provider",
            "openai-compatible",
            "--base-url",
            "http://127.0.0.1:9/v1",
            "--model",
            "stub-model",
            "--dataset",
            str(DATASET),
            "--limit",
            "1",
            "--max-tokens",
            "32",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["config"]["temperature"] == 0.0
    assert payload["generation_settings"]["model"] == "stub-model"
    assert payload["generation_settings"]["max_tokens"] == 32
    assert payload["generation_usage"]["requests"] == 1
    assert payload["generation_usage"]["total_tokens"] > 0
    assert payload["aggregate_metrics"]["graded_answer_count"] == 1
    assert payload["aggregate_metrics"]["answer_accuracy"] == 1.0
    assert payload["task_results"][0]["answer"] == "MongoDB 7"
    assert provider.request_count == 1
