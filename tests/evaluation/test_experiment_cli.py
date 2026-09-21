"""Phase 18: the controlled experiment's CLI surface.

``tests/evaluation/test_experiment.py`` drives the run itself. What it did not
exercise is the part a researcher actually types: the summary that follows a
run, the exit codes that say whether the numbers may be reported at all, and the
credential handling on the command line. Those paths decide whether a failed
comparison is noticed or quoted, so they are pinned here:

* exit **0** — the comparison held;
* exit **2** — the controlled-comparison checks failed (do not report these);
* exit **3** — a cost ceiling stopped the run (partial results, stated).

The failing-provider case uses the *real* OpenAI adapter over an injected client,
so the redaction under test is the production one, not a fake that already
redacts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.experiment import BILLING_NOTICE, build_parser, main
from providers import OpenAIProvider, ProviderConfig
from tests.fakes import RecordingProvider

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
KEY = "sk-cli-SENTINEL-2468013579"


def _dry_run_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--dry-run",
        "--dataset",
        str(DATASET),
        "--limit",
        "1",
        "--modes",
        "full_context,brainos",
        "--output",
        str(tmp_path / "artifact.json"),
        *extra,
    ]


def _single_mode_args(tmp_path: Path, *extra: str) -> list[str]:
    """One mode and one task, so a ceiling cannot truncate the plan's coverage."""

    return [
        "--dataset",
        str(DATASET),
        "--limit",
        "1",
        "--modes",
        "full_context",
        "--model",
        "fake-model",
        "--api-key-env",
        "LAB_CLI_KEY",
        "--output",
        str(tmp_path / "artifact.json"),
        *extra,
    ]


def _generated_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--dataset",
        str(DATASET),
        "--limit",
        "1",
        "--modes",
        "full_context,brainos",
        "--model",
        "fake-model",
        "--api-key-env",
        "LAB_CLI_KEY",
        "--output",
        str(tmp_path / "artifact.json"),
        *extra,
    ]


def _exploding_client(key: str) -> Any:
    """An SDK client whose call raises exactly what an auth failure looks like."""

    class ExplodingCompletions:
        def create(self, **request: Any) -> Any:
            raise RuntimeError(f"401 Unauthorized: invalid api key {key}")

    return SimpleNamespace(
        chat=SimpleNamespace(completions=ExplodingCompletions()),
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[])),
    )


def _provider_from_spec(spec: Any, *, api_key: str) -> OpenAIProvider:
    config = ProviderConfig(
        provider=spec.provider, model=spec.model, api_key=api_key
    )
    return OpenAIProvider(config, client=_exploding_client(api_key))


# --------------------------------------------------------------------------- #
# The summary a researcher reads
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_a_dry_run_summary_states_the_billing_notice_and_the_ungraded_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(_dry_run_args(tmp_path))

    printed = capsys.readouterr().out
    assert code == 0
    assert BILLING_NOTICE in printed
    assert "dry_run=True" in printed
    assert "No answers were graded: accuracy and faithfulness are unset, not zero." in printed
    # The table and the cost report are both part of the receipt.
    assert "accuracy" in printed and "mean_tokens" in printed
    assert "cost: requests=0" in printed
    assert f"artifact: {tmp_path / 'artifact.json'}" in printed


@pytest.mark.requires_runtime
def test_the_quiet_flag_suppresses_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(_dry_run_args(tmp_path, "--quiet"))

    assert code == 0
    assert capsys.readouterr().out == ""


@pytest.mark.requires_runtime
def test_a_generated_summary_reports_graded_answers_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import evaluation.experiment as experiment

    # The first committed task's expected answer, so the run is graded rather
    # than merely generated.
    monkeypatch.setenv("LAB_CLI_KEY", KEY)
    monkeypatch.setattr(
        experiment,
        "build_provider",
        lambda spec, *, api_key, **kwargs: RecordingProvider(text="MongoDB 7"),
    )

    code = main(_generated_args(tmp_path))

    printed = capsys.readouterr().out
    assert code == 0
    assert "No answers were graded" not in printed
    assert "cost: requests=" in printed and "failures=0" in printed


# --------------------------------------------------------------------------- #
# Exit codes: 2 (uncontrolled) and 3 (ceiling)
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_a_gateway_that_routes_elsewhere_exits_two_and_says_not_to_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import evaluation.experiment as experiment

    monkeypatch.setenv("LAB_CLI_KEY", KEY)
    monkeypatch.setattr(
        experiment,
        "build_provider",
        lambda spec, *, api_key, **kwargs: RecordingProvider(
            text="MongoDB 7", report_model="some-other-model"
        ),
    )

    code = main(_generated_args(tmp_path))

    captured = capsys.readouterr()
    assert code == 2
    assert "violation: " in captured.err
    assert "Do not report these numbers" in captured.err or (
        "do not report these numbers" in captured.err
    )


@pytest.mark.requires_runtime
def test_a_ceiling_stop_that_keeps_coverage_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A repeated trial is the case where a ceiling stop is still controlled.

    The task ran under the mode the plan selected; only its second trial was
    refused, so the control check has nothing to complain about and the exit
    code is the ceiling's own (3) rather than the comparison's (2).
    """

    import evaluation.experiment as experiment

    monkeypatch.setenv("LAB_CLI_KEY", KEY)
    monkeypatch.setattr(
        experiment,
        "build_provider",
        lambda spec, *, api_key, **kwargs: RecordingProvider(text="MongoDB 7"),
    )

    code = main(
        _single_mode_args(tmp_path, "--trials", "2", "--max-requests", "1")
    )

    captured = capsys.readouterr()
    assert code == 3
    assert "aborted: " in captured.err
    assert "cost: requests=1" in captured.out


@pytest.mark.requires_runtime
def test_a_ceiling_stop_that_truncates_coverage_exits_two_not_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stricter code wins: a truncated comparison may not be reported.

    When the budget refuses the second mode's first request, the run is both
    ceiling-truncated *and* uncontrolled. Exiting 3 would suggest a partial
    result is reportable, so the controlled-comparison check takes precedence
    and the run says so on stderr.
    """

    import evaluation.experiment as experiment

    monkeypatch.setenv("LAB_CLI_KEY", KEY)
    monkeypatch.setattr(
        experiment,
        "build_provider",
        lambda spec, *, api_key, **kwargs: RecordingProvider(text="MongoDB 7"),
    )

    code = main(_generated_args(tmp_path, "--max-requests", "1"))

    captured = capsys.readouterr()
    assert code == 2
    assert "aborted: " in captured.err
    assert "violation: " in captured.err
    assert "do not report these numbers" in captured.err


@pytest.mark.requires_runtime
def test_a_run_that_never_generated_still_writes_the_requested_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-mode run files exist even when every call failed, with no key in them."""

    import evaluation.experiment as experiment

    monkeypatch.setenv("LAB_CLI_KEY", KEY)
    monkeypatch.setattr(experiment, "build_provider", _provider_from_spec)
    runs_dir = tmp_path / "runs"

    code = main(_generated_args(tmp_path, "--runs-dir", str(runs_dir), "--quiet"))

    assert code == 0
    written = sorted(path.name for path in runs_dir.glob("*.json"))
    assert written == ["brainos.json", "full_context.json"]
    assert KEY not in "".join(
        path.read_text(encoding="utf-8") for path in runs_dir.glob("*.json")
    )


# --------------------------------------------------------------------------- #
# Credentials on the command line
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_a_failing_adapter_never_puts_the_key_in_a_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every call fails with an auth error that quotes the key; nothing echoes it."""

    import evaluation.experiment as experiment

    monkeypatch.setenv("LAB_CLI_KEY", KEY)
    monkeypatch.setattr(experiment, "build_provider", _provider_from_spec)

    code = main(_generated_args(tmp_path))

    captured = capsys.readouterr()
    assert code == 0
    assert KEY not in captured.out + captured.err
    assert KEY not in (tmp_path / "artifact.json").read_text(encoding="utf-8")


def test_a_missing_credential_names_the_variable_and_never_a_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAB_CLI_ABSENT_KEY", raising=False)

    with pytest.raises(SystemExit) as raised:
        main(_generated_args(tmp_path, "--api-key-env", "LAB_CLI_ABSENT_KEY"))

    message = str(raised.value)
    assert "LAB_CLI_ABSENT_KEY" in message
    assert KEY not in message


def test_the_parser_refuses_a_zero_trial_count(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        main(_dry_run_args(tmp_path, "--trials", "0"))

    assert "--trials must be at least 1." in str(raised.value)


def test_the_parser_does_not_abbreviate_the_credential_flag() -> None:
    """``--api-key`` must not be accepted as a short form of ``--api-key-env``."""

    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--api-key", KEY, "--output", "out.json"])

    parsed = parser.parse_args(["--api-key-env", "LAB_CLI_KEY", "--output", "out.json"])

    assert parsed.api_key_env == "LAB_CLI_KEY"
