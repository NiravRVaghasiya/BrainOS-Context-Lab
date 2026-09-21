"""Phase 17 security: an automated pipeline must not widen the credential surface.

Phase 17 is the first surface where a *visitor* can start a benchmark run from a
browser, and the first that writes a whole directory of artifacts per run. So the
question these tests answer is what can leave:

* a key read from the environment by the CLI, or held by a browser session, must
  never appear in a raw run file, an aggregate, a figure, a report, a manifest, a
  persisted row, or a rendered UI view;
* the pipeline must scan its own output — both passes, including the files
  written *after* the first pass ran;
* and when a credential does reach an artifact (a provider that echoes it into an
  answer is enough, because failure examples quote answers), detecting it is not
  the end of the job: the file has to go, and the run has to say so.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.controller import UIController
from app.evaluation import UIEvaluationRunner
from evaluation import pipeline as pipeline_module
from evaluation.datasets import load_jsonl
from evaluation.experiment import ExperimentPlan
from evaluation.generation import ModelSpec
from evaluation.limits import RunLimits
from evaluation.pipeline import EXIT_OK, EXIT_STAGE_FAILED, main, run_pipeline
from providers.base import ProviderConfig
from security.scan import scan_paths
from tests.fakes import FakeLLMProvider, InMemoryEvaluationStore, RecordingProvider

KEY = "sk-pipeline-SECRET-31337"
OTHER_KEY = "sk-foreign-SECRET-99999"
DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)[:1]
SPEC = ModelSpec(model="fake-model", temperature=0.0, max_tokens=32)


def _plan(**overrides: object) -> ExperimentPlan:
    defaults: dict[str, object] = {
        "name": "quick",
        "modes": ("full_context", "rag"),
        "dataset": DATASET,
        "limits": RunLimits(max_requests=50),
    }
    defaults.update(overrides)
    return ExperimentPlan(**defaults)  # type: ignore[arg-type]


def _all_text(root: Path) -> str:
    return "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in root.rglob("*")
        if path.is_file()
    )


def _files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


# --------------------------------------------------------------------------- #
# A clean generated run
# --------------------------------------------------------------------------- #


def test_a_provider_error_carrying_the_key_never_reaches_an_artifact(tmp_path: Path) -> None:
    provider = RecordingProvider(fail=f"401 unauthorized: Bearer {KEY} (api_key={KEY})")

    result = run_pipeline(
        _plan(),
        TASKS,
        SPEC,
        provider,
        output_dir=tmp_path,
        render_plots=False,
        secrets=(KEY,),
        credential_source="environment:TEST_KEY",
    )

    assert KEY not in _all_text(tmp_path)
    assert KEY not in json.dumps(result.artifact, default=str)
    assert KEY not in result.report_markdown
    assert result.artifact["credential_check"]["clean"] is True
    assert scan_paths([str(tmp_path)], secrets=(KEY,)).clean
    # The route is recorded; the value is not.
    assert result.artifact["generation"]["credential_source"] == "environment:TEST_KEY"


@pytest.mark.requires_runtime
def test_the_scan_covers_both_passes_and_names_the_files_it_read(
    tmp_path: Path,
) -> None:
    result = run_pipeline(
        _plan(), TASKS, output_dir=tmp_path, render_plots=False, secrets=(KEY,)
    )
    check = result.artifact["credential_check"]

    assert check["clean"] is True
    assert check["files_scanned"] >= 5  # raw run files, error records, aggregates
    assert check["scanned"] == check["files_scanned"]
    assert "security.scan" in check["note"]
    report_scan = check["report_scan"]
    assert report_scan["files_scanned"] == 2  # report.md and pipeline.json
    assert report_scan["clean"] is True
    assert "Second pass" in report_scan["note"]


@pytest.mark.requires_runtime
def test_the_scan_still_runs_when_the_report_stage_is_skipped(tmp_path: Path) -> None:
    from evaluation.pipeline import expand_stages

    result = run_pipeline(
        _plan(),
        TASKS,
        output_dir=tmp_path,
        render_plots=False,
        stages=expand_stages(["raw"]),
        secrets=(KEY,),
    )
    check = result.artifact["credential_check"]

    assert result.exit_code == EXIT_OK
    assert check["clean"] is True
    assert check["files_scanned"] > 0
    assert "did not run" in check["note"]


@pytest.mark.requires_runtime
def test_figures_are_scanned_too(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib", reason="figures need matplotlib")

    result = run_pipeline(_plan(), TASKS, output_dir=tmp_path, render_plots=True)

    assert list((tmp_path / "plots").glob("*.png"))
    assert result.artifact["credential_check"]["clean"] is True
    assert scan_paths([str(tmp_path)]).clean


# --------------------------------------------------------------------------- #
# Quarantine: detection is not remediation
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_a_credential_echoed_into_an_answer_is_quarantined(tmp_path: Path) -> None:
    provider = RecordingProvider(text=f"Sure — the key is api_key={KEY}, enjoy.")

    result = run_pipeline(
        _plan(),
        TASKS,
        SPEC,
        provider,
        output_dir=tmp_path,
        render_plots=False,
        secrets=(KEY,),
    )

    assert result.exit_code == EXIT_STAGE_FAILED
    check = result.artifact["credential_check"]
    assert check["clean"] is False
    assert check["quarantined"], "a finding must delete the file that holds it"
    # Nothing credential-shaped survives anywhere in the run's output.
    assert KEY not in _all_text(tmp_path)
    assert scan_paths([str(tmp_path)], secrets=(KEY,)).clean
    # The run says so, by file name, on disk and in the report.
    assert (tmp_path / "report" / "QUARANTINED.txt").is_file()
    assert "## Quarantine" in result.report_markdown
    assert KEY not in result.report_markdown
    assert result.artifact["quarantined"] is True
    assert KEY not in json.dumps(result.artifact, default=str)


@pytest.mark.requires_runtime
def test_quarantine_detects_a_credential_it_was_not_told_about(tmp_path: Path) -> None:
    # No ``secrets=``: the scanner still recognises a credential-shaped string,
    # which is the difference between a guard and a checklist.
    provider = RecordingProvider(text=f"api_key={OTHER_KEY}")

    result = run_pipeline(
        _plan(), TASKS, SPEC, provider, output_dir=tmp_path, render_plots=False
    )

    assert result.exit_code == EXIT_STAGE_FAILED
    assert result.artifact["credential_check"]["quarantined"]
    assert OTHER_KEY not in _all_text(tmp_path)
    assert scan_paths([str(tmp_path)]).clean


@pytest.mark.requires_runtime
def test_quarantine_only_ever_deletes_this_runs_own_output(tmp_path: Path) -> None:
    outside = tmp_path / "keep-me.json"
    outside.write_text(json.dumps({"note": f"api_key={KEY}"}), encoding="utf-8")
    output = tmp_path / "run"
    provider = RecordingProvider(text=f"api_key={KEY}")

    result = run_pipeline(
        _plan(), TASKS, SPEC, provider, output_dir=output, render_plots=False, secrets=(KEY,)
    )

    deleted = result.artifact["credential_check"]["quarantined"]
    assert deleted
    root = Path(output).resolve()
    assert all(Path(path).resolve().is_relative_to(root) for path in deleted)
    # A file the pipeline did not write is not the pipeline's to delete, even
    # when it is the very thing the scan was told to look for.
    assert outside.is_file()


@pytest.mark.requires_runtime
def test_a_quarantined_report_is_not_handed_back_to_a_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the report file itself carries a credential, its text must not survive.

    The Evaluation tab renders ``report_markdown`` straight into a Markdown
    widget, so deleting ``report.md`` and then returning its contents would
    publish exactly what the deletion removed. The caller gets the quarantine
    note instead.
    """

    def leaky_report(artifact: object) -> str:
        return f"# report\n\nThe operator's key is {KEY}.\n"

    monkeypatch.setattr(pipeline_module, "markdown_report", leaky_report)

    result = run_pipeline(
        _plan(), TASKS, output_dir=tmp_path, render_plots=False, secrets=(KEY,)
    )

    assert result.exit_code == EXIT_STAGE_FAILED
    assert not (tmp_path / "report" / "report.md").exists()
    assert KEY not in result.report_markdown
    assert "## Quarantine" in result.report_markdown
    assert result.report_markdown.startswith("\n## Quarantine") or "quarantined" in (
        result.report_markdown.lower()
    )


@pytest.mark.requires_runtime
def test_a_quarantined_manifest_is_rewritten_without_its_prose(tmp_path: Path) -> None:
    provider = RecordingProvider(text=f"api_key={KEY}")

    result = run_pipeline(
        _plan(), TASKS, SPEC, provider, output_dir=tmp_path, render_plots=False, secrets=(KEY,)
    )
    manifest_path = tmp_path / "report" / "pipeline.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The rewrite keeps provenance and drops the report preview, which is the
        # block most likely to quote an answer.
        assert manifest["exit_code"] == EXIT_STAGE_FAILED
        assert manifest["credential_check"]["quarantined"]
        assert "report_markdown_preview" not in manifest
        assert KEY not in json.dumps(manifest)
    else:
        # A rewrite that is still credential-shaped is deleted rather than kept.
        assert (tmp_path / "report" / "QUARANTINED.txt").is_file()
    assert result.artifact["credential_check"]["quarantined"]


# --------------------------------------------------------------------------- #
# The CLI's environment credential
# --------------------------------------------------------------------------- #


@pytest.mark.requires_runtime
def test_the_cli_reads_a_key_from_the_environment_and_never_writes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIPELINE_TEST_KEY", KEY)
    monkeypatch.setattr(
        pipeline_module, "build_provider", lambda spec, *, api_key, **kwargs: RecordingProvider()
    )

    code = main(
        [
            "--model",
            "fake-model",
            "--api-key-env",
            "PIPELINE_TEST_KEY",
            "--limit",
            "1",
            "--no-plots",
            "--quiet",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert code == EXIT_OK
    manifest = json.loads((tmp_path / "report" / "pipeline.json").read_text(encoding="utf-8"))
    assert manifest["generation"]["enabled"] is True
    assert manifest["generation"]["credential_source"] == "environment:PIPELINE_TEST_KEY"
    assert KEY not in _all_text(tmp_path)
    assert scan_paths([str(tmp_path)], secrets=(KEY,)).clean


def test_the_cli_reports_a_missing_credential_by_variable_name_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PIPELINE_TEST_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--model",
                "fake-model",
                "--api-key-env",
                "PIPELINE_TEST_KEY",
                "--output-dir",
                str(tmp_path),
            ]
        )

    message = str(excinfo.value)
    assert "PIPELINE_TEST_KEY" in message
    assert KEY not in message
    assert _files(tmp_path) == []  # nothing was written before the refusal


# --------------------------------------------------------------------------- #
# The browser route
# --------------------------------------------------------------------------- #


def _ui_runner(tmp_path: Path, store: InMemoryEvaluationStore) -> UIEvaluationRunner:
    return UIEvaluationRunner(
        results_root=tmp_path / "results",
        evaluation_store=store,
        provider_factory=FakeLLMProvider,
    )


@pytest.mark.requires_runtime
def test_a_session_key_is_never_exported_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store = InMemoryEvaluationStore()
    runner = _ui_runner(tmp_path, store)
    before = dict(os.environ)

    view = runner.run(
        session_id="session-1",
        preset_name="quick",
        limit=1,
        generate=True,
        provider=ProviderConfig(provider="openai", model="gpt-4o-mini", api_key=KEY),
        render_plots=False,
    )

    assert view.ok
    assert dict(os.environ) == before
    assert KEY not in _all_text(tmp_path)
    assert KEY not in json.dumps(view.to_dict(), default=str)
    assert KEY not in view.report_markdown
    assert scan_paths([str(tmp_path)], secrets=(KEY,)).clean


@pytest.mark.requires_runtime
def test_the_persisted_row_holds_no_credential(tmp_path: Path) -> None:
    store = InMemoryEvaluationStore()
    runner = _ui_runner(tmp_path, store)

    runner.run(
        session_id="session-1",
        preset_name="quick",
        limit=1,
        generate=True,
        provider=ProviderConfig(provider="openai", model="gpt-4o-mini", api_key=KEY),
        render_plots=False,
    )

    saved = json.dumps(store.runs, default=str)
    assert KEY not in saved
    assert store.runs
    row = next(iter(store.runs.values()))
    assert row["metadata"]["dry_run"] is False
    assert scan_paths([str(tmp_path)], secrets=(KEY,)).clean


@pytest.mark.requires_runtime
def test_the_controller_redacts_a_session_key_out_of_every_rendered_field(
    tmp_path: Path,
) -> None:
    controller = UIController(provider_factory=FakeLLMProvider)
    controller.evaluation = _ui_runner(tmp_path, InMemoryEvaluationStore())
    sid = controller.ensure_session(None).session_id
    controller.connect(sid, provider="openai", model="gpt-4o-mini", api_key=KEY)

    view = controller.run_evaluation(
        sid, preset="quick", limit=1, generate=True, render_plots=False
    )
    rendered = "\n".join(
        [
            view.status,
            view.cost_markdown,
            view.report_markdown,
            view.artifacts_markdown,
            view.repro_markdown,
            json.dumps(view.__dict__, default=str),
            controller.export_text(sid),
        ]
    )

    assert view.ok
    assert KEY not in rendered
    assert KEY not in _all_text(tmp_path)
