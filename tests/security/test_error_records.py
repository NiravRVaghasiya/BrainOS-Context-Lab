"""Phase 12 security: the error-analysis layer must not widen the secret surface.

Phase 12 reads evaluation artifacts and turns them into reports and per-failure
records. Those files are the most tempting place for a credential to reappear:
they are long, deeply nested, and written to disk by a command anyone can run.
The tests here pin three properties:

* a credential carried in a provider error stays out of the report *and* the
  JSONL failure records;
* the provenance block records paths and digests, never environment values;
* the module reads a fixed set of artifact keys, so adding a field to a run
  configuration cannot silently start copying secrets into a report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import errors
from evaluation.datasets import load_jsonl
from evaluation.experiment import ExperimentPlan, run_controlled_experiment
from evaluation.generation import ModelSpec
from evaluation.limits import RunLimits
from tests.fakes import RecordingProvider

KEY = "sk-phase12-SECRET-4242"
DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)[:1]
SPEC = ModelSpec(model="fake-model")


def _failing_run() -> dict:
    """Produce a real artifact whose provider error carried the credential."""

    plan = ExperimentPlan(
        name="security", modes=("brainos",), limits=RunLimits(max_requests=10)
    )
    provider = RecordingProvider(fail=f"401 unauthorized: Bearer {KEY} (api_key={KEY})")
    run = run_controlled_experiment(
        plan, TASKS, SPEC, provider, secrets=(KEY,), clock=lambda: 1.0
    )
    return run.to_dict()


@pytest.mark.requires_runtime
def test_the_redacted_artifact_still_carries_no_key_into_the_report(tmp_path) -> None:
    artifact = _failing_run()
    assert KEY not in json.dumps(artifact)

    report = errors.error_report([artifact], tasks=TASKS)
    records_path = tmp_path / "records.jsonl"
    records = errors.collect_failures([artifact], tasks=TASKS)
    errors.write_failure_records(records, records_path)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    for rendered in (
        json.dumps(report),
        report_path.read_text(encoding="utf-8"),
        records_path.read_text(encoding="utf-8"),
    ):
        assert KEY not in rendered
        assert "api_key" not in rendered


@pytest.mark.requires_runtime
def test_the_failure_record_is_built_from_named_fields_not_the_whole_artifact() -> None:
    """A configuration field the taxonomy does not read cannot leak by default."""

    artifact = _failing_run()
    artifact["config"] = {**artifact.get("plan", {}), "api_key": KEY}
    artifact["api_key"] = KEY
    for mode in artifact["modes"]:
        mode["config"] = {**mode.get("config", {}), "api_key": KEY}

    records = errors.collect_failures([artifact], tasks=TASKS)
    report = errors.error_report([artifact], tasks=TASKS, records=records)

    rendered = json.dumps(records) + json.dumps(report)
    assert KEY not in rendered
    assert "api_key" not in rendered


@pytest.mark.requires_runtime
def test_provenance_records_paths_and_digests_not_environment_values(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(_failing_run()), encoding="utf-8")
    output = tmp_path / "report.json"

    assert errors.main([str(run_path), "--output", str(output), "--dataset", str(DATASET)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    source = report["provenance"]["sources"][0]
    assert set(source) == {"path", "sha256"}
    assert source["path"] == str(run_path)
    assert len(source["sha256"]) == 64
    assert KEY not in json.dumps(report)


def test_the_module_never_reads_a_credential_field() -> None:
    """The analysis layer has no code path that could copy a key anywhere."""

    source = Path(errors.__file__).read_text(encoding="utf-8")

    assert "api_key" not in source
    assert "Authorization" not in source
    assert "providers" not in source
