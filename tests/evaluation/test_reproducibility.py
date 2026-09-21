"""Phase 16 — reproducibility harness over the evaluation artifacts.

Phase 16 turns the §22 block from a per-artifact convention into a check: a
single-mode run writes a ``repro`` manifest, an experiment artifact carries a
``repro`` block whose version strings match the manifest vocabulary, and the
error report records the same version vocabulary in ``provenance``. These tests
pin the emitted shapes and the two properties that make them trustworthy — no
credential, and the fallback version fields reach ``config`` so older consumers
keep reading them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import run as run_cli
from evaluation.experiment import run_controlled_experiment
from evaluation.generation import ModelSpec
from evaluation.limits import RunLimits
from reproducibility.run_provenance import (
    REPRO_VERSION,
    app_version,
    brainos_version,
)
from tests.fakes import RecordingProvider

DATASET = Path("benchmarks/context_rot/dataset.jsonl")


@pytest.mark.requires_runtime
def test_a_run_writes_a_repro_manifest_without_a_credential(tmp_path) -> None:
    output = tmp_path / "run.json"

    code = run_cli.main(["--mode", "brainos", "--limit", "1", "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert code == 0
    repro = payload["repro"]
    assert repro["repro_version"] == REPRO_VERSION
    assert repro["run_id"] == payload["run_id"]
    assert repro["timestamp"] == payload["timestamp"]
    assert repro["mode"] == "brainos"
    assert repro["task_count"] == 1
    assert repro["task_ids"] == [payload["task_results"][0]["task_id"]]
    assert repro["dataset"]["path"] == str(DATASET)
    assert repro["dataset"]["sha256"] == payload["config"]["dataset_sha256"]

    rendered = json.dumps(payload)
    assert "sk-" not in rendered
    assert '"api_key"' not in rendered


@pytest.mark.requires_runtime
def test_a_run_records_the_repro_vocabulary_on_config(tmp_path) -> None:
    """``config`` keeps the §22 keys so a consumer older than Phase 16 still works."""

    output = tmp_path / "run.json"
    run_cli.main(["--mode", "brainos", "--limit", "1", "--output", str(output)])
    config = json.loads(output.read_text(encoding="utf-8"))["config"]

    assert config["application_version"] == app_version()
    assert config["brainos_version"] == brainos_version()
    assert config["benchmark_version"] == "context-rot-v1"
    assert config["output"] == str(output)


@pytest.mark.requires_runtime
def test_an_experiment_writes_a_repro_block_matching_the_manifest() -> None:
    provider = RecordingProvider(text="MongoDB 7")
    model = ModelSpec(model="fake-model", temperature=0.0, max_tokens=64)

    payload = run_controlled_experiment(
        __import__("evaluation.experiment", fromlist=["ExperimentPlan"]).ExperimentPlan(
            modes=("brainos", "full_context"), limits=RunLimits(max_requests=1000)
        ),
        __import__("evaluation.datasets", fromlist=["load_jsonl"]).load_jsonl(DATASET)[:1],
        model,
        provider,
    ).to_dict()

    repro = payload["repro"]
    assert repro["repro_version"] == REPRO_VERSION
    assert repro["run_id"] == payload["run_id"]
    assert repro["dry_run"] is False
    assert repro["modes"] == ["brainos", "full_context"]
    assert repro["task_count"] == 1
    assert repro["application_version"] == payload["provenance"]["application_version"]
    assert repro["brainos_version"] == payload["provenance"]["brainos_version"]

    rendered = json.dumps(payload)
    assert "sk-" not in rendered
    assert '"api_key":' not in rendered


@pytest.mark.requires_runtime
def test_an_error_report_records_the_repro_versions_in_provenance(tmp_path) -> None:
    """Phase 12 consumers also carry the §22 vocabulary in their provenance block."""

    from evaluation.errors import error_report

    run_output = tmp_path / "run.json"
    run_cli.main(
        [
            "--mode",
            "brainos",
            "--answers",
            "benchmarks/fixtures/scripted_answers.jsonl",
            "--limit",
            "1",
            "--output",
            str(run_output),
        ]
    )
    artifact = json.loads(run_output.read_text(encoding="utf-8"))

    report = error_report(artifact, sources=[{"path": str(run_output), "sha256": "abc"}])

    assert report["provenance"]["application_version"] == app_version()
    assert report["provenance"]["brainos_version"] == brainos_version()


def test_a_manifest_survives_a_round_trip_through_the_evaluation_store(
    tmp_path,
) -> None:
    """``persist_run`` writes through the secret-stripping store protocol."""

    from reproducibility.run_provenance import build_manifest, persist_run
    from storage.sqlite import SqliteEvaluationStore

    store = SqliteEvaluationStore(tmp_path / "lab.sqlite3")
    manifest = build_manifest(
        run_id="run-1",
        config={
            "provider": "openai",
            "model": "gpt-4o-mini",
            "mode": "brainos",
            "context_budget": 4096,
            "benchmark_version": "context-rot-v1",
        },
        aggregate_metrics={"answer_accuracy": 0.95},
    )

    assert persist_run("run-1", manifest, {"answer_accuracy": 0.95}, store=store) is True
    saved = store.get_run("run-1")
    assert saved is not None
    assert saved["metadata"]["mode"] == "brainos"
    assert saved["metrics"]["answer_accuracy"] == 0.95


def test_persist_run_does_not_blow_up_without_a_store() -> None:
    from reproducibility.run_provenance import persist_run

    assert persist_run("run-1", {"mode": "brainos"}, {}) is False
