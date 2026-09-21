"""Phase 16 — the run manifest (§22) and the reproducibility primitives.

The plan's reproducibility block is a fixed shape: run id, timestamp, BrainOS
revision, application version, provider, model, temperature, benchmark
revision, mode, context budget. These tests pin the module that builds it so
every artifact producer (single-mode run, experiment, error report) records
the same vocabulary, and pin the two properties that make a manifest
trustworthy:

* it never contains a credential — not a value, not a key name, not the
  environment variable that held a key;
* it is honest — an unknown version is ``unvalidated``/``pinned:``, not a
  guessed number, and a manifest built without inputs records their absence.
"""

from __future__ import annotations

import json

import pytest

from reproducibility.run_provenance import (
    APPLICATION_FALLBACK_VERSION,
    BENCHMARK_VERSION,
    BRAINOS_PINNED_COMMIT,
    GENERATOR_VERSION,
    REPRO_VERSION,
    UNVALIDATED,
    app_version,
    brainos_version,
    build_manifest,
    global_overrides,
    rerun_command,
)

# --------------------------------------------------------------------------- #
# Version reads
# --------------------------------------------------------------------------- #


def test_application_version_names_its_origin() -> None:
    """An honest version says where it came from, not just the number."""

    version = app_version()
    assert version.startswith(("installed:", "fallback:"))
    assert APPLICATION_FALLBACK_VERSION in version


def test_brainos_version_is_unvalidated_when_the_runtime_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the runtime the answer is ``unvalidated``, never an invented rev."""

    def _missing(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr("builtins.__import__", _missing)
    assert brainos_version() == UNVALIDATED


@pytest.mark.requires_runtime
def test_brainos_version_attaches_the_pin_when_the_runtime_reports_one() -> None:
    """With the runtime present, the pinned Phase 0 revision travels with the number."""

    version = brainos_version()
    assert version != UNVALIDATED
    if version.startswith("pinned:"):
        assert version == f"pinned:{BRAINOS_PINNED_COMMIT}"
    else:
        assert BRAINOS_PINNED_COMMIT in version


# --------------------------------------------------------------------------- #
# Manifest shape
# --------------------------------------------------------------------------- #


def _config(**overrides) -> dict[str, object]:
    config: dict[str, object] = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "mode": "brainos",
        "benchmark": "context_rot",
        "context_budget": 4096,
        "benchmark_version": BENCHMARK_VERSION,
        "dataset_sha256": "abc123",
    }
    config.update(overrides)
    return config


def test_manifest_records_the_plan_field_by_field() -> None:
    manifest = build_manifest(
        run_id="run-1",
        timestamp="2026-09-18T00:00:00+00:00",
        task_ids=("t1", "t2"),
        config=_config(),
    )

    assert manifest["repro_version"] == REPRO_VERSION
    assert manifest["run_id"] == "run-1"
    assert manifest["timestamp"] == "2026-09-18T00:00:00+00:00"
    assert manifest["provider"] == "openai"
    assert manifest["model"] == "gpt-4o-mini"
    assert manifest["temperature"] == 0.0
    assert manifest["mode"] == "brainos"
    assert manifest["context_budget"] == 4096
    assert manifest["benchmark_version"] == BENCHMARK_VERSION
    assert manifest["task_count"] == 2
    assert manifest["task_ids"] == ["t1", "t2"]
    assert manifest["dataset"]["sha256"] == "abc123"
    assert manifest["brainos_version"] != ""
    assert manifest["application_version"] != ""


def test_manifest_records_reality_when_inputs_are_missing() -> None:
    """Absent inputs record their absence; nothing is invented to look complete."""

    manifest = build_manifest(run_id="run-2")

    assert manifest["run_id"] == "run-2"
    assert manifest["task_count"] == 0
    assert manifest["task_ids"] == []
    assert manifest["provider"] == ""
    assert manifest["model"] == ""
    assert manifest["temperature"] is None
    assert manifest["mode"] == ""
    assert manifest["dataset"]["path"] == ""
    assert manifest["timestamp"] != ""  # a timestamp is always producible
    assert isinstance(manifest["environment"], dict)


def test_manifest_never_records_a_credential() -> None:
    """Neither a value, a field name, nor the source env var reaches the manifest."""

    config = _config()
    config["api_key"] = "sk-very-secret"
    config["api_key_env"] = "OPENAI_API_KEY"

    manifest = build_manifest(run_id="run-3", config=config)
    rendered = json.dumps(manifest, sort_keys=True)

    assert "sk-very-secret" not in rendered
    assert '"api_key"' not in rendered
    assert '"api_key_env"' not in rendered
    assert "OPENAI_API_KEY" not in rendered


def test_manifest_records_a_path_digest_when_given_a_dataset_file(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"task_id": "t1"}\n', encoding="utf-8")

    manifest = build_manifest(
        run_id="run-4", dataset_path=dataset, config=_config(dataset_sha256="")
    )

    assert manifest["dataset"]["path"] == str(dataset)
    assert manifest["dataset"]["sha256"]
    assert manifest["dataset"]["sha256"] != ""


def test_global_overrides_carry_the_manifest_versions() -> None:
    overrides = global_overrides()

    assert overrides["repro_version"] == REPRO_VERSION
    assert overrides["application_version"] == app_version()
    assert overrides["benchmark_version"] == BENCHMARK_VERSION
    assert overrides["brainos_version"] == brainos_version()


def test_generator_version_matches_the_benchmark_revision() -> None:
    """The benchmark and generator revisions are one vocabulary, not a coincidence."""

    assert GENERATOR_VERSION == BENCHMARK_VERSION == "context-rot-v1"


def test_every_producer_shares_one_benchmark_revision() -> None:
    """Phase 16 fixes the split vocab: dataset spec, runner, experiment, manifest."""

    from benchmarks.context_rot import spec
    from evaluation.experiment import BENCHMARK_VERSION as EXPERIMENT_REV
    from evaluation.runner import EvaluationConfig

    assert spec.GENERATOR_VERSION == "context-rot-v1"
    assert EvaluationConfig().benchmark_version == "context-rot-v1"
    assert EXPERIMENT_REV == "context-rot-v1"
    assert BENCHMARK_VERSION == "context-rot-v1"


# --------------------------------------------------------------------------- #
# Rerun command
# --------------------------------------------------------------------------- #


def test_rerun_command_contains_no_credential_and_a_full_mode_line() -> None:
    command = rerun_command(
        _config(),
        dataset="benchmarks/context_rot/dataset.jsonl",
        output="results/run.json",
    )

    assert "evaluation.run" in command
    assert "--mode" in command and "brainos" in command
    assert "--dataset" in command
    assert "--output" in command and "results/run.json" in command
    assert "sk-" not in command
    assert "api_key" not in command


def test_rerun_command_reconstructs_from_an_artifact_config() -> None:
    config = {
        "mode": "brainos",
        "benchmark": "context_rot",
        "temperature": 0.0,
        "dataset": "benchmarks/context_rot/dataset.jsonl",
        "output": "results/run.json",
    }

    command = rerun_command(config)

    assert "--mode brainos" in command
    assert "--dataset benchmarks/context_rot/dataset.jsonl" in command
    assert "--output results/run.json" in command
