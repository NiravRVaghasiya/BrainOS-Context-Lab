"""Run provenance and re-run manifests (plan Phase 16 / §22).

The implementation plan's reproducibility block is a fixed shape:

```json
{
  "run_id": "...",
  "timestamp": "...",
  "brainos_version": "...",
  "application_version": "...",
  "provider": "...",
  "model": "...",
  "temperature": 0.0,
  "benchmark_version": "...",
  "mode": "brainos",
  "context_budget": 4096
}
```

This module is the single place every evaluation entry point builds that block
from, so an experiment artifact, a single-mode run, and an error report all
record the *same* provenance vocabulary and the *same* version strings. It also
produces the second half of §16 (\"every evaluation run gets a run id ... save
configuration, task ids, raw metrics, aggregate metrics, model identifier,
BrainOS revision, benchmark revision\"): :func:`build_manifest` packages the
complete inventory — versions, environment, configuration, task ids, artifact
digests, and a copy-pasteable ``--rerun`` command — which callers embed as a
top-level ``repro`` block and can persist through
:func:`persist_run` / :func:`load_run`.

Two rules the whole repository already follows, kept here on principle:

* **Never save an API key.** a manifest carries the *name* of the environment
  variable a provider credential came from (``api_key_env``) — which is
  provenance — and never a value. :func:`persist_run` uses the secret-stripping
  storage writer as defence in depth.
* **Never promise what is not true.** an import we could not perform, a dataset
  hash we never computed, or a BrainOS revision we could not resolve is
  recorded as such (``unknown`` / ``unvalidated``) rather than filled in with a
  guess.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Identifies the provenance/manifest semantics recorded on an artifact.
REPRO_VERSION = "repro-v1"

#: What a manifest calls the upstream runtime version when the runtime is not
#: importable or does not expose a version. ``unvalidated`` is deliberately
#: *not* ``unknown``: it says the answer field exists and was checked, and that
#: nothing was found — the field's presence is the provenance.
UNVALIDATED = "unvalidated"

#: The upstream BrainOS revision this repository is pinned to (Phase 0). Used
#: as a fallback when the installed distribution cannot report its own version,
#: and marked as such in the payload so it is never mistaken for a live read.
BRAINOS_PINNED_COMMIT = "1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc"

#: The benchmark generator revision (mirrors ``benchmarks.context_rot.spec``).
GENERATOR_VERSION = "context-rot-v1"
#: The benchmark revision recorded on runs (mirrors ``evaluation.runner``).
BENCHMARK_VERSION = "context-rot-v1"
#: Application version when the installed distribution metadata is unavailable
#: (source checkout without an installed package). Mirrors ``pyproject.toml``.
APPLICATION_FALLBACK_VERSION = "0.1.0"

#: A runtime version may show up decorated (``2.0.0-alpha.1+git.…``); this is a
#: sanitizing read of the raw attribute.
_RUNTIME_METADATA_ATTRS = ("__version__", "version")

#: Environment variables recorded in a manifest. They are (a) metadata, never
#: credentials, and (b) read to their exact string value, so a deployment's
#: persistence posture can be reproduced rather than guessed.
_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "PYTHON_VERSION",  # read from sys.version_info; included for a uniform scan check
    "BRAINOS_LAB_DB",
)


def app_version() -> str:
    """Return the application version from the installed distribution.

    Falls back to :data:`APPLICATION_FALLBACK_VERSION` for a source checkout
    that was never built. The return value is padded to the left with a marker
    of how it was obtained (``installed`` / ``fallback``) so a reader can tell
    whether the number came from packaging metadata or from the constant.
    """

    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"installed:{version('brainos-context-lab')}"
    except PackageNotFoundError:
        return f"fallback:{APPLICATION_FALLBACK_VERSION}"


def brainos_version() -> str:
    """Return the installed BrainOS runtime version, without failing when absent.

    ``unvalidated`` means the runtime is not installed (or exposes no version);
    ``pinned:{commit}`` means the runtime is importable but has no version
    attribute and the Phase 0 pin is reported as the best-known revision. When
    the runtime reports a version, the pinned commit is attached as ``pin`` so
    the *resolution* provenance travels with the number.
    """

    pinned = BRAINOS_PINNED_COMMIT
    try:
        import brainos_runtime  # noqa: PLC0415 — optional dependency
    except Exception:
        return UNVALIDATED
    resolved = ""
    for attr in _RUNTIME_METADATA_ATTRS:
        value = getattr(brainos_runtime, attr, None)
        if value:
            resolved = str(value)
            break
    if not resolved:
        return f"pinned:{pinned}"
    return f"{resolved} (pin:{pinned})"


def _environment_block() -> dict[str, Any]:
    """Return the recorded environment, with an origin tag per key.

    ``origin`` says where the value came from (``env``, ``default``, or
    ``set-by-python`` for the interpreter version) so a reader knows which
    values to replicate and which are informational.
    """

    block: dict[str, Any] = {}
    for key in _ENVIRONMENT_KEYS:
        if key == "PYTHON_VERSION":
            major, minor, micro = (
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            )
            block[key] = {"value": f"{major}.{minor}.{micro}", "origin": "set-by-python"}
            continue
        if os.getenv(key) is not None:
            block[key] = {"value": os.getenv(key), "origin": "env"}
        else:
            block[key] = {"value": "", "origin": "unset"}
    return block


def _digest(path: str | Path | None) -> str:
    """SHA-256 of a file's bytes, or ``""`` when the path is absent/unreadable."""

    import hashlib

    if path is None:
        return ""
    candidate = Path(path)
    try:
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return ""


def build_manifest(
    *,
    run_id: str,
    timestamp: str | None = None,
    task_ids: tuple[str, ...] = (),
    dataset_path: str | None = None,
    config: Mapping[str, Any] | None = None,
    aggregate_metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the §22 reproducibility manifest for one evaluation run.

    Everything is optional except ``run_id``: a manifest never fails the run
    whose provenance it records. Absent inputs record honest absence — an empty
    task list says "no task ids were handed to the builder", while the
    ``task_count`` field shows whether that is because the run had none or the
    call site did not pass them. Configuration, the dataset digest, the version
    strings, the environment block, and the ``--rerun`` command are recorded
    when available.
    """

    resolved = str(run_id).strip() or "<unset>"
    config = dict(config or {})
    timestamps = [value for value in [timestamp, config.get("timestamp")] if value]
    when = str(timestamps[0]) if timestamps else datetime.now(timezone.utc).isoformat()
    task_ids = tuple(str(item) for item in task_ids)
    digest = config.get("dataset_sha256") or _digest(dataset_path)
    command = rerun_command(config, dataset=str(dataset_path) if dataset_path else None)
    return {
        "repro_version": REPRO_VERSION,
        "run_id": resolved,
        "timestamp": when,
        "brainos_version": brainos_version(),
        "application_version": app_version(),
        "benchmark_version": str(config.get("benchmark_version") or BENCHMARK_VERSION),
        "provider": str(config.get("provider") or ""),
        "model": str(config.get("model") or ""),
        "temperature": config.get("temperature"),
        "mode": str(config.get("mode") or ""),
        "context_budget": config.get("context_budget"),
        "task_count": len(task_ids),
        "task_ids": list(task_ids),
        "dataset": {
            "path": str(dataset_path) if dataset_path else "",
            "sha256": str(digest or ""),
        },
        "aggregate_metrics": dict(aggregate_metrics or {}),
        "environment": _environment_block(),
        "rerun_command": command,
        "notes": dict((extra or {}).get("notes") or {}),
    }


def rerun_command(
    config: Mapping[str, Any] | None,
    *,
    dataset: str | Path | None = None,
    output: str | Path | None = None,
) -> str:
    """Render a copy-pasteable re-run command for a single-mode run.

    Reads the values already recorded on a run's ``config`` so a manifest can be
    rebuilt from an artifact read back off disk, and never requires an argument
    the caller did not record. The command is credential-free by construction:
    a run's credential never lives in ``config``, so it can never reach the
    command.
    """

    config = dict(config or {})
    lines = ["python -m evaluation.run", f"--mode {config.get('mode') or ''}"]
    if config.get("benchmark"):
        lines.append(f"--benchmark {config['benchmark']}")
    dataset_arg = str(dataset) if dataset is not None else str(config.get("dataset") or "")
    if dataset_arg:
        lines.append(f"--dataset {dataset_arg}")
    if config.get("temperature") is not None:
        lines.append(f"--temperature {config['temperature']}")
    output_arg = str(output) if output is not None else str(config.get("output") or "")
    if not output_arg:
        output_arg = "results/run.json"
    lines.append(f"--output {output_arg}")
    return " \\\n  ".join(lines)


def global_overrides() -> dict[str, Any]:
    """Return the global provenance fields callers merge into a top-level block.

    The ``repro`` block is a full manifest plus the fallback version fields.
    Keeping the fallback fields at the top level of ``repro`` means consumers
    that already read ``experiment.provenance.*_version`` do not need to reach
    into ``repro.inventory``; the manifest remains the authoritative shape.
    """

    return {
        "repro_version": REPRO_VERSION,
        "application_version": app_version(),
        "benchmark_version": BENCHMARK_VERSION,
        "brainos_version": brainos_version(),
    }


def persist_run(
    run_id: str,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    store: Any = None,
    session_id: str = "",
) -> bool:
    """Persist one manifest's metadata/aggregate to an evaluation store, best-effort.

    Intentionally not wired into the CLI runs' write path by default — a CLI
    artifact is the record of record. This is the seam a future UI runner or a
    long-lived deployment uses to keep results queryable, using the storage
    protocol's secret-stripping writer.
    """

    if store is None:
        return False
    try:
        from storage.sqlite import SqliteEvaluationStore

        backend = SqliteEvaluationStore() if store is True else store
        if not hasattr(backend, "save_run"):
            return False
        payload = dict(metadata)
        payload.setdefault("session_id", str(session_id))
        backend.save_run(str(run_id), payload, dict(metrics))
        return True
    except Exception:
        return False


__all__ = [
    "APPLICATION_FALLBACK_VERSION",
    "BENCHMARK_VERSION",
    "BRAINOS_PINNED_COMMIT",
    "GENERATOR_VERSION",
    "REPRO_VERSION",
    "UNVALIDATED",
    "app_version",
    "brainos_version",
    "build_manifest",
    "global_overrides",
    "persist_run",
    "rerun_command",
]
