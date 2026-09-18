"""Reproducibility primitives (plan Phase 16 / §22).

Every evaluation result gets a *run manifest*: the versions, environment, and
configuration that produced it, plus a re-run command. The manifest is written
beside the numbers (``repro`` block), embedded in the failure/analysis tooling
that reads artifacts, and persistable to the evaluation store, so a result in a
month can be tied back to the exact code, dataset, model, and settings that
made it.

The package depends only on the standard library plus :mod:`providers` and
:mod:`storage` (both dependency-free in the BrainOS direction), so it can be
imported by the evaluation CLI without pulling in the UI or the BrainOS runtime.
"""

from __future__ import annotations

from .run_provenance import (
    APPLICATION_FALLBACK_VERSION,
    BENCHMARK_VERSION,
    BRAINOS_PINNED_COMMIT,
    GENERATOR_VERSION,
    REPRO_VERSION,
    app_version,
    brainos_version,
    build_manifest,
    global_overrides,
    persist_run,
    rerun_command,
)

__all__ = [
    "APPLICATION_FALLBACK_VERSION",
    "BENCHMARK_VERSION",
    "BRAINOS_PINNED_COMMIT",
    "GENERATOR_VERSION",
    "REPRO_VERSION",
    "app_version",
    "brainos_version",
    "build_manifest",
    "global_overrides",
    "persist_run",
    "rerun_command",
]
