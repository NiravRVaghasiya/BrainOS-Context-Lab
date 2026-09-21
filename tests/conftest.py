"""Shared test configuration and the suite's optional-dependency policy.

The plan's Phase 18 exit criteria are about what the suite *guarantees*, not
about which optional extra a contributor happened to install. Two kinds of test
exist here:

* tests backed by the doubles in ``tests.fakes`` — they run everywhere, with no
  network, no credential, and no BrainOS;
* tests that exercise the application's real composition (its default adapter
  factory, the evaluation pipeline, the deployment configuration) — those need
  the pinned BrainOS runtime, which lives in the optional ``integration`` extra.

The second kind must **skip**, not fail, when the extra is absent: a base
``pip install -e ".[dev]"`` is a supported way to work on this repository, and a
red suite that is really a missing dependency teaches a contributor to ignore
failures. Phase 18 found 86 such tests failing in a base install and pinned the
policy here, so the fix is one marker instead of 86 copies of the same skip.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest

#: Marker name -> (module that must be importable, why it is optional).
OPTIONAL_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "requires_runtime": (
        "brainos_runtime",
        "the pinned BrainOS runtime is not installed "
        "(pip install -e '.[integration]')",
    ),
    "requires_figures": (
        "matplotlib",
        "figures need matplotlib (pip install -e '.[evaluation]')",
    ),
}

RUNTIME_MARKER = "requires_runtime"


def missing_dependency_reason(marker: str) -> str | None:
    """Return the skip reason when the marker's dependency is absent."""

    module, reason = OPTIONAL_DEPENDENCIES[marker]
    if importlib.util.find_spec(module) is None:
        return reason
    return None


def pytest_collection_modifyitems(items: list[Any]) -> None:
    """Skip — do not fail — the tests whose optional dependency is missing."""

    reasons = {
        marker: reason
        for marker in OPTIONAL_DEPENDENCIES
        if (reason := missing_dependency_reason(marker)) is not None
    }
    if not reasons:
        return
    for item in items:
        for marker, reason in reasons.items():
            if marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))
                break
