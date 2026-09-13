# Phase 0 log — Requirements and research baseline

**Date:** 2026-09-13
**Branch:** `arena/01a09b1f-brainos-context-lab`
**Starting commit:** `ec1f29445764f2f58884ee3c5c1afdc0ae6411e5`
**Plan section:** [Phase 0 — Requirements and Research Baseline](../BrainOS_Context_Lab_Implementation_Plan.md#4-phase-0--requirements-and-research-baseline)

## Objective

Establish what the upstream BrainOS runtime actually provides before the
application depends on it. This phase is a research/integration baseline, not
a claim that BrainOS improves the eventual context-rot benchmark.

## Repository and dependency audit

The initial repository already contained the Phase 0 documentation locations,
benchmark fixture, provider/BrainOS boundaries, and dependency placeholder. It
did not contain a validated upstream revision or real runtime smoke test.

The upstream repository was fetched into a temporary checkout outside this
repository:

```text
https://github.com/NiravRVaghasiya/BrainOS.git
HEAD at validation time:
1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc
```

The checkout was not copied into this repository and no generated upstream
artifacts were committed.

## Files inspected upstream

The validation inspected the areas called out in the plan:

- `brainos_runtime/__init__.py` and `brainos_runtime/core/brain.py`;
- `brainos_runtime/core/runtime.py`, `core/cycle.py`, `core/events.py`, and
  cognitive state;
- `brainos_runtime/retrieval/engine.py` and retrieval metrics;
- `brainos_runtime/observability/tracer.py`;
- `brainos_runtime/memory/` and storage backends;
- `brainos_runtime/security/`;
- `brainos_eval/` baselines, scenarios, metrics, ablations, and long-run
  validation;
- v2 documentation and quickstart examples; and
- the full upstream test suite.

## Validation results

The pinned checkout was installed into an isolated temporary virtual
environment (`/tmp/brainos-phase0-venv`) with its development dependencies.
The complete upstream suite passed:

```text
/tmp/brainos-phase0-venv/bin/pytest -q
316 passed in 33.78s
```

The public API smoke test passed with `prefer_generated=False`:

```python
from brainos_runtime import BrainOS

brain = BrainOS(
    session_id="phase0-session",
    actor_id="phase0-actor",
    prefer_generated=False,
)
brain.observe("For Project Atlas, the production database is PostgreSQL 16.")
brain.recall("What database does Project Atlas use?")
```

Observed shapes:

- `observe(...)` → `dict` cycle trace;
- `recall(...)` → `list[str]` containing the stored fact;
- `decide(...)` → a decision string (`retrieve` in the smoke test);
- `why(...)` → a mapping with `query`, `selected`, `decision`, `confidence`,
  and `rationale`; and
- `trace()` → a structured mapping with session state and cycle records.

The upstream offline evaluation command passed and produced the following
baseline observations:

```text
full_history  Recall@3=0.50
simple_rag    Recall@3=0.75
brainos       Recall@3=0.75
BrainOS token reduction vs full history = 17.2%
safety overall = 1.00
```

The upstream 200-event long-run benchmark also passed:

```text
PASS overall      : True
peak/final active : 57/9
salient recall    : 0.667
p95 latency       : 48.411 ms
stale/contra/inject: 1.0/1.0/1.0
```

These numbers are upstream baseline evidence only. They are not results from
BrainOS Context Lab's context-rot benchmark and must not be reported as such.

## Integration decision

The dependency in [`pyproject.toml`](../pyproject.toml) is now pinned to the
validated commit under the correct distribution name, `brainos-cli`. The import
package is `brainos_runtime`, not `brainos`. A clean temporary environment also
successfully installed the application integration extra and imported
`BrainOS`:

```text
pip install -e ".[integration]"
from brainos_runtime import BrainOS
# BrainOS imported successfully
```

The inspection found API differences from the original scaffold's injected
fake runtime:

| Scaffold assumption | Validated upstream API |
| --- | --- |
| `observe(text, metadata=...)` | `observe(event, source=..., event_type=...)` |
| `recall(query, limit=...)` | `recall(query, top_k=...)` |
| boolean/object `decide` | decision string |
| trace list | structured trace mapping |

These differences are recorded in [`docs/brainos-integration.md`](brainos-integration.md)
and are intentionally not papered over during Phase 1. Mapping the real
runtime, preserving session/actor isolation, and adding a real integration test
are Phase 2 adapter work.

## Phase 0 status

| Exit criterion | Status |
| --- | --- |
| BrainOS works independently | **Complete** — upstream suite, eval, and long-run benchmark passed. |
| Application can import a pinned BrainOS dependency | **Complete** — exact commit recorded in `pyproject.toml`; install remains optional. |
| Minimal `observe → recall` integration | **Validated upstream; application mapping pending Phase 2** — exact API shapes are now known. |
| Baseline models/tasks/metrics defined | **Scaffolded** — documented in `docs/evaluation.md`; the checked-in fixture is not a research dataset. |

Phase 0 is therefore complete as a research/API baseline, with the concrete
adapter implementation explicitly handed to Phase 2. Phase 1 provider work
was completed independently and is logged in
[`phase-1-provider-abstraction.md`](phase-1-provider-abstraction.md).

## Reproducibility and security notes

- Tests and benchmarks ran offline; no provider API key was used.
- The temporary upstream checkout and virtual environment remain outside the
  repository.
- No API key, user conversation, or generated result was written to tracked
  files.
- Future evaluation metadata must include the exact upstream commit and must
  never include provider credentials.
