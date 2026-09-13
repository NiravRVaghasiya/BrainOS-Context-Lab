# BrainOS integration

## Validated upstream revision

Phase 0 inspected the upstream repository and validated the following revision:

```text
Repository: https://github.com/NiravRVaghasiya/BrainOS.git
Commit:     1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc
Package:    brainos-cli 1.2.0
Runtime:    brainos_runtime.__version__ = 2.0.0-alpha.1
```

The application dependency is pinned to that commit in the `integration` extra
in [`pyproject.toml`](../pyproject.toml). The distribution name is
`brainos-cli`; the v2 runtime is imported from `brainos_runtime`.

## Upstream API observed at the pinned revision

The public runtime entry point is:

```python
from brainos_runtime import BrainOS

brain = BrainOS(
    session_id="...",
    actor_id="...",
    token_budget=2000,
    wm_slots=6,
    prefer_generated=False,
)
```

The application-relevant methods and return shapes are:

| Method | Observed signature/return | Adapter implication |
| --- | --- | --- |
| `observe(event, *, source="user", event_type=EventType.USER_MESSAGE)` | Returns a cycle-trace `dict`; accepts strings, dict payloads, or a `CognitiveEvent`. | Do not call it with the scaffold's `metadata=` keyword. Convert application metadata into event/source fields in Phase 2. |
| `recall(query, top_k=8)` | Returns `list[str]` containing selected memory content. | Map `top_k`, not `limit`, and assign stable IDs from runtime explanations/state when available. |
| `decide(query)` | Returns a decision string such as `ask`, `retrieve`, or `act`. | Normalize into the application `Decision` dataclass. |
| `assess(query)` | Returns a dict with confidence, decision, rationale, and retrieval signals. | Prefer this for a richer application decision/explanation. |
| `why(query)` | Returns a dict with `query`, selected memories, scores/signals, decision, and rationale. | Sanitize before sending to the UI. |
| `trace(formatted=False)` | Returns a structured session-trace dict; `formatted=True` returns text. | Map the structured records into application `TraceEvent` values in Phase 2. |

The runtime supports `session_id` and `actor_id` tenancy anchors. The adapter
must construct one runtime per isolated application session and must not share a
runtime or storage backend across sessions.

## Phase 0 validation evidence

The pinned upstream checkout was installed in an isolated temporary virtual
environment and its complete test suite was run:

```text
pytest -q
316 passed in 33.78s
```

The following offline runtime smoke test also passed:

```python
brain.observe("For Project Atlas, the production database is PostgreSQL 16.")
brain.recall("What database does Project Atlas use?")
```

Observed result: `observe` returned a `dict` cycle trace and `recall` returned
`["For Project Atlas, the production database is PostgreSQL 16."]`.

The upstream evaluation entry points were run at the same revision:

```text
python -m brainos_runtime.cli eval
```

Relevant deterministic output:

```text
full_history  Recall@3=0.50
simple_rag    Recall@3=0.75
brainos       Recall@3=0.75
brainos token reduction vs full = 17.2%
safety overall = 1.00
```

The 200-event long-running validation was also run:

```text
python -m brainos_runtime.cli benchmark --events 200
```

It reported `PASS overall: True`, peak/final active memories `57/9`, salient
recall `0.667`, and p95 latency `48.411 ms`. These are upstream baseline
observations, not BrainOS Context Lab research results.

## Application adapter status

The application boundary remains intentionally separate in
[`src/brain/adapter.py`](../src/brain/adapter.py). Its injected fake-runtime
contract was created before the upstream API was validated and currently uses
`metadata=`, `limit=`, and a list-style trace. The validated signature mapping
above is the input to **Phase 2 — BrainOS Adapter**; the real runtime should not
be wired by guessing or silently adapting incompatible return shapes.

Until Phase 2 is complete, the application must not claim that the live
BrainOS runtime is active. The Phase 1 provider adapters do not import BrainOS.

## Memory and security boundary

The application must not blindly store every message. Candidate memories should
be classified conservatively into categories such as `FACT`, `PREFERENCE`,
`PROJECT_STATE`, `DECISION`, `GOAL`, `TASK`, `TEMPORAL_EVENT`, `CORRECTION`, and
`CONSTRAINT`.

BrainOS memory is untrusted user data. It must never become a higher-priority
instruction than the application system prompt. Context construction must use
explicit delimiters and state that retrieved memory is data, not instructions.

## Reproducibility

Use the pinned integration extra only when running real integration work:

```bash
pip install -e ".[integration]"
```

The application should record the pinned commit in future evaluation metadata,
never a mutable branch name. No upstream checkout, API key, or generated
benchmark artifact is committed to this repository.
