# BrainOS integration

## Integration status

The adapter boundary is scaffolded, but the upstream BrainOS dependency is not yet validated or pinned. Phase 0 must inspect the upstream repository, run its tests and benchmark, and confirm the v2 runtime signatures before wiring the adapter.

## Application contract

```python
class BrainMemoryAdapter:
    def observe(self, text, *, metadata=None): ...
    def recall(self, query, *, limit=8): ...
    def decide(self, query): ...
    def explain(self, query): ...
    def trace(self): ...
```

The application uses normalized `MemoryRecord`, `Decision`, and `TraceEvent` values. This prevents upstream object types from reaching the UI, provider layer, or evaluator.

## Required validation

Before enabling the integration:

1. Identify and pin a reproducible BrainOS revision.
2. Run upstream tests and existing benchmarks.
3. Inspect `brainos_runtime`, evaluation/ablation tooling, examples, adapters, and storage backends.
4. Validate a minimal `observe → recall` flow.
5. Confirm whether `decide`, `why`, and trace APIs are available and what their return types are.
6. Add adapter integration tests against a deterministic fixture or injected runtime.
7. Record the validated revision in evaluation metadata.

## Memory policy

The application must not blindly store every message. Candidate memories should be classified conservatively into categories such as `FACT`, `PREFERENCE`, `PROJECT_STATE`, `DECISION`, `GOAL`, `TASK`, `TEMPORAL_EVENT`, `CORRECTION`, and `CONSTRAINT`.

Explicit corrections and temporal updates must remain distinguishable so later evaluation can measure stale-memory and conflict-resolution errors.

## Security boundary

BrainOS memory is untrusted user data. It must never become a higher-priority instruction than the application system prompt. Context construction must use explicit delimiters and state that retrieved memory is data, not instructions.
