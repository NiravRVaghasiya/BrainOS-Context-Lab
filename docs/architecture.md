# Architecture

## Scope

BrainOS Context Lab is an application around the upstream BrainOS runtime. BrainOS remains responsible for cognitive memory operations; this repository is responsible for providers, sessions, context orchestration, UI, storage, and evaluation.

```text
Browser
  │
  ▼
Gradio UI
  │
  ▼
Application/session layer
  ├── BrainMemoryAdapter ──► BrainOS runtime
  ├── LLMProvider ─────────► Selected provider
  ├── Context builder
  ├── Conversation/memory stores
  └── Evaluation logger
```

## Request lifecycle

1. Receive a user message inside an isolated session.
2. Apply the memory policy to information eligible for observation.
3. Ask the BrainOS adapter to observe eligible content.
4. Recall memories relevant to the current query.
5. Deduplicate, filter, account for conflicts/recency, and apply the context budget.
6. Send the resulting provider-neutral message list to the selected LLM provider.
7. Render the answer and sanitized inspection data.
8. Observe the assistant response when the memory policy permits it.
9. Record token accounting and evaluation metadata without secrets.

## Boundaries

### BrainOS adapter

The adapter is the only application dependency on BrainOS. It exposes `observe`, `recall`, `decide`, `explain`, and `trace`. The concrete runtime mapping must be validated against a pinned upstream revision before production use.

### Provider adapter

The provider interface normalizes model listing, credential validation, and generation. Provider-specific SDK objects must not leak into the context builder or evaluation runner.

### Context builder

The context builder returns model-ready messages and explicit accounting fields:

- raw history tokens,
- selected recent-history tokens,
- retrieved-memory tokens,
- system tokens, and
- final context tokens.

Retrieved memory is delimited and explicitly described as data rather than instructions.

### Storage

The initial target is session-local persistence with SQLite behind interfaces. API keys are never part of persisted state. Every record must carry sufficient session/conversation identity to prevent cross-session leakage.

## Baseline modes

The evaluation layer must support the same provider and task protocol for:

- Full Context,
- Sliding Window,
- Conventional RAG,
- BrainOS, and optionally
- BrainOS + RAG.

Only the context-management strategy should change in a controlled comparison.
