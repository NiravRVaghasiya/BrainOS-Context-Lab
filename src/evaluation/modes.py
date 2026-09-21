"""Baseline-mode strategies for the evaluation runner (plan Phase 6).

This module is the seam the plan's automated pipeline (Phase 17) plugs into:
:meth:`evaluation.runner.EvaluationRunner.run` accepts an evaluator callable of
shape ``(task, mode) -> record``, and :func:`task_evaluator` produces exactly
that from the application's own context-construction path.

What this module deliberately does **not** do:

* **No model calls of its own.** A replay configures no provider, so no API key
  is needed and the retrieval half of a benchmark runs credential-free. Phase 9
  added an *optional* ``generate`` callable: when a caller supplies one (the
  controlled experiment in :mod:`evaluation.experiment` does), the prompt this
  module built is sent to a model and the answer comes back on the replay
  record. The callable is a parameter, not an import, so the credential-free
  path stays credential-free.
* **No scoring.** It reports what each mode *put in the prompt* and what that
  cost. Calling the ``expected_answer_in_prompt`` flag an accuracy measure would
  be wrong: it says the evidence was available to the model, not that the model
  used it.
* **No shared sessions.** Every (task, mode) pair gets a fresh session and a
  fresh BrainOS runtime, so one task's memory cannot leak into the next and a
  mode cannot benefit from a warm cache built by another.

What it does guarantee is the plan's controlled-comparison rule: every mode runs
the identical transcript and question through the identical
:class:`~app.service.ConversationService` and context builder, with the same
system prompt, and reports the same accounting fields. Only the
context-management strategy differs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from app.service import ConversationService
from app.state import ContextSettings, SessionState
from baselines.modes import MODE_ORDER, mode_label, resolve_mode

from .datasets import BenchmarkTask
from .generation import GenerationResult, Generator, count_messages

#: Type of the factory that hands a replay a fresh, session-scoped service.
ServiceFactory = Callable[[str], ConversationService]


@dataclass(frozen=True)
class ModeReplay:
    """One benchmark task run through one baseline mode.

    Every field is either an accounting number the context builder already
    reported or an identifier list, so a replay is safe to write into an
    evaluation record: it carries no credentials and no provider state.
    """

    task_id: str
    category: str
    mode: str
    mode_label: str
    question: str
    expected_answer: str
    conversation_length: int
    final_context_tokens: int
    full_context_reference_tokens: int
    context_reduction: float
    selected_memory_count: int
    selected_chunk_count: int
    history_messages_considered: int
    history_messages_selected: int
    retrieved_memory_ids: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    #: Phase 7: the *text* the retriever selected, kept because the benchmark's
    #: ledger refers to facts by marker, not by runtime id. Scoring needs to see
    #: what was selected, and a memory id alone cannot be matched to a fact.
    retrieved_memory_texts: tuple[str, ...] = ()
    retrieved_chunk_texts: tuple[str, ...] = ()
    #: How many memories BrainOS held at the end of the replay. Non-zero proves
    #: the runtime actually observed the scripted facts, which is what separates
    #: "the retriever chose badly" from "nothing was ever stored".
    stored_memory_count: int = 0
    #: Transcript sessions the replay used. ``1`` unless session isolation was
    #: requested on a multi-session task.
    sessions_used: int = 1
    #: Evidence-availability proxy only: the expected answer string appears in
    #: the prompt. Not accuracy — see the module docstring.
    expected_answer_in_prompt: bool = False
    prompt_messages: tuple[dict[str, str], ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    #: Phase 12: the Phase 3 retrieval audit for the question's turn — drop
    #: reasons for every memory the policy removed, plus the conflict
    #: resolutions. ``stats`` says what the prompt cost; the audit says why the
    #: evidence in it is what it is, which is what failure attribution needs.
    retrieval_report: dict[str, Any] = field(default_factory=dict)
    #: Phase 9: the model's answer, present only when a generation path was
    #: supplied. ``None`` means "no model was called", not "the model said
    #: nothing" — an empty completion is a generation failure and scores as
    #: ungraded rather than as a wrong answer.
    answer: str | None = None
    #: Phase 9: the generation record (latency, usage, settings fingerprint,
    #: error). Kept as a plain mapping so a replay stays serializable and this
    #: module keeps no dependency on how generation is implemented.
    generation: dict[str, Any] | None = None
    #: Phase 13: the question turn's guard report — injection families seen,
    #: structural rewrites, credentials redacted, history roles downgraded. It
    #: is part of the record for two reasons: a quarantine that changed what
    #: reached the prompt belongs beside the retrieval audit that explains the
    #: same prompt, and an artifact that carries no guard block at all cannot
    #: claim it held no credential.
    security: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float | None:
        """Return the generation latency, or ``None`` when nothing was generated."""

        if not self.generation:
            return None
        value = self.generation.get("latency_ms")
        return None if value is None else float(value)

    def prompt_text(self) -> str:
        """Return the prompt as one string, in message order."""

        return "\n".join(
            str(message.get("content", "")) for message in self.prompt_messages
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record for the runner and the evaluation store."""

        return {
            "task_id": self.task_id,
            "category": self.category,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "generation": dict(self.generation) if self.generation else None,
            "conversation_length": self.conversation_length,
            "final_context_tokens": self.final_context_tokens,
            "full_context_reference_tokens": self.full_context_reference_tokens,
            "context_reduction": round(self.context_reduction, 6),
            "selected_memory_count": self.selected_memory_count,
            "selected_chunk_count": self.selected_chunk_count,
            "history_messages_considered": self.history_messages_considered,
            "history_messages_selected": self.history_messages_selected,
            "retrieved_memory_ids": list(self.retrieved_memory_ids),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieved_memory_texts": list(self.retrieved_memory_texts),
            "retrieved_chunk_texts": list(self.retrieved_chunk_texts),
            "stored_memory_count": self.stored_memory_count,
            "sessions_used": self.sessions_used,
            "expected_answer_in_prompt": self.expected_answer_in_prompt,
            "prompt_messages": [dict(message) for message in self.prompt_messages],
            "stats": dict(self.stats),
            "retrieval_report": dict(self.retrieval_report),
            "security": dict(self.security),
        }


#: Room a replay's ceiling leaves on top of the transcript for the system
#: prompt, the question, and whatever evidence block the mode adds.
REPLAY_CEILING_SLACK = 1024


def replay_ceiling(task: BenchmarkTask) -> int:
    """Return the context ceiling a replay of ``task`` needs.

    The product's ceiling (``ContextSettings.max_tokens``, 4096) exists to bound a
    *live* session. Applied to a replay it silently turns the benchmark's
    reference into a 4k window: Mode A's history budget is derived from
    ``max_tokens``, so above roughly 4k tokens the "full context" prompt stops
    growing and every other mode's reduction is measured against a truncated
    conversation. Phase 20 measured exactly that — Mode A sat at 4,086 / 4,090 /
    4,090 tokens for 5k / 10k / 20k tasks — which is why the replay sizes its own
    ceiling to the transcript it is about to replay.

    The run-level ``max_input_tokens`` ceiling (preset ``standard``: 64k) stays
    the guard: a prompt genuinely too big for a run is refused or reported, not
    quietly truncated by a session default nobody in the benchmark chose.
    """

    settings = ContextSettings()
    # Malformed entries are skipped the same way the replay skips them, so a
    # dataset with a stray string cannot inflate the ceiling.
    messages = [message for message in task.conversation if isinstance(message, Mapping)]
    transcript = count_messages(messages) + count_messages(
        [{"role": "user", "content": task.question}]
    )
    return max(settings.max_tokens, transcript + REPLAY_CEILING_SLACK)


def default_service_factory(mode: str) -> ConversationService:
    """Build a fresh, purely in-memory service configured for ``mode``.

    No storage backends are attached, so a replay never touches disk, and no
    provider configuration is set, so a replay never calls a model.
    """

    state = SessionState(context=ContextSettings().with_mode_defaults(mode))
    return ConversationService(state)


def _raise_session_ceiling(service: ConversationService, ceiling: int) -> None:
    """Let a replay's session hold its whole transcript, before it starts.

    Applied to the service the factory returned — the default factory and any
    injected one — through the same settings object the mode switch edits. The
    mode's budgets are re-derived from the new ceiling, because Mode A's history
    budget *is* the ceiling: without that step a larger ``max_tokens`` would move
    the wall but not the window.
    """

    settings = service.state.context
    if ceiling <= settings.max_tokens:
        return
    service.state.context = replace(settings, max_tokens=ceiling).with_mode_defaults(
        settings.mode
    )


def replay_task(
    task: BenchmarkTask,
    mode: str,
    *,
    service_factory: ServiceFactory | None = None,
    session_isolation: bool = False,
    generate: Generator | None = None,
) -> ModeReplay:
    """Run one benchmark task's conversation through one baseline mode.

    The scripted transcript is replayed turn by turn — user turns through the
    normal path, assistant turns through
    :meth:`~app.service.ConversationService.record_assistant_message` — and then
    the task's question is asked. The returned record describes the prompt that
    question produced.

    A transcript may mark session boundaries with a per-message ``session`` index
    (Phase 7's cross-session category). By default the whole transcript is
    replayed as **one** session, which is the long-range-memory case: the fact is
    far away, and only a memory layer can still reach it. With
    ``session_isolation=True`` a fresh session starts at each boundary, which is
    how the product's session-isolation invariant is measured — the fact is then
    provably unreachable, and declining to answer is the correct behaviour.

    When ``generate`` is supplied (Phase 9), the prompt built for the question is
    sent to the model and the answer is attached to the replay. The generated
    reply is *not* appended to the transcript: the episode ends at the question,
    and letting a mode's own answer extend the conversation would make the next
    task's replay depend on the previous mode's output.
    """

    factory = service_factory or default_service_factory
    service = factory(resolve_mode(mode))
    _raise_session_ceiling(service, replay_ceiling(task))
    sessions_used = 1
    current_session = _message_session(task.conversation[0]) if task.conversation else 0
    for message in task.conversation:
        if not isinstance(message, Mapping):
            continue
        session_index = _message_session(message)
        if session_isolation and session_index != current_session:
            service = factory(resolve_mode(mode))
            sessions_used += 1
            current_session = session_index
        role = str(message.get("role", "user")).strip().lower()
        content = message.get("content", "")
        text = "" if content is None else str(content)
        if role == "assistant":
            service.record_assistant_message(text)
        else:
            service.handle_user_message(text)

    turn = service.handle_user_message(task.question)
    stats = dict(turn.context_stats)
    prompt = "\n".join(
        str(message.get("content", "")) for message in turn.context_messages
    ).casefold()
    expected = str(task.expected_answer or "").strip().casefold()
    length = task.conversation_length
    if length is None:
        length = len(task.conversation)
    generation: GenerationResult | None = None
    if generate is not None:
        generation = generate(turn.context_messages)
    return ModeReplay(
        task_id=task.task_id,
        category=task.category,
        mode=resolve_mode(mode),
        mode_label=mode_label(mode),
        question=task.question,
        expected_answer=task.expected_answer,
        conversation_length=int(length or 0),
        final_context_tokens=int(stats.get("final_context_tokens", 0) or 0),
        full_context_reference_tokens=int(
            stats.get("full_context_reference_tokens", 0) or 0
        ),
        context_reduction=float(stats.get("context_reduction_vs_full_context", 0.0) or 0.0),
        selected_memory_count=int(stats.get("selected_memory_count", 0) or 0),
        selected_chunk_count=int(stats.get("selected_chunk_count", 0) or 0),
        history_messages_considered=int(stats.get("history_messages_considered", 0) or 0),
        history_messages_selected=int(stats.get("history_messages_selected", 0) or 0),
        retrieved_memory_ids=tuple(
            record.memory_id for record in turn.retrieved_memories
        ),
        retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in turn.retrieved_chunks),
        retrieved_memory_texts=tuple(
            record.text for record in turn.retrieved_memories
        ),
        retrieved_chunk_texts=tuple(chunk.text for chunk in turn.retrieved_chunks),
        stored_memory_count=len(service.stored_memories()),
        sessions_used=sessions_used,
        expected_answer_in_prompt=bool(expected) and expected in prompt,
        prompt_messages=tuple(dict(message) for message in turn.context_messages),
        stats=stats,
        retrieval_report=dict(turn.context_report),
        security=_replay_security(turn.security),
        answer=generation.text if generation is not None else None,
        generation=generation.to_dict() if generation is not None else None,
    )


def compare_modes(
    task: BenchmarkTask,
    modes: Sequence[str] = MODE_ORDER,
    *,
    service_factory: ServiceFactory | None = None,
    session_isolation: bool = False,
    generate: Generator | None = None,
) -> list[ModeReplay]:
    """Replay one task through every requested mode, in plan order.

    Each mode gets its own session, so the comparison is between strategies and
    not between a warm and a cold runtime.
    """

    return [
        replay_task(
            task,
            mode,
            service_factory=service_factory,
            session_isolation=session_isolation,
            generate=generate,
        )
        for mode in modes
    ]


def task_evaluator(
    service_factory: ServiceFactory | None = None,
    *,
    session_isolation: bool = False,
    generate: Generator | None = None,
) -> Callable[[BenchmarkTask, str], dict[str, Any]]:
    """Return the evaluator callable :meth:`EvaluationRunner.run` expects.

    The runner supplies the mode from its own configuration, so a single
    evaluator serves every mode and a run cannot accidentally mix strategies.
    """

    def evaluate(task: BenchmarkTask, mode: str) -> dict[str, Any]:
        return replay_task(
            task,
            mode,
            service_factory=service_factory,
            session_isolation=session_isolation,
            generate=generate,
        ).to_dict()

    return evaluate


def _replay_security(block: Mapping[str, Any] | None) -> dict[str, Any]:
    """The turn's guard report, minus the replay's throwaway session id.

    A replay mints a session per (task, mode) and nothing else in the artifact
    can refer to it, so keeping the identifier would be noise that reads like
    provenance. The counts, the bounded findings, and the caveat are what the
    artifact needs.
    """

    if not isinstance(block, Mapping):
        return {}
    payload = dict(block)
    payload.pop("session_id", None)
    return payload


def _message_session(message: Mapping[str, Any]) -> int:
    """Return the transcript session index a message belongs to (default 0)."""

    try:
        return int(message.get("session", 0) or 0)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ModeReplay",
    "ServiceFactory",
    "compare_modes",
    "REPLAY_CEILING_SLACK",
    "default_service_factory",
    "replay_ceiling",
    "replay_task",
    "task_evaluator",
]
