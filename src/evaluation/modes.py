"""Baseline-mode strategies for the evaluation runner (plan Phase 6).

This module is the seam the plan's automated pipeline (Phase 17) plugs into:
:meth:`evaluation.runner.EvaluationRunner.run` accepts an evaluator callable of
shape ``(task, mode) -> record``, and :func:`task_evaluator` produces exactly
that from the application's own context-construction path.

What this module deliberately does **not** do:

* **No model calls.** A replay has no provider configured, so no generation
  happens and no API key is needed. Answer accuracy, faithfulness, and
  abstention need a model and belong to Phase 8.
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
from dataclasses import dataclass, field
from typing import Any

from app.service import ConversationService
from app.state import ContextSettings, SessionState
from baselines.modes import MODE_ORDER, mode_label, resolve_mode

from .datasets import BenchmarkTask

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
    #: Evidence-availability proxy only: the expected answer string appears in
    #: the prompt. Not accuracy — see the module docstring.
    expected_answer_in_prompt: bool = False
    prompt_messages: tuple[dict[str, str], ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record for the runner and the evaluation store."""

        return {
            "task_id": self.task_id,
            "category": self.category,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "question": self.question,
            "expected_answer": self.expected_answer,
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
            "expected_answer_in_prompt": self.expected_answer_in_prompt,
            "prompt_messages": [dict(message) for message in self.prompt_messages],
            "stats": dict(self.stats),
        }


def default_service_factory(mode: str) -> ConversationService:
    """Build a fresh, purely in-memory service configured for ``mode``.

    No storage backends are attached, so a replay never touches disk, and no
    provider configuration is set, so a replay never calls a model.
    """

    state = SessionState(context=ContextSettings().with_mode_defaults(mode))
    return ConversationService(state)


def replay_task(
    task: BenchmarkTask,
    mode: str,
    *,
    service_factory: ServiceFactory | None = None,
) -> ModeReplay:
    """Run one benchmark task's conversation through one baseline mode.

    The scripted transcript is replayed turn by turn — user turns through the
    normal path, assistant turns through
    :meth:`~app.service.ConversationService.record_assistant_message` — and then
    the task's question is asked. The returned record describes the prompt that
    question produced.
    """

    factory = service_factory or default_service_factory
    service = factory(resolve_mode(mode))
    for message in task.conversation:
        if not isinstance(message, Mapping):
            continue
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
        expected_answer_in_prompt=bool(expected) and expected in prompt,
        prompt_messages=tuple(dict(message) for message in turn.context_messages),
        stats=stats,
    )


def compare_modes(
    task: BenchmarkTask,
    modes: Sequence[str] = MODE_ORDER,
    *,
    service_factory: ServiceFactory | None = None,
) -> list[ModeReplay]:
    """Replay one task through every requested mode, in plan order.

    Each mode gets its own session, so the comparison is between strategies and
    not between a warm and a cold runtime.
    """

    return [
        replay_task(task, mode, service_factory=service_factory) for mode in modes
    ]


def task_evaluator(
    service_factory: ServiceFactory | None = None,
) -> Callable[[BenchmarkTask, str], dict[str, Any]]:
    """Return the evaluator callable :meth:`EvaluationRunner.run` expects.

    The runner supplies the mode from its own configuration, so a single
    evaluator serves every mode and a run cannot accidentally mix strategies.
    """

    def evaluate(task: BenchmarkTask, mode: str) -> dict[str, Any]:
        return replay_task(task, mode, service_factory=service_factory).to_dict()

    return evaluate


__all__ = [
    "ModeReplay",
    "ServiceFactory",
    "compare_modes",
    "default_service_factory",
    "replay_task",
    "task_evaluator",
]
