"""Live Phase 6 validation: the five baseline modes against the pinned BrainOS.

The unit tests drive the modes with a fake runtime. This file runs the same
conversation through all five modes with the real runtime and a deterministic
provider double, so the cross-mode numbers reported in the phase log come from
the code that ships rather than from a stand-in.

Two rules the file enforces on itself:

* no provider API key is used — generation goes through ``FakeLLMProvider``, and
  the evaluation replay path configures no provider at all;
* the assertions are about *relationships* between modes (which one replays
  everything, which one keeps the fact, which one is cheaper), never about a
  specific token count, because a token count is a property of one synthetic
  conversation and not a result.
"""

from __future__ import annotations

import pytest

from app.controller import UIController
from baselines.modes import (
    EVIDENCE_BUDGET,
    MODE_BRAINOS,
    MODE_BRAINOS_RAG,
    MODE_FULL_CONTEXT,
    MODE_ORDER,
    MODE_RAG,
    MODE_SLIDING_WINDOW,
    MODES,
)
from evaluation.datasets import BenchmarkTask
from evaluation.modes import compare_modes, replay_task, task_evaluator
from evaluation.runner import EvaluationConfig, EvaluationRunner
from tests.fakes import FakeLLMProvider

pytest.importorskip("brainos_runtime")

FACT = "For Project Atlas, the production database is PostgreSQL 16."
DEPLOY = "We deploy on Fridays at 17:00 UTC."
QUESTION = "What database does Project Atlas use in production?"
FILLER_TURNS = 12


def _transcript() -> list[dict[str, str]]:
    conversation = [
        {"role": "user", "content": FACT},
        {"role": "assistant", "content": "Noted: Project Atlas runs PostgreSQL 16."},
    ]
    for index in range(FILLER_TURNS):
        conversation.append(
            {"role": "user", "content": f"How is the rollout looking today? Check {index}."}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": f"Rollout check {index} is steady; nothing new to report.",
            }
        )
    conversation += [
        {"role": "user", "content": DEPLOY},
        {"role": "assistant", "content": "Understood: the window is Friday 17:00 UTC."},
    ]
    return conversation


def _task() -> BenchmarkTask:
    conversation = _transcript()
    return BenchmarkTask(
        task_id="live-mode-comparison",
        category="cross_session",
        conversation=conversation,
        question=QUESTION,
        expected_answer="PostgreSQL",
        conversation_length=len(conversation),
    )


def _prompt(replay) -> str:
    return "\n".join(message["content"] for message in replay.prompt_messages)


def test_live_all_five_modes_run_against_the_real_runtime() -> None:
    replays = {item.mode: item for item in compare_modes(_task())}

    assert set(replays) == set(MODE_ORDER)
    for mode, replay in replays.items():
        assert replay.prompt_messages, mode
        assert replay.final_context_tokens > 0, mode
        assert replay.stats["token_counter"], mode
        assert replay.mode_label.startswith("Mode "), mode


def test_live_full_context_replays_everything_and_is_the_reference() -> None:
    replay = replay_task(_task(), MODE_FULL_CONTEXT)

    assert replay.history_messages_selected == replay.history_messages_considered
    assert replay.final_context_tokens == replay.full_context_reference_tokens
    assert replay.context_reduction == 0.0
    assert replay.selected_memory_count == 0
    assert replay.selected_chunk_count == 0
    assert replay.expected_answer_in_prompt is True


def test_live_the_sliding_window_loses_the_fact() -> None:
    replay = replay_task(_task(), MODE_SLIDING_WINDOW)

    assert replay.final_context_tokens < replay.full_context_reference_tokens
    assert replay.expected_answer_in_prompt is False, (
        "the fact is older than the window and Mode B has nothing to retrieve it"
    )


def test_live_brainos_keeps_the_fact_for_the_same_cost_as_the_window() -> None:
    """The comparison the project exists to make, on the pinned runtime."""

    windowed = replay_task(_task(), MODE_SLIDING_WINDOW)
    managed = replay_task(_task(), MODE_BRAINOS)

    assert managed.selected_memory_count > 0
    assert managed.expected_answer_in_prompt is True
    assert managed.final_context_tokens <= windowed.final_context_tokens
    assert managed.context_reduction > 0.0
    assert managed.full_context_reference_tokens == windowed.full_context_reference_tokens


def test_live_lexical_rag_reaches_back_without_brainos() -> None:
    replay = replay_task(_task(), MODE_RAG)

    assert replay.selected_memory_count == 0
    assert replay.selected_chunk_count > 0
    assert replay.expected_answer_in_prompt is True
    assert "<retrieved_memory>" not in _prompt(replay)
    assert "<retrieved_history>" in _prompt(replay)


def test_live_the_hybrid_carries_both_sources() -> None:
    replay = replay_task(_task(), MODE_BRAINOS_RAG)

    assert replay.selected_memory_count > 0
    assert replay.selected_chunk_count > 0
    assert replay.expected_answer_in_prompt is True
    assert replay.stats["evidence_tokens"] == (
        replay.stats["memory_block_tokens"] + replay.stats["chunk_block_tokens"]
    )


def test_live_the_retrieval_modes_spend_within_one_shared_evidence_budget() -> None:
    """Equal allowance, unequal spend — the allowance is what must be equal.

    Each block is filled to fit inside its own section budget, so the evidence
    spend can never exceed the shared allowance. How *much* of that allowance a
    mode actually uses is the finding, not a violation.
    """

    replays = {item.mode: item for item in compare_modes(_task())}

    for mode in (MODE_RAG, MODE_BRAINOS, MODE_BRAINOS_RAG):
        stats = replays[mode].stats
        assert stats["evidence_tokens"] <= EVIDENCE_BUDGET, (mode, stats)
        assert stats["evidence_tokens"] == (
            stats["memory_block_tokens"] + stats["chunk_block_tokens"]
        )

    spends = {mode: replays[mode].stats["evidence_tokens"] for mode in replays}
    assert len(set(spends.values())) > 1, spends


def test_live_each_evidence_source_stays_inside_its_own_budget() -> None:
    replays = {item.mode: item for item in compare_modes(_task())}
    hybrid = replays[MODE_BRAINOS_RAG].stats

    assert hybrid["memory_block_tokens"] <= MODES[MODE_BRAINOS_RAG].memory_budget
    assert hybrid["chunk_block_tokens"] <= MODES[MODE_BRAINOS_RAG].chunk_budget


def test_live_memory_is_recorded_in_every_mode() -> None:
    """Observing is a side process; only injection is mode-dependent."""

    controller = UIController(provider_factory=FakeLLMProvider)
    for mode in MODE_ORDER:
        session_id = controller.connect(
            None, provider="openai", model="fake-model", api_key=""
        ).session_id
        controller.apply_mode(session_id, mode)
        controller.chat(session_id, FACT)

        assert controller.service(session_id).stored_memories(), mode


def test_live_the_runner_executes_a_task_through_the_mode_seam() -> None:
    """Phase 6's contribution to the Phase 17 pipeline."""

    runner = EvaluationRunner(EvaluationConfig(mode=MODE_BRAINOS))

    run = runner.run([_task()], evaluator=task_evaluator())

    assert len(run.task_results) == 1
    record = run.task_results[0]
    assert record["mode"] == MODE_BRAINOS
    assert record["mode_label"].startswith("Mode D")
    assert record["final_context_tokens"] > 0
    assert "api_key" not in str(record).lower()


def test_live_the_controller_reports_the_mode_that_ran() -> None:
    controller = UIController(provider_factory=FakeLLMProvider)
    session_id = controller.connect(
        None, provider="openai", model="fake-model", api_key=""
    ).session_id
    controller.apply_mode(session_id, MODE_BRAINOS)
    controller.chat(session_id, FACT)
    for index in range(FILLER_TURNS):
        controller.chat(session_id, f"How is the rollout looking today? Check {index}.")

    view = controller.chat(session_id, QUESTION)

    assert "mode `brainos`" in view.status
    assert "Mode D" in view.summary
    assert view.retrieved_rows
    assert any("PostgreSQL" in str(row[1]) for row in view.retrieved_rows), view.retrieved_rows
