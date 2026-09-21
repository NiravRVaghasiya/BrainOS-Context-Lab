"""Tests for the Phase 6 evaluation seam (mode strategies).

These run on the fake runtime so the seam is covered in an environment without
the integration extra; ``tests/integration/test_baseline_modes_live.py`` repeats
the comparison against the pinned BrainOS revision.
"""

from __future__ import annotations

import pytest

from app.service import ConversationService
from app.state import ContextSettings, SessionState
from baselines.modes import (
    MODE_BRAINOS,
    MODE_BRAINOS_RAG,
    MODE_FULL_CONTEXT,
    MODE_ORDER,
    MODE_RAG,
    MODE_SLIDING_WINDOW,
)
from brain.adapter import BrainOSAdapter
from evaluation.datasets import BenchmarkTask
from evaluation.modes import (
    ModeReplay,
    compare_modes,
    default_service_factory,
    replay_task,
    task_evaluator,
)
from evaluation.runner import EvaluationConfig, EvaluationRunner
from tests.fakes import FakeRuntime

FACT = "For Project Atlas, the production database is PostgreSQL 16."
QUESTION = "What database does Project Atlas use in production?"


def _task(filler: int = 8) -> BenchmarkTask:
    conversation: list[dict[str, str]] = [{"role": "user", "content": FACT}]
    for index in range(filler):
        conversation.append({"role": "user", "content": f"Rollout check {index} is steady."})
    return BenchmarkTask(
        task_id="t1",
        category="cross_session",
        conversation=conversation,
        question=QUESTION,
        expected_answer="PostgreSQL",
        conversation_length=len(conversation),
    )


def _fake_factory(mode: str) -> ConversationService:
    state = SessionState(context=ContextSettings().with_mode_defaults(mode))
    return ConversationService(state, adapter=BrainOSAdapter(FakeRuntime()))


@pytest.mark.requires_runtime
def test_the_default_factory_builds_a_memory_less_provider_less_service() -> None:
    """A replay must not call a model or touch disk."""

    service = default_service_factory(MODE_BRAINOS)

    assert service.state.context.mode == MODE_BRAINOS
    assert service._can_generate() is False
    assert service._conversation_store is None
    assert service._memory_store is None


@pytest.mark.requires_runtime
def test_each_mode_gets_its_own_session() -> None:
    """One task's memory must not leak into the next mode's replay."""

    sessions = [
        default_service_factory(mode).state.session_id for mode in MODE_ORDER
    ]

    assert len(set(sessions)) == len(MODE_ORDER)


def test_replay_uses_the_mode_it_was_given() -> None:
    replay = replay_task(_task(), MODE_RAG, service_factory=_fake_factory)

    assert replay.mode == MODE_RAG
    assert replay.mode_label.startswith("Mode C")
    assert replay.selected_chunk_count > 0
    assert replay.selected_memory_count == 0


def test_replay_resolves_legacy_mode_values() -> None:
    replay = replay_task(_task(filler=0), "no_memory", service_factory=_fake_factory)

    assert replay.mode == MODE_SLIDING_WINDOW


def test_replay_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown baseline mode"):
        replay_task(_task(), "telepathy", service_factory=_fake_factory)


def test_replay_records_the_scripted_assistant_turns() -> None:
    conversation = [
        {"role": "user", "content": FACT},
        {"role": "assistant", "content": "Noted: Project Atlas runs PostgreSQL 16."},
    ]
    task = BenchmarkTask(
        task_id="t2",
        category="single_hop",
        conversation=conversation,
        question=QUESTION,
        expected_answer="PostgreSQL",
        conversation_length=2,
    )

    replay = replay_task(task, MODE_FULL_CONTEXT, service_factory=_fake_factory)

    # 2 scripted messages plus the question the replay asked.
    assert replay.history_messages_considered == 2
    assert replay.prompt_messages[-1]["content"] == QUESTION


def test_replay_reports_the_full_context_reference_for_every_mode() -> None:
    task = _task()
    references = {
        replay_task(task, mode, service_factory=_fake_factory).full_context_reference_tokens
        for mode in MODE_ORDER
    }

    assert len(references) == 1, references


def test_replay_records_which_modes_lost_the_evidence() -> None:
    task = _task()

    windowed = replay_task(task, MODE_SLIDING_WINDOW, service_factory=_fake_factory)
    managed = replay_task(task, MODE_BRAINOS, service_factory=_fake_factory)

    assert windowed.expected_answer_in_prompt is False
    assert managed.expected_answer_in_prompt is True


def test_compare_modes_returns_one_replay_per_mode_in_plan_order() -> None:
    replays = compare_modes(_task(), service_factory=_fake_factory)

    assert [replay.mode for replay in replays] == list(MODE_ORDER)
    assert all(isinstance(replay, ModeReplay) for replay in replays)


def test_compare_modes_accepts_a_subset() -> None:
    replays = compare_modes(
        _task(), modes=(MODE_FULL_CONTEXT, MODE_BRAINOS_RAG), service_factory=_fake_factory
    )

    assert [replay.mode for replay in replays] == [MODE_FULL_CONTEXT, MODE_BRAINOS_RAG]
    assert replays[1].selected_memory_count > 0
    assert replays[1].selected_chunk_count > 0


def test_replay_to_dict_is_json_shaped_and_credential_free() -> None:
    import json

    record = replay_task(_task(), MODE_BRAINOS, service_factory=_fake_factory).to_dict()

    json.dumps(record)
    assert record["mode"] == MODE_BRAINOS
    assert record["stats"]["final_context_tokens"] > 0
    assert "api_key" not in str(record).lower()


def test_the_task_evaluator_plugs_into_the_runner() -> None:
    """The seam the Phase 17 pipeline will use."""

    runner = EvaluationRunner(EvaluationConfig(mode=MODE_BRAINOS))

    run = runner.run(
        [_task(), _task(filler=2)],
        evaluator=task_evaluator(service_factory=_fake_factory),
    )

    assert len(run.task_results) == 2
    assert {result["mode"] for result in run.task_results} == {MODE_BRAINOS}
    assert run.config.mode == MODE_BRAINOS


def test_the_runner_still_refuses_to_run_without_a_strategy() -> None:
    runner = EvaluationRunner(EvaluationConfig(mode=MODE_BRAINOS))

    with pytest.raises(NotImplementedError):
        runner.run([_task()])


def test_conversation_length_falls_back_to_the_transcript() -> None:
    task = BenchmarkTask(
        task_id="t3",
        category="single_hop",
        conversation=[{"role": "user", "content": FACT}],
        question=QUESTION,
        expected_answer="PostgreSQL",
        conversation_length=None,
    )

    replay = replay_task(task, MODE_RAG, service_factory=_fake_factory)

    assert replay.conversation_length == 1


def test_malformed_transcript_entries_are_skipped() -> None:
    task = BenchmarkTask(
        task_id="t4",
        category="single_hop",
        conversation=[{"role": "user", "content": FACT}, "not a message"],  # type: ignore[list-item]
        question=QUESTION,
        expected_answer="PostgreSQL",
        conversation_length=1,
    )

    replay = replay_task(task, MODE_FULL_CONTEXT, service_factory=_fake_factory)

    assert replay.history_messages_considered == 1
