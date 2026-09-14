"""Live Phase 11 validation: the ablations against the pinned BrainOS revision.

The fake-runtime tests pin each removal's mechanism. This file runs the same
conversations through the full system and the four ablations with the real
runtime and no provider key, so the ablation numbers in the phase log come
from the code that ships.

Assertions are about relationships between the full system and each ablation
(which one keeps the stale claim, which one drops the irrelevant memory),
never about a specific token count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.service import ConversationService
from app.state import ContextSettings, SessionState
from baselines.modes import (
    ABLATION_NO_CONFLICT,
    ABLATION_NO_MEMORY,
    ABLATION_NO_RELEVANCE,
    ABLATION_NO_TEMPORAL,
    ABLATION_ORDER,
    MODE_BRAINOS,
)
from evaluation.analysis import pairwise_comparisons, summarize_trial_modes
from evaluation.datasets import load_jsonl
from evaluation.modes import compare_modes, replay_task
from evaluation.scoring import aggregate_scores, score_record

pytest.importorskip("brainos_runtime")

DATASET = Path("benchmarks/context_rot/dataset.jsonl")
TASKS = load_jsonl(DATASET)

OLD = "The production database is PostgreSQL 16."
NEW = "Correction: the production database is MySQL 8 now."
FACT = "For Project Atlas, the production database is PostgreSQL 16."
UNRELATED = "The backup window requires 400 days of audit logs."
QUESTION = "What database does Project Atlas use in production?"


def _service(mode: str) -> ConversationService:
    state = SessionState(context=ContextSettings().with_mode_defaults(mode))
    return ConversationService(state)


def _bury(service: ConversationService, turns: int = 4) -> None:
    for index in range(turns):
        service.handle_user_message(f"How is the rollout looking? Check {index}?")


def _prompt_text(turn) -> str:  # type: ignore[no-untyped-def]
    return "\n".join(message["content"] for message in turn.context_messages)


def test_live_runtime_reports_no_conflicts_on_the_observe_path() -> None:
    """The Phase 11 premise: the observe path never versions by subject, so the
    runtime's contradiction/stale reports are inert here and the application's
    heuristic is the operative conflict handling."""

    service = _service(MODE_BRAINOS)
    service.handle_user_message(OLD)
    service.handle_user_message(NEW)
    _bury(service)
    service.handle_user_message(QUESTION)

    assert service.adapter.conflicts() == []
    assert service.adapter.stale_memory_ids() == set()


def test_live_no_conflict_keeps_the_superseded_claim() -> None:
    full_service = _service(MODE_BRAINOS)
    full_service.handle_user_message(OLD)
    full_service.handle_user_message(NEW)
    _bury(full_service)
    full = full_service.handle_user_message(QUESTION)

    ablated_service = _service(ABLATION_NO_CONFLICT)
    ablated_service.handle_user_message(OLD)
    ablated_service.handle_user_message(NEW)
    _bury(ablated_service)
    ablated = ablated_service.handle_user_message(QUESTION)

    assert full.context_stats["selected_memory_count"] == 1
    assert ablated.context_stats["selected_memory_count"] == 2
    assert "PostgreSQL 16" not in _prompt_text(full)
    assert "MySQL 8" in _prompt_text(full)
    assert "PostgreSQL 16" in _prompt_text(ablated)
    assert "MySQL 8" in _prompt_text(ablated)


def test_live_no_relevance_keeps_the_irrelevant_memory() -> None:
    full_service = _service(MODE_BRAINOS)
    full_service.handle_user_message(FACT)
    full_service.handle_user_message(UNRELATED)
    _bury(full_service)
    full = full_service.handle_user_message(QUESTION)

    ablated_service = _service(ABLATION_NO_RELEVANCE)
    ablated_service.handle_user_message(FACT)
    ablated_service.handle_user_message(UNRELATED)
    _bury(ablated_service)
    ablated = ablated_service.handle_user_message(QUESTION)

    assert ablated.context_stats["selected_memory_count"] >= (
        full.context_stats["selected_memory_count"]
    )
    assert "audit logs" not in _prompt_text(full)
    assert "audit logs" in _prompt_text(ablated)


def test_live_no_memory_sends_the_window_without_memories() -> None:
    full = replay_task(TASKS[0], MODE_BRAINOS)
    ablated = replay_task(TASKS[0], ABLATION_NO_MEMORY)

    assert ablated.selected_memory_count == 0
    assert ablated.retrieved_memory_ids == ()
    assert "<retrieved_memory>" not in ablated.prompt_text()
    assert "<retrieved_memory>" in full.prompt_text()
    assert (
        ablated.history_messages_selected == full.history_messages_selected
    )


def test_live_ablations_share_window_system_prompt_and_question() -> None:
    modes = [MODE_BRAINOS, *ABLATION_ORDER]
    replays = compare_modes(TASKS[0], modes)

    assert [replay.mode for replay in replays] == modes
    windows = {replay.history_messages_selected for replay in replays}
    assert len(windows) == 1, "the window must not change between ablations"
    systems = {replay.prompt_messages[0]["content"] for replay in replays}
    assert len(systems) == 1, "the system prompt must not change"
    for replay in replays:
        assert replay.prompt_messages[-1]["content"] == TASKS[0].question
        assert replay.final_context_tokens > 0
        assert replay.stored_memory_count > 0


def test_live_no_relevance_never_selects_less_than_full() -> None:
    """Disabling a filter can only keep more; the relevance floor only drops."""

    for task in TASKS:
        full = replay_task(task, MODE_BRAINOS)
        ablated = replay_task(task, ABLATION_NO_RELEVANCE)
        assert ablated.selected_memory_count >= full.selected_memory_count, task.task_id


def test_live_ablation_pairs_are_computed_against_full() -> None:
    modes = [MODE_BRAINOS, *ABLATION_ORDER]
    bundle: dict[str, list] = {"modes": []}
    for mode in modes:
        scores = []
        for task in TASKS:
            replay = replay_task(task, mode)
            scores.append(score_record(task, replay.to_dict()))
        bundle["modes"].append(
            {
                "mode": mode,
                "label": mode,
                "trial": 0,
                "aggregate_metrics": aggregate_scores(scores),
                "scores": scores,
            }
        )

    summaries = summarize_trial_modes(bundle)
    assert set(summaries) == set(modes)
    pairs = {
        (item["mode_a"], item["mode_b"])
        for item in pairwise_comparisons(bundle, baseline_mode=MODE_BRAINOS)
    }
    for ablation in ABLATION_ORDER:
        assert (ablation, MODE_BRAINOS) in pairs, ablation
    temporal = summaries[ABLATION_NO_TEMPORAL]
    assert temporal["trials"] == 1
