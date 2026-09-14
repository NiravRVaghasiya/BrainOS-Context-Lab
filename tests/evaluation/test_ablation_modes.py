"""Phase 11 ablation behaviour through the application pipeline.

These run on fake runtimes so the ablation study is covered without the
integration extra; ``tests/integration/test_ablation_live.py`` repeats the
comparison against the pinned BrainOS revision. What is pinned here is the
removal each ablation promises: no temporal influence on ranking, no
relevance floor, no conflict/staleness handling, and no injected memory —
with the system prompt, task wording, window, and budgets held constant.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.service import ConversationService
from app.state import ContextSettings, SessionState
from baselines.ablations import ABLATIONS, prepare_records
from baselines.modes import (
    ABLATION_NO_CONFLICT,
    ABLATION_NO_MEMORY,
    ABLATION_NO_RELEVANCE,
    ABLATION_NO_TEMPORAL,
    ABLATION_ORDER,
    MODE_BRAINOS,
)
from brain.adapter import BrainOSAdapter, MemoryRecord
from brain.retrieval_policy import RetrievalPolicy, select_memories
from evaluation import modes as evaluation_modes
from evaluation.datasets import BenchmarkTask, load_jsonl
from evaluation.experiment import (
    ExperimentPlan,
    _parse_modes,
    run_controlled_experiment,
)
from evaluation.limits import RunLimits
from evaluation.modes import compare_modes, replay_task
from tests.fakes import FakeRuntime, LooseFakeRuntime

FACT = "For Project Atlas, the production database is PostgreSQL 16."
UNRELATED = "The backup window requires 400 days of audit logs."
QUESTION = "What database does Project Atlas use in production?"
DATASET = Path("benchmarks/context_rot/dataset.jsonl")


def _service(mode: str, runtime: FakeRuntime | None = None) -> ConversationService:
    state = SessionState(context=ContextSettings().with_mode_defaults(mode))
    return ConversationService(
        state, adapter=BrainOSAdapter(runtime or FakeRuntime())
    )


def _fake_factory(mode: str) -> ConversationService:
    return _service(mode)


def _task(filler: int = 8) -> BenchmarkTask:
    conversation: list[dict[str, str]] = [{"role": "user", "content": FACT}]
    for index in range(filler):
        conversation.append(
            {"role": "user", "content": f"Rollout check {index} is steady."}
        )
    return BenchmarkTask(
        task_id="ablation-t1",
        category="single_hop",
        conversation=conversation,
        question=QUESTION,
        expected_answer="PostgreSQL",
        conversation_length=len(conversation) + 1,
    )


# --------------------------------------------------------------------------- #
# Removal behaviour
# --------------------------------------------------------------------------- #


def test_no_temporal_ranks_without_recency() -> None:
    """The old memory is lexically stronger; only recency lets the new one win."""

    now = datetime.now(timezone.utc)
    old_stamp = (now - timedelta(days=30)).isoformat()
    query = "What database does Project Atlas use?"
    old = MemoryRecord(
        memory_id="old",
        text="Project Atlas production database uses PostgreSQL for records.",
        observed_at=old_stamp,
        created_at=old_stamp,
        source_turn=1,
        signals={
            "lexical": 0.65,
            "salience": 0.65,
            "recency": 0.0,
            "temporal_relevance": 1.0,
        },
    )
    new = MemoryRecord(
        memory_id="new",
        text="Project Atlas database cache uses memory.",
        observed_at=now.isoformat(),
        created_at=now.isoformat(),
        source_turn=50,
        signals={
            "lexical": 0.45,
            "salience": 0.45,
            "recency": 1.0,
            "temporal_relevance": 1.0,
        },
    )

    full = select_memories(
        query, [old, new], policy=RetrievalPolicy(), current_turn=55, now=now
    )
    assert [item.memory_id for item in full.ordered] == ["new", "old"]

    stripped = prepare_records(ABLATION_NO_TEMPORAL, [old, new])
    assert all(
        "recency" not in record.signals
        and "temporal_relevance" not in record.signals
        for record in stripped
    )
    ablated = select_memories(
        query,
        stripped,
        policy=RetrievalPolicy(weight_recency=0.0),
        current_turn=55,
        now=now,
    )
    assert [item.memory_id for item in ablated.ordered] == ["old", "new"]


def _bury_with_filler(service: ConversationService, turns: int = 4) -> None:
    """Push earlier turns out of the small Mode D window.

    The filler is un-storable (questions are never candidates), so the facts
    stay reachable only through memory — exactly what the ablations remove.
    """

    for index in range(turns):
        service.handle_user_message(f"How is the rollout looking? Check {index}?")


def _prompt_text(turn) -> str:  # type: ignore[no-untyped-def]
    return "\n".join(message["content"] for message in turn.context_messages)


def test_no_relevance_keeps_what_the_floor_would_drop() -> None:
    service_full = _service(MODE_BRAINOS, LooseFakeRuntime())
    service_full.handle_user_message(FACT)
    service_full.handle_user_message(UNRELATED)
    _bury_with_filler(service_full)
    full = service_full.handle_user_message(QUESTION)

    service_ablated = _service(ABLATION_NO_RELEVANCE, LooseFakeRuntime())
    service_ablated.handle_user_message(FACT)
    service_ablated.handle_user_message(UNRELATED)
    _bury_with_filler(service_ablated)
    ablated = service_ablated.handle_user_message(QUESTION)

    assert service_full.state.context.relevance_floor > 0
    assert service_ablated.state.context.relevance_floor == 0.0
    assert full.context_stats["selected_memory_count"] == 1
    assert ablated.context_stats["selected_memory_count"] == 2
    assert "audit logs" not in _prompt_text(full)
    assert "audit logs" in _prompt_text(ablated)
    assert "PostgreSQL 16" in _prompt_text(full)
    reasons = {
        item["reason"] for item in full.context_report["dropped"]
    }
    assert "low_relevance" in reasons


def test_no_conflict_keeps_the_superseded_claim() -> None:
    old = FACT
    new = "Correction: the production database is MySQL 8 now."
    contradiction = [
        {
            "subject": "production database",
            "older_id": "session-a-m1",
            "newer_id": "session-a-m2",
            "older_content": old,
            "newer_content": new,
        }
    ]

    service_full = _service(MODE_BRAINOS, FakeRuntime(contradictions=contradiction))
    service_full.handle_user_message(old)
    service_full.handle_user_message(new)
    _bury_with_filler(service_full)
    full = service_full.handle_user_message(QUESTION)

    service_ablated = _service(
        ABLATION_NO_CONFLICT, FakeRuntime(contradictions=contradiction)
    )
    service_ablated.handle_user_message(old)
    service_ablated.handle_user_message(new)
    _bury_with_filler(service_ablated)
    ablated = service_ablated.handle_user_message(QUESTION)

    assert full.context_stats["selected_memory_count"] == 1
    assert ablated.context_stats["selected_memory_count"] == 2
    assert "PostgreSQL 16" not in _prompt_text(full)
    assert "MySQL 8" in _prompt_text(full)
    assert "PostgreSQL 16" in _prompt_text(ablated)
    assert "MySQL 8" in _prompt_text(ablated)
    reasons = {
        item["reason"] for item in full.context_report["dropped"]
    }
    assert "superseded" in reasons
    ablated_reasons = {
        item["reason"] for item in ablated.context_report["dropped"]
    }
    assert "superseded" not in ablated_reasons


def test_no_memory_injects_nothing_but_keeps_the_window() -> None:
    replay = replay_task(_task(), ABLATION_NO_MEMORY, service_factory=_fake_factory)

    assert replay.selected_memory_count == 0
    assert replay.retrieved_memory_ids == ()
    assert "<retrieved_memory>" not in replay.prompt_text()
    # Mode D's small window, not Mode B's eight turns.
    assert replay.history_messages_selected <= 2


# --------------------------------------------------------------------------- #
# Routing and controlled-comparison plumbing
# --------------------------------------------------------------------------- #


def test_every_ablation_replays_through_the_same_path() -> None:
    replays = compare_modes(
        _task(), [MODE_BRAINOS, *ABLATION_ORDER], service_factory=_fake_factory
    )

    assert [replay.mode for replay in replays] == [MODE_BRAINOS, *ABLATION_ORDER]
    prompts = [replay.prompt_messages for replay in replays]
    systems = [messages[0]["content"] for messages in prompts]
    assert len(set(systems)) == 1, "the system prompt must not change"
    questions = [
        messages[-1]["content"]
        for messages in prompts
        if messages[-1]["role"] == "user"
    ]
    assert questions == [QUESTION] * len(replays)
    for replay in replays:
        assert replay.task_id == "ablation-t1"
        assert replay.mode_label.startswith("Ablation D") or replay.mode == MODE_BRAINOS


def test_parse_modes_ablations_keyword_includes_the_full_reference() -> None:
    assert _parse_modes("ablations") == (MODE_BRAINOS, *ABLATION_ORDER)
    assert _parse_modes("all") == tuple(
        __import__("baselines.modes", fromlist=["MODE_ORDER"]).MODE_ORDER
    )


def test_ablation_experiment_pairs_each_ablation_with_full(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        evaluation_modes, "default_service_factory", _fake_factory
    )
    tasks = load_jsonl(DATASET)[:2]
    plan = ExperimentPlan(
        name="ablation",
        modes=(MODE_BRAINOS, *ABLATION_ORDER),
        trials=1,
        limits=RunLimits(max_requests=1000),
        dataset=str(DATASET),
        dataset_sha256="test-digest",
    )

    run = run_controlled_experiment(plan, tasks, baseline_mode=MODE_BRAINOS)

    assert run.passed is True
    assert run.baseline_mode == MODE_BRAINOS
    assert plan.resolved_modes() == (MODE_BRAINOS, *ABLATION_ORDER)
    summary = run.statistical_summary()
    pairs = {
        (item["mode_a"], item["mode_b"])
        for item in summary["paired_comparisons"]
    }
    for ablation in ABLATION_ORDER:
        assert (ablation, MODE_BRAINOS) in pairs, ablation
    payload = run.to_dict()
    assert payload["baseline_mode"] == MODE_BRAINOS
    assert set(ABLATIONS) <= {
        row["mode"] for row in payload["headline"]
    }
    output = tmp_path / "ablation.json"
    from evaluation.reports import write_json_report

    write_json_report(payload, output)
    assert output.exists()


def test_run_cli_accepts_an_ablation_mode(monkeypatch, tmp_path: Path) -> None:
    from evaluation import run as run_cli

    monkeypatch.setattr(
        evaluation_modes, "default_service_factory", _fake_factory
    )
    output = tmp_path / "ablation-run.json"

    exit_code = run_cli.main(
        [
            "--mode", ABLATION_NO_MEMORY,
            "--dataset", str(DATASET),
            "--limit", "2",
            "--output", str(output),
        ]
    )

    assert exit_code == 0
    import json

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"]["mode"] == ABLATION_NO_MEMORY
    assert payload["aggregate_metrics"]["task_count"] == 2
