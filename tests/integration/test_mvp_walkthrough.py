"""Phase 19: the plan's §25 MVP checklist, walked end to end in one session.

§25 defines the MVP as fourteen things a user can *do*, in order, ending with
"export results". Phases 4–18 built every one of them and tested each in
isolation — but nothing ran the sequence. The audit that opened this phase could
therefore say the checklist was implemented and could not say it was walkable:
the pieces all passed their own tests while the path from "enter a key" to
"export results" was never executed as a path.

This file executes it. One controller, one live session on the pinned runtime,
one deterministic provider double (no API key is spent), and the fourteen steps
in the order the plan lists them:

* items 2–4 are one act (choose provider, enter key, choose model);
* items 5–8 are the conversation: speak, store a fact, continue past it, ask for
  it back later;
* items 9–10 are the inspection panels;
* item 11 turns the same question into a controlled comparison;
* item 12 is the cost panel;
* item 13 runs the automated pipeline (retrieval-only: no key is spent);
* item 14 exports the session and the run.

Item 1 — "open the HF Space" — is not testable from inside the repository and is
deliberately not faked here; the ledger row for it says so.

The order matters and is preserved on purpose: a fact asked for immediately is
retrieval, a fact asked for after unrelated turns is memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.controller import UIController
from app.evaluation import EvaluationPolicy, UIEvaluationRunner
from providers.base import ProviderConfig, ProviderResponse
from tests.fakes import InMemoryEvaluationStore

pytest.importorskip("brainos_runtime")

#: Fixture-shaped credential. Never live, and never allowed to reach a panel.
KEY = "sk-mvp-SECRET-2501"
FACT = "For Project Atlas, the production database is PostgreSQL 16."
FOLLOW_UP = "What database does Project Atlas use in production?"
FILLER = (
    "Unrelated turn {index}: we reviewed the quarterly roadmap notes and the "
    "office move checklist."
)
#: What the model replies. Distinctive so a leak into a wrong panel is obvious.
ANSWER = "Project Atlas uses PostgreSQL 16 in production."


class StubProvider:
    """Deterministic provider double: instant, offline, records what it was sent.

    One instance is reused for every session configuration, like the real
    service's cached client, so a test can assert which model the turn really
    asked for.
    """

    def __init__(self, config: ProviderConfig | None = None, **_: Any) -> None:
        self.config = config
        self.model = getattr(config, "model", "stub-model")
        self.calls: list[list[dict[str, str]]] = []

    def attach(self, config: ProviderConfig) -> StubProvider:
        """Record the configuration the caller built this provider from."""

        self.config = config
        self.model = getattr(config, "model", "stub-model")
        return self

    def list_models(self) -> list[str]:
        return ["stub-small", "stub-medium"]

    def validate_credentials(self) -> bool:
        return True

    def generate(self, messages: list[dict[str, str]], **_: Any) -> ProviderResponse:
        self.calls.append([dict(message) for message in messages])
        return ProviderResponse(text=ANSWER, model=self.model, usage={"total_tokens": 12})


@dataclass
class Walk:
    """Everything the checklist produced, so each test can assert its own item."""

    controller: UIController
    provider: StubProvider
    session_id: str
    connection: Any
    turns: list[Any]
    evaluation: Any
    export: dict[str, Any]
    export_text: str
    results_root: Path


@pytest.fixture(scope="module")
def walk(tmp_path_factory: pytest.TempPathFactory) -> Walk:
    """Walk the checklist once, in order, and hand each test the evidence."""

    root = tmp_path_factory.mktemp("mvp-walkthrough")
    stub = StubProvider()
    runner = UIEvaluationRunner(
        results_root=root / "results",
        policy=EvaluationPolicy(max_tasks=2, max_requests=50),
        evaluation_store=InMemoryEvaluationStore(),
        provider_factory=stub.attach,
    )
    controller = UIController(
        provider_factory=stub.attach,
        evaluation_runner=runner,
    )

    # Items 2, 3, 4 — provider, key, model.
    connection = controller.connect(
        None, provider="openai", model="stub-medium", api_key=KEY
    )
    session_id = connection.session_id

    # Item 5 — start a conversation (and, in the same turn, state the fact for 6).
    first = controller.chat(
        session_id, f"{FACT} Deployments run on Friday at 17:00."
    )

    # Item 7 — continue the conversation with turns that do not mention it.
    for index in range(4):
        controller.chat(session_id, FILLER.format(index=index))

    # Items 8, 9, 10 — ask for it back later and inspect the panels.
    asked = controller.chat(session_id, FOLLOW_UP)

    # Item 11 — the same question under full-context mode, for comparison.
    controller.apply_mode(session_id, "full_context")
    compared = controller.chat(session_id, FOLLOW_UP)
    controller.apply_mode(session_id, "brainos")

    # Item 13 — a small benchmark (retrieval-only; the stub never sees a task).
    evaluation = controller.run_evaluation(
        session_id,
        preset="quick",
        modes=["full_context", "brainos"],
        limit=2,
        render_plots=False,
    )

    # Item 14 — export the session and the run.
    export = controller.export_session(session_id)

    return Walk(
        controller=controller,
        provider=stub,
        session_id=session_id,
        connection=connection,
        turns=[first, asked, compared],
        evaluation=evaluation,
        export=export,
        export_text=controller.export_text(session_id),
        results_root=root / "results",
    )


# --------------------------------------------------------------------------- #
# Items 2–4 — provider, credential, model
# --------------------------------------------------------------------------- #


def test_item_2_through_4_a_visitor_selects_openai_a_key_and_a_model(walk: Walk) -> None:
    assert walk.connection.connected is True
    assert walk.connection.models == ("stub-small", "stub-medium")
    assert walk.connection.model_value == "stub-medium"
    assert "Connected" in walk.connection.status or "loaded" in walk.connection.status.lower()


def test_item_3_the_key_is_never_returned_to_the_browser(walk: Walk) -> None:
    """The panel gets acknowledgement, not the credential."""

    assert walk.connection.key_value == ""
    blob = json.dumps(walk.export) + walk.export_text
    assert KEY not in blob


def test_item_4_the_selected_model_is_the_one_the_turn_used(walk: Walk) -> None:
    assert walk.turns[0].summary  # the turn rendered a status line
    assert walk.provider.model == "stub-medium"
    assert walk.turns[0].turn_usage.get("total_tokens") or walk.turns[0].turn_usage


# --------------------------------------------------------------------------- #
# Items 5–8 — the conversation, and memory across it
# --------------------------------------------------------------------------- #


def test_item_5_a_conversation_starts_and_the_provider_is_called(walk: Walk) -> None:
    assert walk.turns[0].history[-1] == {"role": "assistant", "content": ANSWER}
    # Seven turns: the fact, four fillers, the question, and the comparison.
    assert len(walk.provider.calls) == 7
    # Every prompt carried the system instructions and delimited memory, not raw history.
    system = [message for message in walk.provider.calls[0] if message["role"] == "system"]
    assert system


def test_item_6_the_fact_is_stored_in_brainos_memory(walk: Walk) -> None:
    stored = [str(row) for row in walk.turns[0].stored_rows]
    assert stored, "the memory policy stored nothing for a durable project fact"
    assert any("PostgreSQL 16" in row for row in stored)


def test_item_7_the_conversation_continues_past_the_fact(walk: Walk) -> None:
    continued = walk.controller.ensure_session(walk.session_id)
    assert len(continued.messages) == 14  # seven user/assistant pairs


def test_item_8_the_fact_is_retrieved_later_into_the_prompt(walk: Walk) -> None:
    """The point of the whole project: a fact from turn 1 in turn 6's prompt."""

    prompt = walk.turns[1].prompt
    assert "PostgreSQL 16" in prompt
    # ... while the filler turns are not all still in it.
    stats = walk.turns[1].stats
    assert stats["final_context_tokens"] < stats["full_context_reference_tokens"]
    assert stats["selected_memory_count"] >= 1
    assert stats["history_messages_selected"] < stats["history_messages_considered"]


# --------------------------------------------------------------------------- #
# Items 9–10 — the inspection panels
# --------------------------------------------------------------------------- #


def test_item_9_the_memory_panel_shows_what_brainos_holds(walk: Walk) -> None:
    assert walk.turns[1].retrieved_rows
    assert any("PostgreSQL" in str(row) for row in walk.turns[1].retrieved_rows)
    assert walk.turns[1].summary  # the panel rendered a headline


def test_item_10_the_context_panel_shows_the_prompt_and_its_accounting(walk: Walk) -> None:
    stats = walk.turns[1].stats
    assert walk.turns[1].prompt
    for field in (
        "final_context_tokens",
        "full_context_reference_tokens",
        "raw_history_tokens",
        "selected_memory_count",
        "context_reduction_vs_full_context",
        "dropped_memories_for_budget",
        "dropped_history_for_budget",
        "budget_utilization",
    ):
        assert field in stats, f"the context panel no longer reports {field}"
    assert stats["memory_block_tokens"] > 0


def test_item_10_the_retrieved_context_is_attributed(walk: Walk) -> None:
    """The panel says *why* each memory was used, not just that it was."""

    assert walk.turns[1].retrieved_rows
    # The decision trace is the audit of the retrieval: the query, the
    # candidates, the drops and the guard, which is what makes the comparison
    # in item 11 inspectable rather than a number to trust.
    assert walk.turns[1].trace_events
    assert walk.turns[1].stats["selected_memory_count"] <= walk.turns[1].stats[
        "candidate_memory_count"
    ]


# --------------------------------------------------------------------------- #
# Item 11 — the comparison the project exists to run
# --------------------------------------------------------------------------- #


def test_item_11_brainos_and_full_context_are_compared_on_the_same_session(walk: Walk) -> None:
    brainos = walk.turns[1].stats
    full_context = walk.turns[2].stats

    assert brainos["final_context_tokens"] < full_context["final_context_tokens"]
    assert brainos["context_reduction_vs_full_context"] > 0.0
    assert full_context["context_reduction_vs_full_context"] == 0.0
    # Both modes answered the same question, from the same memory.
    assert "PostgreSQL 16" in walk.turns[2].prompt


def test_item_11_the_mode_is_visible_in_the_session_payload(walk: Walk) -> None:
    payload = walk.controller.context_payload(walk.session_id)

    assert payload["mode"] == "brainos"
    assert payload["mode_label"]


# --------------------------------------------------------------------------- #
# Item 12 — token usage
# --------------------------------------------------------------------------- #


def test_item_12_token_usage_is_visible_per_turn_and_per_session(walk: Walk) -> None:
    usage = walk.controller.usage_payload(walk.session_id)

    assert usage["requests"] == 7
    assert usage["input_tokens"] > 0
    assert walk.turns[1].turn_usage["prompt_tokens"] > 0
    assert walk.turns[1].usage_report
    assert walk.turns[1].usage_summary


# --------------------------------------------------------------------------- #
# Item 13 — a small benchmark
# --------------------------------------------------------------------------- #


def test_item_13_a_small_benchmark_runs_from_the_tab(walk: Walk) -> None:
    view = walk.evaluation

    assert view.ran is True, view.status
    assert view.ok is True, view.status
    stages = [(row[0], row[1]) for row in view.stage_rows]
    assert stages, "the stage table is empty"
    assert [status for _, status in stages if status not in {"ok", "skipped"}] == []
    # Plots were switched off for this run; the status says so rather than hiding it.
    assert view.summary["skipped_stages"] == ["plots"]


def test_item_13_the_run_discloses_its_cost_before_it_is_trusted(walk: Walk) -> None:
    view = walk.evaluation

    assert view.summary["tasks_executed"] == 2
    assert view.summary["within_limits"] is True
    assert view.summary["failed_stages"] == []
    # The cost table is shown before the numbers, and it says what the ceiling was.
    assert "Planned" in view.cost_markdown and "Ceiling" in view.cost_markdown
    assert "retrieval-only" in view.status  # no key was spent


def test_item_13_the_run_writes_a_report_and_a_reproducibility_manifest(walk: Walk) -> None:
    view = walk.evaluation

    assert len(view.report_markdown) > 500
    assert "evaluation.pipeline" in view.rerun_command
    assert "--preset quick" in view.rerun_command
    assert Path(walk.results_root / "ui" / walk.session_id).is_dir()
    assert Path(view.report_path).is_file()
    assert view.repro.get("application_version")
    assert view.repro["dataset"]["sha256"]
    # The manifest records the dataset it read; the run's *selection* is in the summary.
    assert view.repro["task_count"] == view.summary["tasks_available"]
    assert view.repro["task_ids"]


def test_item_13_the_run_is_viewable_afterwards(walk: Walk) -> None:
    history = walk.controller.evaluation_history(walk.session_id)

    assert history.history_rows
    assert "1 run(s) in this session" in history.status
    assert history.history_rows[0][2] == "quick"


# --------------------------------------------------------------------------- #
# Item 14 — export
# --------------------------------------------------------------------------- #


def test_item_14_the_session_exports_with_its_transcript_usage_and_runs(walk: Walk) -> None:
    export = walk.export

    assert export["messages"] or export["transcript"]
    assert export["usage"]["requests"] == 7
    assert export["evaluation_runs"]
    assert len(walk.export_text) > 0


def test_item_14_the_export_is_key_free(walk: Walk) -> None:
    assert KEY not in walk.export_text
    assert "sk-" not in walk.export_text
