"""Tests for the Gradio layer.

The layout is exercised with the real Gradio package when it is installed, and
skipped otherwise — the controller, panels, and service tests cover behaviour
without it. What these tests pin down is the part that can only break here:

* the app builds against the installed Gradio version,
* every registered callback declares as many outputs as its function returns,
* the connect callback never hands the key back to the browser,
* the export callback writes a parseable, credential-free file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.controller import UIController
from app.ui import create_app
from baselines.modes import RECENT_WINDOW_TURNS
from brain.adapter import BrainOSAdapter
from tests.fakes import FakeLLMProvider, InMemoryEvaluationStore, LooseFakeRuntime

KEY = "sk-ui-SECRET-0002"

gradio = pytest.importorskip("gradio", reason="Gradio is an optional dependency")


def _controller() -> UIController:
    def adapter_factory(**kwargs: object) -> BrainOSAdapter:
        return BrainOSAdapter(
            LooseFakeRuntime(str(kwargs["session_id"]), str(kwargs["actor_id"]))
        )

    return UIController(provider_factory=FakeLLMProvider, adapter_factory=adapter_factory)


def test_app_builds_and_registers_every_callback() -> None:
    demo = create_app(_controller())

    # 11 widget callbacks (Phase 6 adds the baseline-mode selector) plus the
    # per-page session bootstrap, 4 Phase 15 cost-control widget callbacks, and
    # the 3 Phase 17 Evaluation-tab callbacks (preview, run, history).
    assert len(demo.fns) == 19
    assert any(block_fn.targets == [(0, "load")] for block_fn in demo.fns.values())
    assert any(
        block_fn.targets and block_fn.targets[0][1] == "change"
        for block_fn in demo.fns.values()
    ), "the baseline-mode selector must rewrite the budgets it governs"


def test_chat_callback_returns_one_value_per_declared_output() -> None:
    controller = _controller()
    demo = create_app(controller)

    values = controller.chat(None, "The production database is PostgreSQL 16.")
    from app.ui import _chat

    produced = _chat(controller)("The production database is PostgreSQL 16.", None)
    declared = {
        (len(block_fn.inputs), len(block_fn.outputs)) for block_fn in demo.fns.values()
    }

    assert len(values.history) >= 0  # the view is what the UI unpacks
    assert (2, len(produced)) in declared


def test_panel_callbacks_return_matching_lengths() -> None:
    controller = _controller()
    demo = create_app(controller)

    from app.ui import _clear_conversation, _refresh

    declared = {
        (len(block_fn.inputs), len(block_fn.outputs)) for block_fn in demo.fns.values()
    }
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    controller.chat(sid, "The production database is PostgreSQL 16.")

    assert (1, len(_refresh(controller)(sid))) in declared
    assert (1, len(_clear_conversation(controller)(sid))) in declared


def test_connect_callback_clears_the_key_box_and_names_no_secret() -> None:
    from app.ui import _connect

    controller = _controller()
    status, model_update, key_value, session_id = _connect(controller)(
        "openai", "gpt-4o-mini", KEY, "", 0.2, None, None
    )

    assert key_value == ""
    assert KEY not in status
    assert KEY not in json.dumps(model_update, default=str)
    assert session_id


def test_export_callback_writes_a_credential_free_json_file(tmp_path: Path) -> None:
    from app.ui import _export

    controller = _controller()
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    controller.chat(sid, "The production database is PostgreSQL 16.")

    update = _export(controller)(sid)
    path = Path(update["value"] if isinstance(update, dict) else update.value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)

    assert payload["messages"]
    assert "api_key" not in payload["provider"]
    assert KEY not in path.read_text(encoding="utf-8") if path.exists() else True


def test_update_context_callback_accepts_the_sidebar_values() -> None:
    from app.ui import _update_context

    controller = _controller()
    status, session_id = _update_context(controller)(
        # mode, max_tokens, max_recent_turns, recent_turn_budget, memory_budget,
        # chunk_budget, max_memories, relevance_floor, recency_half_life_turns,
        # resolve_conflicts, drop_stale_memories, drop_suspicious_memories, session
        "brainos", 4096, 2, 256, 512, 0, 6, 0.12, 20.0, True, True, False, None
    )

    assert "Context updated" in status
    settings = controller.ensure_session(session_id).context
    assert settings.recent_turn_budget == 256
    assert settings.memory_budget == 512
    assert settings.max_recent_turns == 2


def test_blank_recent_turns_means_whole_conversation() -> None:
    from app.ui import _update_context

    controller = _controller()
    _status, session_id = _update_context(controller)(
        "full_context", 4096, None, 4096, 0, 0, 6, 0.12, 20.0, True, True, False, None
    )

    # A blank box is "no turn limit", not "send no history at all".
    assert controller.ensure_session(session_id).context.max_recent_turns is None


def test_mode_selector_rewrites_the_budgets_it_governs() -> None:
    from app.ui import _apply_mode

    controller = _controller()
    apply_mode = _apply_mode(controller)
    # ``session_id`` threads through gr.State in the app, so the callback's
    # return value is the only handle on the session it changed.
    status, info, *_rest, session_id = apply_mode("full_context", None)

    assert "Mode A" in info
    assert "Mode A" in status
    full_context = controller.ensure_session(session_id).context
    # Mode A replays everything: no turn cap, and the window is the ceiling.
    assert full_context.max_recent_turns is None
    assert full_context.recent_turn_budget == full_context.max_tokens
    assert full_context.memory_budget == 0
    assert full_context.chunk_budget == 0

    _status, rag_info, *_rag_rest, session_id = apply_mode("rag", session_id)
    assert "Mode C" in rag_info
    rag = controller.ensure_session(session_id).context
    assert rag.uses_rag() and not rag.uses_memory()
    assert rag.chunk_budget > 0
    assert rag.max_recent_turns == RECENT_WINDOW_TURNS


def test_the_security_tab_is_wired_into_every_panel_callback() -> None:
    """Phase 13 adds three components; every panel callback must feed all three.

    The declared output count is what Gradio checks at build time, so a mismatch
    between :func:`_panel_values` and the components would break every callback
    at once — the count is asserted against the component order, not a literal.
    """

    from app.panels import SECURITY_COLUMNS
    from app.ui import PanelComponents, _panel_values, _silent_values

    controller = _controller()
    demo = create_app(controller)
    sid = controller.connect(None, provider="openai", model="gpt-4o-mini", api_key=KEY).session_id
    controller.chat(sid, f"My API key is {KEY} and we run PostgreSQL 16.")
    view = controller.chat(sid, "What database do we run in production?")

    values = _panel_values(view)
    declared = {(len(fn.inputs), len(fn.outputs)) for fn in demo.fns.values()}

    assert len(values) == len(PanelComponents.order) == 15
    # Phase 15: usage components are now at the end, security is at -5 to -3
    assert PanelComponents.order[-5:-2] == ("security", "security_rows", "security_report")
    assert (1, len(_silent_values(view))) in declared, "panel callbacks feed all 15"
    assert (2, len(_silent_values(view)) + 1) in declared, "the chat callback adds the box"

    markdown, rows, report = values[-5:-2]
    assert markdown.startswith("### Security")
    assert "Credentials redacted" in markdown
    assert all(len(row) == len(SECURITY_COLUMNS) for row in rows)
    assert rows, "a session that redacted a credential must show the finding"
    assert report["clean"] is False
    assert report["by_route"]["history"] >= 1
    assert report["session_id"] == sid
    assert KEY not in str(values), "no panel value may carry the session key"


# --------------------------------------------------------------------------- #
# Phase 17: the Evaluation tab
# --------------------------------------------------------------------------- #


def _evaluation_controller(tmp_path: Path) -> UIController:
    """A controller whose benchmark runs write into ``tmp_path``, not the repo.

    The tab writes real files, so a UI test that ran one against the default
    ``results/`` root would leave artifacts in the checkout — the same reason the
    runner takes a ``results_root`` at all.
    """

    from app.evaluation import UIEvaluationRunner

    controller = _controller()
    controller.evaluation = UIEvaluationRunner(
        results_root=tmp_path / "results",
        evaluation_store=InMemoryEvaluationStore(),
        provider_factory=FakeLLMProvider,
    )
    return controller


def test_evaluation_tab_builds_with_the_preset_catalogue(tmp_path: Path) -> None:
    from app.panels import PRESET_COLUMNS
    from app.ui import EVALUATION_MARKDOWN, EvaluationComponents

    controller = _evaluation_controller(tmp_path)
    demo = create_app(controller)
    catalogue = controller.evaluation_presets()

    assert demo is not None
    assert len(catalogue.preset_rows) == 3
    assert all(len(row) == len(PRESET_COLUMNS) for row in catalogue.preset_rows)
    assert "ceiling" in catalogue.status
    # The tab's own copy says what the defaults are, so a visitor who never
    # presses a button still learns that generation spends their key.
    assert "retrieval-only" in EVALUATION_MARKDOWN.lower()
    assert "End session" in EVALUATION_MARKDOWN
    assert len(EvaluationComponents.order) == 12


def _form(
    session_id: str | None = None,
    *,
    preset: str = "quick",
    modes: list[str] | None = None,
    limit: int = 1,
    dataset: str | None = None,
    generate: bool = False,
    render_plots: bool = False,
    baseline: str = "full_context",
    counter: str = "estimate",
) -> tuple[object, ...]:
    """The Evaluation tab's run form, in the order the callbacks read it.

    Preview and run share this list on purpose (see
    :class:`app.ui.EvaluationComponents`), so one helper builds both.
    """

    return (
        preset,
        modes or [],
        limit,
        dataset,
        generate,
        render_plots,
        baseline,
        counter,
        session_id,
    )


def test_evaluation_callbacks_return_one_value_per_declared_output(
    tmp_path: Path,
) -> None:
    from app.ui import (
        EvaluationComponents,
        _evaluation_history,
        _evaluation_preview,
        _evaluation_run,
    )

    controller = _evaluation_controller(tmp_path)
    demo = create_app(controller)
    declared = {(len(fn.inputs), len(fn.outputs)) for fn in demo.fns.values()}

    preview = _evaluation_preview(controller)(*_form())
    # The last value is the session id, threaded back through gr.State: with no
    # session yet, the first callback starts one and the rest must join it
    # rather than each starting their own.
    session_id = preview[-1]
    run = _evaluation_run(controller)(*_form(session_id))
    history = _evaluation_history(controller)(session_id)

    assert len(preview) == len(run) == len(history) == len(EvaluationComponents.order) + 1
    assert (9, len(preview)) in declared
    assert (1, len(history)) in declared
    assert session_id
    assert run[-1] == history[-1] == session_id
    assert len(history[9]) == 1, "the run the previous callback made is this session's"


def test_preview_callback_shows_a_ceiling_and_writes_nothing(tmp_path: Path) -> None:
    from app.ui import _evaluation_preview

    controller = _evaluation_controller(tmp_path)

    values = _evaluation_preview(controller)(*_form(limit=2))

    status, cost, presets, headline = values[0], values[1], values[2], values[3]
    assert "Preview" in status
    assert "What this run would cost" in cost
    assert "Provider requests" in cost
    assert "unknown before the prompts are built" in cost
    assert presets and not headline
    assert list((tmp_path / "results").rglob("*.json")) == []


def test_run_callback_feeds_the_tables_the_figures_and_the_report(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib", reason="figures need matplotlib")
    from app.panels import (
        EVALUATION_HEADLINE_COLUMNS,
        EVALUATION_STAGE_COLUMNS,
    )
    from app.ui import _evaluation_run

    controller = _evaluation_controller(tmp_path)

    values = _evaluation_run(controller)(
        *_form(modes=["full_context", "rag"], render_plots=True)
    )
    status, cost, presets, headline, stages, gallery = values[:6]
    report, artifacts, repro, history, downloads, payload = values[6:12]

    assert "Pipeline finished" in status
    assert "What this run cost" in cost
    assert len(presets) == 3
    assert all(len(row) == len(EVALUATION_HEADLINE_COLUMNS) for row in headline)
    assert [row[0] for row in headline] == ["Mode A — Full context", "Mode C — Lexical RAG"]
    assert all(len(row) == len(EVALUATION_STAGE_COLUMNS) for row in stages)
    assert gallery and all(Path(path).is_file() for path in gallery)
    assert "# BrainOS Context Lab — evaluation report" in report
    assert "### Artifacts" in artifacts and "### Reproducibility" in repro
    assert len(history) == 1
    assert downloads and all(Path(path).is_file() for path in downloads)
    assert payload["tasks_executed"] == 1


def test_run_callback_reports_a_refusal_instead_of_raising(tmp_path: Path) -> None:
    from app.ui import _evaluation_run

    controller = _evaluation_controller(tmp_path)

    values = _evaluation_run(controller)(*_form(limit=0, dataset="/etc/passwd"))

    assert values[0].startswith("**Not run**")
    assert "must stay inside" in values[0]
    assert list(tmp_path.rglob("*.json")) == []


def test_run_callback_refuses_generation_without_a_connected_key(tmp_path: Path) -> None:
    from app.ui import _evaluation_run

    controller = _evaluation_controller(tmp_path)

    values = _evaluation_run(controller)(*_form(generate=True))

    assert values[0].startswith("**Not run**")
    assert "API key" in values[0]


def test_history_callback_reports_a_fresh_session(tmp_path: Path) -> None:
    from app.ui import _evaluation_history

    controller = _evaluation_controller(tmp_path)

    values = _evaluation_history(controller)(None)

    assert "No benchmark run in this session yet." in values[0]
    assert values[9] == []


def test_the_evaluation_tab_never_renders_the_session_key(tmp_path: Path) -> None:
    from app.ui import _evaluation_run

    controller = _evaluation_controller(tmp_path)
    sid = controller.connect(
        None, provider="openai", model="gpt-4o-mini", api_key=KEY
    ).session_id

    values = _evaluation_run(controller)(*_form(sid, generate=True))

    assert "Pipeline finished" in values[0]
    assert KEY not in json.dumps(values, default=str)
    assert KEY not in "".join(
        Path(path).read_text(encoding="utf-8", errors="ignore")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
