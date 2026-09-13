"""Tests for the rendering helpers behind the inspection panels.

The panels are the only place application data becomes browser-visible text, so
these tests pin both the shape (column counts the UI declares) and the safety
properties (no unbounded text, no raw ``None`` rendered as a value).
"""

from app.panels import (
    CONFLICT_COLUMNS,
    DROPPED_MEMORY_COLUMNS,
    RETRIEVED_MEMORY_COLUMNS,
    STORED_MEMORY_COLUMNS,
    clip,
    conflict_rows,
    context_summary,
    dropped_rows,
    export_payload,
    format_int,
    format_percent,
    memory_rows,
    render_prompt,
    retrieved_rows,
    round_value,
    short_timestamp,
    status_line,
    token_breakdown,
    trace_markdown,
)
from brain.adapter import MemoryRecord


def _record(
    memory_id: str = "m1",
    text: str = "The production database is PostgreSQL 16.",
    **kwargs: object,
) -> MemoryRecord:
    return MemoryRecord(memory_id=memory_id, text=text, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Scalar formatting
# --------------------------------------------------------------------------- #


def test_short_timestamp_trims_iso_values() -> None:
    assert short_timestamp("2026-09-13T16:18:45.719593+00:00") == "2026-09-13 16:18:45"
    assert short_timestamp("") == ""
    assert short_timestamp(None) == ""


def test_clip_bounds_and_flattens_text() -> None:
    assert clip("line one\nline two") == "line one line two"
    assert len(clip("x" * 500)) == 240
    assert clip(None) == ""


def test_round_value_renders_missing_signals_as_a_dash() -> None:
    assert round_value(0.123456) == 0.123
    assert round_value(None) == "—"
    assert round_value("nonsense") == "—"


def test_number_helpers_degrade_gracefully() -> None:
    assert format_int("12") == 12
    assert format_int(None) == 0
    assert format_percent(0.72185) == "72.2%"
    assert format_percent(None) == "—"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

_MEMORY_REPORT = {
    "query": "what database?",
    "candidate_count": 3,
    "selected_count": 1,
    "dropped": [
        {
            "memory_id": "m2",
            "reason": "low_relevance",
            "text": "unrelated memory",
            "detail": "relevance=0.02 below floor=0.12",
            "score": 0.02,
        },
        {"memory_id": "m3", "reason": "duplicate", "text": "", "detail": "", "score": 0.0},
    ],
    "conflicts": [
        {
            "subject": "database",
            "kept_memory_id": "m2",
            "dropped_memory_id": "m1",
            "source": "runtime",
            "reason": "newer observation",
        }
    ],
    "ranking": [
        {
            "memory_id": "m1",
            "score": 0.81,
            "relevance": 0.7,
            "lexical": 0.9,
            "runtime": 0.5,
            "recency": 1.0,
            "rank": 1,
            "contested": False,
            "suspicious": False,
            "merged_ids": ["m9"],
        }
    ],
}


def test_memory_rows_match_the_declared_columns() -> None:
    rows = memory_rows(
        [_record("m1", "fact one", memory_type="FACT", source_turn=3, retrieval_count=2)]
    )
    assert len(rows) == 1
    assert len(rows[0]) == len(STORED_MEMORY_COLUMNS)
    assert rows[0][0] == "FACT"
    assert rows[0][2] == 3
    assert rows[0][3] == 2


def test_memory_rows_render_unknown_values_without_crashing() -> None:
    row = memory_rows([_record("m1")])[0]
    assert row[2] == "—"  # source_turn
    assert row[4] == "—"  # confidence
    assert row[6] == "active"


def test_retrieved_rows_show_only_what_reached_the_prompt() -> None:
    records = [_record("m1", "PostgreSQL 16"), _record("m2", "unscored")]
    rows = retrieved_rows(records, _MEMORY_REPORT["ranking"])

    # The ranking is post-budget: m2 never reached the prompt, so it belongs to
    # the audit table and must not appear here as if the model saw it.
    assert len(rows) == 1
    assert len(rows[0]) == len(RETRIEVED_MEMORY_COLUMNS)
    assert rows[0][0] == 1
    assert rows[0][1] == "PostgreSQL 16"
    assert rows[0][3] == 0.81
    assert "merged×2" in rows[0][8]


def test_retrieved_rows_fall_back_to_records_when_nothing_was_scored() -> None:
    rows = retrieved_rows([_record("m1", "one"), _record("m2", "two")], [])

    assert [row[1] for row in rows] == ["one", "two"]
    assert [row[0] for row in rows] == [1, 2]
    assert rows[0][3] == "—"


def test_retrieved_rows_surface_contested_and_suspicious_flags() -> None:
    ranking = [
        {
            "memory_id": "m1",
            "score": 0.4,
            "rank": 1,
            "contested": True,
            "suspicious": True,
            "merged_ids": [],
        }
    ]
    flags = retrieved_rows([_record("m1")], ranking)[0][8]
    assert "contested" in flags
    assert "suspicious" in flags


def test_dropped_and_conflict_rows_follow_the_audit_records() -> None:
    dropped = dropped_rows(_MEMORY_REPORT)
    assert len(dropped) == 2
    assert len(dropped[0]) == len(DROPPED_MEMORY_COLUMNS)
    assert dropped[0][0] == "low_relevance"
    assert dropped[0][2] == 0.02

    conflicts = conflict_rows(_MEMORY_REPORT)
    assert len(conflicts[0]) == len(CONFLICT_COLUMNS)
    assert conflicts[0][1] == "m2"
    assert conflicts[0][2] == "m1"


def test_tables_tolerate_missing_sections() -> None:
    assert dropped_rows({}) == []
    assert conflict_rows({}) == []
    assert dropped_rows({"dropped": ["not-a-mapping"]}) == []


# --------------------------------------------------------------------------- #
# Prompt and accounting
# --------------------------------------------------------------------------- #


def test_render_prompt_shows_roles_in_order() -> None:
    text = render_prompt(
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
        ]
    )
    assert text == "SYSTEM\nbe helpful\n\nUSER\nhello"


def test_context_summary_reports_the_baseline_next_to_the_savings() -> None:
    stats = {
        "final_context_tokens": 387,
        "full_context_reference_tokens": 1391,
        "raw_history_tokens": 1200,
        "max_tokens": 4096,
        "context_reduction_vs_full_context": 0.7218,
        "token_savings_vs_full_context": 1004,
        "budget_utilization": 0.0944,
        "system_tokens": 42,
        "memory_block_tokens": 118,
        "recent_history_tokens": 205,
        "current_message_tokens": 14,
        "overhead_tokens": 8,
        "history_messages_considered": 60,
        "history_messages_selected": 4,
        "candidate_memory_count": 6,
        "selected_memory_count": 2,
        "dropped_history_for_budget": 56,
        "token_counter": "estimate_tokens",
    }
    summary = context_summary(stats, report=_MEMORY_REPORT, mode="brainos", turn=12)
    assert "**387 tokens**" in summary
    assert "1391 tokens" in summary
    assert "72.2%" in summary
    assert "1,004" not in summary  # raw ints are not thousands-separated
    assert "1004" in summary
    assert "**4** of 60" in summary
    assert "56 dropped for budget" in summary
    assert "**2** of 6" in summary
    assert "low_relevance 1" in summary
    assert "duplicate 1" in summary
    assert token_breakdown(stats) in summary


def test_context_summary_prefers_the_reported_conflict_count() -> None:
    report = {"conflicts": [], "conflict_count": 3}
    assert "conflicts: 3 pair(s) resolved" in context_summary({}, report=report)


def test_context_summary_counts_conflicts_when_no_counter_is_present() -> None:
    assert "conflicts: 1 pair(s) resolved" in context_summary({}, report=_MEMORY_REPORT)


def test_context_summary_flags_truncation_and_overruns() -> None:
    stats = {"system_truncated": True, "exceeds_max_tokens": True}
    summary = context_summary(stats)
    assert "system prompt was truncated" in summary
    assert "exceeds `max_tokens`" in summary


def test_empty_accounting_still_renders() -> None:
    summary = context_summary({})
    assert "Sent to model" in summary
    assert "0 tokens" in summary


# --------------------------------------------------------------------------- #
# Trace and status
# --------------------------------------------------------------------------- #


def test_trace_markdown_follows_the_planned_stage_order() -> None:
    text = trace_markdown(
        query="What database does Project Atlas use?",
        stats={"current_message_tokens": 10, "final_context_tokens": 165},
        report=_MEMORY_REPORT,
        events=[{"name": "recall", "detail": "returned=3"}],
        stored=[_record("m1", memory_type="FACT")],
        generated=True,
        model="gpt-4o-mini",
    )
    stages = (
        "Query",
        "Observe",
        "Recall",
        "Relevance filtering",
        "Context construction",
        "LLM",
        "Observation",
    )
    for index, stage in enumerate(stages, start=1):
        assert f"{index}. **{stage}**" in text
    assert "generated with `gpt-4o-mini`" in text
    assert "low_relevance 1" in text
    assert "`recall` — returned=3" in text
    assert "**Conflicts**" in text


def test_trace_markdown_marks_the_unconfigured_and_failed_paths() -> None:
    unconfigured = trace_markdown(query="hi", stats={}, report={}, generated=False)
    assert "not called — no API key configured" in unconfigured

    failed = trace_markdown(
        query="hi", stats={}, report={}, error="401 unauthorized", generated=False
    )
    assert "failed: 401 unauthorized" in failed


def test_trace_markdown_truncates_long_runtime_traces() -> None:
    events = [{"name": f"event-{index}", "detail": "x"} for index in range(60)]
    text = trace_markdown(query="hi", stats={}, report={}, events=events)
    assert "event-0" in text
    assert "event-39" in text
    assert "event-40" not in text
    assert "20 earlier event(s) omitted" in text


def test_status_line_distinguishes_missing_key_from_silent_turn() -> None:
    assert "no API key configured" in status_line(generated=False, configured=False)
    assert "provider returned no reply" in status_line(generated=False, configured=True)
    # A refresh renders no turn, so it must not claim the model was skipped.
    assert "no model called" not in status_line(generated=False, has_turn=False)
    assert status_line(generated=True, error="boom") == "⚠️ boom"


def test_status_line_reports_tokens_and_reduction() -> None:
    line = status_line(
        turn=7,
        mode="brainos",
        model="gpt-4o-mini",
        generated=True,
        stats={
            "final_context_tokens": 387,
            "context_reduction_vs_full_context": 0.7218,
        },
    )
    assert "Turn 7" in line
    assert "387 tokens sent" in line
    assert "72.2% below baseline" in line


def test_export_payload_produces_sorted_indented_json() -> None:
    text = export_payload({"b": 1, "a": {"z": 2}})
    assert text.startswith("{\n")
    assert text.index('"a"') < text.index('"b"')
