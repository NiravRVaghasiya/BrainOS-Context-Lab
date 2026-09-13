"""Rendering helpers for the inspection panels of the chat UI.

Everything in this module is a pure function from application data to
browser-safe values. It deliberately imports no web framework, so:

* the panels can be unit-tested without Gradio installed, and
* the UI cannot accidentally widen what it renders — every value that reaches
  the browser is produced here, from the already-sanitized data the service
  returns, and is redacted once more against the active session credential.

Two rules govern the formatting:

1. **No credential ever reaches a cell.** Numeric columns are rounded, memory
   text is bounded, and trace details are redacted by the caller.
2. **Numbers stay interpretable.** The token accounting only means something if
   the panel shows what it is compared against, so every context summary
   reports the full-context reference and the reduction next to the raw count.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from brain.adapter import MemoryRecord

#: Column layouts for the memory tables. Kept module-level so the UI and the
#: tests agree on the schema.
STORED_MEMORY_COLUMNS: tuple[str, ...] = (
    "Type",
    "Memory",
    "Turn",
    "Retrieved",
    "Confidence",
    "Observed",
    "Status",
)
#: ``Rank`` … ``Flags`` describe a memory that survived filtering and reached
#: the prompt; the same columns are meaningless for a dropped memory, which is
#: why the audit table carries a reason instead.
RETRIEVED_MEMORY_COLUMNS: tuple[str, ...] = (
    "Rank",
    "Memory",
    "Type",
    "Score",
    "Relevance",
    "Lexical",
    "Runtime",
    "Recency",
    "Flags",
)
DROPPED_MEMORY_COLUMNS: tuple[str, ...] = ("Reason", "Memory", "Score", "Detail")
#: Phase 6: the transcript chunks a non-BrainOS retriever put in the prompt
#: (baseline Modes C/E). Shown separately from memory because they are a
#: different kind of evidence — verbatim conversation, no supersession
#: resolution — and the comparison is the point of the panel.
CHUNK_COLUMNS: tuple[str, ...] = ("Rank", "Chunk", "Said by", "Turn", "Score", "Flags")

_MAX_CELL_CHARS = 240
_MAX_PROMPT_CHARS = 20000
_MAX_TRACE_EVENTS = 40


def short_timestamp(value: Any) -> str:
    """Render an ISO timestamp for a table cell, or ``""`` when unknown."""

    text = str(value or "").strip()
    if len(text) >= 19:
        return text[:19].replace("T", " ")
    return text


def clip(text: Any, limit: int = _MAX_CELL_CHARS) -> str:
    """Bound a string for table/panel rendering."""

    value = str(text or "").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def round_value(value: Any, digits: int = 3) -> float | str:
    """Round a numeric signal for display; ``—`` when the runtime gave none."""

    if value is None or isinstance(value, bool):
        return "—"
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return "—"


def format_int(value: Any) -> int:
    """Coerce a value to an integer for display, defaulting to ``0``."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def format_percent(value: Any, digits: int = 1) -> str:
    """Render a 0–1 ratio as a percentage."""

    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def memory_rows(records: Sequence[MemoryRecord]) -> list[list[Any]]:
    """Rows for the "stored memories" table.

    Mirrors the plan's Memory panel: type, timestamp, retrieval count,
    relevance, and source turn.
    """

    rows: list[list[Any]] = []
    for record in records:
        rows.append(
            [
                str(record.memory_type or "FACT"),
                clip(record.text),
                "—" if record.source_turn is None else int(record.source_turn),
                format_int(record.retrieval_count),
                round_value(record.confidence, 2),
                short_timestamp(record.timestamp),
                str(record.status or "active"),
            ]
        )
    return rows


def retrieved_rows(
    records: Sequence[MemoryRecord],
    ranking: Sequence[Mapping[str, Any]] = (),
) -> list[list[Any]]:
    """Rows for the "memories in this prompt" table.

    The builder's post-budget ``ranking`` drives this table, because the panel
    answers "what did the model actually see, and why was it ranked there".
    Everything BrainOS recalled that did *not* reach the prompt is an audit
    record instead (:func:`dropped_rows`) — showing it in both tables would make
    the panel contradict itself. Records supply the text the ranking does not
    carry; when no ranking exists at all (a caller that did not score), the
    records are listed in recall order so the panel is never silently empty.
    """

    items = [item for item in ranking if isinstance(item, Mapping)]
    by_id = {str(record.memory_id): record for record in records}

    rows: list[list[Any]] = []

    def _row(index: int, item: Mapping[str, Any] | None, record: MemoryRecord | None) -> list[Any]:
        flags: list[str] = []
        if item is not None and item.get("contested"):
            flags.append("contested")
        if item is not None and item.get("suspicious"):
            flags.append("suspicious")
        if item is not None and item.get("merged_ids"):
            flags.append(f"merged×{len(item['merged_ids']) + 1}")
        text = record.text if record is not None else str((item or {}).get("memory_id", ""))
        memory_type = str(record.memory_type or "FACT") if record is not None else "FACT"
        return [
            format_int(item.get("rank")) if item is not None else index,
            clip(text),
            memory_type,
            round_value(item.get("score")) if item is not None else "—",
            round_value(record.relevance) if record is not None else "—",
            round_value(item.get("lexical")) if item is not None else "—",
            round_value(item.get("runtime")) if item is not None else "—",
            round_value(item.get("recency")) if item is not None else "—",
            ", ".join(flags),
        ]

    for index, item in enumerate(items, start=1):
        rows.append(_row(index, item, by_id.get(str(item.get("memory_id", "")))))
    if not items:
        for index, record in enumerate(records, start=1):
            rows.append(_row(index, None, record))
    return rows


def chunk_rows(chunks: Sequence[Any]) -> list[list[Any]]:
    """Rows for the retrieved-transcript table (baseline Modes C/E).

    ``Said by`` is not decoration: a chunk the assistant produced is the model's
    own earlier claim, while one the user produced is user-asserted evidence.
    Presenting them alike would let the panel imply a provenance the prompt does
    not carry.
    """

    rows: list[list[Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        flags = ["suspicious"] if getattr(chunk, "suspicious", False) else []
        rows.append(
            [
                index,
                clip(getattr(chunk, "text", "")),
                str(getattr(chunk, "role", "") or "user"),
                format_int(getattr(chunk, "turn", 0)),
                round_value(getattr(chunk, "score", None)),
                ", ".join(flags),
            ]
        )
    return rows


def dropped_rows(report: Mapping[str, Any]) -> list[list[Any]]:
    """Rows for the per-memory audit trail of the retrieval pipeline.

    These are the same records the evaluation phase maps onto the error
    taxonomy, surfaced to the user as "why this memory did not reach the model".
    """

    dropped = report.get("dropped") if isinstance(report, Mapping) else None
    if not isinstance(dropped, list):
        return []
    rows: list[list[Any]] = []
    for item in dropped:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            [
                str(item.get("reason", "")),
                clip(item.get("text", "")),
                round_value(item.get("score")),
                clip(item.get("detail", ""), 120),
            ]
        )
    return rows


def conflict_rows(report: Mapping[str, Any]) -> list[list[Any]]:
    """Rows for the conflict-resolution table."""

    conflicts = report.get("conflicts") if isinstance(report, Mapping) else None
    if not isinstance(conflicts, list):
        return []
    rows: list[list[Any]] = []
    for item in conflicts:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            [
                clip(item.get("subject", ""), 120),
                str(item.get("kept_memory_id", "") or "kept both"),
                str(item.get("dropped_memory_id", "") or "—"),
                str(item.get("source", "")),
                clip(item.get("reason", ""), 160),
            ]
        )
    return rows


CONFLICT_COLUMNS: tuple[str, ...] = ("Subject", "Kept", "Dropped", "Source", "Reason")


def render_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    """Render the exact message list that was (or would be) sent to the model."""

    blocks: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        content = str(message.get("content", ""))
        blocks.append(f"{role}\n{content}")
    text = "\n\n".join(blocks)
    return text[:_MAX_PROMPT_CHARS]


def token_breakdown(stats: Mapping[str, Any]) -> str:
    """One-line decomposition of the final context cost.

    ``memory`` and ``chunks`` are the two retrieved-evidence sources (Phase 6).
    Both are always shown, including as zeros, so a reader comparing modes sees
    which source a mode actually spent its evidence budget on instead of having
    to notice an absent column.
    """

    parts = (
        ("system", stats.get("system_tokens")),
        ("memory", stats.get("memory_block_tokens")),
        ("chunks", stats.get("chunk_block_tokens")),
        ("history", stats.get("recent_history_tokens")),
        ("question", stats.get("current_message_tokens")),
        ("overhead", stats.get("overhead_tokens")),
    )
    return " · ".join(f"{label} {format_int(value)}" for label, value in parts)


def context_summary(
    stats: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
    mode: str = "brainos",
    turn: int | None = None,
) -> str:
    """Markdown summary of the context accounting for one turn."""

    report = report or {}
    sent = format_int(stats.get("final_context_tokens"))
    reference = format_int(stats.get("full_context_reference_tokens"))
    raw = format_int(stats.get("raw_history_tokens"))
    lines = [
        f"**Context — turn {format_int(turn)} · mode `{mode}`**",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Sent to model | **{sent} tokens** |",
        f"| Full-context baseline | {reference} tokens |",
        (
            "| Reduction vs baseline | "
            f"**{format_percent(stats.get('context_reduction_vs_full_context'))}** "
            f"({format_int(stats.get('token_savings_vs_full_context'))} tokens saved) |"
        ),
        f"| Conversation size | {raw} tokens (history only) |",
        (
            "| Budget | "
            f"{sent} / {format_int(stats.get('max_tokens'))} "
            f"({format_percent(stats.get('budget_utilization'))}) |"
        ),
        "",
        f"`{token_breakdown(stats)}`",
        "",
    ]

    considered = format_int(stats.get("history_messages_considered"))
    selected = format_int(stats.get("history_messages_selected"))
    dropped_history = format_int(stats.get("dropped_history_for_budget"))
    lines.append(
        f"- history: **{selected}** of {considered} messages kept"
        + (f", {dropped_history} dropped for budget" if dropped_history else "")
    )

    candidates = format_int(stats.get("candidate_memory_count"))
    memory_count = format_int(stats.get("selected_memory_count"))
    memory_dropped = format_int(stats.get("dropped_memories_for_budget"))
    lines.append(
        f"- memories: **{memory_count}** of {candidates} candidates selected"
        + (f", {memory_dropped} dropped for budget" if memory_dropped else "")
    )

    chunk_candidates = format_int(stats.get("candidate_chunk_count"))
    chunk_count = format_int(stats.get("selected_chunk_count"))
    chunk_dropped = format_int(stats.get("dropped_chunks_for_budget"))
    lines.append(
        f"- retrieved chunks: **{chunk_count}** of {chunk_candidates} candidates selected"
        + (f", {chunk_dropped} dropped" if chunk_dropped else "")
    )

    evidence = format_int(stats.get("evidence_tokens"))
    if evidence or chunk_count or memory_count:
        lines.append(
            f"- retrieved evidence: {evidence} tokens "
            f"(memory {format_int(stats.get('memory_block_tokens'))} + "
            f"chunks {format_int(stats.get('chunk_block_tokens'))})"
        )

    if isinstance(report, Mapping):
        counts = _reason_counts(report)
        if counts:
            detail = ", ".join(f"{reason} {count}" for reason, count in counts)
            lines.append(f"- filtered out: {detail}")
        # ``conflict_count`` is authoritative, but a report assembled without
        # the counter still knows how many pairs it resolved.
        conflicts = format_int(report.get("conflict_count")) or len(conflict_rows(report))
        if conflicts:
            lines.append(f"- conflicts: {conflicts} pair(s) resolved")
    if stats.get("system_truncated"):
        lines.append("- ⚠️ system prompt was truncated to fit the budget")
    if stats.get("exceeds_max_tokens"):
        lines.append("- ⚠️ final context still exceeds `max_tokens`")
    counter = stats.get("token_counter")
    if counter:
        lines.append(f"- token counter: `{counter}`")
    return "\n".join(lines)


def trace_markdown(
    *,
    query: str,
    stats: Mapping[str, Any],
    report: Mapping[str, Any],
    events: Sequence[Mapping[str, str]] = (),
    stored: Sequence[MemoryRecord] = (),
    generated: bool = False,
    error: str = "",
    model: str = "",
) -> str:
    """Render the sanitized cognitive trace in the plan's stage order.

    The stage sequence is the one the plan specifies:

    ```text
    Query → Observe → Recall → Relevance filtering → Context construction
          → LLM → Observation
    ```

    Counts come from the retrieval report and the token accounting; the
    runtime's own mapped events are appended underneath.
    """

    counts = _reason_counts(report)
    filtered = ", ".join(f"{reason} {count}" for reason, count in counts) or "none"
    llm_stage = (
        f"generated with `{model}`" if generated and model else "generated"
        if generated
        else "not called — no API key configured"
    )
    if error:
        llm_stage = f"failed: {clip(error, 200)}"
    stored_types = ", ".join(
        sorted({str(record.memory_type or "FACT") for record in stored})
    ) or "none"
    question_tokens = format_int(stats.get("current_message_tokens"))
    lines = [
        f"1. **Query** — {clip(query, 160)!r} ({question_tokens} tokens)",
        f"2. **Observe** — {len(stored)} candidate(s) stored ({stored_types})",
        f"3. **Recall** — {format_int(report.get('candidate_count'))} candidate(s) returned",
        (
            "4. **Relevance filtering** — "
            f"{format_int(report.get('selected_count'))} kept, "
            f"{len(list(_iter_dropped(report)))} dropped ({filtered})"
        ),
        (
            "5. **Context construction** — "
            f"{format_int(stats.get('final_context_tokens'))} tokens "
            f"({token_breakdown(stats)})"
        ),
        f"6. **LLM** — {llm_stage}",
        "7. **Observation** — assistant reply observed back into memory",
    ]
    text = "\n".join(lines)

    conflict_rows_data = conflict_rows(report)
    if conflict_rows_data:
        header = " | ".join(CONFLICT_COLUMNS)
        divider = " | ".join("---" for _ in CONFLICT_COLUMNS)
        text += f"\n\n**Conflicts**\n\n| {header} |\n| {divider} |\n"
        text += "\n".join(
            "| " + " | ".join(clip(cell, 80) for cell in row) + " |"
            for row in conflict_rows_data
        )

    visible = list(events)[:_MAX_TRACE_EVENTS]
    if visible:
        text += "\n\n**Runtime events**\n\n"
        text += "\n".join(
            f"- `{clip(event.get('name', ''), 40)}` — {clip(event.get('detail', ''), 160)}"
            for event in visible
            if isinstance(event, Mapping)
        )
        if len(events) > _MAX_TRACE_EVENTS:
            text += f"\n- … {len(events) - _MAX_TRACE_EVENTS} earlier event(s) omitted"
    return text


def status_line(
    *,
    turn: int | None = None,
    mode: str = "brainos",
    model: str = "",
    generated: bool = False,
    stats: Mapping[str, Any] | None = None,
    error: str = "",
    notice: str = "",
    configured: bool = True,
    has_turn: bool = True,
) -> str:
    """One-line status shown under the chat input.

    ``has_turn`` distinguishes "this turn produced no reply" from "no turn was
    run at all" (a refresh, or an empty message): only the former is worth
    explaining to the user.
    """

    stats = stats or {}
    parts: list[str] = []
    if turn is not None:
        parts.append(f"Turn {format_int(turn)}")
    parts.append(f"mode `{mode}`")
    if model:
        parts.append(model)
    if stats:
        parts.append(
            f"{format_int(stats.get('final_context_tokens'))} tokens sent "
            f"({format_percent(stats.get('context_reduction_vs_full_context'))} below baseline)"
        )
    line = " · ".join(parts)
    if error:
        line = f"⚠️ {error}"
    elif has_turn and not generated:
        reason = (
            "no API key configured — memory only"
            if not configured
            else "the provider returned no reply"
        )
        line += f" · **no model called** ({reason})"
    if notice:
        line = f"{line}\n\n{notice}" if line else notice
    return line


def export_payload(payload: Mapping[str, Any]) -> str:
    """Render an export payload as indented JSON text."""

    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _iter_dropped(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    dropped = report.get("dropped") if isinstance(report, Mapping) else None
    if not isinstance(dropped, list):
        return []
    return [item for item in dropped if isinstance(item, Mapping)]


def _reason_counts(report: Mapping[str, Any]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in _iter_dropped(report):
        reason = str(item.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))


__all__ = [
    "CHUNK_COLUMNS",
    "CONFLICT_COLUMNS",
    "DROPPED_MEMORY_COLUMNS",
    "RETRIEVED_MEMORY_COLUMNS",
    "STORED_MEMORY_COLUMNS",
    "chunk_rows",
    "clip",
    "conflict_rows",
    "context_summary",
    "dropped_rows",
    "export_payload",
    "format_int",
    "format_percent",
    "memory_rows",
    "render_prompt",
    "retrieved_rows",
    "round_value",
    "short_timestamp",
    "status_line",
    "token_breakdown",
    "trace_markdown",
]
