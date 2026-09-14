"""Findings vocabulary, ledger, prompt report, and aggregation (plan Phase 13).

The reporting layer has three promises the tests hold it to:

1. the vocabulary is closed — a finding cannot be constructed with a category,
   action, route, or stage nobody can render;
2. a finding never republishes what it caught — previews are clipped, invisible
   characters are stripped, and credentials appear only as a masked shape;
3. counts are cumulative and exact while the detail list is bounded, so a long
   session cannot grow an unbounded report and cannot lose a count either.
"""

from __future__ import annotations

import threading

import pytest

from security.findings import (
    ACTION_DOWNGRADED,
    ACTION_FLAGGED,
    ACTION_NEUTRALIZED,
    ACTION_QUARANTINED,
    ACTION_REDACTED,
    ACTIONS,
    CATEGORIES,
    CATEGORY_CONTROL_CHARACTERS,
    CATEGORY_CREDENTIAL_REDACTED,
    CATEGORY_DELIMITER_BREAKOUT,
    CATEGORY_INJECTION,
    CATEGORY_INVISIBLE_CHARACTERS,
    CATEGORY_ROLE_DOWNGRADED,
    CATEGORY_ROLE_SMUGGLING,
    MAX_PREVIEW_CHARS,
    MAX_RECENT_FINDINGS,
    REPORT_NOTE,
    ROUTE_CHUNK,
    ROUTE_CURRENT_MESSAGE,
    ROUTE_HISTORY,
    ROUTE_MEMORY,
    ROUTES,
    SECURITY_VERSION,
    STAGE_SELECTION,
    STAGES,
    PromptGuardReport,
    SecurityFinding,
    SecurityLedger,
    categories_for_families,
    clip_preview,
    empty_counts,
    merge_counts,
    security_report,
    summarize_security,
)
from security.guard import mask_secret, strip_invisible

#: A credential-shaped value used to prove it never survives into a report.
LIVE_KEY = "sk-live-abcdef0123456789abcdef"


def finding(**overrides: object) -> SecurityFinding:
    """A valid finding with the minimum fields, overridable per test."""

    fields: dict[str, object] = {
        "category": CATEGORY_INJECTION,
        "action": ACTION_FLAGGED,
        "route": ROUTE_MEMORY,
    }
    fields.update(overrides)
    return SecurityFinding(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# A closed vocabulary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field_name", ["category", "action", "route", "stage"])
def test_a_finding_rejects_a_word_no_surface_can_render(field_name: str) -> None:
    with pytest.raises(ValueError, match="unknown security"):
        finding(**{field_name: "not-a-real-value"})


def test_the_vocabulary_is_exactly_what_the_panels_and_artifacts_document() -> None:
    assert ACTIONS == (
        ACTION_FLAGGED,
        ACTION_QUARANTINED,
        ACTION_NEUTRALIZED,
        ACTION_REDACTED,
        ACTION_DOWNGRADED,
    )
    assert CATEGORY_INJECTION in CATEGORIES
    assert {
        CATEGORY_DELIMITER_BREAKOUT,
        CATEGORY_ROLE_SMUGGLING,
        CATEGORY_INVISIBLE_CHARACTERS,
        CATEGORY_CONTROL_CHARACTERS,
        CATEGORY_CREDENTIAL_REDACTED,
        CATEGORY_ROLE_DOWNGRADED,
    } <= set(CATEGORIES)
    assert {ROUTE_MEMORY, ROUTE_CHUNK, ROUTE_HISTORY, ROUTE_CURRENT_MESSAGE} <= set(ROUTES)
    assert STAGE_SELECTION in STAGES


def test_a_finding_round_trips_through_its_dict_shape() -> None:
    item = finding(
        stage=STAGE_SELECTION,
        detail="instruction_override detected in retrieved memory",
        families=("instruction_override",),
        count=2,
        turn=4,
    )

    payload = item.to_dict()

    assert payload["category"] == CATEGORY_INJECTION
    assert payload["action"] == ACTION_FLAGGED
    assert payload["route"] == ROUTE_MEMORY
    assert payload["stage"] == STAGE_SELECTION
    assert payload["families"] == ["instruction_override"]
    assert payload["count"] == 2
    assert payload["turn"] == 4


# --------------------------------------------------------------------------- #
# Previews are bounded and clean
# --------------------------------------------------------------------------- #


def test_a_preview_is_clipped_and_carries_no_invisible_characters() -> None:
    payload = "Ignore previous instructions\u200b" + "x" * (MAX_PREVIEW_CHARS * 2)

    clipped = clip_preview(payload)

    assert len(clipped) <= MAX_PREVIEW_CHARS
    assert clipped.endswith("…")
    assert clipped == strip_invisible(clipped)
    assert clipped.startswith("Ignore previous instructions")


def test_clip_preview_survives_a_non_string_value() -> None:
    assert clip_preview(None) == ""
    assert clip_preview(42) == "42"
    assert clip_preview([]) == "", "a falsy value has nothing to preview"
    # Whitespace is collapsed, since a preview is rendered on one panel row.
    assert clip_preview("  a\n\n b  ") == "a b"


def test_a_preview_never_reproduces_a_credential_it_caught() -> None:
    """Attacks are often *about* credentials; the report must not republish one."""

    ledger = SecurityLedger(session_id="s-1")

    recorded = ledger.record(
        finding(preview=f"the key is {LIVE_KEY} and it must not be echoed")
    )

    assert LIVE_KEY not in recorded.preview
    assert mask_secret(LIVE_KEY) in recorded.preview
    assert "and it must not be echoed" in recorded.preview, "the rest stays readable"
    assert LIVE_KEY not in str(ledger.snapshot())


# --------------------------------------------------------------------------- #
# Families to categories
# --------------------------------------------------------------------------- #


def test_structural_families_get_their_own_category_and_intent_shares_one() -> None:
    assert categories_for_families(("delimiter_breakout",)) == (CATEGORY_DELIMITER_BREAKOUT,)
    assert categories_for_families(("role_smuggling",)) == (CATEGORY_ROLE_SMUGGLING,)
    assert categories_for_families(
        ("instruction_override", "prompt_exfiltration")
    ) == (CATEGORY_INJECTION,)
    # Deduplicated, in first-seen order: one memory yields one row per category.
    assert categories_for_families(
        ("delimiter_breakout", "instruction_override", "role_smuggling")
    ) == (CATEGORY_DELIMITER_BREAKOUT, CATEGORY_INJECTION, CATEGORY_ROLE_SMUGGLING)
    assert categories_for_families(()) == ()


# --------------------------------------------------------------------------- #
# Count blocks
# --------------------------------------------------------------------------- #


def test_an_empty_count_block_has_every_section_and_no_numbers() -> None:
    counts = empty_counts()

    assert set(counts) == {"totals", "by_action", "by_route", "by_stage", "by_family"}
    assert all(values == {} for values in counts.values())


def test_merge_counts_adds_key_by_key_and_ignores_garbage() -> None:
    base = {"totals": {"injection_pattern": 1}, "by_route": {"memory": 1}}
    other = {"totals": {"injection_pattern": 2, "credential_redacted": 1}}

    merged = merge_counts(base, other)

    assert merged["totals"] == {"injection_pattern": 3, "credential_redacted": 1}
    assert merged["by_route"] == {"memory": 1}
    assert merge_counts(base, {"totals": "not-a-mapping"})["totals"] == {
        "injection_pattern": 1
    }
    assert merge_counts(base, {"totals": {"injection_pattern": "NaN"}})["totals"] == {
        "injection_pattern": 1
    }


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


def test_a_fresh_ledger_is_clean_and_says_what_it_is() -> None:
    snapshot = SecurityLedger(session_id="s-2").snapshot()

    assert snapshot["clean"] is True
    assert snapshot["record_count"] == 0
    assert snapshot["finding_count"] == 0
    assert snapshot["recent"] == []
    assert snapshot["security_version"] == SECURITY_VERSION
    assert snapshot["session_id"] == "s-2"
    assert snapshot["note"] == REPORT_NOTE


def test_count_records_nothing_when_the_measured_number_is_zero() -> None:
    ledger = SecurityLedger()

    assert ledger.count(CATEGORY_INJECTION, ACTION_FLAGGED, ROUTE_MEMORY, count=0) is None
    assert ledger.clean is True

    assert (
        ledger.count(CATEGORY_INJECTION, ACTION_FLAGGED, ROUTE_MEMORY, count=3)
        is not None
    )
    assert ledger.finding_count == 3


def test_findings_are_stamped_with_the_turn_they_belong_to() -> None:
    ledger = SecurityLedger()
    ledger.set_turn(7)

    stamped = ledger.record(finding())
    explicit = ledger.record(finding(turn=2))

    assert stamped.turn == 7
    assert explicit.turn == 2, "an explicit turn wins over the ledger's cursor"


def test_counts_are_cumulative_across_actions_routes_stages_and_families() -> None:
    ledger = SecurityLedger()
    ledger.record(
        finding(route=ROUTE_MEMORY, families=("instruction_override",), count=2)
    )
    ledger.record(
        finding(
            category=CATEGORY_CREDENTIAL_REDACTED,
            action=ACTION_REDACTED,
            route=ROUTE_HISTORY,
            stage=STAGE_SELECTION,
        )
    )

    snapshot = ledger.snapshot()

    assert snapshot["totals"] == {
        CATEGORY_INJECTION: 2,
        CATEGORY_CREDENTIAL_REDACTED: 1,
    }
    assert snapshot["by_action"] == {ACTION_FLAGGED: 2, ACTION_REDACTED: 1}
    assert snapshot["by_route"] == {ROUTE_MEMORY: 2, ROUTE_HISTORY: 1}
    assert snapshot["by_stage"] == {"prompt": 2, STAGE_SELECTION: 1}
    assert snapshot["by_family"] == {"instruction_override": 2}
    assert snapshot["record_count"] == 3
    assert snapshot["clean"] is False


def test_the_detail_list_is_bounded_but_the_totals_are_not() -> None:
    ledger = SecurityLedger()
    total = MAX_RECENT_FINDINGS * 3

    for index in range(total):
        ledger.record(finding(count=1, detail=f"finding {index}"))

    snapshot = ledger.snapshot()

    assert snapshot["record_count"] == total
    assert len(snapshot["recent"]) == MAX_RECENT_FINDINGS
    assert snapshot["finding_count"] == MAX_RECENT_FINDINGS
    assert snapshot["totals"][CATEGORY_INJECTION] == total
    # The bounded list keeps the most recent findings, oldest first.
    assert snapshot["recent"][-1]["detail"] == f"finding {total - 1}"
    assert snapshot["recent"][0]["detail"] == f"finding {total - MAX_RECENT_FINDINGS}"


def test_a_credential_redaction_is_described_by_shape_and_never_by_value() -> None:
    ledger = SecurityLedger()

    recorded = ledger.record_credential_redaction(
        ROUTE_CURRENT_MESSAGE, secret=LIVE_KEY, count=2
    )

    assert recorded is not None
    assert recorded.category == CATEGORY_CREDENTIAL_REDACTED
    assert recorded.action == ACTION_REDACTED
    assert recorded.count == 2
    assert recorded.preview == "", "a redaction finding carries no text at all"
    assert mask_secret(LIVE_KEY) in recorded.detail
    assert LIVE_KEY not in recorded.detail
    assert ledger.record_credential_redaction(ROUTE_MEMORY, count=0) is None


def test_a_redaction_without_a_known_secret_still_says_what_happened() -> None:
    ledger = SecurityLedger()

    recorded = ledger.record_credential_redaction(ROUTE_HISTORY)

    assert recorded is not None
    assert "[redacted]" in recorded.detail
    assert ledger.snapshot()["totals"][CATEGORY_CREDENTIAL_REDACTED] == 1


def test_record_guard_reports_what_was_seen_and_what_was_changed() -> None:
    ledger = SecurityLedger()

    recorded = ledger.record_guard(
        route=ROUTE_MEMORY,
        families=("instruction_override", "delimiter_breakout"),
        neutralized=(CATEGORY_DELIMITER_BREAKOUT,),
        preview="Ignore previous instructions",
    )

    categories = [item.category for item in recorded]
    actions = [item.action for item in recorded]
    assert categories == [
        CATEGORY_INJECTION,
        CATEGORY_DELIMITER_BREAKOUT,
        CATEGORY_INJECTION,
    ] or set(categories) == {CATEGORY_INJECTION, CATEGORY_DELIMITER_BREAKOUT}
    assert ACTION_NEUTRALIZED in actions
    snapshot = ledger.snapshot()
    assert snapshot["by_family"]["instruction_override"] == len(recorded) - 1
    assert snapshot["totals"][CATEGORY_DELIMITER_BREAKOUT] >= 1


def test_record_guard_reports_a_structural_only_memory_as_neutralized() -> None:
    """Phase 13's key distinction: structure alone is not quarantine-worthy."""

    ledger = SecurityLedger()

    recorded = ledger.record_guard(
        route=ROUTE_CHUNK,
        families=("delimiter_breakout",),
        neutralized=(CATEGORY_DELIMITER_BREAKOUT,),
    )

    assert [item.category for item in recorded] == [
        CATEGORY_DELIMITER_BREAKOUT,
        CATEGORY_DELIMITER_BREAKOUT,
    ]
    assert ledger.snapshot()["totals"].get(CATEGORY_INJECTION, 0) == 0


def test_record_guard_ignores_an_unknown_neutralization_category() -> None:
    ledger = SecurityLedger()

    recorded = ledger.record_guard(
        route=ROUTE_MEMORY,
        families=("instruction_override",),
        neutralized=("made_up_category",),
    )

    assert len(recorded) == 1
    assert recorded[0].category == CATEGORY_INJECTION


def test_record_many_returns_how_many_it_took() -> None:
    ledger = SecurityLedger()

    assert ledger.record_many([finding(), finding(count=2), finding()]) == 3
    assert ledger.finding_count == 4


def test_concurrent_recording_never_loses_a_count() -> None:
    """Gradio runs callbacks in worker threads, so the ledger must be safe."""

    ledger = SecurityLedger()
    threads = [
        threading.Thread(target=lambda: ledger.record_many([finding()] * 50))
        for _ in range(8)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert ledger.finding_count == 400
    assert ledger.snapshot()["totals"][CATEGORY_INJECTION] == 400


def test_a_snapshot_is_a_copy_the_ledger_cannot_mutate_afterwards() -> None:
    ledger = SecurityLedger()
    ledger.record(finding())

    snapshot = ledger.snapshot()
    ledger.record(finding())

    assert snapshot["record_count"] == 1
    assert snapshot["totals"][CATEGORY_INJECTION] == 1
    assert ledger.to_dict() == ledger.snapshot(), "to_dict is an alias, not a variant"


# --------------------------------------------------------------------------- #
# The prompt report
# --------------------------------------------------------------------------- #


def test_an_untouched_prompt_reports_clean_with_no_findings() -> None:
    report = PromptGuardReport()

    assert report.clean is True
    assert report.findings() == ()
    payload = report.to_dict()
    assert payload["clean"] is True
    assert payload["record_count"] == 0
    assert payload["note"] == REPORT_NOTE
    assert payload["detail"]["drop_suspicious_memories"] is False


def test_an_intent_family_is_flagged_by_default_and_quarantined_on_request() -> None:
    counts = {"instruction_override": 2}

    default = PromptGuardReport(memory_families=counts, suspicious_memories=2)
    quarantine = PromptGuardReport(
        memory_families=counts,
        suspicious_memories=2,
        quarantined_memories=2,
        drop_suspicious_memories=True,
    )

    assert [item.action for item in default.findings()] == [ACTION_FLAGGED]
    assert [item.action for item in quarantine.findings()] == [ACTION_QUARANTINED]
    assert default.findings()[0].count == 2
    assert default.findings()[0].route == ROUTE_MEMORY
    assert default.findings()[0].stage == STAGE_SELECTION
    assert quarantine.to_dict()["by_action"][ACTION_QUARANTINED] == 2


def test_a_chunk_family_is_flagged_on_the_chunk_route() -> None:
    report = PromptGuardReport(chunk_families={"role_assumption": 1}, suspicious_chunks=1)

    found = report.findings()

    assert found[0].route == ROUTE_CHUNK
    assert found[0].category == CATEGORY_INJECTION
    assert "transcript chunk" in found[0].detail
    assert report.clean is False


def test_structural_neutralizations_are_reported_as_neutralized_not_flagged() -> None:
    report = PromptGuardReport(
        memory_neutralized={CATEGORY_DELIMITER_BREAKOUT: 1, CATEGORY_ROLE_SMUGGLING: 1},
        chunk_neutralized={CATEGORY_INVISIBLE_CHARACTERS: 1},
    )

    by_route = {(item.route, item.category): item for item in report.findings()}

    assert set(by_route) == {
        (ROUTE_MEMORY, CATEGORY_DELIMITER_BREAKOUT),
        (ROUTE_MEMORY, CATEGORY_ROLE_SMUGGLING),
        (ROUTE_CHUNK, CATEGORY_INVISIBLE_CHARACTERS),
    }
    assert all(item.action == ACTION_NEUTRALIZED for item in by_route.values())
    assert all(item.detail for item in by_route.values())
    assert report.to_dict()["totals"].get(CATEGORY_INJECTION, 0) == 0


def test_history_and_message_redactions_appear_on_their_own_routes() -> None:
    report = PromptGuardReport(
        history_credentials_redacted=2,
        history_messages_redacted=1,
        current_message_redacted=1,
        history_roles_downgraded=3,
    )

    findings = report.findings()
    routes = {(item.route, item.action) for item in findings}

    assert (ROUTE_HISTORY, ACTION_REDACTED) in routes
    assert (ROUTE_CURRENT_MESSAGE, ACTION_REDACTED) in routes
    assert (ROUTE_HISTORY, ACTION_DOWNGRADED) in routes
    downgraded = [item for item in findings if item.action == ACTION_DOWNGRADED][0]
    assert downgraded.category == CATEGORY_ROLE_DOWNGRADED
    assert downgraded.count == 3
    assert report.clean is False
    assert report.to_dict()["detail"]["history_messages_redacted"] == 1


def test_the_report_detail_block_carries_every_counter_the_panel_shows() -> None:
    report = PromptGuardReport(
        memory_families={"instruction_override": 1},
        chunk_families={"prompt_exfiltration": 1},
        memory_neutralized={CATEGORY_CONTROL_CHARACTERS: 1},
        chunk_neutralized={CATEGORY_DELIMITER_BREAKOUT: 2},
        suspicious_memories=1,
        suspicious_chunks=1,
        quarantined_memories=0,
        flagged_candidates=1,
        history_credentials_redacted=1,
        history_messages_redacted=1,
        current_message_redacted=0,
        history_roles_downgraded=1,
        drop_suspicious_memories=True,
    )

    detail = report.to_dict()["detail"]

    assert detail == {
        "drop_suspicious_memories": True,
        "suspicious_memories": 1,
        "suspicious_chunks": 1,
        "quarantined_memories": 0,
        "flagged_candidates": 1,
        "history_credentials_redacted": 1,
        "history_messages_redacted": 1,
        "current_message_redacted": 0,
        "history_roles_downgraded": 1,
        "memory_families": {"instruction_override": 1},
        "chunk_families": {"prompt_exfiltration": 1},
        "memory_neutralized": {CATEGORY_CONTROL_CHARACTERS: 1},
        "chunk_neutralized": {CATEGORY_DELIMITER_BREAKOUT: 2},
    }


def test_zero_valued_and_unparseable_counts_drop_out_of_the_report() -> None:
    report = PromptGuardReport(memory_families={"instruction_override": 0, "role_assumption": "2"})  # type: ignore[dict-item]

    detail = report.to_dict()["detail"]

    assert detail["memory_families"] == {"role_assumption": 2}
    assert report.findings()[0].count == 2


# --------------------------------------------------------------------------- #
# The session report
# --------------------------------------------------------------------------- #


def test_a_report_without_a_ledger_is_a_clean_block_not_an_error() -> None:
    report = security_report(None)

    assert report["clean"] is True
    assert report["record_count"] == 0
    assert report["recent"] == []
    assert report["security_version"] == SECURITY_VERSION
    assert report["note"] == REPORT_NOTE
    assert report["last_prompt"] == {}


def test_the_session_report_combines_the_ledger_the_turn_and_the_policy() -> None:
    ledger = SecurityLedger(session_id="s-3")
    ledger.record(finding())
    last_prompt = PromptGuardReport(history_credentials_redacted=1).to_dict()

    report = security_report(
        ledger,
        last_prompt=last_prompt,
        policy={"drop_suspicious_memories": True, "neutralize_text": False},
    )

    assert report["session_id"] == "s-3"
    assert report["record_count"] == 1
    assert report["last_prompt"]["detail"]["history_credentials_redacted"] == 1
    assert report["policy"] == {
        "drop_suspicious_memories": True,
        "neutralize_text": False,
    }


def test_a_non_mapping_policy_is_simply_omitted() -> None:
    report = security_report(SecurityLedger(), last_prompt="not-a-mapping", policy=None)

    assert "policy" not in report
    assert report["last_prompt"] == {}


# --------------------------------------------------------------------------- #
# Aggregation across a run
# --------------------------------------------------------------------------- #


def test_summarize_security_accepts_reports_dicts_and_nothing_at_all() -> None:
    blocks = [
        PromptGuardReport(memory_families={"instruction_override": 1}),
        PromptGuardReport(history_credentials_redacted=2).to_dict(),
        None,
        "not-a-mapping",
        {},
    ]

    summary = summarize_security(blocks)

    assert summary["record_count"] == 5, "every record is counted, findings or not"
    assert summary["records_with_findings"] == 2
    assert summary["clean"] is False
    assert summary["credentials_redacted"] == 2
    assert summary["quarantined"] == 0
    assert summary["totals"][CATEGORY_INJECTION] == 1
    assert summary["families_seen"] == ["instruction_override"]
    assert summary["security_version"] == SECURITY_VERSION


def test_summarize_security_counts_quarantines_across_modes() -> None:
    quarantined = PromptGuardReport(
        memory_families={"instruction_override": 3},
        quarantined_memories=3,
        drop_suspicious_memories=True,
    ).to_dict()

    summary = summarize_security([quarantined, quarantined])

    assert summary["quarantined"] == 6
    assert summary["by_action"][ACTION_QUARANTINED] == 6


def test_an_artifact_written_before_this_phase_sums_to_a_clean_zero_block() -> None:
    """Old records have no ``security`` key; aggregation must not fail on them."""

    summary = summarize_security([None, None, None])

    assert summary["clean"] is True
    assert summary["record_count"] == 3
    assert summary["records_with_findings"] == 0
    assert summary["totals"] == {}
    assert summary["families_seen"] == []


def test_an_empty_run_is_clean_and_says_how_many_records_it_saw() -> None:
    summary = summarize_security([])

    assert summary == {
        **empty_counts(),
        "record_count": 0,
        "records_with_findings": 0,
        "quarantined": 0,
        "credentials_redacted": 0,
        "families_seen": [],
        "clean": True,
        "security_version": SECURITY_VERSION,
        "note": REPORT_NOTE,
    }


def test_the_report_note_refuses_to_claim_safety_from_absence() -> None:
    """A clean report is not a safety certificate, and it must say so itself."""

    assert "not that the" in REPORT_NOTE
    assert summarize_security([])["note"] == REPORT_NOTE
    assert SecurityLedger().snapshot()["note"] == REPORT_NOTE
    assert PromptGuardReport().to_dict()["note"] == REPORT_NOTE
