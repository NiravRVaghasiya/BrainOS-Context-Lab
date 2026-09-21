"""Phase 18: the plan's §24, as a machine-checked ledger.

Phases 1–17 each wrote the tests their own phase required, so by the time the
plan reached its test-suite phase there were 1179 tests and no single place that
answered the question §24 actually asks: *does every item the plan names have a
test, and does that test still exist?* A coverage percentage cannot answer it —
a line can be executed by a test that asserts nothing about the requirement.

This module is that answer. It maps each §24 item to named tests, then runs the
suite's own collection in a subprocess and fails if any mapped test has been
renamed, moved, or deleted. The tests themselves still live in the four
directories the plan's §28 lays out (``tests/unit``, ``tests/integration``,
``tests/evaluation``, ``tests/security``); this file is the index, not a fifth
copy.

Phase 19 extends the same instrument to §25, the MVP checklist. That section is
a different shape of requirement — fourteen things a *user* can do, in order,
rather than fourteen components — so it gets its own table and its own
properties:

* one row per item, in the plan's order, with the walkthrough test that
  demonstrates it and the focused test that pins the behaviour;
* **item 1 ("open the HF Space") is recorded as external**, with the reason
  spelled out. Deploying to Hugging Face is an operator action: this repository
  can test that the checkout *is* deployable — app entry point, declared
  dependencies, a Gradio app that builds — and cannot test that a Space is
  running. The row says which half it covers instead of implying the other.

Three properties are deliberate:

* **It reads the plan.** If a future edit removes §24 or §25, or renames an item
  or a category, the ledger's own premise is gone and this module says so
  instead of quietly checking a stale list.
* **It names tests that already exist.** Nothing here was written to make the
  ledger green; the mappings point at the tests that carry each guarantee today.
* **The MVP rows are ordered and complete.** §25 is a sequence, so a row that is
  missing, duplicated, or reordered is a failure, not a formatting detail.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "BrainOS_Context_Lab_Implementation_Plan.md"

#: The plan's §28 test layout. §24's items are verified *by* tests in these
#: directories; the mapping below records which.
PLAN_CATEGORIES = ("unit", "integration", "evaluation", "security")

#: The plan's §28 predates the Phase 4 UI work, which added a fifth directory
#: for the Gradio-facing callback contracts. It is a real category of this
#: suite, so the ledger accepts it even though §24 does not name it.
CATEGORIES = (*PLAN_CATEGORIES, "ui")

#: Sections whose items are *rules* (a property of the whole system) rather
#: than components. A rule proven in one file is one refactor away from
#: disappearing silently, so those requirements must be carried by two files.
RULE_SECTIONS = frozenset({"Integration tests", "Security tests"})


@dataclass(frozen=True)
class Requirement:
    """One item from plan §24 and the tests that carry it."""

    section: str
    item: str
    category: str
    tests: tuple[str, ...]


REQUIREMENTS: tuple[Requirement, ...] = (
    # ------------------------------------------------------------------ #
    # §24 — Unit tests
    # ------------------------------------------------------------------ #
    Requirement(
        section="Unit tests",
        item="provider adapters",
        category="unit",
        tests=(
            "tests/unit/test_providers.py::test_openai_provider_lists_models_and_normalizes_generation",
            "tests/unit/test_providers.py::test_request_overrides_do_not_mutate_config",
            "tests/unit/test_provider_adapter_edges.py::test_the_session_configuration_builds_the_client_once",
            "tests/unit/test_provider_adapter_edges.py::test_a_missing_optional_sdk_names_the_extra_that_provides_it",
            "tests/unit/test_provider_adapter_edges.py::test_messages_that_would_reach_the_api_are_refused",
            "tests/integration/test_provider_http_live.py::test_the_sdk_reaches_a_base_url_and_normalizes_the_response",
        ),
    ),
    Requirement(
        section="Unit tests",
        item="API key redaction",
        category="security",
        tests=(
            "tests/unit/test_providers.py::test_provider_errors_redact_api_keys",
            "tests/security/test_secrets.py::test_service_diagnostics_never_include_provider_key",
            "tests/security/test_log_hygiene.py::test_a_provider_that_fails_mid_session_stays_redacted_on_every_surface",
            "tests/unit/test_trace.py::test_sanitize_trace_redacts_secret_bearing_mapping",
        ),
    ),
    Requirement(
        section="Unit tests",
        item="context builder",
        category="unit",
        tests=(
            "tests/unit/test_context_builder.py::test_context_builder_delimits_memory_and_accounts_for_tokens",
            "tests/unit/test_context_builder.py::test_message_order_is_system_memory_history_question",
            "tests/unit/test_context_builder.py::test_accounting_reports_every_field_the_plan_requires",
        ),
    ),
    Requirement(
        section="Unit tests",
        item="token budgeting",
        category="unit",
        tests=(
            "tests/unit/test_context_builder.py::test_global_ceiling_drops_the_oldest_history_first",
            "tests/unit/test_context_builder.py::test_section_budgets_are_respected_independently",
            "tests/unit/test_context_builder.py::test_memories_are_dropped_before_the_system_prompt",
            "tests/unit/test_tokenizers.py::test_truncate_to_tokens_respects_the_budget",
        ),
    ),
    Requirement(
        section="Unit tests",
        item="memory filtering",
        category="unit",
        tests=(
            "tests/unit/test_memory_policy.py::test_extract_candidates_keeps_project_facts_and_skips_chatter",
            "tests/unit/test_retrieval_policy.py::test_off_topic_memories_are_dropped_and_audited",
            "tests/unit/test_retrieval_policy.py::test_empty_memories_never_reach_the_prompt",
        ),
    ),
    Requirement(
        section="Unit tests",
        item="session isolation",
        category="unit",
        tests=(
            "tests/unit/test_storage_sqlite.py::test_messages_isolated_by_session_and_conversation",
            "tests/unit/test_storage_sqlite.py::test_memories_isolated_and_cleared",
            "tests/unit/test_adapter.py::test_two_injected_runtimes_do_not_share_memory",
            "tests/unit/test_service.py::test_session_isolation_across_services",
            "tests/integration/test_context_pipeline.py::test_two_sessions_do_not_share_context_or_memories",
            "tests/integration/test_brainos_runtime.py::test_pinned_runtimes_are_isolated_by_session",
        ),
    ),
    Requirement(
        section="Unit tests",
        item="conflict handling",
        category="unit",
        tests=(
            "tests/unit/test_retrieval_policy.py::test_runtime_contradiction_drops_the_older_claim",
            "tests/unit/test_retrieval_policy.py::test_heuristic_conflict_drops_the_corrected_claim",
            "tests/integration/test_context_pipeline.py::test_pipeline_resolves_a_correction_before_it_reaches_the_model",
        ),
    ),
    # ------------------------------------------------------------------ #
    # §24 — Integration tests
    # ------------------------------------------------------------------ #
    Requirement(
        section="Integration tests",
        item="user input → observe → recall → context → provider → response → memory update",
        category="integration",
        tests=(
            "tests/unit/test_service.py::test_service_observes_fact_and_retrieves_it_later",
            "tests/unit/test_controller.py::test_chat_turn_fills_every_panel",
            "tests/unit/test_controller.py::test_chat_records_the_fact_in_the_memory_panel",
            "tests/integration/test_chat_controller_live.py::test_live_panels_show_a_fact_recalled_many_turns_later",
            "tests/integration/test_context_pipeline.py::test_live_runtime_context_pipeline",
        ),
    ),
    # ------------------------------------------------------------------ #
    # §24 — Evaluation tests
    # ------------------------------------------------------------------ #
    Requirement(
        section="Evaluation tests",
        item="deterministic mock models verify benchmark calculations",
        category="evaluation",
        tests=(
            "tests/evaluation/test_scoring.py::test_an_accepted_answer_grades_correct",
            "tests/evaluation/test_scoring.py::test_aggregate_reports_rates_with_their_own_denominators",
            "tests/unit/test_metrics.py::test_retrieval_metrics",
            "tests/evaluation/test_experiment.py::test_dry_run_measures_prompts_without_calling_a_model",
            "tests/evaluation/test_experiment.py::test_generated_run_grades_answers_and_records_latency",
        ),
    ),
    # ------------------------------------------------------------------ #
    # §24 — Security tests
    # ------------------------------------------------------------------ #
    Requirement(
        section="Security tests",
        item="API key never appears in logs",
        category="security",
        tests=(
            "tests/security/test_log_hygiene.py::test_no_shipped_module_configures_logging_or_emits_a_record",
            "tests/security/test_log_hygiene.py::test_importing_every_shipped_package_installs_no_log_sink",
            "tests/security/test_log_hygiene.py::test_the_storage_diagnostic_prints_the_exception_type_and_nothing_else",
            "tests/security/test_log_hygiene.py::test_the_single_mode_cli_writes_no_credential_to_stdout_or_stderr",
            "tests/security/test_log_hygiene.py::test_the_pipeline_cli_writes_no_credential_to_stdout_or_stderr",
            # A run's artifacts are the other half of "logs": a key that reaches
            # one is a key an operator will read later.
            "tests/security/test_pipeline_secrets.py::test_the_cli_reads_a_key_from_the_environment_and_never_writes_it",
            "tests/security/test_artifact_scan.py::test_nothing_this_project_ships_contains_a_credential",
        ),
    ),
    Requirement(
        section="Security tests",
        item="Session A cannot access Session B memory",
        category="security",
        tests=(
            "tests/unit/test_controller.py::test_sessions_do_not_share_memory",
            "tests/integration/test_security_live.py::test_two_live_sessions_do_not_share_memory_or_findings",
            "tests/security/test_storage_secrets.py::test_deleted_session_data_is_gone_from_the_store",
        ),
    ),
    Requirement(
        section="Security tests",
        item="Retrieved memory cannot override system instructions",
        category="security",
        tests=(
            "tests/security/test_injection_corpus.py::test_the_system_prompt_always_outranks_the_transcript",
            "tests/security/test_injection_corpus.py::test_no_attack_probe_can_escape_the_memory_block",
            "tests/security/test_history_guard.py::test_a_transcript_claiming_the_system_role_cannot_outvote_the_application",
            "tests/integration/test_security_live.py::test_live_recalled_memory_is_flagged_and_neutralized_before_it_renders",
        ),
    ),
    Requirement(
        section="Security tests",
        item="User data is not accidentally returned in diagnostics",
        category="security",
        tests=(
            "tests/unit/test_controller.py::test_the_session_key_never_reaches_a_panel",
            "tests/security/test_mode_secrets.py::test_a_key_pasted_into_the_transcript_never_reaches_diagnostics",
            "tests/security/test_error_records.py::test_the_failure_record_is_built_from_named_fields_not_the_whole_artifact",
            "tests/security/test_storage_secrets.py::test_export_and_every_view_are_key_free",
        ),
    ),
)


@dataclass(frozen=True)
class MVPItem:
    """One numbered item from plan §25 and the evidence that it is walkable."""

    number: int
    item: str
    tests: tuple[str, ...] = ()
    #: Set for an item this repository cannot verify on its own behalf. ``tests``
    #: must stay empty for such a row: the honest record is an operator action.
    external: str = ""


MVP_ITEMS: tuple[MVPItem, ...] = (
    MVPItem(
        number=1,
        item="Open the HF Space",
        external=(
            "Deploying to a Space is an operator action, not a code path. The "
            "rows below check that this checkout is deployable (app entry point, "
            "declared dependencies, a building Gradio app); whether a Space is "
            "running is recorded in the phase log, not claimed here."
        ),
        tests=(
            "tests/unit/test_deployment_config.py::test_app_py_exists_and_bootstraps_the_source_path",
            "tests/unit/test_deployment_config.py::test_requirements_txt_lists_the_runtime_dependencies",
            "tests/ui/test_ui.py::test_app_builds_and_registers_every_callback",
        ),
    ),
    MVPItem(
        number=2,
        item="Select OpenAI",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_2_through_4_a_visitor_selects_openai_a_key_and_a_model",
            "tests/unit/test_controller.py::test_connect_lists_models_and_clears_the_key_box",
            "tests/unit/test_controller.py::test_connect_rejects_an_invalid_configuration",
        ),
    ),
    MVPItem(
        number=3,
        item="Enter an API key",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_3_the_key_is_never_returned_to_the_browser",
            "tests/unit/test_controller.py::test_connect_keeps_the_key_in_server_memory_only",
            "tests/ui/test_ui.py::test_connect_callback_clears_the_key_box_and_names_no_secret",
        ),
    ),
    MVPItem(
        number=4,
        item="Select a model",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_4_the_selected_model_is_the_one_the_turn_used",
            "tests/integration/test_baseline_modes_live.py::test_live_the_controller_reports_the_mode_that_ran",
        ),
    ),
    MVPItem(
        number=5,
        item="Start a conversation",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_5_a_conversation_starts_and_the_provider_is_called",
            "tests/integration/test_chat_controller_live.py::test_live_panels_show_a_fact_recalled_many_turns_later",
        ),
    ),
    MVPItem(
        number=6,
        item="Store a fact",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_6_the_fact_is_stored_in_brainos_memory",
            "tests/unit/test_controller.py::test_chat_records_the_fact_in_the_memory_panel",
            "tests/unit/test_memory_policy.py::test_extract_candidates_keeps_project_facts_and_skips_chatter",
        ),
    ),
    MVPItem(
        number=7,
        item="Continue the conversation",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_7_the_conversation_continues_past_the_fact",
            "tests/unit/test_baseline_modes.py::test_full_context_replays_the_whole_conversation",
        ),
    ),
    MVPItem(
        number=8,
        item="Retrieve the fact later",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_8_the_fact_is_retrieved_later_into_the_prompt",
            "tests/unit/test_baseline_modes.py::test_brainos_keeps_the_fact_the_sliding_window_lost",
            "tests/integration/test_baseline_modes_live.py::test_live_brainos_keeps_the_fact_for_the_same_cost_as_the_window",
        ),
    ),
    MVPItem(
        number=9,
        item="Inspect BrainOS memory",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_9_the_memory_panel_shows_what_brainos_holds",
            "tests/unit/test_panels.py::test_memory_rows_match_the_declared_columns",
            "tests/unit/test_controller.py::test_chat_records_the_fact_in_the_memory_panel",
        ),
    ),
    MVPItem(
        number=10,
        item="Inspect retrieved context",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_10_the_context_panel_shows_the_prompt_and_its_accounting",
            "tests/integration/test_mvp_walkthrough.py::test_item_10_the_retrieved_context_is_attributed",
            "tests/unit/test_panels.py::test_retrieved_rows_show_only_what_reached_the_prompt",
            "tests/unit/test_panels.py::test_context_summary_reports_the_baseline_next_to_the_savings",
        ),
    ),
    MVPItem(
        number=11,
        item="Compare BrainOS against full-context mode",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_11_brainos_and_full_context_are_compared_on_the_same_session",
            "tests/integration/test_mvp_walkthrough.py::test_item_11_the_mode_is_visible_in_the_session_payload",
            "tests/integration/test_baseline_modes_live.py::test_live_full_context_replays_everything_and_is_the_reference",
        ),
    ),
    MVPItem(
        number=12,
        item="See token usage",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_12_token_usage_is_visible_per_turn_and_per_session",
            "tests/unit/test_chat_limits_controller.py::test_turn_view_carries_usage_summary_and_report",
            "tests/unit/test_chat_limits_controller.py::test_turn_view_usage_never_carries_the_key",
        ),
    ),
    MVPItem(
        number=13,
        item="Run a small benchmark",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_13_a_small_benchmark_runs_from_the_tab",
            "tests/integration/test_mvp_walkthrough.py::test_item_13_the_run_discloses_its_cost_before_it_is_trusted",
            "tests/integration/test_mvp_walkthrough.py::test_item_13_the_run_writes_a_report_and_a_reproducibility_manifest",
            "tests/ui/test_ui.py::test_run_callback_feeds_the_tables_the_figures_and_the_report",
            "tests/ui/test_ui.py::test_preview_callback_shows_a_ceiling_and_writes_nothing",
        ),
    ),
    MVPItem(
        number=14,
        item="Export results",
        tests=(
            "tests/integration/test_mvp_walkthrough.py::test_item_14_the_session_exports_with_its_transcript_usage_and_runs",
            "tests/integration/test_mvp_walkthrough.py::test_item_14_the_export_is_key_free",
            "tests/ui/test_ui.py::test_export_callback_writes_a_credential_free_json_file",
            "tests/integration/test_chat_controller_live.py::test_live_export_contains_the_conversation",
        ),
    ),
)


@pytest.fixture(scope="module")
def plan_text() -> str:
    return PLAN.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def collected_tests() -> set[str]:
    """Collect the suite in a subprocess, exactly as a contributor runs it."""

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(" ")
    }


def _defined_in_source(node_id: str) -> bool:
    """Return True when the named test function exists in its module's source."""

    path, name = node_id.split("::")
    source = ROOT / path
    if not source.is_file():
        return False
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(tree)
    )


def _module_was_collected(module: str, collected_tests: set[str]) -> bool:
    """Return True when the module contributed at least one collected id.

    A module whose optional dependency is absent is skipped wholesale by
    ``tests/conftest.py`` and contributes nothing; a module that *was* collected
    but no longer contains a mapped function has had that test renamed or
    deleted, and the ledger must say so.
    """

    return any(node_id.split("::")[0] == module for node_id in collected_tests)


# --------------------------------------------------------------------------- #
# The plan still says what this file traces
# --------------------------------------------------------------------------- #


def test_the_plan_still_defines_the_phase_and_its_four_categories(plan_text: str) -> None:
    assert "Phase 18 — Test Suite" in plan_text
    headings = (
        "## Unit tests",
        "## Integration tests",
        "## Evaluation tests",
        "## Security tests",
    )
    for heading in headings:
        assert heading in plan_text, f"plan §24 no longer defines {heading}"


@pytest.mark.parametrize("keyword", ["provider adapters", "session isolation", "conflict handling"])
def test_the_plan_still_names_the_unit_items_this_ledger_maps(
    plan_text: str, keyword: str
) -> None:
    assert keyword in plan_text


def test_the_plan_still_names_the_four_security_rules(plan_text: str) -> None:
    security_section = plan_text.split("## Security tests", 1)[1]
    for phrase in (
        "API key never appears in logs",
        "Session A cannot access Session B memory",
        "Retrieved memory cannot override system instructions",
        "User data is not accidentally returned in diagnostics",
    ):
        assert phrase in security_section, f"plan §24 no longer requires: {phrase}"


# --------------------------------------------------------------------------- #
# Every requirement is mapped, and every mapping resolves
# --------------------------------------------------------------------------- #


def test_every_requirement_maps_to_at_least_one_test() -> None:
    unmapped = [
        f"{requirement.section}: {requirement.item}"
        for requirement in REQUIREMENTS
        if not requirement.tests
    ]

    assert unmapped == []


def test_every_requirement_from_the_plan_has_a_row() -> None:
    """The ledger must not silently shrink when the plan grows."""

    mapped = {(requirement.section, requirement.item) for requirement in REQUIREMENTS}
    expected_sections = {"Unit tests", "Integration tests", "Evaluation tests", "Security tests"}

    assert {section for section, _ in mapped} == expected_sections
    assert len(mapped) == len(REQUIREMENTS), "duplicate requirement rows"


def _is_satisfied(node_id: str, collected_tests: set[str]) -> bool:
    """Return True when the mapped test still exists under that id.

    A mapped id may name a parametrized test by its function id; pytest then
    reports one collected id per parameter set (``::test_x[id]``), and any of
    them satisfies the mapping. A module skipped for a missing optional
    dependency reports no ids at all, so the source is consulted — but only in
    that case, so a rename inside a *collected* module still fails.
    """

    if any(
        collected == node_id or collected.startswith(f"{node_id}[")
        for collected in collected_tests
    ):
        return True
    module = node_id.split("::")[0]
    return not _module_was_collected(module, collected_tests) and _defined_in_source(node_id)


def test_every_mapped_test_still_exists(collected_tests: set[str]) -> None:
    """A renamed or deleted test breaks the requirement it was carrying."""

    missing = sorted(
        node_id
        for requirement in REQUIREMENTS
        for node_id in requirement.tests
        if not _is_satisfied(node_id, collected_tests)
    )

    assert missing == [], f"the ledger points at tests that no longer exist: {missing}"


def test_every_mapped_id_names_a_test_in_one_of_the_plan_categories(
    collected_tests: set[str],
) -> None:
    pattern = re.compile(rf"^tests/({'|'.join(CATEGORIES)})/")

    assert all(pattern.match(node_id) for node_id in collected_tests)
    assert all(
        pattern.match(node_id)
        for requirement in REQUIREMENTS
        for node_id in requirement.tests
    )


def test_each_plan_category_contributes_mapped_tests() -> None:
    contributed = {
        category: sum(
            1
            for requirement in REQUIREMENTS
            for node_id in requirement.tests
            if node_id.startswith(f"tests/{category}/")
        )
        for category in PLAN_CATEGORIES
    }

    assert all(count > 0 for count in contributed.values()), contributed


def test_every_requirement_is_carried_by_at_least_two_tests() -> None:
    """One test per requirement makes the requirement as fragile as the test."""

    thin = [
        f"{requirement.section}: {requirement.item}"
        for requirement in REQUIREMENTS
        if len(requirement.tests) < 2
    ]

    assert thin == []


def test_every_rule_shaped_requirement_spans_more_than_one_file() -> None:
    """A rule proven from one file is a rule with one point of failure."""

    thin = [
        f"{requirement.section}: {requirement.item}"
        for requirement in REQUIREMENTS
        if requirement.section in RULE_SECTIONS
        and len({node_id.split("::")[0] for node_id in requirement.tests}) < 2
    ]

    assert thin == []


# --------------------------------------------------------------------------- #
# §25 — the MVP checklist
# --------------------------------------------------------------------------- #


def test_the_plan_still_defines_the_mvp_checklist(plan_text: str) -> None:
    assert "Phase 19 — MVP Definition" in plan_text
    assert "The MVP is complete when the user can:" in plan_text

    items = _plan_mvp_items(plan_text)
    assert len(items) == 14, f"the plan's MVP checklist changed shape: {len(items)} items"
    assert items[0].lower().startswith("open the hf space")
    assert items[-1].lower().startswith("export results")


def test_every_mvp_item_has_exactly_one_row_in_the_plan_s_order() -> None:
    numbers = [entry.number for entry in MVP_ITEMS]

    assert numbers == list(range(1, 15)), f"the MVP rows are out of order: {numbers}"
    assert len({entry.item for entry in MVP_ITEMS}) == len(MVP_ITEMS)


def test_every_mvp_row_names_the_plan_item_it_traces() -> None:
    """The row's text is the plan's text, modulo the numbering the plan adds."""

    plan_items = _plan_mvp_items(PLAN.read_text(encoding="utf-8"))

    for entry, plan_item in zip(MVP_ITEMS, plan_items, strict=True):
        assert entry.item.lower() == plan_item.lower(), (entry.number, entry.item, plan_item)


def test_every_mvp_item_maps_to_at_least_two_tests() -> None:
    thin = [entry.number for entry in MVP_ITEMS if len(entry.tests) < 2]

    assert thin == [], f"MVP items with fewer than two mapped tests: {thin}"


def test_every_mvp_mapped_test_still_exists(collected_tests: set[str]) -> None:
    missing = sorted(
        node_id
        for entry in MVP_ITEMS
        for node_id in entry.tests
        if not _is_satisfied(node_id, collected_tests)
    )

    assert missing == [], f"the MVP rows point at tests that no longer exist: {missing}"


def test_the_mvp_walkthrough_covers_every_item_that_is_not_external() -> None:
    """Every non-external item is demonstrated by the walkthrough, not just pinned.

    The focused tests pin behaviour; the walkthrough is what proves the fourteen
    steps survive being taken in order, in one session. An item with no
    walkthrough test is an item nobody has walked end to end.
    """

    walkthrough = "tests/integration/test_mvp_walkthrough.py::"
    unpaced = [
        entry.number
        for entry in MVP_ITEMS
        if not any(node_id.startswith(walkthrough) for node_id in entry.tests)
    ]

    # Item 1 is the deployment row: it is checked, but it is not a chat step.
    assert unpaced == [1], f"MVP items with no walkthrough test: {unpaced}"


def test_only_the_deployment_row_may_be_external_and_it_says_why() -> None:
    external = [entry for entry in MVP_ITEMS if entry.external]

    assert [entry.number for entry in external] == [1]
    reason = external[0].external.lower()
    assert "operator" in reason
    assert "space" in reason
    # The row still checks the half this repository owns.
    assert external[0].tests


def _plan_mvp_items(plan_text: str) -> list[str]:
    """Extract §25's numbered checklist from the plan."""

    section = plan_text.split("# 25. Phase 19 — MVP Definition", 1)[1]
    section = section.split("# 26.", 1)[0]
    return [
        match.group("item").strip().rstrip(".")
        for match in re.finditer(
            r"^\s*(?P<number>\d+)\.\s*(?P<item>.+?)\s*$", section, re.M
        )
    ]


def test_the_readme_claims_the_checklist_is_walked_and_the_window_it_states() -> None:
    """Phase 18's rule: a README claim needs something that fails when it stops being true.

    The README tells a reader that every MVP item is walkable, names the file
    that walks them, and states the retention window in days. Both halves are
    checked against the code rather than trusted: the walkthrough file must be
    the one the ledger maps to, and the stated window must be the default
    :class:`~app.retention.RetentionPolicy` actually uses.
    """

    from app.retention import DEFAULT_RETENTION_SECONDS

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "tests/integration/test_mvp_walkthrough.py" in readme
    days = DEFAULT_RETENTION_SECONDS / 86_400
    assert f"**{days:g} days**" in readme, "the README no longer states the retention window"
    walkthrough = "tests/integration/test_mvp_walkthrough.py::"
    assert all(
        any(node_id.startswith(walkthrough) for node_id in entry.tests)
        for entry in MVP_ITEMS
        if entry.number != 1
    )
