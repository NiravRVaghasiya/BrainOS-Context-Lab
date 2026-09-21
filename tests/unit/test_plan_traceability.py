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

Two properties are deliberate:

* **It reads the plan.** If a future edit removes §24 or renames a category, the
  ledger's own premise is gone and this module says so instead of quietly
  checking a stale list.
* **It names tests that already exist.** Nothing here was written to make the
  ledger green; the mappings point at the tests that carry each guarantee today.
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


def test_every_mapped_test_still_exists(collected_tests: set[str]) -> None:
    """A renamed or deleted test breaks the requirement it was carrying.

    A mapped id may name a parametrized test by its function id; pytest then
    reports one collected id per parameter set (``::test_x[id]``), and any of
    them satisfies the mapping.
    """

    def satisfied(node_id: str) -> bool:
        if any(
            collected == node_id or collected.startswith(f"{node_id}[")
            for collected in collected_tests
        ):
            return True
        module = node_id.split("::")[0]
        # A module skipped for a missing optional dependency reports no ids at
        # all. Fall back to the source in that case — but only in that case, so
        # a rename inside a *collected* module still fails here.
        return not _module_was_collected(module, collected_tests) and _defined_in_source(
            node_id
        )

    missing = sorted(
        node_id
        for requirement in REQUIREMENTS
        for node_id in requirement.tests
        if not satisfied(node_id)
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
