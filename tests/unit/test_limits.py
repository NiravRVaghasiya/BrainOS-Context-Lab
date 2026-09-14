"""Phase 9 cost controls: the ceilings a run may not cross.

The tests are mostly about *refusals*, because that is what a budget is for: a
limit that is only recorded after the money is spent is a report, not a control.
"""

from __future__ import annotations

import pytest

from baselines.modes import MODE_BRAINOS, MODE_FULL_CONTEXT, MODE_ORDER, MODE_RAG
from evaluation.limits import (
    PRESETS,
    BudgetExceeded,
    RunBudget,
    RunLimits,
    preset,
)


def test_limits_reject_negative_or_non_integer_caps() -> None:
    with pytest.raises(ValueError):
        RunLimits(max_requests=-1)
    with pytest.raises(ValueError):
        RunLimits(max_total_tokens=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RunLimits(timeout_seconds=0)


def test_capped_takes_the_tighter_output_and_timeout() -> None:
    limits = RunLimits(max_output_tokens=512, timeout_seconds=30.0)

    assert limits.capped(4096, 60.0) == (512, 30.0)
    assert limits.capped(None, 60.0) == (512, 30.0)
    assert limits.capped(128, 10.0) == (128, 10.0)
    # No cap on either side leaves the model's own values alone.
    assert RunLimits().capped(4096, 60.0) == (4096, 60.0)


def test_presets_follow_the_plan_budgets() -> None:
    quick = preset("quick")
    standard = preset("standard")
    research = preset("research")

    assert quick.limits.max_tasks == 20
    assert quick.modes == (MODE_FULL_CONTEXT, MODE_RAG, MODE_BRAINOS)
    assert quick.trials == 1
    assert quick.planned_requests == 60

    assert standard.limits.max_tasks == 100
    assert standard.modes == MODE_ORDER
    assert standard.planned_requests == 500

    assert research.limits.max_tasks == 500
    assert research.trials == 3
    assert research.planned_requests == 7500


def test_unknown_preset_and_unknown_override_are_rejected() -> None:
    with pytest.raises(ValueError):
        preset("turbo")
    with pytest.raises(ValueError):
        preset("quick").with_overrides(max_steps=3)


def test_documented_presets_are_all_reachable() -> None:
    assert set(PRESETS) == {"quick", "standard", "research"}
    assert all(item.limits.max_total_tokens for item in PRESETS.values())


def test_budget_skips_an_oversized_prompt_without_charging() -> None:
    budget = RunBudget(RunLimits(max_input_tokens=100))

    fits, reason = budget.fits_prompt(101)

    assert fits is False
    assert "prompt_over_input_limit" in reason
    assert budget.requests == 0
    budget.record_skip()
    assert budget.snapshot()["skips"] == 1
    assert budget.snapshot()["within_limits"] is True


def test_budget_raises_when_the_request_allowance_is_gone() -> None:
    budget = RunBudget(RunLimits(max_requests=1))
    budget.charge(prompt_tokens=10, completion_text="done")

    with pytest.raises(BudgetExceeded) as error:
        budget.check_request(10)

    assert error.value.limit == "max_requests"
    assert "max_requests" in str(error.value)


def test_budget_raises_when_the_token_allowance_is_gone() -> None:
    budget = RunBudget(RunLimits(max_total_tokens=100))
    budget.charge(prompt_tokens=100, completion_text="")

    with pytest.raises(BudgetExceeded):
        budget.check_request(1)


def test_budget_raises_when_the_failure_allowance_is_used_up() -> None:
    budget = RunBudget(RunLimits(max_generation_failures=2))
    budget.charge(failed=True)
    budget.charge(failed=True)

    with pytest.raises(BudgetExceeded) as error:
        budget.check_request(0)

    assert error.value.limit == "max_generation_failures"


def test_budget_prefers_reported_usage_over_the_counter() -> None:
    budget = RunBudget(RunLimits(), counter=lambda text: 999)

    charged = budget.charge(
        prompt_tokens=10,
        completion_text="hello there",
        usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    )

    assert charged == {"input_tokens": 7, "output_tokens": 2}
    assert budget.total_tokens == 9


def test_budget_estimates_the_completion_when_usage_is_missing() -> None:
    budget = RunBudget(RunLimits(), counter=lambda text: len(text), counter_name="chars")

    charged = budget.charge(prompt_tokens=4, completion_text="abcd")

    assert charged == {"input_tokens": 4, "output_tokens": 4}
    assert budget.counter_name == "chars"


def test_snapshot_reports_remaining_allowances_and_limit_state() -> None:
    budget = RunBudget(RunLimits(max_requests=3, max_total_tokens=50))
    budget.charge(prompt_tokens=10, completion_text="")

    snapshot = budget.snapshot()

    assert snapshot["remaining_requests"] == 2
    assert snapshot["remaining_total_tokens"] == 40
    assert snapshot["within_limits"] is True
    assert snapshot["limits"]["limits_version"] == snapshot["limits_version"]


def test_unbounded_limits_report_no_remaining_figure() -> None:
    snapshot = RunBudget(RunLimits()).snapshot()

    assert snapshot["remaining_requests"] is None
    assert snapshot["remaining_total_tokens"] is None
    assert snapshot["within_limits"] is True


def test_estimate_ceiling_multiplies_only_the_known_side() -> None:
    budget = RunBudget(RunLimits(max_output_tokens=512, max_requests=60))

    estimate = budget.estimate_ceiling(60)

    assert estimate["planned_requests"] == 60
    assert estimate["output_token_ceiling"] == 60 * 512
    assert estimate["input_tokens"] == "unknown before the prompts are built"
