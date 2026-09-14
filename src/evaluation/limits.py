"""Cost controls for evaluation runs (plan Phase 15, delivered with Phase 9).

Users bring their own API key, so the application's job is to make a runaway run
impossible rather than to report the damage afterwards. Phase 8 flagged the gap:
*"Grading still needs Phase 15 cost controls before the Evaluation tab exposes a
run; ``--limit`` is the only control today."* Phase 9 puts a model in the loop,
so the controls ship with it.

Two objects, deliberately separate:

``RunLimits``
    A frozen set of ceilings. Immutable because a limit that can change
    mid-run is not a limit: a benchmark that raises its own ceiling partway
    through is a benchmark whose cost nobody can reproduce.

``RunBudget``
    The mutable accounting for one run. It is created once and shared by every
    mode and every trial, because a per-mode reset would let a five-mode
    experiment spend five times what the user configured.

Enforcement is layered by intent:

* a single prompt past ``max_input_tokens`` is **skipped** and recorded — it is a
  measurement ("this mode cannot fit at this length"), not an accident, and
  aborting would destroy the rest of the run;
* exhausting requests, total tokens, or the failure allowance **raises**
  :class:`BudgetExceeded` — continuing would spend money the user did not
  authorise;
* reported provider usage is what gets charged; when a provider reports
  nothing, the shared token counter's estimate is charged instead, because a
  silent zero would make an exhausted run look free.

The prompt ceiling is checked **before** the request is sent, so a refused
request costs nothing. A skip and a failure are counted separately for the same
reason: "the prompt was too large for the budget we set" and "the provider
errored" are different findings, and only the second one is a model failure.

Presets follow the plan's example budgets — Quick ``20 tasks × 3 modes``,
Standard ``100 tasks × 5 modes``, Research ``500 tasks × 5 modes × multiple
trials``. They are **ceilings, not estimates**: a run stops at the ceiling
rather than promising to reach it, and the artifact records both the limits and
what was actually spent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any

from baselines.modes import (
    MODE_BRAINOS,
    MODE_FULL_CONTEXT,
    MODE_ORDER,
    MODE_RAG,
    mode_label,
)
from brain.tokenizers import TokenCounter, estimate_tokens

#: Identifies the cost-control semantics recorded on a run.
LIMITS_VERSION = "limits-v1"

#: Hard-cap field names, used to build readable messages and to serialize.
LIMIT_FIELDS: tuple[str, ...] = (
    "max_tasks",
    "max_requests",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "max_generation_failures",
)


class BudgetExceeded(RuntimeError):
    """A run-level cost limit was reached and the run must stop.

    The message is safe to print and to store: it names the limit and the usage
    that hit it, never a request body, a prompt, or a credential.
    """

    def __init__(self, limit: str, *, allowed: int, used: int) -> None:
        super().__init__(f"Run budget exhausted: {limit} limit {allowed} reached (used {used}).")
        self.limit = limit
        self.allowed = allowed
        self.used = used


@dataclass(frozen=True)
class RunLimits:
    """The ceilings one run may not cross."""

    #: Benchmark examples (tasks) the run may execute.
    max_tasks: int | None = None
    #: Provider requests across every mode and trial.
    max_requests: int | None = None
    #: Tokens a *single* prompt may cost. Larger prompts are skipped, not sent.
    max_input_tokens: int | None = None
    #: Tokens a single completion may produce (also caps the request's max_tokens).
    max_output_tokens: int | None = None
    #: Input + output tokens across the whole run.
    max_total_tokens: int | None = None
    #: Consecutive-or-total generation failures tolerated before aborting.
    max_generation_failures: int | None = None
    #: Per-request transport timeout; combined with the model spec by ``capped``.
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        for name in LIMIT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if (
            not isinstance(self.timeout_seconds, (int, float))
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be greater than zero.")

    def capped(
        self, max_tokens: int | None, timeout_seconds: float
    ) -> tuple[int | None, float]:
        """Return ``(max_tokens, timeout_seconds)`` with this run's caps applied.

        The tighter of the two wins: a model spec that asks for 4096 output
        tokens under a 512-token run ceiling sends 512. ``None`` on either side
        means "no opinion" and the other value passes through.
        """

        effective_tokens = max_tokens
        if self.max_output_tokens is not None:
            effective_tokens = (
                self.max_output_tokens
                if effective_tokens is None
                else min(effective_tokens, self.max_output_tokens)
            )
        return effective_tokens, min(float(timeout_seconds), float(self.timeout_seconds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits_version": LIMITS_VERSION,
            **{name: getattr(self, name) for name in LIMIT_FIELDS},
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class RunPreset:
    """A named budget: which modes to run, how many trials, and under what limits."""

    name: str
    description: str
    modes: tuple[str, ...]
    trials: int
    limits: RunLimits

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "modes": list(self.modes),
            "mode_labels": [mode_label(mode) for mode in self.modes],
            "trials": self.trials,
            "limits": self.limits.to_dict(),
            "planned_requests": self.planned_requests,
        }

    @property
    def planned_requests(self) -> int | None:
        """Requests this preset aims for, or ``None`` when a limit is unbounded."""

        tasks = self.limits.max_tasks
        if tasks is None:
            return None
        return tasks * len(self.modes) * self.trials

    def with_overrides(self, **overrides: Any) -> RunPreset:
        """Return a preset with its limits replaced field by field."""

        unknown = set(overrides) - set(LIMIT_FIELDS)
        if unknown:
            raise ValueError(f"Unknown limit overrides: {', '.join(sorted(unknown))}.")
        return replace(self, limits=replace(self.limits, **overrides))


#: Quick keeps three modes: the reference (A), the strongest baseline (C), and
#: the system under test (D). Dropping Mode B is the cheapest honest cut —
#: the sliding window is a control, not a comparison of interest, and the
#: Standard tier runs all five.
_QUICK_MODES: tuple[str, ...] = (MODE_FULL_CONTEXT, MODE_RAG, MODE_BRAINOS)

PRESETS: dict[str, RunPreset] = {
    "quick": RunPreset(
        name="quick",
        description=(
            "20 tasks × 3 modes, one trial. A smoke budget for checking that a "
            "key, a model, and a dataset work together before spending more."
        ),
        modes=_QUICK_MODES,
        trials=1,
        limits=RunLimits(
            max_tasks=20,
            max_requests=60,
            max_input_tokens=16_000,
            max_output_tokens=512,
            max_total_tokens=250_000,
            max_generation_failures=5,
            timeout_seconds=60.0,
        ),
    ),
    "standard": RunPreset(
        name="standard",
        description="100 tasks × all five modes, one trial. The plan's standard run.",
        modes=MODE_ORDER,
        trials=1,
        limits=RunLimits(
            max_tasks=100,
            max_requests=500,
            max_input_tokens=64_000,
            max_output_tokens=1024,
            max_total_tokens=5_000_000,
            max_generation_failures=10,
            timeout_seconds=120.0,
        ),
    ),
    "research": RunPreset(
        name="research",
        description=(
            "500 tasks × all five modes × 3 trials. Repeated trials because "
            "stochastic sampling without repeats cannot support a claim."
        ),
        modes=MODE_ORDER,
        trials=3,
        limits=RunLimits(
            max_tasks=500,
            max_requests=7_500,
            max_input_tokens=200_000,
            max_output_tokens=2048,
            max_total_tokens=100_000_000,
            max_generation_failures=25,
            timeout_seconds=180.0,
        ),
    ),
}


def preset(name: str) -> RunPreset:
    """Return a named preset, rejecting unknown names loudly."""

    key = str(name or "").strip().lower()
    if key not in PRESETS:
        raise ValueError(
            f"Unknown limit preset {name!r}. Expected one of {', '.join(sorted(PRESETS))}."
        )
    return PRESETS[key]


@dataclass
class RunBudget:
    """Mutable accounting for one run, shared by every mode and trial.

    Every method is credential-blind: it counts tokens and requests, and it is
    never handed a prompt body or a key.
    """

    limits: RunLimits
    counter: TokenCounter = estimate_tokens
    counter_name: str = "estimate_tokens"
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    skips: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    #: Token counts already charged, for the JSON snapshot's remaining figures.
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def fits_prompt(self, prompt_tokens: int) -> tuple[bool, str]:
        """Return whether one prompt may be sent under the per-request ceiling.

        A refusal here is a *skip*: the caller records it and keeps going, so one
        oversized prompt cannot end a benchmark that is measuring how modes
        behave at increasing lengths.
        """

        ceiling = self.limits.max_input_tokens
        if ceiling is not None and prompt_tokens > ceiling:
            return False, (
                f"prompt_over_input_limit: {prompt_tokens} tokens exceeds the "
                f"run's max_input_tokens ceiling of {ceiling}"
            )
        return True, ""

    def check_request(self, prompt_tokens: int = 0) -> None:
        """Raise :class:`BudgetExceeded` when the run may not continue."""

        if self.limits.max_requests is not None and self.requests >= self.limits.max_requests:
            raise BudgetExceeded(
                "max_requests", allowed=self.limits.max_requests, used=self.requests
            )
        if (
            self.limits.max_generation_failures is not None
            and self.failures >= self.limits.max_generation_failures
        ):
            raise BudgetExceeded(
                "max_generation_failures",
                allowed=self.limits.max_generation_failures,
                used=self.failures,
            )
        if self.limits.max_total_tokens is not None:
            projected = self.total_tokens + max(0, int(prompt_tokens))
            if self.total_tokens >= self.limits.max_total_tokens:
                raise BudgetExceeded(
                    "max_total_tokens",
                    allowed=self.limits.max_total_tokens,
                    used=self.total_tokens,
                )
            # A prompt alone can exceed the remaining allowance; the completion
            # is unknown, so the check is conservative and the next one stops
            # the run rather than overshooting silently.
            if projected > self.limits.max_total_tokens:
                raise BudgetExceeded(
                    "max_total_tokens",
                    allowed=self.limits.max_total_tokens,
                    used=projected,
                )

    def charge(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_text: str = "",
        usage: Mapping[str, int] | None = None,
        failed: bool = False,
    ) -> dict[str, int]:
        """Record one request and return how many tokens it was charged."""

        reported_input = _int_field(usage, "prompt_tokens")
        reported_output = _int_field(usage, "completion_tokens")
        if reported_input is None:
            input_tokens = int(prompt_tokens or 0)
        else:
            input_tokens = reported_input
        if reported_output is None:
            output_tokens = self.counter(str(completion_text or ""))
        else:
            output_tokens = reported_output

        self.requests += 1
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        if failed:
            self.failures += 1
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}

    def record_skip(self) -> None:
        """Count a request that was refused before it was sent."""

        self.skips += 1

    def snapshot(self) -> dict[str, Any]:
        """Return the JSON-ready cost report for the run artifact."""

        limits = self.limits
        return {
            "limits_version": LIMITS_VERSION,
            "limits": limits.to_dict(),
            "token_counter": self.counter_name,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "failures": self.failures,
            "skips": self.skips,
            "remaining_requests": _remaining(limits.max_requests, self.requests),
            "remaining_total_tokens": _remaining(limits.max_total_tokens, self.total_tokens),
            "remaining_failures": _remaining(limits.max_generation_failures, self.failures),
            "within_limits": (
                _within(limits.max_requests, self.requests)
                and _within(limits.max_total_tokens, self.total_tokens)
                and _within(limits.max_generation_failures, self.failures)
            ),
        }

    def estimate_ceiling(self, requests: int) -> dict[str, Any]:
        """Return the worst-case spend for ``requests`` requests.

        "Where possible" from the plan: input tokens cannot be known before the
        prompts are built, so the estimate reports the output ceiling
        (``requests × max_output_tokens``) plus the run's hard caps, and marks
        the input side unknown rather than guessing low.
        """

        output_ceiling = None
        if self.limits.max_output_tokens is not None:
            output_ceiling = int(requests) * int(self.limits.max_output_tokens)
        return {
            "planned_requests": int(requests),
            "output_token_ceiling": output_ceiling,
            "input_tokens": "unknown before the prompts are built",
            "max_total_tokens": self.limits.max_total_tokens,
            "max_requests": self.limits.max_requests,
            "note": (
                "Ceilings, not estimates: the run stops at a limit instead of "
                "promising to reach it."
            ),
        }


def _int_field(usage: Mapping[str, int] | None, name: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _remaining(limit: int | None, used: int) -> int | None:
    if limit is None:
        return None
    return max(0, int(limit) - int(used))


def _within(limit: int | None, used: int) -> bool:
    if limit is None:
        return True
    return int(used) <= int(limit)


__all__ = [
    "LIMITS_VERSION",
    "LIMIT_FIELDS",
    "PRESETS",
    "BudgetExceeded",
    "RunBudget",
    "RunLimits",
    "RunPreset",
    "preset",
]
