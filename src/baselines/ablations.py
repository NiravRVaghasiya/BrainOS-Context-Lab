"""BrainOS ablation profiles (plan Phase 11).

The plan's ablation study asks which BrainOS components are responsible for any
observed improvement::

    BrainOS full
    - no temporal signal
    - no working memory
    - no consolidation
    - no relevance filtering
    - no conflict handling
    - no memory

with the constraint that only components actually available and stable in the
integrated BrainOS revision may be ablated. At the pinned revision
(``1d9eb7a``) that constraint removes two of the six:

* **Working memory** (``wm_slots``) is written by the cognitive cycle (topic,
  entities, recalled fact) but never read back by retrieval: ``recall()``
  scores the temporal store directly, and the application adapter never calls
  ``working_memory()``. Varying ``wm_slots`` changes trace snapshots only, so
  there is no retrieval ablation to run for it.
* **Consolidation** (``consolidate()`` / the ``consolidation`` feature flag)
  is an explicit offline pass the application never invokes; the
  observe → recall path runs no consolidation with or without the flag. A
  periodic-consolidation variant would be an *addition* to the pipeline, not an
  ablation of it.

Both exclusions are documented rather than silently dropped: an ablation that
cannot move a metric must not be reported as a null result.

The four runnable ablations are Mode D with exactly one component removed.
``brainos`` itself is the full reference — ablations are always compared
against it, never against each other:

* ``brainos_no_temporal`` — recency removed from application ranking and the
  runtime's temporal signals (``recency``, ``temporal_relevance``) ignored.
  The pinned runtime exposes no retrieval-weight configuration, so its
  internal top-k pre-ranking still uses temporal signals; the ablation removes
  temporal influence from the application's selection stage, which is where
  the prompt is decided. Conflict chronology still uses timestamps: ordering
  two contradictory claims belongs to conflict handling, not to ranking.
* ``brainos_no_relevance`` — the relevance floor and the relative tail trim
  disabled (``relevance_floor=0``, ``relative_relevance_ratio=0``). Every
  recalled memory reaches ranking; the token budget and the ``max_memories``
  cap still apply, so this measures the filter, not the budget.
* ``brainos_no_conflict`` — conflict resolution and staleness handling
  disabled: runtime contradiction/stale reports are ignored, the heuristic
  detector is off, and lifecycle status / validity windows are not honoured.
  Superseded memories are kept and contested pairs are never resolved.
* ``brainos_no_memory`` — Mode D's small window with no memory injected. This
  differs from Mode B on purpose: Mode B keeps eight turns, so B-vs-D mixes
  the window change with the memory change, while D4-vs-D isolates what the
  memory itself contributes on top of the window.

Every ablation keeps Mode D's history window and evidence budget
(see :mod:`baselines.modes`), keeps observing (observation is a side process
in every mode), and flows through the same service, builder, scorer, and
paired statistics as the baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brain.adapter import MemoryRecord

from .modes import (
    ABLATION_NO_CONFLICT,
    ABLATION_NO_MEMORY,
    ABLATION_NO_RELEVANCE,
    ABLATION_NO_TEMPORAL,
    ABLATION_ORDER,
    MODE_BRAINOS,
)

#: Runtime retrieval signals that carry temporal information. Ignored by the
#: ``no_temporal`` ablation so the temporal component cannot leak back into
#: ranking through the runtime blend (which weights ``recency`` at 0.10).
TEMPORAL_SIGNAL_NAMES: frozenset[str] = frozenset({"recency", "temporal_relevance"})


@dataclass(frozen=True)
class AblationProfile:
    """One removable component and how its removal is implemented.

    ``policy_overrides`` keys must be :class:`~app.state.ContextSettings`
    field names; they are applied by
    :meth:`~app.state.ContextSettings.with_mode_defaults` when the mode
    switches to this ablation, exactly like the mode's budget overrides.
    """

    ablation: str
    label: str
    description: str
    removes: str
    hypothesis: str
    policy_overrides: dict[str, Any] = field(default_factory=dict)
    ignore_runtime_conflicts: bool = False
    ignore_runtime_stale: bool = False
    strip_temporal_signals: bool = False
    strip_lifecycle: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready description for diagnostics and export."""

        return {
            "ablation": self.ablation,
            "label": self.label,
            "description": self.description,
            "removes": self.removes,
            "hypothesis": self.hypothesis,
            "policy_overrides": dict(self.policy_overrides),
            "ignore_runtime_conflicts": self.ignore_runtime_conflicts,
            "ignore_runtime_stale": self.ignore_runtime_stale,
            "strip_temporal_signals": self.strip_temporal_signals,
            "strip_lifecycle": self.strip_lifecycle,
        }


ABLATIONS: dict[str, AblationProfile] = {
    ABLATION_NO_TEMPORAL: AblationProfile(
        ablation=ABLATION_NO_TEMPORAL,
        label="Ablation D1 — BrainOS without temporal signals",
        description=(
            "Mode D with recency removed from application ranking and the "
            "runtime's recency/temporal_relevance signals ignored."
        ),
        removes="temporal_signal",
        hypothesis=(
            "Ranking without recency should keep recently-observed facts less "
            "reliably and may surface stale-but-lexically-strong memories."
        ),
        policy_overrides={"weight_recency": 0.0},
        strip_temporal_signals=True,
    ),
    ABLATION_NO_RELEVANCE: AblationProfile(
        ablation=ABLATION_NO_RELEVANCE,
        label="Ablation D2 — BrainOS without relevance filtering",
        description=(
            "Mode D with the relevance floor and relative tail trim disabled."
        ),
        removes="relevance_filtering",
        hypothesis=(
            "Without the filter, precision should fall (more irrelevant "
            "memories reach the budget) while recall stays flat or rises."
        ),
        policy_overrides={
            "relevance_floor": 0.0,
            "relative_relevance_ratio": 0.0,
        },
    ),
    ABLATION_NO_CONFLICT: AblationProfile(
        ablation=ABLATION_NO_CONFLICT,
        label="Ablation D3 — BrainOS without conflict handling",
        description=(
            "Mode D with conflict resolution and staleness handling disabled."
        ),
        removes="conflict_handling",
        hypothesis=(
            "Without conflict handling, corrected facts should reappear "
            "alongside their replacements and conflict-resolution accuracy "
            "should fall."
        ),
        policy_overrides={
            "resolve_conflicts": False,
            "drop_stale_memories": False,
            "heuristic_conflict_detection": False,
        },
        ignore_runtime_conflicts=True,
        ignore_runtime_stale=True,
        strip_lifecycle=True,
    ),
    ABLATION_NO_MEMORY: AblationProfile(
        ablation=ABLATION_NO_MEMORY,
        label="Ablation D4 — BrainOS window without memory",
        description="Mode D's window with no memory injected.",
        removes="memory",
        hypothesis=(
            "Without memory, distant facts should become unreachable while "
            "token cost stays near Mode D's — the window, not the memory, "
            "sets the price."
        ),
    ),
}

#: Retrieval-policy knobs at least one ablation overrides. When a session leaves
#: an ablation for a baseline, these are restored to their defaults: the knobs
#: are the ablation, not user tuning, so carrying them into ``brainos`` would
#: silently run a different system than the label claims. Baseline-to-baseline
#: switches keep the caller's tuning untouched.
POLICY_KNOB_FIELDS: tuple[str, ...] = tuple(
    sorted({key for profile in ABLATIONS.values() for key in profile.policy_overrides})
)

#: Components the plan lists that are not runnable at the pinned revision, with
#: the reason each is excluded instead of reported as a null result.
EXCLUDED_ABLATIONS: dict[str, str] = {
    "no working memory": (
        "The pinned runtime writes working memory (topic, entities, recalled "
        "fact) but retrieval never reads it back, and the application adapter "
        "never calls working_memory(). Varying wm_slots changes trace "
        "snapshots only."
    ),
    "no consolidation": (
        "Consolidation is an explicit offline pass (consolidate()) the "
        "application never invokes; the observe → recall path runs no "
        "consolidation either way. A periodic-consolidation variant would be "
        "an addition to the pipeline, not an ablation of it."
    ),
}


def is_ablation(value: Any) -> bool:
    """Return whether ``value`` names a runnable ablation."""

    return str(value or "").strip() in ABLATIONS


def ablation_profile(value: Any) -> AblationProfile:
    """Return the :class:`AblationProfile` for an ablation selector.

    Raises :class:`ValueError` for anything else, mirroring
    :func:`baselines.modes.resolve_mode`: an unknown ablation must surface as
    an error rather than silently run the full system.
    """

    text = str(value or "").strip()
    if text not in ABLATIONS:
        raise ValueError(
            f"Unknown ablation {str(value)!r}. Expected one of "
            f"{', '.join(ABLATION_ORDER)}."
        )
    return ABLATIONS[text]


def strip_temporal_signals(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Return copies of ``records`` without temporal retrieval signals.

    Only the ``recency`` / ``temporal_relevance`` keys are removed; every
    other signal still ranks. Used by the ``no_temporal`` ablation so the
    temporal component cannot leak back through the runtime blend.
    """

    from dataclasses import replace

    stripped: list[MemoryRecord] = []
    for record in records:
        signals = dict(record.signals or {})
        pruned = {
            name: value
            for name, value in signals.items()
            if name not in TEMPORAL_SIGNAL_NAMES
        }
        stripped.append(
            replace(record, signals=pruned) if len(pruned) != len(signals) else record
        )
    return stripped


def strip_lifecycle(records: list[MemoryRecord]) -> list[MemoryRecord]:
    """Return copies of ``records`` with lifecycle state neutralized.

    Status and validity windows are what let the retrieval pipeline honour
    supersession and expiry; clearing them (without touching text, type, or
    signals) is how the ``no_conflict`` ablation keeps superseded memories.
    """

    from dataclasses import replace

    return [
        (
            replace(record, status=None, valid_until=None)
            if record.status is not None or record.valid_until is not None
            else record
        )
        for record in records
    ]


def prepare_records(
    ablation: AblationProfile | str | None, records: list[MemoryRecord]
) -> list[MemoryRecord]:
    """Apply an ablation's record preparation, or return ``records`` unchanged.

    Accepts a profile, an ablation id, a baseline mode id, or ``None``: only a
    runnable ablation id changes anything, so the service can call this
    unconditionally with the active mode.
    """

    if ablation is None:
        return records
    profile = ablation if isinstance(ablation, AblationProfile) else ABLATIONS.get(
        str(ablation)
    )
    if profile is None:
        return records
    prepared = list(records)
    if profile.strip_temporal_signals:
        prepared = strip_temporal_signals(prepared)
    if profile.strip_lifecycle:
        prepared = strip_lifecycle(prepared)
    return prepared


def ablation_pairs(modes: Any) -> list[tuple[str, str]]:
    """Return ``(ablation, brainos)`` pairs for the ablations in ``modes``.

    Each ablation is compared against the full system, never against another
    ablation: the question is what the removed component contributed.
    """

    present = {str(mode) for mode in (modes or [])}
    if MODE_BRAINOS not in present:
        return []
    return [
        (ablation, MODE_BRAINOS) for ablation in ABLATION_ORDER if ablation in present
    ]


__all__ = [
    "ABLATIONS",
    "ABLATION_ORDER",
    "EXCLUDED_ABLATIONS",
    "POLICY_KNOB_FIELDS",
    "TEMPORAL_SIGNAL_NAMES",
    "AblationProfile",
    "ablation_pairs",
    "ablation_profile",
    "is_ablation",
    "prepare_records",
    "strip_lifecycle",
    "strip_temporal_signals",
]
