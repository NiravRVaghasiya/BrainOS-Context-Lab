"""Security findings: what a guard did, counted, reported, and never secret.

Phase 12 deliberately left the retrieval audit's ``suspicious`` drop reason
mapped to *no* error label: an injection quarantine is a security finding, not
an answer failure, and adding a tenth taxonomy label would have mixed the two
vocabularies. This module is where those findings are reported instead.

The shape is one vocabulary used by every surface:

```text
category   what the guard saw            (injection_pattern, credential_redacted, …)
action     what the guard did about it   (flagged, quarantined, neutralized, redacted, downgraded)
route      which path carried the text   (memory, chunk, history, current_message, persistence)
stage      where in the pipeline         (recall, selection, prompt, persistence, diagnostics)
families   the attack names matched      (instruction_override, delimiter_breakout, …)
```

Three rules the reporting layer exists to keep:

1. **A finding never carries the payload it caught, unbounded.** Previews are
   clipped and pass through :func:`security.guard.strip_invisible`; credentials
   are described by :func:`security.guard.mask_secret`, never reproduced. The
   ledger is rendered in a browser panel and written into evaluation artifacts,
   so it is held to the same standard as everything else on those surfaces.
2. **Counts are cumulative, detail is bounded.** A long session must not grow an
   unbounded report, so category counters always increase while the recorded
   findings list is capped at :data:`MAX_RECENT_FINDINGS`.
3. **Absence of findings is not a claim of safety.** Detection is pattern-based
   (see :mod:`security.guard`); a clean report says nothing matched, and the
   report says so in its own ``note`` field.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .guard import (
    NEUTRALIZED_CONTROL,
    NEUTRALIZED_DELIMITER,
    NEUTRALIZED_INVISIBLE,
    NEUTRALIZED_ROLE_PREFIX,
    mask_secret,
    strip_invisible,
)
from .scan import CREDENTIAL_PATTERNS

SECURITY_VERSION = "security-v1"

#: How many individual findings a ledger keeps for inspection. Totals are exact;
#: only the detail list is capped.
MAX_RECENT_FINDINGS = 50
#: Bound for a finding's preview text.
MAX_PREVIEW_CHARS = 160

# --- actions --------------------------------------------------------------- #
#: Detected and kept in the prompt (the default policy), so retrieval quality
#: stays measurable and the user can see what was caught.
ACTION_FLAGGED = "flagged"
#: Detected and removed before the prompt (``drop_suspicious_memories``).
ACTION_QUARANTINED = "quarantined"
#: Rewritten by the structural guard (delimiter, role prefix, invisible char).
ACTION_NEUTRALIZED = "neutralized"
#: A credential was replaced with ``[redacted]``.
ACTION_REDACTED = "redacted"
#: A message role was forced to ``user`` so history cannot outrank the system prompt.
ACTION_DOWNGRADED = "downgraded"

ACTIONS: tuple[str, ...] = (
    ACTION_FLAGGED,
    ACTION_QUARANTINED,
    ACTION_NEUTRALIZED,
    ACTION_REDACTED,
    ACTION_DOWNGRADED,
)

# --- categories ------------------------------------------------------------ #
CATEGORY_INJECTION = "injection_pattern"
#: The four structural categories are the guard's own vocabulary
#: (:mod:`security.guard`), imported rather than restated so the two modules
#: cannot drift apart.
CATEGORY_DELIMITER_BREAKOUT = NEUTRALIZED_DELIMITER
CATEGORY_ROLE_SMUGGLING = NEUTRALIZED_ROLE_PREFIX
CATEGORY_INVISIBLE_CHARACTERS = NEUTRALIZED_INVISIBLE
CATEGORY_CONTROL_CHARACTERS = NEUTRALIZED_CONTROL
CATEGORY_CREDENTIAL_REDACTED = "credential_redacted"
CATEGORY_SECRET_FIELD_DROPPED = "secret_field_dropped"
CATEGORY_ROLE_DOWNGRADED = "history_role_downgraded"

CATEGORIES: tuple[str, ...] = (
    CATEGORY_INJECTION,
    CATEGORY_DELIMITER_BREAKOUT,
    CATEGORY_ROLE_SMUGGLING,
    CATEGORY_INVISIBLE_CHARACTERS,
    CATEGORY_CONTROL_CHARACTERS,
    CATEGORY_CREDENTIAL_REDACTED,
    CATEGORY_SECRET_FIELD_DROPPED,
    CATEGORY_ROLE_DOWNGRADED,
)

#: Families that are structural breakouts rather than phrasing. They get their
#: own category because they are the attacks the delimiter guard exists for.
_STRUCTURAL_FAMILIES = frozenset({"delimiter_breakout", "role_smuggling"})

# --- routes ---------------------------------------------------------------- #
ROUTE_MEMORY = "memory"
ROUTE_CHUNK = "chunk"
ROUTE_HISTORY = "history"
ROUTE_CURRENT_MESSAGE = "current_message"
ROUTE_PERSISTENCE = "persistence"
ROUTE_DIAGNOSTICS = "diagnostics"
ROUTE_SESSION = "session"

ROUTES: tuple[str, ...] = (
    ROUTE_MEMORY,
    ROUTE_CHUNK,
    ROUTE_HISTORY,
    ROUTE_CURRENT_MESSAGE,
    ROUTE_PERSISTENCE,
    ROUTE_DIAGNOSTICS,
    ROUTE_SESSION,
)

# --- stages ---------------------------------------------------------------- #
STAGE_RECALL = "recall"
STAGE_SELECTION = "selection"
STAGE_PROMPT = "prompt"
STAGE_PERSISTENCE = "persistence"
STAGE_DIAGNOSTICS = "diagnostics"

STAGES: tuple[str, ...] = (
    STAGE_RECALL,
    STAGE_SELECTION,
    STAGE_PROMPT,
    STAGE_PERSISTENCE,
    STAGE_DIAGNOSTICS,
)

#: The standing caveat every report carries, so a clean block cannot be
#: misread as a proof of safety.
REPORT_NOTE = (
    "Counts describe what the guard matched and did. Detection is pattern-based "
    "and English-only: a clean report means nothing matched, not that the "
    "content was safe. The structural control (delimited blocks, guarded text, "
    "role constraints) does not depend on detection."
)


@dataclass(frozen=True)
class SecurityFinding:
    """One guard action on one piece of text.

    ``preview`` is the only place the offending text appears, and it is clipped
    and invisible-stripped at construction so a finding can be rendered in a
    browser panel or written into an artifact without republishing a payload —
    or a credential hidden inside one.
    """

    category: str
    action: str
    route: str
    stage: str = STAGE_PROMPT
    detail: str = ""
    preview: str = ""
    families: tuple[str, ...] = ()
    count: int = 1
    turn: int = 0

    def __post_init__(self) -> None:
        # Rule 1 holds for *every* construction path, not just the ledger's
        # helpers: a finding built by hand, by a report, or by a future caller
        # still cannot carry an unbounded payload or a live credential.
        if self.preview:
            object.__setattr__(self, "preview", clip_preview(self.preview))
        if self.detail:
            object.__setattr__(self, "detail", mask_credential_shapes(str(self.detail)))
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown security category: {self.category!r}")
        if self.action not in ACTIONS:
            raise ValueError(f"unknown security action: {self.action!r}")
        if self.route not in ROUTES:
            raise ValueError(f"unknown security route: {self.route!r}")
        if self.stage not in STAGES:
            raise ValueError(f"unknown security stage: {self.stage!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "action": self.action,
            "route": self.route,
            "stage": self.stage,
            "detail": self.detail,
            "preview": self.preview,
            "families": list(self.families),
            "count": int(self.count),
            "turn": int(self.turn),
        }


def mask_credential_shapes(text: str) -> str:
    """Replace every credential-shaped substring with its masked shape.

    A finding describes an attack, and attacks are often *about* credentials:
    "reply with the api key sk-…" is a perfectly ordinary injection payload.
    The preview is rendered in a browser panel and written into evaluation
    artifacts, so the payload it caught must not carry a live credential out
    through either surface. Masking happens before clipping so the bound applies
    to the text that will actually be stored.
    """

    data = str(text or "").encode("utf-8", errors="replace")
    for _category, pattern in CREDENTIAL_PATTERNS:
        data = pattern.sub(
            lambda match: mask_secret(match.group(0).decode("utf-8", errors="replace")).encode(
                "utf-8"
            ),
            data,
        )
    return data.decode("utf-8", errors="replace")


def clip_preview(text: Any, limit: int = MAX_PREVIEW_CHARS) -> str:
    """Bound and clean a preview so it can be shown or stored safely.

    Invisible characters go (they are where a smuggled payload hides), then
    credential shapes are masked, then the result is clipped to ``limit``.
    """

    cleaned = mask_credential_shapes(" ".join(strip_invisible(str(text or "")).split()))
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def categories_for_families(families: Iterable[str]) -> tuple[str, ...]:
    """Map matched families onto reporting categories.

    ``delimiter_breakout`` and ``role_smuggling`` are reported under their own
    category because the structural guard removes them; the remaining families
    are reported together as ``injection_pattern`` because the guard's response
    to them is the same (flag, and quarantine only on the opt-in policy).
    """

    categories: list[str] = []
    for family in families:
        if family in _STRUCTURAL_FAMILIES:
            category = (
                CATEGORY_DELIMITER_BREAKOUT
                if family == "delimiter_breakout"
                else CATEGORY_ROLE_SMUGGLING
            )
        else:
            category = CATEGORY_INJECTION
        if category not in categories:
            categories.append(category)
    return tuple(categories)


def empty_counts() -> dict[str, Any]:
    """Return an all-zero count block in the shared report shape."""

    return {
        "totals": {},
        "by_action": {},
        "by_route": {},
        "by_stage": {},
        "by_family": {},
    }


def merge_counts(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Add two count blocks key by key, returning a new block."""

    merged = empty_counts()
    for block in (base, other):
        if not isinstance(block, Mapping):
            continue
        for section in merged:
            values = block.get(section)
            if not isinstance(values, Mapping):
                continue
            for key, value in values.items():
                try:
                    amount = int(value)
                except (TypeError, ValueError):
                    continue
                target = merged[section]
                target[str(key)] = int(target.get(str(key), 0)) + amount
    return merged


class SecurityLedger:
    """Cumulative, thread-safe record of what a session's guards did.

    One ledger belongs to one session's service. Gradio runs callbacks in worker
    threads, so the counters are guarded by a lock; the report is a snapshot, so
    a reader never sees a half-updated count.
    """

    def __init__(self, *, session_id: str = "") -> None:
        self.session_id = str(session_id)
        self._lock = threading.Lock()
        self._counts: dict[str, Any] = empty_counts()
        self._findings: list[SecurityFinding] = []
        self._record_count = 0
        self._turn = 0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def set_turn(self, turn: int) -> None:
        """Stamp subsequent findings with the conversation turn they belong to."""

        with self._lock:
            self._turn = int(turn)

    def record(self, finding: SecurityFinding) -> SecurityFinding:
        """Add one finding to the ledger and return it."""

        stamped = finding if finding.turn else _with_turn(finding, self._turn)
        with self._lock:
            self._record_count += max(1, int(stamped.count))
            self._accumulate(stamped)
            self._findings.append(stamped)
            if len(self._findings) > MAX_RECENT_FINDINGS:
                del self._findings[: len(self._findings) - MAX_RECENT_FINDINGS]
        return stamped

    def record_many(self, findings: Iterable[SecurityFinding]) -> int:
        """Add several findings; return how many were recorded."""

        recorded = 0
        for finding in findings:
            self.record(finding)
            recorded += 1
        return recorded

    def count(
        self,
        category: str,
        action: str,
        route: str,
        *,
        stage: str = STAGE_PROMPT,
        families: Sequence[str] = (),
        count: int = 1,
        detail: str = "",
        preview: str = "",
    ) -> SecurityFinding | None:
        """Record ``count`` occurrences of one guard action.

        ``count <= 0`` records nothing and returns ``None``, so callers can pass
        a measured number straight through without an ``if``.
        """

        if count <= 0:
            return None
        return self.record(
            SecurityFinding(
                category=category,
                action=action,
                route=route,
                stage=stage,
                detail=detail,
                preview=clip_preview(preview) if preview else "",
                families=tuple(families),
                count=int(count),
            )
        )

    def record_credential_redaction(
        self,
        route: str,
        *,
        stage: str = STAGE_PROMPT,
        count: int = 1,
        secret: str = "",
        detail: str = "",
    ) -> SecurityFinding | None:
        """Record that a live credential was removed from text.

        The credential itself is never stored — only its masked shape, which is
        enough to tell one key from another in a report without publishing either.
        """

        if count <= 0:
            return None
        description = detail or (
            f"session credential ({mask_secret(secret)}) replaced with [redacted]"
            if secret
            else "credential-shaped value replaced with [redacted]"
        )
        return self.record(
            SecurityFinding(
                category=CATEGORY_CREDENTIAL_REDACTED,
                action=ACTION_REDACTED,
                route=route,
                stage=stage,
                detail=description,
                count=int(count),
            )
        )

    def record_guard(
        self,
        *,
        route: str,
        families: Sequence[str],
        action: str = ACTION_FLAGGED,
        stage: str = STAGE_PROMPT,
        count: int = 1,
        detail: str = "",
        preview: str = "",
        neutralized: Sequence[str] = (),
    ) -> list[SecurityFinding]:
        """Record a detection, and the structural rewrites that came with it.

        ``neutralized`` names the categories the guard physically removed
        (``delimiter_breakout``, ``role_smuggling``, ``invisible_characters``,
        ``control_characters``), so one call reports both what was *seen* and
        what was *changed* — the distinction that decides whether a memory is
        still a candidate for quarantine.
        """

        if count <= 0:
            return []
        recorded: list[SecurityFinding] = []
        for category in categories_for_families(families) or (CATEGORY_INJECTION,):
            finding = self.record(
                SecurityFinding(
                    category=category,
                    action=action,
                    route=route,
                    stage=stage,
                    detail=detail or f"{', '.join(families)} detected in {route} text",
                    preview=clip_preview(preview) if preview else "",
                    families=tuple(families),
                    count=int(count),
                )
            )
            recorded.append(finding)
        for category in neutralized:
            if category not in CATEGORIES:
                continue
            recorded.append(
                self.record(
                    SecurityFinding(
                        category=category,
                        action=ACTION_NEUTRALIZED,
                        route=route,
                        stage=stage,
                        detail=f"{category} removed from {route} text",
                        preview=clip_preview(preview) if preview else "",
                        count=int(count),
                    )
                )
            )
        return recorded

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def _accumulate(self, finding: SecurityFinding) -> None:
        amount = max(1, int(finding.count))
        _bump(self._counts["totals"], finding.category, amount)
        _bump(self._counts["by_action"], finding.action, amount)
        _bump(self._counts["by_route"], finding.route, amount)
        _bump(self._counts["by_stage"], finding.stage, amount)
        for family in finding.families:
            _bump(self._counts["by_family"], family, amount)

    @property
    def finding_count(self) -> int:
        """Total occurrences recorded, capped detail list notwithstanding."""

        with self._lock:
            return self._record_count

    @property
    def clean(self) -> bool:
        with self._lock:
            return self._record_count == 0

    def findings(self) -> tuple[SecurityFinding, ...]:
        """The bounded detail list, oldest first."""

        with self._lock:
            return tuple(self._findings)

    def snapshot(self) -> dict[str, Any]:
        """Return the session-cumulative report in the shared shape."""

        with self._lock:
            counts = {section: dict(values) for section, values in self._counts.items()}
            recent = [finding.to_dict() for finding in self._findings]
            total = self._record_count
        counts["record_count"] = total
        counts["finding_count"] = len(recent)
        counts["clean"] = total == 0
        counts["security_version"] = SECURITY_VERSION
        counts["session_id"] = self.session_id
        counts["recent"] = recent
        counts["note"] = REPORT_NOTE
        return counts

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`snapshot`, so the ledger can be sanitized directly."""

        return self.snapshot()


def _with_turn(finding: SecurityFinding, turn: int) -> SecurityFinding:
    if not turn:
        return finding
    return SecurityFinding(
        category=finding.category,
        action=finding.action,
        route=finding.route,
        stage=finding.stage,
        detail=finding.detail,
        preview=finding.preview,
        families=finding.families,
        count=finding.count,
        turn=turn,
    )


def _bump(target: dict[str, int], key: str, amount: int) -> None:
    if not key:
        return
    target[key] = int(target.get(key, 0)) + int(amount)


@dataclass(frozen=True)
class PromptGuardReport:
    """What the context builder's guards did while assembling one prompt.

    The builder returns this instead of writing to a ledger: it is a pure
    function of its inputs, so an evaluation replay gets the same report a live
    session does, and the service decides where the findings are accumulated.

    Two distinctions the report keeps, because they decide what happens next:

    * **intent versus structure.** ``memory_families`` / ``chunk_families``
      count *intent*-bearing attack families (override, role assumption,
      exfiltration, policy bypass). Structural breakouts are counted separately
      in ``memory_neutralized`` / ``chunk_neutralized``: they were removed, so
      they are reported as neutralized rather than as a live threat, and they do
      not make a memory quarantine-worthy on their own.
    * **this prompt versus the session.** Everything here describes one
      :func:`~brain.context_builder.build_context` call. The ledger is what
      accumulates it across a session.

    ``history_credentials_redacted`` and ``current_message_redacted`` close the
    route Phase 3 left open: a credential pasted into a conversation used to be
    replayed verbatim inside the recent-history window, so it went back to the
    provider on every later turn. ``history_roles_downgraded`` counts transcript
    messages that arrived with a ``system``/``tool`` role and were sent as
    ``user`` instead, because only the application may speak as the system.
    """

    memory_families: Mapping[str, int] = field(default_factory=dict)
    chunk_families: Mapping[str, int] = field(default_factory=dict)
    memory_neutralized: Mapping[str, int] = field(default_factory=dict)
    chunk_neutralized: Mapping[str, int] = field(default_factory=dict)
    suspicious_memories: int = 0
    suspicious_chunks: int = 0
    quarantined_memories: int = 0
    flagged_candidates: int = 0
    history_credentials_redacted: int = 0
    history_messages_redacted: int = 0
    current_message_redacted: int = 0
    history_roles_downgraded: int = 0
    drop_suspicious_memories: bool = False

    @property
    def clean(self) -> bool:
        """Whether no guard fired at all while building this prompt."""

        return not any(
            [
                self.memory_families,
                self.chunk_families,
                self.memory_neutralized,
                self.chunk_neutralized,
                self.suspicious_memories,
                self.suspicious_chunks,
                self.quarantined_memories,
                self.history_credentials_redacted,
                self.current_message_redacted,
                self.history_roles_downgraded,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        counts = empty_counts()
        for finding in self.findings():
            _bump(counts["totals"], finding.category, finding.count)
            _bump(counts["by_action"], finding.action, finding.count)
            _bump(counts["by_route"], finding.route, finding.count)
            _bump(counts["by_stage"], finding.stage, finding.count)
            for family in finding.families:
                _bump(counts["by_family"], family, finding.count)
        counts["record_count"] = sum(int(value) for value in counts["totals"].values())
        counts["clean"] = self.clean
        counts["security_version"] = SECURITY_VERSION
        counts["note"] = REPORT_NOTE
        counts["detail"] = {
            "drop_suspicious_memories": bool(self.drop_suspicious_memories),
            "suspicious_memories": int(self.suspicious_memories),
            "suspicious_chunks": int(self.suspicious_chunks),
            "quarantined_memories": int(self.quarantined_memories),
            "flagged_candidates": int(self.flagged_candidates),
            "history_credentials_redacted": int(self.history_credentials_redacted),
            "history_messages_redacted": int(self.history_messages_redacted),
            "current_message_redacted": int(self.current_message_redacted),
            "history_roles_downgraded": int(self.history_roles_downgraded),
            "memory_families": _sorted_counts(self.memory_families),
            "chunk_families": _sorted_counts(self.chunk_families),
            "memory_neutralized": _sorted_counts(self.memory_neutralized),
            "chunk_neutralized": _sorted_counts(self.chunk_neutralized),
        }
        return counts

    def findings(self) -> tuple[SecurityFinding, ...]:
        """The findings this prompt produced, in a stable order."""

        found: list[SecurityFinding] = []
        quarantine = bool(self.drop_suspicious_memories)
        for family, amount in _sorted_counts(self.memory_families).items():
            found.append(
                SecurityFinding(
                    category=CATEGORY_INJECTION,
                    action=ACTION_QUARANTINED if quarantine else ACTION_FLAGGED,
                    route=ROUTE_MEMORY,
                    stage=STAGE_SELECTION,
                    detail=f"{family} detected in retrieved memory",
                    families=(family,),
                    count=int(amount),
                )
            )
        for family, amount in _sorted_counts(self.chunk_families).items():
            found.append(
                SecurityFinding(
                    category=CATEGORY_INJECTION,
                    action=ACTION_FLAGGED,
                    route=ROUTE_CHUNK,
                    stage=STAGE_SELECTION,
                    detail=f"{family} detected in a retrieved transcript chunk",
                    families=(family,),
                    count=int(amount),
                )
            )
        for category, amount in _sorted_counts(self.memory_neutralized).items():
            found.append(_neutralized_finding(category, ROUTE_MEMORY, int(amount)))
        for category, amount in _sorted_counts(self.chunk_neutralized).items():
            found.append(_neutralized_finding(category, ROUTE_CHUNK, int(amount)))
        if self.history_credentials_redacted:
            found.append(
                SecurityFinding(
                    category=CATEGORY_CREDENTIAL_REDACTED,
                    action=ACTION_REDACTED,
                    route=ROUTE_HISTORY,
                    stage=STAGE_PROMPT,
                    detail="session credential removed from replayed history",
                    count=int(self.history_credentials_redacted),
                )
            )
        if self.current_message_redacted:
            found.append(
                SecurityFinding(
                    category=CATEGORY_CREDENTIAL_REDACTED,
                    action=ACTION_REDACTED,
                    route=ROUTE_CURRENT_MESSAGE,
                    stage=STAGE_PROMPT,
                    detail="session credential removed from the current message",
                    count=int(self.current_message_redacted),
                )
            )
        if self.history_roles_downgraded:
            found.append(
                SecurityFinding(
                    category=CATEGORY_ROLE_DOWNGRADED,
                    action=ACTION_DOWNGRADED,
                    route=ROUTE_HISTORY,
                    stage=STAGE_PROMPT,
                    detail="history arrived with a system/tool role and was sent as user",
                    count=int(self.history_roles_downgraded),
                )
            )
        return tuple(found)


_NEUTRALIZED_DETAIL = {
    CATEGORY_DELIMITER_BREAKOUT: "delimiter or chat-template marker removed from retrieved text",
    CATEGORY_ROLE_SMUGGLING: "leading role prefix stripped from retrieved text",
    CATEGORY_INVISIBLE_CHARACTERS: "zero-width/bidi/tag characters removed from retrieved text",
    CATEGORY_CONTROL_CHARACTERS: "control characters removed from retrieved text",
}


def _neutralized_finding(category: str, route: str, count: int) -> SecurityFinding:
    return SecurityFinding(
        category=category if category in CATEGORIES else CATEGORY_DELIMITER_BREAKOUT,
        action=ACTION_NEUTRALIZED,
        route=route,
        stage=STAGE_PROMPT,
        detail=_NEUTRALIZED_DETAIL.get(category, f"{category} removed from {route} text"),
        count=count,
    )


def _sorted_counts(values: Mapping[str, int] | None) -> dict[str, int]:
    if not isinstance(values, Mapping):
        return {}
    cleaned: dict[str, int] = {}
    for key, value in values.items():
        try:
            amount = int(value)
        except (TypeError, ValueError):
            continue
        if amount:
            cleaned[str(key)] = amount
    return dict(sorted(cleaned.items()))


def security_report(
    ledger: SecurityLedger | None,
    *,
    last_prompt: Mapping[str, Any] | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The session-level report the UI, the export, and diagnostics render.

    ``last_prompt`` is the most recent :class:`PromptGuardReport` (as a dict) so
    the panel can show "this turn" next to "this session" without the ledger
    having to remember turn boundaries.
    """

    counts = ledger.snapshot() if ledger is not None else empty_counts()
    if ledger is None:
        counts["record_count"] = 0
        counts["finding_count"] = 0
        counts["clean"] = True
        counts["security_version"] = SECURITY_VERSION
        counts["recent"] = []
        counts["note"] = REPORT_NOTE
    counts["last_prompt"] = dict(last_prompt) if isinstance(last_prompt, Mapping) else {}
    if isinstance(policy, Mapping):
        counts["policy"] = {
            "drop_suspicious_memories": bool(policy.get("drop_suspicious_memories", False)),
            "neutralize_text": bool(policy.get("neutralize_text", True)),
        }
    return counts


def summarize_security(blocks: Iterable[Any]) -> dict[str, Any]:
    """Aggregate per-record security blocks across a run or an experiment.

    Accepts :class:`PromptGuardReport` objects, their dicts, ledger snapshots,
    or ``None``/missing values, so an artifact written before Phase 13 sums to a
    clean zero block instead of failing. The result is the artifact-level answer
    to "did anything get quarantined or redacted while these numbers were
    produced?" — and, when the answer is no, the evidence that the run held no
    credential to leak.
    """

    merged = empty_counts()
    record_count = 0
    records_with_findings = 0
    quarantined = 0
    redacted = 0
    seen_families: set[str] = set()
    for block in blocks:
        record_count += 1
        payload: Mapping[str, Any] | None = None
        if isinstance(block, PromptGuardReport):
            payload = block.to_dict()
        elif isinstance(block, Mapping):
            payload = block
        if payload is None:
            continue
        totals = payload.get("totals") if isinstance(payload.get("totals"), Mapping) else {}
        by_family = (
            payload.get("by_family") if isinstance(payload.get("by_family"), Mapping) else {}
        )
        if totals or by_family:
            records_with_findings += 1
        seen_families.update(str(name) for name in by_family)
        merged = merge_counts(merged, payload)
        quarantined += int((payload.get("by_action") or {}).get(ACTION_QUARANTINED, 0) or 0)
        redacted += int(totals.get(CATEGORY_CREDENTIAL_REDACTED, 0) or 0)
    merged["record_count"] = record_count
    merged["records_with_findings"] = records_with_findings
    merged["quarantined"] = quarantined
    merged["credentials_redacted"] = redacted
    merged["families_seen"] = sorted(seen_families)
    merged["clean"] = records_with_findings == 0
    merged["security_version"] = SECURITY_VERSION
    merged["note"] = REPORT_NOTE
    return merged


__all__ = [
    "ACTIONS",
    "ACTION_DOWNGRADED",
    "ACTION_FLAGGED",
    "ACTION_NEUTRALIZED",
    "ACTION_QUARANTINED",
    "ACTION_REDACTED",
    "CATEGORIES",
    "CATEGORY_CONTROL_CHARACTERS",
    "CATEGORY_CREDENTIAL_REDACTED",
    "CATEGORY_DELIMITER_BREAKOUT",
    "CATEGORY_INVISIBLE_CHARACTERS",
    "CATEGORY_INJECTION",
    "CATEGORY_ROLE_DOWNGRADED",
    "CATEGORY_ROLE_SMUGGLING",
    "CATEGORY_SECRET_FIELD_DROPPED",
    "MAX_PREVIEW_CHARS",
    "MAX_RECENT_FINDINGS",
    "PromptGuardReport",
    "REPORT_NOTE",
    "ROUTES",
    "ROUTE_CHUNK",
    "ROUTE_CURRENT_MESSAGE",
    "ROUTE_DIAGNOSTICS",
    "ROUTE_HISTORY",
    "ROUTE_MEMORY",
    "ROUTE_PERSISTENCE",
    "ROUTE_SESSION",
    "SECURITY_VERSION",
    "STAGES",
    "STAGE_DIAGNOSTICS",
    "STAGE_PERSISTENCE",
    "STAGE_PROMPT",
    "STAGE_RECALL",
    "STAGE_SELECTION",
    "SecurityFinding",
    "SecurityLedger",
    "categories_for_families",
    "clip_preview",
    "mask_credential_shapes",
    "empty_counts",
    "merge_counts",
    "security_report",
    "summarize_security",
]
