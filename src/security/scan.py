"""Artifact scanner: "no credential in any artifact" as a repeatable check.

Every phase of this project asserts, in tests, that a particular artifact is
key-free. Tests are the right place for that, but they only cover the artifacts
a test wrote. This module is the operational half: point it at whatever a run
produced — ``results/``, an exported session JSON, the SQLite database, a log
file — and it says whether a credential is in there.

```bash
python -m security.scan results/ data/brainos_lab.sqlite3
python -m security.scan results/run.json --secrets-env OPENAI_API_KEY
```

Two detection layers, because they fail differently:

* **structure** — any JSON field whose *name* is a secret name
  (``api_key``, ``authorization``, ``token``, …) and whose value is non-empty.
  A field name survives serialization even when a pattern does not, so this is
  the layer that catches a design mistake rather than a pasted string.
* **content** — credential-shaped strings (``sk-…``, ``ghp_…``, ``AKIA…``,
  ``Bearer …``, ``api_key=…``, PEM private-key blocks) anywhere in the file's
  text or raw bytes, plus an exact match against any secret the caller supplies
  by environment variable.

The ``name=value`` pattern is the noisy one, so it is filtered: a value that is a
placeholder word, a function call, or an all-lowercase identifier is not reported
(see :func:`_assignment_holds_a_credential`). Precision matters more than recall
on that one pattern, because a report full of source-code false positives gets
ignored — and the exact-match layer covers any real credential whose shape is
unremarkable.

Findings never reproduce what they found. A finding names the file, the JSON
pointer or byte offset, the category, and a masked shape
(:func:`security.guard.mask_secret`), so a scan report can itself be committed
or pasted into an issue without becoming the leak it describes.

Exit codes: ``0`` clean, ``1`` findings, ``2`` nothing could be scanned.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .guard import mask_secret

SCAN_VERSION = "scan-v1"

#: Field names that must never hold a value in a written artifact.
SECRET_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "api_secret",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "token",
        "access_token",
        "auth_token",
        "refresh_token",
        "client_secret",
        "private_key",
        "session_key",
    }
)

#: Credential shapes worth flagging in any text or byte stream.
CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("openai_key", re.compile(rb"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{7,}\b")),
    ("anthropic_key", re.compile(rb"\bsk-ant-[A-Za-z0-9_-]{10,}\b")),
    ("github_token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{16,}\b")),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("hf_token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("private_key_block", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "key_value_assignment",
        re.compile(
            rb"(?i)\b(?:api[_-]?key|authorization|password|secret|access[_-]?token|"
            rb"auth[_-]?token|client[_-]?secret|refresh[_-]?token)\b[\"']?\s*[=:]\s*"
            rb"[\"']?(?P<value>[A-Za-z0-9_\-./+=]{12,})"
        ),
    ),
    ("bearer_header", re.compile(rb"(?i)\bbearer\s+[A-Za-z0-9_\-./+=]{16,}")),
)

#: Extensions read as text and parsed as JSON when possible.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".json", ".jsonl", ".txt", ".log", ".md", ".csv", ".tsv", ".yaml", ".yml", ".ndjson"}
)
#: Extensions read as raw bytes only (a database is not text).
BINARY_SUFFIXES: frozenset[str] = frozenset({".sqlite3", ".db", ".sqlite", ".png", ".pdf"})
#: A single file larger than this is reported as skipped rather than read.
MAX_FILE_BYTES = 64 * 1024 * 1024

#: Values that read as prose, placeholders, or identifiers rather than secrets.
#: Documentation has to show *something* after ``api_key=``, and a scanner that
#: flags ``your-key-here`` trains everyone to ignore it.
_PLACEHOLDER_VALUE_RE = re.compile(
    rb"(?i)^(?:your|my|our|the|some|any|dummy|fake|example|sample|placeholder|"
    rb"provided|changeme|todo|tbd|xxx+|redacted|none|null|true|false|unknown|"
    rb"insert|replace|set|add|put|enter|fill)\b"
)


@dataclass(frozen=True)
class ScanFinding:
    """One credential-shaped thing found in one artifact."""

    path: str
    category: str
    #: JSON pointer (``$.config.api_key``) or ``bytes:<offset>`` for raw scans.
    pointer: str
    #: Masked shape only — never the value.
    preview: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "category": self.category,
            "pointer": self.pointer,
            "preview": self.preview,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScanReport:
    """What a scan covered and what it found."""

    files_scanned: int = 0
    bytes_scanned: int = 0
    files_skipped: tuple[str, ...] = ()
    findings: tuple[ScanFinding, ...] = field(default_factory=tuple)
    scan_version: str = SCAN_VERSION

    @property
    def clean(self) -> bool:
        return not self.findings

    def by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_version": self.scan_version,
            "files_scanned": self.files_scanned,
            "bytes_scanned": self.bytes_scanned,
            "files_skipped": list(self.files_skipped),
            "finding_count": len(self.findings),
            "clean": self.clean,
            "by_category": self.by_category(),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _normalize_field(name: Any) -> str:
    return str(name or "").strip().lower().replace("-", "_").replace(" ", "_")


def scan_value(
    value: Any,
    *,
    pointer: str = "$",
    secrets: Sequence[str] = (),
    path: str = "<memory>",
) -> list[ScanFinding]:
    """Walk a JSON-like value and report credentials by field name and by shape.

    ``secrets`` are exact values the caller knows are live (read from an
    environment variable, never from the command line). They are matched before
    the patterns, because an arbitrary key does not have to look like one.
    """

    findings: list[ScanFinding] = []
    exact = tuple(sorted({text for text in secrets if text}, key=len, reverse=True))
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{pointer}.{key}"
            name = _normalize_field(key)
            if name in SECRET_FIELD_NAMES:
                text = "" if item is None else str(item)
                if text.strip() and text.strip().lower() not in {"", "none", "null", "[redacted]"}:
                    findings.append(
                        ScanFinding(
                            path=path,
                            category="secret_field",
                            pointer=child,
                            preview=mask_secret(text),
                            detail=f"field named '{name}' holds a value",
                        )
                    )
            findings.extend(scan_value(item, pointer=child, secrets=exact, path=path))
        return findings
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            findings.extend(
                scan_value(item, pointer=f"{pointer}[{index}]", secrets=exact, path=path)
            )
        return findings
    if isinstance(value, str):
        findings.extend(_scan_text(value, pointer=pointer, secrets=exact, path=path))
    return findings


def _scan_text(
    text: str, *, pointer: str, secrets: Sequence[str], path: str
) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for secret in secrets:
        if secret and secret in text:
            findings.append(
                ScanFinding(
                    path=path,
                    category="known_secret",
                    pointer=pointer,
                    preview=mask_secret(secret),
                    detail="exact match against a live credential supplied via --secrets-env",
                )
            )
    encoded = text.encode("utf-8", errors="replace")
    findings.extend(_scan_bytes(encoded, pointer=pointer, path=path))
    return findings


def _assignment_holds_a_credential(match: re.Match[bytes]) -> bool:
    """Decide whether a ``name = value`` match holds a secret or is just code.

    The assignment pattern is the widest net here and the one most likely to land
    on something that is not a leak:

    * source code — ``api_key = api_key_from_environment(model)`` is a call, and
      ``secret=self._secrets()[0]`` is an expression;
    * documentation — ``api_key="your-key-here"`` and
      ``api_key="provided-for-this-session"`` are placeholders.

    Both are noise, so the value has to look generated: it must not be a
    placeholder word, must not be followed by ``(`` (a call), and must carry the
    digit-or-mixed-case mixture a real credential has. The trade is deliberate —
    an all-lowercase passphrase in a log would be missed by *shape*, which is why
    the exact-match layer (``--secrets-env``, never a command-line literal) and
    the field-name layer exist: those two do not depend on how a secret looks.
    """

    value = match.group("value") if "value" in match.groupdict() else b""
    if not value:
        return False
    tail = match.string[match.end() : match.end() + 1]
    if tail == b"(":  # a function call, not an assignment of a literal
        return False
    if _PLACEHOLDER_VALUE_RE.match(value):
        return False
    has_digit = any(b"0"[0] <= byte <= b"9"[0] for byte in value)
    has_upper = any(65 <= byte <= 90 for byte in value)
    has_lower = any(97 <= byte <= 122 for byte in value)
    return has_digit or (has_upper and has_lower)


def _scan_bytes(data: bytes, *, pointer: str, path: str) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for category, pattern in CREDENTIAL_PATTERNS:
        for match in pattern.finditer(data):
            if category == "key_value_assignment" and not _assignment_holds_a_credential(match):
                continue
            findings.append(
                ScanFinding(
                    path=path,
                    category=category,
                    pointer=f"{pointer}@{match.start()}",
                    preview=mask_secret(match.group(0).decode("utf-8", errors="replace")),
                    detail="credential-shaped string",
                )
            )
    return findings


#: Directory names skipped while expanding a path. Both hold *copies* of content
#: scanned elsewhere — version-control metadata and compiled bytecode — so
#: reporting them would double every finding without covering anything new.
SKIPPED_DIRECTORIES: frozenset[str] = frozenset({".git", "__pycache__"})

#: SQLite sidecars. Phase 14 puts the database in WAL mode, so the newest
#: committed rows live in ``<db>-wal`` until a checkpoint merges them: scanning
#: ``data/brainos_lab.sqlite3`` alone can report clean while the freshest content
#: sits beside it — and a copy of the directory ships the WAL with it. Naming a
#: database file therefore scans its sidecars too. (A directory argument already
#: picked them up as ordinary files; this closes the single-file case the
#: documented command uses.)
SQLITE_SIDECAR_SUFFIXES: tuple[str, ...] = ("-journal", "-shm", "-wal")


def _database_sidecars(path: Path) -> tuple[Path, ...]:
    """Return ``path``'s existing SQLite sidecars, in a stable order."""

    return tuple(
        sidecar
        for sidecar in (Path(f"{path}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES)
        if sidecar.is_file()
    )


def iter_files(paths: Iterable[Path | str]) -> Iterator[Path]:
    """Yield every regular file under ``paths``, directories expanded, sorted.

    Sorting matters: a scan report is compared between runs, and a filesystem
    walk order is not stable. A path that does not exist contributes nothing
    rather than raising, so ``python -m security.scan results/`` works before the
    first run has produced anything.
    """

    collected: list[Path] = []
    for entry in paths:
        candidate = Path(entry)
        if candidate.is_dir():
            collected.extend(item for item in candidate.rglob("*") if item.is_file())
        elif candidate.exists():
            collected.append(candidate)
            collected.extend(_database_sidecars(candidate))
    for item in sorted(set(collected)):
        if SKIPPED_DIRECTORIES.intersection(item.parts):
            continue
        yield item


def scan_file(path: Path, *, secrets: Sequence[str] = ()) -> tuple[list[ScanFinding], int, bool]:
    """Scan one file; return ``(findings, bytes_scanned, skipped)``."""

    try:
        size = path.stat().st_size
    except OSError:
        return [], 0, True
    if size > MAX_FILE_BYTES:
        return [], 0, True
    suffix = path.suffix.lower()
    try:
        raw = path.read_bytes()
    except OSError:
        return [], 0, True
    relative = str(path)
    if suffix in {".json"}:
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError):
            return _scan_text_findings(raw, relative, secrets), size, False
        return scan_value(payload, secrets=secrets, path=relative), size, False
    if suffix in {".jsonl", ".ndjson"}:
        findings: list[ScanFinding] = []
        for index, line in enumerate(
            raw.decode("utf-8", errors="replace").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                findings.extend(_scan_text_findings(line.encode("utf-8"), relative, secrets))
                continue
            findings.extend(
                scan_value(payload, pointer=f"$[{index}]", secrets=secrets, path=relative)
            )
        return findings, size, False
    if suffix in BINARY_SUFFIXES:
        return _scan_text_findings(raw, relative, secrets), size, False
    if suffix in TEXT_SUFFIXES or _looks_like_text(raw):
        return _scan_text_findings(raw, relative, secrets), size, False
    return _scan_text_findings(raw, relative, secrets), size, False


def _scan_text_findings(
    raw: bytes, path: str, secrets: Sequence[str]
) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for secret in secrets:
        if secret and secret.encode("utf-8") in raw:
            findings.append(
                ScanFinding(
                    path=path,
                    category="known_secret",
                    pointer="bytes",
                    preview=mask_secret(secret),
                    detail="exact match against a live credential supplied via --secrets-env",
                )
            )
    findings.extend(_scan_bytes(raw, pointer="bytes", path=path))
    return findings


def _looks_like_text(raw: bytes) -> bool:
    sample = raw[:4096]
    if not sample:
        return True
    return sample.count(b"\x00") == 0


def scan_paths(paths: Iterable[Path | str], *, secrets: Sequence[str] = ()) -> ScanReport:
    """Scan every file under ``paths`` and return one report."""

    findings: list[ScanFinding] = []
    scanned = 0
    total_bytes = 0
    skipped: list[str] = []
    for path in iter_files(paths):
        found, size, was_skipped = scan_file(path, secrets=secrets)
        if was_skipped:
            skipped.append(str(path))
            continue
        scanned += 1
        total_bytes += size
        findings.extend(found)
    return ScanReport(
        files_scanned=scanned,
        bytes_scanned=total_bytes,
        files_skipped=tuple(skipped),
        findings=tuple(findings),
    )


def secrets_from_environment(names: Iterable[str]) -> tuple[str, ...]:
    """Read credential values from the environment, without echoing them.

    A missing or empty variable contributes nothing; the function never raises,
    because a scan should still run when the variable it was told about is unset.
    """

    values: list[str] = []
    for name in names:
        key = str(name or "").strip()
        if not key:
            continue
        value = os.getenv(key, "")
        if value:
            values.append(value)
    return tuple(values)


def render_report(report: ScanReport) -> str:
    """A compact human-readable summary. Findings are already masked."""

    lines = [
        f"scan_version={report.scan_version} files={report.files_scanned} "
        f"bytes={report.bytes_scanned} findings={len(report.findings)} "
        f"clean={report.clean}"
    ]
    if report.files_skipped:
        lines.append(f"skipped: {len(report.files_skipped)} file(s) (unreadable or too large)")
    for category, count in report.by_category().items():
        lines.append(f"  {category:<22} {count}")
    for finding in report.findings[:25]:
        lines.append(
            f"  {finding.path} {finding.pointer} {finding.category} {finding.preview}"
        )
    if len(report.findings) > 25:
        lines.append(f"  … {len(report.findings) - 25} more")
    if report.clean:
        lines.append("No credential-shaped value or secret-named field was found.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan artifacts for credentials (plan Phase 13).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to scan (results/, exports, the SQLite database, logs).",
    )
    parser.add_argument(
        "--secrets-env",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Environment variable holding a live credential to match exactly. "
            "Repeatable. The value is never printed or written."
        ),
    )
    parser.add_argument("--output", type=Path, help="Write the report as JSON here.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--quiet", action="store_true", help="Print only the exit status line."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    secrets = secrets_from_environment(args.secrets_env or [])
    report = scan_paths(args.paths, secrets=secrets)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif not args.quiet:
        print(render_report(report))
    else:
        print(f"findings={len(report.findings)} clean={report.clean}")
    if report.files_scanned == 0:
        print("Nothing could be scanned: check the paths.", flush=True)
        return 2
    return 0 if report.clean else 1


__all__ = [
    "CREDENTIAL_PATTERNS",
    "MAX_FILE_BYTES",
    "SCAN_VERSION",
    "SECRET_FIELD_NAMES",
    "SKIPPED_DIRECTORIES",
    "ScanFinding",
    "ScanReport",
    "build_parser",
    "iter_files",
    "main",
    "render_report",
    "scan_file",
    "scan_paths",
    "scan_value",
    "secrets_from_environment",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
