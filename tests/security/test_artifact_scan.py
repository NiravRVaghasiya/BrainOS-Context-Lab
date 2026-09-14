"""The artifact scanner: "no credential in any artifact" as a runnable check.

Every other test in this directory asserts that *one* artifact a test wrote is
key-free. These tests cover the operational half — point the scanner at whatever
a run produced and get an answer — plus the two properties that decide whether
anybody trusts that answer:

* **it reports what it covers.** ``files_scanned``, ``bytes_scanned``, and
  ``files_skipped`` are in the report, so "clean" always comes with a scope. A
  scan that silently read nothing must not be able to say ``clean=True``.
* **a finding never republishes the credential.** Previews are masked shapes, so
  the report can be committed, pasted into an issue, or printed in CI without
  becoming the leak it describes.

Fake credentials here are deliberately fixture-shaped; scanning ``tests/`` is
expected to find them, which is why the repository-clean assertion at the bottom
covers shipped surfaces only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from security import scan
from security.scan import (
    MAX_FILE_BYTES,
    SCAN_VERSION,
    SECRET_FIELD_NAMES,
    SKIPPED_DIRECTORIES,
    ScanFinding,
    ScanReport,
    build_parser,
    iter_files,
    main,
    render_report,
    scan_file,
    scan_paths,
    scan_value,
    secrets_from_environment,
)

#: Credential-shaped fixtures. None of these is live; all of them must be caught.
OPENAI_SHAPED = "sk-" + "aB3xY9" * 5
ANTHROPIC_SHAPED = "sk-ant-api03-" + "Qw7eR5" * 5
GITHUB_SHAPED = "ghp_" + "A1b2C3d4E5f6G7h8I9j0"
AWS_SHAPED = "AKIA" + "IOSFODNN7EXAMPLE"
GOOGLE_SHAPED = "AIza" + "Sy1Bj2Kk3Ll4Mm5Nn6Oo7Pp8Qq9Rr0Ss1T"
HF_SHAPED = "hf_" + "Xc1Vb2Nn3Mm4Qq5Ww6Ee"
SLACK_SHAPED = "xoxb-123456789012-abcdef"
PEM_SHAPED = "-----BEGIN RSA PRIVATE KEY-----"
BEARER_SHAPED = "Bearer zz99YY88xx77WW66vv55"
#: A live-looking secret with no recognizable shape: only an exact match finds it.
PASSPHRASE = "correct-horse-battery-staple-9"

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Surfaces this project ships: source, docs, the plan, benchmark data, and
#: the Phase 14 HF Space deployment files (``app.py``, ``requirements.txt``,
#: ``packages.txt``, the Space README). Anything added to the repository and
#: shipped to visitors must be added here — the test is the assertion that the
#: artifact scanner actually covers the tree a Space will build from.
SHIPPED_PATHS = (
    "src",
    "docs",
    "README.md",
    "CONTEXT.md",
    "benchmarks",
    "app.py",
    "requirements.txt",
    "packages.txt",
    "pyproject.toml",
    "BrainOS_Context_Lab_Implementation_Plan.md",
)


def raw_values(report: ScanReport) -> str:
    """The whole report as one string, for "the secret is not in here" checks."""

    return json.dumps(report.to_dict(), sort_keys=True)


# --------------------------------------------------------------------------- #
# Layer 1: a field whose name is a secret name
# --------------------------------------------------------------------------- #


def test_a_secret_named_field_holding_a_value_is_reported_by_pointer() -> None:
    findings = scan_value({"config": {"provider": {"api_key": OPENAI_SHAPED}}})

    assert len(findings) >= 1
    secret_field = [item for item in findings if item.category == "secret_field"]
    assert secret_field[0].pointer == "$.config.provider.api_key"
    assert "field named 'api_key'" in secret_field[0].detail


def test_the_name_layer_does_not_depend_on_how_the_value_looks() -> None:
    """A design mistake is caught by the name even when the value is ordinary."""

    findings = scan_value({"authorization": "a-plain-session-cookie-value"})

    assert [item.category for item in findings] == ["secret_field"]
    assert findings[0].pointer == "$.authorization"


@pytest.mark.parametrize("name", sorted(SECRET_FIELD_NAMES))
def test_every_documented_secret_field_name_is_covered(name: str) -> None:
    assert scan_value({name: "some-value-that-is-not-empty"})


@pytest.mark.parametrize("name", ["API-Key", "Authorization", "client secret", "TOKEN"])
def test_field_names_are_normalized_before_they_are_judged(name: str) -> None:
    findings = scan_value({name: "Zx9cvbnmLKJHGFDSAQ12"})

    assert [item.category for item in findings] == ["secret_field"]
    assert name.lower().replace("-", "_").replace(" ", "_") in findings[0].detail


@pytest.mark.parametrize("value", ["", "   ", None, "none", "null", "[redacted]"])
def test_an_empty_or_already_redacted_secret_field_is_not_a_finding(value: object) -> None:
    """Redaction is the goal; flagging it would make the report punish success."""

    assert scan_value({"api_key": value}) == []


def test_an_innocent_field_name_is_not_reported_by_the_name_layer() -> None:
    assert scan_value({"model": "gpt-4o-mini", "turns": 12, "notes": "all good"}) == []


def test_lists_are_walked_with_their_indices_in_the_pointer() -> None:
    findings = scan_value({"items": [{"ok": True}, {"token": "Zx9cvbnmLKJHGFDSAQ12"}]})

    assert findings[0].pointer == "$.items[1].token"


def test_a_nested_value_is_scanned_by_both_layers() -> None:
    payload = {"runs": [{"provider": {"api_key": OPENAI_SHAPED}}]}

    findings = scan_value(payload)
    categories = {item.category for item in findings}

    assert "secret_field" in categories, "the field name is the design mistake"
    assert "openai_key" in categories, "the value's shape is the leak"


# --------------------------------------------------------------------------- #
# Layer 2: credential shapes in text and raw bytes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "category"),
    [
        (OPENAI_SHAPED, "openai_key"),
        (ANTHROPIC_SHAPED, "anthropic_key"),
        (GITHUB_SHAPED, "github_token"),
        (AWS_SHAPED, "aws_access_key"),
        (GOOGLE_SHAPED, "google_api_key"),
        (HF_SHAPED, "hf_token"),
        (SLACK_SHAPED, "slack_token"),
        (PEM_SHAPED, "private_key_block"),
        (BEARER_SHAPED, "bearer_header"),
        ("api_key=Sk1Live99887766abcd", "key_value_assignment"),
    ],
)
def test_a_credential_shape_is_reported_wherever_it_appears(
    value: str, category: str
) -> None:
    findings = scan_value({"notes": f"the run used {value} and then stopped"})

    assert category in {item.category for item in findings}, findings


def test_a_shapeless_live_secret_is_found_by_exact_match(tmp_path: Path) -> None:
    """The exact-match layer exists because shapes are not dependable."""

    target = tmp_path / "session.log"
    target.write_text(f"turn 3 leaked {PASSPHRASE} inside the transcript\n", encoding="utf-8")

    report = scan_paths([target], secrets=(PASSPHRASE,))

    assert [item.category for item in report.findings] == ["known_secret"]
    assert report.findings[0].pointer == "bytes"
    assert "--secrets-env" in report.findings[0].detail


def test_a_finding_masks_the_credential_instead_of_repeating_it(tmp_path: Path) -> None:
    target = tmp_path / "leak.json"
    target.write_text(
        json.dumps({"api_key": OPENAI_SHAPED, "note": f"use {PASSPHRASE} here"}),
        encoding="utf-8",
    )

    report = scan_paths([target], secrets=(PASSPHRASE,))

    assert report.findings, "both layers should have fired"
    assert OPENAI_SHAPED not in raw_values(report)
    assert PASSPHRASE not in raw_values(report)
    previews = [item.preview for item in report.findings]
    assert any(item.startswith("sk-") for item in previews), previews
    assert any(item.startswith("cor") for item in previews), previews
    assert all("chars]" in item for item in previews), "a masked shape names the length"


def test_the_assignment_pattern_ignores_source_code_and_placeholders() -> None:
    """Precision on the widest pattern, or the whole report gets ignored."""

    noise = "\n".join(
        [
            "api_key = api_key_from_environment(model)",
            "secret=self._secrets()[0]",
            'api_key="your-key-here"',
            'api_key="provided-for-this-session"',
            '{"api_key": config.api_key}',
            "password: str = ''",
            "--api-key-env OPENAI_API_KEY",
        ]
    )

    assert scan_value({"code": noise}) == []


def test_the_assignment_pattern_still_catches_a_generated_value() -> None:
    for text in (
        "api_key=Sk1Live99887766abcd",
        'api_key: "AbCdEf1234567890"',
        "AUTHORIZATION: Bearer zz99YY88xx77WW66vv55",
        "password=hunter2X99887766",
    ):
        assert scan_value({"line": text}), text


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def test_a_json_file_is_parsed_so_findings_carry_pointers(tmp_path: Path) -> None:
    target = tmp_path / "run.json"
    target.write_text(json.dumps({"config": {"api_key": OPENAI_SHAPED}}), encoding="utf-8")

    findings, size, skipped = scan_file(target)

    assert skipped is False
    assert size == target.stat().st_size
    assert "$.config.api_key" in {item.pointer for item in findings}


def test_a_jsonl_file_reports_the_line_it_found(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"
    target.write_text(
        '{"turn": 1}\n\n{"turn": 2, "api_key": "Zx9cvbnmLKJHGFDSAQ12"}\n',
        encoding="utf-8",
    )

    findings, _size, skipped = scan_file(target)

    assert skipped is False
    # Line numbering is 1-based over the raw file, blank lines included.
    assert [item.pointer for item in findings] == ["$[3].api_key"]


def test_a_malformed_json_file_falls_back_to_a_raw_byte_scan(tmp_path: Path) -> None:
    """A truncated artifact is the likeliest thing to scan; it must not be skipped."""

    target = tmp_path / "broken.json"
    target.write_text('{not json at all "api_key": "Qw9ertyUiop12345"}', encoding="utf-8")

    findings, _size, skipped = scan_file(target)

    assert skipped is False
    assert [item.category for item in findings] == ["key_value_assignment"]
    assert findings[0].pointer.startswith("bytes@")


def test_a_database_file_is_scanned_as_raw_bytes(tmp_path: Path) -> None:
    target = tmp_path / "brainos_lab.sqlite3"
    target.write_bytes(b"\x00\x01" + OPENAI_SHAPED.encode() + b"\x00\x02")

    findings, size, skipped = scan_file(target)

    assert skipped is False
    assert size == len(target.read_bytes())
    assert "openai_key" in {item.category for item in findings}


def test_a_log_file_and_an_unrecognized_text_file_are_both_scanned(
    tmp_path: Path,
) -> None:
    log = tmp_path / "run.log"
    log.write_text(f"POST 200 key={GITHUB_SHAPED}\n", encoding="utf-8")
    odd = tmp_path / "payload.weird"
    odd.write_text(f"authorization: {BEARER_SHAPED}\n", encoding="utf-8")

    assert "github_token" in {item.category for item in scan_file(log)[0]}
    assert "bearer_header" in {item.category for item in scan_file(odd)[0]}


def test_an_oversized_file_is_reported_as_skipped_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "huge.log"
    target.write_text("x" * 4096, encoding="utf-8")
    monkeypatch.setattr(scan, "MAX_FILE_BYTES", 1024)

    findings, size, skipped = scan_file(target)

    assert (findings, size, skipped) == ([], 0, True)
    assert MAX_FILE_BYTES == 64 * 1024 * 1024, "the real bound stays generous"


def test_an_unreadable_path_is_skipped_rather_than_raising(tmp_path: Path) -> None:
    assert scan_file(tmp_path) == ([], 0, True), "a directory cannot be read as a file"
    assert scan_file(tmp_path / "does-not-exist.log") == ([], 0, True)


# --------------------------------------------------------------------------- #
# Walking a tree
# --------------------------------------------------------------------------- #


def test_directories_are_expanded_sorted_and_deduplicated(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "two.json").write_text("{}", encoding="utf-8")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    single = tmp_path / "c.json"
    single.write_text("{}", encoding="utf-8")

    walked = list(iter_files([tmp_path, single, single]))

    assert walked == [tmp_path / "a.json", tmp_path / "b" / "two.json", single]


def test_version_control_and_bytecode_copies_are_not_scanned(tmp_path: Path) -> None:
    for name in sorted(SKIPPED_DIRECTORIES):
        (tmp_path / name).mkdir()
        (tmp_path / name / "copy.json").write_text("{}", encoding="utf-8")
    (tmp_path / "real.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in iter_files([tmp_path])] == ["real.json"]
    assert {".git", "__pycache__"} <= SKIPPED_DIRECTORIES


def test_a_path_that_does_not_exist_contributes_nothing(tmp_path: Path) -> None:
    assert list(iter_files([tmp_path / "no-such-dir", tmp_path / "nope.json"])) == []


def test_scan_paths_reports_what_it_covered(tmp_path: Path) -> None:
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"model": "gpt-4o-mini"}), encoding="utf-8")
    dirty = tmp_path / "dirty.json"
    dirty.write_text(json.dumps({"api_key": OPENAI_SHAPED}), encoding="utf-8")

    report = scan_paths([tmp_path])

    assert report.files_scanned == 2
    assert report.bytes_scanned == clean.stat().st_size + dirty.stat().st_size
    assert report.files_skipped == ()
    assert report.clean is False
    assert report.by_category() == {"openai_key": 1, "secret_field": 1}
    assert {item.path for item in report.findings} == {str(dirty)}


def test_skipped_files_appear_in_the_report_not_in_the_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = tmp_path / "small.json"
    small.write_text("{}", encoding="utf-8")
    big = tmp_path / "big.log"
    big.write_text("y" * 2048, encoding="utf-8")
    monkeypatch.setattr(scan, "MAX_FILE_BYTES", 1024)

    report = scan_paths([tmp_path])

    assert report.files_scanned == 1
    assert report.bytes_scanned == small.stat().st_size
    assert report.files_skipped == (str(big),)
    assert report.clean is True, "clean, but the scope says one file was not read"


def test_a_clean_directory_produces_a_clean_report(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"answer": "PostgreSQL 16"}), encoding="utf-8")

    report = scan_paths([tmp_path])

    assert report.clean is True
    assert report.findings == ()
    payload = report.to_dict()
    assert payload["scan_version"] == SCAN_VERSION
    assert payload["finding_count"] == 0
    assert payload["findings"] == []
    assert payload["by_category"] == {}


def test_a_scan_of_nothing_is_not_a_clean_scan(tmp_path: Path) -> None:
    """The distinction that keeps "clean" from being vacuous."""

    report = scan_paths([tmp_path / "missing-dir"])

    assert report.files_scanned == 0
    assert report.clean is True
    assert main([str(tmp_path / "missing-dir")]) == 2, "nothing could be scanned"


def test_a_finding_is_a_small_frozen_record() -> None:
    item = ScanFinding(path="a.json", category="secret_field", pointer="$.api_key", preview="sk-…")

    assert item.to_dict() == {
        "path": "a.json",
        "category": "secret_field",
        "pointer": "$.api_key",
        "preview": "sk-…",
        "detail": "",
    }
    with pytest.raises(Exception):
        item.path = "b.json"  # type: ignore[misc]


def test_an_empty_report_defaults_are_consistent() -> None:
    report = ScanReport()

    assert report.clean is True
    assert report.to_dict()["files_scanned"] == 0
    assert report.to_dict()["scan_version"] == SCAN_VERSION


# --------------------------------------------------------------------------- #
# Secrets from the environment, never from the command line
# --------------------------------------------------------------------------- #


def test_secrets_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROBE_LIVE_KEY", PASSPHRASE)
    monkeypatch.setenv("PROBE_EMPTY", "")
    monkeypatch.delenv("PROBE_UNSET", raising=False)

    assert secrets_from_environment(["PROBE_LIVE_KEY", "PROBE_EMPTY", "PROBE_UNSET", "  "]) == (
        PASSPHRASE,
    )


def test_secrets_from_an_unset_environment_never_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PROBE_NOT_THERE", raising=False)

    assert secrets_from_environment(["PROBE_NOT_THERE"]) == ()
    assert secrets_from_environment([]) == ()


# --------------------------------------------------------------------------- #
# The rendered report and the CLI
# --------------------------------------------------------------------------- #


def test_the_rendered_report_states_its_scope_and_masks_its_findings(
    tmp_path: Path,
) -> None:
    target = tmp_path / "leak.json"
    target.write_text(json.dumps({"api_key": OPENAI_SHAPED}), encoding="utf-8")

    text = render_report(scan_paths([target]))

    assert f"scan_version={SCAN_VERSION}" in text
    assert "files=1" in text
    assert "clean=False" in text
    assert OPENAI_SHAPED not in text
    assert "secret_field" in text


def test_a_clean_report_says_so_in_words(tmp_path: Path) -> None:
    (tmp_path / "ok.json").write_text("{}", encoding="utf-8")

    text = render_report(scan_paths([tmp_path]))

    assert "No credential-shaped value or secret-named field was found." in text


def test_a_long_report_is_truncated_but_counted(tmp_path: Path) -> None:
    payload = {f"api_key_{index}": OPENAI_SHAPED for index in range(40)}
    target = tmp_path / "many.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    text = render_report(scan_paths([target]))

    assert "more" in text
    assert text.count(OPENAI_SHAPED) == 0


def test_the_cli_exits_zero_on_a_clean_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"answer": "42"}), encoding="utf-8")

    assert main([str(tmp_path)]) == 0
    assert "clean=True" in capsys.readouterr().out


def test_the_cli_exits_one_when_it_finds_a_credential(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"api_key": OPENAI_SHAPED}), encoding="utf-8")

    assert main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "clean=False" in output
    assert OPENAI_SHAPED not in output


def test_the_cli_writes_a_json_report_and_creates_its_parent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"api_key": OPENAI_SHAPED}), encoding="utf-8")
    destination = tmp_path / "reports" / "scan.json"

    assert main([str(tmp_path / "run.json"), "--output", str(destination)]) == 1

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["clean"] is False
    assert payload["files_scanned"] == 1
    assert payload["by_category"]["secret_field"] == 1
    assert OPENAI_SHAPED not in json.dumps(payload)
    assert "--output" not in capsys.readouterr().out


def test_the_cli_can_print_json_instead_of_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")

    assert main([str(tmp_path), "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["scan_version"] == SCAN_VERSION
    assert payload["clean"] is True


def test_quiet_prints_only_the_status_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"api_key": OPENAI_SHAPED}), encoding="utf-8")

    assert main([str(tmp_path), "--quiet"]) == 1

    assert capsys.readouterr().out.strip() == "findings=2 clean=False"


def test_a_live_secret_supplied_by_environment_is_matched_but_never_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PROBE_LIVE_KEY", PASSPHRASE)
    target = tmp_path / "session.log"
    target.write_text(f"the transcript contains {PASSPHRASE}\n", encoding="utf-8")
    destination = tmp_path / "scan.json"

    assert (
        main([str(target), "--secrets-env", "PROBE_LIVE_KEY", "--output", str(destination)]) == 1
    )

    output = capsys.readouterr().out
    written = destination.read_text(encoding="utf-8")
    assert PASSPHRASE not in output
    assert PASSPHRASE not in written
    assert "known_secret" in output
    assert "known_secret" in written


def test_an_unset_secrets_env_variable_does_not_break_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PROBE_NOT_THERE", raising=False)
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")

    assert main([str(tmp_path), "--secrets-env", "PROBE_NOT_THERE"]) == 0


def test_the_cli_refuses_abbreviations_and_unknown_flags() -> None:
    parser = build_parser()

    assert parser.parse_args(["results/"]).paths == [Path("results/")]
    with pytest.raises(SystemExit):
        parser.parse_args(["results/", "--out", "x.json"])
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_the_cli_can_be_run_as_a_module() -> None:
    assert (REPO_ROOT / "src" / "security" / "scan.py").is_file()
    assert scan.__name__ == "security.scan"


# --------------------------------------------------------------------------- #
# The repository's own shipped surfaces
# --------------------------------------------------------------------------- #


def test_nothing_this_project_ships_contains_a_credential() -> None:
    """The scanner's own claim, applied to the project rather than to a fixture.

    ``tests/`` is excluded on purpose: it holds deliberate fake credentials and
    injection payloads as fixtures, and finding them is the correct behaviour.
    What must stay clean is everything a reader or a run actually receives.
    """

    existing = [str(REPO_ROOT / name) for name in SHIPPED_PATHS]
    report = scan_paths(existing)

    assert report.files_scanned > 50, "the scan must actually cover the tree"
    assert report.bytes_scanned > 100_000
    assert report.findings == (), render_report(report)
    assert report.clean is True
