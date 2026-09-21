"""Phase 14 — deployment configuration is an enforced contract, not folklore.

These tests pin down the things an HF Space (or any other public deployment)
relies on that aren't visible from inside a running chat turn:

* ``app.py`` resolves ``src/`` on ``sys.path`` and exposes ``main``,
* ``requirements.txt`` lists the UI, provider, BrainOS, and evaluation deps
  and contains **no** shared API key and no hard-coded endpoint,
* ``packages.txt`` exists (the Space SDK refuses a missing one),
* the SQLite backend reads ``BRAINOS_LAB_DB=:memory:`` as a signal to run in
  memory (ephemeral/stateless deployments), and opens disk-backed stores in
  WAL mode with a busy timeout so concurrent Gradio workers don't lock,
* the Gradio entry point calls ``.queue()`` with bounded concurrency so a
  public Space cannot be driven into unbounded parallel provider calls,
* the UI header honestly discloses whether persistence is enabled,
* Phase 20: the README opens with the Hugging Face manifest a Space build reads
  (``sdk`` / ``app_file`` / a version that does not lag ``requirements.txt``),
  and the five ``BRAINOS_LAB_EVAL_*`` variables narrow the Evaluation tab
  without editing source — including the failure mode, where an unparsable value
  stops startup instead of silently hosting the widest policy.

These tests do not require Gradio or BrainOS to be installed; they read files
and exercise the storage module with dependency-free assertions.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _reload_sqlite(monkeypatch: pytest.MonkeyPatch, db_env: str | None):
    """Reload ``storage.sqlite`` with ``BRAINOS_LAB_DB`` pinned to ``db_env``."""

    if db_env is None:
        monkeypatch.delenv("BRAINOS_LAB_DB", raising=False)
    else:
        monkeypatch.setenv("BRAINOS_LAB_DB", db_env)
    import storage.sqlite as sqlite_mod

    return importlib.reload(sqlite_mod)


def _reload_ui(monkeypatch: pytest.MonkeyPatch, db_env: str | None):
    """Reload ``app.ui`` so ``_header_markdown()`` picks up the new env."""

    sqlite_mod = _reload_sqlite(monkeypatch, db_env)
    import app.ui as ui_mod

    importlib.reload(ui_mod)
    return ui_mod, sqlite_mod


# --------------------------------------------------------------------------- #
# Space files exist and are well-formed
# --------------------------------------------------------------------------- #


def test_app_py_exists_and_bootstraps_the_source_path() -> None:
    text = _read("app.py")
    assert "src_dir" in text
    assert 'sys.path.insert(0, str(src_dir))' in text
    assert "from app.ui import main" in text


def test_requirements_txt_lists_the_runtime_dependencies() -> None:
    text = _read("requirements.txt")
    for required in ("gradio", "openai", "brainos-cli", "matplotlib", "pandas"):
        assert required in text, f"requirements.txt must list {required}"
    assert "-e ." in text
    # No shared key / demo token / placeholder endpoint.
    assert "sk-" not in text
    assert "hf_" not in text
    assert "api_key" not in text.lower()
    # Pinned BrainOS commit hash from Phase 0 is present (not just "main").
    assert "1d9eb7a0ca537e7278e29809cda4f4c5da6c1dcc" in text


def test_packages_txt_exists_and_is_comment_or_apt_packages_only() -> None:
    """Gradio Spaces accept an empty packages.txt but not a missing one."""

    path = REPO_ROOT / "packages.txt"
    assert path.exists()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert re.match(r"^[a-z0-9][a-z0-9.+-]*$", stripped), (
            f"packages.txt line does not look like an apt package: {stripped!r}"
        )


# --------------------------------------------------------------------------- #
# SQLite backend deployment behaviour
# --------------------------------------------------------------------------- #


def test_memory_sentinel_switches_to_shared_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_mod = _reload_sqlite(monkeypatch, ":memory:")

    assert sqlite_mod.default_database_path() == ":memory:"
    assert sqlite_mod.persistence_enabled() is False

    store = sqlite_mod.SqliteConversationStore()
    assert store.database_path == ":memory:"
    with store._connect() as conn:
        jm = conn.execute("PRAGMA journal_mode").fetchone()
        assert jm[0] == "memory", "in-memory stores must not try WAL"


def test_disk_backed_store_uses_wal_busy_timeout_secure_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sqlite_mod = _reload_sqlite(monkeypatch, str(tmp_path / "lab.sqlite3"))

    store = sqlite_mod.SqliteConversationStore(tmp_path / "lab.sqlite3")
    with store._connect() as conn:
        jm = conn.execute("PRAGMA journal_mode").fetchone()
        busy = conn.execute("PRAGMA busy_timeout").fetchone()
        sync = conn.execute("PRAGMA synchronous").fetchone()
        sd = conn.execute("PRAGMA secure_delete").fetchone()
    assert jm[0] == "wal", "Phase 14 requires WAL for concurrent readers"
    assert busy[0] >= 30000, "busy timeout must be at least 30s"
    assert sync[0] in (1, 2), "synchronous must be NORMAL or FULL"
    assert sd[0] == 1, "secure_delete stays on in Phase 14"


def test_two_memory_stores_share_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared-cache URI is what makes per-op connections see data."""

    sqlite_mod = _reload_sqlite(monkeypatch, ":memory:")
    store_a = sqlite_mod.SqliteConversationStore()
    store_b = sqlite_mod.SqliteConversationStore()
    from storage.conversations import ConversationMessage

    store_a.append(
        ConversationMessage(
            session_id="s",
            conversation_id="s-main",
            role="user",
            content="hello",
            created_at="2026-01-01T00:00:00+00:00",
            metadata={},
        )
    )
    rows = store_b.list_messages("s", "s-main")
    assert len(rows) == 1
    assert rows[0].content == "hello"


def test_deleting_a_session_checkpoints_the_wal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After a session delete, the WAL must not retain a plaintext marker."""

    sqlite_mod = _reload_sqlite(monkeypatch, str(tmp_path / "lab.sqlite3"))
    MARKER = "DEPLOY-TEST-MARKER-4a9f2c"
    store = sqlite_mod.SqliteConversationStore(tmp_path / "lab.sqlite3")
    from storage.conversations import ConversationMessage

    store.append(
        ConversationMessage(
            session_id="s",
            conversation_id="s-main",
            role="user",
            content=f"the secret is {MARKER}",
            created_at="2026-01-01T00:00:00+00:00",
            metadata={},
        )
    )
    store.delete_session("s")
    wal = tmp_path / "lab.sqlite3-wal"
    if wal.exists():
        assert MARKER.encode() not in wal.read_bytes(), (
            "delete_session must TRUNCATE the WAL so deleted content is gone"
        )


# --------------------------------------------------------------------------- #
# UI entry point
# --------------------------------------------------------------------------- #


def test_main_enables_a_bounded_queue_for_public_traffic() -> None:
    """Phase 14: a public surface must bound concurrency via Gradio's queue."""

    src = _read("src/app/ui.py")
    assert "demo.queue(" in src, "main() must call demo.queue() before launch"
    assert "default_concurrency_limit" in src
    assert "max_size" in src
    assert "api_open=False" in src, "internal API endpoints must not be exposed"
    queue_pos = src.index("demo.queue(")
    launch_pos = src.index("demo.launch(")
    assert queue_pos < launch_pos


def test_header_discloses_in_memory_when_persistence_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui_mod, _sqlite_mod = _reload_ui(monkeypatch, ":memory:")
    header = ui_mod._header_markdown()
    assert "fully in memory" in header
    assert "server-side SQLite" not in header


def test_header_discloses_persistence_when_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ui_mod, _sqlite_mod = _reload_ui(monkeypatch, str(tmp_path / "lab.sqlite3"))
    header = ui_mod._header_markdown()
    assert "server-side" in header and "SQLite" in header
    assert "fully in memory" not in header


def test_default_database_path_is_repo_local_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite_mod = _reload_sqlite(monkeypatch, None)
    assert sqlite_mod.persistence_enabled() is True
    import pathlib

    assert sqlite_mod.default_database_path() == pathlib.Path("data") / "brainos_lab.sqlite3"


# --------------------------------------------------------------------------- #
# Phase 20: the Space manifest and the deployment-narrowed Evaluation tab
# --------------------------------------------------------------------------- #


#: The colours the Space server accepts in front matter; anything else is
#: rejected when the Space is created.
_HF_SPACE_COLORS = frozenset(
    {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
)


def _front_matter(text: str) -> dict[str, str]:
    """Parse the ``---`` block at the top of the README as ``key: value`` pairs."""

    assert text.startswith("---\n"), "README.md must open with the Space manifest"
    end = text.index("\n---\n", 4)
    block: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition(":")
        block[key.strip()] = value.strip()
    return block


def test_readme_carries_the_hugging_face_space_manifest() -> None:
    """A Space created from this repository reads these keys or builds nothing."""

    manifest = _front_matter(_read("README.md"))
    assert manifest["sdk"] == "gradio"
    assert manifest["app_file"] == "app.py"
    assert (REPO_ROOT / manifest["app_file"]).exists()
    assert manifest["title"] == "BrainOS Context Lab"
    assert manifest["emoji"]
    # The Space server rejects a longer short_description, so this is a build
    # failure waiting to happen rather than a style preference.
    assert 0 < len(manifest["short_description"]) <= 60
    assert manifest["colorFrom"] in _HF_SPACE_COLORS
    assert manifest["colorTo"] in _HF_SPACE_COLORS

    # The manifest's SDK version must match the requirement the build installs;
    # pinning an older major version would build a different Gradio than the one
    # this suite is validated against.
    requirement = re.search(r"gradio>=(\d+)", _read("requirements.txt"))
    assert requirement is not None, "requirements.txt no longer pins a gradio floor"
    assert manifest["sdk_version"].split(".")[0] == requirement.group(1)


def test_the_deployment_document_lists_every_variable_the_code_reads() -> None:
    """A variable an operator cannot find is a variable an operator will guess."""

    import app.evaluation as evaluation

    documented = _read("docs/deployment.md")
    for name in evaluation.POLICY_ENVIRONMENT_VARIABLES:
        assert name in documented, f"{name} is read by the code but not documented"
    assert "docs/deployment.md" in _read("README.md")


def test_the_ui_builds_its_controller_with_the_deployments_policy() -> None:
    """The variables only matter if the shipped entry point reads them."""

    src = _read("src/app/ui.py")
    assert "EvaluationPolicy.from_environment()" in src
    assert "evaluation_policy=" in src




def test_a_generation_flag_accepts_the_strings_a_deployment_uses() -> None:
    from app.evaluation import EvaluationPolicy

    for off in ("0", "false", "no", "off", "FALSE"):
        policy = EvaluationPolicy.from_environment({"BRAINOS_LAB_EVAL_ALLOW_GENERATION": off})
        assert policy.allow_generation is False, off
    # A blank value is "unset", not "off": an operator who clears the field in the
    # Space settings should get the documented default back, not silence.
    assert (
        EvaluationPolicy.from_environment(
            {"BRAINOS_LAB_EVAL_ALLOW_GENERATION": "  "}
        ).allow_generation
        is True
    )
    for on in ("1", "true", "yes", "on"):
        assert EvaluationPolicy.from_environment(
            {"BRAINOS_LAB_EVAL_ALLOW_GENERATION": on}
        ).allow_generation is True



# --------------------------------------------------------------------------- #
# Space manifest (Phase 20)
# --------------------------------------------------------------------------- #




def test_the_space_app_file_exposes_the_entry_point_the_manifest_names() -> None:
    text = _read("app.py")
    assert "from app.ui import main" in text
    assert "main()" in text
    assert "sys.path.insert(0, str(src_dir))" in text




# --------------------------------------------------------------------------- #
# Evaluation-policy variables (Phase 20)
# --------------------------------------------------------------------------- #


def test_no_variables_means_the_plan_defaults() -> None:
    from app.evaluation import EvaluationPolicy

    assert EvaluationPolicy.from_environment({}) == EvaluationPolicy()
    assert EvaluationPolicy.from_environment({"BRAINOS_LAB_EVAL_PRESETS": "  "}) == (
        EvaluationPolicy()
    )


def test_a_public_space_can_narrow_the_tab_without_editing_source() -> None:
    from app.evaluation import EvaluationPolicy, EvaluationRequestError

    policy = EvaluationPolicy.from_environment(
        {
            "BRAINOS_LAB_EVAL_PRESETS": "quick",
            "BRAINOS_LAB_EVAL_MAX_TASKS": "20",
            "BRAINOS_LAB_EVAL_MAX_REQUESTS": "60",
            "BRAINOS_LAB_EVAL_ALLOW_GENERATION": "0",
            "BRAINOS_LAB_EVAL_RETENTION_DAYS": "0",
        }
    )

    assert policy.allowed_presets == ("quick",)
    assert policy.max_tasks == 20
    assert policy.max_requests == 60
    assert policy.allow_generation is False
    assert policy.results_retention_seconds is None
    assert "no automatic sweep" in policy.retention_notice()

    with pytest.raises(EvaluationRequestError, match="retrieval-only"):
        policy.check(preset_name="quick", generate=True)
    with pytest.raises(EvaluationRequestError, match="does not allow"):
        policy.check(preset_name="standard", generate=False)
    policy.check(preset_name="quick", generate=False)  # the dry run still works


def test_a_narrowed_policy_still_obeys_the_presets_ceiling() -> None:
    """The variables can only tighten — ``apply`` keeps the preset's own limits."""

    from app.evaluation import EvaluationPolicy
    from evaluation.experiment import ExperimentPlan

    plan = ExperimentPlan.from_preset("standard", modes=("full_context",))
    assert (plan.limits.max_tasks, plan.limits.max_requests) == (100, 500)

    narrowed = EvaluationPolicy.from_environment({"BRAINOS_LAB_EVAL_MAX_TASKS": "20"}).apply(plan)
    assert narrowed.limits.max_tasks == 20
    # Everything the policy did not narrow is still the preset's own number.
    assert narrowed.limits.max_requests == 500

    looser = EvaluationPolicy.from_environment({"BRAINOS_LAB_EVAL_MAX_TASKS": "500"}).apply(plan)
    assert looser.limits.max_tasks == 100, "a policy can never widen a preset's ceiling"


def test_a_registered_retention_window_survives_the_conversion() -> None:
    from app.evaluation import EvaluationPolicy

    policy = EvaluationPolicy.from_environment({"BRAINOS_LAB_EVAL_RETENTION_DAYS": "3"})
    assert policy.results_retention_seconds == 3 * 86_400
    assert "3 days" in policy.retention_notice()


def test_a_variable_that_cannot_be_obeyed_stops_the_process_at_startup() -> None:
    from app.evaluation import EvaluationPolicy

    for name, value, expected in (
        ("BRAINOS_LAB_EVAL_MAX_TASKS", "twenty", "must be an integer"),
        ("BRAINOS_LAB_EVAL_MAX_TASKS", "0", "must be at least 1"),
        ("BRAINOS_LAB_EVAL_MAX_REQUESTS", "-5", "must be at least 1"),
        ("BRAINOS_LAB_EVAL_RETENTION_DAYS", "-1", "must be at least 0"),
    ):
        with pytest.raises(ValueError, match=expected) as error:
            EvaluationPolicy.from_environment({name: value})
        assert name in str(error.value)

    with pytest.raises(ValueError, match="Unknown preset"):
        EvaluationPolicy.from_environment({"BRAINOS_LAB_EVAL_PRESETS": "gigantic"})
