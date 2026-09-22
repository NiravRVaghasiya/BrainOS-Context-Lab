"""The Vercel entrypoint is an enforced contract, not folklore.

Vercel's FastAPI builder finds the application in two steps that disagree
about what counts as "the app":

* a **static** read of the recognised entrypoint locations (``app.py``,
  ``index.py``, ``server.py``, ``main.py``, ``wsgi.py``, ``asgi.py`` at the
  root or under ``src/``, ``app/``, ``api/``) that looks for a *module-level*
  assignment to ``app`` and does not look inside ``try``/``except``;
* an **import** of that module, which asks whether the object it got really is
  a FastAPI application.

The first is the one that broke this repository's deployment:

    Error: Found app.py, api/index.py but none define a top-level "app" FastAPI
    instance.

``app.py`` is the Hugging Face Space launcher — it has no ASGI app at all, and
it cannot grow one without importing the UI at module scope, which closes an
import cycle through ``src/evaluation``. ``api/index.py`` did define ``app``,
but only inside ``try:`` (the real application) and ``except:`` (the fallback),
so the static read saw nothing.

These tests pin down both halves: the file assigns ``app`` at column zero,
``pyproject.toml`` names the entrypoint so the builder never has to guess
between the two candidate files, and the object an import produces is a FastAPI
application on the healthy path *and* on the degraded one.

They need nothing beyond FastAPI; the Gradio UI and the BrainOS runtime are
optional, which is exactly why ``api/index.py`` degrades to a REST-only app.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "api" / "index.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The variable names Vercel's Python runtime accepts as a handler.
HANDLER_NAMES = frozenset({"app", "application", "handler"})

#: Loaded under its own name so it never collides with the ``app`` package.
_ENTRYPOINT_MODULE = "brainos_vercel_entrypoint"


def _module_level_names(path: Path) -> list[str]:
    """Return every name bound by a module-level statement in ``path``.

    This mirrors Vercel's scan: statements in the module body only. A binding
    inside ``try``/``except``, ``if``, or a function is invisible to it, which
    is the failure this file guards against.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.extend(alias.asname or alias.name for alias in node.names)
    return names


def _declared_entrypoint() -> tuple[str, str]:
    """Return ``(module, attribute)`` from ``[tool.vercel] entrypoint``."""

    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(
        r"^\[tool\.vercel\]\s*?\n(?:[^\[].*?\n)*?entrypoint\s*=\s*\"(?P<value>[^\"]+)\"",
        text,
        re.MULTILINE,
    )
    assert match is not None, "pyproject.toml must declare [tool.vercel] entrypoint"
    module, _, attribute = match.group("value").partition(":")
    return module, attribute


@contextlib.contextmanager
def _load_entrypoint() -> Iterator[object]:
    """Import ``api/index.py`` the way Vercel does, then undo the side effects.

    Importing the entrypoint builds a controller, opens SQLite stores, and sets
    deployment defaults in ``os.environ``; all of that must be unwound so the
    rest of the suite — which reloads ``storage.sqlite`` under its own
    environment — still sees the process it started with.
    """

    saved_env = dict(os.environ)
    saved_path = list(sys.path)
    saved_modules = dict(sys.modules)
    spec = importlib.util.spec_from_file_location(_ENTRYPOINT_MODULE, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ENTRYPOINT_MODULE] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        os.environ.clear()
        os.environ.update(saved_env)


# --------------------------------------------------------------------------- #
# What the builder reads statically
# --------------------------------------------------------------------------- #


def test_the_entrypoint_assigns_app_at_module_scope() -> None:
    """A column-zero ``app =`` is what the builder's scan looks for."""

    names = _module_level_names(ENTRYPOINT)
    assert "app" in names, (
        "api/index.py must bind `app` in the module body; a binding inside "
        "try/except is invisible to Vercel's scan"
    )


def test_app_is_not_bound_by_a_module_level_try_or_except() -> None:
    """The regression itself: a binding inside ``try``/``except`` is invisible.

    ``api/index.py`` used to assign ``app`` twice at module scope — once inside
    ``try:``, once inside ``except:`` — and both were unreadable to the scan.
    Bindings inside functions are fine (the scan reads the module body), so
    only the module-level compound statements are checked here.
    """

    tree = ast.parse(ENTRYPOINT.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Try, ast.If, ast.With, ast.For, ast.While)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for target in inner.targets:
                    if isinstance(target, ast.Name) and target.id in HANDLER_NAMES:
                        pytest.fail(
                            f"`{target.id}` is bound at line {inner.lineno}, "
                            "inside a compound statement the scan cannot see"
                        )


def test_pyproject_names_the_entrypoint_instead_of_leaving_it_to_the_scan() -> None:
    """Two candidate files, one real app: the builder must be told which."""

    module, attribute = _declared_entrypoint()
    assert attribute == "app"

    path = REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    assert path.exists(), f"the declared entrypoint {module} does not exist"
    assert path == ENTRYPOINT
    # The named module has to satisfy the same scan, or the declaration buys
    # nothing.
    assert "app" in _module_level_names(path)


def test_the_entrypoint_sits_where_vercel_looks_by_default() -> None:
    """index.py under api/ is a recognised location; src/app/vercel_app.py is not."""

    assert ENTRYPOINT.parent.name in {"src", "app", "api", "."}
    assert ENTRYPOINT.name in {
        "app.py",
        "index.py",
        "server.py",
        "main.py",
        "wsgi.py",
        "asgi.py",
    }


# --------------------------------------------------------------------------- #
# What the runtime imports
# --------------------------------------------------------------------------- #


def test_the_entrypoint_exposes_a_fastapi_application() -> None:
    fastapi = pytest.importorskip("fastapi")

    with _load_entrypoint() as module:
        assert isinstance(module.app, fastapi.FastAPI)


def test_the_health_endpoint_answers_over_the_imported_app() -> None:
    """The check a deployment actually performs after a successful build."""

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    with _load_entrypoint() as module:
        with TestClient(module.app) as client:
            response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_an_unimportable_application_degrades_to_a_fastapi_app(monkeypatch) -> None:
    """The fallback must survive the same two checks as the healthy path.

    A bare ASGI function reports the reason just as well to a human, but the
    runtime asks whether ``app`` is a FastAPI instance — answering with
    anything else trades a readable error for a failed build.
    """

    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "app.vercel_app":
            raise ImportError("blocked by test: app.vercel_app cannot be imported")
        return real_import(name, globals, locals, fromlist, level)

    # Patching ``__import__`` rather than planting ``None`` in ``sys.modules``
    # is what makes this deterministic: the import statement consults it before
    # any cache, so a copy of the module another test already imported cannot
    # satisfy the import and quietly turn this into the healthy-path test.
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with _load_entrypoint() as module:
        assert isinstance(module.app, fastapi.FastAPI)
        with TestClient(module.app) as client:
            health = client.get("/api/health")
            root = client.get("/")

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert root.status_code == 200
    assert "vercel_app" in root.json()["detail"]


def test_the_space_launcher_does_not_shadow_the_source_package() -> None:
    """``app.py`` (the Space launcher) and ``src/app`` (the package) share a name.

    Whichever is imported first wins for the whole process. Importing the
    launcher first happens as soon as anything imports ``app`` with the checkout
    root ahead of ``src/`` on ``sys.path`` — which is what Vercel's own scan of
    ``app.py`` does. The entrypoint must still load the package.
    """

    fastapi = pytest.importorskip("fastapi")
    saved_modules = dict(sys.modules)
    launcher_path = REPO_ROOT / "app.py"
    package_init = REPO_ROOT / "src" / "app" / "__init__.py"
    try:
        spec = importlib.util.spec_from_file_location("app", launcher_path)
        assert spec is not None and spec.loader is not None
        launcher = importlib.util.module_from_spec(spec)
        sys.modules["app"] = launcher
        spec.loader.exec_module(launcher)
        assert Path(sys.modules["app"].__file__) == launcher_path

        with _load_entrypoint() as module:
            assert Path(sys.modules["app"].__file__) == package_init
            assert isinstance(module.app, fastapi.FastAPI)
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_vercel_config_leaves_the_runtime_pythonpath_alone() -> None:
    """The entrypoint adds src/; deployment config must not replace vendor paths."""

    config = json.loads((REPO_ROOT / "vercel.json").read_text(encoding="utf-8"))
    for section in ("env", "build", "builds"):
        value = config.get(section, {})
        assert '"PYTHONPATH"' not in json.dumps(value), (
            "Do not override Vercel's dependency path with PYTHONPATH=src"
        )


def test_entrypoint_preserves_the_runtime_dependency_path(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    vendor = str(tmp_path / "_vendor")
    monkeypatch.setenv("PYTHONPATH", vendor)
    monkeypatch.syspath_prepend(vendor)

    with _load_entrypoint():
        assert sys.path[0] == str(REPO_ROOT / "src")
        assert vendor in sys.path
        assert os.environ["PYTHONPATH"] == vendor


def test_vercel_startup_serves_the_ui_not_just_health() -> None:
    """The API can be healthy even when the optional Gradio mount failed."""

    for dependency in ("fastapi", "httpx", "gradio"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{dependency} is not installed")

    # A cold process is essential: the import-isolation helper above unwinds
    # sys.modules, whereas native modules such as NumPy cannot be reloaded.
    script = """
from api.index import app
from fastapi.testclient import TestClient

with TestClient(app) as client:
    health = client.get("/api/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok", health.text
    page = client.get("/")
    assert page.status_code == 200, page.text
    assert "text/html" in page.headers["content-type"]
    assert "gradio" in page.text.lower()
    config = client.get("/config")
    assert config.status_code == 200, config.text
    assert config.json()["components"]
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**os.environ, "VERCEL": "1", "GRADIO_ANALYTICS_ENABLED": "False"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_vercel_installs_the_deployment_requirements_and_checks_startup() -> None:
    """An empty base dependency list must not become the deployment environment."""

    config = json.loads((REPO_ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config["installCommand"] == "uv pip install -r requirements-vercel.txt"
    assert config["buildCommand"] == (
        "python scripts/prepare_vercel_bundle.py && python scripts/check_vercel_runtime.py"
    )
    assert (REPO_ROOT / "scripts/check_vercel_runtime.py").is_file()


def test_deployment_smoke_check_does_not_accept_a_missing_fastapi() -> None:
    # No site-packages (-S) reproduces the incomplete runtime in the logs.
    # The deployment check must fail, not skip or accept the degraded app.
    result = subprocess.run(
        [sys.executable, "-S", "scripts/check_vercel_runtime.py"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "No module named 'fastapi'" in result.stderr
    assert "Vercel runtime smoke check passed" not in result.stdout


def test_deployment_smoke_check_in_the_installed_environment() -> None:
    for dependency in (
        "fastapi", "uvicorn", "gradio", "openai", "brainos_runtime", "pandas"
    ):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{dependency} is not installed")
    result = subprocess.run(
        [sys.executable, "scripts/check_vercel_runtime.py"],
        cwd=REPO_ROOT,
        env={**os.environ, "VERCEL": "1", "GRADIO_ANALYTICS_ENABLED": "False"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Vercel runtime smoke check passed" in result.stdout
