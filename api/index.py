"""Vercel serverless entry point.

Vercel's Python runtime looks for `api/index.py` and expects an ASGI `app`
variable. This file bootstraps the repo's src/ layout and delegates to
`app.vercel_app` which implements both:

- REST API under /api/* (always available)
- Optional Gradio UI mounted at / (if gradio import succeeds)

Why ``app`` is assigned by a helper
-----------------------------------

Vercel's FastAPI builder finds the application by a *static* read of the
entrypoint file: it looks for a module-level (column-zero) assignment to
``app`` and does not look inside ``try``/``except``. An earlier version of this
file assigned ``app`` twice — once inside ``try:`` (the real application) and
once inside ``except:`` (the diagnostic fallback) — which is invisible to that
scan and failed the build with::

    Error: Found app.py, api/index.py but none define a top-level "app" FastAPI
    instance.

The error handling now lives in :func:`_build_app`, and ``app = _build_app()``
is a plain module-level statement. The fallback is still a real ``FastAPI``
instance rather than a bare ASGI function: the runtime half of the same check
asks whether ``app`` *is* one, and a deployment that cannot import its own code
should say why instead of returning ``FUNCTION_INVOCATION_FAILED``.

Environment defaults for Vercel (ephemeral, no file persistence):
- BRAINOS_LAB_DB=:memory:  -> SqliteStore uses shared-cache URI, no file
- BRAINOS_LAB_EVAL_ALLOW_GENERATION=0 -> evaluation tab retrieval-only
- BRAINOS_LAB_EVAL_PRESETS=quick
- RESULTS_ROOT=/tmp/results -> avoid read-only repo root

To test locally:
    pip install fastapi uvicorn gradio openai
    BRAINOS_LAB_DB=:memory: python -m uvicorn api.index:app --host 0.0.0.0 --port 7860 --reload

To deploy:
    vercel --prod
    # Set env vars in Vercel dashboard as per vercel.json or docs/vercel-deployment.md
    # pyproject.toml names this file as the entrypoint ([tool.vercel] entrypoint),
    # so the builder does not have to guess between app.py (the Space launcher)
    # and this file.

Note: On Vercel, sessions are in-memory per Lambda instance. Two requests
may hit different instances and lose session. For production multi-user,
replace SessionManager with Redis (Upstash) - see VERCEL_DEPLOYMENT_ANALYSIS.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap - same as app.py
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"

# ``app`` names two things in this checkout: the repository-root launcher
# (``app.py``, the Hugging Face Space entry point) and the source package
# (``src/app/``) that holds the ASGI application. Whichever Python resolves
# first wins for the whole process, so ``src/`` is moved to the front of
# ``sys.path`` rather than merely appended to it — a host that puts the
# checkout root on the path (Vercel's runtime, pytest, ``python -m``) would
# otherwise hand ``import app`` the launcher.
if str(src_dir) in sys.path:
    sys.path.remove(str(src_dir))
sys.path.insert(0, str(src_dir))

# A launcher imported *before* this file runs is cached under the same name and
# would be reused no matter what the path now says. It only works as a
# stand-in because it re-points ``__path__`` at the package — a hack this file
# should not have to depend on — so it is dropped, with anything imported
# through it, and the package is imported on its own terms.
_cached_app = sys.modules.get("app")
if _cached_app is not None and Path(getattr(_cached_app, "__file__", "") or "") == (
    repo_root / "app.py"
):
    for _name in [n for n in list(sys.modules) if n == "app" or n.startswith("app.")]:
        del sys.modules[_name]
    del _cached_app

# ---------------------------------------------------------------------------
# Vercel-safe env defaults (must be before importing storage)
# ---------------------------------------------------------------------------
os.environ.setdefault("BRAINOS_LAB_DB", ":memory:")
os.environ.setdefault("BRAINOS_LAB_EVAL_ALLOW_GENERATION", "0")
os.environ.setdefault("BRAINOS_LAB_EVAL_PRESETS", "quick")
os.environ.setdefault("BRAINOS_LAB_EVAL_MAX_TASKS", "5")
os.environ.setdefault("BRAINOS_LAB_EVAL_MAX_REQUESTS", "15")
os.environ.setdefault("BRAINOS_LAB_CONCURRENCY", "1")
os.environ.setdefault("BRAINOS_LAB_MAX_QUEUE", "8")

# On Vercel, the filesystem is read-only except /tmp. The repo's default
# results root is `results/` which would fail. Override to /tmp.
if os.getenv("VERCEL") == "1":
    os.environ.setdefault("RESULTS_ROOT", "/tmp/results")
    # Also ensure /tmp/results exists
    try:
        Path("/tmp/results").mkdir(parents=True, exist_ok=True)
        Path("/tmp/results/ui").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Import the FastAPI+Gradio app
# ---------------------------------------------------------------------------


def _degraded_app(exc: BaseException) -> Any:
    """Build a FastAPI app that reports why the real one could not load.

    The fallback answers with the reason instead of letting the deployment
    return an opaque 500, but it is still a ``FastAPI`` instance: Vercel's
    entrypoint check reads this file statically *and* asks whether the object
    it loaded is a FastAPI application.
    """

    print(f"[api/index] Failed to import vercel_app: {exc}", file=sys.stderr)
    import traceback

    traceback.print_exc()

    from fastapi import FastAPI

    detail = f"{type(exc).__name__}: {exc}"
    application = FastAPI(title="BrainOS Context Lab - degraded")

    @application.get("/")
    def degraded_root() -> dict[str, Any]:
        return {
            "error": "Failed to load BrainOS app",
            "detail": detail,
            "hint": (
                "Check the runtime logs; requirements.txt must include fastapi, "
                "gradio, openai and brainos-cli"
            ),
            "env": {
                "BRAINOS_LAB_DB": os.getenv("BRAINOS_LAB_DB"),
                "VERCEL": os.getenv("VERCEL"),
                "PYTHONPATH": os.getenv("PYTHONPATH"),
            },
        }

    @application.get("/api/health")
    def degraded_health() -> dict[str, Any]:
        return {"status": "degraded", "error": detail}

    return application


def _build_app() -> Any:
    """Return the deployed application: the real one, or why it is missing.

    Kept separate from the module body on purpose — see the module docstring.
    """

    try:
        from app.vercel_app import app as application
    except Exception as exc:  # noqa: BLE001 - a broken import must not break the build
        return _degraded_app(exc)
    return application


# Vercel's builder scans this file for a column-zero assignment to ``app``.
# Keep it here: moving the import back into ``try``/``except`` at module scope
# hides it from that scan and fails the build.
app = _build_app()

__all__ = ["app"]
