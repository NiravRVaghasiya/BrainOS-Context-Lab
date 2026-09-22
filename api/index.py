"""Vercel serverless entry point.

Vercel's Python runtime looks for `api/index.py` and expects an ASGI `app`
variable. This file bootstraps the repo's src/ layout and delegates to
`app.vercel_app` which implements both:

- REST API under /api/* (always available)
- Optional Gradio UI mounted at / (if gradio import succeeds)

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

Note: On Vercel, sessions are in-memory per Lambda instance. Two requests
may hit different instances and lose session. For production multi-user,
replace SessionManager with Redis (Upstash) - see VERCEL_DEPLOYMENT_ANALYSIS.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap - same as app.py
# ---------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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
try:
    from app.vercel_app import app  # noqa: F401 - Vercel expects `app`
except Exception as exc:
    # Fallback: minimal FastAPI app that explains the failure
    # This ensures Vercel still returns 200 with diagnostic, not 500 crash
    print(f"[api/index] Failed to import vercel_app: {exc}", file=sys.stderr)
    import traceback

    traceback.print_exc()

    from fastapi import FastAPI

    app = FastAPI(title="BrainOS Context Lab - Fallback")

    @app.get("/")
    def fallback_root():
        return {
            "error": "Failed to load BrainOS app",
            "detail": str(exc),
            "hint": "Check Vercel logs, ensure requirements.txt includes fastapi, gradio, openai, brainos-cli",
            "env": {
                "BRAINOS_LAB_DB": os.getenv("BRAINOS_LAB_DB"),
                "VERCEL": os.getenv("VERCEL"),
                "PYTHONPATH": os.getenv("PYTHONPATH"),
            },
        }

    @app.get("/api/health")
    def fallback_health():
        return {"status": "degraded", "error": str(exc)}

# Vercel expects `app` to be ASGI callable - we have it.
__all__ = ["app"]
