"""Fail the deployment build if its installed runtime cannot serve the app.

Run after the install command, in the same virtual environment. Unlike tests
which permit missing optional dependencies, a deployment must have all of them.
No provider requests or credentials are used.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path


def main() -> None:
    # Direct execution sets sys.path[0] to scripts/, not the checkout root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ["BRAINOS_LAB_DB"] = ":memory:"

    # Check before importing the entrypoint: its degraded app intentionally
    # catches import failures, and optional UI mounting can otherwise hide them.
    for module in (
        "fastapi", "uvicorn", "gradio", "openai", "brainos_runtime", "pandas"
    ):
        importlib.import_module(module)

    from fastapi.testclient import TestClient

    from api.index import app

    with TestClient(app) as client:
        health = client.get("/api/health")
        if health.status_code != 200 or health.json().get("status") != "ok":
            raise RuntimeError("Vercel API health check failed; inspect startup logs")
        page = client.get("/")
        if page.status_code != 200 or "text/html" not in page.headers.get("content-type", ""):
            raise RuntimeError("Vercel UI is unavailable; inspect Gradio mount logs")
        # The HTML shell alone can pass while a pruned JS/CSS asset is missing.
        assets = re.findall(r'(?:src|href)="(\./assets/[^"]+)"', page.text)
        if not assets:
            raise RuntimeError("No Gradio frontend assets found in the page")
        for asset in assets:
            response = client.get("/" + asset.removeprefix("./"))
            if response.status_code != 200 or not response.content:
                raise RuntimeError(f"Gradio frontend asset unavailable: {asset}")
        config = client.get("/config")
        if config.status_code != 200 or not config.json().get("components"):
            raise RuntimeError("Vercel Gradio configuration is unavailable")

        components = config.json()["components"]
        if any(c["type"] in {"audio", "video", "model3d"} for c in components):
            raise RuntimeError("Revalidate trimmed media assets before adding media components")
        if importlib.util.find_spec("matplotlib") is None:
            plot_controls = [
                c["props"] for c in components
                if "Figures unavailable" in c["props"].get("label", "")
            ]
            if not plot_controls or any(
                c.get("value") or c.get("interactive") for c in plot_controls
            ):
                raise RuntimeError("Figure rendering must be disabled without matplotlib")

    print("Vercel runtime smoke check passed")


if __name__ == "__main__":
    main()
