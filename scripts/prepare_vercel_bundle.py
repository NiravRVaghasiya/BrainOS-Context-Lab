"""Trim unused Gradio assets in Vercel's disposable build venv, before bundling.

Keep Python modules, frontend JS/CSS, NumPy/Pandas, and native libraries intact.
Gradio's Node SSR server, demo media, and browser video transcoder are not used
by this app. SSR is explicitly disabled in app.vercel_app. Update wheel RECORD
as well: Vercel enumerates and sizes dependencies using that metadata.
"""

from __future__ import annotations

import csv
import importlib.metadata
import os
import shutil
import sys
from pathlib import Path

GRADIO_VERSION = "6.28.0"
UNUSED_GRADIO_PATHS = (
    "gradio/templates/frontend/static/ffmpeg",
    "gradio/templates/node",
    "gradio/media_assets",
)
# Leave room beneath the reported 225 MB cap for app source and Vercel runtime.
# This is a dependency budget, not a measurement of Vercel's final artifact.
DEPENDENCY_BUDGET_BYTES = 210_000_000


def prune_gradio(site_packages: Path, record: Path) -> int:
    """Remove only the allowlisted assets and their RECORD rows; idempotent."""
    root = site_packages.resolve()
    targets = [root / relative for relative in UNUSED_GRADIO_PATHS]
    for target in targets:
        if not target.resolve().is_relative_to(root):
            raise RuntimeError(f"Refusing to prune a path outside the build environment: {target}")
    with record.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    removed_bytes = 0
    for target in targets:
        if target.is_dir():
            removed_bytes += sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
            shutil.rmtree(target)
    with record.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for row in rows:
            if row and any(
                row[0] == prefix or row[0].startswith(prefix + "/")
                for prefix in UNUSED_GRADIO_PATHS
            ):
                continue
            writer.writerow(row)
    return removed_bytes


def dependency_size(site_packages: Path) -> int:
    """Count on-disk dependency files, excluding generated Python bytecode."""
    return sum(
        path.stat().st_size
        for path in site_packages.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def main() -> None:
    if os.getenv("VERCEL") != "1" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run only in Vercel's disposable build virtual environment (VERCEL=1)")
    distribution = importlib.metadata.distribution("gradio")
    if distribution.version != GRADIO_VERSION:
        raise RuntimeError("Revalidate the asset allowlist before changing the Gradio version")
    root = Path(distribution.locate_file("")).resolve()
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Gradio must be installed inside the active build virtual environment")
    records = [p for p in distribution.files or [] if str(p).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise RuntimeError("Cannot locate Gradio's wheel RECORD")
    removed = prune_gradio(root, Path(distribution.locate_file(records[0])))
    size = dependency_size(root)
    print(f"Removed {removed / 1e6:.2f} MB of unused Gradio assets")
    print(
        f"Runtime dependencies: {size / 1e6:.2f} MB "
        f"(budget: {DEPENDENCY_BUDGET_BYTES / 1e6:.0f} MB)"
    )
    if size > DEPENDENCY_BUDGET_BYTES:
        raise RuntimeError("Dependency budget exceeded; inspect installed packages")


if __name__ == "__main__":
    main()
