"""Application entry point for local development and Hugging Face Spaces."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep ``python app.py`` usable from a source checkout without requiring an
# editable install. Installed packages still use the normal package layout.
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# The repository entry point is named ``app.py`` while the source package is
# also named ``app`` to match the planned layout. When this file is imported
# from the checkout (for example by tests), expose the source directory as the
# package path so ``import app.session`` resolves to ``src/app/session.py``.
if __name__ == "app":
    __path__ = [str(src_dir / "app")]  # type: ignore[name-defined]

from app.ui import main  # noqa: E402  (path bootstrap must happen first)


if __name__ == "__main__":
    main()
