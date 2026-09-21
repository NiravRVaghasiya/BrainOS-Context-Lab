"""Where this checkout lives, and when a repository-relative path must be anchored.

The plan's §28 layout is relative on purpose: ``benchmarks/context_rot/
dataset.jsonl`` is how a reader of the plan finds the committed dataset, and
``results/`` is where §23 says a run writes. Inside the checkout, relative is
also what makes a re-run command portable — the artifact says
``--dataset benchmarks/context_rot/dataset.jsonl`` and that works on any machine.

The rule breaks the moment the process' working directory is not the checkout.
A supervisor that starts the app from ``/``, a ``pip install .`` console script
run from a home directory, or a Gradio Space whose worker changes directory all
turn a relative default into a path that does not exist — and, worse, into an
*output* root that gets created somewhere nobody will look. Phase 19's audit
found exactly that in the Evaluation tab: every dataset path and the whole
``results/ui/`` tree resolved against ``os.getcwd()``.

Two functions, one rule: **a path that belongs to the repository is anchored to
the repository when the process is running outside it, and left alone when the
process is running inside it.**

* :func:`in_checkout` — is this path inside the checkout?
* :func:`repo_anchored` — resolve a repository-owned location, with an honest
  fallback when the anchored location is not there (an installed wheel has no
  ``benchmarks/``, and a "file not found" message should name the documented
  path rather than a machine-specific guess).

Module is dependency-free on purpose: ``storage``, ``app``, and ``evaluation``
all import it, so it must not import any of them.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The checkout this module was imported from. ``src/checkout.py`` → the
#: repository root in a source checkout and in an editable install (which is
#: how the project is deployed, locally and on a Space).
REPO_ROOT = Path(__file__).resolve().parents[1]


def in_checkout(path: str | Path | None = None) -> bool:
    """Return True when ``path`` (default: the working directory) is in the checkout.

    ``resolve()`` is used on both sides so a symlinked checkout and a path that
    travels through ``..`` still compare correctly. An unresolvable path (a
    deleted directory, a permission error) is reported as outside the checkout:
    the caller then anchors to :data:`REPO_ROOT`, which is a known location.
    """

    candidate = Path(path) if path is not None else Path(os.getcwd())
    try:
        resolved = candidate.resolve()
    except OSError:  # pragma: no cover - depends on the filesystem, not on logic
        return False
    return resolved == REPO_ROOT or resolved.is_relative_to(REPO_ROOT)


def repo_anchored(relative: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a repository-owned path, anchoring it to the checkout when needed.

    Inside the checkout the path is returned unchanged, so the documented
    workflow keeps producing the plan's relative paths in artifacts and re-run
    commands. Outside it the path is resolved against :data:`REPO_ROOT`.

    ``must_exist`` is for inputs. A dataset a reader must be able to open is
    only anchored when the anchored file is really there; otherwise the
    original (relative) path is returned, so the failure message names the
    documented location instead of an invented one. Outputs — which are created
    rather than opened — always anchor, because writing ``results/ui/`` into an
    unknown working directory is the bug this function exists to prevent.
    """

    candidate = Path(relative)
    if in_checkout():
        return candidate
    anchored = REPO_ROOT / candidate
    if must_exist and not anchored.exists():
        return candidate
    return anchored


__all__ = ["REPO_ROOT", "in_checkout", "repo_anchored"]
