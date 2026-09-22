"""Trim only unused assets, and keep the wheel metadata consistent for Vercel."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.prepare_vercel_bundle import (
    GRADIO_VERSION,
    UNUSED_GRADIO_PATHS,
    dependency_size,
    main,
    prune_gradio,
)


def test_pruning_is_allowlisted_updates_record_and_is_idempotent(tmp_path: Path) -> None:
    removed = [f"{p}/sample.bin" for p in UNUSED_GRADIO_PATHS]
    kept = [
        "gradio/templates/frontend/assets/app.js",
        "gradio/templates/frontend/assets/style.css",
        "gradio/routes.py",
        "gradio/templates/frontend/static/other.txt",
        "pandas/core/frame.py",
        "numpy.libs/native.so",
    ]
    record = tmp_path / "gradio.dist-info/RECORD"
    record.parent.mkdir()
    rows = []
    for relative in removed + kept:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"data")
        rows.append([relative, "sha256=unchanged", "4"])
    rows.append(["gradio.dist-info/RECORD", "", ""])
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)

    assert prune_gradio(tmp_path, record) == len(removed) * 4
    assert all(not (tmp_path / p).exists() for p in removed)
    assert all((tmp_path / p).read_bytes() == b"data" for p in kept)
    with record.open(newline="") as stream:
        assert list(csv.reader(stream)) == rows[len(removed):]
    assert prune_gradio(tmp_path, record) == 0


def test_pruning_refuses_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "gradio").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="outside"):
        prune_gradio(root, root / "RECORD")
    assert outside.is_dir()


def test_prepare_requires_a_disposable_vercel_environment(monkeypatch) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    with pytest.raises(RuntimeError, match="disposable"):
        main()


def test_size_excludes_generated_bytecode_but_counts_native_libraries(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_bytes(b"123")
    (tmp_path / "native.so").write_bytes(b"1234")
    (tmp_path / "module.pyc").write_bytes(b"12345")
    assert dependency_size(tmp_path) == 7


def test_vercel_profile_pins_the_pruned_gradio_and_omits_optional_charts() -> None:
    root = Path(__file__).resolve().parents[2]
    requirements = (root / "requirements-vercel.txt").read_text()
    active = [line for line in requirements.splitlines() if line and not line.startswith("#")]
    assert f"gradio=={GRADIO_VERSION}" in active
    assert not any(line.startswith("matplotlib") for line in active)
    assert "pandas>=2.0" in active
    assert "matplotlib>=3.7" in (root / "requirements.txt").read_text()
