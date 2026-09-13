"""Report generation extension point for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(report: dict[str, Any], output: str | Path) -> Path:
    """Write a structured report without credentials or provider clients."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    destination.write_text(rendered, encoding="utf-8")
    return destination
