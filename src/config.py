from __future__ import annotations
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
