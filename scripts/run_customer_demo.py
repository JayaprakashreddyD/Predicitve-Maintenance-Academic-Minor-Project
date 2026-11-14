from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
dashboard = ROOT / "ui" / "dashboard.py"

if not dashboard.exists():
    raise FileNotFoundError(f"Dashboard not found: {dashboard}")

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(dashboard)],
    check=True,
)
