from __future__ import annotations
import subprocess
import sys

COMMANDS = [
    [sys.executable, "-m", "src.generate_data"],
    [sys.executable, "-m", "src.preprocess"],
    [sys.executable, "-m", "src.train"],
    [sys.executable, "-m", "src.evaluate"]
]

for command in COMMANDS:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, check=True)
