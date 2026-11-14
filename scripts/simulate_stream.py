from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import requests

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default="data/processed/prepared.npz")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    data = np.load(Path(args.prepared))
    windows = data["test_x"]
    for i, window in enumerate(windows[:args.limit]):
        payload = {"machine_id": f"stream-{i % 5}", "sensor_window": window.tolist()}
        response = requests.post(args.url, json=payload, timeout=10)
        response.raise_for_status()
        print(json.dumps(response.json()))
        time.sleep(args.sleep)

if __name__ == "__main__":
    main()
