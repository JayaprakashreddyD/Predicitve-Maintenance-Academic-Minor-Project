from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_config, PROJECT_ROOT

SENSOR_COLUMNS = ["temperature", "vibration", "pressure", "rpm", "load", "ambient_temp", "humidity"]

def generate_dataset(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    dcfg = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    freq = int(dcfg["frequency_minutes"])
    periods = int(dcfg["days"] * 24 * 60 / freq)
    start = pd.Timestamp("2026-01-01T00:00:00")
    frames: list[pd.DataFrame] = []

    for machine_idx in range(int(dcfg["n_machines"])):
        machine_id = f"M-{machine_idx + 1:03d}"
        machine_type = ["pump", "motor", "compressor"][machine_idx % 3]
        timestamps = pd.date_range(start, periods=periods, freq=f"{freq}min")
        t = np.arange(periods)
        age_factor = 0.2 + machine_idx / max(1, dcfg["n_machines"] - 1)

        load = np.clip(0.55 + 0.18*np.sin(t/48) + rng.normal(0, 0.07, periods), 0.05, 1.0)
        rpm = 1200 + 850*load + rng.normal(0, 45, periods)
        ambient = 26 + 5*np.sin(t/(24*60/freq)*2*np.pi) + rng.normal(0, 0.8, periods)
        humidity = np.clip(55 + 15*np.sin(t/(24*60/freq)*2*np.pi + 0.7) + rng.normal(0, 4, periods), 20, 95)
        degradation = np.linspace(0, 1.6 + 1.2*age_factor, periods)

        temperature = 45 + 16*load + 0.7*ambient + 3.2*degradation + rng.normal(0, 1.5, periods)
        vibration = 1.1 + 1.8*load + 1.5*degradation + rng.normal(0, 0.25, periods)
        pressure = 8.2 - 1.4*load - 0.9*degradation + rng.normal(0, 0.18, periods)

        # Create realistic failure events concentrated where degradation and stress are high.
        stress = (
            0.035*(temperature - 65)
            + 0.45*(vibration - 3.5)
            + 0.8*np.maximum(0, 6.2 - pressure)
            + 0.7*np.maximum(0, load - 0.82)
        )
        failure_probability = 1 / (1 + np.exp(-(stress - 2.5)))
        failure_event = rng.random(periods) < (failure_probability * 0.012)
        # Enforce spacing so events resemble maintenance cycles.
        event_indices = np.flatnonzero(failure_event)
        accepted: list[int] = []
        min_gap = int(72 * 60 / freq)
        for idx in event_indices:
            if not accepted or idx - accepted[-1] >= min_gap:
                accepted.append(int(idx))
        failure_event[:] = False
        failure_event[accepted] = True

        frame = pd.DataFrame({
            "timestamp": timestamps,
            "machine_id": machine_id,
            "machine_type": machine_type,
            "shift": np.where(timestamps.hour < 8, "A", np.where(timestamps.hour < 16, "B", "C")),
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,
            "load": load,
            "production_rate": np.clip(85*load + rng.normal(0, 3, periods), 0, None),
            "ambient_temp": ambient,
            "humidity": humidity,
            "failure_event": failure_event.astype(int),
            "failure_type": np.where(failure_event, rng.choice(["bearing", "overheat", "pressure_loss"], periods), "none")
        })

        # Inject limited missingness to exercise preprocessing.
        for col in ["temperature", "vibration", "pressure"]:
            missing_idx = rng.choice(periods, size=max(1, periods // 150), replace=False)
            frame.loc[missing_idx, col] = np.nan
        frames.append(frame)

    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "machine_id"]).reset_index(drop=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"))
    args = parser.parse_args()
    df = generate_dataset(args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")
    print(df["failure_event"].value_counts().to_dict())

if __name__ == "__main__":
    main()
