from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import load_config, PROJECT_ROOT

BASE_FEATURES = [
    "temperature", "vibration", "pressure", "rpm", "load",
    "production_rate", "ambient_temp", "humidity"
]

@dataclass
class PreparedData:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_meta: pd.DataFrame
    feature_names: list[str]
    scaler: StandardScaler

def add_failure_horizon_label(df: pd.DataFrame, horizon_hours: int, frequency_minutes: int) -> pd.DataFrame:
    result = df.sort_values(["machine_id", "timestamp"]).copy()
    horizon_steps = max(1, int(horizon_hours * 60 / frequency_minutes))
    labels = np.zeros(len(result), dtype=np.int8)
    for _, idx in result.groupby("machine_id", sort=False).groups.items():
        pos = np.asarray(list(idx))
        failures = result.loc[pos, "failure_event"].to_numpy(dtype=int)
        # Future event within (t, t+horizon].
        future = pd.Series(failures[::-1]).rolling(horizon_steps, min_periods=1).max().to_numpy()[::-1]
        labels[pos] = np.roll(future, -1)
        labels[pos[-1]] = 0
    result["failure_within_horizon"] = labels
    return result

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    result[BASE_FEATURES] = (
        result.groupby("machine_id")[BASE_FEATURES]
        .transform(lambda x: x.interpolate(limit_direction="both").ffill().bfill())
    )
    engineered = list(BASE_FEATURES)
    for col in ["temperature", "vibration", "pressure", "rpm", "load"]:
        grouped = result.groupby("machine_id")[col]
        for window in (6, 18, 36):
            mean_col = f"{col}_roll_mean_{window}"
            std_col = f"{col}_roll_std_{window}"
            result[mean_col] = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
            result[std_col] = grouped.transform(lambda s: s.rolling(window, min_periods=2).std()).fillna(0.0)
            engineered.extend([mean_col, std_col])
        diff_col = f"{col}_diff"
        result[diff_col] = grouped.diff().fillna(0.0)
        engineered.append(diff_col)
    return result, engineered

def create_windows(frame: pd.DataFrame, feature_names: list[str], window_size: int, stride: int):
    xs, ys, meta_rows = [], [], []
    for machine_id, group in frame.groupby("machine_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        values = group[feature_names].to_numpy(dtype=np.float32)
        labels = group["failure_within_horizon"].to_numpy(dtype=np.float32)
        for end in range(window_size - 1, len(group), stride):
            start = end - window_size + 1
            xs.append(values[start:end+1])
            ys.append(labels[end])
            meta_rows.append({
                "timestamp": group.loc[end, "timestamp"],
                "machine_id": machine_id,
                "failure_event": int(group.loc[end, "failure_event"])
            })
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta_rows)

def prepare_data(input_path: str | Path, config_path: str | None = None) -> PreparedData:
    cfg = load_config(config_path)
    dcfg, tcfg = cfg["data"], cfg["training"]
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = add_failure_horizon_label(df, int(dcfg["failure_horizon_hours"]), int(dcfg["frequency_minutes"]))
    df, features = engineer_features(df)

    timestamps = np.sort(df["timestamp"].unique())
    n = len(timestamps)
    train_end = timestamps[int(n * (1 - tcfg["validation_fraction"] - tcfg["test_fraction"])) - 1]
    val_end = timestamps[int(n * (1 - tcfg["test_fraction"])) - 1]

    train_df = df[df["timestamp"] <= train_end].copy()
    val_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].copy()
    test_df = df[df["timestamp"] > val_end].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[features])
    for split in (train_df, val_df, test_df):
        split.loc[:, features] = scaler.transform(split[features])

    args = (features, int(dcfg["window_size"]), int(dcfg["stride"]))
    train_x, train_y, _ = create_windows(train_df, *args)
    val_x, val_y, _ = create_windows(val_df, *args)
    test_x, test_y, test_meta = create_windows(test_df, *args)
    return PreparedData(train_x, train_y, val_x, val_y, test_x, test_y, test_meta, features, scaler)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "raw" / "sensor_data.csv"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "processed" / "prepared.npz"))
    args = parser.parse_args()
    prepared = prepare_data(args.input, args.config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_x=prepared.train_x, train_y=prepared.train_y,
        val_x=prepared.val_x, val_y=prepared.val_y,
        test_x=prepared.test_x, test_y=prepared.test_y
    )
    joblib.dump(
        {"scaler": prepared.scaler, "feature_names": prepared.feature_names},
        output.parent / "preprocessor.joblib"
    )
    prepared.test_meta.to_csv(output.parent / "test_meta.csv", index=False)
    print("Shapes:", prepared.train_x.shape, prepared.val_x.shape, prepared.test_x.shape)
    print("Positive rates:", prepared.train_y.mean(), prepared.val_y.mean(), prepared.test_y.mean())

if __name__ == "__main__":
    main()
