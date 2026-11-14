import pandas as pd
from src.preprocess import add_failure_horizon_label

def test_future_failure_label():
    df = pd.DataFrame({
        "machine_id": ["M1"] * 5,
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"),
        "failure_event": [0, 0, 0, 1, 0]
    })
    out = add_failure_horizon_label(df, horizon_hours=2, frequency_minutes=60)
    assert out["failure_within_horizon"].tolist()[1] == 1
