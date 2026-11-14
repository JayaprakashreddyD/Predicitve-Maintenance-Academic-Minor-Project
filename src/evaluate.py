from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
)
from .config import load_config, PROJECT_ROOT
from .model import focal_loss

def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(np.nanargmax(f1[:-1])) if thresholds.size else 0
    return float(thresholds[best]) if thresholds.size else 0.5

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default=str(PROJECT_ROOT / "data" / "processed" / "prepared.npz"))
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "best.keras"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = np.load(args.prepared)
    model = tf.keras.models.load_model(args.model, custom_objects={"loss": focal_loss()})
    val_prob = model.predict(data["val_x"], verbose=0).reshape(-1)
    test_prob = model.predict(data["test_x"], verbose=0).reshape(-1)
    threshold = choose_threshold(data["val_y"], val_prob)
    pred = (test_prob >= threshold).astype(int)
    y = data["test_y"].astype(int)

    metrics = {
        "threshold": threshold,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, test_prob)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, test_prob)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, zero_division=0, output_dict=True)
    }
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    meta_path = PROJECT_ROOT / "data" / "processed" / "test_meta.csv"
    meta = pd.read_csv(meta_path)
    meta["actual"] = y
    meta["failure_probability"] = test_prob
    meta["alert"] = pred
    meta.to_csv(report_dir / "test_predictions.csv", index=False)

    precision, recall, _ = precision_recall_curve(y, test_prob)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(report_dir / "precision_recall_curve.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
