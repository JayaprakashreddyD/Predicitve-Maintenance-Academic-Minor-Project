from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from .config import load_config, PROJECT_ROOT
from .model import build_model

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default=str(PROJECT_ROOT / "data" / "processed" / "prepared.npz"))
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    data = np.load(args.prepared)
    train_x, train_y = data["train_x"], data["train_y"]
    val_x, val_y = data["val_x"], data["val_y"]

    tf.keras.utils.set_random_seed(cfg["seed"])
    model = build_model(train_x.shape[1], train_x.shape[2], float(tcfg["learning_rate"]))
    model_dir = PROJECT_ROOT / "models"
    report_dir = PROJECT_ROOT / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(model_dir / "best.keras"), monitor="val_pr_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(PROJECT_ROOT / "monitoring" / "tensorboard"))
    ]
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=int(tcfg["epochs"]),
        batch_size=int(tcfg["batch_size"]),
        callbacks=callbacks,
        verbose=2
    )
    model.save(model_dir / "predictive_maintenance.keras")
    export_path = model_dir / "predictive_maintenance" / "1"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.export(export_path)
    with (report_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"Saved Keras model and TF-Serving export under {model_dir}")

if __name__ == "__main__":
    main()
