from __future__ import annotations
from pathlib import Path
import time
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "predictive_maintenance.keras"
PREPROCESSOR_PATH = ROOT / "data" / "processed" / "preprocessor.joblib"

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")
model = None
preprocessor = None

class PredictionRequest(BaseModel):
    machine_id: str = Field(examples=["M-001"])
    sensor_window: list[list[float]]

@app.on_event("startup")
def load_artifacts() -> None:
    global model, preprocessor
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(503, "Model is not available. Run training first.")
    array = np.asarray(request.sensor_window, dtype=np.float32)
    if array.ndim != 2:
        raise HTTPException(422, "sensor_window must be a 2-D list [time, features].")
    expected = tuple(model.input_shape[1:])
    if tuple(array.shape) != expected:
        raise HTTPException(422, f"Expected sensor window shape {expected}, received {array.shape}.")
    start = time.perf_counter()
    probability = float(model.predict(array[None, ...], verbose=0)[0, 0])
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "machine_id": request.machine_id,
        "failure_probability": probability,
        "alert": probability >= 0.65,
        "latency_ms": latency_ms,
        "model_version": "1"
    }
