# Live Customer Simulation UI

This Streamlit dashboard provides a visual demonstration of the predictive-maintenance project.

## Features

- Live simulation of multiple factory machines
- Five selectable customer scenarios
- Temperature, vibration, pressure, load and failure-risk movement
- Fleet health and machine-level health cards
- Warning and critical alerts
- Diagnostic factor explanation
- Work-order simulation
- Illustrative downtime and savings calculation
- Built-in demo mode that runs without TensorFlow
- Optional FastAPI mode that calls the project's `/predict` endpoint

## Install

From the repository root:

```bash
python -m pip install -r requirements-ui.txt
```

## Start in self-contained demo mode

```bash
streamlit run ui/dashboard.py
```

The browser normally opens automatically. Otherwise visit:

```text
http://localhost:8501
```

## Start with the trained FastAPI model

Terminal 1:

```bash
uvicorn api.main:app --reload --port 8000
```

Terminal 2:

```bash
streamlit run ui/dashboard.py
```

Inside the dashboard:

1. Enable **Use live FastAPI model**.
2. Keep the API URL as `http://localhost:8000`.
3. Set model window size to the value in `configs/config.json`.
4. Set model feature count to the second dimension expected by the trained model.

## Important integration note

The existing model expects preprocessed and scaled features, including engineered rolling features. The UI's API mode provides a generic demonstration window. For exact production scoring, add a raw-reading endpoint that reuses the saved preprocessing pipeline before calling the model.

The built-in simulation mode is recommended for immediate customer demonstrations because it does not depend on model availability or feature-shape compatibility.

## Suggested customer demonstration

1. Begin with **Normal plant operation**.
2. Explain the fleet-health cards.
3. Select **Gradual bearing wear**.
4. Turn on **Run live simulation**.
5. Open machine `M-002`.
6. Show vibration and failure probability rising.
7. Explain the warning or critical alert.
8. Show the diagnostic factors and estimated downtime avoided.
