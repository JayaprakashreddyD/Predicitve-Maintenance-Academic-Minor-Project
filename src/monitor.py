from __future__ import annotations
import numpy as np

def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Simple PSI drift score; larger values imply stronger distribution shift."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.maximum(ref_hist / max(ref_hist.sum(), 1), 1e-6)
    cur_pct = np.maximum(cur_hist / max(cur_hist.sum(), 1), 1e-6)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def alert_decision(probability: float, vibration: float, temperature: float,
                   probability_threshold: float = 0.65) -> dict:
    rule_confirmation = vibration >= 5.0 or temperature >= 82.0
    level = "critical" if probability >= 0.85 and rule_confirmation else (
        "warning" if probability >= probability_threshold else "normal"
    )
    return {
        "level": level,
        "open_work_order": level == "critical",
        "technician_review": level in {"warning", "critical"},
        "reason": "Model probability combined with sensor safety rules."
    }
