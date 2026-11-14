import numpy as np
from src.monitor import population_stability_index, alert_decision

def test_psi_identical_is_small():
    x = np.linspace(0, 1, 1000)
    assert population_stability_index(x, x) < 1e-9

def test_critical_alert_requires_confirmation():
    result = alert_decision(0.92, vibration=5.5, temperature=75)
    assert result["level"] == "critical"
    assert result["open_work_order"] is True
