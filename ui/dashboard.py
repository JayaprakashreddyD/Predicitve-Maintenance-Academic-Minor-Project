from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Predictive Maintenance Live Demo",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      .hero {
        border: 1px solid rgba(120,120,120,.24);
        border-radius: 18px;
        padding: 18px 20px;
        margin-bottom: 14px;
        background: linear-gradient(135deg, rgba(22,163,74,.10), rgba(2,132,199,.07));
      }
      .hero h1 {margin: 0 0 4px 0; font-size: 1.85rem;}
      .hero p {margin: 0; opacity: .8;}
      .machine-card {
        border: 1px solid rgba(120,120,120,.25);
        border-radius: 16px;
        padding: 14px;
        min-height: 138px;
      }
      .status-normal {border-left: 8px solid #16a34a;}
      .status-warning {border-left: 8px solid #f59e0b;}
      .status-critical {border-left: 8px solid #dc2626;}
      .small-label {font-size: .78rem; opacity: .7;}
      .big-value {font-size: 1.45rem; font-weight: 700;}
      .alert-row {
        border: 1px solid rgba(120,120,120,.22);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 8px;
      }
      .pulse {
        display:inline-block; width:10px; height:10px; border-radius:50%;
        background:#16a34a; margin-right:8px;
        box-shadow:0 0 0 rgba(22,163,74,.4);
        animation:pulse 1.6s infinite;
      }
      @keyframes pulse {
        0% {box-shadow:0 0 0 0 rgba(22,163,74,.45)}
        70% {box-shadow:0 0 0 10px rgba(22,163,74,0)}
        100% {box-shadow:0 0 0 0 rgba(22,163,74,0)}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class MachineState:
    machine_id: str
    machine_type: str
    health: float
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    load: float
    probability: float
    status: str
    degradation: float
    maintenance_due: int
    history: Deque[dict] = field(default_factory=lambda: deque(maxlen=80))


MACHINE_TYPES = ["Pump", "Motor", "Compressor", "Conveyor", "Robot Arm", "Cooling Unit"]


def classify_status(probability: float, temperature: float, vibration: float) -> str:
    if probability >= 0.85 and (temperature >= 82 or vibration >= 5.0):
        return "Critical"
    if probability >= 0.65:
        return "Warning"
    return "Normal"


def initialise_machine(index: int) -> MachineState:
    rng = random.Random(900 + index)
    degradation = rng.uniform(0.05, 0.58)
    load = rng.uniform(0.38, 0.82)
    temperature = 46 + 20 * load + 18 * degradation
    vibration = 1.0 + 2.0 * load + 2.5 * degradation
    pressure = 8.4 - 1.5 * load - 1.3 * degradation
    probability = min(0.97, max(0.02, 0.04 + degradation ** 2.3))
    status = classify_status(probability, temperature, vibration)
    return MachineState(
        machine_id=f"M-{index + 1:03d}",
        machine_type=MACHINE_TYPES[index % len(MACHINE_TYPES)],
        health=max(0, 100 * (1 - degradation)),
        temperature=temperature,
        vibration=vibration,
        pressure=pressure,
        rpm=1150 + 900 * load,
        load=load,
        probability=probability,
        status=status,
        degradation=degradation,
        maintenance_due=max(1, int(30 * (1 - degradation))),
    )


def reset_state(machine_count: int) -> None:
    st.session_state.machines = [initialise_machine(i) for i in range(machine_count)]
    st.session_state.alerts = deque(maxlen=30)
    st.session_state.tick = 0
    st.session_state.started_at = datetime.now()
    st.session_state.total_predictions = 0
    st.session_state.work_orders = 0


def simulated_probability(machine: MachineState) -> float:
    temperature_risk = max(0.0, (machine.temperature - 68) / 22)
    vibration_risk = max(0.0, (machine.vibration - 3.0) / 3.0)
    pressure_risk = max(0.0, (6.7 - machine.pressure) / 1.7)
    load_risk = max(0.0, (machine.load - 0.70) / 0.30)
    raw = (
        -3.2
        + 1.9 * temperature_risk
        + 2.1 * vibration_risk
        + 1.4 * pressure_risk
        + 1.0 * load_risk
        + 2.7 * machine.degradation
    )
    return float(1 / (1 + math.exp(-raw)))


def call_api(api_url: str, machine: MachineState, window_size: int, feature_count: int) -> float | None:
    # Demo windows are already scaled-like values. In a real deployment, send the
    # exact processed feature window produced by the repository preprocessor.
    row = [
        machine.temperature, machine.vibration, machine.pressure,
        machine.rpm, machine.load, machine.load * 85, 26.0, 55.0,
    ]
    while len(row) < feature_count:
        row.append(0.0)
    row = row[:feature_count]
    payload = {
        "machine_id": machine.machine_id,
        "sensor_window": [row for _ in range(window_size)],
    }
    try:
        response = requests.post(api_url.rstrip("/") + "/predict", json=payload, timeout=1.6)
        response.raise_for_status()
        return float(response.json()["failure_probability"])
    except Exception:
        return None


def advance_simulation(
    speed: float,
    scenario: str,
    api_enabled: bool,
    api_url: str,
    api_window_size: int,
    api_feature_count: int,
) -> None:
    st.session_state.tick += 1

    for index, machine in enumerate(st.session_state.machines):
        scenario_target = {
            "Normal plant operation": 0.0008,
            "Gradual bearing wear": 0.0035 if index == 1 else 0.0011,
            "Cooling system failure": 0.0050 if index == 2 else 0.0010,
            "High-load production surge": 0.0030,
            "Multiple machine incident": 0.0042 if index in {0, 2, 4} else 0.0012,
        }[scenario]

        machine.degradation = float(np.clip(
            machine.degradation + scenario_target * speed + random.uniform(-0.001, 0.001),
            0.02, 0.98,
        ))

        if scenario == "High-load production surge":
            target_load = 0.90
        elif scenario == "Cooling system failure" and index == 2:
            target_load = 0.78
        else:
            target_load = 0.55 + 0.18 * math.sin(st.session_state.tick / 8 + index)

        machine.load = float(np.clip(
            0.78 * machine.load + 0.22 * target_load + random.uniform(-0.025, 0.025),
            0.15, 1.0,
        ))

        cooling_penalty = 14 if scenario == "Cooling system failure" and index == 2 else 0
        bearing_penalty = 1.4 if scenario == "Gradual bearing wear" and index == 1 else 0

        machine.temperature = float(
            43 + 21 * machine.load + 20 * machine.degradation
            + cooling_penalty + random.uniform(-1.2, 1.2)
        )
        machine.vibration = float(
            0.9 + 2.1 * machine.load + 2.7 * machine.degradation
            + bearing_penalty + random.uniform(-0.18, 0.18)
        )
        machine.pressure = float(
            8.5 - 1.5 * machine.load - 1.4 * machine.degradation
            + random.uniform(-0.12, 0.12)
        )
        machine.rpm = float(1100 + 950 * machine.load + random.uniform(-35, 35))

        api_probability = None
        if api_enabled:
            api_probability = call_api(
                api_url, machine, api_window_size, api_feature_count
            )

        machine.probability = (
            api_probability if api_probability is not None
            else simulated_probability(machine)
        )
        old_status = machine.status
        machine.status = classify_status(
            machine.probability, machine.temperature, machine.vibration
        )
        machine.health = float(np.clip(100 * (1 - machine.degradation), 0, 100))
        machine.maintenance_due = max(0, int(30 * (1 - machine.degradation)))

        timestamp = datetime.now().strftime("%H:%M:%S")
        machine.history.append({
            "timestamp": timestamp,
            "temperature": machine.temperature,
            "vibration": machine.vibration,
            "pressure": machine.pressure,
            "failure_probability": machine.probability * 100,
            "health": machine.health,
        })

        if machine.status != old_status and machine.status in {"Warning", "Critical"}:
            message = {
                "time": timestamp,
                "machine_id": machine.machine_id,
                "machine_type": machine.machine_type,
                "status": machine.status,
                "probability": machine.probability,
                "temperature": machine.temperature,
                "vibration": machine.vibration,
            }
            st.session_state.alerts.appendleft(message)
            if machine.status == "Critical":
                st.session_state.work_orders += 1

        st.session_state.total_predictions += 1


def render_machine_card(machine: MachineState) -> None:
    status_class = {
        "Normal": "status-normal",
        "Warning": "status-warning",
        "Critical": "status-critical",
    }[machine.status]

    status_icon = {"Normal": "●", "Warning": "▲", "Critical": "◆"}[machine.status]
    st.markdown(
        f"""
        <div class="machine-card {status_class}">
          <div style="display:flex;justify-content:space-between;gap:8px;">
            <div>
              <div class="small-label">{machine.machine_type}</div>
              <div class="big-value">{machine.machine_id}</div>
            </div>
            <div style="text-align:right;">
              <div class="small-label">STATUS</div>
              <div><strong>{status_icon} {machine.status}</strong></div>
            </div>
          </div>
          <div style="margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:7px;">
            <div><span class="small-label">Health</span><br><strong>{machine.health:.0f}%</strong></div>
            <div><span class="small-label">Failure risk</span><br><strong>{machine.probability*100:.1f}%</strong></div>
            <div><span class="small-label">Temperature</span><br><strong>{machine.temperature:.1f} °C</strong></div>
            <div><span class="small-label">Vibration</span><br><strong>{machine.vibration:.2f} mm/s</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("Simulation controls")

    machine_count = st.slider("Number of machines", 3, 12, 6)
    scenario = st.selectbox(
        "Customer demonstration scenario",
        [
            "Normal plant operation",
            "Gradual bearing wear",
            "Cooling system failure",
            "High-load production surge",
            "Multiple machine incident",
        ],
        index=1,
    )
    speed = st.slider("Simulation speed", 0.5, 5.0, 1.5, 0.5)
    refresh_seconds = st.slider("Refresh interval (seconds)", 0.5, 4.0, 1.0, 0.5)

    st.divider()
    st.subheader("Prediction source")
    api_enabled = st.toggle("Use live FastAPI model", value=False)
    api_url = st.text_input("FastAPI base URL", "http://localhost:8000", disabled=not api_enabled)
    api_window_size = st.number_input("Model window size", 1, 500, 36, disabled=not api_enabled)
    api_feature_count = st.number_input("Model feature count", 1, 200, 43, disabled=not api_enabled)

    st.caption(
        "Simulation mode works without a trained model. API mode sends windows to "
        "your repository's `/predict` endpoint."
    )

    st.divider()
    run_simulation = st.toggle("Run live simulation", value=False)
    col_a, col_b = st.columns(2)
    with col_a:
        step_once = st.button("Step once", use_container_width=True)
    with col_b:
        reset_clicked = st.button("Reset", use_container_width=True)

if "machines" not in st.session_state or len(st.session_state.machines) != machine_count:
    reset_state(machine_count)

if reset_clicked:
    reset_state(machine_count)

if run_simulation or step_once:
    advance_simulation(
        speed=speed,
        scenario=scenario,
        api_enabled=api_enabled,
        api_url=api_url,
        api_window_size=int(api_window_size),
        api_feature_count=int(api_feature_count),
    )

st.markdown(
    f"""
    <div class="hero">
      <h1>Predictive Maintenance & Health Monitoring — Live Plant Simulation</h1>
      <p><span class="pulse"></span>Scenario: <strong>{scenario}</strong> ·
      Prediction source: <strong>{"FastAPI model" if api_enabled else "Built-in demo engine"}</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

machines = st.session_state.machines
normal_count = sum(m.status == "Normal" for m in machines)
warning_count = sum(m.status == "Warning" for m in machines)
critical_count = sum(m.status == "Critical" for m in machines)
fleet_health = np.mean([m.health for m in machines])
average_risk = np.mean([m.probability for m in machines]) * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fleet health", f"{fleet_health:.1f}%")
k2.metric("Average failure risk", f"{average_risk:.1f}%")
k3.metric("Warnings", warning_count)
k4.metric("Critical machines", critical_count)
k5.metric("Work orders created", st.session_state.work_orders)

st.subheader("Machine health overview")
card_columns = st.columns(3)
for index, machine in enumerate(machines):
    with card_columns[index % 3]:
        render_machine_card(machine)

st.divider()

left, right = st.columns([1.65, 1])

with left:
    st.subheader("Live sensor and risk trend")
    selected_machine_id = st.selectbox(
        "Machine to inspect",
        [m.machine_id for m in machines],
        index=min(1, len(machines) - 1),
    )
    selected = next(m for m in machines if m.machine_id == selected_machine_id)

    if len(selected.history) == 0:
        for _ in range(12):
            advance_simulation(
                speed=0.35,
                scenario="Normal plant operation",
                api_enabled=False,
                api_url=api_url,
                api_window_size=int(api_window_size),
                api_feature_count=int(api_feature_count),
            )

    history_df = pd.DataFrame(selected.history)
    if not history_df.empty:
        history_df = history_df.set_index("timestamp")
        tab1, tab2, tab3 = st.tabs(["Failure risk", "Sensors", "Machine health"])
        with tab1:
            st.line_chart(history_df[["failure_probability"]], height=270)
            st.caption("Warning threshold: 65% · Critical threshold: 85% plus unsafe sensor evidence.")
        with tab2:
            st.line_chart(history_df[["temperature", "vibration", "pressure"]], height=270)
        with tab3:
            st.line_chart(history_df[["health"]], height=270)

with right:
    st.subheader("Current diagnostic explanation")
    factors = [
        ("Temperature stress", max(0, min(100, (selected.temperature - 60) / 30 * 100))),
        ("Vibration stress", max(0, min(100, (selected.vibration - 2) / 4 * 100))),
        ("Pressure loss", max(0, min(100, (7.5 - selected.pressure) / 2.2 * 100))),
        ("High load", max(0, min(100, (selected.load - 0.45) / 0.55 * 100))),
        ("Long-term degradation", selected.degradation * 100),
    ]
    for name, value in factors:
        st.write(f"**{name}** — {value:.0f}%")
        st.progress(int(value))

    recommendation = (
        "Open an urgent work order and request immediate technician confirmation."
        if selected.status == "Critical"
        else "Schedule inspection during the next planned maintenance window."
        if selected.status == "Warning"
        else "Continue monitoring. No maintenance action is required."
    )
    st.info(recommendation)

st.divider()
alert_col, business_col = st.columns([1.25, 1])

with alert_col:
    st.subheader("Live alerts and maintenance actions")
    if not st.session_state.alerts:
        st.success("No active warning transitions have been detected.")
    else:
        for alert in list(st.session_state.alerts)[:8]:
            icon = "🚨" if alert["status"] == "Critical" else "⚠️"
            st.markdown(
                f"""
                <div class="alert-row">
                  <strong>{icon} {alert["status"]}: {alert["machine_id"]}</strong>
                  <div class="small-label">{alert["machine_type"]} · {alert["time"]}</div>
                  Risk {alert["probability"]*100:.1f}% ·
                  Temperature {alert["temperature"]:.1f} °C ·
                  Vibration {alert["vibration"]:.2f} mm/s
                </div>
                """,
                unsafe_allow_html=True,
            )

with business_col:
    st.subheader("Customer value simulation")
    predicted_failures = warning_count + critical_count
    avoided_downtime_hours = critical_count * 8 + warning_count * 3
    estimated_savings = avoided_downtime_hours * 45000
    st.metric("Potential incidents detected", predicted_failures)
    st.metric("Estimated downtime avoided", f"{avoided_downtime_hours} hours")
    st.metric("Illustrative production savings", f"₹{estimated_savings:,.0f}")
    st.caption(
        "These values are demonstration assumptions. Replace them with each customer's "
        "actual downtime cost, maintenance cost and production-loss model."
    )

with st.expander("How to narrate this demo to a customer"):
    st.markdown(
        """
        1. Start with normal machines and explain that the system continuously watches sensor patterns.
        2. Select **Gradual bearing wear** or **Cooling system failure**.
        3. Start the simulation and show temperature, vibration and risk increasing together.
        4. Explain that the AI does not wait for a breakdown; it estimates future failure probability.
        5. Show the alert, diagnostic factors and recommended maintenance action.
        6. Finish with estimated avoided downtime and savings using the customer's own numbers.
        """
    )

if run_simulation:
    time.sleep(refresh_seconds)
    st.rerun()
