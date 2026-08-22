"""
generate_data.py
Generates a synthetic but medically-grounded dataset of:
    - Systolic BP, Diastolic BP (mmHg)
    - Body Temperature (°F)
    - SpO2 / Oxygen Saturation (%)
and labels each record as: Normal, At Risk, or Critical
based on standard clinical reference ranges.

Reference ranges used (approx, for adults):
  BP (Systolic/Diastolic):
    Normal:      90-120 / 60-80
    At Risk:     120-140 / 80-90  (elevated / stage 1 hypertension) OR <90/<60 (mild hypotension)
    Critical:    >140/>90 (hypertensive crisis) OR <80/<50 (severe hypotension)
  Temperature (°F):
    Normal:      97.0 - 99.0
    At Risk:     99.1 - 100.9 (mild fever) OR 95.1-96.9 (mild hypothermia)
    Critical:    >=103 (high fever) OR <=95.0 (hypothermia)
  SpO2 (%):
    Normal:      95-100
    At Risk:     90-94
    Critical:    <90
"""

import numpy as np
import pandas as pd

np.random.seed(42)

def generate_record():
    # Randomly decide the "true" health state to keep classes balanced
    state = np.random.choice(["normal", "at_risk", "critical"], p=[0.45, 0.35, 0.20])

    if state == "normal":
        sys_bp = np.random.uniform(95, 120)
        dia_bp = np.random.uniform(60, 80)
        temp = np.random.uniform(97.0, 99.0)
        spo2 = np.random.uniform(95, 100)
    elif state == "at_risk":
        # mix of elevated/low BP, mild fever/hypothermia, mild hypoxia
        sys_bp = np.random.choice([np.random.uniform(120, 140), np.random.uniform(85, 90)])
        dia_bp = np.random.choice([np.random.uniform(80, 90), np.random.uniform(55, 60)])
        temp = np.random.choice([np.random.uniform(99.1, 100.9), np.random.uniform(95.1, 96.9)])
        spo2 = np.random.uniform(90, 94)
    else:  # critical
        sys_bp = np.random.choice([np.random.uniform(140, 190), np.random.uniform(60, 80)])
        dia_bp = np.random.choice([np.random.uniform(90, 120), np.random.uniform(40, 50)])
        temp = np.random.choice([np.random.uniform(103, 106), np.random.uniform(92, 95)])
        spo2 = np.random.uniform(70, 89)

    return round(sys_bp, 1), round(dia_bp, 1), round(temp, 1), round(spo2, 1), state


def label_from_rules(sys_bp, dia_bp, temp, spo2):
    """Independent rule-based labeler used as ground truth (clinically inspired)."""
    critical = (
        sys_bp >= 140 or sys_bp <= 80 or
        dia_bp >= 90 or dia_bp <= 50 or
        temp >= 103 or temp <= 95.0 or
        spo2 < 90
    )
    if critical:
        return "critical"

    at_risk = (
        (120 < sys_bp < 140) or (80 <= sys_bp < 90) or
        (80 < dia_bp < 90) or (50 < dia_bp < 60) or
        (99.1 <= temp < 103) or (95.0 < temp < 97.0) or
        (90 <= spo2 < 95)
    )
    if at_risk:
        return "at_risk"

    return "normal"


def build_dataset(n=5000):
    rows = []
    for _ in range(n):
        sys_bp, dia_bp, temp, spo2, _ = generate_record()
        label = label_from_rules(sys_bp, dia_bp, temp, spo2)
        rows.append([sys_bp, dia_bp, temp, spo2, label])

    df = pd.DataFrame(rows, columns=[
        "systolic_bp", "diastolic_bp", "temperature_f", "spo2", "health_status"
    ])
    return df


if __name__ == "__main__":
    df = build_dataset(5000)
    df.to_csv("health_data.csv", index=False)
    print(df["health_status"].value_counts())
    print("\nSaved health_data.csv with", len(df), "records")
