"""
predict.py
Load the trained model and predict health status for a new person
given: systolic_bp, diastolic_bp, temperature_f, spo2

Usage:
    python3 predict.py --sys 118 --dia 76 --temp 98.6 --spo2 97
"""

import argparse
import joblib
import numpy as np
import pandas as pd

FEATURES = ["systolic_bp", "diastolic_bp", "temperature_f", "spo2"]

def load_artifacts():
    model = joblib.load("health_model.joblib")
    scaler = joblib.load("scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    return model, scaler, le


def predict(sys_bp, dia_bp, temp, spo2):
    model, scaler, le = load_artifacts()

    X = pd.DataFrame([[sys_bp, dia_bp, temp, spo2]], columns=FEATURES)
    X_scaled = scaler.transform(X)

    pred_idx = model.predict(X_scaled)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    proba = model.predict_proba(X_scaled)[0]
    proba_dict = {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)}

    return pred_label, proba_dict


def explain(sys_bp, dia_bp, temp, spo2):
    """Simple human-readable reasoning for the prediction (explainability)."""
    notes = []
    if spo2 < 90:
        notes.append(f"SpO2 {spo2}% is critically low (<90%)")
    elif spo2 < 95:
        notes.append(f"SpO2 {spo2}% is below normal (90-94%)")

    if sys_bp >= 140 or dia_bp >= 90:
        notes.append(f"BP {sys_bp}/{dia_bp} indicates hypertensive range")
    elif sys_bp <= 80 or dia_bp <= 50:
        notes.append(f"BP {sys_bp}/{dia_bp} indicates hypotensive range")
    elif sys_bp > 120 or dia_bp > 80:
        notes.append(f"BP {sys_bp}/{dia_bp} is elevated")

    if temp >= 103 or temp <= 95.0:
        notes.append(f"Temperature {temp}°F is in a dangerous range")
    elif temp >= 99.1 or temp <= 96.9:
        notes.append(f"Temperature {temp}°F is outside normal range")

    return notes if notes else ["All vitals within normal range"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys", type=float, required=True, help="Systolic BP (mmHg)")
    parser.add_argument("--dia", type=float, required=True, help="Diastolic BP (mmHg)")
    parser.add_argument("--temp", type=float, required=True, help="Temperature (F)")
    parser.add_argument("--spo2", type=float, required=True, help="Oxygen saturation (%)")
    args = parser.parse_args()

    label, proba = predict(args.sys, args.dia, args.temp, args.spo2)
    notes = explain(args.sys, args.dia, args.temp, args.spo2)

    print(f"\nInput -> Sys BP: {args.sys}, Dia BP: {args.dia}, Temp: {args.temp}F, SpO2: {args.spo2}%")
    print(f"Predicted Health Status: {label.upper()}")
    print(f"Confidence scores: {proba}")
    print("Reasoning:")
    for n in notes:
        print("  -", n)
