"""
app.py
FastAPI wrapper around the trained health-status model.

Run locally:
    pip install fastapi uvicorn pandas scikit-learn joblib
    uvicorn app:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"systolic_bp": 128, "diastolic_bp": 84, "temperature_f": 100.2, "spo2": 92}'

Deploy this as a GCP Cloud Function / Cloud Run service, then point your
Next.js API route's HEALTH_MODEL_ENDPOINT env var to the deployed URL.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

FEATURES = ["systolic_bp", "diastolic_bp", "temperature_f", "spo2"]

app = FastAPI(title="Health Status Prediction API", version="1.0")

# Allow calls from your Next.js frontend (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts once at startup
model = joblib.load("health_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")


class VitalsInput(BaseModel):
    systolic_bp: float = Field(..., gt=0, le=300, description="Systolic BP in mmHg")
    diastolic_bp: float = Field(..., gt=0, le=200, description="Diastolic BP in mmHg")
    temperature_f: float = Field(..., gt=80, le=115, description="Body temperature in Fahrenheit")
    spo2: float = Field(..., gt=0, le=100, description="Oxygen saturation in %")


class PredictionResponse(BaseModel):
    status: str
    confidence: dict
    reasoning: list[str]


def explain(sys_bp, dia_bp, temp, spo2):
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


@app.get("/")
def root():
    return {"message": "Health Status Prediction API is running", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(vitals: VitalsInput):
    try:
        X = pd.DataFrame([[
            vitals.systolic_bp, vitals.diastolic_bp,
            vitals.temperature_f, vitals.spo2
        ]], columns=FEATURES)

        X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        proba = model.predict_proba(X_scaled)[0]
        confidence = {
            cls: round(float(p), 3)
            for cls, p in zip(label_encoder.classes_, proba)
        }

        reasoning = explain(
            vitals.systolic_bp, vitals.diastolic_bp,
            vitals.temperature_f, vitals.spo2
        )

        return PredictionResponse(
            status=pred_label,
            confidence=confidence,
            reasoning=reasoning,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
