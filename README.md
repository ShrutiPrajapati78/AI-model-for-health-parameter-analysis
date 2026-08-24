# Health Parameter Analysis — AI/ML Model

Predicts health status (**Normal / At Risk / Critical**) from:
- Systolic & Diastolic Blood Pressure (mmHg)
- Body Temperature (°F)
- SpO2 / Oxygen Saturation (%)

## Files
| File | Purpose |
|---|---|
| `generate_data.py` | Builds a synthetic dataset (`health_data.csv`) using clinically-grounded thresholds |
| `train_model.py` | Trains a Random Forest classifier (+ Logistic Regression baseline), saves model artifacts |
| `predict.py` | Loads the saved model and predicts status for new input, with human-readable reasoning |
| `health_model.joblib`, `scaler.joblib`, `label_encoder.joblib` | Saved trained model artifacts |

## Run it

```bash
pip install pandas numpy scikit-learn joblib

python3 generate_data.py      # creates health_data.csv
python3 train_model.py        # trains & saves model
python3 predict.py --sys 118 --dia 76 --temp 98.6 --spo2 97
```

## Clinical reference ranges used
| Parameter | Normal | At Risk | Critical |
|---|---|---|---|
| Systolic BP | 90–120 | 120–140 or 85–90 | ≥140 or ≤80 |
| Diastolic BP | 60–80 | 80–90 or 55–60 | ≥90 or ≤50 |
| Temperature | 97.0–99.0°F | 99.1–100.9°F or 95.1–96.9°F | ≥103°F or ≤95.0°F |
| SpO2 | 95–100% | 90–94% | <90% |

> These are general adult reference ranges for demo purposes — **not a substitute for actual medical guidelines**. For a production/healthcare product, validate thresholds with a clinician and ideally train on real (de-identified, consented) patient data instead of synthetic data.

## Using this from your Next.js app (since you're not on Node/Express)

Since the model is Python-based, the cleanest way to call it from a **Next.js API Route** is to either:

1. **Spin up a tiny Python inference service** (FastAPI, one file) and call it from your API route with `fetch()`, OR
2. **Export the model to run in Python only, deployed separately** (e.g. as a serverless function via AWS Lambda / GCP Cloud Function using Python runtime — fits well with your GCP experience) and have the Next.js API route call that endpoint.

Example Next.js API route (`app/api/health-check/route.ts`) calling a deployed Python inference endpoint:

```ts
export async function POST(req: Request) {
  const { sys, dia, temp, spo2 } = await req.json();

  const res = await fetch(process.env.HEALTH_MODEL_ENDPOINT!, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sys, dia, temp, spo2 }),
  });

  const result = await res.json();
  return Response.json(result);
}
```

A minimal FastAPI wrapper around `predict.py`'s `predict()` function would be a natural next step if you want to expose this as an HTTP endpoint.

## Streamlit app (`streamlit_app.py`) — for presentation/demo

**Full-featured version with:**
- 🔐 **Login + Sign Up** — multi-user (username/password), and anyone can create their own account via the "Create Account" tab on the login screen — no need to share one login for everyone
- 🎨 **Custom dark theme** with branded header, colored result cards, gauge charts
- 📜 **Prediction history** — every prediction is saved (per user) to `history.csv` and shown in a History tab with a status trend chart
- 📄 **PDF report download** — generates a downloadable report for each prediction

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. You'll see a login screen first.

**Demo logins** (defined in `config.yaml` — change before presenting to DRDO if you want custom names):
| Username | Password |
|---|---|
| shruti | shruti123 |
| admin | admin123 |
| guest | guest123 |

**To add/change users:** either use the in-app "Create Account" tab (easiest), or manually edit `config.yaml` — add a new entry under `credentials: usernames:` with `email`, `name`, and a plain-text `password`.

⚠️ **Streamlit Cloud caveat:** new accounts created via "Create Account" are saved to `config.yaml` on the server's filesystem. On Streamlit Community Cloud's free tier, this storage is **ephemeral** — if the app restarts (which happens periodically, e.g., after inactivity or a redeploy), any accounts created through the app (not the ones already in the committed `config.yaml`) will be lost. This is fine for a live demo but not for permanent multi-user accounts — mention this if asked during presentation.

⚠️ **Security note:** This is a simple demo-grade auth suitable for a college/internship presentation — passwords live in a local YAML file in plain text. It is **not** production-grade security (no password hashing, no reset flow, no rate-limiting). Good enough to show "the app requires login and users can sign up" in a demo; not something to expose on the public internet with real patient data.

## FastAPI service (`app.py`)

A ready-to-run HTTP API wrapping the model, with request validation and CORS enabled.

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Endpoints:
- `GET /health` — health check
- `GET /docs` — interactive Swagger UI (auto-generated)
- `POST /predict` — body: `{"systolic_bp": 128, "diastolic_bp": 84, "temperature_f": 100.2, "spo2": 92}`

Response:
```json
{
  "status": "at_risk",
  "confidence": {"at_risk": 1.0, "critical": 0.0, "normal": 0.0},
  "reasoning": ["SpO2 92.0% is below normal (90-94%)", "BP 128.0/84.0 is elevated", "Temperature 100.2°F is outside normal range"]
}
```

Deploy this on **GCP Cloud Run** (Dockerize it) or **Cloud Functions** — fits naturally with your existing GCP experience. Then set `HEALTH_MODEL_ENDPOINT` in your Next.js app to the deployed URL and call it from your API route as shown above.

## Extending this
- Add more vitals: heart rate, respiratory rate, glucose level
- Swap synthetic data for a real anonymized dataset (e.g. MIMIC-III/IV, PhysioNet) for more realistic modeling
- Try XGBoost or a small neural net for comparison
- Add SHAP for proper explainability instead of the simple rule-based `explain()` function
