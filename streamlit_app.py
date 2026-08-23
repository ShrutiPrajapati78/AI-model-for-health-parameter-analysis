"""
streamlit_app.py
Health Status Prediction — full-featured Streamlit web app.

Features:
    - Multi-user login (username/password) — simple session-based auth
    - Custom themed UI (sidebar branding, colored result cards, gauges)
    - Prediction history per user (saved to CSV, viewable in-app)
    - Downloadable PDF report for each prediction

Run:
    pip install -r requirements.txt
    python -m streamlit run streamlit_app.py

Default login credentials (change these in config.yaml before presenting!):
    username: shruti   | password: shruti123
    username: admin    | password: admin123
    username: guest    | password: guest123
"""

import streamlit as st
import pandas as pd
import joblib
import yaml
from yaml.loader import SafeLoader
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

FEATURES = ["systolic_bp", "diastolic_bp", "temperature_f", "spo2"]
HISTORY_FILE = Path("history.csv")

# ---------- Page config ----------
st.set_page_config(
    page_title="Health Status Prediction",
    page_icon="🩺",
    layout="wide",
)

# ---------- Custom theme (CSS) ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

    .stApp {
        background: radial-gradient(circle at 10% 0%, #0f1c3f 0%, #060a16 55%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1530 0%, #090d1e 100%);
        border-right: 1px solid rgba(96,165,250,0.15);
    }
    section[data-testid="stSidebar"] * { color: #dbe4ff; }

    h1, h2, h3 { color: #f1f5ff; letter-spacing: 0.3px; }
    p, span, label, .stMarkdown { color: #c7d2e8; }

    /* Top banner */
    .top-banner {
        background: linear-gradient(120deg, #1e3a8a 0%, #0891b2 100%);
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 28px;
        box-shadow: 0 8px 30px rgba(8,145,178,0.25);
        display: flex; align-items: center; gap: 18px;
    }
    .top-banner .icon-badge {
        background: rgba(255,255,255,0.15);
        border-radius: 14px;
        width: 58px; height: 58px;
        display: flex; align-items: center; justify-content: center;
        font-size: 30px;
        flex-shrink: 0;
    }
    .top-banner h1 { margin: 0; color: #ffffff; font-size: 26px; }
    .top-banner p { margin: 4px 0 0 0; color: #dff3fa; font-size: 14px; }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        height: 3em;
        background: linear-gradient(120deg, #2563eb, #0891b2);
        color: white;
        border: none;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(37,99,235,0.35);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37,99,235,0.5);
    }
    .stDownloadButton>button {
        border-radius: 10px; font-weight: 600; height: 3em;
        background: #10182b; color: #7dd3fc; border: 1px solid #1e3a5f;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 18px;
        backdrop-filter: blur(6px);
    }
    .result-card {
        border-radius: 16px;
        padding: 26px;
        text-align: center;
        margin-bottom: 22px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }
    .brand-header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }

    /* Sidebar user chip */
    .user-chip {
        background: rgba(96,165,250,0.1);
        border: 1px solid rgba(96,165,250,0.25);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 14px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        color: #9fb0d0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(8,145,178,0.18) !important;
        color: #7dd3fc !important;
    }

    div[data-testid="stMetricValue"] { color: #60a5fa; }

    /* Divider */
    hr { border-color: rgba(148,163,184,0.15) !important; }

    /* Login form container polish */
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 18px;
        padding: 10px 26px 22px 26px;
        max-width: 420px;
        margin: 10px auto 0 auto;
        box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    }

    footer, #MainMenu { visibility: hidden; }
    .app-footer {
        text-align: center; color: #64748b; font-size: 12.5px;
        padding: 18px 0 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Auth setup (simple, custom — no external cookie component) ----------
with open("config.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

USERS = config["credentials"]["usernames"]

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.name = None

if not st.session_state.authenticated:
    st.markdown(
        """
        <div style="text-align:center; margin-top:30px; margin-bottom:10px;">
            <div style="font-size:44px;">🩺</div>
            <h1 style="margin:6px 0 2px 0;">Health Status Prediction System</h1>
            <p style="color:#93a4c7; font-size:14px;">DRDO Internship Project &nbsp;•&nbsp; AI/ML Vitals Monitoring</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.form("login_form"):
            st.subheader("Login")
            input_user = st.text_input("Username")
            input_pass = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            user_record = USERS.get(input_user)
            if user_record and str(user_record.get("password")) == input_pass:
                st.session_state.authenticated = True
                st.session_state.username = input_user
                st.session_state.name = user_record.get("name", input_user)
                st.rerun()
            else:
                st.error("❌ Username / password incorrect")

    st.info("")
    st.stop()
    # 🔑 Demo logins — **shruti / shruti123**, **admin / admin123**, **guest / guest123

# ---------- Authenticated from here on ----------
username = st.session_state.username
name = st.session_state.name

with st.sidebar:
    st.markdown(
        f"""
        <div class="user-chip">
            <div style="font-size:13px; color:#93a4c7;">Signed in as</div>
            <div style="font-size:16px; font-weight:600; color:#f1f5ff;">👤 {name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.name = None
        st.rerun()
    st.divider()

# ---------- Load model (cached) ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("health_model.joblib")
    scaler = joblib.load("scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    return model, scaler, le

model, scaler, label_encoder = load_artifacts()

STATUS_COLORS = {"normal": "#22c55e", "at_risk": "#f59e0b", "critical": "#ef4444"}
STATUS_LABELS = {"normal": "✅ Normal", "at_risk": "⚠️ At Risk", "critical": "🚨 Critical"}


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


def get_recommendations(status):
    """Status-based general triage guidance — not a diagnosis or treatment plan."""
    if status == "critical":
        return {
            "urgency": "🚨 Seek Emergency Medical Attention Immediately",
            "color": "#ef4444",
            "actions": [
                "Contact emergency medical services right away — in India, call 108 or 112",
                "Do not drive yourself — have someone else take you, or wait for an ambulance",
                "Go to the nearest hospital emergency department",
                "If someone is with the person, keep them calm, seated/lying down, and monitored until help arrives",
                "Bring a list of current medications and medical history if possible",
            ],
        }
    elif status == "at_risk":
        return {
            "urgency": "⚠️ Consult a Doctor Soon",
            "color": "#f59e0b",
            "actions": [
                "Schedule an appointment with a general physician within the next 24–48 hours",
                "Re-check vitals after 30–60 minutes of rest to see if they return to normal range",
                "Avoid strenuous physical activity until checked by a doctor",
                "Note any other symptoms (dizziness, chest pain, breathlessness) and mention them to the doctor",
                "If symptoms worsen at any point, treat it as an emergency and seek immediate care",
            ],
        }
    else:
        return {
            "urgency": "✅ No Immediate Action Needed",
            "color": "#22c55e",
            "actions": [
                "Vitals are within a healthy range — continue regular monitoring",
                "Maintain a balanced diet, regular exercise, and adequate sleep",
                "Stay hydrated and manage stress levels",
                "Continue routine health check-ups as normally scheduled",
            ],
        }

def make_gauge(value, title, min_v, max_v, normal_range, unit=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": unit},
        gauge={
            "axis": {"range": [min_v, max_v]},
            "bar": {"color": "#60a5fa"},
            "steps": [
                {"range": [min_v, normal_range[0]], "color": "#3f1d1d"},
                {"range": normal_range, "color": "#123a24"},
                {"range": [normal_range[1], max_v], "color": "#3f1d1d"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=15, r=15, t=35, b=5),
                       paper_bgcolor="rgba(0,0,0,0)", font_color="#e5edff")
    return fig


def save_to_history(username, sys_bp, dia_bp, temp, spo2, status, confidence):
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "systolic_bp": sys_bp, "diastolic_bp": dia_bp,
        "temperature_f": temp, "spo2": spo2,
        "status": status, "confidence": round(confidence * 100, 1),
    }])
    if HISTORY_FILE.exists():
        row.to_csv(HISTORY_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(HISTORY_FILE, mode="w", header=True, index=False)


def _pdf_safe(text):
    """Strip/replace characters the core Helvetica PDF font can't render (emoji, em-dash, etc.)."""
    text = text.replace("—", "-").replace("–", "-").replace("’", "'").replace("‘", "'")
    return text.encode("latin-1", "ignore").decode("latin-1").strip()


def generate_pdf(username, sys_bp, dia_bp, temp, spo2, status, confidence, notes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Health Status Prediction Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 8, f"User: {username}", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Vitals Recorded", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for label, val in [("Systolic BP", f"{sys_bp} mmHg"), ("Diastolic BP", f"{dia_bp} mmHg"),
                        ("Temperature", f"{temp} F"), ("SpO2", f"{spo2} %")]:
        pdf.cell(0, 8, f"{label}: {val}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, f"Predicted Status: {status.upper()} ({confidence*100:.1f}% confidence)", ln=True)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Reasoning:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for n in notes:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, f"- {_pdf_safe(n)}")

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    rec = get_recommendations(status)
    pdf.cell(0, 8, f"Recommended Action: {_pdf_safe(rec['urgency'])}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    for a in rec["actions"]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 7, f"- {_pdf_safe(a)}")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 6, "Disclaimer: This is a demo/educational ML system and not a substitute for professional medical diagnosis.")

    return bytes(pdf.output())


# ---------- Header ----------
st.markdown(
    """
    <div class="top-banner">
        <div class="icon-badge">🩺</div>
        <div>
            <h1>Health Status Prediction System</h1>
            <p>AI/ML model using Blood Pressure, Temperature &amp; SpO2 — Random Forest Classifier</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_history = st.tabs(["🔍  Predict", "📜  History"])

# ================= PREDICT TAB =================
with tab_predict:
    st.sidebar.header("Enter Vitals")
    systolic_bp = st.sidebar.slider("Systolic BP (mmHg)", 60, 220, 118)
    diastolic_bp = st.sidebar.slider("Diastolic BP (mmHg)", 30, 140, 76)
    temperature_f = st.sidebar.slider("Temperature (°F)", 90.0, 108.0, 98.6, step=0.1)
    spo2 = st.sidebar.slider("SpO2 (%)", 50, 100, 97)
    predict_btn = st.sidebar.button("🔍 Predict Health Status", use_container_width=True, type="primary")

    if predict_btn:
        X = pd.DataFrame([[systolic_bp, diastolic_bp, temperature_f, spo2]], columns=FEATURES)
        X_scaled = scaler.transform(X)
        pred_idx = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = {cls: float(p) for cls, p in zip(label_encoder.classes_, proba)}
        notes = explain(systolic_bp, diastolic_bp, temperature_f, spo2)

        save_to_history(username, systolic_bp, diastolic_bp, temperature_f, spo2,
                         pred_label, confidence[pred_label])

        color = STATUS_COLORS[pred_label]
        st.markdown(
            f"""
            <div class="result-card" style="background-color:{color}22; border:2px solid {color};">
                <h2 style="color:{color}; margin:0;">{STATUS_LABELS[pred_label]}</h2>
                <p style="margin:4px 0 0 0; font-size:15px; color:#e5edff;">
                    Confidence: {confidence[pred_label]*100:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rec = get_recommendations(pred_label)
        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid {rec['color']};">
                <h3 style="color:{rec['color']}; margin-top:0;">{rec['urgency']}</h3>
                <ul style="margin-bottom:0;">
                    {"".join(f"<li style='margin-bottom:6px;'>{a}</li>" for a in rec['actions'])}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Prediction Confidence")
            conf_df = pd.DataFrame({
                "Status": [STATUS_LABELS[k] for k in confidence.keys()],
                "Confidence": [v * 100 for v in confidence.values()],
            })
            st.bar_chart(conf_df.set_index("Status"))

            st.subheader("🧠 Why this result?")
            for note in notes:
                st.write("•", note)
            st.markdown('</div>', unsafe_allow_html=True)

            pdf_bytes = generate_pdf(username, systolic_bp, diastolic_bp, temperature_f,
                                      spo2, pred_label, confidence[pred_label], notes)
            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with col_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Vitals Overview")
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(make_gauge(systolic_bp, "Systolic BP", 60, 220, (90, 120), " mmHg"), use_container_width=True)
                st.plotly_chart(make_gauge(temperature_f, "Temperature", 90, 108, (97, 99), "°F"), use_container_width=True)
            with g2:
                st.plotly_chart(make_gauge(diastolic_bp, "Diastolic BP", 30, 140, (60, 80), " mmHg"), use_container_width=True)
                st.plotly_chart(make_gauge(spo2, "SpO2", 50, 100, (95, 100), "%"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👈 Enter vitals in the sidebar and click **Predict Health Status** to see the result.")

# ================= HISTORY TAB =================
with tab_history:
    st.subheader("Your Prediction History")
    if HISTORY_FILE.exists():
        hist_df = pd.read_csv(HISTORY_FILE)
        user_hist = hist_df[hist_df["username"] == username].sort_values("timestamp", ascending=False)
        if len(user_hist) == 0:
            st.info("No predictions yet. Make one in the Predict tab.")
        else:
            st.dataframe(user_hist.drop(columns=["username"]), use_container_width=True, hide_index=True)

            st.subheader("Status Trend")
            trend_df = user_hist.sort_values("timestamp")[["timestamp", "status"]]
            status_map = {"normal": 0, "at_risk": 1, "critical": 2}
            trend_df["level"] = trend_df["status"].map(status_map)
            st.line_chart(trend_df.set_index("timestamp")["level"])

            csv_data = user_hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download History (CSV)", data=csv_data,
                                file_name=f"{username}_history.csv", mime="text/csv")
    else:
        st.info("No predictions yet. Make one in the Predict tab.")

st.markdown(
     '<div class="app-footer"></div>',
     unsafe_allow_html=True,
)
