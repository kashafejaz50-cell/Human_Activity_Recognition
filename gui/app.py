"""
╔══════════════════════════════════════════════════════════════════╗
║       Human Activity Recognition System — Streamlit Dashboard    ║
╚══════════════════════════════════════════════════════════════════╝
Run with:
    streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Project preprocess import ─────────────────────────────────────────────────
try:
    from preprocess import load_data, preprocess
    CUSTOM_PREPROCESS = True
except ImportError:
    CUSTOM_PREPROCESS = False

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
    --bg:          #0D1117;
    --surface:     #161B22;
    --surface2:    #1C2330;
    --border:      #30363D;
    --accent:      #58A6FF;
    --accent-glow: rgba(88,166,255,.18);
    --green:       #3FB950;
    --green-bg:    rgba(63,185,80,.12);
    --red:         #F85149;
    --red-bg:      rgba(248,81,73,.12);
    --amber:       #D29922;
    --amber-bg:    rgba(210,153,34,.12);
    --text:        #E6EDF3;
    --muted:       #8B949E;
    --radius:      10px;
    --radius-lg:   16px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Main canvas ── */
.main .block-container {
    background: var(--bg) !important;
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 1300px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero ── */
.hero {
    position: relative;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2.8rem 3rem 2.4rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 70% 60% at 50% -10%,
        rgba(88,166,255,.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: .45rem;
    background: var(--accent-glow);
    border: 1px solid rgba(88,166,255,.3);
    border-radius: 999px;
    padding: .28rem .9rem;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -1px;
    line-height: 1.15;
    margin: 0 0 .6rem;
}
.hero h1 span { color: var(--accent); }
.hero p {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 400;
    margin: 0;
}

/* ── Section heading ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -.2px;
    margin: 0 0 1.2rem;
    padding-bottom: .7rem;
    border-bottom: 1px solid var(--border);
}
.sec-head .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    flex-shrink: 0;
}

/* ── Card wrapper ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
}

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.4rem; }
.metric-tile {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.metric-tile:hover { border-color: var(--accent); }
.metric-tile::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.metric-tile .mt-label {
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .5rem;
}
.metric-tile .mt-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    letter-spacing: -1px;
}
.metric-tile .mt-sub {
    font-size: .78rem;
    color: var(--muted);
    margin-top: .3rem;
}

/* ── Prediction result boxes ── */
.pred-box {
    border-radius: var(--radius);
    padding: 1rem 1.3rem;
    margin: .5rem 0;
    border-left: 3px solid;
    font-size: .93rem;
}
.pred-box strong { display: block; font-size: .68rem; letter-spacing: .09em; text-transform: uppercase; margin-bottom: .35rem; font-weight: 700; }
.pred-box .big  { font-size: 1.3rem; font-weight: 700; }
.box-blue   { background: rgba(88,166,255,.1);  border-color: var(--accent);  color: #A5C8FF; }
.box-green  { background: var(--green-bg);       border-color: var(--green);   color: #7EE8A2; }
.box-amber  { background: var(--amber-bg);       border-color: var(--amber);   color: #F0C674; }
.verdict    {
    text-align: center;
    font-size: 1rem;
    font-weight: 700;
    border-radius: var(--radius);
    padding: .9rem;
    margin-top: .6rem;
    letter-spacing: .02em;
}
.verdict-ok  { background: var(--green-bg); color: var(--green); border: 1px solid rgba(63,185,80,.3); }
.verdict-err { background: var(--red-bg);   color: var(--red);   border: 1px solid rgba(248,81,73,.3); }

/* ── Sidebar pills ── */
.sb-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .6rem .9rem;
    margin-bottom: .5rem;
}
.sb-item .sbi-label {
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .2rem;
}
.sb-item .sbi-value {
    font-size: .9rem;
    font-weight: 600;
    color: var(--text);
}

/* ── Report / code block ── */
.report-pre {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.3rem 1.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    line-height: 1.75;
    color: #8B949E;
    overflow-x: auto;
    white-space: pre;
}

/* ── Image frame ── */
.img-frame {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--surface2);
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > div { padding-top: .4rem; }
.stButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #0D1117 !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: .95rem !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: .75rem 1.5rem !important;
    letter-spacing: .02em;
    transition: opacity .15s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: .95rem !important;
    color: var(--text) !important;
}

/* ── Number input ── */
input[type=number] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

ACTIVITY_EMOJI = {
    "WALKING":            "🚶",
    "WALKING_UPSTAIRS":   "🪜",
    "WALKING_DOWNSTAIRS": "⬇️",
    "SITTING":            "🪑",
    "STANDING":           "🧍",
    "LAYING":             "🛌",
}

def get_emoji(label: str) -> str:
    return ACTIVITY_EMOJI.get(str(label).upper().replace(" ", "_"), "🏃")

def uses_scaled_data(model) -> bool:
    return isinstance(model, (SVC, LogisticRegression))

def model_display_name(model) -> str:
    return type(model).__name__

def sec(icon: str, title: str):
    """Render a styled section heading."""
    st.markdown(f"""
    <div class="sec-head">
        <span class="dot"></span>
        <span>{icon}&nbsp; {title}</span>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    model  = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_data(show_spinner="Loading dataset …")
def load_test_data():
    train_df, test_df = load_data()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess(train_df, test_df)
    return X_test.values, y_test.values

@st.cache_data(show_spinner="Loading dataset …")
def load_test_data_fallback():
    df        = pd.read_csv("data/test.csv")
    drop_cols = [c for c in df.columns if c.lower() in ("subject", "id")]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    label_col = df.columns[-1]
    return df.drop(columns=[label_col]).values, df[label_col].values


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def predict_sample(model, scaler, X_test, index):
    sample       = X_test[index].reshape(1, -1)
    sample_input = scaler.transform(sample) if uses_scaled_data(model) else sample
    prediction   = model.predict(sample_input)[0]
    confidence   = None
    if hasattr(model, "predict_proba"):
        confidence = round(float(model.predict_proba(sample_input)[0].max()) * 100, 2)
    return str(prediction), confidence


# ═════════════════════════════════════════════════════════════════════════════
# LOAD EVERYTHING
# ═════════════════════════════════════════════════════════════════════════════

try:
    model, scaler = load_artifacts()
    if CUSTOM_PREPROCESS:
        X_test, y_test = load_test_data()
    else:
        X_test, y_test = load_test_data_fallback()
    N          = len(X_test)
    load_ok    = True
except Exception as exc:
    load_ok    = False
    load_error = str(exc)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🧠 Project Info")
    st.markdown("<hr style='border-color:#30363D;margin:.4rem 0 1rem'>",
                unsafe_allow_html=True)

    items = (
        [
            ("Best Model",    model_display_name(model)),
            ("Accuracy",      "96.17 %"),
            ("Activities",    str(len(np.unique(y_test)))),
            ("Dataset",       "UCI HAR Dataset"),
            ("Test Samples",  str(N)),
        ]
        if load_ok else
        [
            ("Best Model",    "Best Model"),
            ("Accuracy",      "96.17 %"),
            ("Activities",    "6"),
            ("Dataset",       "UCI HAR Dataset"),
            ("Test Samples",  "—"),
        ]
    )

    for label, value in items:
        st.markdown(f"""
        <div class="sb-item">
            <div class="sbi-label">{label}</div>
            <div class="sbi-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.8rem;color:#8B949E;line-height:1.65;padding:.2rem .1rem'>
    Classifies smartphone sensor signals into 6 daily activities using a
    fine-tuned ML model trained on the UCI HAR Dataset.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem;color:#30363D;text-align:center'>
        Streamlit · scikit-learn · UCI HAR
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HERO
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">⚡ Machine Learning Dashboard</div>
    <h1>Human <span>Activity</span> Recognition</h1>
    <p>Real-time prediction of human activities from smartphone sensor data</p>
</div>
""", unsafe_allow_html=True)

# ── Error banner ──────────────────────────────────────────────────────────────
if not load_ok:
    st.error(f"⚠️  Failed to load model / data: **{load_error}**")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# METRIC TILES
# ═════════════════════════════════════════════════════════════════════════════

c1, c2, c3, c4 = st.columns(4)

tiles = [
    ("🏆 Model",         model_display_name(model),          "Best classifier"),
    ("🎯 Accuracy",      "96.17 %",                          "On held-out test set"),
    ("🏷️ Activities",   str(len(np.unique(y_test))),         "Unique class labels"),
    ("📋 Test Samples", str(N),                              "Available for prediction"),
]

for col, (label, value, sub) in zip([c1, c2, c3, c4], tiles):
    with col:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="mt-label">{label}</div>
            <div class="mt-value">{value}</div>
            <div class="mt-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION SECTION
# ═════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="card">', unsafe_allow_html=True)
sec("🔮", "Activity Prediction")

left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown("<p style='color:#8B949E;font-size:.87rem;margin-bottom:.6rem'>Choose a test-set sample to classify:</p>",
                unsafe_allow_html=True)

    row_index = st.slider("Row index", 0, N - 1, 0, label_visibility="collapsed")
    manual    = st.number_input("Or enter index", 0, N - 1, row_index, step=1)

    if manual != row_index:
        row_index = int(manual)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀  Predict Activity")

with right:
    if predict_btn:
        with st.spinner("Running inference …"):
            pred, conf  = predict_sample(model, scaler, X_test, row_index)
            actual      = str(y_test[row_index])
            correct     = pred.upper() == actual.upper()

        pe = get_emoji(pred)
        ae = get_emoji(actual)

        st.markdown(f"""
        <div class="pred-box box-blue">
            <strong>🔍 Predicted Activity</strong>
            <div class="big">{pe} {pred}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pred-box box-green">
            <strong>✔️ Actual Activity</strong>
            <div class="big">{ae} {actual}</div>
        </div>""", unsafe_allow_html=True)

        if conf is not None:
            st.markdown(f"""
            <div class="pred-box box-amber">
                <strong>📊 Model Confidence</strong>
                <div class="big">{conf} %</div>
            </div>""", unsafe_allow_html=True)

        verdict = ("✅ &nbsp; Prediction Correct" if correct
                   else "❌ &nbsp; Prediction Incorrect")
        cls     = "verdict-ok" if correct else "verdict-err"
        st.markdown(f'<div class="verdict {cls}">{verdict}</div>',
                    unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='height:100%;display:flex;align-items:center;
                    justify-content:center;color:#8B949E;font-size:.9rem;
                    padding:2rem;text-align:center;
                    border:1px dashed #30363D;border-radius:10px'>
            👈 Select a row and click <strong>&nbsp;Predict Activity</strong>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("📊  Model Visualisations", expanded=True):
    v1, v2 = st.columns(2, gap="large")

    with v1:
        sec("🗺️", "Confusion Matrix")
        cm_path = "reports/best_model_confusion_matrix.png"
        if os.path.exists(cm_path):
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(Image.open(cm_path), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("confusion_matrix.png not found in reports/")

    with v2:
        sec("📈", "Model Accuracy Comparison")
        acc_path = "reports/tuned_accuracy_comparison.png"
        if os.path.exists(acc_path):
            st.markdown('<div class="img-frame">', unsafe_allow_html=True)
            st.image(Image.open(acc_path), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("tuned_accuracy_comparison.png not found in reports/")


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION REPORT
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("📄  Classification Report", expanded=False):
    sec("📄", "Best Model Classification Report")
    report_path = "reports/best_model_classification_report.txt"
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report_text = f.read()
        st.markdown(f'<div class="report-pre">{report_text}</div>',
                    unsafe_allow_html=True)
    else:
        st.warning("Classification report not found.")


# ═════════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("🗄️  Dataset Preview", expanded=False):
    sec("🗄️", "Test Set — First 10 Rows")
    preview_df = pd.DataFrame(X_test)
    st.dataframe(preview_df.head(10), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='text-align:center;color:#30363D;font-size:.78rem;
            margin-top:3rem;padding-top:1rem;border-top:1px solid #30363D'>
    🏃 Human Activity Recognition System &nbsp;·&nbsp;
    Streamlit · scikit-learn · UCI HAR Dataset
</div>
""", unsafe_allow_html=True)