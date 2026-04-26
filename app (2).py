"""
app.py  –  Streamlit User Interface (Task 9 - Option B)
=========================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Evaluator",
    page_icon="🎓",
    layout="centered",
)

# ── Robust path finder ────────────────────────────────────
def _find(filename):
    """Search next to app.py, then in cwd."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"'{filename}' not found.\nSearched:\n" +
        "\n".join(f"  {p}" for p in candidates) +
        "\n\nRun  python train_ann.py  first to generate the model files."
    )

# ── Load artefacts (cached for the Streamlit session) ─────
@st.cache_resource
def load_model():
    m = joblib.load(_find("model.joblib"))
    s = joblib.load(_find("scaler.joblib"))
    return m, s

try:
    model, scaler = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Evaluation function ───────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    features = np.array([[attendance, assignment, quiz, mid, study_hours]], dtype=float)
    features_scaled = scaler.transform(features)
    prediction  = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])
    if probability >= 0.75:
        performance = "High 🌟"
    elif probability >= 0.50:
        performance = "Medium 📈"
    else:
        performance = "Low 📉"
    return prediction, probability, performance

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("🎓 Student Performance Evaluator")
st.markdown(
    """
    Powered by an **Artificial Neural Network** trained on historical
    student data.  Enter a student's academic profile below and click
    **Evaluate** to get an instant Pass / Fail prediction.
    """
)
st.divider()

# ─────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────
st.subheader("📋 Student Profile")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider(
        "Attendance (%)", min_value=0, max_value=100, value=75,
        help="Percentage of classes the student attended"
    )
    quiz = st.slider(
        "Quiz Marks", min_value=0, max_value=100, value=65,
        help="Marks obtained in quizzes (out of 100)"
    )
    study_hours = st.slider(
        "Study Hours / Week", min_value=0, max_value=30, value=6,
        help="Self-reported weekly study hours"
    )

with col2:
    assignment = st.slider(
        "Assignment Marks", min_value=0, max_value=100, value=70,
        help="Marks obtained in assignments (out of 100)"
    )
    mid = st.slider(
        "Mid-term Marks", min_value=0, max_value=100, value=55,
        help="Mid-term exam marks (out of 100)"
    )

st.divider()

# ─────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────
if st.button("🔍 Evaluate Student", use_container_width=True, type="primary"):

    prediction, probability, performance = evaluate_student(
        attendance, assignment, quiz, mid, study_hours
    )

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success(f"### ✅ PASS  —  Pass Probability: {probability:.1%}")
    else:
        st.error(f"### ❌ FAIL  —  Pass Probability: {probability:.1%}")

    st.metric("Performance Band", performance)

    # Probability gauge bar
    st.markdown("**Pass Probability Gauge**")
    st.progress(int(probability * 100))

    # Input summary table
    st.markdown("**Input Summary**")
    summary = pd.DataFrame({
        "Feature":   ["Attendance", "Assignment", "Quiz", "Mid-term", "Study Hours/Week"],
        "Value":     [attendance, assignment, quiz, mid, study_hours],
    })
    st.table(summary.set_index("Feature"))

    # ── Radar chart of student profile ──────────────────
    st.markdown("**Student Profile Radar**")
    features_norm = [
        attendance / 100,
        assignment / 100,
        quiz / 100,
        mid / 100,
        min(study_hours / 20, 1.0),
    ]
    labels = ["Attendance", "Assignment", "Quiz", "Mid-term", "Study Hours"]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = features_norm + features_norm[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    color = "#22c55e" if prediction == 1 else "#ef4444"
    ax.fill(angles, values, alpha=0.25, color=color)
    ax.plot(angles, values, linewidth=2, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=6)
    ax.set_title("Normalised Profile", size=10, pad=12)
    st.pyplot(fig)
    plt.close(fig)

    # ── Interpretation message ───────────────────────────
    st.divider()
    st.subheader("🧠 Interpretation")
    if prediction == 1 and probability >= 0.75:
        st.info(
            "The student shows **strong** academic performance across all features. "
            "They are very likely to pass."
        )
    elif prediction == 1:
        st.info(
            "The student is likely to **pass**, but there is some uncertainty. "
            "Improving quiz or mid-term scores would increase confidence."
        )
    else:
        low_feat = []
        if attendance < 60:  low_feat.append("Attendance")
        if assignment < 55:  low_feat.append("Assignment")
        if quiz < 50:        low_feat.append("Quiz")
        if mid < 40:         low_feat.append("Mid-term")
        if study_hours < 4:  low_feat.append("Study Hours")
        areas = ", ".join(low_feat) if low_feat else "multiple areas"
        st.warning(
            f"The student is at risk of **failing**. "
            f"Key areas to improve: **{areas}**."
        )

# ─────────────────────────────────────────────────────────
# SIDEBAR – About
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** MLPClassifier (Scikit-learn)

        **Architecture:**
        - Input : 5 features
        - Hidden 1 : 64 neurons (ReLU)
        - Hidden 2 : 32 neurons (ReLU)
        - Output : 2 classes (Pass / Fail)

        **Features used:**
        - Attendance (%)
        - Assignment marks
        - Quiz marks
        - Mid-term marks
        - Weekly study hours

        **Performance Bands:**
        | Band   | Pass Prob |
        |--------|-----------|
        | High 🌟  | ≥ 75%   |
        | Medium 📈 | 50-74% |
        | Low 📉   | < 50%   |
        """
    )
    st.divider()
    st.caption("ANN Student Evaluator — Assignment Project")
