"""
app.py — Streamlit UI for Student Performance Evaluator
========================================================
Task 9B: Streamlit-based user interface
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from predict import evaluate_student

# ── Page config
st.set_page_config(
    page_title="Student ANN Evaluator",
    page_icon="🎓",
    layout="centered",
)

# ── Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0;
    }
    .subtitle {
        color: #555;
        margin-top: 0;
        font-size: 1rem;
    }
    .result-pass {
        background: #d1fae5;
        border-left: 6px solid #10b981;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-fail {
        background: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header
st.markdown('<p class="main-title">🎓 Student Performance Evaluator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by an Artificial Neural Network (ANN)</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar — About
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "This tool uses a trained ANN (MLPClassifier) to predict "
        "whether a student will **Pass** or **Fail** based on their "
        "academic features."
    )
    st.markdown("**Input Features:**")
    st.markdown(
        "- 📅 Attendance (%)\n"
        "- 📝 Assignment Score\n"
        "- 📋 Quiz Score\n"
        "- 📖 Mid-term Score\n"
        "- ⏱️ Weekly Study Hours"
    )
    st.markdown("---")
    st.markdown("**Model:** `MLPClassifier`  \n**Layers:** 5 → 64 → 32 → 1")

# ── Input Panel
st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider(
        "Attendance (%)", min_value=0, max_value=100, value=75,
        help="Percentage of classes attended"
    )
    assignment = st.slider(
        "Assignment Score", min_value=0, max_value=100, value=60,
        help="Total assignment marks obtained"
    )
    quiz = st.slider(
        "Quiz Score", min_value=0, max_value=100, value=55,
        help="Quiz marks out of 100"
    )

with col2:
    mid = st.slider(
        "Mid-term Score", min_value=0, max_value=100, value=50,
        help="Mid-term examination score"
    )
    study_hours = st.slider(
        "Weekly Study Hours", min_value=0, max_value=30, value=5,
        help="Average hours studied per week"
    )

st.divider()

# ── Predict Button
if st.button("🔍 Predict Performance", use_container_width=True, type="primary"):

    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    # ── Result Display
    label      = result["label"]
    confidence = result["confidence"] * 100
    perf       = result["performance"]
    interp     = result["interpretation"]

    if label == "Pass":
        st.markdown(f'<div class="result-pass">'
                    f'<h3 style="margin:0;color:#065f46">✅ Predicted: PASS</h3>'
                    f'<p style="margin:0.3rem 0 0">{interp}</p>'
                    f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-fail">'
                    f'<h3 style="margin:0;color:#991b1b">❌ Predicted: FAIL</h3>'
                    f'<p style="margin:0.3rem 0 0">{interp}</p>'
                    f'</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction",  label)
    m2.metric("Confidence",  f"{confidence:.1f}%")
    m3.metric("Performance", perf)

    # ── Radar / Feature Chart
    st.subheader("📊 Feature Snapshot")

    categories  = ["Attendance", "Assignment", "Quiz", "Mid-term", "Study\nHours"]
    values_raw  = [attendance, assignment, quiz, mid, study_hours]
    # Normalize study_hours to 0-100 for display
    values_norm = [
        attendance,
        assignment,
        quiz,
        mid,
        min(study_hours / 30 * 100, 100)
    ]

    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
    bars = ax.barh(categories, values_norm, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score / Normalized Value (%)")
    ax.axvline(50, color="#9ca3af", linestyle="--", linewidth=0.8, label="50% mark")
    for bar, val in zip(bars, values_raw):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, color="#374151")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Confidence gauge
    st.subheader("🎯 Model Confidence")
    prog_color = "#10b981" if label == "Pass" else "#ef4444"
    st.progress(int(confidence), text=f"{confidence:.1f}% confident → **{label}**")

# ── Footer
st.divider()
st.caption("🧠 ANN model trained on 600 synthetic student records · scikit-learn MLPClassifier")
