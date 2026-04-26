"""
rnn_app.py  –  Streamlit UI for RNN Student Evaluator
======================================================
Run with:  streamlit run rnn_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="RNN Student Evaluator",
    page_icon="🧠",
    layout="centered",
)

# ── Path finder ───────────────────────────────────────────
def _find(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(script_dir, filename),
                 os.path.join(os.getcwd(), filename)]:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"'{filename}' not found.\n"
        f"Run  python train_rnn.py  first to generate the model files."
    )

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    rnn    = joblib.load(_find("rnn_model.joblib"))
    scaler = joblib.load(_find("rnn_scaler.joblib"))
    return rnn, scaler

try:
    rnn, scaler = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Evaluation function ───────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    raw    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = scaler.transform(raw)
    x_seq  = scaled.reshape(5, 1)
    prob   = rnn.predict_proba(x_seq)
    pred   = 1 if prob >= 0.5 else 0
    perf   = ("High 🌟" if prob >= 0.75 else
              "Medium 📈" if prob >= 0.50 else "Low 📉")
    return pred, float(prob), perf

# ── RNN sequence visualiser ───────────────────────────────
def draw_rnn_sequence(features, feature_names, hidden_states):
    """Draw the RNN unrolled through 5 time steps."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")

    colors = ["#dbeafe", "#dcfce7", "#fef9c3", "#fce7f3", "#ede9fe"]
    FEATURE_LABELS = feature_names

    for t in range(5):
        x = t + 0.5

        # Input node
        ax.add_patch(plt.Circle((x, 0.3), 0.22, color=colors[t],
                                ec="#64748b", linewidth=1.5, zorder=3))
        ax.text(x, 0.3, f"{features[t]:.0f}", ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.text(x, -0.1, FEATURE_LABELS[t], ha="center", va="center",
                fontsize=7, color="#475569")

        # Hidden state node
        h_val = hidden_states[t]
        h_color = "#bbf7d0" if h_val > 0 else "#fecaca"
        ax.add_patch(plt.Circle((x, 1.5), 0.28, color=h_color,
                                ec="#334155", linewidth=1.5, zorder=3))
        ax.text(x, 1.5, f"h{t}\n{h_val:.2f}", ha="center", va="center",
                fontsize=7)

        # Vertical arrow input → hidden
        ax.annotate("", xy=(x, 1.2), xytext=(x, 0.52),
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))

        # Horizontal arrow hidden → hidden
        if t < 4:
            ax.annotate("", xy=(x + 0.78, 1.5), xytext=(x + 0.28, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5))

    # Output arrow from last hidden
    ax.annotate("", xy=(5.4, 1.5), xytext=(5.05, 1.5),
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2))
    ax.text(5.55, 1.5, "y", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#16a34a")

    ax.text(2.75, 2.3, "RNN Unrolled Through Time →", ha="center",
            fontsize=10, fontweight="bold", color="#1e293b")

    blue_patch  = mpatches.Patch(color="#93c5fd", label="Input features")
    green_patch = mpatches.Patch(color="#bbf7d0", label="Hidden state h>0")
    red_patch   = mpatches.Patch(color="#fecaca", label="Hidden state h<0")
    ax.legend(handles=[blue_patch, green_patch, red_patch],
              loc="upper left", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("🧠 RNN Student Performance Evaluator")
st.markdown("""
Powered by a **Recurrent Neural Network (RNN)** built from scratch.
The RNN reads student features **one at a time as a sequence**,
building up memory with each step before making a final prediction.
""")
st.divider()

# ─────────────────────────────────────────────────────────
# INPUT SLIDERS
# ─────────────────────────────────────────────────────────
st.subheader("📋 Student Profile")

col1, col2 = st.columns(2)
with col1:
    attendance  = st.slider("Attendance (%)",        0, 100, 75)
    quiz        = st.slider("Quiz Marks",            0, 100, 65)
    study_hours = st.slider("Study Hours / Week",    0,  30,  6)
with col2:
    assignment  = st.slider("Assignment Marks",      0, 100, 70)
    mid         = st.slider("Mid-term Marks",        0, 100, 55)

st.divider()

# ─────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────
if st.button("🔍 Evaluate with RNN", use_container_width=True, type="primary"):

    prediction, probability, performance = evaluate_student(
        attendance, assignment, quiz, mid, study_hours
    )

    # ── Result banner ─────────────────────────────────────
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.success(f"### ✅  PASS  —  Pass Probability: {probability:.1%}")
    else:
        st.error(f"### ❌  FAIL  —  Pass Probability: {probability:.1%}")

    col_a, col_b = st.columns(2)
    col_a.metric("Performance Band", performance)
    col_b.metric("Pass Probability", f"{probability:.1%}")

    st.progress(int(probability * 100))
    st.divider()

    # ── RNN sequence diagram ──────────────────────────────
    st.subheader("🔄 How the RNN Processed This Student")
    st.markdown(
        "Each feature is read **one step at a time**. "
        "The hidden state h(t) carries memory from previous steps."
    )

    # Get actual hidden states
    raw    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = scaler.transform(raw)
    x_seq  = scaled.reshape(5, 1)

    # Run forward pass and collect hidden states
    h = np.zeros((rnn.hidden_size, 1))
    hidden_scalars = []
    for t in range(5):
        h_raw = rnn.Wxh @ x_seq[t].reshape(-1,1) + rnn.Whh @ h + rnn.bh
        h     = np.tanh(h_raw)
        hidden_scalars.append(float(h.mean()))   # mean of hidden vector

    feat_vals   = [attendance, assignment, quiz, mid, study_hours]
    feat_names  = ["Attend.", "Assign.", "Quiz", "Mid", "Study\nHrs"]
    fig_rnn = draw_rnn_sequence(feat_vals, feat_names, hidden_scalars)
    st.pyplot(fig_rnn)
    plt.close(fig_rnn)

    st.caption(
        "Each h(t) value shown is the **mean** of the 32-dimensional "
        "hidden vector. Green = mostly positive activation, red = negative."
    )
    st.divider()

    # ── Radar chart ───────────────────────────────────────
    st.subheader("📡 Student Profile Radar")
    norm = [attendance/100, assignment/100, quiz/100,
            mid/100, min(study_hours/20, 1.0)]
    labels = ["Attendance", "Assignment", "Quiz", "Mid-term", "Study Hours"]
    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    vals   = norm + [norm[0]]

    fig2, ax2 = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    color = "#22c55e" if prediction == 1 else "#ef4444"
    ax2.fill(angles, vals, alpha=0.25, color=color)
    ax2.plot(angles, vals, color=color, linewidth=2)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(labels, size=8)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["25%","50%","75%","100%"], size=6)
    ax2.set_title("Normalised Profile", size=10, pad=12)
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Input table ───────────────────────────────────────
    st.subheader("📋 Input Summary")
    st.table(pd.DataFrame({
        "Feature": ["Attendance","Assignment","Quiz","Mid-term","Study Hours/Week"],
        "Value":   [attendance, assignment, quiz, mid, study_hours],
    }).set_index("Feature"))

    # ── Interpretation ────────────────────────────────────
    st.divider()
    st.subheader("🧠 Interpretation")
    if prediction == 1 and probability >= 0.75:
        st.info("Strong performance across the board — very likely to **pass**.")
    elif prediction == 1:
        st.info("Likely to **pass**, but some features could be stronger. "
                "Focus on quiz and mid-term scores.")
    else:
        low = []
        if attendance  < 60: low.append("Attendance")
        if assignment  < 55: low.append("Assignment")
        if quiz        < 50: low.append("Quiz")
        if mid         < 40: low.append("Mid-term")
        if study_hours <  4: low.append("Study Hours")
        areas = ", ".join(low) if low else "overall performance"
        st.warning(f"At risk of **failing**. Improve: **{areas}**.")

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About This Model")
    st.markdown("""
    **Model Type:** Elman RNN (from scratch)

    **Architecture:**
    - Input size : 1 (per time step)
    - Hidden size : 32 neurons
    - Time steps : 5 (one per feature)
    - Output : sigmoid → P(Pass)

    **How it works:**
    ```
    h(t) = tanh(Wxh·x(t) + Whh·h(t-1) + bh)
    y    = sigmoid(Why·h(5) + by)
    ```

    **Sequence order:**
    1. Attendance
    2. Assignment
    3. Quiz
    4. Mid-term
    5. Study Hours

    **Performance Bands:**
    | Band | Pass Prob |
    |------|-----------|
    | High 🌟 | ≥ 75% |
    | Medium 📈 | 50–74% |
    | Low 📉 | < 50% |
    """)
    st.divider()
    st.caption("RNN Student Evaluator — Built with NumPy")
