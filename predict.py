"""
predict.py — Reusable Student Evaluation Function
==================================================
Task 7 : evaluate_student()   — core prediction function
Task 9A: Command-line interface
"""

import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# Load model artifacts
# ─────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(_BASE, "model.joblib"))
scaler = joblib.load(os.path.join(_BASE, "scaler.joblib"))


# ─────────────────────────────────────────────
# TASK 7 — Core Evaluation Function
# ─────────────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict whether a student will pass or fail.

    Parameters
    ----------
    attendance  : int/float  – attendance percentage (0–100)
    assignment  : int/float  – assignment score (0–100)
    quiz        : int/float  – quiz score (0–100)
    mid         : int/float  – mid-term score (0–100)
    study_hours : int/float  – weekly study hours

    Returns
    -------
    dict with keys:
        result       : int   – 0 (Fail) or 1 (Pass)
        label        : str   – "Pass" or "Fail"
        confidence   : float – model's confidence (0–1)
        performance  : str   – Low / Medium / High  [Bonus]
        interpretation: str  – human-readable message
    """
    import pandas as pd
    features = pd.DataFrame(
        [[attendance, assignment, quiz, mid, study_hours]],
        columns=["attendance", "assignment", "quiz", "mid", "study_hours"]
    )
    features_sc = scaler.transform(features)

    pred        = model.predict(features_sc)[0]
    proba       = model.predict_proba(features_sc)[0]
    confidence  = float(proba[pred])

    # ── Bonus: Low / Medium / High performance label
    # Based on weighted composite score
    composite = (
        0.25 * (attendance / 100) +
        0.20 * (assignment / 100) +
        0.20 * (quiz / 100) +
        0.25 * (mid / 100) +
        0.10 * min(study_hours / 15, 1.0)
    )
    if composite < 0.40:
        performance = "Low"
    elif composite < 0.65:
        performance = "Medium"
    else:
        performance = "High"

    # ── Human-readable interpretation
    if pred == 1:
        interp = (
            f"✅ This student is predicted to PASS with "
            f"{confidence * 100:.1f}% confidence. "
            f"Performance level: {performance}."
        )
    else:
        tips = []
        if attendance < 75:
            tips.append("improve attendance (currently low)")
        if study_hours < 5:
            tips.append("increase study hours")
        if quiz < 50:
            tips.append("focus on quiz preparation")
        tip_str = ("; ".join(tips) + ".") if tips else "review all subjects."
        interp = (
            f"❌ This student is predicted to FAIL with "
            f"{confidence * 100:.1f}% confidence. "
            f"Suggestion: {tip_str}"
        )

    return {
        "result":        int(pred),
        "label":         "Pass" if pred == 1 else "Fail",
        "confidence":    round(confidence, 4),
        "performance":   performance,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────
# TASK 9A — Command-Line Interface
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Student Performance Evaluator (CLI)")
    print("=" * 50)

    try:
        attendance  = float(input("\nAttendance  (0–100) : "))
        assignment  = float(input("Assignment  (0–100) : "))
        quiz        = float(input("Quiz Score  (0–100) : "))
        mid         = float(input("Mid-term    (0–100) : "))
        study_hours = float(input("Study Hours/week    : "))
    except ValueError:
        print("❌ Please enter valid numeric values.")
        exit(1)

    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    print("\n" + "─" * 50)
    print(f"  Prediction   : {result['label']}")
    print(f"  Confidence   : {result['confidence'] * 100:.1f}%")
    print(f"  Performance  : {result['performance']}")
    print("─" * 50)
    print(f"\n{result['interpretation']}\n")
