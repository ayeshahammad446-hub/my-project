"""
predict_rnn.py
==============
Reusable RNN evaluation function + CLI interface.

Usage
─────
  from predict_rnn import evaluate_student
  result = evaluate_student(85, 78, 72, 65, 8)

  python predict_rnn.py 85 78 72 65 8
  python predict_rnn.py          ← interactive
"""

import sys
import numpy as np
import joblib
import os

# ── Path finder ───────────────────────────────────────────
def _find(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(script_dir, filename),
                 os.path.join(os.getcwd(), filename)]:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"'{filename}' not found. Run  python train_rnn.py  first."
    )

# ── Lazy singletons ───────────────────────────────────────
_rnn    = None
_scaler = None

def _load():
    global _rnn, _scaler
    if _rnn is None:
        _rnn    = joblib.load(_find("rnn_model.joblib"))
        _scaler = joblib.load(_find("rnn_scaler.joblib"))
    return _rnn, _scaler


# ── Core function ─────────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Maps student profile → RNN Pass/Fail prediction.

    The 5 features are fed into the RNN one at a time:
      t=0: attendance  →  t=1: assignment  →  t=2: quiz
      t=3: mid-term    →  t=4: study_hours
    """
    rnn, scaler = _load()

    raw    = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled = scaler.transform(raw)
    x_seq  = scaled.reshape(5, 1)

    probability = rnn.predict_proba(x_seq)
    prediction  = 1 if probability >= 0.5 else 0
    label       = "Pass ✅" if prediction == 1 else "Fail ❌"
    performance = ("High 🌟"   if probability >= 0.75 else
                   "Medium 📈" if probability >= 0.50 else
                   "Low 📉")

    return {
        "prediction":  prediction,
        "label":       label,
        "probability": float(probability),
        "performance": performance,
    }


# ── CLI ───────────────────────────────────────────────────
def _interactive():
    print("\n" + "=" * 50)
    print("   🎓 RNN Student Performance Evaluator (CLI)")
    print("=" * 50)
    try:
        attendance  = float(input("  Attendance  (0-100) : "))
        assignment  = float(input("  Assignment  (0-100) : "))
        quiz        = float(input("  Quiz        (0-100) : "))
        mid         = float(input("  Mid-term    (0-100) : "))
        study_hours = float(input("  Study hours/week    : "))
    except ValueError:
        print("❌ Please enter numeric values.")
        return

    r = evaluate_student(attendance, assignment, quiz, mid, study_hours)
    print(f"\n  Prediction  : {r['label']}")
    print(f"  Pass Prob   : {r['probability']:.1%}")
    print(f"  Performance : {r['performance']}\n")


if __name__ == "__main__":
    if len(sys.argv) == 6:
        _, a, b, c, d, e = sys.argv
        r = evaluate_student(float(a), float(b), float(c), float(d), float(e))
        print(f"{r['label']}  |  {r['probability']:.1%}  |  {r['performance']}")
    else:
        _interactive()
