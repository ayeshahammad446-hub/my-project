"""
predict.py
==========
Reusable evaluation function + Command-Line Interface (Task 9 - Option A).

Usage examples
──────────────
  # As a module
  from predict import evaluate_student
  result = evaluate_student(85, 78, 72, 65, 8)
  print(result)

  # As a CLI
  python predict.py 85 78 72 65 8
  python predict.py          ← interactive mode
"""

import sys
import numpy as np
import joblib
import os

# ── Path resolution ───────────────────────────────────────
# Look for model files next to this script first,
# then fall back to the current working directory.
def _find(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, filename),   # same folder as predict.py
        os.path.join(os.getcwd(), filename),  # current working directory
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find '{filename}'.\n"
        f"Searched in:\n" + "\n".join(f"  {p}" for p in candidates) + "\n"
        f"Make sure you ran  python train_ann.py  first to generate "
        f"model.joblib and scaler.joblib."
    )

# ── Lazy singletons (loaded once on first call) ───────────
_model  = None
_scaler = None

def _load():
    global _model, _scaler
    if _model is None:
        _model  = joblib.load(_find("model.joblib"))
        _scaler = joblib.load(_find("scaler.joblib"))
    return _model, _scaler


# ─────────────────────────────────────────────
# TASK 7 – Core Evaluation Function
# ─────────────────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Maps student attributes  →  predicted performance.

    This function represents the learned ANN: f(x) → y.
    For any new student (not in the training dataset), it
    returns a Pass/Fail prediction along with the probability.

    Parameters
    ----------
    attendance  : float  – attendance percentage  (0-100)
    assignment  : float  – assignment marks        (0-100)
    quiz        : float  – quiz marks              (0-100)
    mid         : float  – mid-term marks          (0-100)
    study_hours : float  – weekly self-study hours

    Returns
    -------
    dict:
        prediction  (int)   0 = Fail,  1 = Pass
        label       (str)   human-readable result
        probability (float) probability of passing (0.0-1.0)
        performance (str)   Low / Medium / High  (bonus)
    """
    model, scaler = _load()

    features = np.array([[attendance, assignment, quiz, mid, study_hours]],
                        dtype=float)
    features_scaled = scaler.transform(features)

    prediction  = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0][1])

    label = "Pass ✅" if prediction == 1 else "Fail ❌"

    # Bonus – three-tier performance band
    if probability >= 0.75:
        performance = "High 🌟"
    elif probability >= 0.50:
        performance = "Medium 📈"
    else:
        performance = "Low 📉"

    return {
        "prediction":  prediction,
        "label":       label,
        "probability": probability,
        "performance": performance,
    }


# ─────────────────────────────────────────────
# CLI – Option A (beginner interface)
# ─────────────────────────────────────────────
def _cli_interactive():
    print("\n" + "=" * 50)
    print("   🎓 Student Performance Evaluator (CLI)   ")
    print("=" * 50)
    try:
        attendance  = float(input("  Attendance  (0-100) : "))
        assignment  = float(input("  Assignment  (0-100) : "))
        quiz        = float(input("  Quiz        (0-100) : "))
        mid         = float(input("  Mid-term    (0-100) : "))
        study_hours = float(input("  Study hours/week    : "))
    except ValueError:
        print("❌  Please enter numeric values only.")
        return

    result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

    print("\n" + "─" * 50)
    print(f"  Prediction  : {result['label']}")
    print(f"  Pass Prob   : {result['probability']:.1%}")
    print(f"  Performance : {result['performance']}")
    print("─" * 50 + "\n")


def _cli_args():
    _, att, asgn, qz, mid, sh = sys.argv
    result = evaluate_student(float(att), float(asgn), float(qz),
                              float(mid), float(sh))
    print(f"Result: {result['label']}  |  Pass Prob: {result['probability']:.1%}"
          f"  |  Performance: {result['performance']}")


if __name__ == "__main__":
    if len(sys.argv) == 6:
        _cli_args()
    else:
        _cli_interactive()
