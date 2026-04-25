"""
train_ann.py — ANN Training Script for Student Performance Evaluator
=====================================================================
Task 1 : Understand the dataset
Task 3 : Preprocess
Task 4 : Build ANN
Task 5 : Train
Task 6 : Evaluate
Task 8 : Save model + scaler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# TASK 1 — Understand the Dataset
# ─────────────────────────────────────────────
print("=" * 55)
print("  TASK 1 — Dataset Overview")
print("=" * 55)

df = pd.read_excel("dataset.xlsx")

print(f"\n📐 Shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Column Names : {df.columns.tolist()}")
print("\n👀 First 5 Rows:")
print(df.head().to_string(index=False))
print("\n📊 Basic Statistics:")
print(df.describe().round(2).to_string())
print("\n🎯 Target Distribution (result):")
print(df["result"].value_counts().rename({0: "Fail (0)", 1: "Pass (1)"}))

print("""
📖 Column Explanations:
  attendance  – % of classes attended (0–100)
  assignment  – assignment score (0–100)
  quiz        – quiz score (0–100)
  mid         – mid-term exam score (0–100)
  study_hours – weekly study hours
  result      – 0 = Fail, 1 = Pass  ← TARGET

🔵 Problem Type: CLASSIFICATION
   Because 'result' is binary (0 or 1), not a
   continuous value, so we predict a category.
""")

# ─────────────────────────────────────────────
# TASK 3 — Data Preprocessing
# ─────────────────────────────────────────────
print("=" * 55)
print("  TASK 3 — Preprocessing")
print("=" * 55)

X = df[["attendance", "assignment", "quiz", "mid", "study_hours"]]
y = df["result"]

print(f"\n✅ Features (X) shape : {X.shape}")
print(f"✅ Target  (y) shape : {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n📂 Train size : {X_train.shape[0]}")
print(f"📂 Test  size : {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("""
🔍 Why Scaling?
   ANN neurons compute weighted sums. Features with
   large ranges (e.g. attendance 0–100 vs study_hours 0–20)
   dominate the gradient and slow/distort learning.
   StandardScaler converts every feature to mean=0, std=1
   so all features contribute equally.
""")

# ─────────────────────────────────────────────
# TASK 4 — Build ANN
# ─────────────────────────────────────────────
print("=" * 55)
print("  TASK 4 — ANN Architecture")
print("=" * 55)

print("""
🧠 ANN Concepts:
  Neurons         – processing units that apply a weighted
                    sum + activation to their inputs.
  Activation Fn   – introduces non-linearity (relu used here).
  Hidden Layers   – layers between input and output that let
                    the model learn complex patterns.

Architecture chosen:
  Input  layer  → 5 neurons (one per feature)
  Hidden layer1 → 64 neurons  (relu)
  Hidden layer2 → 32 neurons  (relu)
  Output layer  → 1 neuron   (sigmoid-equivalent via logistic)
""")

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False,
    n_iter_no_change=20,
)

# ─────────────────────────────────────────────
# TASK 5 — Train
# ─────────────────────────────────────────────
print("=" * 55)
print("  TASK 5 — Training")
print("=" * 55)

model.fit(X_train_sc, y_train)

print(f"✅ Training complete!")
print(f"   Iterations ran   : {model.n_iter_}")
print(f"   Best val score   : {model.best_validation_score_:.4f}")
print(f"   Loss at stop     : {model.loss_:.6f}")

# ─────────────────────────────────────────────
# TASK 6 — Evaluate
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 6 — Evaluation")
print("=" * 55)

y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Test Accuracy : {acc * 100:.2f}%")
print(f"""
📖 What does accuracy mean?
   Out of {len(y_test)} test students, the model
   correctly predicted {int(acc * len(y_test))} of them.
""")

print("📋 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Fail (0)", "Pass (1)"]))

cm = confusion_matrix(y_test, y_pred)
print("🔲 Confusion Matrix:")
print(f"              Predicted")
print(f"              Fail  Pass")
print(f"Actual Fail  [{cm[0,0]:4d}  {cm[0,1]:4d}]")
print(f"Actual Pass  [{cm[1,0]:4d}  {cm[1,1]:4d}]")
print(f"""
🔍 Mistakes analysis:
   False Positives (model says Pass, actually Fail): {cm[0,1]}
   False Negatives (model says Fail, actually Pass): {cm[1,0]}
""")

# ── Confusion matrix heatmap (Bonus)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"], ax=axes[0])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")

axes[1].plot(model.loss_curve_, label="Training Loss", color="#2563EB")
if hasattr(model, "validation_scores_"):
    val_acc = model.validation_scores_
    axes[1].plot(val_acc, label="Val Accuracy", color="#16A34A", linestyle="--")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss / Score")
axes[1].set_title("Training Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_report.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Saved: training_report.png")

# ─────────────────────────────────────────────
# TASK 8 — Save Model & Scaler
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 8 — Saving Model & Scaler")
print("=" * 55)

joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("""
✅ model.joblib  — saved
✅ scaler.joblib — saved

💡 Why save BOTH?
   The scaler transforms raw feature values into the
   normalized form the model was trained on. Without
   the exact same scaler, predictions on new data will
   be wrong even with the correct model weights.
""")

print("🎉 Training pipeline complete!\n")
