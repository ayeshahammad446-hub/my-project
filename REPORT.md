# 📘 ANN Student Performance Evaluator — Final Report

**Student Assignment: Artificial Neural Networks**  
**Dataset:** 600 synthetic student records | **Accuracy: 82.50%**

---

## 🧠 Task 11 — Final Explanation

### 1. What is an ANN in your own words?

An **Artificial Neural Network (ANN)** is a computational system loosely inspired by how the human brain works. It consists of interconnected layers of **neurons** — simple mathematical units that each receive inputs, multiply them by learned *weights*, add a *bias*, apply a non-linear *activation function*, and pass the result forward to the next layer.

Through a process called **backpropagation**, the network repeatedly adjusts these weights by comparing its predictions against correct answers and minimizing the error. Over many iterations (called *epochs*), the network gradually "learns" to distinguish patterns in data — in our case, the difference between students likely to pass versus fail.

---

### 2. What function did your model learn?

The model learned to approximate the function:

```
f(attendance, assignment, quiz, mid, study_hours) → {0: Fail, 1: Pass}
```

Specifically, the ANN internally learned a complex combination of thresholds and weighted relationships among features. For example:
- Students with **high attendance (>75%)** AND **mid-term score (>50)** strongly lean toward Pass.
- Students with **very low study hours (<3)** combined with **low quiz scores (<40)** tend toward Fail.

The model doesn't follow explicit if-else rules. Instead, it approximates these boundaries through 64+32 hidden neurons across two layers, making it much more flexible than a simple linear classifier.

---

### 3. How does your system evaluate a new student?

When a new student's data is entered:

1. **Input** → The 5 features (attendance, assignment, quiz, mid, study_hours) are collected.
2. **Scaling** → The scaler transforms raw values to standardized form (mean=0, std=1) using the same statistics computed on training data.
3. **Forward Pass** → Scaled values flow through the ANN: Input Layer (5) → Hidden Layer 1 (64 ReLU) → Hidden Layer 2 (32 ReLU) → Output Layer (1 sigmoid).
4. **Prediction** → The output neuron produces a probability. If ≥ 0.5 → **Pass (1)**, otherwise → **Fail (0)**.
5. **Interpretation** → The system also computes a composite performance score (Low/Medium/High) and generates a human-readable suggestion.

---

### 4. Why is scaling important?

ANN neurons compute **weighted sums**: `z = w₁x₁ + w₂x₂ + ... + b`

Without scaling:
- `attendance` ranges 0–100
- `study_hours` ranges 0–15

The weight updates during training (via gradient descent) will be **dominated by features with larger numerical ranges**, causing slower convergence, unstable training, and biased predictions.

`StandardScaler` converts every feature to **mean = 0, standard deviation = 1**, ensuring all features contribute equally to the learned function. This is essential for gradient-based optimizers like Adam used here.

---

### 5. Limitations of the model

| Limitation | Explanation |
|---|---|
| **Synthetic data** | Trained on generated data, not real student records. Real-world accuracy may differ. |
| **Binary output only** | Only predicts Pass/Fail, not a grade or percentage. |
| **No temporal awareness** | Doesn't track improvement over time — each prediction is independent. |
| **Feature scope** | Ignores many real factors: socioeconomic status, mental health, subject difficulty. |
| **Small dataset** | 600 records is relatively small; larger datasets would improve generalization. |
| **Fixed threshold** | Uses 0.5 decision threshold — this could be tuned for specific use cases (e.g., reduce false negatives). |
| **Black-box** | ANN decisions are not easily explainable to students or teachers without explainability tools (e.g., SHAP). |

---

## 📊 Model Performance Summary

| Metric | Value |
|---|---|
| Test Accuracy | **82.50%** |
| Precision (Fail) | 79% |
| Precision (Pass) | 86% |
| Recall (Fail) | 86% |
| Recall (Pass) | 80% |
| F1 Score (avg) | 0.82 |
| Training Iterations | 35 |
| Architecture | 5 → 64 → 32 → 1 |

---

## 🗂️ Project Structure

```
student_ann_project/
│
├── dataset.xlsx          ← Original student data (600 records)
├── train_ann.py          ← Tasks 1, 3, 4, 5, 6, 8: Full training pipeline
├── predict.py            ← Task 7, 9A: evaluate_student() + CLI
├── app.py                ← Task 9B: Streamlit web UI
├── model.joblib          ← Trained ANN weights
├── scaler.joblib         ← Fitted StandardScaler
├── training_report.png   ← Confusion matrix + loss curve
├── requirements.txt      ← All dependencies
└── REPORT.md             ← This file
```

---

## 🚀 How to Run

```bash
# 1. Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_ann.py

# 4. Test via command line
python predict.py

# 5. Launch web UI
streamlit run app.py
```

---

## 🏆 Bonus Completed

- ✅ **Low / Medium / High** performance classification (composite score)
- ✅ **Confusion matrix heatmap** (seaborn, saved as PNG)
- ✅ **Training loss curve** plotted and saved
- ✅ **Actionable suggestions** for failing students

---

*Report prepared for ANN Assignment | Model: scikit-learn MLPClassifier | Python 3.x*
