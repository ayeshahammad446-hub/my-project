"""
rnn_model.py
============
A simple Elman RNN (Recurrent Neural Network) built from scratch
using only NumPy — no TensorFlow or PyTorch required.

Architecture
────────────
  Input  x(t) : shape (input_size,)
  Hidden h(t) : shape (hidden_size,)   ← carries "memory" across time steps
  Output y    : shape (output_size,)   ← Pass / Fail probability

Forward pass at each time step t
─────────────────────────────────
  h(t) = tanh( Wxh · x(t) + Whh · h(t-1) + bh )
  y    = sigmoid( Why · h(T) + by )          ← use final hidden state only

We treat each student's 5 features as a sequence of 5 time steps
(one feature per step), so the RNN reads the student profile
"one feature at a time" — simulating sequential decision making.
"""

import numpy as np


# ─────────────────────────────────────────────
# Activation functions
# ─────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


# ─────────────────────────────────────────────
# RNN Class
# ─────────────────────────────────────────────
class SimpleRNN:
    """
    Elman RNN trained with Backpropagation Through Time (BPTT).

    Parameters
    ----------
    input_size  : number of features per time step (1 here)
    hidden_size : number of hidden (recurrent) neurons
    output_size : 1 for binary classification
    """

    def __init__(self, input_size=1, hidden_size=32, output_size=1, seed=42):
        np.random.seed(seed)
        scale = 0.1

        # Weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size)  * scale  # input → hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale  # hidden → hidden
        self.Why = np.random.randn(output_size, hidden_size) * scale  # hidden → output

        # Biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        self.hidden_size = hidden_size
        self.input_size  = input_size

    # ── Forward pass ──────────────────────────────────────
    def forward(self, x_seq):
        """
        x_seq : numpy array shape (T, input_size)
                T = number of time steps (= number of features)

        Returns
        -------
        y_pred   : scalar probability of passing (0-1)
        cache    : stored values needed for BPTT
        """
        T = x_seq.shape[0]
        h = np.zeros((self.hidden_size, 1))   # initial hidden state

        hs     = {}          # hidden states at each time step
        xs     = {}          # inputs at each time step
        h_raws = {}          # pre-activation hidden values

        hs[-1] = h.copy()

        for t in range(T):
            xs[t]     = x_seq[t].reshape(-1, 1)
            h_raw     = self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh
            h_raws[t] = h_raw
            hs[t]     = np.tanh(h_raw)

        # Output layer uses the FINAL hidden state
        y_raw  = self.Why @ hs[T-1] + self.by
        y_pred = sigmoid(y_raw)

        cache = (xs, hs, h_raws, y_raw, T)
        return float(y_pred.squeeze()), cache

    # ── Backward pass (BPTT) ─────────────────────────────
    def backward(self, cache, y_true):
        """
        Computes gradients via Backpropagation Through Time.

        Returns dict of gradients.
        """
        xs, hs, h_raws, y_raw, T = cache

        # ── Output layer gradient ─────────────────────
        y_pred  = sigmoid(y_raw)
        dL_dy   = y_pred - y_true              # binary cross-entropy deriv

        dWhy = dL_dy * hs[T-1].T
        dby  = dL_dy.copy()

        # ── Back through time ─────────────────────────
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh  = np.zeros_like(self.bh)

        dh_next = self.Why.T * dL_dy           # gradient into hidden at T-1

        for t in reversed(range(T)):
            dh_raw  = dh_next * tanh_deriv(h_raws[t])
            dWxh   += dh_raw @ xs[t].T
            dWhh   += dh_raw @ hs[t-1].T
            dbh    += dh_raw
            dh_next = self.Whh.T @ dh_raw

        # Clip gradients to prevent explosion
        for grad in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(grad, -5, 5, out=grad)

        return {
            "dWxh": dWxh, "dWhh": dWhh, "dWhy": dWhy,
            "dbh":  dbh,  "dby":  dby,
        }

    # ── Parameter update (SGD with momentum) ─────────────
    def update(self, grads, lr=0.01, momentum=0.9):
        if not hasattr(self, "_velocity"):
            self._velocity = {k: np.zeros_like(v) for k, v in grads.items()}

        self._velocity["dWxh"] = momentum * self._velocity["dWxh"] - lr * grads["dWxh"]
        self._velocity["dWhh"] = momentum * self._velocity["dWhh"] - lr * grads["dWhh"]
        self._velocity["dWhy"] = momentum * self._velocity["dWhy"] - lr * grads["dWhy"]
        self._velocity["dbh"]  = momentum * self._velocity["dbh"]  - lr * grads["dbh"]
        self._velocity["dby"]  = momentum * self._velocity["dby"]  - lr * grads["dby"]

        self.Wxh += self._velocity["dWxh"]
        self.Whh += self._velocity["dWhh"]
        self.Why += self._velocity["dWhy"]
        self.bh  += self._velocity["dbh"]
        self.by  += self._velocity["dby"]

    # ── Predict single sample ─────────────────────────────
    def predict_proba(self, x_seq):
        prob, _ = self.forward(x_seq)
        return prob

    def predict(self, x_seq, threshold=0.5):
        return 1 if self.predict_proba(x_seq) >= threshold else 0
