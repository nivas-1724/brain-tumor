"""
Temperature Scaling & Calibration Engine
Fits temperature T parameter on validation logits to minimize NLL and computes Expected Calibration Error (ECE).
"""

import numpy as np
from scipy.optimize import minimize


def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and reliability bin data.
    probs: (N, C) probability outputs
    labels: (N,) true class indices
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": int(np.sum(in_bin)),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
            })
        else:
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": 0,
                "accuracy": 0.0,
                "confidence": (bin_lower + bin_upper) / 2.0,
            })

    return float(ece), bin_data


class TemperatureScaler:
    def __init__(self, temperature=1.0):
        self.temperature = float(temperature)

    def fit(self, logits, labels):
        """
        Fits optimal temperature T > 0 on validation logits using Negative Log-Likelihood minimization.
        logits: (N, C) unscaled raw output scores/logits
        labels: (N,) class indices
        """
        def nll_loss(t):
            temp = t[0]
            scaled_logits = logits / temp
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # Clip for numerical stability
            probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
            return nll

        res = minimize(nll_loss, [1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        return self.temperature

    def calibrate(self, probs):
        """
        Calibrates probability distribution probs (N, C) using temperature scaling.
        Converts probabilities back to pseudo-logits, divides by T, and re-applies softmax.
        """
        if abs(self.temperature - 1.0) < 1e-4:
            return probs

        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(probs_clipped)
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        calibrated_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return calibrated_probs
