"""
Comprehensive Model Evaluation Metrics Module
Computes overall & per-class precision, recall, F1, accuracy, confusion matrix, and misclassification statistics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def evaluate_predictions(y_true, y_pred_probs, class_names=CLASSES):
    """
    Computes full metrics dictionary from true one-hot labels and predicted probabilities.
    y_true: (N, 4) one-hot or (N,) class indices
    y_pred_probs: (N, 4) predicted softmax probabilities
    """
    if y_true.ndim > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true_indices, y_pred_indices))
    prec_macro = float(precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(len(class_names))))

    # Per-class metrics
    prec_per_class = precision_score(y_true_indices, y_pred_indices, average=None, zero_division=0)
    rec_per_class = recall_score(y_true_indices, y_pred_indices, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_indices, y_pred_indices, average=None, zero_division=0)

    per_class_summary = {}
    for idx, cname in enumerate(class_names):
        per_class_summary[cname] = {
            "precision": round(float(prec_per_class[idx]) * 100.0, 2),
            "recall": round(float(rec_per_class[idx]) * 100.0, 2),
            "f1_score": round(float(f1_per_class[idx]) * 100.0, 2),
        }

    # Correct vs Incorrect confidence analysis
    correct_mask = (y_true_indices == y_pred_indices)
    incorrect_mask = ~correct_mask

    avg_conf_correct = float(np.mean(confidences[correct_mask])) if np.sum(correct_mask) > 0 else 0.0
    avg_conf_incorrect = float(np.mean(confidences[incorrect_mask])) if np.sum(incorrect_mask) > 0 else 0.0

    # Specific Glioma <-> Meningioma confusion counts
    # Glioma = index 0, Meningioma = index 1
    glioma_as_meningioma = int(np.sum((y_true_indices == 0) & (y_pred_indices == 1)))
    meningioma_as_glioma = int(np.sum((y_true_indices == 1) & (y_pred_indices == 0)))

    return {
        "accuracy": round(acc * 100.0, 2),
        "precision_macro": round(prec_macro * 100.0, 2),
        "recall_macro": round(rec_macro * 100.0, 2),
        "f1_macro": round(f1_macro * 100.0, 2),
        "precision_weighted": round(prec_weighted * 100.0, 2),
        "recall_weighted": round(rec_weighted * 100.0, 2),
        "f1_weighted": round(f1_weighted * 100.0, 2),
        "per_class": per_class_summary,
        "confusion_matrix": cm.tolist(),
        "total_test_samples": int(len(y_true_indices)),
        "correct_predictions": int(np.sum(correct_mask)),
        "incorrect_predictions": int(np.sum(incorrect_mask)),
        "avg_confidence_correct": round(avg_conf_correct * 100.0, 2),
        "avg_confidence_incorrect": round(avg_conf_incorrect * 100.0, 2),
        "glioma_as_meningioma_errors": glioma_as_meningioma,
        "meningioma_as_glioma_errors": meningioma_as_glioma,
    }
