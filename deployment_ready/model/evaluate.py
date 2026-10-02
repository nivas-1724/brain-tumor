"""
Brain Tumor Detection — Model Evaluation Script

Evaluates the trained EfficientNetB0 classification model on the independent testing set (dataset/Testing).
Calculates and prints:
- Overall Accuracy, Loss, AUC
- Confusion Matrix
- Per-Class Accuracy, Precision, Recall, and F1-Score for Glioma, Meningioma, No Tumor, Pituitary.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}
IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "brain_tumor_model.h5")


def load_test_dataset(data_dir="dataset/Testing"):
    """Loads independent test set images and labels."""
    images = []
    labels = []
    
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        cls_idx = CLASS_MAP[cls]
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                    images.append(np.array(img, dtype=np.float32))
                    labels.append(cls_idx)
                except Exception:
                    pass

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def evaluate_model():
    """Main evaluation routine."""
    test_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "Testing")
    if not os.path.exists(test_dir):
        print(f"❌ Test dataset directory '{test_dir}' not found.")
        return

    print(f"\n⚡ Loading independent test dataset from '{test_dir}'...")
    X_test, y_test = load_test_dataset(test_dir)
    print(f"   Loaded {len(X_test)} testing images across {len(CLASSES)} classes.")

    print(f"\n⚡ Loading trained model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": tf.keras.applications.efficientnet.preprocess_input}
    )

    print("\n⚡ Running inference on test dataset...")
    y_pred_probs = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

    print("\n==================================================")
    print("  🧠  MODEL EVALUATION METRICS REPORT")
    print("==================================================")
    print(f"  Overall Accuracy : {acc * 100:.2f}%")
    print(f"  Total Test Scans : {len(y_test)}")
    
    print("\n--- Confusion Matrix ---")
    print(f"               Pred Glioma  Pred Meningioma  Pred NoTumor  Pred Pituitary")
    for i, row in enumerate(cm):
        print(f"True {CLASSES[i]:<10s} :    {row[0]:5d}          {row[1]:5d}        {row[2]:5d}        {row[3]:5d}")

    print("\n--- Per-Class Metrics ---")
    for i, cls_name in enumerate(CLASSES):
        cls_acc = cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) > 0 else 0.0
        print(f"  Class: {cls_name.capitalize():<12s}")
        print(f"    - Accuracy  : {cls_acc * 100:.2f}% ({cm[i, i]}/{sum(cm[i, :])})")
        print(f"    - Precision : {precision[i] * 100:.2f}%")
        print(f"    - Recall    : {recall[i] * 100:.2f}%")
        print(f"    - F1-Score  : {f1[i] * 100:.2f}%")

    print("\n--- Detailed Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=[c.capitalize() for c in CLASSES], digits=4)
    print(report)
    print("==================================================")

    return {
        "overall_accuracy": acc,
        "confusion_matrix": cm,
        "per_class": {
            CLASSES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASSES))
        }
    }


if __name__ == "__main__":
    evaluate_model()
