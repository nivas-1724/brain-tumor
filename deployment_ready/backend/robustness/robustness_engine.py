"""
Robustness Testing & Perturbation Stress Testing Engine
Applies 7 controlled image transformations and evaluates model prediction stability, confidence change, and accuracy drop.
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


def apply_transformation(img_np, transform_type):
    """
    Applies controlled perturbation to a single image array (224, 224, 3) [0..255].
    """
    img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)

    if transform_type == "Original":
        return img_np.copy()

    elif transform_type == "Brightness":
        # +20% brightness
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.20, 0, 255)
        res = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return res.astype(np.float32)

    elif transform_type == "Contrast":
        # x1.3 contrast scale
        mean = np.mean(img_uint8, axis=(0, 1), keepdims=True)
        res = np.clip((img_uint8.astype(np.float32) - mean) * 1.3 + mean, 0, 255)
        return res

    elif transform_type == "Noise":
        # Gaussian noise (sigma = 15)
        noise = np.random.normal(0, 15, img_uint8.shape)
        res = np.clip(img_uint8.astype(np.float32) + noise, 0, 255)
        return res

    elif transform_type == "Rotation":
        # 15 degree rotation
        (h, w) = img_uint8.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 15, 1.0)
        res = cv2.warpAffine(img_uint8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return res.astype(np.float32)

    elif transform_type == "Compression":
        # JPEG compression quality = 30
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode('.jpg', bgr, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        res = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return res.astype(np.float32)

    elif transform_type == "Blur":
        # Gaussian Blur sigma = 1.0
        res = cv2.GaussianBlur(img_uint8, (5, 5), 1.0)
        return res.astype(np.float32)

    else:
        return img_np.copy()


def evaluate_model_robustness(model, X_test, y_test, num_samples=300):
    """
    Evaluates model across 7 transformation types on a test subset.
    Returns dictionary of results and list of rows for robustness_results.csv.
    """
    if len(X_test) > num_samples:
        indices = np.random.choice(len(X_test), size=num_samples, replace=False)
        X_sub = X_test[indices]
        y_sub = y_test[indices]
    else:
        X_sub = X_test
        y_sub = y_test

    y_true = np.argmax(y_sub, axis=1)

    # 1. Baseline Original
    orig_preds = model.predict(X_sub, verbose=0)
    orig_classes = np.argmax(orig_preds, axis=1)
    orig_confs = np.max(orig_preds, axis=1)
    orig_acc = accuracy_score(y_true, orig_classes)
    orig_f1 = f1_score(y_true, orig_classes, average="weighted")
    orig_avg_conf = np.mean(orig_confs)

    transformations = [
        "Original",
        "Brightness",
        "Contrast",
        "Noise",
        "Rotation",
        "Compression",
        "Blur",
    ]

    results = []

    for t_type in transformations:
        if t_type == "Original":
            t_X = X_sub
        else:
            t_X = np.array([apply_transformation(img, t_type) for img in X_sub], dtype=np.float32)

        preds = model.predict(t_X, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        confs = np.max(preds, axis=1)

        acc = float(accuracy_score(y_true, pred_classes))
        f1 = float(f1_score(y_true, pred_classes, average="weighted"))
        avg_conf = float(np.mean(confs))
        conf_change = float((avg_conf - orig_avg_conf) * 100.0)
        stability = float(np.mean(pred_classes == orig_classes) * 100.0)

        results.append({
            "transformation": t_type,
            "accuracy": round(acc * 100.0, 2),
            "f1_score": round(f1 * 100.0, 2),
            "avg_confidence": round(avg_conf * 100.0, 2),
            "confidence_change_pct": round(conf_change, 2),
            "stability_pct": round(stability, 2),
        })

    return results
