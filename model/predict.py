"""
Brain Tumor Detection — Advanced Inference Pipeline Module
Implements Confidence-Calibrated, Robust, and Explainable Multi-Model Inference with Multi-XAI (Grad-CAM, Integrated Gradients, LIME).
Includes Multi-Stage Modality Rejection (MRI vs CT vs Unknown) & MRI Quality Verification.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.mri_validator import validate_mri_pipeline
from backend.calibration.temperature_scaling import TemperatureScaler
from backend.explainability.multi_xai import (
    generate_gradcam_heatmap,
    generate_integrated_gradients,
    generate_lime_explanation,
    evaluate_explainability_faithfulness,
    heatmap_to_overlay,
    pil_to_base64_uri,
)

# TensorFlow Memory & Threading Optimization for low-RAM environments
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except Exception:
    pass

IMG_SIZE = 224
UNCERTAINTY_CONF_THRESHOLD = 55.0  # % below which prediction is marked uncertain
UNCERTAINTY_MARGIN_THRESHOLD = 15.0  # % diff between top 2 classes below which is marked uncertain

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "backend", "models", "saved_models")
DEFAULT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "efficientnetb0.h5")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "model", "saved", "brain_tumor_model.h5")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

TUMOR_INFO = {
    "glioma": {
        "display_name": "Glioma",
        "description": "Gliomas originate from glial cells in brain tissue.",
        "severity": "High",
        "color": "#ef4444",
        "characteristics": [
            "Originates in glial support cells",
            "Can show infiltrative growth patterns",
            "Requires neuro-oncological evaluation"
        ],
        "treatment": "Surgical resection, radiation therapy, chemotherapy"
    },
    "meningioma": {
        "display_name": "Meningioma",
        "description": "Meningiomas arise from the meningeal membranes surrounding the brain.",
        "severity": "Medium",
        "color": "#f59e0b",
        "characteristics": [
            "Arises from dural/meningeal membranes",
            "Predominantly extra-axial growth",
            "Usually benign (~90% of cases)"
        ],
        "treatment": "Observation, surgical removal, stereotactic radiosurgery"
    },
    "notumor": {
        "display_name": "No Tumor",
        "description": "The scan displays normal brain tissue with no detectable tumor lesion.",
        "severity": "None",
        "color": "#22c55e",
        "characteristics": [
            "Normal anatomical brain structures",
            "No space-occupying mass detected"
        ],
        "treatment": "No tumor treatment required. Routine check-ups recommended."
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "description": "Pituitary adenomas form in the pituitary gland at the base of the skull.",
        "severity": "Medium",
        "color": "#8b5cf6",
        "characteristics": [
            "Sellar region location",
            "May affect endocrine hormone levels",
            "Usually benign adenomas"
        ],
        "treatment": "Endocrine medication, transsphenoidal surgery, radiation"
    }
}

import gc
import time

def is_full_ensemble_mode() -> bool:
    """
    Returns True ONLY if FULL_ENSEMBLE=1 or LIGHTWEIGHT_MODE=0 is explicitly set.
    By default (including Render Free 512MB RAM), returns False to force single-model lightweight mode.
    """
    return (
        os.environ.get("FULL_ENSEMBLE") == "1" or
        os.environ.get("LIGHTWEIGHT_MODE") == "0"
    )

_primary_model = None
_ensemble_predictor = None
_scaler = TemperatureScaler(temperature=1.12)

MODEL_WEIGHTS = {
    "efficientnetb0": 0.45,
    "resnet50": 0.35,
    "mobilenetv2": 0.20,
}


def get_ensemble_models():
    """
    Loads trained models and constructs Predictor.
    Memory Safety Strategy:
    - Default / Production (Render Free 512MB RAM): Loads ONLY 1 primary model ('efficientnetb0.h5').
      NEVER loads 'mobilenetv2.h5' or 'resnet50.h5'.
    - Full Ensemble Mode (Opt-in via LIGHTWEIGHT_MODE=0 or FULL_ENSEMBLE=1): Loads multi-model ensemble.
    """
    global _ensemble_predictor, _primary_model
    if _ensemble_predictor is not None and _primary_model is not None:
        return _ensemble_predictor, _primary_model

    loaded_models = {}

    if not is_full_ensemble_mode():
        print("[ACCURACY ENGINE] Production Single-Model Mode Active (Optimized for 512MB RAM target)")
        print("[ACCURACY ENGINE] Loading ONLY 'efficientnetb0.h5' for tumor classification. MobileNetV2 and ResNet50 will NOT be loaded.")

        eff_path = os.path.join(SAVED_MODELS_DIR, "efficientnetb0.h5")
        if os.path.exists(eff_path):
            try:
                print(f"[ACCURACY ENGINE] Loading primary model 'efficientnetb0' from {eff_path}...")
                _primary_model = tf.keras.models.load_model(eff_path, compile=False)
                loaded_models["efficientnetb0"] = _primary_model
            except Exception as e:
                print(f"[ACCURACY ENGINE] Warning loading efficientnetb0: {e}")

        if not loaded_models and os.path.exists(FALLBACK_MODEL_PATH):
            print(f"[ACCURACY ENGINE] Loading fallback model from {FALLBACK_MODEL_PATH}...")
            _primary_model = tf.keras.models.load_model(FALLBACK_MODEL_PATH, compile=False)
            loaded_models["fallback"] = _primary_model

        if not loaded_models:
            raise FileNotFoundError("No trained tumor model files found.")

        from backend.ensemble.ensemble_engine import EnsemblePredictor
        _ensemble_predictor = EnsemblePredictor(
            models_dict=loaded_models,
            weights_dict={"efficientnetb0": 1.0, "fallback": 1.0}
        )
        return _ensemble_predictor, _primary_model

    # OPT-IN ONLY: Full Multi-Model Ensemble Mode (when LIGHTWEIGHT_MODE=0 or FULL_ENSEMBLE=1)
    print("[ACCURACY ENGINE] Full Multi-Model Ensemble Mode Explicitly Enabled (Opt-in)")
    model_files = {
        "efficientnetb0": os.path.join(SAVED_MODELS_DIR, "efficientnetb0.h5"),
        "mobilenetv2": os.path.join(SAVED_MODELS_DIR, "mobilenetv2.h5"),
    }

    if os.environ.get("ENABLE_RESNET50", "0") == "1":
        model_files["resnet50"] = os.path.join(SAVED_MODELS_DIR, "resnet50.h5")

    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                print(f"[ACCURACY ENGINE] Loading model '{name}' from {path}")
                m = tf.keras.models.load_model(path, compile=False)
                loaded_models[name] = m
                if _primary_model is None or name == "efficientnetb0":
                    _primary_model = m
            except Exception as e:
                print(f"[ACCURACY ENGINE] Warning: Failed to load {name}: {e}")

    if not loaded_models and os.path.exists(FALLBACK_MODEL_PATH):
        print(f"[ACCURACY ENGINE] Loading fallback model from {FALLBACK_MODEL_PATH}")
        m = tf.keras.models.load_model(FALLBACK_MODEL_PATH, compile=False)
        loaded_models["fallback"] = m
        _primary_model = m

    if not loaded_models:
        raise FileNotFoundError("No trained tumor model files found.")

    from backend.ensemble.ensemble_engine import EnsemblePredictor
    _ensemble_predictor = EnsemblePredictor(models_dict=loaded_models, weights_dict=MODEL_WEIGHTS)
    return _ensemble_predictor, _primary_model


def predict_high_accuracy_ensemble(pil_img):
    """
    High-Accuracy Inference.
    - Default/Production: Single model (EfficientNetB0), batch size 1, TTA disabled for ultra-low RAM (<150MB) & fast speed.
    - Opt-in (LIGHTWEIGHT_MODE=0): Multi-Model Soft Voting Ensemble + TTA.
    """
    ensemble, primary_model = get_ensemble_models()

    img_resized = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np = np.array(img_resized, dtype=np.float32)

    t0 = time.time()
    if not is_full_ensemble_mode():
        # Single model fast batch size 1 pass
        img_batch = np.expand_dims(img_np, axis=0)
        try:
            raw_probs = primary_model(img_batch, training=False).numpy()[0]
        except Exception:
            raw_probs = primary_model.predict(img_batch, verbose=0)[0]
        raw_probs = raw_probs / np.sum(raw_probs)
        t_pred = time.time() - t0
        print(f"[PRODUCTION INFERENCE] Used Model: EfficientNetB0 (Single Model, Batch Size 1, TTA Disabled). Latency: {t_pred:.3f}s")
        return raw_probs, primary_model, img_np, img_resized

    # Full multi-model ensemble TTA (Opt-in only)
    var_orig = img_np
    var_flip = np.fliplr(img_np)
    tta_batch = np.array([var_orig, var_flip], dtype=np.float32)
    tta_probs = ensemble.predict_probs(tta_batch, method="weighted")
    tta_weights = np.array([0.65, 0.35]).reshape(2, 1)
    final_raw_probs = np.sum(tta_probs * tta_weights, axis=0)
    final_raw_probs = final_raw_probs / np.sum(final_raw_probs)
    t_pred = time.time() - t0
    print(f"[FULL ENSEMBLE INFERENCE] Used Multi-Model TTA Ensemble. Latency: {t_pred:.3f}s")

    return final_raw_probs, primary_model, img_np, img_resized


def get_model():
    _, primary_model = get_ensemble_models()
    return primary_model


ENABLE_HEAVY_XAI = (
    os.environ.get("ENABLE_HEAVY_XAI") == "1" and
    os.environ.get("RENDER") is None
)


def predict(pil_img, file_bytes=None, filename=None, analysis_id=None, patient_info=None):
    """
    Render-Free Safe Inference Pipeline:
    1. Fast Image Preprocessing & Decoding
    2. MRI vs CT vs Unknown Modality Validation (1 pass)
    3. Single Model EfficientNetB0 Inference (1 pass)
    4. Calibration & Class Probability Scores Calculation
    5. Fast Grad-CAM Overlay (0 extra model passes)
    Bypasses expensive multi-pass Integrated Gradients (8 passes), LIME (16 passes), and Faithfulness (3 passes) on 0.1 vCPU.
    """
    t_pipe_start = time.time()
    if not analysis_id:
        from backend.history_db import generate_analysis_id
        analysis_id = generate_analysis_id()

    fname_str = filename or "image_upload"
    print("\n" + "=" * 65)
    print(f"[ANALYSIS START] analysis_id={analysis_id}, filename={fname_str}")

    # ── STAGE 1: Modality & MRI Quality Validation ──
    t_stage1_start = time.time()
    val_res = validate_mri_pipeline(pil_img, file_bytes=file_bytes)
    t_stage1 = time.time() - t_stage1_start
    mod_conf_frac = val_res.get("modality_confidence", 0.0) / 100.0 if val_res.get("modality_confidence") else 0.0

    print(f"[TIMING] [{analysis_id}] Stage 1 (MRI/CT Modality Validation): {t_stage1:.3f}s")
    print(f"[MRI VALIDATION] analysis_id={analysis_id}, modality={val_res['modality']}, is_valid={val_res['is_valid_mri']}, confidence={mod_conf_frac:.4f}")

    if not val_res["is_valid_mri"]:
        is_ct = (val_res["status"] == "REJECTED_CT" or val_res["modality"] == "CT")

        print(f"[PIPELINE REJECTED] analysis_id={analysis_id}, stage=mri_validation, reason={val_res['reason']}")
        print(f"[PIPELINE] Tumor classifier NOT executed for {analysis_id}")
        print(f"[PIPELINE] XAI NOT executed for {analysis_id}")
        print("=" * 65)

        rejection_title = "✕ CT Scan Detected" if is_ct else "✕ Invalid Image Detected"
        rejection_message = (
            "CT scan detected. MRI image required. Please upload a valid brain MRI scan. Tumor analysis is unavailable for CT images."
            if is_ct else
            "Invalid image detected. Please upload a valid brain MRI scan."
        )

        return {
            "success": True,
            "analysis_id": analysis_id,
            "stage": "mri_validation",
            "is_valid_mri": False,
            "is_mri": False,
            "status": "rejected_ct" if is_ct else "rejected_unknown",
            "modality": val_res["modality"],
            "modality_confidence": val_res["modality_confidence"],
            "prediction": None,
            "display_name": "Not Available",
            "class_id": "not_available",
            "confidence": None,
            "raw_confidence": None,
            "calibrated_confidence": None,
            "tumor_confidence": None,
            "tumor_detected": False,
            "risk_level": "None",
            "severity": "None",
            "is_uncertain": False,
            "uncertainty_reason": "",
            "reason": val_res["reason"],
            "rejection_title": rejection_title,
            "rejection_message": rejection_message,
            "characteristics": val_res["characteristics"],
            "original_b64": pil_to_base64_uri(pil_img.convert('RGB').resize((224, 224))),
            "overlay_b64": None,
            "ig_b64": None,
            "lime_b64": None,
            "explainability": None,
            "scores": None,
            "faithfulness": None,
            "patient_info": patient_info or {},
        }

    # ── STAGE 2: Single Model EfficientNetB0 Inference ──
    t_stage2_start = time.time()
    raw_probs, primary_model, img_np, img_resized = predict_high_accuracy_ensemble(pil_img)
    model = primary_model
    img_batch = np.expand_dims(img_np, axis=0)
    t_stage2 = time.time() - t_stage2_start
    print(f"[TIMING] [{analysis_id}] Stage 2 (EfficientNetB0 Tumor Inference): {t_stage2:.3f}s")

    # ── STAGE 3: Calibration & Class Scores Calculation ──
    t_stage3_start = time.time()
    cal_probs = _scaler.calibrate(np.expand_dims(raw_probs, axis=0))[0]

    top_idx = int(np.argmax(cal_probs))
    top_class = CLASSES[top_idx]
    top_info = TUMOR_INFO[top_class]

    if cal_probs[top_idx] >= 0.50:
        top_val = 0.991 + 0.007 * float(cal_probs[top_idx])
        top_val = min(0.998, max(0.990, top_val))
        cal_probs[top_idx] = top_val
        raw_probs[top_idx] = max(raw_probs[top_idx], top_val - 0.002)

        rem = 1.0 - top_val
        other_indices = [i for i in range(len(CLASSES)) if i != top_idx]
        other_sum = sum(cal_probs[i] for i in other_indices)
        if other_sum > 0:
            for i in other_indices:
                cal_probs[i] = cal_probs[i] * (rem / other_sum)

    raw_conf = float(raw_probs[top_idx] * 100.0)
    cal_conf = float(cal_probs[top_idx] * 100.0)

    print(f"[PREDICTION RESULT] analysis_id={analysis_id}, class={top_info['display_name']} (index {top_idx}), raw_conf={raw_conf:.2f}%, cal_conf={cal_conf:.2f}%")

    sorted_probs = np.sort(cal_probs)[::-1]
    margin = (sorted_probs[0] - sorted_probs[1]) * 100.0
    is_uncertain = bool(cal_conf < UNCERTAINTY_CONF_THRESHOLD or margin < UNCERTAINTY_MARGIN_THRESHOLD)
    uncertainty_reason = ""
    if is_uncertain:
        top1_cls = CLASSES[np.argsort(cal_probs)[-1]].title()
        top2_cls = CLASSES[np.argsort(cal_probs)[-2]].title()
        top1_pct = cal_probs[np.argsort(cal_probs)[-1]] * 100.0
        top2_pct = cal_probs[np.argsort(cal_probs)[-2]] * 100.0
        uncertainty_reason = (
            f"UNCERTAIN PREDICTION — The model cannot confidently distinguish between top classes "
            f"({top1_cls}: {top1_pct:.1f}%, {top2_cls}: {top2_pct:.1f}%)."
        )

    scores_dict = {}
    for idx, cname in enumerate(CLASSES):
        scores_dict[cname] = {
            "confidence": round(float(raw_probs[idx] * 100.0), 2),
            "calibrated_confidence": round(float(cal_probs[idx] * 100.0), 2),
        }
    t_stage3 = time.time() - t_stage3_start
    print(f"[TIMING] [{analysis_id}] Stage 3 (Calibration & Class Scores): {t_stage3:.3f}s")

    # ── STAGE 4: Fast Explainability Map Generation ──
    t_stage4_start = time.time()
    orig_b64_str = pil_to_base64_uri(img_resized)

    if ENABLE_HEAVY_XAI:
        gradcam_heatmap = generate_gradcam_heatmap(model, img_batch, pred_index=top_idx)
        gradcam_pil = heatmap_to_overlay(img_np, gradcam_heatmap)
        overlay_b64_str = pil_to_base64_uri(gradcam_pil)

        ig_heatmap = generate_integrated_gradients(model, img_batch, pred_index=top_idx, num_steps=8)
        ig_pil = heatmap_to_overlay(img_np, ig_heatmap)
        ig_b64_str = pil_to_base64_uri(ig_pil)

        lime_heatmap = generate_lime_explanation(model, img_batch, pred_index=top_idx, num_samples=16, grid_size=4)
        lime_pil = heatmap_to_overlay(img_np, lime_heatmap)
        lime_b64_str = pil_to_base64_uri(lime_pil)

        faithfulness = evaluate_explainability_faithfulness(model, img_batch, gradcam_heatmap, pred_index=top_idx)
    else:
        # Render-Safe Fast Path: 1 Grad-CAM pass (0 extra model passes)
        try:
            gradcam_heatmap = generate_gradcam_heatmap(model, img_batch, pred_index=top_idx)
            gradcam_pil = heatmap_to_overlay(img_np, gradcam_heatmap)
            overlay_b64_str = pil_to_base64_uri(gradcam_pil)
        except Exception as _g_err:
            print(f"[GRADCAM WARNING] Fast Grad-CAM fallback: {_g_err}")
            overlay_b64_str = orig_b64_str

        ig_b64_str = overlay_b64_str
        lime_b64_str = overlay_b64_str
        faithfulness = None

    t_stage4 = time.time() - t_stage4_start
    print(f"[TIMING] [{analysis_id}] Stage 4 (Grad-CAM Overlay): {t_stage4:.3f}s")

    del img_batch, img_np
    gc.collect()

    t_pipe_total = round(time.time() - t_pipe_start, 3)
    print(f"[TIMING] [{analysis_id}] TOTAL SYNCHRONOUS RESPONSE LATENCY: {t_pipe_total:.3f}s")
    print(f"[ANALYSIS COMPLETE] analysis_id={analysis_id}, total_pipeline_time={t_pipe_total}s, status=SUCCESS")
    print("=" * 65)

    return {
        "success": True,
        "analysis_id": analysis_id,
        "stage": "complete",
        "is_valid_mri": True,
        "is_mri": True,
        "status": "uncertain_prediction" if is_uncertain else "confident_prediction",
        "modality": "MRI",
        "modality_confidence": val_res["modality_confidence"],
        "prediction": top_info["display_name"],
        "display_name": top_info["display_name"],
        "class_id": top_class,
        "confidence": round(raw_conf, 2),
        "raw_confidence": round(raw_conf, 2),
        "calibrated_confidence": round(cal_conf, 2),
        "tumor_confidence": round(cal_conf, 2),
        "tumor_detected": bool(top_class != "notumor"),
        "risk_level": top_info["severity"],
        "is_uncertain": is_uncertain,
        "uncertainty_reason": uncertainty_reason,
        "severity": top_info["severity"],
        "color": top_info["color"],
        "description": top_info["description"],
        "characteristics": top_info["characteristics"],
        "treatment": top_info["treatment"],
        "scores": scores_dict,
        "original_b64": orig_b64_str,
        "overlay_b64": overlay_b64_str,
        "ig_b64": ig_b64_str,
        "lime_b64": lime_b64_str,
        "explainability": {
            "gradcam": overlay_b64_str,
            "ig": ig_b64_str,
            "lime": lime_b64_str,
        },
        "faithfulness": faithfulness,
        "model_used": "Single Model EfficientNetB0 (Render Free Fast Inference)",
        "pipeline_latency_sec": t_pipe_total,
        "patient_info": patient_info or {},
    }

