"""
Brain Tumor Detection — Research REST API
Provides endpoints for MRI image classification, Multi-XAI, Calibration, Model Benchmarking, Error Analysis, and CSV Downloads.
"""

import os
import sys
import json
import traceback
import io
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "model"))
sys.path.insert(0, BASE_DIR)

from model.predict import predict
from report import generate_report
from history_db import (
    init_db,
    generate_analysis_id,
    save_analysis_record,
    get_all_history,
    get_history_stats,
    get_history_detail,
    delete_history_record,
    clear_all_history_records,
    STORAGE_DIR,
)

init_db()

def decode_image_bytes(file_bytes: bytes, filename: str = "") -> Image.Image:
    """
    Decodes raw file bytes into a PIL Image (RGB) supporting DICOM, PNG, JPG, BMP, WEBP, etc.
    """
    if not file_bytes:
        raise ValueError("Empty file bytes.")

    is_dcm_ext = filename.lower().endswith(".dcm")
    has_dicm_magic = len(file_bytes) >= 132 and file_bytes[128:132] == b"DICM"

    if is_dcm_ext or has_dicm_magic:
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            arr = ds.pixel_array.astype(np.float32)
            arr_min, arr_max = np.min(arr), np.max(arr)
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
            else:
                arr = np.zeros_like(arr)
            arr = arr.astype(np.uint8)

            if arr.ndim == 2:
                return Image.fromarray(arr).convert("RGB")
            elif arr.ndim == 3:
                return Image.fromarray(arr).convert("RGB")
            else:
                return Image.fromarray(arr[0]).convert("RGB")
        except Exception as dicom_err:
            print(f"[IMAGE DECODER] DICOM decode warning: {dicom_err}, trying standard image decode...")

    return Image.open(io.BytesIO(file_bytes)).convert("RGB")

# Pre-load & warm-up AI models once at application startup for low-RAM efficiency
try:
    import time
    import gc
    t_start_load = time.time()
    print("[STARTUP] Pre-loading AI models into memory...")
    from model.mri_validator import load_modality_model
    from model.predict import get_ensemble_models
    
    mod_model = load_modality_model()
    ens_pred, prim_model = get_ensemble_models()
    
    # Warm-up TensorFlow execution graph once at startup
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    if mod_model is not None:
        try:
            _ = mod_model(dummy_input, training=False)
        except Exception:
            pass
    if prim_model is not None:
        try:
            _ = prim_model(dummy_input, training=False)
        except Exception:
            pass
            
    print(f"[STARTUP] AI models pre-loaded and warmed up in {time.time() - t_start_load:.2f}s successfully.")
    gc.collect()
except Exception as _preload_err:
    print(f"[STARTUP] Model pre-loading warning: {_preload_err}")

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp", ".dcm"}
MAX_FILE_SIZE_MB = 16


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def error_response(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "running",
        "title": "Confidence-Calibrated, Robust and Explainable Multi-Model Brain MRI Classification",
        "version": "2.0.0-research",
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return error_response("No image file provided. Send as 'image' in multipart form.")

    file = request.files["image"]
    if file.filename == "":
        return error_response("Empty filename.")

    if not allowed_file(file.filename):
        return error_response(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return error_response(f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB.")

    analysis_id = generate_analysis_id()
    print("\n" + "=" * 65)
    print(f"[ANALYSIS START] analysis_id={analysis_id}, filename={file.filename}")

    patient_info = {
        "analysis_id": analysis_id,
        "patient_name": request.form.get("patient_name", "Anonymous Patient").strip() or "Anonymous Patient",
        "patient_id": request.form.get("patient_id", "PT-2026-EX").strip() or "PT-2026-EX",
        "age": request.form.get("age", "").strip(),
        "gender": request.form.get("gender", "Male").strip() or "Male",
        "referring_doctor": request.form.get("referring_doctor", "Dr. Nivas").strip() or "Dr. Nivas",
        "image_filename": file.filename,
    }

    ext = os.path.splitext(file.filename.lower())[1] or ".png"
    unique_filename = f"{analysis_id}_original{ext}"
    saved_upload_path = os.path.join(STORAGE_DIR, unique_filename)

    try:
        img_bytes = file.read()
        with open(saved_upload_path, "wb") as f_out:
            f_out.write(img_bytes)
        print(f"[UPLOAD] analysis_id={analysis_id}, saved_path={saved_upload_path}")
    except Exception as upload_err:
        print(f"[UPLOAD WARNING] Could not save raw file to storage: {upload_err}")

    try:
        pil_img = decode_image_bytes(img_bytes, filename=file.filename)
    except Exception as e:
        print(f"[DECODE ERROR] analysis_id={analysis_id}, error={e}")
        return error_response(f"Could not open image: {str(e)}")

    try:
        result = predict(
            pil_img,
            file_bytes=img_bytes,
            filename=file.filename,
            analysis_id=analysis_id,
            patient_info=patient_info
        )

        # Save history ONLY if valid MRI scan
        if result.get("is_valid_mri", True):
            try:
                saved_id = save_analysis_record(result, patient_info, analysis_id=analysis_id)
                result["analysis_id"] = saved_id
                result["saved_to_history"] = True
                print(f"[HISTORY] analysis_id={saved_id}, record saved successfully")
            except Exception as hist_err:
                print(f"[HISTORY ERROR] Failed to save history record for {analysis_id}: {hist_err}")
                traceback.print_exc()
                result["saved_to_history"] = False
        else:
            print(f"[PIPELINE REJECTED] analysis_id={analysis_id}, stage=mri_validation, is_mri=False")
            result["saved_to_history"] = False

        result["patient_info"] = patient_info
        print(f"[ANALYSIS COMPLETE] analysis_id={analysis_id}")
        print("=" * 65 + "\n")

        gc.collect()
        return jsonify({"success": True, **result})
    except FileNotFoundError as e:
        return error_response(str(e), 503)
    except Exception as e:
        traceback.print_exc()
        return error_response(f"Prediction failed: {str(e)}", 500)


# ─────────────────────────────────────────────
# ANALYSIS HISTORY API ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def get_history_api():
    search = request.args.get("search", "")
    prediction = request.args.get("prediction", "all")
    date_filter = request.args.get("date", "all")
    sort = request.args.get("sort", "newest")

    try:
        records = get_all_history(
            search=search,
            prediction_filter=prediction,
            date_filter=date_filter,
            sort_order=sort
        )
        stats = get_history_stats()
        return jsonify({"success": True, "history": records, "stats": stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to fetch history: {str(e)}"}), 500


@app.route("/api/history/<analysis_id>", methods=["GET"])
def get_history_detail_api(analysis_id):
    if not analysis_id or not (analysis_id.startswith("ANL-") or analysis_id.startswith("ANA-")):
        return error_response("Invalid Analysis ID format.")

    try:
        detail = get_history_detail(analysis_id)
        if not detail:
            return jsonify({"success": False, "error": f"Analysis record '{analysis_id}' not found."}), 404
        return jsonify({"success": True, "detail": detail})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to fetch record: {str(e)}"}), 500


@app.route("/api/history/<analysis_id>", methods=["DELETE"])
def delete_history_api(analysis_id):
    if not analysis_id or not (analysis_id.startswith("ANL-") or analysis_id.startswith("ANA-")):
        return error_response("Invalid Analysis ID format.")

    try:
        success = delete_history_record(analysis_id)
        if not success:
            return jsonify({"success": False, "error": f"Analysis record '{analysis_id}' not found."}), 404
        stats = get_history_stats()
        return jsonify({"success": True, "message": f"Deleted {analysis_id}", "stats": stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to delete record: {str(e)}"}), 500


@app.route("/api/history", methods=["DELETE"])
def clear_history_api():
    try:
        cleared_count = clear_all_history_records()
        stats = get_history_stats()
        return jsonify({"success": True, "message": f"Cleared all history ({cleared_count} records).", "stats": stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to clear history: {str(e)}"}), 500


@app.route("/api/history/image/<filename>", methods=["GET"])
def get_history_image_api(filename):
    if ".." in filename or "/" in filename or "\\" in filename:
        return error_response("Invalid image filename.")

    filepath = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(filepath):
        return error_response("Image file not found.", 404)

    ext = os.path.splitext(filename)[1].lower()
    mimetype = "image/png" if ext == ".png" else "image/jpeg"
    return send_file(filepath, mimetype=mimetype)


@app.route("/api/report", methods=["POST"])
def api_report():
    data = request.get_json(force=True, silent=True)
    if not data:
        return error_response("Invalid JSON body.")

    prediction_data = data.get("prediction_data")
    patient_info = data.get("patient_info", {})

    if not prediction_data:
        return error_response("Missing 'prediction_data' in request body.")

    try:
        pdf_bytes = generate_report(prediction_data, patient_info)
    except Exception as e:
        traceback.print_exc()
        return error_response(f"Report generation failed: {str(e)}", 500)

    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = 'attachment; filename="brain_mri_research_report.pdf"'
    return response


# ─────────────────────────────────────────────
# EXPERIMENTAL RESEARCH API ENDPOINTS
# ─────────────────────────────────────────────
@app.route("/api/experiments/comparison", methods=["GET"])
def get_model_comparison():
    summary_path = os.path.join(ROOT_DIR, "results", "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "comparison": data})
    
    # Run dynamic evaluation if json not created yet
    try:
        from experiments.evaluate_models import run_evaluation_experiment
        data, _ = run_evaluation_experiment()
        return jsonify({"success": True, "comparison": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/experiments/calibration", methods=["GET"])
def get_calibration_results():
    summary_path = os.path.join(ROOT_DIR, "results", "calibration_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "calibration": data})
    
    try:
        from experiments.calibration_exp import run_calibration_experiment
        data = run_calibration_experiment()
        return jsonify({"success": True, "calibration": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/experiments/robustness", methods=["GET"])
def get_robustness_results():
    summary_path = os.path.join(ROOT_DIR, "results", "robustness_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "robustness": data})

    try:
        from experiments.robustness_exp import run_robustness_experiment
        data = run_robustness_experiment()
        return jsonify({"success": True, "robustness": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/experiments/error-analysis", methods=["GET"])
def get_error_analysis():
    summary_path = os.path.join(ROOT_DIR, "results", "error_analysis_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "error_analysis": data})

    try:
        from experiments.error_analysis import run_error_analysis
        data = run_error_analysis()
        return jsonify({"success": True, "error_analysis": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/experiments/ablation", methods=["GET"])
def get_ablation_study():
    summary_path = os.path.join(ROOT_DIR, "results", "ablation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "ablation": data})

    try:
        from experiments.ablation_exp import run_ablation_study
        data = run_ablation_study()
        return jsonify({"success": True, "ablation": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/results/csv/<filename>", methods=["GET"])
def download_csv(filename):
    allowed_csvs = {
        "model_comparison.csv",
        "calibration_results.csv",
        "robustness_results.csv",
        "ablation_results.csv",
        "misclassification_results.csv",
        "prediction_results.csv",
    }
    if filename not in allowed_csvs:
        return error_response("CSV file not found or disallowed.")

    csv_path = os.path.join(ROOT_DIR, "results", "csv", filename)
    if not os.path.exists(csv_path):
        return error_response(f"File {filename} has not been generated yet.", 404)

    return send_file(csv_path, as_attachment=True, download_name=filename, mimetype="text/csv")


@app.route("/api/samples", methods=["GET"])
def api_samples():
    import base64
    dataset_dir = os.path.join(ROOT_DIR, "dataset")
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    samples = {}

    for cls in classes:
        img_path = None
        for split in ["Testing", "Training"]:
            cls_dir = os.path.join(dataset_dir, split, cls)
            if os.path.isdir(cls_dir):
                files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if files:
                    img_path = os.path.join(cls_dir, sorted(files)[0])
                    break
        
        if img_path and os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                mime = 'png' if ext == 'png' else 'jpeg'
                samples[cls] = {
                    "class": cls,
                    "filename": os.path.basename(img_path),
                    "data_url": f"data:image/{mime};base64,{b64}",
                }
            except Exception:
                samples[cls] = None
        else:
            samples[cls] = None

    return jsonify({"success": True, "samples": samples})


if __name__ == "__main__":
    print("=" * 65)
    print(" 🧠 NeuroScan AI — Research Platform Server")
    print("    Framework: Confidence-Calibrated, Robust and Explainable Multi-Model Brain MRI Classification")
    print("=" * 65)
    print(" URL  : http://localhost:5000")
    print(" API  : http://localhost:5000/api/predict")
    print("=" * 65)
    app.run(host="0.0.0.0", port=5000, debug=False)
