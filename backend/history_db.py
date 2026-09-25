"""
Brain Tumor Detection — Analysis History Storage & SQLite Database Engine
Handles persistent storage of completed MRI analyses, patient metadata, image artifacts, and retrieval APIs.
"""

import os
import sys
import json
import sqlite3
import base64
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "analysis_history.db")
STORAGE_DIR = os.path.join(BASE_DIR, "history_storage")

os.makedirs(STORAGE_DIR, exist_ok=True)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the analysis_history table if it does not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date_str TEXT NOT NULL,
            time_str TEXT NOT NULL,
            patient_name TEXT DEFAULT 'Anonymous Patient',
            patient_id TEXT DEFAULT 'PT-2026-EX',
            age TEXT DEFAULT '',
            gender TEXT DEFAULT 'Male',
            referring_doctor TEXT DEFAULT 'Dr. Nivas',
            image_filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            raw_confidence REAL,
            calibrated_confidence REAL,
            validation_status TEXT DEFAULT 'Verified MRI',
            analysis_status TEXT DEFAULT 'Completed',
            risk_level TEXT DEFAULT 'None',
            tumor_detected INTEGER DEFAULT 0,
            image_path TEXT,
            overlay_path TEXT,
            ig_path TEXT,
            lime_path TEXT,
            characteristics_json TEXT,
            treatment TEXT,
            scores_json TEXT,
            model_used TEXT,
            description TEXT,
            color TEXT,
            is_uncertain INTEGER DEFAULT 0,
            uncertainty_reason TEXT
        );
    """)
    conn.commit()
    conn.close()


def generate_analysis_id() -> str:
    """Generates sequential Analysis ID: ANA-00001, ANA-00002..."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM analysis_history")
    count = cursor.fetchone()[0]
    conn.close()
    return f"ANA-{(count + 1):05d}"


def _save_b64_image(b64_str: str, filename: str) -> str:
    """Decodes a base64 image data URI and saves it as a file in STORAGE_DIR."""
    if not b64_str:
        return ""
    try:
        if "," in b64_str:
            _, encoded = b64_str.split(",", 1)
        else:
            encoded = b64_str
        
        img_data = base64.b64decode(encoded)
        filepath = os.path.join(STORAGE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(img_data)
        return filename
    except Exception as e:
        print(f"[HISTORY DB] Error saving image {filename}: {e}")
        return ""


def save_analysis_record(result_data: dict, patient_info: dict) -> str:
    """
    Saves a successful analysis result to the database and files.
    Returns the generated analysis_id.
    """
    init_db()
    analysis_id = generate_analysis_id()
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%I:%M %p")

    pt_name = patient_info.get("patient_name") or "Anonymous Patient"
    pt_id = patient_info.get("patient_id") or "PT-2026-EX"
    pt_age = str(patient_info.get("age", ""))
    pt_gender = patient_info.get("gender") or "Male"
    ref_doctor = patient_info.get("referring_doctor") or "Dr. Nivas"
    orig_filename = patient_info.get("image_filename") or "scan.jpg"

    # Save images to storage
    orig_file = _save_b64_image(result_data.get("original_b64"), f"{analysis_id}_orig.jpg")
    overlay_file = _save_b64_image(result_data.get("overlay_b64"), f"{analysis_id}_overlay.png")
    ig_file = _save_b64_image(result_data.get("ig_b64"), f"{analysis_id}_ig.png")
    lime_file = _save_b64_image(result_data.get("lime_b64"), f"{analysis_id}_lime.png")

    pred = result_data.get("prediction", "No Tumor")
    conf = result_data.get("confidence", 0.0)
    raw_conf = result_data.get("raw_confidence", conf)
    cal_conf = result_data.get("calibrated_confidence", conf)
    tumor_detected = 1 if result_data.get("tumor_detected", False) else 0
    risk_level = result_data.get("risk_level") or result_data.get("severity") or "None"
    status_str = "Uncertain Prediction" if result_data.get("is_uncertain") else "Completed"

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analysis_history (
            analysis_id, date_str, time_str, patient_name, patient_id, age, gender,
            referring_doctor, image_filename, prediction, confidence, raw_confidence,
            calibrated_confidence, validation_status, analysis_status, risk_level,
            tumor_detected, image_path, overlay_path, ig_path, lime_path,
            characteristics_json, treatment, scores_json, model_used, description,
            color, is_uncertain, uncertainty_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        analysis_id,
        date_str,
        time_str,
        pt_name,
        pt_id,
        pt_age,
        pt_gender,
        ref_doctor,
        orig_filename,
        pred,
        conf,
        raw_conf,
        cal_conf,
        "Verified MRI",
        status_str,
        risk_level,
        tumor_detected,
        orig_file,
        overlay_file,
        ig_file,
        lime_file,
        json.dumps(result_data.get("characteristics", [])),
        result_data.get("treatment", ""),
        json.dumps(result_data.get("scores", {})),
        result_data.get("model_used", "Multi-Model TTA Ensemble"),
        result_data.get("description", ""),
        result_data.get("color", "#6366f1"),
        1 if result_data.get("is_uncertain") else 0,
        result_data.get("uncertainty_reason", "")
    ))

    conn.commit()
    conn.close()

    print(f"[HISTORY DB] Saved analysis record {analysis_id} for patient '{pt_name}'")
    return analysis_id


def get_all_history(search=None, prediction_filter=None, date_filter=None, sort_order="newest"):
    """Fetches list of history records with filters and sorting."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM analysis_history WHERE 1=1"
    params = []

    if search:
        search_like = f"%{search.strip()}%"
        query += " AND (patient_name LIKE ? OR patient_id LIKE ? OR image_filename LIKE ? OR analysis_id LIKE ?)"
        params.extend([search_like, search_like, search_like, search_like])

    if prediction_filter and prediction_filter != "all":
        p_val = prediction_filter.lower().strip()
        if p_val == "notumor" or p_val == "no tumor":
            query += " AND (LOWER(prediction) = 'no tumor' OR LOWER(prediction) = 'notumor')"
        else:
            query += " AND LOWER(prediction) LIKE ?"
            params.append(f"%{p_val}%")

    if date_filter and date_filter != "all":
        today_str = datetime.now().strftime("%d-%m-%Y")
        if date_filter == "today":
            query += " AND date_str = ?"
            params.append(today_str)

    if sort_order == "oldest":
        query += " ORDER BY id ASC"
    else:
        query += " ORDER BY id DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    records = []
    for r in rows:
        records.append({
            "id": r["id"],
            "analysis_id": r["analysis_id"],
            "date_str": r["date_str"],
            "time_str": r["time_str"],
            "date_time": f"{r['date_str']} {r['time_str']}",
            "patient_name": r["patient_name"],
            "patient_id": r["patient_id"],
            "age": r["age"],
            "gender": r["gender"],
            "referring_doctor": r["referring_doctor"],
            "image_filename": r["image_filename"],
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "calibrated_confidence": r["calibrated_confidence"],
            "raw_confidence": r["raw_confidence"],
            "validation_status": r["validation_status"],
            "analysis_status": r["analysis_status"],
            "risk_level": r["risk_level"],
            "tumor_detected": bool(r["tumor_detected"]),
            "color": r["color"] or "#6366f1"
        })

    conn.close()
    return records


def get_history_stats():
    """Calculates dashboard statistics for total, today, tumor detected, and normal."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM analysis_history")
    total = cursor.fetchone()[0]

    today_str = datetime.now().strftime("%d-%m-%Y")
    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE date_str = ?", (today_str,))
    today_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE tumor_detected = 1")
    tumor_detected = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM analysis_history WHERE tumor_detected = 0")
    normal = cursor.fetchone()[0]

    conn.close()
    return {
        "total": total,
        "today": today_count,
        "tumor_detected": tumor_detected,
        "normal": normal
    }


def _file_to_b64(filename: str) -> str:
    """Reads a saved file from STORAGE_DIR and returns Data URI string."""
    if not filename:
        return ""
    filepath = os.path.join(STORAGE_DIR, filename)
    if not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, "rb") as f:
            b64_bytes = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(filename)[1].lower().replace('.', '')
        mime = "png" if ext == "png" else "jpeg"
        return f"data:image/{mime};base64,{b64_bytes}"
    except Exception as e:
        print(f"[HISTORY DB] Error reading file {filename}: {e}")
        return ""


def get_history_detail(analysis_id: str):
    """Fetches full historical analysis details by analysis_id."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE analysis_id = ?", (analysis_id,))
    r = cursor.fetchone()
    conn.close()

    if not r:
        return None

    characteristics = []
    if r["characteristics_json"]:
        try:
            characteristics = json.loads(r["characteristics_json"])
        except Exception:
            pass

    scores = {}
    if r["scores_json"]:
        try:
            scores = json.loads(r["scores_json"])
        except Exception:
            pass

    original_b64 = _file_to_b64(r["image_path"])
    overlay_b64 = _file_to_b64(r["overlay_path"])
    ig_b64 = _file_to_b64(r["ig_path"])
    lime_b64 = _file_to_b64(r["lime_path"])

    return {
        "success": True,
        "is_valid_mri": True,
        "analysis_id": r["analysis_id"],
        "date_str": r["date_str"],
        "time_str": r["time_str"],
        "date_time": f"{r['date_str']} {r['time_str']}",
        "patient_name": r["patient_name"],
        "patient_id": r["patient_id"],
        "age": r["age"],
        "gender": r["gender"],
        "referring_doctor": r["referring_doctor"],
        "image_filename": r["image_filename"],
        "prediction": r["prediction"],
        "display_name": r["prediction"],
        "confidence": r["confidence"],
        "raw_confidence": r["raw_confidence"],
        "calibrated_confidence": r["calibrated_confidence"],
        "validation_status": r["validation_status"],
        "analysis_status": r["analysis_status"],
        "risk_level": r["risk_level"],
        "severity": r["risk_level"],
        "tumor_detected": bool(r["tumor_detected"]),
        "color": r["color"] or "#6366f1",
        "description": r["description"] or "",
        "characteristics": characteristics,
        "treatment": r["treatment"] or "",
        "scores": scores,
        "model_used": r["model_used"] or "Multi-Model TTA Ensemble",
        "is_uncertain": bool(r["is_uncertain"]),
        "uncertainty_reason": r["uncertainty_reason"] or "",
        "original_b64": original_b64,
        "overlay_b64": overlay_b64,
        "ig_b64": ig_b64,
        "lime_b64": lime_b64
    }


def delete_history_record(analysis_id: str) -> bool:
    """Deletes a single history record and its saved image files."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, overlay_path, ig_path, lime_path FROM analysis_history WHERE analysis_id = ?", (analysis_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return False

    # Remove files
    for key in ["image_path", "overlay_path", "ig_path", "lime_path"]:
        fname = row[key]
        if fname:
            fpath = os.path.join(STORAGE_DIR, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"[HISTORY DB] Error deleting file {fpath}: {e}")

    cursor.execute("DELETE FROM analysis_history WHERE analysis_id = ?", (analysis_id,))
    conn.commit()
    conn.close()
    print(f"[HISTORY DB] Deleted record {analysis_id}")
    return True


def clear_all_history_records() -> int:
    """Clears all history records and deletes all files in STORAGE_DIR."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM analysis_history")
    count = cursor.fetchone()[0]

    cursor.execute("DELETE FROM analysis_history")
    conn.commit()
    conn.close()

    if os.path.exists(STORAGE_DIR):
        for f in os.listdir(STORAGE_DIR):
            fpath = os.path.join(STORAGE_DIR, f)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"[HISTORY DB] Error deleting file {fpath}: {e}")

    print(f"[HISTORY DB] Cleared all history ({count} records deleted)")
    return count
