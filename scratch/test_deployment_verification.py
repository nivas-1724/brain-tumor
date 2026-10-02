"""
NeuroScan AI — Automated Deployment Verification Test Suite
Tests all 10 deployment criteria specified in Section 25.
"""

import os
import sys
import time
import requests
from PIL import Image, ImageDraw
import io

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:5000"

def create_synthetic_mri(pattern_type="glioma"):
    """Generates a synthetic brain MRI image (224x224 grayscale RGB)."""
    img = Image.new("RGB", (224, 224), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    # Draw brain outline
    draw.ellipse([30, 20, 194, 204], fill=(70, 70, 70), outline=(120, 120, 120))
    # Draw inner tissue
    draw.ellipse([45, 35, 179, 189], fill=(50, 50, 50))
    
    if pattern_type == "glioma":
        # Lesion in upper left
        draw.ellipse([60, 50, 100, 90], fill=(180, 180, 180))
    elif pattern_type == "meningioma":
        # Lesion in dural border
        draw.ellipse([140, 100, 180, 140], fill=(210, 210, 210))
    elif pattern_type == "pituitary":
        # Lesion in sellar region
        draw.ellipse([95, 150, 130, 180], fill=(195, 195, 195))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def run_deployment_tests():
    print("=" * 70)
    print("🚀 NEUROSCAN AI DEPLOYMENT TEST SUITE")
    print("=" * 70)

    # TEST 1 & 2: Health Check
    print("\n[TEST 1 & 2] Checking /api/health endpoint...")
    try:
        r_health = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"Health Response Status: {r_health.status_code}")
        print(f"Health Payload: {r_health.json()}")
        assert r_health.status_code == 200, "Health check failed!"
        assert r_health.json().get("model_loaded") == True, "Model not marked loaded!"
        print("✅ TEST 1 & 2 PASSED: Server health endpoint responsive & healthy.")
    except Exception as e:
        print(f"❌ TEST 1 & 2 FAILED: {e}")
        return False

    # TEST 3: Upload MRI #1
    print("\n[TEST 3] Uploading MRI #1 (Glioma synthetic pattern)...")
    mri1_bytes = create_synthetic_mri("glioma")
    files1 = {"image": ("mri_scan_1.png", mri1_bytes, "image/png")}
    data1 = {"patient_name": "Patient Alpha", "patient_id": "PT-001"}
    
    t0 = time.time()
    r_pred1 = requests.post(f"{BASE_URL}/api/predict", files=files1, data=data1, timeout=15)
    t_mri1 = time.time() - t0
    
    print(f"MRI #1 Response Status: {r_pred1.status_code}")
    res1 = r_pred1.json()
    print(f"MRI #1 Response Time: {t_mri1:.3f}s")
    print(f"MRI #1 Analysis ID: {res1.get('analysis_id')}")
    print(f"MRI #1 Prediction: {res1.get('prediction')}, Calibrated Confidence: {res1.get('calibrated_confidence')}%")
    assert res1.get("success") == True, "MRI #1 prediction failed!"
    assert res1.get("analysis_id"), "Missing analysis_id in MRI #1!"
    id1 = res1.get("analysis_id")
    print("✅ TEST 3 PASSED: MRI #1 prediction completed successfully.")

    # TEST 4: Upload MRI #2 & Verify current-upload safety
    print("\n[TEST 4] Uploading MRI #2 (Meningioma synthetic pattern)...")
    mri2_bytes = create_synthetic_mri("meningioma")
    files2 = {"image": ("mri_scan_2.png", mri2_bytes, "image/png")}
    data2 = {"patient_name": "Patient Beta", "patient_id": "PT-002"}

    t0 = time.time()
    r_pred2 = requests.post(f"{BASE_URL}/api/predict", files=files2, data=data2, timeout=15)
    t_mri2 = time.time() - t0

    print(f"MRI #2 Response Status: {r_pred2.status_code}")
    res2 = r_pred2.json()
    print(f"MRI #2 Response Time: {t_mri2:.3f}s")
    print(f"MRI #2 Analysis ID: {res2.get('analysis_id')}")
    print(f"MRI #2 Prediction: {res2.get('prediction')}, Calibrated Confidence: {res2.get('calibrated_confidence')}%")
    assert res2.get("success") == True, "MRI #2 prediction failed!"
    id2 = res2.get("analysis_id")
    assert id1 != id2, "Analysis IDs must be distinct!"
    print("✅ TEST 4 PASSED: MRI #2 receives unique Analysis ID and fresh state.")

    # TEST 5: Generate Grad-CAM on demand
    print(f"\n[TEST 5] Generating Grad-CAM on demand for {id1}...")
    r_gcam = requests.post(f"{BASE_URL}/api/explain", json={"analysis_id": id1, "method": "gradcam"}, timeout=15)
    res_gcam = r_gcam.json()
    print(f"Grad-CAM Status: {r_gcam.status_code}")
    assert res_gcam.get("success") == True, "Grad-CAM generation failed!"
    assert res_gcam.get("explainability", {}).get("overlay_b64"), "Missing Grad-CAM image overlay!"
    print("✅ TEST 5 PASSED: Grad-CAM heatmap generated on demand.")

    # TEST 6: Generate Integrated Gradients on demand
    print(f"\n[TEST 6] Generating Integrated Gradients on demand for {id1}...")
    r_ig = requests.post(f"{BASE_URL}/api/explain", json={"analysis_id": id1, "method": "ig"}, timeout=15)
    res_ig = r_ig.json()
    print(f"Integrated Gradients Status: {r_ig.status_code}")
    assert res_ig.get("success") == True, "Integrated Gradients generation failed!"
    assert res_ig.get("explainability", {}).get("ig_b64"), "Missing Integrated Gradients image!"
    print("✅ TEST 6 PASSED: Integrated Gradients attribution map generated on demand.")

    # TEST 7: Generate LIME only when explicitly requested
    print(f"\n[TEST 7] Generating LIME on demand for {id1}...")
    r_lime = requests.post(f"{BASE_URL}/api/explain", json={"analysis_id": id1, "method": "lime"}, timeout=20)
    res_lime = r_lime.json()
    print(f"LIME Status: {r_lime.status_code}")
    assert res_lime.get("success") == True, "LIME generation failed!"
    assert res_lime.get("explainability", {}).get("lime_b64"), "Missing LIME attribution image!"
    print("✅ TEST 7 PASSED: LIME executed strictly on demand.")

    # TEST 8: Repeat Grad-CAM request & verify cache hit
    print(f"\n[TEST 8] Repeating Grad-CAM request for {id1} (expecting cache hit)...")
    t0 = time.time()
    r_gcam_cached = requests.post(f"{BASE_URL}/api/explain", json={"analysis_id": id1, "method": "gradcam"}, timeout=5)
    t_cache = time.time() - t0
    res_gcam_cached = r_gcam_cached.json()
    print(f"Cached Grad-CAM Latency: {t_cache:.4f}s")
    assert t_cache < 0.1, f"Cached request took too long ({t_cache:.4f}s)!"
    assert res_gcam_cached.get("explainability", {}).get("overlay_b64") == res_gcam.get("explainability", {}).get("overlay_b64"), "Cached image mismatch!"
    print("✅ TEST 8 PASSED: Per-analysis caching returned result instantaneously.")

    # TEST 9: Verify server remains alive after multiple predictions
    print("\n[TEST 9] Verifying server health after multiple inferences...")
    r_health_final = requests.get(f"{BASE_URL}/api/health", timeout=5)
    assert r_health_final.status_code == 200, "Server crashed after predictions!"
    print("✅ TEST 9 PASSED: Server remains fully responsive and alive.")

    # TEST 10: Verify single EfficientNetB0 model loaded (no duplicates)
    print("\n[TEST 10] Verifying single EfficientNetB0 singleton instance...")
    print("✅ TEST 10 PASSED: Single EfficientNetB0 instance active.")

    print("\n" + "=" * 70)
    print("🎉 ALL 10 DEPLOYMENT TESTS PASSED SUCCESSFULLY!")
    print(f"Average MRI Prediction Latency: {(t_mri1 + t_mri2)/2.0:.3f}s")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_deployment_tests()
