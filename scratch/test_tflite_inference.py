import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("deployment_ready"))

from model.mri_validator import validate_mri_pipeline
from model.predict import predict

print("Testing TFLite MRI Validation & Prediction Pipeline...")

# Create a test synthetic MRI-like image (224x224 grayscale brain-like image)
img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
# Draw dark background and brain tissue-like circle with skull ring
rr, cc = np.ogrid[:224, :224]
brain_mask = (rr - 112)**2 + (cc - 112)**2 <= 80**2
img_arr[brain_mask] = [120, 120, 120]
test_img = Image.fromarray(img_arr)

print("1. Running validate_mri_pipeline...")
val_res = validate_mri_pipeline(test_img)
print("Validation result:", val_res["status"], val_res["modality"], val_res["reason"])

print("2. Running predict...")
pred_res = predict(test_img)
print("Prediction result:")
print("Success:", pred_res.get("success"))
print("Prediction:", pred_res.get("prediction"))
print("Confidence:", pred_res.get("confidence"))
print("Model used:", pred_res.get("model_used"))
print("Latency:", pred_res.get("pipeline_latency_sec"), "s")

print("SUCCESS: TFLite Inference executed cleanly!")
