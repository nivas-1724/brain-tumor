"""
Brain Tumor Detection — Single & Batch Test Pipeline Script

Accepts an MRI image path (or tests a batch of known images), prints:
- Filename
- MRI validation status
- Predicted class
- Top Confidence
- Glioma, Meningioma, No Tumor, Pituitary probabilities
- Saves original MRI image and Grad-CAM activation heatmap overlay.
"""

import os
import sys
import glob
import base64
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import predict
from gradcam import overlay_gradcam, make_gradcam_heatmap


def test_single_mri(image_path, output_dir="test_results"):
    """Runs complete pipeline test on a single image file."""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    img_pil = Image.open(image_path).convert("RGB")
    result = predict(img_pil)

    filename = os.path.basename(image_path)
    scores = result.get("scores") or {}

    print("==================================================")
    print(f"Filename               : {filename}")
    print(f"MRI validation         : {'Valid Brain MRI' if result.get('is_valid_mri') else 'INVALID INPUT'}")
    print(f"Predicted class        : {result.get('display_name')}")
    print(f"Confidence             : {result.get('confidence'):.2f}%")
    print(f"Glioma probability     : {scores.get('glioma', {}).get('confidence', 0.0):.2f}%")
    print(f"Meningioma probability : {scores.get('meningioma', {}).get('confidence', 0.0):.2f}%")
    print(f"No Tumor probability   : {scores.get('notumor', {}).get('confidence', 0.0):.2f}%")
    print(f"Pituitary probability  : {scores.get('pituitary', {}).get('confidence', 0.0):.2f}%")
    print("==================================================")

    # Generate and save original image and Grad-CAM image
    if result.get("original_b64") and result.get("overlay_b64"):
        orig_save = os.path.join(output_dir, f"original_{filename}")
        grad_save = os.path.join(output_dir, f"gradcam_{filename}")
        img_pil.save(orig_save)
        
        # Save overlay image
        b64_str = result["overlay_b64"]
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_str)
        with open(grad_save, "wb") as f:
            f.write(raw_bytes)

        print(f"Saved original image to : {orig_save}")
        print(f"Saved Grad-CAM image to : {grad_save}")

    return result


def test_known_samples(samples_per_class=2):
    """Runs pipeline tests across multiple known samples from all classes + non-MRI inputs."""
    dataset_test_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "Testing")
    classes = ["glioma", "meningioma", "notumor", "pituitary"]

    print("\n🔬 RUNNING BATCH PIPELINE VERIFICATION TESTS ACROSS KNOWN SAMPLES...")

    for cls in classes:
        cls_dir = os.path.join(dataset_test_dir, cls)
        if os.path.exists(cls_dir):
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:samples_per_class]
            for fpath in files:
                test_single_mri(fpath)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single_mri(sys.argv[1])
    else:
        test_known_samples()
