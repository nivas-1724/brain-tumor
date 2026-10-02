"""
Brain Tumor Detection — Modality Classifier Training & Evaluation Script
Trains a 3-way multi-class Convolutional Neural Network:
- Class 0: MRI
- Class 1: CT
- Class 2: UNKNOWN

Outputs comprehensive classification metrics, confusion matrix, and CT false-negative analysis.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from model.modality_classifier import build_modality_classifier, IMG_SIZE, MODALITY_CLASSES, CLASS_INDICES, MODALITY_MODEL_PATH


def generate_ct_brain_samples(count=1500, img_size=224):
    """
    Generates realistic CT brain scan representations:
    - High-density bright skull bone calvarium ring (HU > +1000 -> intensity > 220)
    - CT brain parenchyma density mapping (mid-gray 75-115)
    - Ventricular Dark CSF attenuation (dark gray 15-40)
    - CT field-of-view radial boundaries and beam artifacts
    """
    print(f"[Train Validator] Generating {count} CT brain scan samples...")
    samples = []
    
    for i in range(count):
        img = Image.new("L", (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)
        
        center_x = img_size // 2 + np.random.randint(-4, 5)
        center_y = img_size // 2 + np.random.randint(-4, 5)
        radius_x = np.random.randint(70, 88)
        radius_y = np.random.randint(80, 96)
        
        # 1. Outer Bright Bone Skull Calvarium Ring (Characteristic CT feature: dense bone appears bright white >220)
        skull_bbox = [center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y]
        bone_brightness = np.random.randint(235, 256)
        draw.ellipse(skull_bbox, fill=bone_brightness)
        
        # 2. Inner Brain Tissue Parenchyma (CT brain window attenuation ~30-40 HU -> mid-gray)
        skull_thick = np.random.randint(8, 15)
        inner_bbox = [
            center_x - radius_x + skull_thick,
            center_y - radius_y + skull_thick,
            center_x + radius_x - skull_thick,
            center_y + radius_y - skull_thick
        ]
        tissue_gray = np.random.randint(75, 115)
        draw.ellipse(inner_bbox, fill=tissue_gray)
        
        # 3. Inner Sulci & Ventricles (Dark CSF attenuation)
        vent_w = np.random.randint(10, 18)
        vent_h = np.random.randint(22, 38)
        draw.ellipse(
            [center_x - vent_w, center_y - vent_h, center_x + vent_w, center_y + vent_h],
            fill=np.random.randint(15, 40)
        )
        
        # Add random CT sulci line attenuation patterns
        for _ in range(np.random.randint(4, 10)):
            sx1 = center_x + np.random.randint(-40, 40)
            sy1 = center_y + np.random.randint(-50, 50)
            sx2 = sx1 + np.random.randint(-20, 20)
            sy2 = sy1 + np.random.randint(-20, 20)
            draw.line([(sx1, sy1), (sx2, sy2)], fill=np.random.randint(40, 70), width=np.random.randint(2, 5))

        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, np.random.uniform(2, 6), arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        
        img_rgb = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        samples.append(np.array(img_rgb, dtype=np.float32))
        
    return np.array(samples, dtype=np.float32)


def generate_unknown_samples(count=1500, img_size=224):
    """
    Generates non-medical unknown images:
    - Random solid color fields
    - High-contrast geometric shapes & patterns
    - White paper / document text scans
    - Noise & natural scenes
    """
    print(f"[Train Validator] Generating {count} UNKNOWN non-medical samples...")
    samples = []
    
    for i in range(count):
        stype = i % 4
        if stype == 0:
            # Document / paper text scan
            img = Image.new("RGB", (img_size, img_size), color=(245, 245, 245))
            draw = ImageDraw.Draw(img)
            for y in range(15, img_size - 15, np.random.randint(12, 20)):
                draw.line([(15, y), (img_size - 15, y)], fill=(30, 30, 30), width=np.random.randint(1, 3))
        elif stype == 1:
            # Color photograph / natural texture
            arr = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
        elif stype == 2:
            # Geometric shapes on light background
            img = Image.new("RGB", (img_size, img_size), color=(220, 230, 240))
            draw = ImageDraw.Draw(img)
            for _ in range(5):
                box = [np.random.randint(0, 100), np.random.randint(0, 100), np.random.randint(100, 220), np.random.randint(100, 220)]
                draw.rectangle(box, fill=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        else:
            # Flat low-contrast gray/black noise
            val = np.random.randint(20, 180)
            arr = np.full((img_size, img_size, 3), val, dtype=np.uint8)
            img = Image.fromarray(arr)
            
        samples.append(np.array(img, dtype=np.float32))
        
    return np.array(samples, dtype=np.float32)


def load_dataset_mri_samples(dataset_dir="dataset", max_samples=1500, img_size=224):
    """Loads positive Brain MRI samples from dataset folder."""
    mri_paths = glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True)
    
    print(f"[Train Validator] Found {len(mri_paths)} total MRI files in dataset.")
    if len(mri_paths) > max_samples:
        np.random.seed(42)
        mri_paths = list(np.random.choice(mri_paths, max_samples, replace=False))

    samples = []
    for p in mri_paths:
        try:
            img = Image.open(p).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
            samples.append(np.array(img, dtype=np.float32))
        except Exception:
            pass

    print(f"[Train Validator] Successfully loaded {len(samples)} positive Brain MRI samples.")
    return np.array(samples, dtype=np.float32)


def train_and_evaluate_modality_classifier():
    """Main training and evaluation routine for the Modality Classifier."""
    print("=" * 65)
    print(" 🚀 STARTING MODALITY CLASSIFIER TRAINING & EVALUATION")
    print(f" Modality model classes: {MODALITY_CLASSES}")
    print(f" Class indices mapping: {CLASS_INDICES}")
    print("=" * 65)

    dataset_dir = os.path.join(BASE_DIR, "dataset")
    
    # 1. Load MRI samples (Class 0)
    mri_imgs = load_dataset_mri_samples(dataset_dir=dataset_dir, max_samples=1500, img_size=IMG_SIZE)
    mri_labels = np.zeros(len(mri_imgs), dtype=np.int32) # Class 0: MRI
    
    # 2. Generate CT samples (Class 1)
    ct_imgs = generate_ct_brain_samples(count=len(mri_imgs), img_size=IMG_SIZE)
    ct_labels = np.ones(len(ct_imgs), dtype=np.int32) # Class 1: CT
    
    # 3. Generate UNKNOWN samples (Class 2)
    unknown_imgs = generate_unknown_samples(count=len(mri_imgs), img_size=IMG_SIZE)
    unknown_labels = np.full(len(unknown_imgs), 2, dtype=np.int32) # Class 2: UNKNOWN
    
    # Combine datasets
    X = np.concatenate([mri_imgs, ct_imgs, unknown_imgs], axis=0)
    y = np.concatenate([mri_labels, ct_labels, unknown_labels], axis=0)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Split 80/20 train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n[Train Validator] Training dataset size: {len(X_train)} samples")
    print(f"[Train Validator] Validation dataset size: {len(X_val)} samples")
    
    model = build_modality_classifier(IMG_SIZE)
    model.summary()
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(MODALITY_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]
    
    print("\n[Train Validator] Training multi-class Modality Classifier (MRI vs CT vs UNKNOWN)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=4,
        batch_size=32,
        callbacks=callbacks
    )
    
    os.makedirs(os.path.dirname(MODALITY_MODEL_PATH), exist_ok=True)
    model.save(MODALITY_MODEL_PATH)
    print(f"\n✅ [Train Validator] Modality Classifier Model saved to '{MODALITY_MODEL_PATH}'")
    
    # ── EVALUATION ──
    print("\n" + "=" * 65)
    print(" 📊 MODALITY CLASSIFIER EVALUATION REPORT")
    print("=" * 65)
    val_probs = model.predict(X_val, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    
    print("\nClassification Report:")
    report = classification_report(y_val, val_preds, target_names=MODALITY_CLASSES, digits=4)
    print(report)
    
    cm = confusion_matrix(y_val, val_preds)
    print("Confusion Matrix:")
    print("               Pred MRI  Pred CT  Pred UNKNOWN")
    for idx, row_name in enumerate(MODALITY_CLASSES):
        print(f"True {row_name:<10}  {cm[idx][0]:<8} {cm[idx][1]:<8} {cm[idx][2]:<8}")
        
    # Check False MRI rate on CT
    ct_total = np.sum(y_val == 1)
    ct_as_mri = cm[1][0] if len(cm) > 1 else 0
    false_mri_rate = (ct_as_mri / ct_total) * 100.0 if ct_total > 0 else 0.0
    print(f"\nCRITICAL SAFETY METRIC — CT Misclassified as MRI: {ct_as_mri}/{ct_total} ({false_mri_rate:.2f}%)")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate_modality_classifier()
