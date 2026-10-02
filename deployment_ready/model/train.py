"""
Brain Tumor Detection - Model Training Script
High-Accuracy EfficientNetB0 Transfer Learning + Fine-Tuning
Classes: Glioma, Meningioma, No Tumor, Pituitary
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "IMG_SIZE":        224,
    "BATCH_SIZE":      32,
    "PHASE1_EPOCHS":   12,
    "PHASE2_EPOCHS":   15,
    "PHASE1_LR":       1e-3,
    "PHASE2_LR":       1e-4,
    "CLASSES":         ["glioma", "meningioma", "notumor", "pituitary"],
    "DATA_DIR":        "dataset",
    "MODEL_DIR":       "model/saved",
    "MODEL_NAME":      "brain_tumor_model.h5",
    "HISTORY_FILE":    "model/saved/training_history.json",
}

IMG_SIZE    = CONFIG["IMG_SIZE"]
BATCH_SIZE  = CONFIG["BATCH_SIZE"]
NUM_CLASSES = len(CONFIG["CLASSES"])
os.makedirs(CONFIG["MODEL_DIR"], exist_ok=True)

CLASS_MAP = {cls: idx for idx, cls in enumerate(CONFIG["CLASSES"])}


# ─────────────────────────────────────────────
# FAST IN-MEMORY DATASET LOADING
# ─────────────────────────────────────────────
def load_split_into_memory(split_dir):
    """Load images [0..255] and labels into numpy arrays."""
    images = []
    labels = []
    
    for cls in CONFIG["CLASSES"]:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        cls_idx = CLASS_MAP[cls]
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath).convert('RGB').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                    images.append(np.array(img, dtype=np.float32)) # 0..255 range
                    labels.append(cls_idx)
                except Exception:
                    pass

    X = np.array(images, dtype=np.float32)
    y = keras.utils.to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return X, y


def prepare_datasets():
    train_dir = os.path.join(CONFIG["DATA_DIR"], "Training")
    test_dir  = os.path.join(CONFIG["DATA_DIR"], "Testing")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at '{CONFIG['DATA_DIR']}'.\n"
            "Please run: python setup_dataset.py"
        )

    print("\n⚡ Loading Training split into memory...")
    X_train_full, y_train_full = load_split_into_memory(train_dir)
    print(f"   Loaded {len(X_train_full)} training images")

    print("\n⚡ Loading Testing split into memory...")
    X_test, y_test = load_split_into_memory(test_dir)
    print(f"   Loaded {len(X_test)} testing images")

    # Stratified validation split (15%)
    num_samples = len(X_train_full)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    val_count = int(num_samples * 0.15)
    val_idx   = indices[:val_count]
    train_idx = indices[val_count:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val     = X_train_full[val_idx], y_train_full[val_idx]

    print(f"\n✅ Dataset Summary:")
    print(f"   Train samples      : {len(X_train)}")
    print(f"   Validation samples : {len(X_val)}")
    print(f"   Test samples       : {len(X_test)}")
    print(f"   Classes            : {CONFIG['CLASSES']}")

    with open(os.path.join(CONFIG["MODEL_DIR"], "class_indices.json"), "w") as f:
        json.dump(CLASS_MAP, f, indent=2)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ─────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(num_classes=4):
    """Build EfficientNetB0 model with preprocess_input and augmentation."""
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.05, 0.05),
    ], name="data_augmentation")

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Lambda(preprocess_input, name="efficientnet_preprocess")(x)

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["PHASE1_LR"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model, base_model


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────
def train(model, base_model, train_data, val_data):
    X_train, y_train = train_data
    X_val, y_val     = val_data
    model_path = os.path.join(CONFIG["MODEL_DIR"], CONFIG["MODEL_NAME"])

    callbacks_phase1 = [
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    print("\n🔵 Phase 1: Training classifier head (base frozen)...")
    history1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["PHASE1_EPOCHS"],
        batch_size=BATCH_SIZE,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    # ── Phase 2: Unfreeze top 50 layers of EfficientNetB0 for fine-tuning ──
    print("\n🟡 Phase 2: Fine-tuning top base layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["PHASE2_LR"]),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    callbacks_phase2 = [
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]

    history2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["PHASE2_EPOCHS"],
        batch_size=BATCH_SIZE,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history.get(key, [])

    with open(CONFIG["HISTORY_FILE"], "w") as f:
        json.dump(combined, f, indent=2)

    return model, combined


# ─────────────────────────────────────────────
# EVALUATION & PLOTS
# ─────────────────────────────────────────────
def evaluate(model, test_data):
    X_test, y_test = test_data
    print("\n📊 Evaluating model on independent test set...")
    results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    print(f"\n   🎯 Test Accuracy : {results[1]*100:.2f}%")
    print(f"   🎯 Test AUC      : {results[2]:.4f}")

    preds   = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred  = np.argmax(preds, axis=1)
    y_true  = np.argmax(y_test, axis=1)
    labels  = CONFIG["CLASSES"]

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Brain Tumor Detection")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plot_path = os.path.join(CONFIG["MODEL_DIR"], "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Confusion matrix saved → {plot_path}")
    plt.close()

    return results


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#334155")

    axes[0].plot(history["accuracy"], color="#6366f1", linewidth=2, label="Train")
    axes[0].plot(history["val_accuracy"], color="#22d3ee", linewidth=2, label="Val")
    axes[0].set_title("Accuracy", color="white", fontsize=14)
    axes[0].legend(facecolor="#1e293b", labelcolor="white")
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].set_ylabel("Accuracy", color="white")

    axes[1].plot(history["loss"], color="#f43f5e", linewidth=2, label="Train")
    axes[1].plot(history["val_loss"], color="#fb923c", linewidth=2, label="Val")
    axes[1].set_title("Loss", color="white", fontsize=14)
    axes[1].legend(facecolor="#1e293b", labelcolor="white")
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss", color="white")

    plt.tight_layout()
    plot_path = os.path.join(CONFIG["MODEL_DIR"], "training_history.png")
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"✅ Training history plot saved → {plot_path}")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🧠  Brain Tumor Detection — Model Training (High Accuracy)")
    print("=" * 60)

    train_data, val_data, test_data = prepare_datasets()
    model, base_model = build_model(num_classes=NUM_CLASSES)

    model, history = train(model, base_model, train_data, val_data)
    evaluate(model, test_data)
    plot_training_history(history)

    print("\n🎉 High-accuracy model training completed successfully!")
    print("   Model saved to:", os.path.join(CONFIG["MODEL_DIR"], CONFIG["MODEL_NAME"]))
