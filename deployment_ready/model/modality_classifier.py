"""
Brain Tumor Detection — Modality Classifier Architecture & Class Mapping
Defines the deep learning model architecture for multi-class modality classification (MRI, CT, UNKNOWN).

Class Mapping:
- Class 0: MRI (Brain Magnetic Resonance Imaging)
- Class 1: CT (Brain Computed Tomography)
- Class 2: UNKNOWN (Non-medical / invalid / document / photo / non-brain)
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224
MODALITY_CLASSES = ["MRI", "CT", "UNKNOWN"]
CLASS_INDICES = {"MRI": 0, "CT": 1, "UNKNOWN": 2}
INDEX_TO_CLASS = {0: "MRI", 1: "CT", 2: "UNKNOWN"}
MODALITY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved", "modality_classifier_model.h5")


def get_class_mapping():
    """Returns official class mapping dictionary."""
    return CLASS_INDICES


def build_modality_classifier(img_size=224):
    """
    Builds a robust convolutional neural network for 3-way modality classification:
    - Class 0: MRI
    - Class 1: CT
    - Class 2: UNKNOWN
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image_input")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Convolutional Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Global Classifier Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(3, activation="softmax", name="modality_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Modality_Classifier_CNN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
