"""
Image Preprocessing Module for Brain MRI Classification
Provides memory-efficient data loading, normalization, and data augmentation.
"""

import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224
NUM_CLASSES = 4
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_MAP = {cls: idx for idx, cls in enumerate(CLASSES)}
INDEX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


def load_split_into_memory(split_dir, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Loads images from a split directory (Training or Testing) into numpy arrays.
    Returns:
        X: float32 numpy array of shape (N, 224, 224, 3) in range [0..255]
        y: one-hot encoded labels of shape (N, 4)
        filenames: list of file names
        filepaths: list of full paths
    """
    images = []
    labels = []
    filenames = []
    filepaths = []

    for cls in CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        cls_idx = CLASS_MAP[cls]
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = Image.open(fpath).convert('RGB').resize(target_size, Image.LANCZOS)
                    images.append(np.array(img, dtype=np.float32))
                    labels.append(cls_idx)
                    filenames.append(fname)
                    filepaths.append(fpath)
                except Exception:
                    pass

    X = np.array(images, dtype=np.float32)
    y = keras.utils.to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return X, y, filenames, filepaths


def get_data_augmentation():
    """Builds Keras Sequential data augmentation pipeline."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.03, 0.03),
    ], name="data_augmentation")


def preprocess_pil_image(pil_img, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocesses a single PIL Image into a batch array (1, 224, 224, 3) [0..255]."""
    img_resized = pil_img.convert('RGB').resize(target_size, Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32)
    return np.expand_dims(arr, axis=0)
