"""
Model Architectures Module
Defines Custom Baseline CNN, ResNet50, EfficientNetB0, and MobileNetV2 for Brain MRI Classification.
Uses native Keras layers (Rescaling) to ensure 100% serialization compatibility without Lambda errors.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2

NUM_CLASSES = 4
IMG_SIZE = 224


def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Custom 4-Block Convolutional Neural Network (Baseline Model).
    Trained from scratch.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Custom_Baseline_CNN")
    return model


def build_resnet50_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """ResNet50 Transfer Learning Model."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=x)
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_Transfer")
    return model, base_model


def build_efficientnetb0_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """EfficientNetB0 Transfer Learning Model."""
    inputs = layers.Input(shape=input_shape)
    # EfficientNetB0 has internal scaling layer
    x = layers.Rescaling(1.0)(inputs)
    
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=x)
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Transfer")
    return model, base_model


def build_mobilenetv2_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """MobileNetV2 Transfer Learning Model."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=x)
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer")
    return model, base_model
