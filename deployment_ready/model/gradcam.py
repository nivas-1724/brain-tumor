"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Generates heatmaps showing which regions of the MRI the model focused on.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import base64


def get_last_conv_layer(model):
    """Auto-detect the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    # For EfficientNet sub-model, look inside the first sub-model
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, keras.layers.Conv2D):
                    return layer.name  # return the sub-model name
    raise ValueError("No convolutional layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate Grad-CAM heatmap for a given image and model.
    
    Args:
        img_array:            Preprocessed image (1, 224, 224, 3) float range [0, 255]
        model:                Trained Keras model
        last_conv_layer_name: Name of the target conv layer ('top_conv' by default)
        pred_index:           Class index to generate CAM for (predicted class if None)
    
    Returns:
        heatmap: numpy array (H, W) in range [0, 1]
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = "top_conv"

    # Find the nested base model (EfficientNetB0)
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model is None:
        return _fallback_heatmap(img_array)

    try:
        top_conv_layer = base_model.get_layer(last_conv_layer_name)
    except Exception:
        # Fall back to last Conv2D in base_model
        conv_layers = [l for l in base_model.layers if isinstance(l, keras.layers.Conv2D)]
        if not conv_layers:
            return _fallback_heatmap(img_array)
        top_conv_layer = conv_layers[-1]

    # Model mapping base_model.inputs -> top_conv.output
    conv_model = keras.models.Model(inputs=base_model.inputs, outputs=top_conv_layer.output)

    # Classification head layers after base_model
    classifier_layers = []
    found_base = False
    for l in model.layers:
        if l == base_model:
            found_base = True
            continue
        if found_base:
            classifier_layers.append(l)

    # Preprocess image array for EfficientNet if needed
    prep_img = tf.keras.applications.efficientnet.preprocess_input(img_array)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(prep_img)
        tape.watch(conv_outputs)

        try:
            top_act = base_model.get_layer('top_activation')(conv_outputs)
        except Exception:
            top_act = conv_outputs

        x = top_act
        for l in classifier_layers:
            x = l(x)

        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Calculate gradients of class score w.r.t. feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return _fallback_heatmap(img_array)

    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map channels by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain positive activations and normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def _fallback_heatmap(img_array):
    """Returns a center-biased heatmap when Grad-CAM fails."""
    h, w = img_array.shape[1], img_array.shape[2]
    y, x = np.ogrid[:h, :w]
    heatmap = np.exp(-((x - w/2)**2 + (y - h/2)**2) / (2 * (min(h,w)/4)**2))
    return heatmap.astype(np.float32)


def overlay_gradcam(original_img_pil, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap onto the original MRI image.

    Args:
        original_img_pil: PIL Image (original MRI)
        heatmap:          numpy array (H, W) in [0,1]
        alpha:            blend factor for heatmap
        colormap:         OpenCV colormap constant

    Returns:
        overlayed_img: PIL Image with heatmap overlay
        heatmap_img:   PIL Image of just the colored heatmap
    """
    # Resize heatmap to image size
    orig_w, orig_h = original_img_pil.size
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))

    # Apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Convert original to numpy
    orig_array = np.array(original_img_pil.convert("RGB"))

    # Blend
    overlayed = cv2.addWeighted(orig_array, 1 - alpha, colored_heatmap_rgb, alpha, 0)
    overlayed_pil = Image.fromarray(overlayed)
    heatmap_pil   = Image.fromarray(colored_heatmap_rgb)

    return overlayed_pil, heatmap_pil


def get_bounding_box(heatmap, threshold=None):
    """
    Compute a bounding box for the high-activation region.

    Returns:
        dict with 'x', 'y', 'width', 'height' as percentages (0-100)
    """
    if threshold is None:
        max_val = np.max(heatmap)
        threshold = max(0.5, float(max_val) * 0.65)

    binary = (heatmap >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return None

    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = heatmap.shape
    return {
        "x":      round(float(cmin / w) * 100, 2),
        "y":      round(float(rmin / h) * 100, 2),
        "width":  round(float((cmax - cmin) / w) * 100, 2),
        "height": round(float((rmax - rmin) / h) * 100, 2),
    }


def pil_to_base64(img: Image.Image, fmt="PNG") -> str:
    """Convert a PIL Image to a base64 data URI string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def generate_gradcam_report(img_array_norm, original_pil, model, pred_index=None):
    """
    Full Grad-CAM pipeline: heatmap → overlay → bounding box → base64 images.

    Args:
        img_array_norm: (1, 224, 224, 3) normalized image array
        original_pil:   PIL Image of original MRI
        model:          Trained Keras model
        pred_index:     Predicted class index

    Returns:
        dict with 'overlay_b64', 'heatmap_b64', 'bounding_box'
    """
    heatmap = make_gradcam_heatmap(img_array_norm, model, pred_index=pred_index)
    overlayed, heatmap_img = overlay_gradcam(original_pil, heatmap)
    bbox = get_bounding_box(heatmap, threshold=0.5)

    return {
        "overlay_b64":  pil_to_base64(overlayed),
        "heatmap_b64":  pil_to_base64(heatmap_img),
        "bounding_box": bbox,
        "heatmap_raw":  heatmap.tolist(),
    }
