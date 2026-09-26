"""
Multi-Method Explainable AI (XAI) & Quantitative Faithfulness Module
Implements Grad-CAM, Integrated Gradients, LIME, and Quantitative Faithfulness Verification.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.linear_model import Ridge


def find_target_conv_layer(model):
    """Finds the last convolutional layer in the Keras model."""
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        # If nested base model
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    return sub_layer.name
    return None


def generate_gradcam_heatmap(model, img_batch, pred_index=None):
    """
    Generates Grad-CAM activation heatmap for the target predicted class.
    img_batch: (1, 224, 224, 3) [0..255]
    """
    grad_model = getattr(model, "_cached_grad_model", None)
    if grad_model is None:
        target_layer_name = find_target_conv_layer(model)
        if not target_layer_name:
            return np.zeros((224, 224), dtype=np.float32)

        try:
            conv_layer = model.get_layer(target_layer_name)
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[conv_layer.output, model.output]
            )
        except Exception:
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    try:
                        conv_layer = layer.get_layer(target_layer_name)
                        grad_model = tf.keras.models.Model(
                            inputs=[model.inputs],
                            outputs=[conv_layer.output, model.output]
                        )
                        break
                    except Exception:
                        pass

        if grad_model is not None:
            model._cached_grad_model = grad_model

    if grad_model is None:
        return np.zeros((224, 224), dtype=np.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap_np = heatmap.numpy()
    heatmap_resized = cv2.resize(heatmap_np, (224, 224))
    return heatmap_resized


def generate_integrated_gradients(model, img_batch, pred_index=None, num_steps=8):
    """
    Computes Integrated Gradients attribution map relative to a black baseline image.
    img_batch: (1, 224, 224, 3) [0..255]
    """
    baseline = np.zeros_like(img_batch)
    if pred_index is None:
        try:
            preds = model(img_batch, training=False).numpy()
        except Exception:
            preds = model.predict(img_batch, verbose=0)
        pred_index = np.argmax(preds[0])

    alphas = np.linspace(0.0, 1.0, num_steps)
    interpolated_batch = np.concatenate([baseline + a * (img_batch - baseline) for a in alphas], axis=0)

    interpolated_tensor = tf.convert_to_tensor(interpolated_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_tensor)
        preds = model(interpolated_tensor)
        target_preds = preds[:, pred_index]

    grads = tape.gradient(target_preds, interpolated_tensor).numpy()

    # Riemann sum approximation
    avg_grads = np.mean(grads, axis=0)
    delta = (img_batch - baseline)[0]
    ig_map = np.abs(delta * avg_grads[0])
    ig_map = np.mean(ig_map, axis=-1)  # Average across color channels

    # Normalize [0..1]
    ig_map = (ig_map - np.min(ig_map)) / (np.max(ig_map) - np.min(ig_map) + 1e-10)
    return ig_map


def generate_lime_explanation(model, img_batch, pred_index=None, num_samples=16, grid_size=4):
    """
    Generates ultra-fast LIME superpixel attribution explanation.
    Splits image into grid_size x grid_size superpixels and fits a linear surrogate model.
    """
    img_2d = img_batch[0]  # (224, 224, 3)
    H, W, C = img_2d.shape
    patch_h = H // grid_size
    patch_w = W // grid_size
    num_masks = grid_size * grid_size

    if pred_index is None:
        try:
            preds = model(img_batch, training=False).numpy()
        except Exception:
            preds = model.predict(img_batch, verbose=0)
        pred_index = np.argmax(preds[0])

    # Generate random binary perturbation masks
    binary_masks = np.random.binomial(1, 0.7, size=(num_samples, num_masks))
    # Always include full unmasked image
    binary_masks[0] = 1

    perturbed_images = []
    for mask in binary_masks:
        p_img = img_2d.copy()
        for idx, val in enumerate(mask):
            if val == 0:
                r = idx // grid_size
                c = idx % grid_size
                p_img[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w, :] = 0
        perturbed_images.append(p_img)

    perturbed_batch = np.array(perturbed_images, dtype=np.float32)
    try:
        preds = model(perturbed_batch, training=False).numpy()
    except Exception:
        preds = model.predict(perturbed_batch, verbose=0)
    target_probs = preds[:, pred_index]

    # Fit Ridge linear surrogate model
    ridge = Ridge(alpha=1.0)
    ridge.fit(binary_masks, target_probs)

    weights = ridge.coef_
    lime_map = np.zeros((H, W), dtype=np.float32)
    for idx, w in enumerate(weights):
        r = idx // grid_size
        c = idx % grid_size
        lime_map[r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w] = max(0, w)

    lime_map = (lime_map - np.min(lime_map)) / (np.max(lime_map) - np.min(lime_map) + 1e-10)
    return lime_map


def evaluate_explainability_faithfulness(model, img_batch, attribution_map, pred_index=None, mask_pct=0.15):
    """
    Quantitatively evaluates explanation faithfulness by masking top important pixels vs random pixels.
    Uses a single batch size=3 pass for ultra-fast execution.
    """
    flat_attr = attribution_map.flatten()
    k = int(len(flat_attr) * mask_pct)
    top_indices = np.argpartition(flat_attr, -k)[-k:]

    # Mask top important pixels (set to 0)
    img_important_masked = img_batch.copy()
    flat_imp = img_important_masked[0].reshape(-1, 3)
    flat_imp[top_indices] = 0
    img_important_masked[0] = flat_imp.reshape(224, 224, 3)

    # Mask random pixels
    random_indices = np.random.choice(len(flat_attr), size=k, replace=False)
    img_random_masked = img_batch.copy()
    flat_rand = img_random_masked[0].reshape(-1, 3)
    flat_rand[random_indices] = 0
    img_random_masked[0] = flat_rand.reshape(224, 224, 3)

    # Run single batch of 3 images [orig, important_masked, random_masked]
    batch_3 = np.concatenate([img_batch, img_important_masked, img_random_masked], axis=0)
    try:
        preds_3 = model(batch_3, training=False).numpy()
    except Exception:
        preds_3 = model.predict(batch_3, verbose=0)

    if pred_index is None:
        pred_index = np.argmax(preds_3[0])

    orig_conf = float(preds_3[0, pred_index])
    imp_conf = float(preds_3[1, pred_index])
    rand_conf = float(preds_3[2, pred_index])

    important_drop = max(0.0, (orig_conf - imp_conf) * 100.0)
    random_drop = max(0.0, (orig_conf - rand_conf) * 100.0)

    return {
        "original_confidence": round(orig_conf * 100.0, 2),
        "important_drop_pct": round(important_drop, 2),
        "random_drop_pct": round(random_drop, 2),
        "is_faithful": bool(important_drop > random_drop),
    }


def heatmap_to_overlay(img_np, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.45):
    """
    Overlays a normalized heatmap [0..1] onto an RGB image array (224, 224, 3) [0..255].
    Returns RGB PIL Image.
    """
    heatmap_255 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_255, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    img_uint8 = np.uint8(img_np)
    overlay = cv2.addWeighted(img_uint8, 1.0 - alpha, colored_heatmap, alpha, 0)
    return Image.fromarray(overlay)


def pil_to_base64_uri(pil_img):
    """Converts a PIL Image to a base64 data URI string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
