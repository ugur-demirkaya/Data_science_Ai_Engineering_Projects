import os
import sys
import numpy as np
import cv2
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

IMG_SIZE = 224
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")


def get_gradcam(image_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(image_array, axis=0))
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    return heatmap


def overlay_gradcam(original_image, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(
        heatmap, (original_image.shape[1], original_image.shape[0])
    )
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    if original_image.max() <= 1.0:
        overlay = original_image * 255.0 * (1 - alpha) + heatmap_colored * alpha
    else:
        overlay = (
            original_image.astype(np.float32) * (1 - alpha)
            + heatmap_colored.astype(np.float32) * alpha
        )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def generate_gradcam_image(image_path, model_path=MODEL_PATH):
    from src.model import build_model, compile_model

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32)

    model, _ = build_model(freeze_base=False)
    model = compile_model(model)
    model.load_weights(model_path)

    heatmap = get_gradcam(img_normalized, model)
    overlay = overlay_gradcam(img_resized, heatmap)

    return img_resized, overlay, heatmap
