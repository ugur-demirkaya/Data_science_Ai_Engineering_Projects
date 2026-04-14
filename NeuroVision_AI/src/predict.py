import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.model import build_model, compile_model

IMG_SIZE = 224
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")


def predict_image(image_path, model_path=MODEL_PATH):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    model, _ = build_model(freeze_base=False)
    model = compile_model(model)
    model.load_weights(model_path)

    prediction = model.predict(img)[0][0]
    label = "Tümör VAR" if prediction >= 0.5 else "Sağlıklı"
    confidence = prediction if prediction >= 0.5 else 1.0 - prediction

    print(f"Tahmin: {label}")
    print(f"Güven: %{confidence * 100:.1f}")
    print(f"Ham olasılık: {prediction:.4f}")

    return {
        "label": label,
        "confidence": float(confidence),
        "raw_probability": float(prediction),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroVision AI - Beyin Tümörü Tahmini"
    )
    parser.add_argument("--image", required=True, help="MRI görüntü dosya yolu")
    parser.add_argument("--model", default=MODEL_PATH, help="Model dosya yolu")
    args = parser.parse_args()
    predict_image(args.image, args.model)
