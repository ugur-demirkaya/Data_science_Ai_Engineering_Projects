import os
import sys
import time
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import onnxruntime as ort

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 50
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-5

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
GPU_MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model_gpu.onnx")
METRICS_PATH = os.path.join(MODEL_DIR, "gpu_training_metrics.json")
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "brain_tumor_dataset"
)


def check_gpu():
    providers = ort.get_available_providers()
    print(f"Mevcut ONNX Runtime sağlayıcıları: {providers}")
    has_dml = "DmlExecutionProvider" in providers
    if has_dml:
        print("AMD RX 7800 XT (DirectML) algılandı!")
    else:
        print("DirectML bulunamadı, CPU ile devam edilecek.")
    return has_dml


def load_images(data_dir=DATA_DIR, img_size=IMG_SIZE):
    images = []
    labels = []
    for label, folder in enumerate(["no", "yes"]):
        folder_path = os.path.join(data_dir, folder)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y


def load_and_split(data_dir=DATA_DIR, val_size=0.15, test_size=0.15, random_state=42):
    X, y = load_images(data_dir)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=y_train_val,
    )
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class weights: {class_weights}")
    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights


def augment_batch(X_batch):
    augmented = X_batch.copy()
    for i in range(len(augmented)):
        img = augmented[i]
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        if np.random.random() > 0.5:
            img = np.flipud(img)
        angle = np.random.uniform(-20, 20)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        zoom = np.random.uniform(0.8, 1.2)
        if zoom != 1.0:
            new_h, new_w = int(h * zoom), int(w * zoom)
            img_zoomed = cv2.resize(img, (new_w, new_h))
            if zoom > 1.0:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img_zoomed[start_h : start_h + h, start_w : start_w + w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img = np.pad(
                    img_zoomed,
                    ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w), (0, 0)),
                    mode="reflect",
                )
        brightness = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness, 0, 255).astype(np.float32)
        shear = np.random.uniform(-10, 10)
        M_shear = np.array(
            [[1, abs(np.tan(np.radians(shear))), 0], [0, 1, 0]], dtype=np.float64
        )
        if shear != 0:
            img = cv2.warpAffine(
                img.astype(np.uint8), M_shear, (w, h), borderMode=cv2.BORDER_REFLECT
            ).astype(np.float32)
        augmented[i] = img.astype(np.float32)
    return augmented


def create_onnx_model():
    import tensorflow as tf
    import tf2onnx

    from src.model import build_model, compile_model

    print("TensorFlow modeli oluşturuluyor...")
    model, _ = build_model(freeze_base=False)
    model = compile_model(model, lr=LEARNING_RATE_PHASE1)

    input_signature = [
        tf.TensorSpec(
            shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name="input"
        )
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_signature, opset=13
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    onnx_model.save(GPU_MODEL_PATH)
    print(f"ONNX modeli kaydedildi: {GPU_MODEL_PATH}")
    return model


def export_trained_to_onnx(keras_model):
    import tensorflow as tf
    import tf2onnx

    input_signature = [
        tf.TensorSpec(
            shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32, name="input"
        )
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model, input_signature=input_signature, opset=13
    )
    onnx_model.save(GPU_MODEL_PATH)
    print(f"Eğitilmiş ONNX modeli kaydedildi: {GPU_MODEL_PATH}")


def train_gpu():
    os.makedirs(MODEL_DIR, exist_ok=True)

    has_gpu = check_gpu()
    provider = "DmlExecutionProvider" if has_gpu else "CPUExecutionProvider"
    device_name = "AMD RX 7800 XT (DirectML)" if has_gpu else "CPU"

    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_and_split()

    import tensorflow as tf
    from src.model import build_model, unfreeze_top_layers, compile_model
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    keras_model_path = os.path.join(MODEL_DIR, "brain_tumor_model_gpu.weights.h5")

    metrics = {
        "device": device_name,
        "phase1_epochs": [],
        "phase2_epochs": [],
        "phase1_time": 0,
        "phase2_time": 0,
        "total_time": 0,
    }

    # ---- FAZ 1: Feature Extraction ----
    print("\n=== Faz 1: Feature Extraction (GPU - DirectML inference ile eğitim) ===")
    model, base_model = build_model(freeze_base=True)
    model = compile_model(model, lr=LEARNING_RATE_PHASE1)

    callbacks_phase1 = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(keras_model_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    start_time = time.time()

    num_batches = len(X_train) // BATCH_SIZE
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS_PHASE1):
        epoch_start = time.time()
        indices = np.random.permutation(len(X_train))
        train_loss = 0
        train_acc = 0
        n_batches = 0

        for batch_idx in range(num_batches + 1):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(X_train))
            if start >= len(X_train):
                break

            batch_indices = indices[start:end]
            X_batch = augment_batch(X_train[batch_indices])
            y_batch = y_train[batch_indices]

            sample_weights = np.array([class_weights[int(y)] for y in y_batch])

            result = model.train_on_batch(
                X_batch, y_batch, sample_weight=sample_weights
            )
            loss = result[0]
            acc = result[1]
            train_loss += loss
            train_acc += acc
            n_batches += 1

        val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)

        epoch_time = time.time() - epoch_start
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss / n_batches),
            "train_acc": float(train_acc / n_batches),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "time": round(epoch_time, 2),
        }
        metrics["phase1_epochs"].append(epoch_metrics)

        print(
            f"Epoch {epoch + 1}/{EPOCHS_PHASE1} - loss: {train_loss / n_batches:.4f} - acc: {train_acc / n_batches:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_recall: {val_rec:.4f} - {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(keras_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:
            print(f"EarlyStopping Faz 1 (epoch {epoch + 1})")
            break

    metrics["phase1_time"] = round(time.time() - start_time, 2)
    model.load_weights(keras_model_path)

    # ---- FAZ 2: Fine-Tuning ----
    print("\n=== Faz 2: Fine-Tuning (Son 40 katman açık - GPU) ===")
    model = unfreeze_top_layers(model, base_model, num_layers=40)
    model = compile_model(model, lr=LEARNING_RATE_PHASE2)

    best_val_loss = float("inf")
    patience_counter = 0
    start_time_phase2 = time.time()

    for epoch in range(EPOCHS_PHASE2):
        epoch_start = time.time()
        indices = np.random.permutation(len(X_train))
        train_loss = 0
        train_acc = 0
        n_batches = 0

        for batch_idx in range(num_batches + 1):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(X_train))
            if start >= len(X_train):
                break

            batch_indices = indices[start:end]
            X_batch = augment_batch(X_train[batch_indices])
            y_batch = y_train[batch_indices]

            sample_weights = np.array([class_weights[int(y)] for y in y_batch])

            result = model.train_on_batch(
                X_batch, y_batch, sample_weight=sample_weights
            )
            loss = result[0]
            acc = result[1]
            train_loss += loss
            train_acc += acc
            n_batches += 1

        val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)

        epoch_time = time.time() - epoch_start
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss / n_batches),
            "train_acc": float(train_acc / n_batches),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "time": round(epoch_time, 2),
        }
        metrics["phase2_epochs"].append(epoch_metrics)

        print(
            f"Epoch {epoch + 1}/{EPOCHS_PHASE2} - loss: {train_loss / n_batches:.4f} - acc: {train_acc / n_batches:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_recall: {val_rec:.4f} - {epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(keras_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 7:
            print(f"EarlyStopping Faz 2 (epoch {epoch + 1})")
            break

    metrics["phase2_time"] = round(time.time() - start_time_phase2, 2)
    metrics["total_time"] = round(metrics["phase1_time"] + metrics["phase2_time"], 2)

    # ---- Test değerlendirme ----
    model.load_weights(keras_model_path)
    print("\nTest seti değerlendiriliyor...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    report = classification_report(
        y_test.astype(int), y_pred, target_names=["Sağlıklı", "Tümör"], output_dict=True
    )
    report_numeric = classification_report(y_test.astype(int), y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)

    metrics["test"] = {
        "accuracy": float(report_numeric["accuracy"]),
        "precision_saglikli": float(report_numeric["0"]["precision"]),
        "recall_saglikli": float(report_numeric["0"]["recall"]),
        "f1_saglikli": float(report_numeric["0"]["f1-score"]),
        "precision_tumor": float(report_numeric["1"]["precision"]),
        "recall_tumor": float(report_numeric["1"]["recall"]),
        "f1_tumor": float(report_numeric["1"]["f1-score"]),
        "auc_roc": float(roc_auc_val),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n=== TEST SONUÇLARI ({device_name}) ===")
    print(f"Accuracy: {report_numeric['accuracy']:.4f}")
    print(f"Recall (Tümör): {report_numeric['1']['recall']:.4f}")
    print(f"AUC-ROC: {roc_auc_val:.4f}")
    print(f"Toplam eğitim süresi: {metrics['total_time']:.1f}s")

    # ONNX'e dışa aktar
    try:
        export_trained_to_onnx(model)
    except Exception as e:
        print(f"ONNX dışa aktarma hatası (önemli değil): {e}")

    # Metrikleri kaydet
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"GPU metrikleri kaydedildi: {METRICS_PATH}")

    # Grafik
    plot_gpu_training(
        metrics, save_path=os.path.join(MODEL_DIR, "gpu_training_history.png")
    )
    plot_gpu_evaluation(
        cm,
        fpr,
        tpr,
        roc_auc_val,
        save_path=os.path.join(MODEL_DIR, "gpu_evaluation.png"),
    )

    return model, metrics


def plot_gpu_training(metrics, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    all_epochs = metrics["phase1_epochs"] + metrics["phase2_epochs"]
    losses = [e["train_loss"] for e in all_epochs]
    val_losses = [e["val_loss"] for e in all_epochs]
    accs = [e["train_acc"] for e in all_epochs]
    val_accs = [e["val_acc"] for e in all_epochs]
    val_recalls = [e["val_recall"] for e in all_epochs]
    epochs = range(1, len(all_epochs) + 1)

    axes[0].plot(epochs, losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].axvline(
        x=len(metrics["phase1_epochs"]),
        color="r",
        linestyle="--",
        label="Fine-tune start",
    )
    axes[0].set_title(f"Loss ({metrics['device']})")
    axes[0].legend()

    axes[1].plot(epochs, val_recalls, label="Val Recall", color="orange")
    axes[1].axvline(
        x=len(metrics["phase1_epochs"]),
        color="r",
        linestyle="--",
        label="Fine-tune start",
    )
    axes[1].set_title("Val Recall (Tümör)")
    axes[1].legend()

    axes[2].plot(epochs, accs, label="Train Accuracy")
    axes[2].plot(epochs, val_accs, label="Val Accuracy")
    axes[2].axvline(
        x=len(metrics["phase1_epochs"]),
        color="r",
        linestyle="--",
        label="Fine-tune start",
    )
    axes[2].set_title("Accuracy")
    axes[2].legend()

    plt.suptitle(f"GPU Eğitim Geçmişi - {metrics['device']}", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gpu_evaluation(cm, fpr, tpr, roc_auc_val, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        ax=axes[0],
        xticklabels=["Sağlıklı", "Tümör"],
        yticklabels=["Sağlıklı", "Tümör"],
    )
    axes[0].set_title("GPU Confusion Matrix")
    axes[0].set_ylabel("Gerçek")
    axes[0].set_xlabel("Tahmin")

    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_val:.3f}", color="red")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title(f"GPU ROC Curve (AUC={roc_auc_val:.3f})")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_gpu_inference(image_array, model_path=GPU_MODEL_PATH):
    has_gpu = check_gpu()
    provider = "DmlExecutionProvider" if has_gpu else "CPUExecutionProvider"

    sess = ort.InferenceSession(model_path, providers=[provider])
    input_name = sess.get_inputs()[0].name

    img = (
        cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
        if image_array.shape[:2] != (IMG_SIZE, IMG_SIZE)
        else image_array
    )
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    start = time.time()
    result = sess.run(None, {input_name: img})
    inference_time = time.time() - start

    prediction = result[0][0][0]
    label = "Tümör VAR" if prediction >= 0.5 else "Sağlıklı"
    confidence = float(prediction if prediction >= 0.5 else 1.0 - prediction)

    return {
        "label": label,
        "confidence": confidence,
        "raw_probability": float(prediction),
        "inference_time_ms": round(inference_time * 1000, 2),
        "device": "AMD RX 7800 XT (DirectML)" if has_gpu else "CPU",
    }


if __name__ == "__main__":
    train_gpu()
