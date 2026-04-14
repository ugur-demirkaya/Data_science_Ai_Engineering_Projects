import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.data_loader import load_and_split, get_augmentation
from src.model import build_model, unfreeze_top_layers, compile_model

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_model.keras")
CPU_METRICS_PATH = os.path.join(MODEL_DIR, "cpu_training_metrics.json")
BATCH_SIZE = 16
EPOCHS_PHASE1 = 15
EPOCHS_PHASE2 = 50


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["recall"], label="Train Recall")
    axes[1].plot(history.history["val_recall"], label="Val Recall")
    axes[1].set_title("Recall (Tümör)")
    axes[1].legend()

    axes[2].plot(history.history["accuracy"], label="Train Accuracy")
    axes[2].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[2].set_title("Accuracy")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Eğitim grafiği kaydedildi: {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test, save_path=None):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("\n=== SINIFLANDIRMA RAPORU ===")
    print(classification_report(y_test, y_pred, target_names=["Sağlıklı", "Tümör"]))

    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Sağlıklı", "Tümör"],
        yticklabels=["Sağlıklı", "Tümör"],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Gerçek")
    axes[0].set_xlabel("Tahmin")

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title(f"ROC Curve (AUC={roc_auc:.3f})")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Değerlendirme grafiği kaydedildi: {save_path}")
    plt.close()

    return classification_report(y_test, y_pred, output_dict=True)


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    cpu_metrics = {
        "device": "CPU",
        "phase1_time": 0,
        "phase2_time": 0,
        "total_time": 0,
        "test": {},
    }

    print("Veri yükleniyor...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_weights = load_and_split()

    print("\n=== Faz 1: Feature Extraction (Dondurulmuş base) ===")
    model, base_model = build_model(freeze_base=True)
    model = compile_model(model, lr=1e-4)
    model.summary()

    callbacks_phase1 = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    aug = get_augmentation()
    aug.fit(X_train)

    start_phase1 = time.time()
    history1 = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS_PHASE1,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks_phase1,
    )
    cpu_metrics["phase1_time"] = round(time.time() - start_phase1, 2)

    print("\n=== Faz 2: Fine-Tuning (Son 40 katman açık) ===")
    model = unfreeze_top_layers(model, base_model, num_layers=40)
    model = compile_model(model, lr=1e-5)

    callbacks_phase2 = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8),
    ]

    start_phase2 = time.time()
    history2 = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS_PHASE2,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks_phase2,
    )
    cpu_metrics["phase2_time"] = round(time.time() - start_phase2, 2)
    cpu_metrics["total_time"] = round(
        cpu_metrics["phase1_time"] + cpu_metrics["phase2_time"], 2
    )

    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    class HistoryContainer:
        pass

    h = HistoryContainer()
    h.history = combined_history

    plot_training_history(h, save_path=os.path.join(MODEL_DIR, "training_history.png"))

    print("\nModel değerlendiriliyor (test seti)...")
    model.load_weights(MODEL_PATH)
    report = evaluate_model(
        model, X_test, y_test, save_path=os.path.join(MODEL_DIR, "evaluation.png")
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc_val = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)

    report_numeric = classification_report(y_test.astype(int), y_pred, output_dict=True)
    cpu_metrics["test"] = {
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

    with open(CPU_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(cpu_metrics, f, ensure_ascii=False, indent=2)
    print(f"CPU metrikleri kaydedildi: {CPU_METRICS_PATH}")

    print(f"\nModel kaydedildi: {MODEL_PATH}")
    print(f"Toplam eğitim süresi: {cpu_metrics['total_time']:.1f}s")
    return model


if __name__ == "__main__":
    train()
