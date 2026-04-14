import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "brain_tumor_dataset"
)


def load_images(data_dir=DATA_DIR, img_size=IMG_SIZE):
    images = []
    labels = []
    for label, folder in enumerate(["no", "yes"]):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Klasör bulunamadı: {folder_path}")
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


def preprocess(X):
    return X


def get_augmentation():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=10,
        fill_mode="nearest",
    )


def compute_class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(enumerate(weights))


def load_and_split(
    data_dir=DATA_DIR, img_size=IMG_SIZE, val_size=0.15, test_size=0.15, random_state=42
):
    X, y = load_images(data_dir, img_size)
    X = preprocess(X)

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

    class_weights = compute_class_weights(y_train)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(
        f"Sınıf dağılımı (train): no={int(sum(y_train == 0))}, yes={int(sum(y_train == 1))}"
    )
    print(f"Class weights: {class_weights}")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights
