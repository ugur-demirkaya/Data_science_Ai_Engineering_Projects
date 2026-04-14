import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    BatchNormalization,
    Dropout,
    Input,
)
from tensorflow.keras.models import Model

IMG_SIZE = 224


def build_model(img_size=IMG_SIZE, freeze_base=True):
    inputs = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_tensor=inputs
    )

    if freeze_base:
        base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers=20):
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True
    return model


def compile_model(model, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
