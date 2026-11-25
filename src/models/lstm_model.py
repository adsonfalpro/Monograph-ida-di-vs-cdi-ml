"""LSTM (arquitetura e treino do Apêndice B / LST.py), reescrita para Keras 3 e com sementes controladas.

LSTM(64, seq) -> Dropout(0,3) -> LSTM(32) -> BatchNorm -> Dense(32, relu) -> Dropout(0,2) -> Dense(1, sigmoid)
Adam 1e-3, binary_crossentropy, até 50 épocas, batch 32, class_weight balanceado (calculado no ``fit``),
EarlyStopping(val_loss, paciência 5, restaura melhores pesos), ReduceLROnPlateau(0,5; paciência 3).
A calibração (Platt) e o threshold são da Fase 7.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

LSTM_PARAMS = dict(units1=64, dropout1=0.3, units2=32, dense=32, dropout2=0.2, learning_rate=1e-3,
                   epochs=50, batch_size=32, es_patience=5, lr_factor=0.5, lr_patience=3)


def build_lstm(lookback: int, n_features: int, p: dict | None = None):
    import keras
    from keras import layers
    p = {**LSTM_PARAMS, **(p or {})}
    model = keras.Sequential([
        keras.Input(shape=(lookback, n_features)),
        layers.LSTM(p["units1"], return_sequences=True),
        layers.Dropout(p["dropout1"]),
        layers.LSTM(p["units2"], return_sequences=False),
        layers.BatchNormalization(),
        layers.Dense(p["dense"], activation="relu"),
        layers.Dropout(p["dropout2"]),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=p["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(x_fit, y_fit, x_val, y_val, seed: int, p: dict | None = None, verbose: int = 0):
    """Treina com a semente dada. Devolve (modelo, histórico, class_weight)."""
    import keras
    import tensorflow as tf
    p = {**LSTM_PARAMS, **(p or {})}
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    model = build_lstm(x_fit.shape[1], x_fit.shape[2], p)
    classes = np.unique(y_fit)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_fit)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=p["es_patience"], restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=p["lr_factor"], patience=p["lr_patience"]),
    ]
    hist = model.fit(x_fit, y_fit, epochs=p["epochs"], batch_size=p["batch_size"], validation_data=(x_val, y_val),
                     callbacks=callbacks, class_weight=class_weight, shuffle=True, verbose=verbose)
    return model, hist, class_weight
