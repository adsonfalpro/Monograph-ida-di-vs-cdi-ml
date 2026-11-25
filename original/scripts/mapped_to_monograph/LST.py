# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Métricas e utilitários
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, confusion_matrix, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# ==========
# Configuração
# ==========
INPUT_PATH = r"IDA_DI.xlsx"  # caminho anonimizado (era um caminho local do autor); IDA_DI.xlsx equivale a data/raw/Monografia - Dados.xlsx
RESULTS_DIR = "Results_modelo"
OUT_DIR = os.path.join(RESULTS_DIR, "LSTM_10d_ajustada")
os.makedirs(OUT_DIR, exist_ok=True)

LOOKBACK = 100
HORIZON = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ==========
# Carregar dados
# ==========
df = pd.read_excel(INPUT_PATH, engine="openpyxl")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("%", "")
df_raw = df.copy()

# CDI acumulado
df_raw["CDI_dia_decimal"] = df_raw["CDI_dia"] / 100.0
df_raw["CDI_acumulado"] = (1 + df_raw["CDI_dia_decimal"]).cumprod()

# Variáveis derivadas
if "Excesso_dia" not in df.columns:
    df["Excesso_dia"] = df["IDA_DI"] - df["CDI_dia"]
if "Excesso_Retorno" not in df.columns:
    df["Excesso_Retorno"] = df["IDA_10_dias"] - df["CDI_10_dias"]

# Target binário
df["drawdown"] = (df["Excesso_Retorno"] < 0).astype(int)

# ==========
# Feature engineering
# ==========
df["vol_20"] = df["Excesso_dia"].rolling(20).std()
df["trend_20"] = df["IDA_DI"].rolling(20).mean()
df["ret_20"] = df["Excesso_dia"].rolling(20).sum()

delta = df["IDA_DI"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean().replace(0, np.nan)
rs = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

ema12 = df["IDA_DI"].ewm(span=12, adjust=False).mean()
ema26 = df["IDA_DI"].ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

features = ["IDA_DI", "Duration", "IPCA_Deb", "BOVA", "CDI_dia", "Excesso_dia",
            "vol_20", "trend_20", "ret_20", "rsi_14", "macd", "macd_signal"]
features = [f for f in features if f in df.columns]

features_df = df[features].copy().fillna(0)

# ==========
# Normalização (simples)
# ==========
scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features_df),
    index=features_df.index,
    columns=features_df.columns
)

# ==========
# Sequências
# ==========
def create_sequences(data, target, lookback=LOOKBACK, horizon=HORIZON):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        window = data.iloc[i:i+lookback].values
        target_y = target.iloc[i+lookback:i+lookback+horizon].max()
        X.append(window)
        y.append(target_y)
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled, df["drawdown"], lookback=LOOKBACK, horizon=HORIZON)

n = len(X)
train_end = int(n*0.6)
val_end = int(n*0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# ==========
# Class weights (balanceamento)
# ==========
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}

# ==========
# Modelo LSTM ajustado
# ==========
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, len(features))),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
]

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# ==========
# Calibração de probabilidades (Platt scaling)
# ==========
val_probs_raw = model.predict(X_val).flatten()
calibrator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
calibrator.fit(val_probs_raw.reshape(-1, 1), y_val)

# Probabilidades calibradas
val_probs = calibrator.predict_proba(val_probs_raw.reshape(-1, 1))[:, 1]

# ==========
# Threshold ótimo pelo F1-score (com probabilidades calibradas)
# ==========
prec_val, rec_val, thr_val = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * prec_val * rec_val / (prec_val + rec_val + 1e-9)
best_idx = np.argmax(f1_scores)
best_thresh = thr_val[best_idx]
best_f1 = f1_scores[best_idx]

print(f"🔍 Threshold ótimo (F1, calibrado): {best_thresh:.3f} | F1 validação: {best_f1:.3f}")

# ==========
# Avaliação no teste (com probabilidades calibradas)
# ==========
test_probs_raw = model.predict(X_test).flatten()
test_probs = calibrator.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]
test_pred = (test_probs > best_thresh).astype(int)

f1 = f1_score(y_test, test_pred, zero_division=0)
prec_test = precision_score(y_test, test_pred, zero_division=0)
rec_test = recall_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float("nan")
brier = brier_score_loss(y_test, test_probs)

print("\nResultados LSTM 10 dias (ajustada + calibrada):")
print(f"F1: {f1:.3f} | Precisão: {prec_test:.3f} | Recall: {rec_test:.3f} | AUC: {auc:.3f} | Brier: {brier:.3f}")

# ==========
# Alertas alinhados à base original (usando probabilidades calibradas)
# ==========
decision_idx_all = np.arange(LOOKBACK, LOOKBACK + len(X))
probs_all_raw = model.predict(X).flatten()
probs_all = calibrator.predict_proba(probs_all_raw.reshape(-1, 1))[:, 1]
signals_all = (probs_all > best_thresh).astype(int)

alerta_col = pd.Series(np.nan, index=df_raw.index)
alerta_col.iloc[decision_idx_all] = signals_all
df_raw["Alerta_Modelo"] = alerta_col

# ==========
# Exportar base completa com alertas
# ==========
alertas_path = os.path.join(OUT_DIR, "alertas_modelo_base_completa.xlsx")
df_raw.to_excel(alertas_path, index=False)
print(f"📄 Excel com a base completa + Alerta_Modelo salvo em: {alertas_path}")

# ==========
# Gráficos salvos em arquivos
# ==========

# Calibration Curve (teste)
prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label="Modelo (calibrado)")
plt.plot([0,1],[0,1], linestyle="--", color="gray", label="Perfeito")
plt.title("Calibration Curve - LSTM 10d (calibrada)")
plt.xlabel("Probabilidade prevista")
plt.ylabel("Frequência observada")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "calibration_curve.png"))
plt.close()

# Precision-Recall Curve (teste)
prec, rec, _ = precision_recall_curve(y_test, test_probs)
plt.figure(figsize=(6,6))
plt.plot(rec, prec, label="LSTM (calibrada)")
plt.title("Precision-Recall Curve - LSTM 10d")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "precision_recall_curve.png"))
plt.close()

# Real vs Predito (200 amostras)
plt.figure(figsize=(12,4))
plt.plot(y_test[:200], label="Real", color="blue")
plt.plot(test_pred[:200], label="Predito", color="orange")
plt.title("Real vs Predito - LSTM 10d (200 amostras)")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "real_vs_predito.png"))
plt.close()

# ROC Curve (teste)
fpr, tpr, _ = roc_curve(y_test, test_probs)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1], linestyle="--", color="gray")
plt.title("ROC Curve - LSTM 10d")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
plt.close()

# Matriz de Confusão (teste)
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Não < CDI","< CDI"],
            yticklabels=["Não < CDI","< CDI"])
plt.title(f"Matriz de Confusão - LSTM 10d (Threshold {best_thresh:.2f})")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.savefig(os.path.join(OUT_DIR, "matriz_confusao.png"))
plt.close()

# Tabela de métricas (salvar CSV)
metrics_table = pd.DataFrame({
    "Modelo":["LSTM_10d_ajustada_calibrada"],
    "Horizonte (Dias)":[HORIZON],
    "Threshold Ótimo":[best_thresh],
    "F1-Score":[f1],
    "Precisão":[prec_test],
    "Recall":[rec_test],
    "AUC":[auc],
    "Brier":[brier]
})
metrics_path = os.path.join(OUT_DIR, "tabela_metricas.csv")
metrics_table.to_csv(metrics_path, index=False)
print(f"📄 Tabela de métricas salva em: {metrics_path}")

# Comparação de desempenho (IDA, CDI, Modelo)
df_raw["Retorno_IDA_decimal"] = df_raw["IDA_DI"] / 100.0
df_raw["Acumulado_IDA"] = (1 + df_raw["Retorno_IDA_decimal"]).cumprod()

df_raw["Retorno_Modelo"] = np.where(
    df_raw["Alerta_Modelo"] == 1,
    df_raw["CDI_dia_decimal"],
    df_raw["Retorno_IDA_decimal"]
)
df_raw["Acumulado_Modelo"] = (1 + df_raw["Retorno_Modelo"]).cumprod()

plt.figure(figsize=(12,6))
plt.plot(df_raw.index, df_raw["CDI_acumulado"], label="CDI", color="blue")
plt.plot(df_raw.index, df_raw["Acumulado_IDA"], label="IDA", color="red")
plt.plot(df_raw.index, df_raw["Acumulado_Modelo"], label="Modelo", color="green")
plt.title("Comparação")
plt.xlabel("Data")
plt.ylabel("Valor acumulado")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "comparacao.png"))
plt.close()

print(f"✅ Gráficos e resultados salvos em: {OUT_DIR}")


