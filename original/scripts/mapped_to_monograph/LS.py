# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss,
    confusion_matrix, precision_recall_curve, roc_curve
)

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# ==========
# Configuração
# ==========
INPUT_PATH = r"IDA_DI.xlsx"  # caminho anonimizado (era um caminho local do autor); IDA_DI.xlsx equivale a data/raw/Monografia - Dados.xlsx
RESULTS_DIR = "Results_modelo"
OUT_DIR = os.path.join(RESULTS_DIR, "RF_10d_otimizado")
os.makedirs(OUT_DIR, exist_ok=True)

LOOKBACK = 100
HORIZON = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==========
# Carregar dados e preservar base original
# ==========
df = pd.read_excel(INPUT_PATH, engine="openpyxl")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("%", "")
df_raw = df.copy()  # preserva base original

# Calcular CDI acumulado corretamente
df_raw["CDI_dia_decimal"] = df_raw["CDI_dia"] / 100.0
df_raw["CDI_acumulado"] = (1 + df_raw["CDI_dia_decimal"]).cumprod()

# Criar variáveis se não existirem
if "Excesso_dia" not in df.columns:
    df["Excesso_dia"] = df["IDA_DI"] - df["CDI_dia"]
if "Excesso_Retorno" not in df.columns:
    df["Excesso_Retorno"] = df["IDA_10_dias"] - df["CDI_10_dias"]

# Target: drawdown em 10 dias
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

# ==========
# Seleção de features e normalização separada
# ==========
features = ["IDA_DI", "Duration", "IPCA_Deb", "BOVA", "CDI_dia", "Excesso_dia",
            "vol_20", "trend_20", "ret_20", "rsi_14", "macd", "macd_signal"]
features = [f for f in features if f in df.columns]

features_df = df[features].copy().fillna(0)
scaler = StandardScaler()
features_df_scaled = pd.DataFrame(
    scaler.fit_transform(features_df),
    index=features_df.index,
    columns=features_df.columns
)

# ==========
# Funções de sequência e agregação
# ==========
def create_sequences(data, target, lookback=LOOKBACK, horizon=HORIZON):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        window = data.iloc[i:i+lookback].values
        target_y = target.iloc[i+lookback:i+lookback+horizon].max()
        X.append(window)
        y.append(target_y)
    return np.array(X), np.array(y)

def aggregate_sequences(X_seq):
    X_mean = X_seq.mean(axis=1)
    X_std = X_seq.std(axis=1)
    X_min = X_seq.min(axis=1)
    X_max = X_seq.max(axis=1)
    X_med = np.median(X_seq, axis=1)
    X_skew = skew(X_seq, axis=1, bias=False, nan_policy='omit')
    X_kurt = kurtosis(X_seq, axis=1, fisher=True, bias=False, nan_policy='omit')
    X_last = X_seq[:, -1, :]
    X_p25 = np.percentile(X_seq, 25, axis=1)
    X_p75 = np.percentile(X_seq, 75, axis=1)
    return np.concatenate(
        [X_mean, X_std, X_min, X_max, X_med, X_skew, X_kurt, X_p25, X_p75, X_last],
        axis=1
    )

# ==========
# Preparar dados
# ==========
X_seq, y = create_sequences(features_df_scaled, df["drawdown"], lookback=LOOKBACK, horizon=HORIZON)
X = aggregate_sequences(X_seq)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

n = len(X)
train_end = int(n*0.6)
val_end = int(n*0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# ==========
# Modelo RF + calibração
# ==========
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    class_weight="balanced_subsample",
    n_jobs=-1
)
rf.fit(X_train, y_train)

calibrated = CalibratedClassifierCV(rf, method="isotonic", cv="prefit")
calibrated.fit(X_val, y_val)

# ==========
# Threshold ótimo pelo F1-score
# ==========
val_probs = calibrated.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1_scores)
best_thresh = thr[best_idx]
best_f1 = f1_scores[best_idx]

print(f"🔍 Threshold ótimo (F1): {best_thresh:.3f} | F1 validação: {best_f1:.3f}")

# ==========
# Avaliação no teste
# ==========
test_probs = calibrated.predict_proba(X_test)[:, 1]
test_pred = (test_probs > best_thresh).astype(int)

f1 = f1_score(y_test, test_pred, zero_division=0)
prec_test = precision_score(y_test, test_pred, zero_division=0)
rec_test = recall_score(y_test, test_pred, zero_division=0)
auc = roc_auc_score(y_test, test_probs) if len(np.unique(y_test)) > 1 else float("nan")
brier = brier_score_loss(y_test, test_probs)

print("\nResultados RF 10 dias (otimizado):")
print(f"F1: {f1:.3f} | Precisão: {prec_test:.3f} | Recall: {rec_test:.3f} | AUC: {auc:.3f} | Brier: {brier:.3f}")

# ==========
# Alertas alinhados à base original
# ==========
decision_idx_all = np.arange(LOOKBACK, LOOKBACK + len(X))
probs_all = calibrated.predict_proba(X)[:, 1]
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
# Gráfico com alertas sobre o índice
# ==========
nivel_col = None
for cand in ["Número_índice", "Numero_indice", "Número_indice", "IDA_DI"]:
    if cand in df_raw.columns:
        nivel_col = cand

