# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss,
    confusion_matrix, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning (Keras/TensorFlow)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# ==========
# Configuração
# ==========
INPUT_PATH = r"IDA_DI.xlsx"  # caminho anonimizado (era um caminho local do autor); IDA_DI.xlsx equivale a data/raw/Monografia - Dados.xlsx
RESULTS_DIR = "Results_modelo"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

LOOKBACK = 100
HORIZONS = [10, 20, 30, 60]

# custos relativos: FN (não prever underperformance) é mais caro
FN_COST = 5
FP_COST = 1

# ==================
# Funções auxiliares
# ==================
def safe_pct_to_float(s):
    if s is None:
        return np.nan
    if isinstance(s, (int, float)):
        return s
    s = str(s).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return np.nan
    had_pct = "%" in s
    s_clean = s.replace("%", "")
    s_clean = s_clean.replace(",", ".")
    try:
        val = float(s_clean)
    except:
        return np.nan
    return val / 100.0 if had_pct else val

def create_technical_features(df, ida_col="IDA DI"):
    # Retorno base do IDA via Número Índice (se disponível)
    if "Número Índice" in df.columns:
        df["ret_ida_idx"] = pd.to_numeric(df["Número Índice"], errors="coerce").pct_change()
        base_ret = df["ret_ida_idx"].fillna(0)
        price = pd.to_numeric(df["Número Índice"], errors="coerce")
    else:
        base_ret = pd.to_numeric(df[ida_col], errors="coerce").fillna(0)
        price = pd.to_numeric(df[ida_col], errors="coerce")

    # Volatilidade e tendência
    df["vol_20"] = base_ret.rolling(20).std()
    df["trend_20"] = price.rolling(20).mean()
    df["ret_20"] = base_ret.rolling(20).sum()

    # RSI 14
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df

def build_targets(df):
    mapping = {
        10: ("IDA 10_dias", "CDI_10_dias"),
        20: ("IDA 20_dias", "CDI_20_dias"),
        30: ("IDA 30_dias", "CDI_30_dias"),
        60: ("IDA 60_dias", "CDI_60_dias"),
    }
    targets = {}
    for h, (ida_col, cdi_col) in mapping.items():
        ida = pd.to_numeric(df.get(ida_col), errors="coerce")
        cdi = pd.to_numeric(df.get(cdi_col), errors="coerce")
        targets[h] = (ida < cdi).astype(int).values
    return targets

def create_sequences(data_2d, target_1d, lookback=LOOKBACK, horizon=10):
    X, y = [], []
    for i in range(len(data_2d) - lookback - horizon):
        seq_x = data_2d.iloc[i:i+lookback].values
        # rótulo em t+horizon (binário: ida acumulado < cdi acumulado)
        y.append(target_1d[i+lookback+horizon-1])
        X.append(seq_x)
    return np.array(X), np.array(y)

def aggregate_sequences_for_rf(X_seq):
    X_mean = X_seq.mean(axis=1)
    X_std = X_seq.std(axis=1)
    X_last = X_seq[:, -1, :]
    return np.concatenate([X_mean, X_std, X_last], axis=1)

def temporal_split(X, y, train_ratio=0.8, val_ratio=0.1):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])

def best_threshold_cost(y_val, proba_val, fn_cost=FN_COST, fp_cost=FP_COST):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        preds = (proba_val > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_t = t
    return best_t, best_cost

def evaluate_and_save(y_test, proba_test, preds_bin, model_name, horizon, out_dir):
    f1 = f1_score(y_test, preds_bin, zero_division=0)
    prec = precision_score(y_test, preds_bin, zero_division=0)
    rec = recall_score(y_test, preds_bin, zero_division=0)
    auc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else np.nan
    brier = brier_score_loss(y_test, proba_test)

    # Matriz de Confusão
    cm = confusion_matrix(y_test, preds_bin)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Não < CDI","< CDI"], yticklabels=["Não < CDI","< CDI"])
    plt.title(f"Matriz de Confusão - {model_name} {horizon}d")
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusao_{model_name}_{horizon}d.png"), dpi=120)
    plt.close()

    # Matriz Normalizada
    cmn = confusion_matrix(y_test, preds_bin, normalize="true")
    plt.figure(figsize=(6,5))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=["Não < CDI","< CDI"], yticklabels=["Não < CDI","< CDI"])
    plt.title(f"Matriz Normalizada - {model_name} {horizon}d")
    plt.xlabel("Predito"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusao_norm_{model_name}_{horizon}d.png"), dpi=120)
    plt.close()

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={auc:.2f}")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC - {model_name} {horizon}d")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_{model_name}_{horizon}d.png"), dpi=120)
        plt.close()
    except Exception:
        pass

    # Precision-Recall
    prec_curve, rec_curve, thr_pr = precision_recall_curve(y_test, proba_test)
    plt.figure(figsize=(6,5))
    plt.plot(rec_curve, prec_curve)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall - {model_name} {horizon}d")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pr_{model_name}_{horizon}d.png"), dpi=120)
    plt.close()

    # Calibração
    try:
        prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10, strategy="uniform")
        plt.figure(figsize=(6,5))
        plt.plot(prob_pred, prob_true, marker="o", label="Modelo")
        plt.plot([0,1],[0,1],"--",color="gray", label="Perfeito")
        plt.xlabel("Probabilidade prevista"); plt.ylabel("Frequência observada")
        plt.title(f"Calibração - {model_name} {horizon}d")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"calibracao_{model_name}_{horizon}d.png"), dpi=120)
        plt.close()
    except Exception:
        pass

    return {
        "Modelo": model_name,
        "Horizonte (Dias)": horizon,
        "F1-Score": f1,
        "Precisão": prec,
        "Recall": rec,
        "AUC": auc,
        "Brier": brier
    }

# ===============
# Carregar e limpar
# ===============
df_raw = pd.read_excel(INPUT_PATH, engine="openpyxl")
df_raw.columns = [c.strip() for c in df_raw.columns]

percent_cols = [
    "IDA DI", "BOVA", "CDI_dia", "Diferença CDIxIDA_Dia",
    "IDA 60_dias", "CDI_60_dias", "Diferença CDIxIDA_60Dias",
    "IDA 30_dias", "CDI_30_dias", "Diferença CDIxIDA_30Dias",
    "IDA 20_dias", "CDI_20_dias", "Diferença CDIxIDA_20Dias",
    "IDA 10_dias", "CDI_10_dias", "Diferença CDIxIDA_10Dias"
]
for c in percent_cols:
    if c in df_raw.columns:
        df_raw[c] = df_raw[c].apply(safe_pct_to_float)

for c in ["Duration", "IPCA Deb", "Número Índice"]:
    if c in df_raw.columns:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

if "Data de Referência" in df_raw.columns:
    df_raw["Data de Referência"] = pd.to_datetime(df_raw["Data de Referência"], dayfirst=True, errors="coerce")
    df_raw = df_raw.sort_values("Data de Referência").reset_index(drop=True)

df = create_technical_features(df_raw.copy(), ida_col="IDA DI")

feature_cols = [col for col in [
    "Número Índice", "IDA DI", "Duration", "IPCA Deb", "BOVA", "CDI_dia",
    "Diferença CDIxIDA_Dia", "vol_20", "trend_20", "ret_20", "rsi_14", "macd", "macd_signal"
] if col in df.columns]

# tratar NaNs e escalar
df_features = df[feature_cols].copy()
df_features = df_features.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

scaler = StandardScaler()
df_features_scaled = pd.DataFrame(
    scaler.fit_transform(df_features.values),
    columns=feature_cols,
    index=df_features.index
)

targets = build_targets(df)

# ============================
# Treino e avaliação dos modelos
# ============================
result_rows = []

for horizon in HORIZONS:
    y = targets[horizon]
    X_seq, y_seq = create_sequences(df_features_scaled, y, lookback=LOOKBACK, horizon=horizon)

    # split temporal
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X_seq, y_seq, train_ratio=0.8, val_ratio=0.1)

    # -------------
    # Random Forest
    # -------------
    Xrf_train = aggregate_sequences_for_rf(X_train)
    Xrf_val = aggregate_sequences_for_rf(X_val)
    Xrf_test = aggregate_sequences_for_rf(X_test)

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(Xrf_train, y_train)
    rf_val_proba = rf.predict_proba(Xrf_val)[:, 1]
    rf_thresh, rf_val_cost = best_threshold_cost(y_val, rf_val_proba, fn_cost=FN_COST, fp_cost=FP_COST)
    rf_test_proba = rf.predict_proba(Xrf_test)[:, 1]
    rf_test_pred = (rf_test_proba > rf_thresh).astype(int)

    out_dir_rf = os.path.join(RESULTS_DIR, f"RF_{horizon}d")
    os.makedirs(out_dir_rf, exist_ok=True)
    rf_metrics = evaluate_and_save(y_test, rf_test_proba, rf_test_pred, "RF", horizon, out_dir_rf)
    rf_metrics["Threshold Ótimo"] = rf_thresh
    rf_metrics["Custo (validação)"] = rf_val_cost
    result_rows.append(rf_metrics)

    # ------
    # RNR
    # ------
    rnr = Sequential([
        SimpleRNN(64, activation="tanh", input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    rnr.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=["AUC"])

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    rnr.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=100, batch_size=64, callbacks=[es], verbose=0)

    rnr_val_proba = rnr.predict(X_val, verbose=0).ravel()
    rnr_thresh, rnr_val_cost = best_threshold_cost(y_val, rnr_val_proba, fn_cost=FN_COST, fp_cost=FP_COST)
    rnr_test_proba = rnr.predict(X_test, verbose=0).ravel()
    rnr_test_pred = (rnr_test_proba > rnr_thresh).astype(int)

    out_dir_rnr = os.path.join(RESULTS_DIR, f"RNR_{horizon}d")
    os.makedirs(out_dir_rnr, exist_ok=True)
    rnr_metrics = evaluate_and_save(y_test, rnr_test_proba, rnr_test_pred, "RNR", horizon, out_dir_rnr)
    rnr_metrics["Threshold Ótimo"] = rnr_thresh
    rnr_metrics["Custo (validação)"] = rnr_val_cost
    result_rows.append(rnr_metrics)

    # ------
    # LSTM
    # ------
    lstm = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                 loss="binary_crossentropy",
                 metrics=["AUC"])

    es2 = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lstm.fit(X_train, y_train, validation_data=(X_val, y_val),
             epochs=100, batch_size=64, callbacks=[es2], verbose=0)

    lstm_val_proba = lstm.predict(X_val, verbose=0).ravel()
    lstm_thresh, lstm_val_cost = best_threshold_cost(y_val, lstm_val_proba, fn_cost=FN_COST, fp_cost=FP_COST)
    lstm_test_proba = lstm.predict(X_test, verbose=0).ravel()
    lstm_test_pred = (lstm_test_proba > lstm_thresh).astype(int)

    out_dir_lstm = os.path.join(RESULTS_DIR, f"LSTM_{horizon}d")
    os.makedirs(out_dir_lstm, exist_ok=True)
    lstm_metrics = evaluate_and_save(y_test, lstm_test_proba, lstm_test_pred, "LSTM", horizon, out_dir_lstm)
    lstm_metrics["Threshold Ótimo"] = lstm_thresh
    lstm_metrics["Custo (validação)"] = lstm_val_cost
    result_rows.append(lstm_metrics)

# ===========================
# Consolidação e salvamento
# ===========================
results_df = pd.DataFrame(result_rows)
# ordenar colunas
cols_order = [
    "Modelo", "Horizonte (Dias)", "Threshold Ótimo", "Custo (validação)",
    "F1-Score", "Precisão", "Recall", "AUC", "Brier"
]
results_df = results_df[[c for c in cols_order if c in results_df.columns]]

results_csv = os.path.join(RESULTS_DIR, "resultados_consolidados.csv")
results_xlsx = os.path.join(RESULTS_DIR, "resultados_consolidados.xlsx")
results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
try:
    results_df.to_excel(results_xlsx, index=False)
except Exception:
    pass

print("Concluído. Resultados salvos em:", RESULTS_DIR)
