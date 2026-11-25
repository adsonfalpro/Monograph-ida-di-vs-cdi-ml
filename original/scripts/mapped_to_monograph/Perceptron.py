# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve
)

# ==========
# Configuração
# ==========
INPUT_PATH = r"IDA_DI.xlsx"  # caminho anonimizado (era um caminho local do autor); IDA_DI.xlsx equivale a data/raw/Monografia - Dados.xlsx
LOOKBACK = 100
HORIZON = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==========
# Carregar dados
# ==========
df = pd.read_excel(INPUT_PATH, engine="openpyxl")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("%", "")

# Criar variáveis alvo
if "Excesso_dia" not in df.columns:
    df["Excesso_dia"] = df["IDA_DI"] - df["CDI_dia"]
if "Excesso_Retorno" not in df.columns:
    df["Excesso_Retorno"] = df["IDA_10_dias"] - df["CDI_10_dias"]

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
# Seleção de features
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
# Implementação do Perceptron do zero
# ==========
class PerceptronScratch:
    def __init__(self, learning_rate=0.01, n_epochs=50, random_state=RANDOM_STATE):
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            for i in range(n_samples):
                linear_output = np.dot(X_shuf[i], self.weights) + self.bias
                y_pred = self.activation(linear_output)
                update = self.learning_rate * (y_shuf[i] - y_pred)
                self.weights += update * X_shuf[i]
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

# ==========
# Treinar e avaliar Perceptron
# ==========
perc = PerceptronScratch(learning_rate=0.01, n_epochs=100)
perc.fit(X_train, y_train)
perc_pred = perc.predict(X_test)

f1_perc = f1_score(y_test, perc_pred, zero_division=0)
prec_perc = precision_score(y_test, perc_pred, zero_division=0)
rec_perc = recall_score(y_test, perc_pred, zero_division=0)
cm_perc = confusion_matrix(y_test, perc_pred)

print("\nResultados Perceptron (do zero) 10 dias:")
print(f"F1: {f1_perc:.3f} | Precisão: {prec_perc:.3f} | Recall: {rec_perc:.3f}")
print("Matriz de Confusão (Perceptron):")
print(cm_perc)

# ==========
# Criar pasta de resultados
# ==========
PERCEPTRON_DIR = "Perceptron_10d_resultados"
os.makedirs(PERCEPTRON_DIR, exist_ok=True)

# ==========
# Gráfico 1: Matriz de Confusão
# ==========
plt.figure(figsize=(6, 5))
sns.heatmap(cm_perc, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Não Drawdown", "Drawdown"],
            yticklabels=["Não Drawdown", "Drawdown"])
plt.title("Matriz de Confusão - Perceptron")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(PERCEPTRON_DIR, "matriz_confusao_perceptron.png"))
plt.close()

# ==========
# Gráfico 2: Matriz de Confusão Normalizada
# ==========
cm_norm = cm_perc.astype('float') / cm_perc.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=["Não Drawdown", "Drawdown"],
            yticklabels=["Não Drawdown", "Drawdown"])
plt.title("Matriz de Confusão Normalizada - Perceptron")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(PERCEPTRON_DIR, "matriz_confusao_norm_perceptron.png"))
plt.close()

# ==========
# Gráfico 3: Curva de Precisão vs Recall
# ==========
prec_curve, rec_curve, _ = precision_recall_curve(y_test, perc_pred)
plt.figure(figsize=(6, 5))
plt.plot(rec_curve, prec_curve, color="darkorange", lw=2)
plt.title("Curva Precisão vs Recall - Perceptron")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PERCEPTRON_DIR, "precision_recall_perceptron.png"))
plt.close()

# ==========
# Gráfico 4: Real vs Predito
# ==========
plt.figure(figsize=(10, 4))
plt.plot(y_test[:200], label="Real", color="blue", lw=2)
plt.plot(perc_pred[:200], label="Predito", color="orange", lw=2)
plt.title("Real vs Predito - Perceptron (200 amostras)")
plt.xlabel("Índice da Amostra")
plt.ylabel("Classe")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PERCEPTRON_DIR, "real_vs_predito_perceptron.png"))
plt.close()

print(f"✅ Gráficos do Perceptron salvos em: {PERCEPTRON_DIR}")
