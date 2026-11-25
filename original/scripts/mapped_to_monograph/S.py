import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, confusion_matrix
from tabulate import tabulate

# 1. Carregar dados
df = pd.read_excel("IDA_DI.xlsx", engine="openpyxl")
df["drawdown"] = (df["Excesso_Retorno"] < 0).astype(int)

# 2. Feature engineering
df["vol_20"] = df["Excesso_dia"].rolling(20).std()
df["trend_20"] = df["IDA_DI"].rolling(20).mean()
df["ret_20"] = df["Excesso_dia"].rolling(20).sum()

delta = df["IDA_DI"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df["rsi_14"] = 100 - (100 / (1 + rs))

ema12 = df["IDA_DI"].ewm(span=12, adjust=False).mean()
ema26 = df["IDA_DI"].ewm(span=26, adjust=False).mean()
df["macd"] = ema12 - ema26
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

# 3. Selecionar features
features = ["IDA_DI","Duration","IPCA_Deb","IBOVA","CDI","Excesso_dia",
            "vol_20","trend_20","ret_20","rsi_14","macd","macd_signal"]

X = df[features].fillna(0)
y = df["drawdown"]

# 4. Divisão temporal
n = len(X)
train_end = int(n*0.8)
val_end = int(n*0.9)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# 5. Treinar modelo RF
rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# 6. Função para buscar threshold com recall mínimo
def find_threshold_recall(model, X_val, y_val, min_recall=0.80):
    probs = model.predict_proba(X_val)[:,1]
    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh, best_prec = 0.5, 0
    for t in thresholds:
        preds_bin = (probs > t).astype(int)
        rec = recall_score(y_val, preds_bin, zero_division=0)
        prec = precision_score(y_val, preds_bin, zero_division=0)
        if rec >= min_recall and prec > best_prec:
            best_prec = prec
            best_thresh = t
    return best_thresh

# 7. Buscar threshold com recall mínimo
thresh_rf = find_threshold_recall(rf, X_val, y_val, min_recall=0.80)
print(f"🔍 Threshold com recall mínimo 0.80: {thresh_rf:.2f}")

# 8. Avaliação no teste
test_probs = rf.predict_proba(X_test)[:,1]
test_preds = (test_probs > thresh_rf).astype(int)

f1 = f1_score(y_test, test_preds, zero_division=0)
prec = precision_score(y_test, test_preds, zero_division=0)
rec = recall_score(y_test, test_preds, zero_division=0)
auc = roc_auc_score(y_test, test_probs)
brier = brier_score_loss(y_test, test_probs)

print("\n=== Resultados RF com threshold orientado a recall mínimo ===")
print(f"F1: {f1:.4f}, Precisão: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc:.4f}, Brier: {brier:.4f}")

# 9. Matriz de confusão
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativo","Positivo"], yticklabels=["Negativo","Positivo"])
plt.title("Matriz de Confusão - RF (Recall mínimo)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# 10. Relatório final
resultados = [{
    "Modelo": "Random Forest (Recall mínimo)",
    "Threshold": round(thresh_rf,2),
    "F1": f1,
    "Precisão": prec,
    "Recall": rec,
    "AUC": auc,
    "Brier": brier
}]
print("\n=== RELATÓRIO FINAL ===")
print(tabulate(resultados, headers="keys", tablefmt="fancy_grid", floatfmt=".4f"))
