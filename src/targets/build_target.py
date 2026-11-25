"""Variável‑alvo (Fase 3).

Definição (data de decisão t = último dia observado):

    R_IDA(t, t+h) = prod_{i=1..h} (1 + r_IDA[t+i]) - 1
    R_CDI(t, t+h) = prod_{i=1..h} (1 + r_CDI[t+i]) - 1
    Y_t = 1  se R_IDA(t, t+h) <  R_CDI(t, t+h)      (empate => 0)

Só retornos de t+1 .. t+h entram no alvo. Nenhuma feature pode usar r[t+1..]. As últimas h linhas ficam
sem alvo (NaN); nunca são preenchidas.

``target_end_idx`` é a posição (linha) do último retorno usado pelo alvo; serve para purge na Fase 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 10


def forward_compound(returns: pd.Series, horizon: int) -> pd.Series:
    """prod(1+r[t+1..t+horizon]) - 1 na posição t."""
    trailing = (1.0 + returns).rolling(horizon).apply(np.prod, raw=True) - 1.0
    return trailing.shift(-horizon)


def build_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Devolve colunas: fwd_ida, fwd_cdi, fwd_excess, y, target_end_idx (índice posicional)."""
    if not df["date"].is_monotonic_increasing:
        raise ValueError("df precisa estar em ordem cronológica")
    out = pd.DataFrame({"date": df["date"].values})
    out["fwd_ida"] = forward_compound(df["ida_ret"].reset_index(drop=True), horizon)
    out["fwd_cdi"] = forward_compound(df["cdi_ret"].reset_index(drop=True), horizon)
    out["fwd_excess"] = (1.0 + out["fwd_ida"]) / (1.0 + out["fwd_cdi"]) - 1.0
    y = (out["fwd_ida"] < out["fwd_cdi"]).astype("float")
    y[out["fwd_ida"].isna() | out["fwd_cdi"].isna()] = np.nan
    out["y"] = y
    idx = np.arange(len(out)) + horizon
    out["target_end_idx"] = np.where(idx < len(out), idx, -1)
    return out


def original_target(df: pd.DataFrame, lookback: int = 100, horizon: int = HORIZON) -> pd.DataFrame:
    """Reproduz EXATAMENTE o alvo dos scripts LS.py/LST.py/Perceptron.py (apenas para comparação).

    drawdown[t] = 1[ida_trail_10[t] < cdi_trail_10[t]]      (janela retroativa terminando em t)
    y_i         = max(drawdown[i+lookback : i+lookback+horizon]),  janela de features = linhas i..i+lookback-1
    A data de decisão é a última linha da janela (i + lookback - 1).
    """
    dd = (df["ida_trail_10"] < df["cdi_trail_10"]).astype(int).reset_index(drop=True)
    n = len(df) - lookback - horizon
    rows = []
    for i in range(n):
        rows.append((i + lookback - 1, int(dd.iloc[i + lookback:i + lookback + horizon].max())))
    out = pd.DataFrame(rows, columns=["decision_idx", "y_original"])
    out["date"] = df["date"].reset_index(drop=True).iloc[out["decision_idx"]].values
    return out
