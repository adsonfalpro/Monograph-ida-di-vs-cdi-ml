"""Engenharia de variáveis (Fase 4).

Mantém a lógica da monografia (12 variáveis, mesmas janelas) com quatro mudanças documentadas:

1. Nenhum ``fillna(0)``/``ffill``/``bfill``: o aquecimento vem das 59 linhas anteriores à amostra original e
   as janelas ``rolling``/``ewm`` ficam NaN até terem observações suficientes (``min_periods``).
2. RSI: quando não há perdas na janela de 14 dias o RSI é 100 (50 se também não há ganhos). No original virava NaN e
   depois 0 (``fillna(0)``), isto é, o oposto.
3. ``StandardScaler`` ajustado SOMENTE com linhas de treino (``fit_scaler``); nunca na base inteira.
4. Estatísticas de janela indefinidas (assimetria/curtose de série constante) valem 0 apenas nesse caso, e são contadas.

Todas as variáveis em t usam apenas informação até t (testado em tests/test_features.py).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "ida_ret", "duration", "ipca_deb_ret", "ibov_ret", "cdi_ret", "excesso_dia",
    "vol_20", "trend_20", "ret_20", "rsi_14", "macd", "macd_signal",
]
# Nome na monografia -> nome aqui (para rastreabilidade)
ORIGINAL_NAMES = {
    "IDA_DI": "ida_ret", "Duration": "duration", "IPCA_Deb": "ipca_deb_ret", "BOVA": "ibov_ret",
    "CDI_dia": "cdi_ret", "Excesso_dia": "excesso_dia", "vol_20": "vol_20", "trend_20": "trend_20",
    "ret_20": "ret_20", "rsi_14": "rsi_14", "macd": "macd", "macd_signal": "macd_signal",
}
AGG_NAMES = ["mean", "std", "min", "max", "median", "skew", "kurt", "p25", "p75", "last"]


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI com média simples (como no original), aplicado à variação diária de ``series``."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    valid = gain.notna() & loss.notna()
    out[valid & (loss == 0) & (gain > 0)] = 100.0
    out[valid & (loss == 0) & (gain == 0)] = 50.0
    return out


def build_features(clean: pd.DataFrame) -> pd.DataFrame:
    """12 variáveis brutas (sem padronizar), alinhadas às linhas de ``clean``."""
    ida = clean["ida_ret"]
    f = pd.DataFrame({"date": clean["date"].values})
    f["ida_ret"] = ida.values
    f["duration"] = clean["duration"].values
    f["ipca_deb_ret"] = clean["ipca_deb_ret"].values
    f["ibov_ret"] = clean["ibov_ret"].values
    f["cdi_ret"] = clean["cdi_ret"].values
    # Mantida a forma simples da monografia (IDA − CDI); difere da geométrica em < 4e-6.
    f["excesso_dia"] = (clean["ida_ret"] - clean["cdi_ret"]).values
    exc = f["excesso_dia"]
    f["vol_20"] = exc.rolling(20).std()
    f["trend_20"] = ida.reset_index(drop=True).rolling(20).mean()
    f["ret_20"] = exc.rolling(20).sum()
    f["rsi_14"] = rsi(ida.reset_index(drop=True), 14)
    ida_r = ida.reset_index(drop=True)
    ema12 = ida_r.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = ida_r.ewm(span=26, adjust=False, min_periods=26).mean()
    f["macd"] = ema12 - ema26
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    return f


def first_valid_row(features: pd.DataFrame) -> int:
    """Primeira linha em que todas as 12 variáveis existem."""
    return int(features[FEATURE_COLUMNS].notna().all(axis=1).values.argmax())


def fit_scaler(features: pd.DataFrame, end_row: int, start_row: int | None = None) -> StandardScaler:
    """Ajusta o StandardScaler com as linhas ``start_row..end_row`` (inclusive). Nada além de ``end_row`` é lido."""
    start = first_valid_row(features) if start_row is None else start_row
    block = features.loc[start:end_row, FEATURE_COLUMNS]
    if block.isna().any().any():
        raise ValueError("Linhas de treino com valores ausentes")
    return StandardScaler().fit(block.values)


def transform(features: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Aplica o scaler já ajustado (NaN de aquecimento permanecem NaN)."""
    return scaler.transform(features[FEATURE_COLUMNS].values)


def make_windows(x: np.ndarray, decision_rows: np.ndarray, lookback: int = 100) -> np.ndarray:
    """Janela de ``lookback`` linhas terminando na linha de decisão t (inclusive): linhas t-lookback+1 .. t."""
    decision_rows = np.asarray(decision_rows)
    if decision_rows.min() < lookback - 1:
        raise ValueError("decision_rows antes do fim da primeira janela completa")
    return np.stack([x[t - lookback + 1:t + 1] for t in decision_rows])


def aggregate_windows(w: np.ndarray) -> tuple[np.ndarray, int]:
    """(n, lookback, 12) -> (n, 120) com mean, std, min, max, mediana, assimetria, curtose, P25, P75, último.

    Retorna também quantas estatísticas indefinidas (janela constante) foram trocadas por 0.
    """
    std = w.std(axis=1)
    const = std < 1e-10  # janela (quase) constante: assimetria/curtose indefinidas (e numericamente instáveis)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = skew(w, axis=1, bias=False)
        ku = kurtosis(w, axis=1, fisher=True, bias=False)
    undefined = int((const | np.isnan(sk)).sum() + (const | np.isnan(ku)).sum())
    sk = np.where(const | np.isnan(sk), 0.0, sk)
    ku = np.where(const | np.isnan(ku), 0.0, ku)
    agg = np.concatenate([
        w.mean(axis=1), w.std(axis=1), w.min(axis=1), w.max(axis=1), np.median(w, axis=1),
        sk, ku, np.percentile(w, 25, axis=1), np.percentile(w, 75, axis=1), w[:, -1, :],
    ], axis=1)
    if not np.isfinite(agg).all():
        raise ValueError("Agregação gerou valores não finitos")
    return agg, undefined


def aggregated_names() -> list[str]:
    return [f"{feat}__{a}" for a in AGG_NAMES for feat in FEATURE_COLUMNS]
