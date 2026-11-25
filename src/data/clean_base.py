"""Construção da base limpa (Fase 2).

Base primária: ``data/raw/Dados-IDA.xls`` (fonte bruta, 4240 linhas, 2009-01-05 a 2025-11-19).
``Monografia - Dados.xlsx`` é uma versão derivada (4181 linhas) usada só para reconciliação.

Convenção única: todos os retornos em **decimal** (0,01 = 1 %). Nenhum preenchimento de ausentes.
Colunas ``*_trail_*`` são retornos acumulados em janela RETROATIVA terminando em t (inclusive).
Elas NÃO são alvos: o alvo (retorno dos 10 dias seguintes) é construído na Fase 3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_PRIMARY = RAW_DIR / "Dados-IDA.xls"
RAW_MONOGRAPH = RAW_DIR / "Monografia - Dados.xlsx"

HORIZONS = (10, 20, 30, 60)
ORIGINAL_SAMPLE_START = pd.Timestamp("2009-03-31")  # primeira data da base da monografia

RAW_COLUMNS = [
    "date", "ida_index", "ida_ret", "duration", "ipca_deb_pct", "ibov_raw", "cdi_ret",
    "pct_cdi_raw", "excess_raw", "ida_60_raw", "cdi_60_raw", "pct_cdi_60_raw", "excess_60_raw",
]
MONO_COLUMNS = [
    "date", "ida_index", "ida_ret", "duration", "ipca_deb_pct", "ibov_ret", "cdi_ret",
    "excess_1", "ida_60", "cdi_60", "excess_60", "ida_30", "cdi_30", "excess_30",
    "ida_20", "cdi_20", "excess_20", "ida_10", "cdi_10", "excess_10",
]


def _parse_ibov(value) -> tuple[float, bool]:
    """BOVA vem como número, mas em 18 linhas de 2009 vem como texto ('-2,69%')."""
    if isinstance(value, str):
        text = value.strip().replace("%", "").replace(".", "").replace(",", ".")
        return float(text) / 100.0, True
    return float(value), False


def trailing_compound(returns: pd.Series, window: int) -> pd.Series:
    """Retorno composto de ``window`` dias terminando em t (inclui o retorno de t)."""
    return (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0


def load_raw(path: Path = RAW_PRIMARY) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"Esperadas {len(RAW_COLUMNS)} colunas, encontradas {df.shape[1]}")
    df.columns = RAW_COLUMNS
    return df


def load_monograph(path: Path = RAW_MONOGRAPH) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.shape[1] != len(MONO_COLUMNS):
        raise ValueError(f"Esperadas {len(MONO_COLUMNS)} colunas, encontradas {df.shape[1]}")
    df.columns = MONO_COLUMNS
    return df


def build_clean_base(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    raw = load_raw() if raw is None else raw
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    parsed = raw["ibov_raw"].map(_parse_ibov)
    out = pd.DataFrame({"date": raw["date"]})
    out["ida_index"] = raw["ida_index"].astype(float)
    out["ida_ret"] = raw["ida_ret"].astype(float)
    out["duration"] = raw["duration"].astype(float)            # NaN nas 59 primeiras linhas (não preencher)
    out["ipca_deb_ret"] = raw["ipca_deb_pct"].astype(float) / 100.0  # fonte em %, convertido p/ decimal
    out["ibov_ret"] = [p[0] for p in parsed]
    out["cdi_ret"] = raw["cdi_ret"].astype(float)
    out["excess_ret"] = (1.0 + out["ida_ret"]) / (1.0 + out["cdi_ret"]) - 1.0  # excesso geométrico

    # Retorno do IDA implícito no nível do índice (usado só para verificação e backtest alternativo).
    out["ida_index_ret"] = out["ida_index"].pct_change()

    for h in HORIZONS:
        out[f"ida_trail_{h}"] = trailing_compound(out["ida_ret"], h)
        out[f"cdi_trail_{h}"] = trailing_compound(out["cdi_ret"], h)
        out[f"excess_trail_{h}"] = (1.0 + out[f"ida_trail_{h}"]) / (1.0 + out[f"cdi_trail_{h}"]) - 1.0

    # Flags de rastreabilidade (nenhum dado é removido ou alterado por causa delas).
    out["in_original_sample"] = out["date"] >= ORIGINAL_SAMPLE_START
    out["has_duration"] = out["duration"].notna()
    out["ibov_from_text"] = [p[1] for p in parsed]
    out["ibov_zero_no_trading"] = out["ibov_ret"] == 0.0
    out["ida_ret_index_mismatch"] = (out["ida_ret"] - out["ida_index_ret"]).abs() > 1e-4
    return out


def save_clean_base(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = PROCESSED_DIR / "ida_clean.csv" if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


if __name__ == "__main__":
    base = build_clean_base()
    print(save_clean_base(base), base.shape)
