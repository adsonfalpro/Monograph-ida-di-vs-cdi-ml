"""Reproduz o backtest da monografia (alertas em todo o período) e separa trecho in-sample do trecho de teste.

Convenção do original: o alerta da linha k (janela terminando em k-1) vale para o retorno da linha k; retorno = CDI se alerta, senão IDA.
Usa os alertas salvos pelos scripts originais (copiados em original/results/) e os retornos diários da base da monografia.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "data"))
from backtest import equity, metrics  # noqa: E402
from clean_base import load_monograph  # noqa: E402

RES = ROOT / "original" / "results"
FILES = {"Random Forest": RES / "alertas_RF_original.xlsx", "LSTM": RES / "alertas_LSTM_original.xlsx"}
N_WIN = 4071
FIRST = 100                                  # primeira linha com alerta
TEST_START = int(N_WIN * 0.8) + 100          # primeira linha do teste original (3356)
VAL_START = int(N_WIN * 0.6) + 100


def main() -> None:
    mono = load_monograph()
    ida, cdi = mono["ida_ret"].values, mono["cdi_ret"].values
    out = []
    for name, f in FILES.items():
        df = pd.read_excel(f)
        alert = df["Alerta_Modelo"].values
        assert len(alert) == len(mono)
        a = alert[FIRST:FIRST + N_WIN].astype(int)
        for seg, (lo, hi) in {"período inteiro (como na monografia)": (FIRST, FIRST + N_WIN),
                              "treino (in-sample)": (FIRST, VAL_START), "validação (usada para limiar)": (VAL_START, TEST_START),
                              "teste original (fora da amostra)": (TEST_START, FIRST + N_WIN)}.items():
            al = alert[lo:hi].astype(int)
            r = np.where(al == 1, cdi[lo:hi], ida[lo:hi])
            m = metrics(r, cdi[lo:hi], ida[lo:hi], int((np.diff(np.concatenate([[0], al])) != 0).sum()), float(al.mean()))
            mi = metrics(ida[lo:hi], cdi[lo:hi], ida[lo:hi])
            mc = metrics(cdi[lo:hi], cdi[lo:hi], ida[lo:hi])
            out.append({"modelo": name, "trecho": seg, "dias": hi - lo, "estrategia_base100": m["final_base100"], "ida_base100": mi["final_base100"],
                        "cdi_base100": mc["final_base100"], "ganho_vs_ida_pp": m["final_base100"] / mi["final_base100"] * 100 - 100,
                        "ganho_vs_cdi_pp": m["final_base100"] / mc["final_base100"] * 100 - 100, "taxa_alerta": float(al.mean()),
                        "trocas": m["trocas"]})
        # conferência com a coluna do Excel (série base 100 da monografia)
        col = "Random Forest" if name == "Random Forest" else "Modelo"
        excel_final = float(df[col].dropna().iloc[-1]) if col in df else np.nan
        full = np.where(a == 1, cdi[FIRST:FIRST + N_WIN], ida[FIRST:FIRST + N_WIN])
        out.append({"modelo": name, "trecho": "conferência: série Excel x recálculo", "dias": N_WIN, "estrategia_base100": equity(full)[-1],
                    "ida_base100": np.nan, "cdi_base100": np.nan, "ganho_vs_ida_pp": np.nan, "ganho_vs_cdi_pp": np.nan,
                    "taxa_alerta": excel_final, "trocas": None})
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "outputs" / "tables" / "phase9_original_backtest_in_out_sample.csv", index=False)
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
