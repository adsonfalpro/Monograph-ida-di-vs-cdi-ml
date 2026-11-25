"""Planilha de conferência do backtest, com fórmulas do Excel, e verificação independente das fórmulas.

Gera outputs/backtest_check.xlsx (custo editável em Parametros!B1) e recalcula as fórmulas com a biblioteca `formulas`
(motor independente do pandas/numpy) para comparar com o backtest em Python (custo de 10 bps).
Obs.: o recálculo aqui é feito por um motor de fórmulas em Python, não pelo Excel; ao abrir a planilha no Excel a coluna de
conferência deve mostrar VERDADEIRO em todas as linhas.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter as L

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from backtest import RULES, equity, load_test_alignment, max_drawdown, metrics, strategy_returns, test_returns  # noqa: E402

COST = 10
STRATS = [("rf_seed42", "estilo_original"), ("lstm_seed42", "estilo_original"), ("lstm_seed42", "valor")]


def build() -> tuple[Path, dict]:
    clean, pred, fr = load_test_alignment()
    base = pred[pred.model == "rf_seed42"].sort_values("date")
    ida, cdi, dates, _ = test_returns(clean, base["date"])
    n = len(ida)
    wb = Workbook()
    par = wb.active; par.title = "Parametros"
    par["A1"] = "Custo por troca (bps)"; par["B1"] = COST
    par["A3"] = "Regras: sinal de t vale para o retorno de t+1; alerta=1 -> CDI, alerta=0 -> IDA; custo descontado no dia da troca; posição inicial no IDA."
    par["A4"] = "A coluna 'esperado' foi gerada pelo Python com custo = 10 bps; com outro custo, as colunas de conferência deixam de valer."
    d = wb.create_sheet("Dados")
    heads = ["data (t+1)", "ret IDA", "ret CDI", "IDA base 100", "CDI base 100"]
    for j, h in enumerate(heads, 1):
        d.cell(1, j, h).font = Font(bold=True)
    for i in range(n):
        r = i + 2
        d.cell(r, 1, pd.Timestamp(dates[i]).date()); d.cell(r, 2, float(ida[i])); d.cell(r, 3, float(cdi[i]))
        d.cell(r, 4, f"=100*(1+B{r})" if i == 0 else f"=D{r-1}*(1+B{r})")
        d.cell(r, 5, f"=100*(1+C{r})" if i == 0 else f"=E{r-1}*(1+C{r})")
    expected = {}
    col = 6
    spans = {}
    for model, rule in STRATS:
        g = pred[pred.model == model].sort_values("date")
        proc, thr_rule = RULES[rule]
        alert = (g["prob"].values >= fr["models"][model]["thresholds"][proc][thr_rule] - 1e-9).astype(int)
        r_, sw = strategy_returns(ida, cdi, alert, COST)
        cum = equity(r_)
        key = f"{model}|{rule}"
        expected[key] = {"final": float(cum[-1]), "trocas": sw, "mdd": max_drawdown(cum), "cum": cum}
        names = ["alerta", "troca", "retorno", "base 100", "pico", "drawdown", "esperado base 100", "confere"]
        for k, nm in enumerate(names):
            d.cell(1, col + k, f"{key}: {nm}").font = Font(bold=True)
        A, T, R, C, P, D_, E, F = (L(col + k) for k in range(8))
        for i in range(n):
            r = i + 2
            d[f"{A}{r}"] = int(alert[i])
            d[f"{T}{r}"] = f"=IF({A}{r}<>0,1,0)" if i == 0 else f"=IF({A}{r}<>{A}{r-1},1,0)"
            d[f"{R}{r}"] = f"=IF({A}{r}=1,C{r},B{r})-{T}{r}*Parametros!$B$1/10000"
            d[f"{C}{r}"] = f"=100*(1+{R}{r})" if i == 0 else f"={C}{r-1}*(1+{R}{r})"
            d[f"{P}{r}"] = f"=MAX(100,{C}{r})" if i == 0 else f"=MAX({P}{r-1},{C}{r})"
            d[f"{D_}{r}"] = f"={C}{r}/{P}{r}-1"
            d[f"{E}{r}"] = float(cum[i])
            d[f"{F}{r}"] = f"=ABS({C}{r}-{E}{r})<0.000001"
        spans[key] = (T, R, C, D_, F)
        col += len(names)
    s = wb.create_sheet("Resumo")
    for j, h in enumerate(["estratégia", "base 100 final", "retorno total", "retorno anual", "vol anual", "max drawdown", "trocas", "conferências VERDADEIRO"], 1):
        s.cell(1, j, h).font = Font(bold=True)
    last = n + 1
    rowi = 2
    for lbl, cumcol, retcol in (("IDA (comprar e manter)", "D", "B"), ("CDI 100 %", "E", "C")):
        s.cell(rowi, 1, lbl)
        s.cell(rowi, 2, f"=Dados!{cumcol}{last}"); s.cell(rowi, 3, f"=B{rowi}/100-1")
        s.cell(rowi, 4, f"=(1+C{rowi})^(252/COUNT(Dados!{retcol}2:{retcol}{last}))-1"); s.cell(rowi, 5, f"=STDEV(Dados!{retcol}2:{retcol}{last})*SQRT(252)")
        rowi += 1
    for key in expected:
        T, R, C, D_, F = spans[key]
        s.cell(rowi, 1, key)
        s.cell(rowi, 2, f"=Dados!{C}{last}"); s.cell(rowi, 3, f"=B{rowi}/100-1")
        s.cell(rowi, 4, f"=(1+C{rowi})^(252/COUNT(Dados!{R}2:{R}{last}))-1"); s.cell(rowi, 5, f"=STDEV(Dados!{R}2:{R}{last})*SQRT(252)")
        s.cell(rowi, 6, f"=MIN(Dados!{D_}2:{D_}{last})"); s.cell(rowi, 7, f"=SUM(Dados!{T}2:{T}{last})")
        s.cell(rowi, 8, f"=COUNTIF(Dados!{F}2:{F}{last},TRUE)")
        rowi += 1
    out = ROOT / "outputs" / "backtest_check.xlsx"
    wb.save(out)
    return out, {"expected": expected, "n": n, "ida": ida, "cdi": cdi}


def verify(path: Path, info: dict) -> pd.DataFrame:
    import formulas
    warnings.filterwarnings("ignore")
    xl = formulas.ExcelModel().loads(str(path)).finish()
    sol = xl.calculate()
    def val(sheet: str, cell: str):
        k = f"'[{path.name}]{sheet.upper()}'!{cell}"
        return float(np.asarray(sol[k].value).ravel()[0]) if k in sol else np.nan
    rows = []
    ida_cum = equity(info["ida"])[-1]; cdi_cum = equity(info["cdi"])[-1]
    rows.append(("IDA base 100 final", val("Resumo", "B2"), ida_cum)); rows.append(("CDI base 100 final", val("Resumo", "B3"), cdi_cum))
    for i, (key, e) in enumerate(info["expected"].items()):
        r = 4 + i
        rows.append((f"{key} base 100 final", val("Resumo", f"B{r}"), e["final"]))
        rows.append((f"{key} trocas", val("Resumo", f"G{r}"), e["trocas"]))
        rows.append((f"{key} max drawdown", val("Resumo", f"F{r}"), e["mdd"]))
        rows.append((f"{key} conferências VERDADEIRO", val("Resumo", f"H{r}"), info["n"]))
    df = pd.DataFrame(rows, columns=["item", "excel_formulas", "python"])
    df["dif_abs"] = (df.excel_formulas - df.python).abs()
    return df


def main() -> None:
    path, info = build()
    df = verify(path, info)
    df.to_csv(ROOT / "outputs" / "tables" / "phase9_excel_check.csv", index=False)
    print(df.to_string())
    print("máx dif:", df.dif_abs.max())


if __name__ == "__main__":
    main()
