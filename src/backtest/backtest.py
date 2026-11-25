"""Backtest (Fase 9).

Regras de execução (explícitas):
* O sinal de decisão em t usa informação até o fechamento de t. A posição vale para o retorno do dia t+1 (defasagem de 1 dia).
* alerta = 1  -> posição no CDI no dia t+1;  alerta = 0 -> posição no IDA no dia t+1. O sinal é reavaliado todo dia.
* Custo de transação: aplicado em cada troca de posição (IDA <-> CDI), em pontos-base do patrimônio, descontado do retorno do dia
  da troca. Cenários: 0, 10 e 25 bps por troca (suposição; o IDA é carteira teórica e não negociável, ver limitações).
  Posição inicial: IDA (o primeiro alerta conta como uma troca).
* Benchmarks: IDA comprado e mantido e CDI 100 %, sem custos, no mesmo intervalo.
* Período: apenas o bloco de teste (datas de decisão do teste; retornos de t+1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("data",):
    sys.path.insert(0, str(ROOT / "src" / sub))
from clean_base import build_clean_base, load_raw  # noqa: E402

ANN = 252
COSTS_BPS = (0, 10, 25)


def strategy_returns(ida: np.ndarray, cdi: np.ndarray, alert: np.ndarray, cost_bps: float = 0.0) -> tuple[np.ndarray, int]:
    """ida, cdi: retornos do dia t+1 (já alinhados à decisão em t); alert: sinal de t. Devolve (retornos, nº de trocas)."""
    alert = np.asarray(alert).astype(int)
    pos = np.concatenate([[0], alert[:-1]])          # posição anterior (inicia no IDA)
    switch = (alert != pos).astype(int)
    r = np.where(alert == 1, cdi, ida) - switch * cost_bps / 1e4
    return r, int(switch.sum())


def equity(r: np.ndarray, base: float = 100.0) -> np.ndarray:
    return base * np.cumprod(1.0 + r)


def max_drawdown(curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.concatenate([[100.0], curve]))[1:]
    return float((curve / peak - 1.0).min())


def metrics(r: np.ndarray, cdi: np.ndarray, ida: np.ndarray, switches: int | None = None, in_cdi: float | None = None) -> dict:
    curve = equity(r)
    n = len(r)
    total = curve[-1] / 100 - 1
    ann = (1 + total) ** (ANN / n) - 1
    vol = float(np.std(r, ddof=1) * np.sqrt(ANN))
    ex_cdi = r - cdi
    return {"final_base100": float(curve[-1]), "retorno_total": float(total), "retorno_anual": float(ann), "vol_anual": vol,
            "max_drawdown": max_drawdown(curve), "excesso_vs_cdi_anual": float(np.mean(ex_cdi) * ANN),
            "excesso_vs_ida_anual": float(np.mean(r - ida) * ANN),
            "trocas": switches, "dias_no_cdi": in_cdi, "n_dias": n}


def load_test_alignment():
    clean = build_clean_base(load_raw())
    pred = pd.read_csv(ROOT / "data" / "processed" / "test_predictions.csv", parse_dates=["date"])
    fr = json.loads((ROOT / "outputs" / "metrics" / "frozen_config.json").read_text(encoding="utf-8"))
    return clean, pred, fr


def test_returns(clean: pd.DataFrame, decision_dates: pd.Series):
    """Retornos do dia seguinte a cada decisão (t+1)."""
    idx = pd.Index(clean["date"])
    rows = idx.get_indexer(decision_dates)
    nxt = rows + 1
    assert (rows >= 0).all() and nxt.max() < len(clean)
    return clean["ida_ret"].values[nxt], clean["cdi_ret"].values[nxt], clean["date"].values[nxt], clean["ida_index_ret"].values[nxt]


RULES = {"principal": ("crossfit", "f1"), "valor": ("crossfit", "value"), "estilo_original": ("original_style", "f1")}


def block_bootstrap_ci(x: np.ndarray, block: int = 20, B: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    m = []
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        m.append(x[idx].mean() * ANN)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    clean, pred, fr = load_test_alignment()
    rows, curves = [], {}
    dates = None
    for name, grp in pred.groupby("model"):
        grp = grp.sort_values("date")
        ida, cdi, d_next, ida_idx = test_returns(clean, grp["date"])
        dates = d_next
        kind, seed = grp["kind"].iloc[0], int(grp["seed"].iloc[0])
        for rule, (proc, thr_rule) in RULES.items():
            thr = fr["models"][name]["thresholds"][proc][thr_rule]
            alert = (grp["prob"].values >= thr - 1e-9).astype(int)
            for c in COSTS_BPS:
                r, sw = strategy_returns(ida, cdi, alert, c)
                m = metrics(r, cdi, ida, sw, float(alert.mean()))
                rows.append({"modelo": name, "tipo": kind, "semente": seed, "regra": rule, "custo_bps": c, **m})
                if c == 0:
                    curves[(kind, seed, rule)] = equity(r)
    ida0, cdi0, d0, ida_idx0 = test_returns(clean, pred[pred.model == "rf_seed42"].sort_values("date")["date"])
    bench = {"IDA (comprar e manter)": metrics(ida0, cdi0, ida0, 0, 0.0), "CDI 100 %": metrics(cdi0, cdi0, ida0, 0, 1.0),
             "IDA pelo nível do índice (alternativa)": metrics(np.nan_to_num(ida_idx0), cdi0, ida0, 0, 0.0)}
    bt = pd.DataFrame(rows)
    bt.to_csv(ROOT / "outputs" / "tables" / "phase9_backtest_by_seed.csv", index=False)
    summ = (bt.groupby(["tipo", "regra", "custo_bps"])
              .agg(final_media=("final_base100", "mean"), final_min=("final_base100", "min"), final_max=("final_base100", "max"),
                   anual_media=("retorno_anual", "mean"), vol_media=("vol_anual", "mean"), mdd_media=("max_drawdown", "mean"),
                   exc_cdi_media=("excesso_vs_cdi_anual", "mean"), exc_ida_media=("excesso_vs_ida_anual", "mean"),
                   trocas_media=("trocas", "mean"), dias_cdi=("dias_no_cdi", "mean"), n=("semente", "count")).reset_index())
    summ.to_csv(ROOT / "outputs" / "tables" / "phase9_backtest_summary.csv", index=False)
    pd.DataFrame(bench).T.to_csv(ROOT / "outputs" / "tables" / "phase9_benchmarks.csv")
    # IC do excesso anualizado vs CDI e vs IDA (média entre sementes, custo 0), por tipo e regra
    ci_rows = []
    for (kind, rule), _ in bt[bt.custo_bps == 0].groupby(["tipo", "regra"]):
        seeds = sorted(bt[(bt.tipo == kind)]["semente"].unique())
        per_seed_ret = []
        for sd in seeds:
            grp = pred[pred.model == f"{kind}_seed{sd}"].sort_values("date")
            ida, cdi, _, _ = test_returns(clean, grp["date"])
            thr = fr["models"][f"{kind}_seed{sd}"]["thresholds"][RULES[rule][0]][RULES[rule][1]]
            r, _ = strategy_returns(ida, cdi, (grp["prob"].values >= thr - 1e-9).astype(int), 0)
            per_seed_ret.append(r)
        r_mean = np.mean(per_seed_ret, axis=0)
        lo, hi = block_bootstrap_ci(r_mean - cdi0)
        lo2, hi2 = block_bootstrap_ci(r_mean - ida0)
        ci_rows.append({"tipo": kind, "regra": rule, "exc_cdi_anual": float((r_mean - cdi0).mean() * ANN), "ic_cdi_baixo": lo, "ic_cdi_alto": hi,
                        "exc_ida_anual": float((r_mean - ida0).mean() * ANN), "ic_ida_baixo": lo2, "ic_ida_alto": hi2})
    pd.DataFrame(ci_rows).to_csv(ROOT / "outputs" / "tables" / "phase9_excess_ci.csv", index=False)
    np.savez(ROOT / "data" / "processed" / "phase9_curves.npz", dates=d0.astype("datetime64[D]").astype(str),
             ida=equity(ida0), cdi=equity(cdi0), **{f"{k}|{s}|{r}": v for (k, s, r), v in curves.items()})
    print(summ[summ.custo_bps == 0].round(3).to_string())
    print(pd.DataFrame(bench).T.round(3).to_string())
    print(pd.DataFrame(ci_rows).round(3).to_string())


if __name__ == "__main__":
    main()
