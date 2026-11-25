"""Decomposição passo a passo: do pipeline ORIGINAL ao corrigido (Fase 8).

Cada passo aplica UMA correção a mais, mantendo o resto igual ao anterior:

  S0 original      : código literal da monografia (features com fillna(0), scaler na base inteira, alvo `max`, sem purge,
                     calibração e threshold F1 na mesma validação, regra `>`)
  S1 +alvo         : alvo formal Y_t (Fase 3)
  S2 +scaler       : StandardScaler ajustado só até a última decisão de treino
  S3 +variáveis    : variáveis da Fase 4 (sem fillna(0), aquecimento, RSI/EWM corrigidos)
  S4 +purge        : purge de 9 amostras no fim de treino e de validação
  S5 +cross-fit    : calibração/threshold por validação cruzada em blocos (regra `>=`)  == pipeline da Fase 7 (aqui com 4071 janelas)

As janelas são sempre as 4071 do original (decisões i+99, i = 0..4070) para isolar o efeito de cada correção.
Resultados em cache: outputs/metrics/ablation_cache/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("data", "targets", "features", "validation", "models", "calibration"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from build_target import build_target  # noqa: E402
from calibration import (IsotonicCalibrator, PlattCalibrator, crossfit_calibrate, prf, select_threshold)  # noqa: E402
from clean_base import build_clean_base, load_monograph, load_raw  # noqa: E402
from compare_features import original_features  # noqa: E402
from engineering import FEATURE_COLUMNS, build_features, fit_scaler, transform  # noqa: E402
from lstm_model import train_lstm  # noqa: E402
from perceptron import PerceptronScratch  # noqa: E402
from random_forest import make_rf  # noqa: E402

LOOKBACK, HORIZON = 100, 10
N_WIN = 4071
STEPS = ["S0_original", "S1_alvo", "S2_scaler", "S3_variaveis", "S4_purge", "S5_crossfit"]
ORIG_COLS = ["IDA_DI", "Duration", "IPCA_Deb", "BOVA", "CDI_dia", "Excesso_dia", "vol_20", "trend_20", "ret_20",
             "rsi_14", "macd", "macd_signal"]
CACHE = ROOT / "outputs" / "metrics" / "ablation_cache"


# ---------------------------------------------------------------- insumos
def load_inputs():
    clean = build_clean_base(load_raw())
    mono = load_monograph()
    start = int(np.where(clean["in_original_sample"])[0][0])
    sample = clean.iloc[start:].reset_index(drop=True)
    # variáveis: versão literal do original (com fillna(0)) e versão da Fase 4, ambas indexadas pela amostra (4181 linhas)
    x_orig = original_features(mono)[ORIG_COLS].fillna(0).values
    f = build_features(clean)
    x_new = f.iloc[start:].reset_index(drop=True)[FEATURE_COLUMNS]
    assert not x_new.isna().any().any()
    x_new = x_new.values
    # alvos
    dd = (sample["ida_trail_10"] < sample["cdi_trail_10"]).astype(int).values
    y_orig = np.array([dd[i + LOOKBACK:i + LOOKBACK + HORIZON].max() for i in range(N_WIN)])
    tgt = build_target(sample)
    y_new = tgt["y"].iloc[LOOKBACK - 1:LOOKBACK - 1 + N_WIN].astype(int).values
    exc = tgt["fwd_excess"].iloc[LOOKBACK - 1:LOOKBACK - 1 + N_WIN].values
    return x_orig, x_new, y_orig, y_new, exc


def windows(x_scaled):
    return np.stack([x_scaled[i:i + LOOKBACK] for i in range(N_WIN)])


def aggregate_original(w):
    """Agregação literal do original: nan_to_num em toda a matriz (janela constante -> NaN -> 0)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.concatenate([w.mean(1), w.std(1), w.min(1), w.max(1), np.median(w, 1),
                            skew(w, axis=1, bias=False, nan_policy="omit"),
                            kurtosis(w, axis=1, fisher=True, bias=False, nan_policy="omit"),
                            np.percentile(w, 25, axis=1), np.percentile(w, 75, axis=1), w[:, -1, :]], axis=1)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- um passo
def make_split_idx(purge: bool):
    a, b = int(N_WIN * 0.6), int(N_WIN * 0.8)
    tr, va, te = np.arange(0, a), np.arange(a, b), np.arange(b, N_WIN)
    if purge:
        tr = tr[(tr + HORIZON) <= a]      # target_end = d+10 = (i+99)+10; decisão seguinte = a+99 -> i+10 <= a  (9 removidos)
        va = va[(va + HORIZON) <= b]
    return tr, va, te


def original_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[int(np.argmax(f1))])


def run(step: str, kind: str, seed: int, inputs) -> dict:
    x_orig, x_new, y_orig, y_new, exc = inputs
    cfg = {"S0_original": dict(tgt="orig", scaler="full", feats="orig", purge=False, calib="same"),
           "S1_alvo": dict(tgt="new", scaler="full", feats="orig", purge=False, calib="same"),
           "S2_scaler": dict(tgt="new", scaler="train", feats="orig", purge=False, calib="same"),
           "S3_variaveis": dict(tgt="new", scaler="train", feats="new", purge=False, calib="same"),
           "S4_purge": dict(tgt="new", scaler="train", feats="new", purge=True, calib="same"),
           "S5_crossfit": dict(tgt="new", scaler="train", feats="new", purge=True, calib="crossfit")}[step]
    x = x_orig if cfg["feats"] == "orig" else x_new
    y = y_orig if cfg["tgt"] == "orig" else y_new
    tr, va, te = make_split_idx(cfg["purge"])
    last_train_decision_row = int(tr.max()) + LOOKBACK - 1
    sc = StandardScaler().fit(x if cfg["scaler"] == "full" else x[:last_train_decision_row + 1])
    w = windows(sc.transform(x))
    out = {"step": step, "kind": kind, "seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "pos_train": int(y[tr].sum()), "pos_val": int(y[va].sum()), "pos_test": int(y[te].sum()), **cfg}

    if kind == "perceptron":
        agg = aggregate_original(w)
        m = PerceptronScratch(learning_rate=0.01, n_epochs=100, random_state=seed).fit(agg[tr], y[tr])
        s_te = m.decision_function(agg[te])
        out.update(auc=float(roc_auc_score(y[te], s_te)), brier=None, threshold=0.0, **prf(y[te], s_te, 0.0))
        return out

    if kind == "rf":
        agg = aggregate_original(w)
        rf = make_rf(random_state=seed).fit(agg[tr], y[tr])
        s_va, s_te = rf.predict_proba(agg[va])[:, 1], rf.predict_proba(agg[te])[:, 1]
        cal_cls = IsotonicCalibrator
    else:
        _, _, _ = None, None, None
        model, _, _ = train_lstm(w[tr].astype(np.float32), y[tr], w[va].astype(np.float32), y[va], seed)
        s_va = model.predict(w[va].astype(np.float32), verbose=0).ravel()
        s_te = model.predict(w[te].astype(np.float32), verbose=0).ravel()
        cal_cls = PlattCalibrator

    cal = cal_cls().fit(s_va, y[va])
    p_te = cal.predict(s_te)
    if cfg["calib"] == "same":
        p_va = cal.predict(s_va)
        thr = original_threshold(y[va], p_va)
        pred_p = (p_te > thr).astype(int)                      # regra literal do original: `>`
        m = prf(y[te], pred_p.astype(float), 0.5)               # métricas do alerta binário
    else:
        method = "isotonic" if kind == "rf" else "platt"
        decision_rows = np.arange(N_WIN) + LOOKBACK - 1
        oof = crossfit_calibrate(s_va, y[va], va, decision_rows, HORIZON, method, k=3)
        thr = select_threshold(y[va], oof, "f1")
        m = prf(y[te], p_te, thr)
    out.update(auc=float(roc_auc_score(y[te], s_te)), auc_cal=float(roc_auc_score(y[te], p_te)),
               brier=float(brier_score_loss(y[te], p_te)), threshold=float(thr), **m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="*", default=["perceptron", "rf", "lstm"])
    ap.add_argument("--rf-seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--lstm-seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--steps", nargs="*", default=STEPS)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    seeds = {"perceptron": [42], "rf": a.rf_seeds, "lstm": a.lstm_seeds}
    for kind in a.kinds:
        for step in a.steps:
            for seed in seeds[kind]:
                f = CACHE / f"{step}__{kind}__{seed}.json"
                if f.exists():
                    continue
                res = run(step, kind, seed, inputs)
                f.write_text(json.dumps(res, default=float), encoding="utf-8")
                print(step, kind, seed, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                                         if k in ("auc", "brier", "threshold", "precision", "recall", "f1", "tn", "fp", "fn", "tp")}, flush=True)


if __name__ == "__main__":
    main()
