"""Avalia UMA vez no teste os calibradores/thresholds congelados (frozen_config.json). Fase 7.

Cada modelo/semente libera o teste uma única vez (TestSetGuard). Salva as previsões para a Fase 8 (nada é recalculado lá).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "models"))
sys.path.insert(0, str(HERE.parents[0] / "validation"))
from calibration import prf  # noqa: E402
from dataset import ROOT, build_dataset  # noqa: E402
from split import TestSetGuard  # noqa: E402

MODELS = ROOT / "outputs" / "models"


def apply_calibration(params: dict, score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if params["method"] == "platt":
        return 1.0 / (1.0 + np.exp(-(params["coef"] * score + params["intercept"])))
    return np.interp(score, params["x"], params["y"])     # isotônica com clip nas pontas


def score_arrays(kind: str, seed: int, x_seq, x_agg) -> np.ndarray:
    if kind == "perceptron":
        return joblib.load(MODELS / "perceptron_seed42.joblib").decision_function(x_agg)
    if kind == "rf":
        return joblib.load(MODELS / f"rf_seed{seed}.joblib").predict_proba(x_agg)[:, 1]
    import keras
    return keras.models.load_model(MODELS / f"lstm_seed{seed}.keras").predict(x_seq, verbose=0).ravel()


def main() -> None:
    frozen = json.loads((ROOT / "outputs" / "metrics" / "frozen_config.json").read_text(encoding="utf-8"))
    sha = frozen["sha256"]
    ds = build_dataset()
    dates = pd.read_csv(ROOT / "data" / "processed" / "split_indices.csv", parse_dates=["date"])
    test_dates = dates.loc[dates["block"] == "test", "date"].values
    guard = TestSetGuard()
    pred_rows, metric_rows = [], []
    for name, cfg in frozen["models"].items():
        x_seq, x_agg, y, exc = ds.release_test(guard, name, {"frozen_sha256": sha, "primary": frozen["primary"]})
        s = score_arrays(cfg["kind"], cfg["seed"], x_seq, x_agg)
        p = apply_calibration(cfg["calibration"], s)
        pred_rows.append(pd.DataFrame({"model": name, "kind": cfg["kind"], "seed": cfg["seed"], "date": test_dates,
                                       "y": y, "fwd_excess": exc, "raw_score": s, "prob": p}))
        auc = roc_auc_score(y, s)
        brier = brier_score_loss(y, p)
        for proc in ("crossfit", "original_style"):
            for rule, thr in cfg["thresholds"][proc].items():
                m = prf(y, p, thr)
                metric_rows.append({"model": name, "kind": cfg["kind"], "seed": cfg["seed"], "procedure": proc, "rule": rule,
                                    "threshold": thr, **m, "auc": auc, "brier": brier,
                                    "primary": proc == frozen["primary"]["procedure"] and rule == frozen["primary"]["threshold_rule"]})
        if "native_step_on_score" in cfg["thresholds"]:
            m = prf(y, s, 0.0)
            metric_rows.append({"model": name, "kind": cfg["kind"], "seed": cfg["seed"], "procedure": "native", "rule": "step_score>=0",
                                "threshold": 0.0, **m, "auc": auc, "brier": np.nan, "primary": False})
        print(name, f"AUC={auc:.3f} Brier={brier:.3f}", flush=True)
    base_rate = float(y.mean())
    pd.concat(pred_rows).to_csv(ROOT / "data" / "processed" / "test_predictions.csv", index=False, date_format="%Y-%m-%d", float_format="%.17g")
    mt = pd.DataFrame(metric_rows)
    mt.to_csv(ROOT / "outputs" / "tables" / "phase7_test_metrics.csv", index=False)
    (ROOT / "outputs" / "metrics" / "test_baselines.json").write_text(json.dumps({
        "prevalence": base_rate, "brier_constant_prevalence": base_rate * (1 - base_rate),
        "f1_always_alert": 2 * base_rate / (1 + base_rate), "precision_always_alert": base_rate}, indent=1), encoding="utf-8")
    print("prevalence", base_rate, "sha", sha)


if __name__ == "__main__":
    main()
