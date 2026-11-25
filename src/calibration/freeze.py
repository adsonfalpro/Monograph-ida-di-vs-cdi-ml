"""Congela calibradores e thresholds usando APENAS fit/val. Não toca o teste.

Saída: outputs/metrics/frozen_config.json (com hash), outputs/tables/phase7_validation_diagnostics.csv,
       outputs/tables/phase7_cost_table.csv
"""
from __future__ import annotations

import hashlib
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
from calibration import cost_table, crossfit_calibrate, make_calibrator, prf, select_threshold  # noqa: E402
from dataset import ROOT, build_dataset  # noqa: E402

MODELS = ROOT / "outputs" / "models"
HORIZON = 10
SEEDS = [42, 43, 44, 45, 46]
CAL_METHOD = {"perceptron": "platt", "rf": "isotonic", "lstm": "platt"}
PRIMARY_RULE = {"procedure": "crossfit", "threshold_rule": "f1"}   # declarado ANTES de qualquer acesso ao teste


def raw_scores(kind: str, seed: int, ds, block: str) -> np.ndarray:
    if kind == "perceptron":
        return joblib.load(MODELS / "perceptron_seed42.joblib").decision_function(ds.agg(block))
    if kind == "rf":
        return joblib.load(MODELS / f"rf_seed{seed}.joblib").predict_proba(ds.agg(block))[:, 1]
    import keras
    return keras.models.load_model(MODELS / f"lstm_seed{seed}.keras").predict(ds.seq(block), verbose=0).ravel()


def model_list():
    return [("perceptron", 42)] + [("rf", s) for s in SEEDS] + [("lstm", s) for s in SEEDS]


def main() -> None:
    ds = build_dataset()
    yv, ev = ds.y("val"), ds.excess("val")
    val_pos = ds.split.blocks["val"]
    rows = ds.split.decision_rows
    cost = {b: cost_table(ds.y(b), ds.excess(b)) for b in ("fit", "val")}
    pd.DataFrame(cost).T.to_csv(ROOT / "outputs" / "tables" / "phase7_cost_table.csv")

    frozen = {"primary": PRIMARY_RULE, "cost_table_fit_val": cost, "models": {}}
    diag = []
    for kind, seed in model_list():
        name = f"{kind}_seed{seed}"
        method = CAL_METHOD[kind]
        sv = raw_scores(kind, seed, ds, "val")
        oof = crossfit_calibrate(sv, yv, val_pos, rows, HORIZON, method, k=3)
        final = make_calibrator(method).fit(sv, yv)              # calibrador final: todo `val`
        insample = final.predict(sv)                             # procedimento original: mesma validação para tudo
        thr = {
            "crossfit": {r: select_threshold(yv, oof, r, ev) for r in ("fixed_0.5", "f1", "value")},
            "original_style": {r: select_threshold(yv, insample, r, ev) for r in ("fixed_0.5", "f1", "value")},
        }
        if kind == "perceptron":
            thr["native_step_on_score"] = 0.0
        frozen["models"][name] = {"kind": kind, "seed": seed, "calibration": final.params(), "thresholds": thr}
        for proc, probs in (("crossfit", oof), ("original_style", insample)):
            for rule, t in thr[proc].items():
                m = prf(yv, probs, t)
                diag.append({"model": name, "procedure": proc, "rule": rule, "threshold": t, "val_precision": m["precision"],
                             "val_recall": m["recall"], "val_f1": m["f1"], "val_alert_rate": m["alert_rate"],
                             "val_brier": brier_score_loss(yv, probs), "val_auc_raw": roc_auc_score(yv, sv)})
        print(name, {p: {r: round(v, 3) for r, v in t.items()} for p, t in thr.items() if isinstance(t, dict)}, flush=True)

    blob = json.dumps(frozen, sort_keys=True, ensure_ascii=False)
    frozen["sha256"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    (ROOT / "outputs" / "metrics" / "frozen_config.json").write_text(json.dumps(frozen, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(diag).to_csv(ROOT / "outputs" / "tables" / "phase7_validation_diagnostics.csv", index=False)
    print("congelado:", frozen["sha256"])


if __name__ == "__main__":
    main()
