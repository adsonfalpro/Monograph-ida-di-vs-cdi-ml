"""Treina Perceptron, Random Forest e LSTM nos blocos fit/val (sem tocar o teste) e salva modelos e diagnósticos.

Uso:  python -I src/models/train_all.py [--lstm-seeds 42 43 44 45 46] [--rf-seeds 42 43 44 45 46]
Saídas: outputs/models/*, outputs/metrics/phase6_training_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import ROOT, build_dataset  # noqa: E402
from lstm_model import LSTM_PARAMS, train_lstm  # noqa: E402
from perceptron import PERCEPTRON_PARAMS, PerceptronScratch  # noqa: E402
from random_forest import RF_PARAMS, make_rf  # noqa: E402

MODELS = ROOT / "outputs" / "models"
METRICS = ROOT / "outputs" / "metrics"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lstm-seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--rf-seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--skip", nargs="*", default=[], choices=["perceptron", "rf", "lstm"])
    args = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    ds = build_dataset()
    yf, yv = ds.y("fit"), ds.y("val")
    summary: dict = {
        "data": {"n_fit": len(yf), "n_val": len(yv), "pos_fit": int(yf.sum()), "pos_val": int(yv.sum()),
                 "scaler_rows": list(ds.scaler_rows), "undefined_agg_stats": ds.undefined_stats},
    }

    if "perceptron" not in args.skip:
        t0 = time.time()
        per = PerceptronScratch(**PERCEPTRON_PARAMS).fit(ds.agg("fit"), yf)
        joblib.dump(per, MODELS / "perceptron_seed42.joblib")
        pv = per.predict(ds.agg("val"))
        summary["perceptron"] = {
            "params": PERCEPTRON_PARAMS, "auc_fit": roc_auc_score(yf, per.decision_function(ds.agg("fit"))),
            "auc_val": roc_auc_score(yv, per.decision_function(ds.agg("val"))),
            "val_pred_positive_rate": float(pv.mean()), "errors_first_epoch": per.errors_per_epoch[0],
            "errors_last_epoch": per.errors_per_epoch[-1], "seconds": round(time.time() - t0, 1),
        }
        print("perceptron", summary["perceptron"], flush=True)

    if "rf" not in args.skip:
        runs = []
        for seed in args.rf_seeds:
            t0 = time.time()
            rf = make_rf(random_state=seed).fit(ds.agg("fit"), yf)
            joblib.dump(rf, MODELS / f"rf_seed{seed}.joblib", compress=3)
            runs.append({"seed": seed, "auc_fit": roc_auc_score(yf, rf.predict_proba(ds.agg("fit"))[:, 1]),
                         "auc_val": roc_auc_score(yv, rf.predict_proba(ds.agg("val"))[:, 1]),
                         "seconds": round(time.time() - t0, 1)})
            print("rf", runs[-1], flush=True)
        summary["rf"] = {"params": {k: v for k, v in RF_PARAMS.items()}, "runs": runs}

    if "lstm" not in args.skip:
        runs = []
        for seed in args.lstm_seeds:
            t0 = time.time()
            model, hist, cw = train_lstm(ds.seq("fit"), yf, ds.seq("val"), yv, seed)
            model.save(MODELS / f"lstm_seed{seed}.keras")
            pf = model.predict(ds.seq("fit"), verbose=0).ravel()
            pv = model.predict(ds.seq("val"), verbose=0).ravel()
            val_loss = hist.history["val_loss"]
            runs.append({"seed": seed, "epochs_run": len(val_loss), "best_epoch": int(np.argmin(val_loss)) + 1,
                         "best_val_loss": float(np.min(val_loss)), "auc_fit": roc_auc_score(yf, pf),
                         "auc_val": roc_auc_score(yv, pv), "class_weight": cw, "seconds": round(time.time() - t0, 1)})
            print("lstm", runs[-1], flush=True)
        summary["lstm"] = {"params": LSTM_PARAMS, "runs": runs}

    out = METRICS / "phase6_training_summary.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    old.update(summary)
    out.write_text(json.dumps(old, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("salvo:", out)


if __name__ == "__main__":
    main()
