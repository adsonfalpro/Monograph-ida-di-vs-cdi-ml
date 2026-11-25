"""Calibração probabilística e escolha de threshold (Fase 7).

* Calibradores como no original: isotônica (RF) e Platt (`LogisticRegression` padrão sobre o score; LSTM). O Perceptron não tinha
  probabilidades na monografia; aqui ganha Platt sobre o score contínuo (acréscimo documentado).
* ``crossfit``: cada observação de ``val`` é calibrada por um calibrador ajustado em OUTRAS dobras (blocos contíguos, com purge).
  Os thresholds são escolhidos sobre essas probabilidades fora da amostra, nunca sobre dados vistos pelo calibrador que as gerou.
* Regra de decisão: alerta se ``p >= limiar`` (convenção do ``precision_recall_curve``; o original usava ``>`` com um limiar vindo
  de ``thr[argmax(F1)]``, o que deixava de fora o próprio ponto ótimo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from split import blocked_folds  # noqa: E402


ALERT_TOL = 1e-9  # os limiares são valores de platô; uma ida e volta por CSV pode alterar o último dígito de p


class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000)

    def fit(self, score, y):
        self.lr.fit(np.asarray(score).reshape(-1, 1), y)
        return self

    def predict(self, score):
        return self.lr.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]

    def params(self):
        return {"method": "platt", "coef": float(self.lr.coef_[0, 0]), "intercept": float(self.lr.intercept_[0])}


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, score, y):
        self.iso.fit(np.asarray(score), y)
        return self

    def predict(self, score):
        return self.iso.predict(np.asarray(score))

    def params(self):
        return {"method": "isotonic", "x": self.iso.X_thresholds_.tolist(), "y": self.iso.y_thresholds_.tolist()}


def make_calibrator(method: str):
    return {"platt": PlattCalibrator, "isotonic": IsotonicCalibrator}[method]()


def crossfit_calibrate(score_val, y_val, val_positions, decision_rows, horizon, method, k=3):
    """Probabilidades calibradas fora da amostra para todo ``val`` (ordem de ``val_positions``)."""
    score_val, y_val = np.asarray(score_val), np.asarray(y_val)
    local = {p: i for i, p in enumerate(val_positions)}
    oof = np.full(len(val_positions), np.nan)
    for train, held in blocked_folds(val_positions, decision_rows, horizon, k):
        ti = [local[p] for p in train]
        hi = [local[p] for p in held]
        cal = make_calibrator(method).fit(score_val[ti], y_val[ti])
        oof[hi] = cal.predict(score_val[hi])
    assert not np.isnan(oof).any()
    return oof


# ---------------------------------------------------------------- métricas por limiar
def confusion(y, p, thr):
    pred = (np.asarray(p) >= thr - ALERT_TOL).astype(int)
    y = np.asarray(y)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tn, fp, fn, tp


def prf(y, p, thr):
    tn, fp, fn, tp = confusion(y, p, thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "alert_rate": (tp + fp) / len(y)}


def select_threshold(y, p, rule: str, excess=None):
    """Regras: 'fixed_0.5'; 'f1' (máximo F1); 'value' (maximiza o ganho acumulado sobre o IDA ao migrar para o CDI).

    Empates: escolhe o MAIOR limiar (menos alertas). ``value`` usa a soma de ``-excesso`` dos dias com alerta, proxy do
    ganho de migrar para o CDI (janelas se sobrepõem; é um critério, não um retorno de carteira).
    """
    y, p = np.asarray(y), np.asarray(p)
    if rule == "fixed_0.5":
        return 0.5
    cands = np.unique(p)
    if rule == "f1":
        scores = np.array([prf(y, p, t)["f1"] for t in cands])
    elif rule == "value":
        e = np.asarray(excess)
        scores = np.array([-(e[p >= t - ALERT_TOL]).sum() for t in cands])
    else:
        raise ValueError(rule)
    best = scores.max()
    return float(cands[np.where(scores >= best - 1e-12)[0].max()])


def cost_table(y, excess):
    """Custos assimétricos medidos nos dados (excesso dos próximos h dias, decimal)."""
    y, e = np.asarray(y), np.asarray(excess)
    gain_tp = float(-e[y == 1].mean())       # perda evitada por alerta correto (>0)
    cost_fp = float(e[y == 0].mean())        # ganho do IDA perdido por falso alerta (>0)
    return {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()), "gain_per_TP": gain_tp, "cost_per_FP": cost_fp,
            "cost_per_FN_equals_gain_TP": gain_tp, "ratio_FN_over_FP": gain_tp / cost_fp,
            "breakeven_precision": cost_fp / (cost_fp + gain_tp)}
