"""Fase 8: figuras e intervalos de confiança sobre as previsões do teste salvas na Fase 7 (nada é retreinado).

Cores: três primeiras posições da paleta categórica validada (azul, laranja, água-marinha), em ordem fixa por modelo.
Referências (acaso, linha de base) em cinza; texto em tons de tinta neutros.
Intervalos: bootstrap em blocos móveis (bloco = 20 dias úteis) sobre as datas do teste, mesmas amostras para todas as sementes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {"perceptron": "#2a78d6", "rf": "#eb6834", "lstm": "#1baf7a"}
NAMES = {"perceptron": "Perceptron", "rf": "Random Forest", "lstm": "LSTM"}
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLOCK, B, RNG = 20, 1000, np.random.default_rng(42)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "legend.frameon": False, "savefig.dpi": 150})


def load():
    pred = pd.read_csv(ROOT / "data" / "processed" / "test_predictions.csv", parse_dates=["date"])
    fr = json.loads((ROOT / "outputs" / "metrics" / "frozen_config.json").read_text(encoding="utf-8"))
    return pred, fr


def alerts(pred_m: pd.DataFrame, fr, model: str, proc: str, rule: str) -> np.ndarray:
    return (pred_m["prob"].values >= fr["models"][model]["thresholds"][proc][rule] - 1e-9).astype(int)


def conf(y, a):
    return np.array([[((a == 0) & (y == 0)).sum(), ((a == 1) & (y == 0)).sum()],
                     [((a == 0) & (y == 1)).sum(), ((a == 1) & (y == 1)).sum()]])


def block_indices(n: int) -> np.ndarray:
    nb = int(np.ceil(n / BLOCK))
    starts = RNG.integers(0, n - BLOCK + 1, size=nb)
    return np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]


def fig_confusion(pred, fr, normalized: bool):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for r, (proc, label) in enumerate([("crossfit", "Principal (cross-fit + F1)"), ("original_style", "Estilo original (mesma validação + F1)")]):
        for c, kind in enumerate(["perceptron", "rf", "lstm"]):
            name = f"{kind}_seed42"
            m = pred[pred.model == name]
            y = m["y"].values
            cm = conf(y, alerts(m, fr, name, proc, "f1"))
            data = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
            ax = axes[r, c]
            ax.imshow(data, cmap="Blues", vmin=0, vmax=(1 if normalized else cm.max()))
            ax.grid(False)
            for i in range(2):
                for j in range(2):
                    v = data[i, j]
                    ax.text(j, i, f"{v:.2f}" if normalized else f"{int(v)}", ha="center", va="center",
                            color="white" if v > (0.5 if normalized else cm.max() / 2) else INK, fontsize=12)
            ax.set_xticks([0, 1], ["Não abaixo", "Abaixo do CDI"]); ax.set_yticks([0, 1], ["Não abaixo", "Abaixo do CDI"])
            ax.set_title(f"{NAMES[kind]}" + (f"\n{label}" if c == 0 else ""), fontsize=10)
            ax.set_xlabel("Previsto"); ax.set_ylabel("Real" if c == 0 else "")
    fig.suptitle("Matriz de confusão no teste (semente 42" + ("; linhas normalizadas)" if normalized else ")"), color=INK)
    fig.tight_layout()
    fig.savefig(FIG / ("confusion_normalized.png" if normalized else "confusion_counts.png")); plt.close(fig)


def fig_curves(pred, kind_curve: str):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    prev = pred.drop_duplicates(["date", "model"]).y.mean()
    for ax, kind in zip(axes, ["perceptron", "rf", "lstm"]):
        aucs = []
        for name, m in pred[pred.kind == kind].groupby("model"):
            y, s = m["y"].values, m["raw_score"].values
            if kind_curve == "roc":
                x_, y_, _ = roc_curve(y, s); aucs.append(roc_auc_score(y, s))
            else:
                y_, x_, _ = precision_recall_curve(y, s); aucs.append(np.nan)
            ax.plot(x_, y_, color=COLORS[kind], lw=1.4, alpha=0.8)
        if kind_curve == "roc":
            ax.plot([0, 1], [0, 1], color=REF, ls="--", lw=1)
            ax.set_title(f"{NAMES[kind]}  (AUC {np.mean(aucs):.2f}; {len(aucs)} exec.)")
            ax.set_xlabel("Taxa de falsos positivos")
        else:
            ax.axhline(prev, color=REF, ls="--", lw=1)
            ax.set_title(NAMES[kind]); ax.set_xlabel("Recall")
        ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
    axes[0].set_ylabel("Taxa de verdadeiros positivos" if kind_curve == "roc" else "Precisão")
    if kind_curve == "pr":
        axes[2].text(0.98, prev + 0.03, f"prevalência {prev:.0%}", ha="right", color=INK2, fontsize=9)
    fig.suptitle("Curva ROC no teste (uma linha por semente)" if kind_curve == "roc" else "Curva precisão × recall no teste (uma linha por semente)", color=INK)
    fig.tight_layout(); fig.savefig(FIG / f"{kind_curve}_curve.png"); plt.close(fig)


def fig_calibration(pred):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True, sharey=True)
    for ax, kind in zip(axes, ["perceptron", "rf", "lstm"]):
        for name, m in pred[pred.kind == kind].groupby("model"):
            fp, mp = calibration_curve(m["y"], m["prob"], n_bins=8, strategy="uniform")
            ax.plot(mp, fp, color=COLORS[kind], lw=1.3, marker="o", ms=4, alpha=0.85)
        ax.plot([0, 1], [0, 1], color=REF, ls="--", lw=1)
        ax.set_title(NAMES[kind]); ax.set_xlabel("Probabilidade prevista"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Frequência observada")
    fig.suptitle("Calibração no teste (uma linha por semente; faixas sem dados omitidas)", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "calibration_curve.png"); plt.close(fig)


def fig_real_vs_pred(pred, fr):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
    for ax, kind in zip(axes, ["perceptron", "rf", "lstm"]):
        name = f"{kind}_seed42"
        m = pred[pred.model == name].reset_index(drop=True).iloc[:200]
        a = alerts(m, fr, name, "crossfit", "f1")
        ax.step(range(len(m)), m["y"], where="post", color=REF, lw=2.2, label="Real (abaixo do CDI)")
        ax.step(range(len(m)), a * 0.92, where="post", color=COLORS[kind], lw=1.4, label=f"{NAMES[kind]}: alerta (regra principal)")
        ax.set_yticks([0, 1]); ax.set_ylim(-0.1, 1.15); ax.legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Amostra do teste (primeiras 200)")
    fig.suptitle("Real × alerta nas 200 primeiras amostras do teste (semente 42)", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "real_vs_predicted.png"); plt.close(fig)


def bootstrap_metrics(pred, fr):
    """Médias entre sementes de AUC, Brier e F1 (regra principal), com IC 95 % por bootstrap em blocos."""
    models = {k: sorted(pred[pred.kind == k].model.unique()) for k in ("perceptron", "rf", "lstm")}
    wide = {n: pred[pred.model == n].sort_values("date").reset_index(drop=True) for ns in models.values() for n in ns}
    n = len(next(iter(wide.values())))
    y = next(iter(wide.values()))["y"].values
    thr = {nm: fr["models"][nm]["thresholds"]["crossfit"]["f1"] for nm in wide}
    point, boots = {}, {k: {"auc": [], "brier": [], "f1": []} for k in models}

    def metrics(idx, nm):
        m = wide[nm]; yy = y[idx]
        a = (m["prob"].values[idx] >= thr[nm] - 1e-9).astype(int)
        tp = ((a == 1) & (yy == 1)).sum(); fp = ((a == 1) & (yy == 0)).sum(); fn = ((a == 0) & (yy == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        auc = roc_auc_score(yy, m["raw_score"].values[idx]) if 0 < yy.sum() < len(yy) else np.nan
        return auc, brier_score_loss(yy, m["prob"].values[idx]), f1

    full = np.arange(n)
    for k, ns in models.items():
        point[k] = np.nanmean([metrics(full, nm) for nm in ns], axis=0)
    for _ in range(B):
        idx = block_indices(n)
        for k, ns in models.items():
            v = np.nanmean([metrics(idx, nm) for nm in ns], axis=0)
            for key, val in zip(("auc", "brier", "f1"), v):
                boots[k][key].append(val)
    rows = []
    for k in models:
        for j, key in enumerate(("auc", "brier", "f1")):
            b_ = np.array(boots[k][key])
            rows.append({"model": NAMES[k], "metric": key, "point": point[k][j], "ci_low": np.nanpercentile(b_, 2.5),
                         "ci_high": np.nanpercentile(b_, 97.5), "n_runs": len(models[k]),
                         "p_auc_le_0.5": float((b_ <= 0.5).mean()) if key == "auc" else np.nan})
    pair = []
    for a_, b_ in (("rf", "perceptron"), ("lstm", "perceptron"), ("lstm", "rf")):
        d = np.array(boots[a_]["auc"]) - np.array(boots[b_]["auc"])
        pair.append({"diff": f"AUC {NAMES[a_]} − {NAMES[b_]}", "point": point[a_][0] - point[b_][0],
                     "ci_low": np.nanpercentile(d, 2.5), "ci_high": np.nanpercentile(d, 97.5)})
    return pd.DataFrame(rows), pd.DataFrame(pair), models


def fig_metrics(ci: pd.DataFrame, base: dict):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    refs = {"auc": (0.5, "acaso 0,50"), "brier": (base["brier_constant_prevalence"], "prevalência 0,164"), "f1": (base["f1_always_alert"], "alertar sempre 0,343")}
    titles = {"auc": "AUC (maior é melhor)", "brier": "Brier (menor é melhor)", "f1": "F1 da regra principal"}
    for ax, key in zip(axes, ["auc", "brier", "f1"]):
        d = ci[ci.metric == key].reset_index(drop=True)
        for i, r in d.iterrows():
            kind = [k for k, v in NAMES.items() if v == r.model][0]
            ax.errorbar(r.point, i, xerr=[[r.point - r.ci_low], [r.ci_high - r.point]], fmt="o", color=COLORS[kind], capsize=3, lw=1.6)
        ax.axvline(refs[key][0], color=REF, ls="--", lw=1)
        ax.text(refs[key][0], 2.55, refs[key][1], color=INK2, fontsize=8, ha="center")
        ax.set_yticks(range(3), d.model); ax.set_ylim(-0.5, 2.8); ax.invert_yaxis(); ax.set_title(titles[key], fontsize=10)
    fig.suptitle("Métricas no teste: média entre sementes e IC 95 % (bootstrap em blocos de 20 dias)", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "metrics_with_ci.png"); plt.close(fig)


def main():
    pred, fr = load()
    base = json.loads((ROOT / "outputs" / "metrics" / "test_baselines.json").read_text(encoding="utf-8"))
    for normalized in (False, True):
        fig_confusion(pred, fr, normalized)
    fig_curves(pred, "roc"); fig_curves(pred, "pr"); fig_calibration(pred); fig_real_vs_pred(pred, fr)
    ci, pair, _ = bootstrap_metrics(pred, fr)
    ci.to_csv(TAB / "phase8_metrics_ci.csv", index=False); pair.to_csv(TAB / "phase8_pairwise_auc.csv", index=False)
    fig_metrics(ci, base)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # consoles Windows (cp1252) não imprimem o sinal de menos
    print(ci.round(3).to_string()); print(pair.round(3).to_string())


if __name__ == "__main__":
    main()
