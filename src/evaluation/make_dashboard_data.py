"""Gera os arquivos de dados (JSON {rows, meta}) usados pelo painel de resultados. Não recalcula nada: só reorganiza tabelas já geradas."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from ablation import STEPS  # noqa: E402

OUT = ROOT / "outputs" / "dashboard_data"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"perceptron": "Perceptron", "rf": "Random Forest", "lstm": "LSTM"}
STEP_SHORT = {"S0_original": "Original", "S1_alvo": "+ alvo formal", "S2_scaler": "+ scaler no treino", "S3_variaveis": "+ variáveis",
              "S4_purge": "+ purge", "S5_crossfit": "+ cross-fit"}


def dump(name: str, rows: list[dict], meta: list[dict]) -> None:
    clean = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()} for r in rows]
    (OUT / f"{name}.json").write_text(json.dumps({"rows": clean, "meta": meta}, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    # 1) métricas no teste com IC
    ci = pd.read_csv(ROOT / "outputs" / "tables" / "phase8_metrics_ci.csv")
    rows = [{"modelo": r.model, "metrica": r.metric, "ponto": float(r.point), "ic_baixo": float(r.ci_low), "ic_alto": float(r.ci_high),
             "execucoes": int(r.n_runs)} for r in ci.itertuples()]
    base = json.loads((ROOT / "outputs" / "metrics" / "test_baselines.json").read_text(encoding="utf-8"))
    for met, val in (("auc", 0.5), ("brier", base["brier_constant_prevalence"]), ("f1", base["f1_always_alert"])):
        rows.append({"modelo": "Referência", "metrica": met, "ponto": float(val), "ic_baixo": None, "ic_alto": None, "execucoes": 0})
    dump("ci_metrics", rows, [{"label": "Gerado por", "value": "src/evaluation/evaluate.py (bootstrap em blocos de 20 dias, 1000 reamostragens)"},
                              {"label": "Origem", "value": "outputs/tables/phase8_metrics_ci.csv"}])

    # 2) decomposição passo a passo
    ab = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "outputs" / "metrics" / "ablation_cache").glob("*.json"))])
    ab["auc_report"] = ab["auc_cal"].where(ab["auc_cal"].notna(), ab["auc"])
    rows = []
    for kind, nm in NAMES.items():
        for i, st in enumerate(STEPS):
            x = ab[(ab.kind == kind) & (ab.step == st)]
            if x.empty:
                continue
            sd = float(x.auc_report.std(ddof=1)) if len(x) > 1 else None
            rows.append({"modelo": nm, "passo_ordem": i, "passo": STEP_SHORT[st], "auc": float(x.auc_report.mean()), "auc_dp": sd,
                         "brier": None if x.brier.isna().all() else float(x.brier.mean()), "precisao": float(x.precision.mean()),
                         "recall": float(x.recall.mean()), "f1": float(x.f1.mean()), "positivos_teste": int(x.pos_test.iloc[0]),
                         "sementes": int(len(x))})
    dump("ablation", rows, [{"label": "Gerado por", "value": "src/evaluation/ablation.py (média das sementes)"},
                            {"label": "Origem", "value": "outputs/metrics/ablation_cache/*.json"}])

    # 3) reprodução da monografia (S0)
    s0 = ab[ab.step == "S0_original"]
    pub = {("Random Forest", "F1"): 0.562, ("Random Forest", "AUC"): 0.737, ("Random Forest", "Brier"): 0.175, ("Random Forest", "Limiar F1"): 0.237,
           ("LSTM", "F1"): 0.389, ("LSTM", "AUC"): 0.810, ("LSTM", "Brier"): 0.198, ("LSTM", "Limiar F1"): 0.269,
           ("Perceptron", "Precisão"): 0.558, ("Perceptron", "Recall"): 0.446, ("Perceptron", "F1"): 0.496}
    col = {"F1": "f1", "AUC": "auc_report", "Brier": "brier", "Limiar F1": "threshold", "Precisão": "precision", "Recall": "recall"}
    kind_of = {v: k for k, v in NAMES.items()}
    rows = []
    for (nm, met), val in pub.items():
        x = s0[s0.kind == kind_of[nm]][col[met]]
        rows.append({"modelo": nm, "metrica": met, "monografia": val, "reproduzido": float(x.mean()), "sementes": int(len(x))})
    for nm, k in (("Perceptron", "perceptron"), ("Random Forest", "rf"), ("LSTM", "lstm")):
        rows.append({"modelo": nm, "metrica": "Positivos no teste", "monografia": {"Perceptron": 314, "Random Forest": 251, "LSTM": 251}[nm],
                     "reproduzido": int(s0[s0.kind == k].pos_test.iloc[0]), "sementes": int(len(s0[s0.kind == k]))})
    dump("recon", rows, [{"label": "Gerado por", "value": "src/evaluation/ablation.py, passo S0 (código original literal)"},
                         {"label": "Valores da monografia", "value": "Figuras e tabelas do PDF (RF: Fig. 6.x; LSTM: Fig. 5.x; Perceptron: Fig. 4.1)"}])

    # 4) prevalência do alvo por conjunto (split 60/20/20 de referência da Fase 3)
    pr = pd.read_csv(ROOT / "outputs" / "tables" / "target_rate_by_split.csv")
    sizes = {"treino": 2442, "validacao": 814, "teste": 815}
    rows = []
    for r in pr.itertuples():
        rows.append({"conjunto": r.split, "n": sizes[r.split], "alvo_original": float(r.y_original), "alvo_formal": float(r.y_new)})
    dump("prevalence", rows, [{"label": "Gerado por", "value": "src/targets/compare_targets.py"},
                              {"label": "Origem", "value": "outputs/tables/target_rate_by_split.csv"}])

    # 5) divisão definitiva (Fase 5)
    sp = pd.read_csv(ROOT / "outputs" / "tables" / "split_summary.csv")
    rows = [{"bloco": r.bloco, "n": int(r.n), "primeira_decisao": str(r.primeira_decisao), "ultima_decisao": str(r.ultima_decisao),
             "positivos": int(r.positivos), "taxa": float(r.taxa_positivos)} for r in sp.itertuples()]
    dump("split", rows, [{"label": "Gerado por", "value": "src/validation/build_split.py"}, {"label": "Origem", "value": "outputs/tables/split_summary.csv"}])
    for p in sorted(OUT.glob("*.json")):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
