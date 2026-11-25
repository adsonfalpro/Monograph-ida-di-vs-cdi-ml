"""Gera docs/calibration_threshold.md a partir das tabelas da Fase 7."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from dataset import ROOT  # noqa: E402

T = ROOT / "outputs" / "tables"


def ms(x, d=3):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x[0]:.{d}f}"


def main() -> None:
    fr = json.loads((ROOT / "outputs" / "metrics" / "frozen_config.json").read_text(encoding="utf-8"))
    base = json.loads((ROOT / "outputs" / "metrics" / "test_baselines.json").read_text(encoding="utf-8"))
    cost = pd.read_csv(T / "phase7_cost_table.csv", index_col=0)
    dv = pd.read_csv(T / "phase7_validation_diagnostics.csv")
    mt = pd.read_csv(T / "phase7_test_metrics.csv")
    kinds = ["perceptron", "rf", "lstm"]
    names = {"perceptron": "Perceptron", "rf": "Random Forest", "lstm": "LSTM"}

    L = ["# Calibração e escolha do threshold (Fase 7)\n",
         "Gerado por `src/calibration/report_phase7.py`. Código: `src/calibration/`. Testes: `tests/test_calibration.py`.\n",
         "## 1. Procedimento\n",
         "1. Modelo treinado em `fit` (Fase 6). 2. Pontuações brutas em `val`. 3. **Calibração em validação cruzada em blocos** "
         "(3 dobras contíguas de `val`, purge de 9 amostras de cada lado): cada observação é calibrada por um calibrador que não a viu. "
         "4. Thresholds escolhidos sobre essas probabilidades fora da amostra. 5. Calibrador final ajustado em todo `val`. "
         "6. Tudo congelado em `outputs/metrics/frozen_config.json` "
         f"(SHA‑256 `{fr['sha256'][:16]}…`). 7. Teste avaliado **uma vez por modelo/semente** (`TestSetGuard`; registro em "
         "`outputs/metrics/test_access_log.json`).\n",
         "Calibradores (como no original): isotônica para o RF; Platt (`LogisticRegression`) para o LSTM e, **acréscimo**, para o Perceptron "
         "(sobre o score contínuo; o original não tinha probabilidades). Regra de alerta: `p ≥ limiar` (o original usava `>`).\n",
         "Regras de threshold avaliadas: `fixed_0.5`; `f1` (máximo F1, como na monografia); `value` (maximiza o ganho acumulado de migrar para o CDI, "
         "isto é, a soma de −excesso dos dias com alerta). Procedimentos: `crossfit` (limpo) e `original_style` (calibra e escolhe o limiar na mesma "
         "validação). Perceptron: também `native` (degrau no score ≥ 0).\n",
         f"**Critério principal, declarado antes de abrir o teste:** `{fr['primary']['procedure']}` + `{fr['primary']['threshold_rule']}`. "
         "Os demais são reportados sem escolha posterior.\n",
         "## 2. Custos dos erros (medidos em `fit` e `val`, nunca no teste)\n",
         "Retorno em excesso dos 10 dias seguintes (decimal). Falso positivo: migrar para o CDI quando o IDA ganharia (`cost_per_FP`). "
         "Verdadeiro positivo: perda evitada (`gain_per_TP`). Falso negativo: perda não evitada (= `gain_per_TP`). "
         "**Precisão de equilíbrio**: abaixo dela, alertar não compensa.\n",
         "| bloco | positivos | negativos | ganho por TP / custo de FN | custo por FP | razão FN/FP | precisão de equilíbrio |\n|---|---|---|---|---|---|---|"]
    for b, r in cost.iterrows():
        L.append(f"| {b} | {int(r.n_pos)} | {int(r.n_neg)} | {r.gain_per_TP:.4f} | {r.cost_per_FP:.4f} | {r.ratio_FN_over_FP:.1f} | {r.breakeven_precision:.1%} |")
    L += ["", "FN custa 2× (fit) a 4× (val) o FP em retorno esperado, mas um evento é raro; por isso o alerta só compensa quando a precisão "
          "supera 19–32 %. A prevalência no teste é "
          f"{base['prevalence']:.1%}, ou seja, **alertar todo dia** já tem precisão {base['precision_always_alert']:.1%} e F1 {base['f1_always_alert']:.3f}.\n",
          "## 3. Validação: efeito do procedimento (média das sementes)\n",
          "| modelo | procedimento | Brier em val | F1 em val (regra f1) | taxa de alerta |\n|---|---|---|---|---|"]
    for k in kinds:
        for proc in ("crossfit", "original_style"):
            x = dv[(dv.model.str.startswith(k)) & (dv.procedure == proc) & (dv.rule == "f1")]
            L.append(f"| {names[k]} | {proc} | {x.val_brier.mean():.3f} | {x.val_f1.mean():.3f} | {x.val_alert_rate.mean():.1%} |")
    L += ["", "Referências em `val`: Brier de prever só a prevalência = 0,129; F1 de alertar sempre = 0,263.\n",
          "Leitura: com a calibração fora da amostra, o Brier de RF e LSTM em `val` é **pior** que o de uma constante, e a regra F1 vira "
          "\"alertar quase sempre\" (limiar ≈ 0). O procedimento original parece bem melhor (Brier ≈ 0,12; F1 0,3–0,5), mas porque calibra e "
          "mede nos mesmos dados. As dobras de `val` têm regimes muito diferentes (110 dos 122 eventos estão em 2019‑20), então o calibrador "
          "treinado nas outras dobras não generaliza: isso é a mudança de regime, e não um defeito do cross‑fit.\n",
          "## 4. Teste (uma única avaliação por modelo/semente)\n",
          "AUC e Brier não dependem do limiar. Média ± desvio entre sementes (Perceptron: 1 execução):\n",
          "| modelo | AUC no teste | Brier no teste |\n|---|---|---|"]
    for k in kinds:
        x = mt[(mt.kind == k)].drop_duplicates("model")
        L.append(f"| {names[k]} | {ms(x.auc)} | {ms(x.brier)} |")
    L += ["", f"Referências: AUC de um acaso = 0,500; Brier prevendo a prevalência = {base['brier_constant_prevalence']:.3f}.\n",
          "Por semente (LSTM e RF):\n", "| modelo | AUC | Brier |\n|---|---|---|"]
    for m_, x in mt.drop_duplicates("model").iterrows():
        pass
    for r in mt.drop_duplicates("model").itertuples():
        L.append(f"| {r.model} | {r.auc:.3f} | {r.brier:.3f} |" if not np.isnan(r.brier) else f"| {r.model} | {r.auc:.3f} | – |")
    L += ["", "### Critério principal (cross‑fit + F1) e demais regras no teste\n",
          "Média entre sementes de precisão / recall / F1 / taxa de alerta:\n",
          "| modelo | procedimento | regra | precisão | recall | F1 | alerta |\n|---|---|---|---|---|---|---|"]
    for k in kinds:
        for (proc, rule), x in mt[mt.kind == k].groupby(["procedure", "rule"]):
            star = " **(principal)**" if (proc == fr["primary"]["procedure"] and rule == fr["primary"]["threshold_rule"]) else ""
            L.append(f"| {names[k]} | {proc} | {rule}{star} | {x.precision.mean():.3f} | {x.recall.mean():.3f} | {x.f1.mean():.3f} | {x.alert_rate.mean():.1%} |")
    L += ["",
          f"Linha de base \"alertar sempre\": precisão {base['precision_always_alert']:.3f}, recall 1,000, F1 {base['f1_always_alert']:.3f}, alerta 100 %.\n",
          "Tabela completa por semente: `outputs/tables/phase7_test_metrics.csv`; previsões salvas em `data/processed/test_predictions.csv` "
          "(a Fase 8 não recalcula nada).\n",
          "## 5. O que se pode e não se pode dizer\n",
          "- O AUC no teste fica entre ~0,3 e ~0,7 conforme a semente, em média próximo de 0,5; o Brier médio de todos os modelos é pior ou igual ao de "
          "prever a prevalência do teste (0,164; uma linha de base que na prática não se conhece de antemão), e só 3 de 11 execuções ficam abaixo, por menos de 0,002. **Com este alvo, esta divisão e estes dados, não há evidência de capacidade preditiva fora da amostra** "
          "para nenhum dos três modelos.\n"
          "- A regra principal (F1 com cross‑fit) resulta em alertar em quase todos os dias, ou seja, F1 ≈ o de \"alertar sempre\"; não é uma "
          "previsão, é uma carteira que fica no CDI. Isso é um achado, não foi \"corrigido\" depois de ver o teste.\n"
          "- Esses números **não reproduzem** os da monografia (RF F1 0,56; LSTM AUC 0,81). O período de teste é o mesmo (as últimas 815 amostras); "
          "a Fase 8 decompõe a diferença passo a passo e mostra que ela vem principalmente da troca do alvo (o alvo original compartilhava "
          "9 de 10 dias com as variáveis), e não das demais correções. Intervalos de confiança: bootstrap em blocos (Fase 8).\n"
          "- Limitações: poucos eventos (fit 80, val 122, teste 169), regime instável e janelas sobrepostas; os testes de significância "
          "devem ser lidos com essas ressalvas.\n"]
    (ROOT / "docs" / "calibration_threshold.md").write_text("\n".join(L), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
