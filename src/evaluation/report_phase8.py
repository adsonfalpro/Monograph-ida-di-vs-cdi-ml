"""Gera docs/evaluation_report.md (Fase 8)."""
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

TAB = ROOT / "outputs" / "tables"
NAMES = {"perceptron": "Perceptron", "rf": "Random Forest", "lstm": "LSTM"}
STEP_LABEL = {"S0_original": "S0 original (literal)", "S1_alvo": "S1 + alvo formal", "S2_scaler": "S2 + scaler só no treino",
              "S3_variaveis": "S3 + variáveis corrigidas", "S4_purge": "S4 + purge", "S5_crossfit": "S5 + cross-fit (4071 janelas, retreinado)"}


def load_cache() -> pd.DataFrame:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "outputs" / "metrics" / "ablation_cache").glob("*.json"))]
    return pd.DataFrame(rows)


def ms(x, d=3):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(x) == 0:
        return "–"
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x[0]:.{d}f}"


def main() -> None:
    ab = load_cache()
    ab["auc_report"] = ab["auc_cal"].where(ab["auc_cal"].notna(), ab["auc"])   # como a monografia: AUC sobre a probabilidade calibrada
    ci = pd.read_csv(TAB / "phase8_metrics_ci.csv")
    pair = pd.read_csv(TAB / "phase8_pairwise_auc.csv")
    base = json.loads((ROOT / "outputs" / "metrics" / "test_baselines.json").read_text(encoding="utf-8"))
    p7 = pd.read_csv(TAB / "phase7_test_metrics.csv")

    L = ["# Avaliação e comparação dos modelos (Fase 8)\n",
         "Gerado por `src/evaluation/report_phase8.py`. Figuras em `outputs/figures/`. Tudo é calculado sobre as previsões do teste "
         "salvas na Fase 7 (`data/processed/test_predictions.csv`); nada foi retreinado nem reajustado. A decomposição da §4 retreina "
         "pipelines intermediários, mas **só mede o teste depois de cada pipeline estar definido**; ela não escolhe nada.\n",
         "## 1. Resultado principal (pipeline corrigido, teste = 815 amostras, 169 positivos)\n",
         "Critério principal pré‑declarado: calibração por cross‑fit + limiar de F1. Média entre sementes (RF e LSTM: 5; Perceptron: 1) "
         "e IC 95 % por bootstrap em blocos móveis de 20 dias (1000 reamostragens, mesmas datas para todas as sementes).\n",
         "| modelo | AUC | Brier | F1 (regra principal) | P(AUC ≤ 0,5) |\n|---|---|---|---|---|"]
    for m in NAMES.values():
        r = {k: ci[(ci.model == m) & (ci.metric == k)].iloc[0] for k in ("auc", "brier", "f1")}
        f = lambda x: f"{x.point:.3f} [{x.ci_low:.3f}; {x.ci_high:.3f}]"
        L.append(f"| {m} | {f(r['auc'])} | {f(r['brier'])} | {f(r['f1'])} | {r['auc']['p_auc_le_0.5']:.2f} |")
    L += ["", f"Referências no teste: AUC de um acaso 0,500; Brier prevendo a prevalência {base['brier_constant_prevalence']:.3f}; "
          f"F1 de alertar sempre {base['f1_always_alert']:.3f} (precisão {base['precision_always_alert']:.3f}, recall 1). "
          "(A prevalência do teste só é conhecida depois; é uma linha de base ideal.)\n",
          "Diferenças de AUC entre modelos:\n", "| comparação | diferença [IC 95 %] |\n|---|---|"]
    for r in pair.itertuples():
        L.append(f"| {r.diff} | {r.point:+.3f} [{r.ci_low:+.3f}; {r.ci_high:+.3f}] |")
    L += ["", "**Leitura.** Todos os intervalos de AUC contêm 0,5, nenhum Brier é distinguível da previsão constante e o F1 de RF e LSTM "
          "coincide com o de alertar sempre (a regra principal degenera). **Nenhum modelo se distingue do acaso**. Entre os modelos, "
          "a única diferença com intervalo fora de zero é RF − Perceptron (+0,133; IC 95 %: +0,009 a +0,269), que vem em boa parte de o Perceptron ficar abaixo de 0,50 "
          "no ponto e é evidência fraca (três comparações, sem correção para múltiplas comparações); a LSTM não se distingue de nenhum dos dois. Tudo isso com este teste (169 eventos, janelas muito autocorrelacionadas). Isso é ausência de evidência, não evidência de que os "
          "modelos sejam inúteis; o poder do teste é baixo.\n",
          "Figuras (mesmo conjunto de teste e critérios para os três modelos): `confusion_counts.png`, `confusion_normalized.png` "
          "(semente 42, escolhida de antemão), `roc_curve.png`, `pr_curve.png`, `calibration_curve.png` (todas as sementes), `real_vs_predicted.png`, "
          "`metrics_with_ci.png`.\n",
          "## 2. Outras regras de decisão no teste (média entre sementes)\n",
          "| modelo | procedimento | regra | precisão | recall | F1 | alerta |\n|---|---|---|---|---|---|---|"]
    for k, nm in NAMES.items():
        for (proc, rule), x in p7[p7.kind == k].groupby(["procedure", "rule"]):
            L.append(f"| {nm} | {proc} | {rule} | {x.precision.mean():.3f} | {x.recall.mean():.3f} | {x.f1.mean():.3f} | {x.alert_rate.mean():.0%} |")
    L += ["", "O limiar 0,50 nunca emite alerta (probabilidades calibradas não chegam a 0,5 num problema com prevalência de 3–20 %).\n",
          "## 3. Reconciliação com os números da monografia\n",
          "O passo S0 executa o código original (literal) sobre `Monografia - Dados.xlsx`, para saber o que da monografia é reproduzível.\n",
          "| item | monografia | S0 reproduzido |\n|---|---|---|"]
    s0 = ab[ab.step == "S0_original"]
    rf0, ls0, pe0 = s0[s0.kind == "rf"], s0[s0.kind == "lstm"], s0[s0.kind == "perceptron"]
    cm = lambda d: f"{d.tn.mean():.0f} / {d.fp.mean():.0f} / {d.fn.mean():.0f} / {d.tp.mean():.0f}"
    L += [f"| RF: matriz (VN/FP/FN/VP) | 319 / 245 / 57 / 194 | {cm(rf0)} (média de {len(rf0)} sementes) |",
          f"| RF: F1 | 0,562 | {ms(rf0.f1)} |", f"| RF: AUC (prob. calibrada) | 0,737 | {ms(rf0.auc_report)} |",
          f"| RF: Brier | 0,175 | {ms(rf0.brier)} |", f"| RF: limiar F1 | 0,237 | {ms(rf0.threshold)} |",
          f"| LSTM: matriz | 562 / 2 / 190 / 61 | {cm(ls0)} (média de {len(ls0)} sementes) |",
          f"| LSTM: F1 | 0,389 | {ms(ls0.f1)} |", f"| LSTM: AUC | 0,810 | {ms(ls0.auc_report)} |",
          f"| LSTM: Brier | 0,198 | {ms(ls0.brier)} |", f"| LSTM: limiar F1 | 0,269 | {ms(ls0.threshold)} |",
          f"| Perceptron: matriz | 390 / 111 / 174 / 140 (314 positivos) | {cm(pe0)} ({int(pe0.pos_test.iloc[0])} positivos) |", "",
          "- **RF: reproduzido aproximadamente** (mesma ordem de grandeza em F1, AUC e matriz, mesma prevalência de teste de 251 positivos). A diferença "
          "restante é compatível com a versão do scikit‑learn e com a base `IDA_DI.xlsx`, que não temos.\n"
          "- **LSTM: reprodução apenas qualitativa.** O resultado da monografia (AUC 0,81, recall 24 %, 2 falsos positivos) é uma única execução "
          "(TensorFlow/semente não registrados); aqui o resultado varia entre sementes (ver a dispersão acima). O valor publicado não pode ser confirmado nem refutado.\n"
          "- **Perceptron: não reproduzido.** A figura da monografia tem 314 positivos no teste; o código `Perceptron.py` com os dados disponíveis gera "
          f"{int(pe0.pos_test.iloc[0])}, igual ao RF e ao LSTM. Testei 168 combinações de coluna/horizonte/janela do alvo e nenhuma produz 314. "
          "A origem dessa figura permanece desconhecida e fica registrada como tal.\n",
          "## 4. De onde vem a diferença: decomposição passo a passo\n",
          "Cada linha acrescenta **uma** correção à anterior (mesmas 4071 janelas do original). Média entre sementes (RF: 5, LSTM: 3, Perceptron: 1) "
          "de AUC sobre a probabilidade calibrada (score bruto no Perceptron), Brier, F1 e prevalência no teste. Os passos S0–S4 usam a regra do original "
          "(calibração e limiar F1 na mesma validação, alerta se `p > limiar`); S5 usa cross‑fit.\n"]
    for k, nm in NAMES.items():
        d = ab[ab.kind == k]
        L += [f"### {nm}\n", "| passo | AUC | Brier | precisão | recall | F1 | positivos no teste |\n|---|---|---|---|---|---|---|"]
        for st in STEPS:
            x = d[d.step == st]
            if x.empty:
                continue
            L.append(f"| {STEP_LABEL[st]} | {ms(x.auc_report)} | {ms(x.brier)} | {ms(x.precision)} | {ms(x.recall)} | {ms(x.f1)} | {int(x.pos_test.iloc[0])} |")
        L.append("")
    L += ["Linha de base 'alertar sempre' em cada passo: F1 = 2p/(1+p) com p = prevalência do teste (0,308 → 0,471; 0,207 → 0,343).\n"]
    def auc_of(kind, step):
        return ab[(ab.kind == kind) & (ab.step == step)].auc_report.mean()

    L += ["## 5. Conclusões da Fase 8\n",
          f"1. **A maior parte da diferença vem do alvo.** Só trocar o alvo original pelo formal derruba o AUC de {auc_of('rf','S0_original'):.2f} para "
          f"{auc_of('rf','S1_alvo'):.2f} no RF, de {auc_of('lstm','S0_original'):.2f} para {auc_of('lstm','S1_alvo'):.2f} no LSTM e de "
          f"{auc_of('perceptron','S0_original'):.2f} para {auc_of('perceptron','S1_alvo'):.2f} no Perceptron. O alvo original marcava 1 se **qualquer** das 10 "
          "janelas deslizantes fosse negativa, e 9 dos 10 dias de cada janela já estavam nas variáveis de entrada (Fase 3): parte do rótulo "
          "era informação passada que o modelo já via. Isso, e não a previsão do futuro, explica a maior parte do desempenho da monografia.\n",
          "2. **As demais correções (scaler só no treino, variáveis, purge, cross‑fit) movem o AUC bem menos que a troca do alvo**, e as mudanças "
          "(até cerca de 0,05 no RF e 0,10 no LSTM entre passos) são da ordem do desvio‑padrão entre sementes (RF ≈ 0,03; LSTM 0,09–0,18). "
          "Não fiz um teste formal de cada passo; a leitura é que nenhuma dessas correções, isolada, explica a queda. O cross‑fit muda a regra de decisão "
          "(F1 → alertar sempre), não a capacidade de ordenar.\n"
          "   *Nota sobre S5:* não é idêntico ao pipeline final da Fase 7. O S5 retreina os modelos com 4071 janelas (as do original), e o AUC da tabela é "
          "medido sobre a probabilidade calibrada; a Fase 7 usa 4072 janelas e o score bruto (RF 0,54 e LSTM 0,54). A distância entre os dois (RF 0,58 contra 0,54) "
          "mostra o quanto o AUC varia com pequenas mudanças do pipeline, e reforça que as diferenças entre S2 e S5 estão dentro do ruído.\n"
          "   *Nota sobre o teste:* o teste só foi aberto uma vez por modelo congelado (Fase 7). Os pipelines retreinados nesta decomposição também foram medidos no teste, "
          "apenas como diagnóstico; nenhuma escolha foi feita com base neles.\n",
          "3. **Com o alvo formal, nenhum modelo se distingue do acaso** (§1). O Perceptron em particular fica abaixo de 0,5 no ponto, sem significância.\n",
          "4. **O que isso muda para a monografia:** o resultado \"RF e LSTM superam IDA e CDI\" depende de um alvo cujo desempenho vem em grande parte da "
          "sobreposição com as variáveis e de um backtest majoritariamente in‑sample (Fase 9 mede isso). A conclusão correta, com os dados e o método "
          "corrigidos, é **ausência de evidência de previsibilidade** de um desempenho do IDA‑DI abaixo do CDI em 10 dias com estas variáveis.\n",
          "5. **Limites desta conclusão:** teste com 169 eventos e janelas autocorrelacionadas (poder baixo); uma única divisão cronológica; "
          "hiperparâmetros da monografia sem busca; LSTM com 3–5 sementes; o CDI diário da base foi conferido pelo autor do projeto. "
          "Sugestões, não executadas aqui para não misturar com a reconstrução fiel: walk‑forward, outros horizontes, alvo contínuo (excesso de retorno).\n"]
    (ROOT / "docs" / "evaluation_report.md").write_text("\n".join(L), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
