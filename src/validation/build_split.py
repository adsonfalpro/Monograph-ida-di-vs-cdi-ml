"""Gera a divisão definitiva e docs/leakage_prevention.md."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("data", "targets", "features", "validation"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from build_target import HORIZON, build_target  # noqa: E402
from clean_base import build_clean_base, load_raw  # noqa: E402
from split import BLOCKS, make_split, split_table  # noqa: E402

LOOKBACK = 100


def load_samples():
    clean = build_clean_base(load_raw())
    tgt = build_target(clean)
    start = int(np.where(clean["in_original_sample"])[0][0])
    first_decision = start + LOOKBACK - 1
    rows = np.arange(first_decision, len(clean))
    rows = rows[tgt["y"].iloc[rows].notna().values]
    y = tgt["y"].iloc[rows].astype(int).values
    return clean, tgt, start, rows, y


def main() -> None:
    clean, tgt, start, rows, y = load_samples()
    split = make_split(rows, HORIZON)
    tab = split_table(split, clean["date"], y)
    # Alternativa descartada: val em duas metades contíguas
    v = tab[tab["block"] == "val"]
    HALF_CAL = int(v["y"].iloc[: len(v) // 2].sum())
    HALF_THR = int(v["y"].iloc[len(v) // 2:].sum())
    tab.to_csv(ROOT / "data" / "processed" / "split_indices.csv", index=False, date_format="%Y-%m-%d")

    stats = []
    for b in list(BLOCKS) + ["purge"]:
        t = tab[tab["block"] == b]
        stats.append({"bloco": b, "n": len(t), "primeira_decisao": t["date"].min().date(),
                      "ultima_decisao": t["date"].max().date(), "fim_alvo_max": t["target_end_date"].max().date(),
                      "positivos": int(t["y"].sum()), "taxa_positivos": t["y"].mean()})
    st = pd.DataFrame(stats)
    st.to_csv(ROOT / "outputs" / "tables" / "split_summary.csv", index=False)
    by = st.set_index("bloco")

    # Fronteiras do split original (60/20/20 sobre 4071 janelas, sem purge)
    n_orig = 4071
    a, b = int(n_orig * .6), int(n_orig * .8)
    cross_tv = int((np.arange(a - 12, a) + 10 > a).sum())
    cross_vt = int((np.arange(b - 12, b) + 10 > b).sum())
    n_total = len(clean)
    n_purge = int((tab["block"] == "purge").sum())
    usable = int((tab["block"] != "purge").sum())

    L = ["# Divisão temporal e prevenção de vazamento (Fase 5)\n",
         "Gerado por `src/validation/build_split.py`. Código: `src/validation/split.py`. Testes: `tests/test_split.py`. "
         "Índices por amostra: `data/processed/split_indices.csv`.\n",
         "## 1. Unidade e blocos\n",
         "Cada amostra tem data de decisão `t`: features = linhas `t−99..t` (100 dias); alvo = retornos `t+1..t+10`. "
         "A divisão é **cronológica, sem embaralhar**, na proporção da monografia (60/20/20):\n",
         "```\n| fit 60 % | val 20 % | test 20 % |\n```\n",
         "| bloco | uso |\n|---|---|",
         "| fit | treino dos modelos; ajuste do `StandardScaler` (linhas ≤ última decisão do fit) |",
         "| val | parada antecipada do LSTM, calibração e threshold, **separados por validação cruzada em blocos** (§2b; Fase 7) |",
         "| test | avaliação final, uma única vez por modelo (`TestSetGuard`) |\n",
         "## 2. Purge\n",
         "Uma amostra do bloco anterior é removida se o seu alvo usa retornos **posteriores** à primeira data de decisão do bloco seguinte "
         "(`target_end_row > t_próximo`). Com horizonte 10 isso remove **9 amostras** no fim de cada bloco seguido por outro "
         f"(fit e val). No split original (sem purge) havia 9 rótulos por fronteira olhando para dentro do bloco seguinte: "
         f"{cross_tv} em treino→validação e {cross_vt} em validação→teste (mais a fronteira interna que não existia).\n",
         "### 2b. Por que `val` não é dividido em duas metades\n",
         f"Testado: com duas metades contíguas, a metade de calibração ficaria com {HALF_CAL} eventos (2019‑20) e a de threshold com apenas "
         f"{HALF_THR} (2021‑jul/2022), insuficiente para escolher um limiar. Em vez disso, `blocked_folds(k=3)` gera três dobras contíguas de `val`; "
         "o calibrador de cada dobra é ajustado nas outras duas (com purge de 9 amostras de cada lado) e o threshold é escolhido sobre as "
         "probabilidades assim obtidas. O calibrador final (para o teste) usa `val` inteiro. A Fase 7 compara com o procedimento original "
         "(mesma validação para tudo), para mostrar quanto isso muda.\n",
         "Nota: as **features** de uma amostra de validação/teste incluem dias do período anterior. Isso não é vazamento "
         "(são dados passados, conhecidos em `t`); o risco está nos **rótulos** e nas estatísticas ajustadas, tratados aqui.\n",
         "## 3. Resultado\n",
         "| bloco | n | primeira decisão | última decisão | último dia usado no alvo | positivos | taxa |\n|---|---|---|---|---|---|---|"]
    for r in st.itertuples(index=False):
        L.append(f"| {r.bloco} | {r.n} | {r.primeira_decisao} | {r.ultima_decisao} | {r.fim_alvo_max} | {r.positivos} | {r.taxa_positivos:.1%} |")
    L += ["",
          "### Observações perdidas\n",
          "| etapa | linhas |\n|---|---|",
          f"| base bruta | {n_total} |",
          f"| linhas antes da amostra original (Duration ausente; só aquecimento) | {start} |",
          f"| primeiras {LOOKBACK-1} linhas da amostra (completar a janela de {LOOKBACK} dias) | {LOOKBACK-1} |",
          f"| últimas {HORIZON} linhas (sem alvo) | {HORIZON} |",
          f"| amostras disponíveis | {len(tab)} |",
          f"| removidas por purge | {n_purge} |",
          f"| **amostras utilizáveis** | **{usable}** |\n",
          "## 4. Vazamentos do original e situação agora\n",
          "| # | Vazamento no original | Agora |\n|---|---|---|",
          "| L1 | `StandardScaler` ajustado na base inteira | ajustado só com linhas ≤ última decisão do `fit` (Fase 4) |",
          "| L2 | Sem purge: 9 rótulos por fronteira usavam retornos do bloco seguinte | purge automático (18 amostras) |",
          "| L3 | RF: calibrador isotônico e threshold na **mesma** validação | calibração e threshold por validação cruzada em blocos (Fase 7) |",
          "| L4 | LSTM: parada antecipada, calibração e threshold na mesma validação | parada antecipada em `val`; calibração/threshold como em L3 |",
          "| L5 | Backtest com alertas em todo o período (≈ 80 % in‑sample) | o backtest (Fase 9) usará só o bloco `test` |",
          "| L6 | Nenhum controle de acesso ao teste | `TestSetGuard`: uma liberação por modelo, em `outputs/metrics/test_access_log.json` |\n",
          "## 5. Riscos que permanecem (não são vazamento, mas limitam as conclusões)\n",
          f"- **Poucos positivos por bloco**: fit {int(by.loc['fit','positivos'])}, val {int(by.loc['val','positivos'])}, "
          f"test {int(by.loc['test','positivos'])}. As métricas do teste terão intervalos largos (Fase 8).",
          f"- **Mudança de regime**: a taxa de positivos vai de {by.loc['fit','taxa_positivos']:.1%} (fit) a "
          f"{by.loc['test','taxa_positivos']:.1%} (teste). Calibração e threshold aprendidos em um regime podem não valer no seguinte; "
          "é uma limitação do problema, não do código.",
          "- **Autocorrelação**: janelas consecutivas compartilham 99 de 100 dias e rótulos se sobrepõem; o número efetivo de observações "
          "independentes é bem menor que n. Intervalos de confiança devem usar bootstrap em blocos (Fase 8).",
          "- Uma única divisão: a Fase 8 poderá incluir avaliação em janelas expansivas (walk‑forward), separada e rotulada.\n"]
    (ROOT / "docs" / "leakage_prevention.md").write_text("\n".join(L), encoding="utf-8")
    print(st.to_string())
    print(cross_tv, cross_vt, "usable", usable, "purge", n_purge)


if __name__ == "__main__":
    main()
