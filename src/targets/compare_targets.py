"""Compara o alvo original (max de 10 dias) com a definição formal. Gera docs/target_definition.md."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_target import HORIZON, build_target, original_target  # noqa: E402
from clean_base import build_clean_base, load_raw  # noqa: E402

LOOKBACK = 100


def main() -> None:
    clean = build_clean_base(load_raw())
    full_tgt = build_target(clean)
    sample = clean.loc[clean["in_original_sample"]].reset_index(drop=True)  # 4181 linhas = base da monografia
    tgt = build_target(sample)
    orig = original_target(sample, LOOKBACK, HORIZON)
    comp = orig.merge(tgt[["date", "y", "fwd_ida", "fwd_cdi", "fwd_excess"]], on="date", how="left")
    comp = comp.rename(columns={"y": "y_new"})
    assert comp["y_new"].notna().all()
    comp["y_new"] = comp["y_new"].astype(int)
    n = len(comp)
    tr, va = int(n * 0.6), int(n * 0.8)
    comp["split"] = np.where(np.arange(n) < tr, "treino", np.where(np.arange(n) < va, "validacao", "teste"))
    comp["year"] = comp["date"].dt.year

    ct = pd.crosstab(comp["y_original"], comp["y_new"], rownames=["original"], colnames=["novo"])
    agree = float((comp["y_original"] == comp["y_new"]).mean())
    by_split = comp.groupby("split")[["y_original", "y_new"]].mean().loc[["treino", "validacao", "teste"]]
    by_year = comp.groupby("year")[["y_original", "y_new"]].mean()
    counts = comp.groupby("split").size().loc[["treino", "validacao", "teste"]]
    # Sanidade: o alvo novo equivale ao de M.py (retorno retroativo de 10d terminando em t+10)?
    m_style = (sample["ida_trail_10"] < sample["cdi_trail_10"]).astype(int).reset_index(drop=True)
    idx = comp["decision_idx"] + HORIZON
    m_equal = bool((m_style.iloc[idx].values == comp["y_new"].values).all())

    comp.to_csv(ROOT / "outputs" / "tables" / "target_comparison_by_window.csv", index=False)
    by_split.to_csv(ROOT / "outputs" / "tables" / "target_rate_by_split.csv")
    by_year.to_csv(ROOT / "outputs" / "tables" / "target_rate_by_year.csv")
    # Alvo completo (inclui as 59 linhas de aquecimento) para uso nas fases seguintes
    full_tgt.to_csv(ROOT / "data" / "processed" / "target_10d.csv", index=False, date_format="%Y-%m-%d")

    only_old = int(((comp.y_original == 1) & (comp.y_new == 0)).sum())
    only_new = int(((comp.y_original == 0) & (comp.y_new == 1)).sum())
    both = int(((comp.y_original == 1) & (comp.y_new == 1)).sum())
    lines = ["# Definição da variável‑alvo (Fase 3)\n",
             "Gerado por `src/targets/compare_targets.py`. Código: `src/targets/build_target.py`. Testes: `tests/test_target.py`.\n",
             "## 1. Definição adotada\n",
             "Com `t` = última data observada (data de decisão) e `h = 10` dias úteis:\n",
             "```\nR_IDA(t,t+h) = Π_{i=1..h}(1 + r_IDA[t+i]) − 1\nR_CDI(t,t+h) = Π_{i=1..h}(1 + r_CDI[t+i]) − 1\n"
             "Y_t = 1  se R_IDA(t,t+h) < R_CDI(t,t+h);   Y_t = 0  caso contrário (empate = 0)\n```\n",
             "- Só entram retornos de `t+1 … t+h`. As últimas `h` linhas não têm alvo (NaN, nunca preenchido).\n"
             "- `r_IDA` = `ida_ret` (decisão: série da monografia) e `r_CDI` = `cdi_ret`, ambos em decimal.\n"
             "- A coluna `target_end_idx` guarda a linha do último retorno usado pelo alvo, para o purge da Fase 5.\n"
             f"- É equivalente à convenção de `M.py` (retorno retroativo de 10 dias terminando em `t+10`): `{m_equal}`.\n"
             "- Usar a forma simples (`R_IDA < R_CDI`) ou o excesso geométrico `(1+R_IDA)/(1+R_CDI) − 1 < 0` dá o mesmo rótulo.\n",
             "## 2. Alvo original (scripts finais)\n",
             "```\ndrawdown[t] = 1[ IDA_10_dias[t] < CDI_10_dias[t] ]      # janela retroativa t-9..t\n"
             "janela i = linhas i..i+99;   y_i = max(drawdown[i+100 .. i+109])\n```\n",
             "Os 10 rótulos `drawdown[i+100..i+109]` usam janelas retroativas que cobrem os dias `i+91 … i+109`. Portanto, "
             "**os dias `i+91 … i+99` (9 dias) já estão na janela de features** e só `i+100 … i+109` são futuros. Além disso, o `max` "
             "marca 1 se **qualquer** das 10 janelas deslizantes for negativa. Não é \"retorno dos próximos 10 dias\". "
             "Como cada janela contém dias do passado, o alvo original é um **superconjunto** do novo: nunca há `novo=1` com `original=0` (ver tabela).\n",
             "## 3. Diferença medida (mesmas 4071 janelas, split 60/20/20 de referência)\n",
             f"Concordância entre os dois alvos: **{agree:.1%}**.\n",
             "| original \\ novo | 0 | 1 |\n|---|---|---|",
             f"| 0 | {ct.loc[0,0]} | {ct.loc[0,1]} |",
             f"| 1 | {ct.loc[1,0]} | {ct.loc[1,1]} |\n",
             f"Positivos só no original: {only_old}; só no novo: {only_new}; em ambos: {both}.\n",
             "Taxa de positivos por conjunto (split 60/20/20 apenas para comparação; a divisão definitiva é a da Fase 5):\n",
             "| conjunto | n | original | novo |\n|---|---|---|---|"]
    for s in ["treino", "validacao", "teste"]:
        lines.append(f"| {s} | {counts[s]} | {by_split.loc[s,'y_original']:.1%} | {by_split.loc[s,'y_new']:.1%} |")
    lines += ["", "Taxa por ano:\n", "| ano | original | novo |\n|---|---|---|"]
    for y_, r in by_year.iterrows():
        lines.append(f"| {y_} | {r['y_original']:.1%} | {r['y_new']:.1%} |")
    lines += ["",
              "## 4. O que isto implica\n",
              "- Os resultados da monografia (RF, LSTM, Perceptron) foram treinados e avaliados com o alvo original; **não são "
              "diretamente comparáveis** aos da versão corrigida. A diferença entre eles passa a ser parte da análise (Fase 8).\n"
              "- A prevalência do alvo muda muito com o regime (veja a tabela por ano): é um problema de não estacionariedade que "
              "a divisão temporal e a escolha do threshold terão de tratar; não será \"corrigido\" ajustando o alvo.\n",
              "## 5. Testes de alinhamento temporal (`tests/test_target.py`)\n",
              "1. Alvo = laço explícito de referência em todas as linhas.\n"
              "2. Alterar `r[t+1]` ou `r[t+h]` pode mudar `Y_t`; alterar `r[t]` ou `r[t+h+1]` **não** muda.\n"
              "3. Últimas `h` linhas sem alvo; nenhuma outra linha sem alvo.\n"
              "4. Empate ⇒ 0.\n"
              "5. `target_end_idx = t + h`.\n"
              "6. O alvo original é reproduzido pelo código de comparação e coincide com o do `drawdown` retroativo.\n"
              "7. O alvo novo coincide com o rótulo `IDA_10 < CDI_10` da linha `t+10`.\n"]
    (ROOT / "docs" / "target_definition.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"agreement={agree:.4f} m_equal={m_equal}")
    print(by_split)
    print(ct)


if __name__ == "__main__":
    main()
