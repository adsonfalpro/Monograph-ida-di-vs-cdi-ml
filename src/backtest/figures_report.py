"""Figuras e docs/backtest_report.md (Fase 9)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = ROOT / "outputs" / "figures"
T = ROOT / "outputs" / "tables"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RULE_COL = {"principal": "#2a78d6", "valor": "#eb6834", "estilo_original": "#1baf7a"}
RULE_LAB = {"principal": "Regra principal (cross-fit + F1)", "valor": "Regra de valor (cross-fit)", "estilo_original": "Estilo original (mesma validação)"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False, "savefig.dpi": 150})


def fig_curves() -> None:
    z = np.load(ROOT / "data" / "processed" / "phase9_curves.npz", allow_pickle=True)
    dates = pd.to_datetime(z["dates"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (kind, title) in zip(axes, [("rf", "Random Forest"), ("lstm", "LSTM")]):
        ax.plot(dates, z["ida"], color=INK2, lw=2, label="IDA (comprar e manter)")
        ax.plot(dates, z["cdi"], color="#8a8984", lw=1.6, ls="--", label="CDI 100 %")
        for rule in ("principal", "valor", "estilo_original"):
            cs = [z[k] for k in z.files if k.startswith(f"{kind}|") and k.endswith(f"|{rule}")]
            ax.plot(dates, np.mean(cs, axis=0), color=RULE_COL[rule], lw=1.4, label=RULE_LAB[rule])
        ax.set_title(f"{title} (média das sementes, sem custos)", fontsize=10, color=INK)
        ax.set_ylabel("Base 100")
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Backtest apenas no período de teste (decisão em t, posição no dia t+1)", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "backtest_curves.png"); plt.close(fig)


def fig_original() -> None:
    d = pd.read_csv(T / "phase9_original_backtest_in_out_sample.csv")
    d = d[~d.trecho.str.startswith("conferência")]
    order = ["treino (in-sample)", "validação (usada para limiar)", "teste original (fora da amostra)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, name in zip(axes, ["Random Forest", "LSTM"]):
        x = d[(d.modelo == name) & (d.trecho.isin(order))].set_index("trecho").loc[order]
        bars = ax.bar(range(3), x["ganho_vs_ida_pp"], color=["#8a8984", "#8a8984", "#2a78d6"][:3], width=0.6)
        for i, v in enumerate(x["ganho_vs_ida_pp"]):
            ax.text(i, v + (0.15 if v >= 0 else -0.5), f"{v:+.1f}".replace(".", ","), ha="center", color=INK, fontsize=10)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(3), ["Treino", "Validação", "Teste"]); ax.set_title(name, fontsize=10, color=INK)
    axes[0].set_ylabel("Ganho sobre o IDA no trecho\n(pontos percentuais)")
    fig.suptitle("Backtest original (alertas dos scripts da monografia): onde o ganho sobre o IDA aparece", color=INK)
    fig.tight_layout(); fig.savefig(FIG / "backtest_original_split.png"); plt.close(fig)


def f(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def report() -> None:
    s = pd.read_csv(T / "phase9_backtest_summary.csv")
    b = pd.read_csv(T / "phase9_benchmarks.csv", index_col=0)
    ci = pd.read_csv(T / "phase9_excess_ci.csv")
    o = pd.read_csv(T / "phase9_original_backtest_in_out_sample.csv")
    ck = pd.read_csv(T / "phase9_excel_check.csv")
    names = {"rf": "Random Forest", "lstm": "LSTM", "perceptron": "Perceptron (extra)"}
    L = ["# Backtest (Fase 9)\n",
         "Gerado por `src/backtest/figures_report.py`. Código: `src/backtest/`. Testes: `tests/test_backtest.py`. Planilha de conferência: `outputs/backtest_check.xlsx`.\n",
         "## 1. Regras de execução\n",
         "- **Período:** apenas o bloco de teste (815 dias; decisões de 2022‑08‑10 a 2025‑11‑05, retornos de 2022‑08‑11 a 2025‑11‑06).",
         "- **Sinal:** a decisão em `t` usa informação até o fechamento de `t`; a posição vale para o retorno de `t+1`. Alerta → CDI; sem alerta → IDA. Reavaliado todo dia.",
         "- **Custos:** 0, 10 e 25 bps por troca de posição (suposição minha, descontada no dia da troca). O IDA é uma carteira teórica, não negociável; "
         "esses custos são um proxy, não uma medida de mercado. Impostos não considerados.",
         "- **Sinais avaliados (rotulados, nenhum escolhido depois de ver o resultado):** `principal` (cross‑fit + F1, pré‑declarada), `valor` (cross‑fit, regra de valor) "
         "e `estilo_original` (calibração e limiar na mesma validação, como na monografia). RF e LSTM: 5 sementes; Perceptron: 1 (extra, a monografia o descartou).",
         "- **Retorno do IDA:** coluna `ida_ret` (decisão da Fase 2). Alternativa pelo nível do índice na tabela de benchmarks.\n",
         "## 2. Resultado no teste (média das sementes, sem custos)\n",
         "Benchmarks no mesmo intervalo:\n",
         "| | base 100 final | retorno anual | vol. anual | máx. drawdown |\n|---|---|---|---|---|"]
    for k, r in b.iterrows():
        L.append(f"| {k} | {f(r.final_base100)} | {f(r.retorno_anual*100)} % | {f(r.vol_anual*100,2)} % | {f(r.max_drawdown*100,2)} % |")
    L += ["", "Estratégias (custo 0):\n",
          "| modelo | regra | base 100 final (mín.–máx. entre sementes) | trocas (média) | dias no CDI | excesso anual vs CDI [IC 95 %] | excesso anual vs IDA [IC 95 %] |\n|---|---|---|---|---|---|---|"]
    for r in s[s.custo_bps == 0].itertuples():
        c = ci[(ci.tipo == r.tipo) & (ci.regra == r.regra)].iloc[0]
        L.append(f"| {names[r.tipo]} | {r.regra} | {f(r.final_media)} ({f(r.final_min)}–{f(r.final_max)}) | {f(r.trocas_media)} | {r.dias_cdi:.0%} | "
                 f"{c.exc_cdi_anual*100:+.2f} pp [{c.ic_cdi_baixo*100:+.2f}; {c.ic_cdi_alto*100:+.2f}] | {c.exc_ida_anual*100:+.2f} pp [{c.ic_ida_baixo*100:+.2f}; {c.ic_ida_alto*100:+.2f}] |")
    L += ["", "IC 95 % por bootstrap em blocos de 20 dias sobre o excesso diário médio entre sementes, anualizado (252 dias).\n",
          "### Sensibilidade a custos (base 100 final, média das sementes)\n",
          "| modelo | regra | 0 bps | 10 bps | 25 bps | trocas |\n|---|---|---|---|---|---|"]
    for (t, rg), g in s.groupby(["tipo", "regra"]):
        g = g.set_index("custo_bps")
        L.append(f"| {names[t]} | {rg} | {f(g.loc[0,'final_media'])} | {f(g.loc[10,'final_media'])} | {f(g.loc[25,'final_media'])} | {f(g.loc[0,'trocas_media'])} |")
    L += ["", "Figuras: `outputs/figures/backtest_curves.png` e `backtest_original_split.png`.\n",
          "## 3. O que o backtest mostra\n",
          "1. **No período de teste o IDA rendeu mais que o CDI** (veja os benchmarks). Qualquer estratégia que fique no IDA 'supera o CDI' sem prever nada. "
          "Por isso a comparação relevante é com o IDA, não com o CDI.\n"
          "2. **A regra principal fica 100 % no CDI** em RF e LSTM (um único alerta no início) e, por construção, perde para o IDA no teste.\n"
          "3. **A regra de valor** praticamente nunca alerta e reproduz o IDA (no RF, nenhuma troca).\n"
          "4. **O estilo original** troca de posição com frequência (dezenas de vezes no RF). Antes de custos o resultado sobre o IDA é pequeno (negativo no RF, positivo no LSTM) e o IC inclui zero; "
          "com 10 bps por troca o RF fica abaixo do IDA e o LSTM empata, e com 25 bps os dois ficam abaixo, principalmente o RF e o Perceptron.\n"
          "5. **Nenhuma estratégia baseada em modelo supera o IDA de forma estatisticamente distinguível** no teste, com ou sem custos. É coerente com o AUC próximo de 0,5 da Fase 8.\n",
          "## 4. O backtest original da monografia, decomposto\n",
          "Recalculei a Fig. 7 a partir dos alertas salvos pelos scripts originais. A série reproduz **exatamente** as colunas do Excel "
          "(RF 570,9 e LSTM 572,8 na base 100; IDA 534,1; CDI 443,4). O gráfico da monografia cobre o período inteiro, que inclui treino e validação (usados para ajustar o modelo e o limiar).\n",
          "| modelo | trecho | dias | estratégia | IDA | CDI | ganho vs IDA (pp) | taxa de alerta | trocas |\n|---|---|---|---|---|---|---|---|---|"]
    for r in o[~o.trecho.str.startswith("conferência")].itertuples():
        L.append(f"| {r.modelo} | {r.trecho} | {r.dias} | {f(r.estrategia_base100)} | {f(r.ida_base100)} | {f(r.cdi_base100)} | {r.ganho_vs_ida_pp:+.1f} | {r.taxa_alerta:.0%} | {int(r.trocas)} |")
    L += ["", "Leitura: o ganho sobre o IDA do período inteiro vem em boa parte da **validação** (2019‑2022, que contém a crise de 2020 e foi usada para escolher o limiar). "
          "No trecho de treino o ganho é próximo de zero (ou negativo, no LSTM) e no teste original é pequeno (RF) ou depende de poucas trocas (LSTM, 6 trocas, uma única execução). "
          "O ganho sobre o **CDI** existe em todos os trechos porque, nesses anos, o IDA rendeu mais que o CDI.\n",
          "## 5. Conferência independente (planilha)\n",
          "`outputs/backtest_check.xlsx` reproduz três estratégias (RF e LSTM no estilo original e LSTM de valor, semente 42, custo 10 bps) com fórmulas do Excel: troca, retorno, base 100, pico e drawdown, mais uma coluna de conferência contra o Python. "
          "O custo é editável em `Parametros!B1` (as conferências só valem com 10 bps).\n",
          "Recalculei as fórmulas com um motor independente (biblioteca `formulas`, não o Excel): diferença máxima em relação ao Python = "
          f"{ck.dif_abs.max():.1e} (base 100 final, trocas, drawdown e 815 de 815 conferências VERDADEIRO por estratégia). **O Excel em si não foi aberto aqui**; ao abrir a planilha, a coluna de conferência deve mostrar VERDADEIRO.\n",
          "## 6. Limitações\n",
          "- Custos de 0/10/25 bps são suposições; o IDA não é negociável e não há spread de debêntures real. Impostos e prazos de liquidação ignorados.\n"
          "- Um único período de teste (815 dias), em que o IDA superou o CDI; o resultado dependeria do regime.\n"
          "- A comparação 'original × corrigido' no mesmo teste usa pipelines diferentes (alvo e vazamentos), então o trecho de teste original da §4 não é diretamente comparável à §2.\n"]
    (ROOT / "docs" / "backtest_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    fig_curves(); fig_original(); report(); print("ok")
