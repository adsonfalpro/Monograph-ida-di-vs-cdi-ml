"""Preparação e verificação da base (Fase 2). Gera docs/data_quality_report.md e outputs/tables/*.csv.

Uso:  python -I src/data/audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_base import (  # noqa: E402
    HORIZONS, ROOT, build_clean_base, load_monograph, load_raw, save_clean_base, trailing_compound,
)

TABLES = ROOT / "outputs" / "tables"
REPORT = ROOT / "docs" / "data_quality_report.md"
TOL = 1e-12


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    mono = load_monograph()
    clean = build_clean_base(raw)
    save_clean_base(clean)
    lines: list[str] = []
    add = lines.append
    res: dict[str, object] = {}

    # 1) Estrutura -----------------------------------------------------------------------------
    d = clean["date"]
    res["rows_raw"], res["rows_mono"] = len(raw), len(mono)
    res["date_min"], res["date_max"] = d.min().date(), d.max().date()
    res["dup_dates"] = int(d.duplicated().sum())
    res["sorted"] = bool(d.is_monotonic_increasing)
    res["weekend"] = int((d.dt.dayofweek >= 5).sum())
    gaps = d.diff().dt.days
    res["max_gap_days"] = int(gaps.max())
    res["ibov_text_rows"] = int(clean["ibov_from_text"].sum())
    res["ibov_text_last"] = clean.loc[clean["ibov_from_text"], "date"].max().date()
    res["duration_nan"] = int(clean["duration"].isna().sum())
    res["duration_nan_last"] = clean.loc[~clean["has_duration"], "date"].max().date()
    nan_tbl = clean.isna().sum().rename("n_nan").to_frame()
    nan_tbl["n_nan_original_sample"] = clean.loc[clean["in_original_sample"]].isna().sum()
    nan_tbl.to_csv(TABLES / "missing_values.csv")

    # 2) Reconciliação bruta x monografia -------------------------------------------------------
    m = clean.merge(mono, on="date", how="inner", suffixes=("", "_m"))
    res["rows_common"] = len(m)
    res["dates_equal_original_sample"] = bool(
        (clean.loc[clean["in_original_sample"], "date"].reset_index(drop=True) == mono["date"]).all()
    )
    rec = []
    for c_clean, c_mono, scale in [
        ("ida_index", "ida_index_m", 1.0), ("ida_ret", "ida_ret_m", 1.0), ("duration", "duration_m", 1.0),
        ("ipca_deb_ret", "ipca_deb_pct", 0.01), ("ibov_ret", "ibov_ret_m", 1.0), ("cdi_ret", "cdi_ret_m", 1.0),
    ]:
        diff = (m[c_clean] - m[c_mono] * scale).abs().max()
        rec.append((c_clean, c_mono, diff))
    for h in HORIZONS:
        for kind, a, b in [("ida", f"ida_trail_{h}", f"ida_{h}"), ("cdi", f"cdi_trail_{h}", f"cdi_{h}"),
                           ("excess", f"excess_trail_{h}", f"excess_{h}")]:
            diff = (m[a] - m[b]).abs()
            rec.append((a, b, diff.max()))
    rec_df = pd.DataFrame(rec, columns=["clean_col", "monograph_col", "max_abs_diff"])
    rec_df["n_rows_diff_gt_1e-9"] = [
        int(((m[a] - (m[b] * (0.01 if b == "ipca_deb_pct" else 1.0))).abs() > 1e-9).sum())
        for a, b in zip(rec_df["clean_col"], rec_df["monograph_col"])
    ]
    rec_df.to_csv(TABLES / "reconciliation_raw_vs_monograph.csv", index=False)

    # 3) Erro na coluna CDI_20_dias da monografia -----------------------------------------------
    err20 = (m["cdi_trail_20"] - m["cdi_20"]).abs()
    cdi19 = trailing_compound(clean["cdi_ret"], 19)
    m19 = clean[["date"]].assign(cdi19=cdi19).merge(mono[["date", "cdi_20"]], on="date")
    res["cdi20_bad_rows"] = int((err20 > 1e-9).sum())
    res["cdi20_max_err"] = float(err20.max())
    res["cdi20_matches_19d"] = float((m19["cdi19"] - m19["cdi_20"]).abs().max())

    # 4) Excesso: definição geométrica ----------------------------------------------------------
    res["excess_geo_vs_raw"] = float((clean["excess_ret"] - raw["excess_raw"]).abs().max())
    res["excess_simple_vs_raw"] = float(((clean["ida_ret"] - clean["cdi_ret"]) - raw["excess_raw"]).abs().max())
    sign_agree = ((clean["excess_trail_10"] < 0) == (clean["ida_trail_10"] < clean["cdi_trail_10"]))
    res["sign_geo_equals_simple_10d"] = bool(sign_agree[clean["ida_trail_10"].notna()].all())

    # 5) Retorno diário do IDA x variação do nível do índice -------------------------------------
    mis = clean.loc[clean["ida_ret_index_mismatch"], ["date", "ida_index", "ida_ret", "ida_index_ret"]].copy()
    mis["diff"] = mis["ida_ret"] - mis["ida_index_ret"]
    mis.to_csv(TABLES / "ida_return_vs_index_mismatch.csv", index=False)
    res["mismatch_days"] = len(mis)
    res["index_total"] = float(clean["ida_index"].iloc[-1] / clean["ida_index"].iloc[0])
    res["compounded_total"] = float((1 + clean["ida_ret"].iloc[1:]).prod())
    alt10 = trailing_compound(clean["ida_index_ret"].fillna(0.0), 10)  # só para medir impacto
    flips = ((alt10 < clean["cdi_trail_10"]) != (clean["ida_trail_10"] < clean["cdi_trail_10"]))
    res["flips_trail10_sign"] = int(flips[clean["ida_trail_10"].notna()].sum())
    res["corr_ida_vs_index"] = float(clean["ida_ret"].corr(clean["ida_index_ret"]))
    ok = ~clean["ida_ret_index_mismatch"] & clean["ida_index_ret"].notna()
    res["max_diff_other_days"] = float((clean["ida_ret"] - clean["ida_index_ret"]).abs()[ok].max())

    # 6) Escalas e valores extremos -------------------------------------------------------------
    desc = clean[["ida_ret", "ipca_deb_ret", "ibov_ret", "cdi_ret", "excess_ret", "duration"]].describe().T
    desc.to_csv(TABLES / "descriptive_stats.csv")
    res["ibov_zero"] = int(clean["ibov_zero_no_trading"].sum())
    res["cdi_changes"] = int((clean["cdi_ret"].diff() != 0).sum() - 1)
    res["cdi_ann_first"] = float((1 + clean["cdi_ret"].iloc[0]) ** 252 - 1)
    res["cdi_ann_last"] = float((1 + clean["cdi_ret"].iloc[-1]) ** 252 - 1)
    res["cdi_zero_diff_share"] = float((clean["cdi_ret"].diff() == 0).mean())

    # 7) Warm-up para a Fase 4 ------------------------------------------------------------------
    res["warmup_rows_before_original"] = int((~clean["in_original_sample"]).sum())
    res["orig_rows"] = int(clean["in_original_sample"].sum())

    # ---------------------------- Relatório -------------------------------------------------
    add("# Relatório de qualidade dos dados (Fase 2)\n")
    add("Gerado por `src/data/audit.py`. Fonte única: arquivos da pasta de materiais originais (nenhum dado externo).\n")
    add("## 1. Base oficial\n")
    add(f"**Primária: `Dados-IDA.xls`** ({res['rows_raw']} linhas, {res['date_min']} a {res['date_max']}). "
        f"`Monografia - Dados.xlsx` ({res['rows_mono']} linhas) é uma derivação dela: nas {res['rows_common']} datas em comum, "
        "`Número Índice`, `IDA DI`, `Duration`, `IPCA Deb`, `CDI` e `BOVA` são **idênticos** (diferença máxima 0), e as datas da "
        f"monografia coincidem exatamente com as de `data >= 2009-03-31` da base bruta (`{res['dates_equal_original_sample']}`).\n")
    add("Justificativa: (i) é a fonte bruta e um superconjunto; (ii) todas as colunas derivadas da monografia podem ser "
        "recalculadas a partir dela e uma delas está errada (§3); (iii) as 59 linhas extras servem de aquecimento causal para "
        "indicadores técnicos, sem preencher nada com informação futura. A amostra de modelagem fica marcada em "
        f"`in_original_sample` ({res['orig_rows']} linhas), idêntica à da monografia.\n")
    add("`LSTM.xlsx` (414 849 linhas por ativo, colunas `Ativo`, `taxa_indicativa`…) e `Mono.zip` (cópia dos mesmos arquivos) "
        "**não** fazem parte deste experimento. `IDA_DI.xlsx`, lido pelos scripts, não existe na pasta; as saídas "
        "`alertas_modelo_base_completa.xlsx` reproduzem `Monografia - Dados.xlsx` com diferença máxima 4e-16, então foi "
        "tratado como equivalente a ela.\n")
    add("## 2. Verificações estruturais\n")
    add("| Verificação | Resultado |\n|---|---|")
    add(f"| Datas duplicadas | {res['dup_dates']} |")
    add(f"| Ordem cronológica | {res['sorted']} |")
    add(f"| Datas em fim de semana | {res['weekend']} |")
    add(f"| Maior intervalo entre datas | {res['max_gap_days']} dias |")
    add(f"| `Duration` ausente | {res['duration_nan']} linhas (até {res['duration_nan_last']}); todas fora da amostra original |")
    add(f"| `BOVA` armazenado como texto (`'-2,69%'`) | {res['ibov_text_rows']} linhas (até {res['ibov_text_last']}); fora da amostra original |")
    add(f"| `BOVA` = 0 (sem pregão: 24/12, 31/12, 25/01, 09/07, 20/11) | {res['ibov_zero']} dias |")
    add("| Demais colunas | sem valores ausentes |\n")
    add("Tabela completa: `outputs/tables/missing_values.csv`.\n")
    add("## 3. Ajustes realizados nos dados\n")
    add("| # | Problema | Evidência | Tratamento na base limpa |\n|---|---|---|---|")
    add(f"| E1 | `BOVA` em texto com `%` e vírgula (18 linhas, mar/2009) | leitura como `object` | convertido para decimal; flag `ibov_from_text` |")
    add(f"| E2 | `IPCA Deb` em **%** enquanto IDA, CDI e BOVA estão em decimal | média 0,046 vs 0,0004 | dividido por 100 → `ipca_deb_ret` (decimal) |")
    add(f"| E3 | Coluna `CDI_20_dias` da monografia está **errada** | difere do composto de 20 dias do `CDI_dia` em {res['cdi20_bad_rows']} linhas, erro máx. {res['cdi20_max_err']:.2e}; coincide com janela de **19** dias (erro máx. {res['cdi20_matches_19d']:.1e}) | recalculado a partir do retorno diário; `Diferença…20Dias` também corrigida. Não afeta o alvo de 10 dias |")
    add(f"| E4 | Colunas `Diferença CDIxIDA_*` e `Excesso` são **excesso geométrico** `(1+IDA)/(1+CDI)−1`, não `IDA−CDI` | diferença máx. vs geométrico {res['excess_geo_vs_raw']:.1e}; vs simples {res['excess_simple_vs_raw']:.1e} | `excess_ret` e `excess_trail_h` usam a forma geométrica. O sinal de `IDA_10 < CDI_10` é idêntico nas duas formas ({res['sign_geo_equals_simple_10d']}), então o alvo original não muda por isso |")
    add(f"| E5 | `CDI_dia`: scripts dividem por 100 | `CDI_dia` já é decimal (≈ {res['cdi_ann_first']:.1%} a.a. em 2009) | **não** dividir; `cdi_ret` único, em decimal |")
    add(f"| E6 | Retorno diário do IDA diverge da variação do nível do índice em **{res['mismatch_days']} dias** | ver `ida_return_vs_index_mismatch.csv`; nos demais dias a diferença máxima é {res['max_diff_other_days']:.1e} | mantida `ida_ret` (base de todas as colunas acumuladas, igual à monografia), com `ida_index_ret` e a flag `ida_ret_index_mismatch`. **Decisão pendente para o backtest (Fase 9)** |")
    add(f"| E7 | `CDI_dia` muda em apenas {res['cdi_changes']} dias (igual ao dia anterior em {res['cdi_zero_diff_share']:.1%}); anualizado (252 d.u.) vai de {res['cdi_ann_first']:.1%} (jan/2009) a {res['cdi_ann_last']:.1%} (nov/2025) | padrão de atualização em degraus | **conferido pelo autor do projeto** contra a fonte do CDI |")
    add("")
    add("### E6 em detalhe\n")
    add(f"Correlação entre `ida_ret` e `ida_index_ret`: {res['corr_ida_vs_index']:.4f}. Nos demais dias a igualdade vale até arredondamento. "
        "Nos dias divergentes: em 2025-04-01 e 2025-09-16 o nível do índice **reverte o salto do dia anterior** (fim/início de mês) e a "
        "coluna de retorno não registra a reversão; em 2020-11-30 e 2022-05-30 o índice sobe 0,2 % e 0,5 % enquanto a coluna de retorno mostra "
        "apenas o carregamento normal. Os dois cálculos são mostrados e a escolha está registrada nas decisões. Efeito acumulado: "
        f"índice {res['index_total']:.4f}× vs composto dos retornos {res['compounded_total']:.4f}× no período todo. "
        f"Se o alvo usasse retornos derivados do índice, o sinal de `IDA_10 < CDI_10` mudaria em {res['flips_trail10_sign']} dias "
        "(janelas de 10 dias que contêm algum dos 4 dias).\n")
    add("## 4. Reconciliação das colunas derivadas\n")
    add("`outputs/tables/reconciliation_raw_vs_monograph.csv` compara cada coluna da base limpa com a da monografia. "
        "Resumo (máx. |diferença|):\n")
    add("| coluna limpa | coluna monografia | máx. abs. dif. | linhas com dif. > 1e-9 |\n|---|---|---|---|")
    for r in rec_df.itertuples(index=False):
        add(f"| {r.clean_col} | {r.monograph_col} | {r.max_abs_diff:.2e} | {r[3]} |")
    add("\nLeitura: acumulados de 10, 30 e 60 dias do IDA e do CDI e o de 20 dias do IDA coincidem até 1e-15 "
        "(**janela retroativa terminando em t, inclusive**). As colunas `Diferença…` da monografia são excessos geométricos "
        "e coincidem com `excess_*` (exceto o de 20 dias, afetado por E3). `ipca_deb_ret` difere só por escala (E2).\n")
    add("## 5. Retornos e estatísticas\n")
    add("Ver `outputs/tables/descriptive_stats.csv`. Valores extremos: `BOVA` (até ±14 %) em 08/08/2011, 18/05/2017 e 09–25/03/2020; "
        "`IPCA Deb` (até ±3,4 %) em 24/09/2015, 18/05/2017, 12–18/03/2020, 24/04/2020 e 12/01/2023. Coincidem com episódios de estresse; "
        "**não foram removidos nem winsorizados** (a revisar na Fase 4 se afetarem a padronização).\n")
    add("## 6. Causalidade e dados futuros\n")
    add("- Nenhum valor ausente foi preenchido (`fillna`, `ffill`, `bfill` não são usados). As 59 primeiras linhas têm `Duration` ausente e "
        "ficam fora da amostra de modelagem; servem apenas de histórico para janelas móveis.\n"
        "- Os acumulados `*_trail_*` usam apenas retornos até t. Indicadores técnicos (RSI, MACD, volatilidade, etc.) serão "
        "construídos na Fase 4 sobre esta base e testados quanto à causalidade.\n"
        "- Nenhum `StandardScaler` é aplicado nesta fase (será ajustado só no treino, Fase 4/5).\n")
    add("## 7. Versão definitiva\n")
    add("`data/processed/ida_clean.csv` (não versionada: os dados da ANBIMA não são redistribuídos, veja `data/README.md`). "
        "Reproduzir com `python -I src/data/clean_base.py`. Dicionário em `docs/data_dictionary.md`.\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    pd.Series(res).to_csv(TABLES / "audit_summary.csv", header=["value"])
    print("Relatório:", REPORT)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
