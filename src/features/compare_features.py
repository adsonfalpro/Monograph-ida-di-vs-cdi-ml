"""Compara as variáveis da versão corrigida com o código original (LS.py) e mede o efeito do scaler.

Gera docs/feature_engineering.md e outputs/tables/features_*.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("data", "targets", "features"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from build_target import build_target  # noqa: E402
from clean_base import build_clean_base, load_monograph, load_raw  # noqa: E402
from engineering import (  # noqa: E402
    FEATURE_COLUMNS, ORIGINAL_NAMES, aggregate_windows, build_features, first_valid_row, fit_scaler,
    make_windows, transform,
)

LOOKBACK = 100


def original_features(mono: pd.DataFrame) -> pd.DataFrame:
    """Cópia literal da engenharia de variáveis de LS.py/LST.py (incluindo fillna(0))."""
    df = pd.DataFrame({
        "IDA_DI": mono["ida_ret"], "Duration": mono["duration"], "IPCA_Deb": mono["ipca_deb_pct"],
        "BOVA": mono["ibov_ret"], "CDI_dia": mono["cdi_ret"],
    })
    df["Excesso_dia"] = df["IDA_DI"] - df["CDI_dia"]
    df["vol_20"] = df["Excesso_dia"].rolling(20).std()
    df["trend_20"] = df["IDA_DI"].rolling(20).mean()
    df["ret_20"] = df["Excesso_dia"].rolling(20).sum()
    delta = df["IDA_DI"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean().replace(0, np.nan)
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    ema12 = df["IDA_DI"].ewm(span=12, adjust=False).mean()
    ema26 = df["IDA_DI"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df


def main() -> None:
    clean = build_clean_base(load_raw())
    mono = load_monograph()
    feats = build_features(clean)
    sample_start = int(np.where(clean["in_original_sample"])[0][0])
    fv = first_valid_row(feats)

    # 1) Todas as variáveis válidas desde o início da amostra, sem preencher nada
    nan_in_sample = int(feats.loc[sample_start:, FEATURE_COLUMNS].isna().sum().sum())

    # 2) Comparação com o original (alinhada por data)
    orig = original_features(mono)
    orig["date"] = mono["date"].values
    ours = feats.iloc[sample_start:].reset_index(drop=True)
    rows = []
    for o_name, n_name in ORIGINAL_NAMES.items():
        d = (ours[n_name].values - orig[o_name].values)
        if o_name == "IPCA_Deb":
            d = (ours[n_name].values * 100 - orig[o_name].values)
        d = np.abs(d)
        warm = d[:150]
        rows.append({
            "variavel_original": o_name, "variavel_nova": n_name,
            "max_abs_diff_total": np.nanmax(d), "max_abs_diff_apos_150_linhas": np.nanmax(d[150:]),
            "linhas_dif_gt_1e9": int((d > 1e-9).sum()), "primeiras_20_linhas_NaN_no_original_zeradas": o_name in
            ("vol_20", "trend_20", "ret_20", "rsi_14"),
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(ROOT / "outputs" / "tables" / "features_vs_original.csv", index=False)
    # RSI com ausência de perdas
    d = clean["ida_ret"].diff()
    loss14 = (-d.clip(upper=0)).rolling(14).mean()
    rsi_no_loss = int(((loss14 == 0) & loss14.notna()).iloc[sample_start:].sum())
    orig_rsi_zero_rows = int((orig["rsi_14"].isna()).sum())

    # 3) Scaler: base inteira (original) x só treino (corrigido) — fronteira PROVISÓRIA (split 60% da Fase 3)
    tgt = build_target(clean.iloc[sample_start:].reset_index(drop=True))
    n_win = int(tgt["y"].notna().sum() - (LOOKBACK - 1))  # janelas com alvo (decisão de 99 em diante)
    tr_windows = int(n_win * 0.6)
    train_end_row_in_sample = (LOOKBACK - 1) + tr_windows - 1       # última data de decisão de treino
    train_end_row = sample_start + train_end_row_in_sample
    sc_train = fit_scaler(feats, train_end_row)
    sc_full = fit_scaler(feats, len(feats) - 1)
    leak = pd.DataFrame({
        "variavel": FEATURE_COLUMNS,
        "media_treino": sc_train.mean_, "media_base_inteira": sc_full.mean_,
        "dp_treino": sc_train.scale_, "dp_base_inteira": sc_full.scale_,
    })
    leak["deslocamento_em_dp_treino"] = (leak["media_base_inteira"] - leak["media_treino"]) / leak["dp_treino"]
    leak["razao_dp"] = leak["dp_base_inteira"] / leak["dp_treino"]
    leak.to_csv(ROOT / "outputs" / "tables" / "scaler_train_vs_full.csv", index=False)

    # 4) Agregação das janelas (RF/Perceptron)
    x = transform(feats, sc_train)
    decision_rows = np.arange(sample_start + LOOKBACK - 1, sample_start + LOOKBACK - 1 + n_win)
    w = make_windows(x, decision_rows, LOOKBACK)
    agg, undefined = aggregate_windows(w)
    summary = {
        "first_valid_row": fv, "sample_start_row": sample_start, "nan_in_sample": nan_in_sample,
        "n_windows_with_target": n_win, "agg_shape": agg.shape, "undefined_stats_set_to_0": undefined,
        "rsi_windows_without_losses": rsi_no_loss, "orig_rsi_nan_rows": orig_rsi_zero_rows,
        "train_end_decision_date": str(clean["date"].iloc[train_end_row].date()),
    }
    pd.Series(summary).to_csv(ROOT / "outputs" / "tables" / "features_summary.csv", header=["value"])

    L = ["# Engenharia de variáveis (Fase 4)\n",
         "Gerado por `src/features/compare_features.py`. Código: `src/features/engineering.py`. Testes: `tests/test_features.py`.\n",
         "## 1. Variáveis (mesma lista da monografia)\n",
         "| Grupo | Variável (original → nova) | Fórmula | Disponível em t? |\n|---|---|---|---|",
         "| Crédito | `IDA_DI` → `ida_ret` | retorno diário do IDA‑DI | sim (fechamento de t) |",
         "| Crédito | `Duration` → `duration` | duration média do índice | sim |",
         "| Crédito | `IPCA_Deb` → `ipca_deb_ret` | retorno diário IDA‑IPCA (÷100) | sim |",
         "| Mercado | `BOVA` → `ibov_ret` | retorno diário Ibovespa | sim |",
         "| Mercado | `CDI_dia` → `cdi_ret` | CDI diário | sim |",
         "| Excesso | `Excesso_dia` → `excesso_dia` | `ida_ret − cdi_ret` (forma simples, como no original) | sim |",
         "| Volatilidade | `vol_20` | desvio‑padrão (ddof=1) de `excesso_dia` em 20 dias | sim |",
         "| Tendência | `trend_20` | média de `ida_ret` em 20 dias | sim |",
         "| Retorno | `ret_20` | soma de `excesso_dia` em 20 dias | sim |",
         "| Momentum | `rsi_14` | RSI (média simples, 14 d.) da **variação diária do retorno** do IDA | sim |",
         "| Técnica | `macd`, `macd_signal` | EMA12−EMA26 e EMA9 do `ida_ret` (`adjust=False`) | sim |",
         "",
         "Todas só usam observações ≤ t (teste de perturbação em `tests/test_features.py`: alterar dados em t+1 em diante não muda nenhuma variável em t).\n",
         "## 2. Mudanças em relação ao código original\n",
         "| # | Original | Corrigido | Motivo |\n|---|---|---|---|",
         "| F1 | `fillna(0)` em todas as variáveis | nenhum preenchimento; aquecimento vem das 59 linhas anteriores à amostra | `fillna(0)` colocava 0 (média falsa) nas ~25 primeiras linhas; com a padronização isso virava um valor atípico |",
         f"| F2 | RSI: `loss=0 → NaN → fillna(0)` | RSI = 100 sem perdas (50 sem ganhos nem perdas) | correção defensiva: o original daria RSI 0 quando só há ganhos. **Nenhuma ocorrência nesta amostra** ({rsi_no_loss} janelas), então não altera os resultados |",
         "| F3 | `StandardScaler` ajustado na base inteira | ajustado só nas linhas de treino (`fit_scaler`) | vazamento de média/desvio de validação e teste |",
         "| F4 | `ewm` sem `min_periods` | `min_periods` = período da média | os primeiros valores do MACD eram baseados em 1–2 observações |",
         f"| F5 | `nan_to_num` em toda a matriz agregada | só assimetria/curtose de janela constante → 0; {undefined} casos de {agg.shape[0]*12*2} | evita esconder outros NaN; qualquer outro valor não finito levanta erro |",
         "",
         "Não alterados (fiéis à monografia, mas questionáveis): RSI e MACD aplicados ao **retorno diário** e não ao nível do índice; "
         "`ret_20` é soma simples; `Excesso_dia` simples. Vão para `docs/limitations.md`.\n",
         "## 3. Equivalência com o original\n",
         f"Variáveis ausentes na amostra original (4181 linhas): **{nan_in_sample}**; primeira linha com as 12 válidas: {fv} "
         f"(a amostra começa na linha {sample_start}, logo todas estão disponíveis sem preencher nada).\n",
         "Diferença máxima absoluta nas linhas da amostra (alinhadas por data):\n",
         "| variável | máx. total | máx. após 150 linhas | linhas com dif > 1e‑9 |\n|---|---|---|---|"]
    for r in cmp_df.itertuples(index=False):
        L.append(f"| {r.variavel_original} | {r.max_abs_diff_total:.3g} | {r.max_abs_diff_apos_150_linhas:.3g} | {r.linhas_dif_gt_1e9} |")
    L += ["",
          "Leitura (comparação feita antes do `fillna(0)` do original, ignorando NaN): variáveis de nível, `vol_20`, `trend_20`, `ret_20` e RSI são idênticas "
          "onde o original tinha valor. Nas ~20–25 primeiras linhas o original tinha NaN e depois usava **0** (`fillna(0)`); aqui há valores reais, "
          "calculados com o aquecimento anterior à amostra. MACD e `macd_signal` diferem no início (até ~2e‑5) porque a média móvel aqui começa "
          "59 linhas antes e converge; após 150 linhas a diferença é < 4e‑10.\n",
          "## 4. Efeito do vazamento do scaler (fronteira de treino provisória)\n",
          f"Fronteira provisória: split 60% das {n_win} janelas com alvo, última data de decisão de treino = {summary['train_end_decision_date']} "
          "(a divisão definitiva é a da Fase 5). Média e desvio‑padrão do treino versus da base inteira:\n",
          "| variável | média treino | média base inteira | desloc. (dp treino) | razão dp |\n|---|---|---|---|---|"]
    for r in leak.itertuples(index=False):
        L.append(f"| {r.variavel} | {r.media_treino:.4g} | {r.media_base_inteira:.4g} | {r.deslocamento_em_dp_treino:+.2f} | {r.razao_dp:.2f} |")
    L += ["",
          "Quanto maior o deslocamento, mais informação do futuro (validação/teste) o original embutia nas variáveis padronizadas. "
          "Variáveis como `cdi_ret`, `duration` e `ida_ret` mudam de nível entre períodos (CDI alto em 2022‑25), então o efeito não é pequeno.\n",
          "## 5. Janelas e agregação\n",
          f"- Janela de {LOOKBACK} linhas terminando na data de decisão t (`make_windows`); a janela de t não contém linhas > t.\n"
          f"- Janelas com alvo na amostra original: **{n_win}** (o original tinha 4071: o laço `range(len−lookback−horizon)` descartava a última janela válida, "
          "cujo alvo usa o último retorno da base).\n"
          f"- Agregação RF/Perceptron: {agg.shape[1]} atributos (10 estatísticas × 12 variáveis); estatísticas indefinidas trocadas por 0: {undefined}.\n"
          "- O LSTM usa a janela 100×12 padronizada, sem agregar.\n"]
    (ROOT / "docs" / "feature_engineering.md").write_text("\n".join(L), encoding="utf-8")
    print(summary)
    print(cmp_df.to_string())
    print(leak.round(3).to_string())


if __name__ == "__main__":
    main()
