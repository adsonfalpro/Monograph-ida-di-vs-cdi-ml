"""Gera docs/relatorio_completo.html: relatório autocontido (CSS, figuras e gráficos embutidos) de toda a reconstrução.

Números das tabelas vêm dos arquivos em outputs/; nada é recalculado aqui.
Uso:  python -I src/report/build_html_report.py
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from html_lib import (CSS, MODELS, M_COL, callout, esc, figure, kpi, legend, pc, pt, svg_ci, svg_equity, svg_prev_year,  # noqa: E402
                      svg_steps, svg_timeline, table)

T = ROOT / "outputs" / "tables"
MET = ROOT / "outputs" / "metrics"
DD = ROOT / "outputs" / "dashboard_data"
STEPS = ["S0_original", "S1_alvo", "S2_scaler", "S3_variaveis", "S4_purge", "S5_crossfit"]
STEP_LABEL = {"S0_original": "S0 · original (código literal)", "S1_alvo": "S1 · + alvo formal", "S2_scaler": "S2 · + scaler só no treino",
              "S3_variaveis": "S3 · + variáveis sem fillna(0)", "S4_purge": "S4 · + purge", "S5_crossfit": "S5 · + cross-fit (4071 janelas, retreinado)"}
KIND = {"perceptron": "Perceptron", "rf": "Random Forest", "lstm": "LSTM"}


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> None:
    ci = pd.DataFrame(rj(DD / "ci_metrics.json")["rows"])
    ab = pd.DataFrame(rj(DD / "ablation.json")["rows"])
    recon = pd.DataFrame(rj(DD / "recon.json")["rows"])
    aud_raw = pd.read_csv(T / "audit_summary.csv", index_col=0)["value"]

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    aud = {k: _num(v) for k, v in aud_raw.items()}
    split = pd.read_csv(T / "split_summary.csv")
    p7 = pd.read_csv(T / "phase7_test_metrics.csv")
    p7v = pd.read_csv(T / "phase7_validation_diagnostics.csv")
    cost = pd.read_csv(T / "phase7_cost_table.csv", index_col=0)
    pair = pd.read_csv(T / "phase8_pairwise_auc.csv")
    bt = pd.read_csv(T / "phase9_backtest_summary.csv")
    bench = pd.read_csv(T / "phase9_benchmarks.csv", index_col=0)
    bci = pd.read_csv(T / "phase9_excess_ci.csv")
    orig = pd.read_csv(T / "phase9_original_backtest_in_out_sample.csv")
    xl = pd.read_csv(T / "phase9_excel_check.csv")
    tr6 = rj(MET / "phase6_training_summary.json")
    base = rj(MET / "test_baselines.json")
    frozen = rj(MET / "frozen_config.json")
    ablc = pd.DataFrame([rj(p) for p in sorted(glob.glob(str(MET / "ablation_cache" / "*.json")))])
    ablc["auc_rep"] = ablc["auc_cal"].where(ablc["auc_cal"].notna(), ablc["auc"])
    by_year = pd.read_csv(T / "target_rate_by_year.csv")
    by_split = pd.read_csv(T / "target_rate_by_split.csv").set_index("split")
    cmpw = pd.read_csv(T / "target_comparison_by_window.csv")
    feat_cmp = pd.read_csv(T / "features_vs_original.csv")
    leak = pd.read_csv(T / "scaler_train_vs_full.csv")
    rec = pd.read_csv(T / "reconciliation_raw_vs_monograph.csv")
    mism = pd.read_csv(T / "ida_return_vs_index_mismatch.csv")

    g = lambda m, k: ci[(ci.modelo == m) & (ci.metrica == k)].iloc[0]
    auc_s = {m: ab[(ab.modelo == m)].sort_values("passo_ordem").auc.tolist() for m in MODELS}
    col = {"Perceptron": "var(--m1)", "Random Forest": "var(--m2)", "LSTM": "var(--m3)"}

    S: list[str] = []
    add = S.append

    # ============================================================== cabeçalho
    toc = [("resumo", "Resumo executivo"), ("metodo", "Como o trabalho foi conduzido"), ("f1", "Fase 1 · Metodologia original"), ("f2", "Fase 2 · Dados"),
           ("f3", "Fase 3 · Variável‑alvo"), ("f4", "Fase 4 · Variáveis"), ("f5", "Fase 5 · Divisão e vazamento"), ("f6", "Fase 6 · Modelos"),
           ("f7", "Fase 7 · Calibração e limiar"), ("f8", "Fase 8 · Avaliação e decomposição"), ("f9", "Fase 9 · Backtest"),
           ("monografia", "Comparação com a monografia"), ("limites", "Limitações conhecidas"), ("decisoes", "Decisões e suposições"),
           ("repro", "Reprodutibilidade"), ("apendice", "Apêndice: tabelas completas e glossário")]
    nav = "".join(f'<li><a href="#{i}"><span class="num">{k}</span>{esc(t)}</a></li>' for k, (i, t) in enumerate(toc))
    add(f"""<div class="layout"><nav class="toc" aria-label="Sumário"><div class="brand">Relatório completo</div><ol>{nav}</ol></nav><main>""")
    add(f"""<header class="hero"><div class="eyebrow">Reprodução e avaliação · Engenharia Financeira</div>
<h1>Modelos de machine learning para prever retornos do IDA‑DI abaixo do CDI</h1>
<p class="lead">Relatório completo da implementação reproduzível, com a metodologia revisada, do experimento da monografia sobre previsão de desempenho do índice de debêntures ANBIMA frente ao CDI com 10 dias de antecedência.
Cobre as fases executadas, os resultados finais, como eles se comparam aos da monografia e os limites da análise.</p>
<p class="cap" style="margin:0">Gerado em {date.today().strftime('%d/%m/%Y')} a partir dos arquivos do projeto. Todos os números vêm de <code>outputs/</code>; nada foi recalculado para este documento.</p></header>""")

    # ============================================================== resumo
    add('<h2 id="resumo"><span class="n">0 · Resumo</span>Resumo executivo</h2>')
    add(f'<p class="lead">Com o alvo definido como no texto da monografia e sem vazamento de dados, <b>nenhum dos três modelos se distingue do acaso no teste</b>, '
        f'e nenhuma estratégia baseada em modelo supera o IDA no backtest. Esse resultado difere do apresentado na monografia.</p>')
    add('<div class="grid kpis">' +
        kpi("Perceptron · AUC no teste", pt(g("Perceptron", "auc").ponto), f"IC 95 % [{pt(g('Perceptron','auc').ic_baixo)}; {pt(g('Perceptron','auc').ic_alto)}]", col["Perceptron"]) +
        kpi("Random Forest · AUC no teste", pt(g("Random Forest", "auc").ponto), f"IC 95 % [{pt(g('Random Forest','auc').ic_baixo)}; {pt(g('Random Forest','auc').ic_alto)}]", col["Random Forest"]) +
        kpi("LSTM · AUC no teste", pt(g("LSTM", "auc").ponto), f"IC 95 % [{pt(g('LSTM','auc').ic_baixo)}; {pt(g('LSTM','auc').ic_alto)}]", col["LSTM"]) +
        kpi("AUC de um acaso", "0,50", "todos os intervalos contêm este valor") + "</div>")
    add('<div class="grid kpis">' +
        kpi("Efeito de trocar só o alvo", f"{pt(auc_s['Random Forest'][0])} → {pt(auc_s['Random Forest'][1])}", "AUC do Random Forest, do código original ao alvo formal") +
        kpi("Backtest no teste · IDA", pt(bench.loc["IDA (comprar e manter)", "final_base100"], 1), "base 100; o CDI chegou a " + pt(bench.loc["CDI 100 %", "final_base100"], 1)) +
        kpi("Testes automatizados", "66", "todos passando") +
        kpi("Acessos ao teste", "11", "uma vez por modelo congelado") + "</div>")
    add('<h3>Os achados mais importantes</h3><ol class="l">'
        '<li><b>O alvo da monografia se sobrepunha às variáveis de entrada.</b> O rótulo marcava 1 se <i>qualquer</i> das 10 janelas deslizantes seguintes fosse negativa, e 9 dos 10 dias de cada janela já estavam nas variáveis de entrada. '
        'Trocar apenas o alvo pelo definido no texto derruba o AUC de todos os modelos (veja a Fase 8).</li>'
        '<li><b>As outras correções movem pouco.</b> Padronização só no treino, variáveis sem preenchimento com zero, purge e calibração fora da amostra alteram o AUC em magnitudes da ordem da variação entre sementes.</li>'
        '<li><b>O backtest da monografia cobre o período inteiro (treino, validação e teste).</b> A Fig. 7 é reproduzida exatamente; o ganho sobre o IDA vem em boa parte da validação (usada para escolher o limiar) e não se sustenta no teste.</li>'
        '<li><b>Calibrar e escolher o limiar nos mesmos dados favorece o desempenho aparente.</b> Com calibração fora da amostra, o F1 degenera para “alertar todos os dias” em RF e LSTM.</li>'
        '<li><b>No período de teste o IDA rendeu mais que o CDI.</b> Qualquer estratégia que fique no IDA “supera o CDI” sem prever nada; a comparação relevante é com o IDA.</li></ol>')
    add(callout("warn", "Como ler estas conclusões",
                "<p>É <b>ausência de evidência</b>, não prova de que os modelos sejam inúteis: o teste tem 169 eventos, as janelas são muito autocorrelacionadas, a taxa de eventos muda muito entre períodos e há uma única divisão cronológica. O poder estatístico é baixo.</p>"))

    # ============================================================== método
    add('<h2 id="metodo"><span class="n">1 · Método</span>Como o trabalho foi conduzido</h2>')
    add('<p>A pergunta da monografia foi mantida: <i>é possível prever, com até 10 dias de antecedência, se o IDA‑DI renderá menos que o CDI?</i> '
        'Também foram mantidos os três modelos (Perceptron, Random Forest e LSTM), as variáveis, o lookback de 100 dias, o horizonte de 10 dias e a divisão 60/20/20. '
        'O trabalho preservou <b>duas versões</b> para que se saiba se os resultados mudam por causa da metodologia e não da implementação:</p>')
    add('<div class="two"><div class="callout"><b class="t">Versão original</b><p>O que os scripts da monografia fazem, sem alterações (só um caminho local foi anonimizado). Serve de referência e de ponto de partida da decomposição.</p></div>'
        '<div class="callout info"><b class="t">Versão revisada</b><p>O mesmo experimento, com o alvo do texto, controle de informação posterior, calibração e limiar fora da amostra e teste protegido. Toda diferença está documentada.</p></div></div>')
    add(table(["Fase", "Entrega", "Critério de aprovação"], [
        ["1", "Metodologia original reconstruída, com inconsistências", "Reconstrução documentada"],
        ["2", "Base limpa, dicionário de dados, relatório de qualidade", "Base validada e consistente"],
        ["3", "Alvo formal e comparação com o original", "Fórmula e alinhamento temporal testados"],
        ["4", "Variáveis sem informação futura; scaler só no treino", "Ausência de informação futura (teste de perturbação)"],
        ["5", "Divisão cronológica, purge, teste protegido", "Ausência de vazamento temporal"],
        ["6", "Três modelos reproduzíveis, 5 sementes", "Treinamento reproduzível"],
        ["7", "Calibração e limiar fora da amostra; teste aberto uma vez", "Calibração e seleção isoladas do teste"],
        ["8", "Avaliação, intervalos de confiança, decomposição", "Métricas reconciliadas"],
        ["9", "Backtest, comparação com o original, planilha de conferência", "Estratégias reproduzíveis em Python e Excel"],
        ["10", "Organização para o GitHub", "Documentação e estrutura completas"]], cls=""))
    add(callout("info", "Disciplina de teste",
                "<p>Nenhum resultado de teste foi visto antes de a Fase 7 congelar modelos, calibradores e limiares (com hash SHA‑256 em <code>frozen_config.json</code>). "
                "O teste só é liberado por um mecanismo que registra cada acesso em <code>test_access_log.json</code> e recusa a repetir um modelo já registrado. "
                "A decomposição da Fase 8 retreinou outros pipelines e também mediu o teste, apenas como diagnóstico; nenhuma escolha foi feita com base nela.</p>"))

    # ============================================================== fase 1
    add('<h2 id="f1"><span class="n">Fase 1</span>Metodologia original: o que a monografia executa</h2>')
    add('<p>Antes de corrigir qualquer coisa foi preciso saber exatamente o que se estava tentando reproduzir. Os scripts da pasta de trabalho foram mapeados contra o texto e as figuras do PDF.</p>')
    add(table(["Papel", "Arquivo", "Evidência de vínculo"], [
        ["Random Forest final (Apêndice A)", "<code>LS.py</code>", "Idêntico ao apêndice; as métricas salvas (limiar 0,237, F1 0,562, AUC 0,737) batem com a figura e a matriz 319/245/57/194"],
        ["LSTM final (Apêndice B)", "<code>LST.py</code>", "Idêntico ao apêndice; limiar 0,269, precisão 0,968, recall 0,243 e AUC 0,810 batem com a matriz 562/2/190/61"],
        ["Perceptron final (Apêndice C)", "<code>Perceptron.py</code>", "Idêntico ao apêndice; <b>mas a Fig. 4.1 não é reproduzida por este script</b>"],
        ["Matrizes normalizadas à mão", "<code>IDA.py</code>, <code>norm.py</code>", "Valores digitados no código, não calculados"],
        ["Versões anteriores", "<code>M.py</code>, <code>S.py</code>", "Outro alvo (retorno dos h dias seguintes), split 80/10/10, custo FN=5/FP=1"],
        ["Backtest (Fig. 7)", "—", "<b>Não existe código</b>: foi feito no Excel sobre o período inteiro"]]))
    add('<h3>A definição de alvo que os scripts finais usam</h3>')
    add('<pre><code>drawdown[t] = 1  se  IDA_10_dias[t] &lt; CDI_10_dias[t]      # janela RETROATIVA, dias t-9 … t\n'
        'janela i  = linhas i … i+99          (variáveis de entrada)\n'
        'y_i       = max( drawdown[i+100 … i+109] )   # “algum” dos 10 dias seguintes</code></pre>')
    add('<p>Os rótulos <code>drawdown[i+100…i+109]</code> usam janelas retroativas que cobrem os dias <code>i+91 … i+109</code>. Os dias <code>i+91 … i+99</code> (nove) já estão na janela de variáveis. '
        'Além disso, o <code>max</code> marca 1 se qualquer das 10 janelas deslizantes for negativa. Isso não é “o retorno dos próximos 10 dias” descrito no texto.</p>')
    add('<h3>Pontos de atenção entre o texto, as figuras e o código</h3>')
    add(table(["#", "Inconsistência", "Natureza"], [
        ["1", "O alvo do código (máximo de 10 janelas retroativas) difere do descrito no texto (excesso de retorno dos 10 dias seguintes)", '<span class="tag bad">metodológica</span>'],
        ["2", "<code>StandardScaler</code> ajustado na base inteira antes da divisão", '<span class="tag bad">vazamento</span>'],
        ["3", "Sem purge: 9 rótulos por fronteira olham para dentro do bloco seguinte", '<span class="tag bad">vazamento</span>'],
        ["4", "RF: calibrador isotônico e limiar na mesma validação; LSTM: parada antecipada, calibração e limiar na mesma validação", '<span class="tag bad">otimismo</span>'],
        ["5", "Backtest em todo o período, cerca de 80 % dentro da amostra de ajuste", '<span class="tag bad">avaliação</span>'],
        ["6", "Fig. 4.1 (Perceptron) tem 314 positivos no teste; RF e LSTM têm 251 com o mesmo split", '<span class="tag warn">inexplicada</span>'],
        ["7", "Texto: Perceptron acerta 69 % / 56 %; Fig. 4.2: 78 % / 45 %", '<span class="tag warn">texto × figura</span>'],
        ["8", "“Multilayer Perceptron” no texto; o código é de camada única", '<span class="tag warn">terminologia</span>'],
        ["9", "Texto diz maximizar o recall; o limiar é escolhido por F1", '<span class="tag warn">texto × código</span>'],
        ["10", "Brier prometido nas métricas, nunca reportado; numeração de figuras repetida; curva de calibração do LSTM descrita como “segue a diagonal” mas fica bem acima", '<span class="tag warn">texto × figura</span>'],
        ["11", "CDI dividido por 100 no código (já estava em decimal); gráfico “Comparação” dos scripts fica errado", '<span class="tag warn">unidade</span>']]))

    # ============================================================== fase 2
    add('<h2 id="f2"><span class="n">Fase 2</span>Dados: base de referência e ajustes</h2>')
    add(f'<p>A base primária é <code>Dados-IDA.xls</code> ({int(aud["rows_raw"])} linhas, {aud["date_min"]} a {aud["date_max"]}), a fonte bruta. '
        f'<code>Monografia - Dados.xlsx</code> ({int(aud["rows_mono"])} linhas) é uma derivação dela: nas {int(aud["rows_common"])} datas em comum, índice, IDA, duration, IPCA, BOVA e CDI são <b>idênticos</b>. '
        'As 59 linhas extras servem de aquecimento para os indicadores sem preencher nada com informação futura. Os scripts leem <code>IDA_DI.xlsx</code>, que não existe na pasta; '
        'as saídas dos scripts reproduzem a base da monografia com diferença máxima de 4e‑16, então os dois foram tratados como equivalentes.</p>')
    add(table(["#", "Ajuste", "Evidência", "Tratamento"], [
        ["E1", "<code>BOVA</code> em texto (“-2,69%”) em parte de 2009", f"{int(aud['ibov_text_rows'])} linhas, todas antes de 2009‑03‑31", "Convertido para decimal"],
        ["E2", "<code>IPCA Deb</code> em %, demais séries em decimal", "média 0,046 contra 0,0004", "Dividido por 100"],
        ["E3", "<code>CDI_20_dias</code> da monografia usa janela de 19 dias", f"erro em {int(aud['cdi20_bad_rows'])} linhas, máx. {aud['cdi20_max_err']:.1e}; coincide com 19 dias", "Recalculado; não afeta o alvo de 10 dias"],
        ["E4", "Colunas “Diferença” são excesso geométrico, não IDA − CDI", "o sinal do alvo é idêntico nas duas formas", "Forma geométrica adotada"],
        ["E5", "Scripts dividem <code>CDI_dia</code> por 100", "CDI já é decimal (≈ 13,6 % a.a. em 2009)", "Removido"],
        ["E6", "Retorno diário do IDA diverge da variação do índice", f"{int(aud['mismatch_days'])} dias; se o alvo usasse o índice, o sinal mudaria em {int(aud['flips_trail10_sign'])} dias", "Mantido o retorno (como na monografia); dias marcados"],
        ["E7", "<code>CDI_dia</code> muda em poucos dias", f"{int(aud['cdi_changes'])} mudanças; igual ao dia anterior em {pc(aud['cdi_zero_diff_share'])}", '<span class="tag ok">conferido</span> pelo autor do projeto']]))
    add('<details><summary>Dias em que o retorno do IDA diverge do nível do índice</summary><div>' +
        table(["Data", "Nível do índice", "Retorno do IDA", "Variação do índice", "Diferença"],
              [[r.date, pt(r.ida_index, 2), pt(100 * r.ida_ret, 3) + " %", pt(100 * r.ida_index_ret, 3) + " %", pt(100 * r.diff, 3) + " p.p."] for r in mism.itertuples()], num=(1, 2, 3, 4)) +
        f'<p class="cap">Efeito acumulado: índice {pt(aud["index_total"],4)}× contra {pt(aud["compounded_total"],4)}× pelo composto dos retornos, em todo o período. Os dois cálculos são mostrados e a escolha está registrada nas decisões.</p></div></details>')
    add('<details><summary>Reconciliação das colunas derivadas (máx. |diferença| entre a base limpa e a monografia)</summary><div>' +
        table(["Coluna limpa", "Coluna da monografia", "Máx. |dif.|", "Linhas com dif. > 1e‑9"],
              [[f"<code>{r.clean_col}</code>", f"<code>{r.monograph_col}</code>", f"{r.max_abs_diff:.2e}", ("<b>" + str(int(r._4)) + "</b>") if r._4 else "0"] for r in rec.itertuples()], num=(2, 3)) + "</div></details>")
    add(callout("ok", "Resultado", "<p>Base limpa e documentada, com flags de rastreabilidade (amostra original, duration ausente, BOVA convertido, divergência de índice). Nada foi preenchido com <code>fillna</code>, <code>ffill</code> ou <code>bfill</code>, nada foi padronizado e nenhum valor extremo (março de 2020) foi removido.</p>"))

    # ============================================================== fase 3
    add('<h2 id="f3"><span class="n">Fase 3</span>Variável‑alvo</h2>')
    add('<p>Esta é provavelmente a etapa mais importante do projeto. Com <code>t</code> a última data observada e <code>h = 10</code> dias úteis:</p>')
    add('<pre><code>R_IDA(t, t+h) = Π (1 + r_IDA[t+i]) − 1    para i = 1 … h\nR_CDI(t, t+h) = Π (1 + r_CDI[t+i]) − 1\nY_t = 1  se  R_IDA(t, t+h) &lt; R_CDI(t, t+h);   Y_t = 0  caso contrário (empate = 0)</code></pre>')
    add('<p>Só entram retornos de <code>t+1 … t+h</code>; as últimas 10 linhas ficam sem alvo (nunca preenchidas). Esta definição equivale à convenção do <code>M.py</code> (versão anterior da monografia). '
        'Usar a forma simples ou o excesso geométrico dá o mesmo rótulo. Testes verificam o alinhamento temporal: alterar o retorno de <code>t+1</code> ou <code>t+h</code> pode mudar <code>Y_t</code>; alterar o de <code>t</code> ou <code>t+h+1</code> não muda.</p>')
    ct = pd.crosstab(cmpw.y_original, cmpw.y_new)
    agree = float((cmpw.y_original == cmpw.y_new).mean())
    add(f'<h3>Alvo original contra alvo formal ({len(cmpw)} janelas)</h3>')
    add(table(["original ↓ · formal →", "0", "1"], [["0", str(ct.loc[0, 0]), str(ct.loc[0, 1])], ["1", f"<b>{ct.loc[1, 0]}</b>", str(ct.loc[1, 1])]], num=(1, 2)))
    add(f'<p>Concordância de {pc(agree)}. O alvo original é um <b>superconjunto</b> do formal: há {ct.loc[1,0]} casos só no original e nenhum só no formal, porque cada janela retroativa contém dias do passado.</p>')
    add(table(["Conjunto (split 60/20/20 de referência)", "Amostras", "Alvo original", "Alvo formal"],
              [[n_, str(c), pc(by_split.loc[k, "y_original"]), pc(by_split.loc[k, "y_new"])] for n_, k, c in (("Treino", "treino", 2442), ("Validação", "validacao", 814), ("Teste", "teste", 815))], num=(1, 2, 3)))
    add('<div class="chart">' + legend([("Alvo original", "var(--ref)"), ("Alvo formal", "var(--m1)")]) + svg_prev_year(by_year) + "</div>")
    add('<p class="cap">Anos inteiros sem nenhum positivo (2009, 2013, 2014, 2021 no alvo formal) e prevalências de 28 a 31 % em 2019, 2023 e 2025: o problema é fortemente não estacionário, e isso não será corrigido ajustando o alvo.</p>')

    # ============================================================== fase 4
    add('<h2 id="f4"><span class="n">Fase 4</span>Engenharia de variáveis</h2>')
    add('<p>A lógica da monografia foi mantida: as mesmas 12 variáveis, as mesmas janelas e as mesmas fórmulas. Todas usam apenas observações até <code>t</code>, o que é verificado por um teste de perturbação (alterar dados de <code>t+1</code> em diante não muda nenhuma variável em <code>t</code>).</p>')
    add(table(["Grupo", "Variável", "Fórmula", "Observação"], [
        ["Crédito", "<code>ida_ret</code>", "retorno diário do IDA‑DI", ""],
        ["Crédito", "<code>duration</code>", "duration média do índice", "ausente nas 59 primeiras linhas (não preenchida)"],
        ["Crédito", "<code>ipca_deb_ret</code>", "retorno diário das debêntures IPCA", "convertido de % para decimal"],
        ["Mercado", "<code>ibov_ret</code>", "retorno diário do Ibovespa", "0 = sem pregão"],
        ["Mercado", "<code>cdi_ret</code>", "CDI diário", ""],
        ["Excesso", "<code>excesso_dia</code>", "IDA − CDI (forma simples, como no original)", ""],
        ["Volatilidade", "<code>vol_20</code>", "desvio‑padrão de 20 dias do excesso", ""],
        ["Tendência", "<code>trend_20</code>", "média de 20 dias do retorno do IDA", ""],
        ["Retorno", "<code>ret_20</code>", "soma de 20 dias do excesso", "soma simples"],
        ["Momentum", "<code>rsi_14</code>", "RSI de 14 dias (média simples)", "aplicado ao retorno diário, não ao nível"],
        ["Técnica", "<code>macd</code>, <code>macd_signal</code>", "EMA12 − EMA26 e EMA9", "aplicado ao retorno diário"]]))
    add('<h3>O que foi ajustado</h3>')
    add(table(["#", "Original", "Corrigido", "Motivo"], [
        ["F1", "<code>fillna(0)</code> em todas as variáveis", "Sem preenchimento; aquecimento vem das 59 linhas anteriores", "O zero criava valores atípicos nas ~20 primeiras linhas"],
        ["F2", "RSI sem perdas → NaN → 0", "RSI = 100 sem perdas (50 sem ganhos nem perdas)", "Correção defensiva; <b>não ocorre nesta amostra</b>"],
        ["F3", "<code>StandardScaler</code> na base inteira", "Ajustado só nas linhas de treino", "Vazamento de média e desvio de validação e teste"],
        ["F4", "<code>ewm</code> sem <code>min_periods</code>", "<code>min_periods</code> igual ao período", "Primeiros valores do MACD vinham de 1–2 observações"],
        ["F5", "<code>nan_to_num</code> em toda a matriz agregada", "Só assimetria e curtose de janela constante viram 0 (0,9 %)", "Evita esconder outros NaN"]]))
    add('<h3>Efeito de ajustar o scaler na base inteira</h3>')
    add('<p>Média e desvio‑padrão do treino contra os da base inteira (fronteira de treino provisória do momento da análise). “Razão de dp” maior que 1 significa que o original comprimia a variável usando informação de períodos posteriores.</p>')
    add(table(["Variável", "Média (treino)", "Média (base inteira)", "Deslocamento (dp do treino)", "Razão de dp"],
              [[f"<code>{r.variavel}</code>", pt(r.media_treino, 4), pt(r.media_base_inteira, 4), pt(r.deslocamento_em_dp_treino, 2, True), pt(r.razao_dp, 2)] for r in leak.itertuples()], num=(1, 2, 3, 4)))
    add(callout("ok", "Equivalência com o original", "<p>As variáveis de nível, <code>vol_20</code>, <code>trend_20</code>, <code>ret_20</code> e o RSI são idênticas ao original onde ele tinha valor. MACD e <code>macd_signal</code> diferem no início da amostra (até ~2e‑5) porque a média móvel agora começa 59 linhas antes e converge; após 150 linhas a diferença é menor que 4e‑10. Nenhuma das 12 variáveis fica ausente na amostra original.</p>"))
    add('<p>RF e Perceptron recebem 120 atributos por janela (10 estatísticas × 12 variáveis: média, desvio, mínimo, máximo, mediana, assimetria, curtose, P25, P75, último valor); o LSTM recebe a sequência 100 × 12 padronizada.</p>')

    # ============================================================== fase 5
    add('<h2 id="f5"><span class="n">Fase 5</span>Divisão temporal e proteção do teste</h2>')
    add('<p>Cada amostra tem uma data de decisão <code>t</code>: variáveis nas linhas <code>t−99 … t</code>; alvo nos retornos <code>t+1 … t+10</code>. A divisão é cronológica, na proporção da monografia, sem embaralhar.</p>')
    sp = split.set_index("bloco")
    add('<div class="chart">' + svg_timeline([
        ("aquecimento", 59, "var(--ref)", ""), ("janela", 99, "var(--ref)", ""), ("treino", int(sp.loc["fit", "n"]), "var(--m1)", "treino"), ("purge", 9, "var(--bad)", ""),
        ("validação", int(sp.loc["val", "n"]), "var(--m2)", "validação"), ("purge", 9, "var(--bad)", ""), ("teste", int(sp.loc["test", "n"]), "var(--m3)", "teste"), ("horizonte", 10, "var(--ref)", "")]) + "</div>")
    add('<p class="cap">Linhas da base bruta (4240) por função. Cinza: linhas que só aquecem indicadores, completam a primeira janela ou formam o horizonte do último alvo. Vermelho: amostras removidas por purge.</p>')
    add(table(["Bloco", "Amostras", "Primeira decisão", "Última decisão", "Último dia usado no alvo", "Positivos", "Taxa"],
              [[{"fit": "Treino", "val": "Validação", "test": "Teste", "purge": "Purge"}[r.bloco], str(r.n), r.primeira_decisao, r.ultima_decisao, r.fim_alvo_max, str(r.positivos), pc(r.taxa_positivos)] for r in split.itertuples()], num=(1, 5, 6)))
    add('<h3>Purge</h3><p>Uma amostra do bloco anterior é removida se o seu alvo usa retornos <b>posteriores</b> à primeira data de decisão do bloco seguinte. Com horizonte 10 isso remove 9 amostras no fim do treino e 9 no fim da validação (18 no total; 4072 amostras disponíveis, 4054 utilizáveis). '
        'As <i>variáveis</i> de uma amostra de validação ou teste incluem dias do período anterior, o que não é vazamento (são dados passados, conhecidos em <code>t</code>); o risco está nos rótulos e nas estatísticas ajustadas.</p>')
    add('<h3>Por que a validação não foi dividida em duas metades</h3><p>O plano pedia calibração e escolha do limiar em dados separados. Com duas metades contíguas, a de calibração ficaria com 110 eventos (2019–2020) e a de limiar com apenas 12, insuficientes. '
        'A validação ficou como um bloco só e a separação é feita na Fase 7 por validação cruzada em blocos.</p>')
    add(table(["#", "No original", "Agora"], [
        ["L1", "<code>StandardScaler</code> ajustado na base inteira", "Ajustado só com linhas até a última decisão do treino"],
        ["L2", "Sem purge", "Purge automático (18 amostras)"],
        ["L3", "RF: calibrador e limiar na mesma validação", "Validação cruzada em blocos (Fase 7)"],
        ["L4", "LSTM: parada antecipada, calibração e limiar na mesma validação", "Parada antecipada em <code>val</code>; calibração e limiar como em L3"],
        ["L5", "Backtest em todo o período (≈ 80 % dentro da amostra)", "Backtest só no bloco de teste"],
        ["L6", "Nenhum controle de acesso ao teste", "<code>TestSetGuard</code>: uma liberação por modelo, registrada em arquivo"]]))
    add(callout("warn", "Riscos que permanecem (não são vazamento)",
                f"<ul><li>Poucos eventos: {int(sp.loc['fit','positivos'])} no treino, {int(sp.loc['val','positivos'])} na validação e {int(sp.loc['test','positivos'])} no teste.</li>"
                f"<li>A taxa de eventos vai de {pc(sp.loc['fit','taxa_positivos'])} (treino) a {pc(sp.loc['test','taxa_positivos'])} (teste): calibração e limiar aprendidos em um regime podem não valer no seguinte.</li>"
                "<li>Janelas consecutivas compartilham 99 de 100 dias; o número efetivo de observações independentes é bem menor que o tamanho da amostra.</li></ul>"))

    # ============================================================== fase 6
    add('<h2 id="f6"><span class="n">Fase 6</span>Modelos</h2>')
    add('<p>Os parâmetros da monografia foram mantidos e <b>nenhum hiperparâmetro foi ajustado</b>. Random Forest e LSTM foram treinados com 5 sementes (42 a 46) em vez de uma só. Versões: Python 3.11, TensorFlow 2.20, Keras 3.11, scikit‑learn 1.7.2.</p>')
    rfp = tr6["rf"]["params"]
    add(table(["", "Perceptron", "Random Forest", "LSTM"], [
        ["Entrada", "120 estatísticas por janela", "120 estatísticas por janela", "sequência 100 × 12"],
        ["Estrutura", "camada única, degrau, pesos iniciais 0", f"{rfp['n_estimators']} árvores, <code>min_samples_leaf={rfp['min_samples_leaf']}</code>, <code>max_features=sqrt</code>", "LSTM(64) → Dropout(0,3) → LSTM(32) → BatchNorm → Dense(32, ReLU) → Dropout(0,2) → Dense(1, sigmoide); 32 897 parâmetros"],
        ["Treino", "taxa 0,01; 100 épocas; sem pesos de classe", f"<code>class_weight={rfp['class_weight']}</code>", "Adam 1e‑3; até 50 épocas; lote 32; pesos de classe balanceados; parada antecipada (paciência 5) e redução de taxa"],
        ["Calibração", "Platt sobre o score (acréscimo)", "isotônica", "Platt"],
        ["Sementes", "1", "5", "5"]]))
    add('<p>O Perceptron foi acrescido de um score contínuo (<code>w·x + b</code>) para curvas ROC e PR, em vez de fabricar curvas a partir de saídas binárias, como o original fazia.</p>')
    pe = tr6["perceptron"]
    rows6 = [["Perceptron", pt(pe["auc_fit"], 3), pt(pe["auc_val"], 3), f"erros por época: {pe['errors_first_epoch']} → {pe['errors_last_epoch']}"]]
    for k, nm in (("rf", "Random Forest"), ("lstm", "LSTM")):
        a, b = [r["auc_fit"] for r in tr6[k]["runs"]], [r["auc_val"] for r in tr6[k]["runs"]]
        rows6.append([nm, f"{pt(np.mean(a),3)} ± {pt(np.std(a, ddof=1),3)}", f"{pt(np.mean(b),3)} ± {pt(np.std(b, ddof=1),3)}", "5 sementes"])
    add('<h3>Diagnóstico do treino (sem tocar o teste)</h3>')
    add(table(["Modelo", "AUC em treino", "AUC em validação", "Observação"], rows6, num=(1, 2)))
    add(callout("info", "Leitura", "<p>RF e LSTM sobreajustam muito (AUC de treino próximo de 1) com apenas 80 positivos de treino. O LSTM atinge o melhor <code>val_loss</code> entre a 2ª e a 5ª época, e a dispersão entre sementes é grande (AUC em validação de 0,62 a 0,70). "
                "O AUC em validação não é a medida final, porque a validação foi usada para a parada antecipada.</p>"))
    add('<details><summary>LSTM por semente</summary><div>' + table(["Semente", "Épocas", "Melhor época", "Melhor val_loss", "AUC treino", "AUC validação"],
        [[str(r["seed"]), str(r["epochs_run"]), str(r["best_epoch"]), pt(r["best_val_loss"], 4), pt(r["auc_fit"], 3), pt(r["auc_val"], 3)] for r in tr6["lstm"]["runs"]], num=(0, 1, 2, 3, 4, 5)) + "</div></details>")

    # ============================================================== fase 7
    add('<h2 id="f7"><span class="n">Fase 7</span>Calibração e escolha do limiar</h2>')
    add('<ol class="l"><li>Modelo treinado no bloco de treino.</li><li>Pontuações brutas na validação.</li>'
        '<li><b>Calibração por validação cruzada em blocos</b> dentro da validação (3 dobras contíguas, com purge de 9 amostras de cada lado): cada observação é calibrada por um calibrador que não a viu.</li>'
        '<li>Limiares escolhidos sobre essas probabilidades fora da amostra.</li><li>Calibrador final ajustado em toda a validação.</li>'
        f'<li>Tudo congelado (SHA‑256 <code>{frozen["sha256"][:16]}…</code>) e só então o teste foi liberado, uma vez por modelo e semente.</li></ol>')
    add('<p>Regras de limiar avaliadas: <code>fixed_0.5</code>; <code>f1</code> (máximo F1, como na monografia); <code>value</code> (maximiza o ganho acumulado de migrar para o CDI). '
        'Procedimentos: <code>crossfit</code> (limpo) e <code>original_style</code> (calibra e escolhe o limiar na mesma validação). O alerta é emitido quando <code>p ≥ limiar</code> (o original usava <code>&gt;</code>, deixando de fora o próprio ponto ótimo). '
        '<b>Critério principal, declarado antes de abrir o teste:</b> <code>crossfit</code> + <code>f1</code>.</p>')
    add('<h3>Custos assimétricos dos erros</h3><p>Medidos em treino e validação (nunca no teste), em retorno em excesso dos 10 dias seguintes. Falso positivo: migrar para o CDI quando o IDA ganharia. Verdadeiro positivo: perda evitada. Falso negativo: perda não evitada.</p>')
    add(table(["Bloco", "Positivos", "Negativos", "Ganho por VP = custo de FN", "Custo por FP", "Razão FN/FP", "Precisão de equilíbrio"],
              [[{"fit": "Treino", "val": "Validação"}[i], str(int(r.n_pos)), str(int(r.n_neg)), pc(r.gain_per_TP, 2), pc(r.cost_per_FP, 2), pt(r.ratio_FN_over_FP, 1), pc(r.breakeven_precision)] for i, r in cost.iterrows()], num=(1, 2, 3, 4, 5, 6)))
    add('<p class="cap">Abaixo da precisão de equilíbrio, alertar não compensa. Um falso negativo custa de 2 a 4 vezes um falso positivo em retorno esperado, mas os eventos são raros.</p>')
    add('<h3>O efeito do procedimento na validação</h3>')
    rows7 = []
    for k, nm in KIND.items():
        for proc in ("crossfit", "original_style"):
            x = p7v[(p7v.model.str.startswith(k)) & (p7v.procedure == proc) & (p7v.rule == "f1")]
            rows7.append([nm, proc, pt(x.val_brier.mean(), 3), pt(x.val_f1.mean(), 3), pc(x.val_alert_rate.mean(), 0)])
    add(table(["Modelo", "Procedimento", "Brier (validação)", "F1 (validação)", "Taxa de alerta"], rows7, num=(2, 3, 4)))
    add('<p class="cap">Referências na validação: Brier de prever só a prevalência = 0,129; F1 de alertar sempre = 0,263.</p>')
    add(callout("warn", "O achado desta fase", "<p>Com a calibração honesta, o Brier de RF e LSTM na validação é <b>pior</b> que o de uma constante, e a regra de F1 vira “alertar quase sempre”. O procedimento original parece muito melhor (Brier ≈ 0,12) porque calibra e mede nos mesmos dados. "
                "As dobras da validação têm regimes muito diferentes (110 dos 122 eventos estão em 2019–2020), então o calibrador treinado nas outras dobras não generaliza: é a mudança de regime, não um defeito do cross‑fit.</p>"))

    # ============================================================== fase 8
    add('<h2 id="f8"><span class="n">Fase 8</span>Avaliação e comparação dos modelos</h2>')
    add(f'<p>Teste com 815 amostras e 169 eventos ({pc(base["prevalence"])}). Intervalos de 95 % por bootstrap em blocos móveis de 20 dias (1000 reamostragens, mesmas datas para todas as sementes). Média entre sementes: RF e LSTM, 5; Perceptron, 1.</p>')
    add('<div class="chart">' + legend([(m, M_COL[m]) for m in MODELS]) + svg_ci(ci) + "</div>")
    add('<p class="cap">Pontos: média entre sementes; barras: intervalo de 95 %. Linha tracejada: referência (acaso no AUC; previsão constante da prevalência no Brier; alertar todo dia no F1).</p>')
    rows8 = []
    for m in MODELS:
        a, b_, f_ = g(m, "auc"), g(m, "brier"), g(m, "f1")
        rows8.append([m, f"{pt(a.ponto,3)} [{pt(a.ic_baixo,3)}; {pt(a.ic_alto,3)}]", f"{pt(b_.ponto,3)} [{pt(b_.ic_baixo,3)}; {pt(b_.ic_alto,3)}]", f"{pt(f_.ponto,3)} [{pt(f_.ic_baixo,3)}; {pt(f_.ic_alto,3)}]"])
    add(table(["Modelo", "AUC", "Brier", "F1 (regra principal)"], rows8, num=(1, 2, 3)))
    add(table(["Diferença de AUC", "Estimativa [IC 95 %]"], [[r.diff, f"{pt(r.point,3,True)} [{pt(r.ci_low,3,True)}; {pt(r.ci_high,3,True)}]"] for r in pair.itertuples()], num=(1,)))
    add(callout("info", "Leitura", f"<p>Todos os intervalos de AUC contêm 0,50, nenhum Brier é distinguível da previsão constante ({pt(base['brier_constant_prevalence'],3)}) e o F1 de RF e LSTM coincide com o de alertar sempre ({pt(base['f1_always_alert'],3)}). "
                "Nenhum modelo se distingue do acaso. Entre os modelos, a única diferença com intervalo fora de zero é RF − Perceptron (+0,133; IC 95 %: +0,009 a +0,269), que vem em boa parte de o Perceptron ficar abaixo de 0,50 no ponto e é evidência fraca (três comparações, sem correção para múltiplas comparações); a LSTM não se distingue de nenhum dos dois. A prevalência do teste só é conhecida depois; é uma linha de base ideal.</p>"))
    add('<h3>Figuras do teste</h3>')
    add(figure("confusion_counts.png", "Matrizes de confusão (semente 42, escolhida de antemão). Em cima, a regra principal; embaixo, o estilo original. A regra principal alerta todos os dias no RF e no LSTM.", "Matrizes de confusão no teste"))
    add(figure("confusion_normalized.png", "As mesmas matrizes, com as linhas normalizadas.", "Matrizes de confusão normalizadas"))
    add(figure("roc_curve.png", "Curvas ROC, uma linha por semente. A diagonal é o acaso.", "Curvas ROC"))
    add(figure("pr_curve.png", "Curvas precisão × recall, uma linha por semente. A linha tracejada é a prevalência.", "Curvas precisão e recall"))
    add(figure("calibration_curve.png", "Calibração no teste, uma linha por semente (faixas sem dados omitidas).", "Curvas de calibração"))
    add(figure("real_vs_predicted.png", "Real contra alerta nas 200 primeiras amostras do teste (semente 42).", "Real contra previsto"))

    add('<h3>Reconciliação com os números da monografia</h3><p>O passo S0 executa o código original, literalmente, sobre os mesmos dados.</p>')
    rr = []
    for r in recon.itertuples():
        integer = r.metrica == "Positivos no teste"
        rr.append([r.modelo, r.metrica, str(int(r.monografia)) if integer else pt(r.monografia, 3), str(int(r.reproduzido)) if integer else pt(r.reproduzido, 3), str(int(r.sementes))])
    add(table(["Modelo", "Métrica", "Monografia", "Reproduzido (S0)", "Sementes"], rr, num=(2, 3, 4)))
    add('<ul><li><b>Random Forest:</b> reproduzido aproximadamente (mesma prevalência de teste de 251 positivos). A diferença restante cabe na versão das bibliotecas e na base original, que não temos.</li>'
        '<li><b>LSTM:</b> reprodução apenas qualitativa; o valor publicado é uma execução sem semente nem versão de TensorFlow registradas.</li>'
        '<li><b>Perceptron:</b> a figura publicada tem mais positivos do que qualquer variante do código gera. Foram testadas 168 combinações de coluna, horizonte e janela do alvo e nenhuma produz 314. A origem fica registrada como desconhecida.</li></ul>')

    add('<h3>Decomposição: de onde vem a diferença</h3>')
    add('<p>Cada passo acrescenta <b>uma</b> correção ao anterior, nas mesmas 4071 janelas do original. AUC médio entre sementes (RF 5, LSTM 3, Perceptron 1); barras: ± 1 desvio‑padrão entre sementes.</p>')
    add('<div class="chart">' + legend([(m, M_COL[m]) for m in MODELS]) + svg_steps(ab, STEPS) + "</div>")
    add('<p class="cap">S0 original · S1 + alvo formal · S2 + scaler ajustado só no treino · S3 + variáveis sem preenchimento com zero · S4 + purge · S5 + cross‑fit.</p>')
    for k, nm in KIND.items():
        d = ablc[ablc.kind == k]
        rows = []
        for st in STEPS:
            x = d[d.step == st]
            if x.empty:
                continue
            f = lambda c, dd=3: (f"{pt(x[c].mean(), dd)} ± {pt(x[c].std(ddof=1), dd)}" if len(x) > 1 and x[c].notna().any() else (pt(x[c].mean(), dd) if x[c].notna().any() else "–"))
            rows.append([STEP_LABEL[st], f("auc_rep"), f("brier"), f("precision"), f("recall"), f("f1"), str(int(x.pos_test.iloc[0]))])
        add(f"<h4>{nm}</h4>" + table(["Passo", "AUC", "Brier", "Precisão", "Recall", "F1", "Positivos no teste"], rows, num=(1, 2, 3, 4, 5, 6)))
    add(callout("bad", "A principal causa da diferença", f"<p>Só trocar o alvo original pelo formal derruba o AUC do RF de {pt(auc_s['Random Forest'][0])} para {pt(auc_s['Random Forest'][1])}, do LSTM de {pt(auc_s['LSTM'][0])} para {pt(auc_s['LSTM'][1])} e do Perceptron de {pt(auc_s['Perceptron'][0])} para {pt(auc_s['Perceptron'][1])}. "
                "Parte do rótulo original era informação passada que o modelo já via.</p>"))
    add('<p>As demais correções movem o AUC bem menos, em magnitudes da ordem do desvio‑padrão entre sementes; não foi feito um teste formal de cada passo. O cross‑fit muda a regra de decisão (F1 → alertar sempre), não a capacidade de ordenar.</p>')
    add(callout("warn", "Sobre o passo S5", "<p>O S5 não é idêntico ao pipeline final da Fase 7. Ele retreina os modelos com 4071 janelas (as do original) e mede o AUC sobre a probabilidade calibrada; a Fase 7 usa 4072 janelas e o score bruto (RF 0,54 contra 0,58). "
                "A distância mostra o quanto o AUC varia com pequenas mudanças do pipeline, o que reforça que as diferenças entre S2 e S5 estão dentro do ruído.</p>"))
    add('<h3>Outras regras de decisão no teste (média entre sementes)</h3>')
    rows = []
    for k, nm in KIND.items():
        for (proc, rule), x in p7[p7.kind == k].groupby(["procedure", "rule"]):
            star = " ★" if (proc == frozen["primary"]["procedure"] and rule == frozen["primary"]["threshold_rule"]) else ""
            rows.append([nm, proc, rule + star, pt(x.precision.mean(), 3), pt(x.recall.mean(), 3), pt(x.f1.mean(), 3), pc(x.alert_rate.mean(), 0)])
    add(table(["Modelo", "Procedimento", "Regra (★ principal)", "Precisão", "Recall", "F1", "Alerta"], rows, num=(3, 4, 5, 6)))
    add('<p class="cap">O limiar 0,50 nunca emite alerta: probabilidades calibradas não chegam a 0,5 num problema com prevalência de 3 a 20 %.</p>')

    # ============================================================== fase 9
    add('<h2 id="f9"><span class="n">Fase 9</span>Backtest</h2>')
    add('<h3>Regras de execução</h3><ul><li><b>Período:</b> apenas o bloco de teste (815 dias).</li><li><b>Sinal:</b> a decisão em <code>t</code> usa informação até o fechamento de <code>t</code>; a posição vale para o retorno de <code>t+1</code>. Alerta → CDI; sem alerta → IDA; reavaliado todo dia.</li>'
        '<li><b>Custos:</b> 0, 10 e 25 pontos‑base por troca de posição (suposição minha; o IDA é uma carteira teórica, não negociável). Impostos não considerados.</li>'
        '<li><b>Sinais avaliados, rotulados, nenhum escolhido depois de ver o resultado:</b> <code>principal</code> (pré‑declarado), <code>valor</code> e <code>estilo_original</code>. RF e LSTM: 5 sementes; Perceptron entra como extra.</li></ul>')
    add('<h3>Benchmarks no mesmo intervalo</h3>')
    add(table(["", "Base 100 final", "Retorno anual", "Vol. anual", "Máx. drawdown"],
              [[i, pt(r.final_base100, 1), pc(r.retorno_anual), pc(r.vol_anual, 2), pc(r.max_drawdown, 2)] for i, r in bench.iterrows()], num=(1, 2, 3, 4)))
    add(callout("warn", "O IDA rendeu mais que o CDI no período de teste", "<p>Qualquer estratégia que fique no IDA “supera o CDI” sem prever nada. Por isso a comparação relevante é com o IDA, não com o CDI.</p>"))
    z = np.load(ROOT / "data" / "processed" / "phase9_curves.npz", allow_pickle=True)
    dates = [str(d) for d in z["dates"]]
    def mean_curve(kind, rule):
        return np.mean([z[k] for k in z.files if k.startswith(f"{kind}|") and k.endswith(f"|{rule}")], axis=0)
    add('<h3>Curvas no teste (média das sementes, sem custos)</h3><div class="chart">' + svg_equity(dates, {
        "IDA": (z["ida"], "var(--fg2)", ""), "CDI": (z["cdi"], "var(--ref)", "5 4"),
        "LSTM · estilo original": (mean_curve("lstm", "estilo_original"), "var(--m3)", ""),
        "RF · estilo original": (mean_curve("rf", "estilo_original"), "var(--m2)", ""),
        "RF · regra principal (= CDI)": (mean_curve("rf", "principal"), "var(--m1)", "2 3")}, None) + "</div>")
    add('<p class="cap">A regra de valor sobrepõe a curva do IDA (quase nunca alerta); a regra principal sobrepõe a do CDI (alerta o tempo todo).</p>')
    rows = []
    for r in bt[bt.custo_bps == 0].itertuples():
        c = bci[(bci.tipo == r.tipo) & (bci.regra == r.regra)].iloc[0]
        rows.append([KIND[r.tipo] + (" (extra)" if r.tipo == "perceptron" else ""), r.regra, f"{pt(r.final_media,1)} ({pt(r.final_min,1)}–{pt(r.final_max,1)})", pt(r.trocas_media, 1), pc(r.dias_cdi, 0),
                     f"{pt(100*c.exc_cdi_anual,2,True)} p.p. [{pt(100*c.ic_cdi_baixo,2,True)}; {pt(100*c.ic_cdi_alto,2,True)}]", f"{pt(100*c.exc_ida_anual,2,True)} p.p. [{pt(100*c.ic_ida_baixo,2,True)}; {pt(100*c.ic_ida_alto,2,True)}]"])
    add('<h3>Estratégias (sem custos)</h3>' + table(["Modelo", "Regra", "Base 100 final (mín.–máx.)", "Trocas", "Dias no CDI", "Excesso anual vs CDI [IC 95 %]", "Excesso anual vs IDA [IC 95 %]"], rows, num=(2, 3, 4)))
    add('<p class="cap">IC 95 % por bootstrap em blocos de 20 dias sobre o excesso diário médio entre sementes, anualizado em 252 dias.</p>')
    rows = []
    for (t_, r_), gg in bt.groupby(["tipo", "regra"]):
        gg = gg.set_index("custo_bps")
        rows.append([KIND[t_], r_, pt(gg.loc[0, "final_media"], 1), pt(gg.loc[10, "final_media"], 1), pt(gg.loc[25, "final_media"], 1), pt(gg.loc[0, "trocas_media"], 1)])
    add('<h3>Sensibilidade a custos (base 100 final)</h3>' + table(["Modelo", "Regra", "0 bps", "10 bps", "25 bps", "Trocas"], rows, num=(2, 3, 4, 5)))
    add('<ul><li><b>Regra principal:</b> fica 100 % no CDI em RF e LSTM e, por construção, perde para o IDA.</li><li><b>Regra de valor:</b> quase nunca alerta e reproduz o IDA.</li>'
        '<li><b>Estilo original:</b> troca de posição com frequência. Sem custos o resultado sobre o IDA é pequeno (negativo no RF, positivo no LSTM) e o intervalo inclui zero; com 10 bps o RF fica abaixo do IDA e o LSTM empata, e com 25 bps os dois ficam abaixo, principalmente o RF.</li>'
        '<li><b>Nenhuma estratégia baseada em modelo supera o IDA de forma estatisticamente distinguível</b>, com ou sem custos, o que é coerente com o AUC próximo de 0,5 da Fase 8.</li></ul>')
    add('<h3>O backtest original da monografia, decomposto</h3><p>A Fig. 7 foi recalculada a partir dos alertas salvos pelos scripts originais. A série reproduz <b>exatamente</b> as colunas do Excel (RF 570,9; LSTM 572,8; IDA 534,1; CDI 443,4 em base 100). '
        'O gráfico da monografia cobre o período inteiro, que inclui o treino e a validação (usados para ajustar o modelo e o limiar).</p>')
    rows = []
    for r in orig[~orig.trecho.str.startswith("conferência")].itertuples():
        rows.append([r.modelo, r.trecho, str(int(r.dias)), pt(r.estrategia_base100, 1), pt(r.ida_base100, 1), pt(r.cdi_base100, 1), pt(r.ganho_vs_ida_pp, 1, True), pc(r.taxa_alerta, 0), str(int(r.trocas))])
    add(table(["Modelo", "Trecho", "Dias", "Estratégia", "IDA", "CDI", "Ganho vs IDA (p.p.)", "Alerta", "Trocas"], rows, num=(2, 3, 4, 5, 6, 7, 8)))
    add(figure("backtest_original_split.png", "Ganho sobre o IDA, por trecho, nos alertas originais da monografia.", "Ganho sobre o IDA por trecho no backtest original"))
    add('<p>O ganho sobre o IDA do período inteiro vem em boa parte da <b>validação</b> (2019–2022, que contém a crise de 2020 e foi usada para escolher o limiar). No treino o ganho é próximo de zero (negativo no LSTM) e no teste original é pequeno (RF) ou depende de poucas trocas (LSTM, 6 trocas, uma única execução). '
        'O ganho sobre o <b>CDI</b> existe em todos os trechos porque, nesses anos, o IDA rendeu mais que o CDI.</p>')
    add('<h3>Conferência independente em planilha</h3>')
    add(f'<p><code>outputs/backtest_check.xlsx</code> reproduz três estratégias (RF e LSTM no estilo original e LSTM de valor, semente 42, custo 10 bps) com fórmulas do Excel: troca, retorno, base 100, pico e drawdown, mais uma coluna de conferência. '
        f'As fórmulas foram recalculadas com um motor independente (biblioteca <code>formulas</code>, não o Excel) e a diferença máxima em relação ao Python foi {xl.dif_abs.max():.1e}. <b>O Excel em si não foi aberto</b>; ao abrir a planilha, a coluna de conferência deve mostrar VERDADEIRO.</p>')

    # ============================================================== monografia
    add('<h2 id="monografia"><span class="n">Síntese</span>Pontos da monografia à luz dos resultados</h2>')
    add(table(["Ponto da monografia", "Com a metodologia revisada"], [
        ["RF e LSTM superam buy‑and‑hold no IDA e CDI contínuo, preservando capital", '<span class="tag warn">não se replica</span> O ganho do período inteiro vem em boa parte da validação e depende de um alvo com sobreposição; no teste não se distingue do IDA'],
        ["LSTM tem desempenho “superior em equilíbrio entre precisão e recall”, com “F1 elevado”", '<span class="tag warn">não se replica</span> O F1 publicado é 0,389, menor que o do RF (0,562); o recall é 24 %'],
        ["Perceptron descartado por baixo desempenho", '<span class="tag warn">sem base estatística</span> Com o método revisado nenhum modelo se distingue do acaso e as diferenças entre eles são pequenas'],
        ["LSTM bem calibrado (Fig. 5.5)", '<span class="tag warn">não se replica</span> A curva fica bem acima da diagonal: o modelo subestima o risco'],
        ["RF “moderadamente bem calibrado” (Fig. 6.4)", '<span class="tag warn">frágil</span> Curva com 6 pontos, dois em 1,0 e queda no último'],
        ["Prioridade é maximizar o recall", '<span class="tag warn">texto × código</span> O limiar foi escolhido por F1'],
        ["Brier Score como métrica", '<span class="tag warn">omitido</span> Nunca aparece nos resultados'],
        ["Perceptron: 69 % / 56 % de acertos (p. 30)", '<span class="tag warn">texto × figura</span> A Fig. 4.2 mostra 78 % / 45 %'],
        ["“Multilayer” Perceptron", '<span class="tag warn">terminologia</span> O código é de camada única'],
        ["Backtest: CDI 436, base 100 em nov/2009", '<span class="tag warn">não verificado</span> O Excel mostra base em ago–set/2009 e CDI 443 até nov/2025; 436 parece um corte em set/2025'],
        ["Numeração de figuras", '<span class="tag warn">revisar</span> A Fig. 5.2 é usada duas vezes, entre outras repetições']]))

    # ============================================================== limites
    add('<h2 id="limites"><span class="n">Limites</span>Limitações conhecidas</h2>')
    add('<h3>Limitações estatísticas e de desenho</h3><ul><li><b>Poder baixo:</b> 169 eventos no teste, 80 no treino, janelas que compartilham 99 de 100 dias.</li><li><b>Uma só divisão cronológica</b>, com a taxa de eventos variando de 3,3 % a 20,7 % entre blocos.</li>'
        '<li><b>Sem busca de hiperparâmetros, outros horizontes ou alvo contínuo</b>, para não misturar extensões com a reconstrução fiel.</li><li><b>Fiéis, mas questionáveis:</b> RSI e MACD sobre o retorno diário; <code>ret_20</code> como soma simples; horizonte único.</li>'
        '<li><b>Backtest:</b> custos supostos; o IDA não é negociável; um único período de teste em que o IDA superou o CDI.</li></ul>')
    add('<h3>Questões em aberto</h3>')
    add(table(["Questão", "Situação"], [
        ["Origem da Fig. 4.1 (Perceptron, 314 positivos)", "O código gera 251; 168 variantes de alvo testadas; origem desconhecida"],
        ["Resultado exato do LSTM publicado", "Uma execução sem semente nem versão de TensorFlow; só reprodução qualitativa"],
        ["Planilha de conferência do backtest", "Gerada localmente e validada por motor de fórmulas em Python; não foi aberta no Excel"],
        ["Base <code>IDA_DI.xlsx</code> original", "Assumida equivalente a <code>Monografia - Dados.xlsx</code> pelas saídas dos scripts"]]))

    # ============================================================== decisões
    add('<h2 id="decisoes"><span class="n">Decisões</span>Decisões e suposições que podem ser contestadas</h2>')
    add(table(["#", "Decisão", "Alternativa"], [
        ["1", "Base primária <code>Dados-IDA.xls</code>; <code>IDA_DI.xlsx ≡ Monografia - Dados.xlsx</code>", "Usar só a base da monografia"],
        ["2", "Retorno do IDA pela coluna <code>IDA_DI</code>, não pelo nível do índice", "Índice (diferença acumulada de 0,2 %; muda o sinal do alvo em 14 dias)"],
        ["3", "Validação cruzada em blocos em vez de duas metades", "Duas metades (12 eventos para o limiar)"],
        ["4", "Critério principal: F1 com cross‑fit (confirmado pelo autor)", "Regra de valor; estilo original"],
        ["5", "LSTM com 5 sementes (3 na decomposição)", "Uma execução, como na monografia"],
        ["6", "Custos de 0, 10 e 25 bps por troca", "Outros custos ou custos de mercado reais"],
        ["7", "Perceptron incluído no backtest como extra", "Excluir, como a monografia"]]))

    # ============================================================== repro
    add('<h2 id="repro"><span class="n">Reprodução</span>Reprodutibilidade</h2>')
    add('<p>Python 3.11. Versões usadas: numpy 2.3.4, pandas 2.3.3, scipy 1.16.2, scikit‑learn 1.7.2, TensorFlow 2.20.0, Keras 3.11.3. O treino leva cerca de 15 minutos e a decomposição cerca de 25, em CPU.</p>')
    add('<pre><code>pip install -r requirements.txt\n'
        'python -I src/data/audit.py                  # base limpa e relatório de qualidade\n'
        'python -I src/targets/compare_targets.py     # alvo formal\n'
        'python -I src/features/compare_features.py   # variáveis e efeito do scaler\n'
        'python -I src/validation/build_split.py      # divisão com purge\n'
        'python -I src/models/train_all.py            # três modelos, 5 sementes\n'
        'python -I src/calibration/freeze.py          # congela calibradores e limiares (fit/val)\n'
        'python -I src/calibration/evaluate_test.py   # abre o teste uma vez por modelo\n'
        'python -I src/evaluation/evaluate.py         # figuras e intervalos\n'
        'python -I src/evaluation/ablation.py         # decomposição passo a passo\n'
        'python -I src/backtest/backtest.py &amp;&amp; python -I src/backtest/original_backtest.py\n'
        'python -I src/backtest/excel_check.py &amp;&amp; python -I src/backtest/figures_report.py\n'
        'python -m pytest -q tests                    # 66 testes</code></pre>')
    add('<p><code>evaluate_test.py</code> se recusa a reavaliar um modelo já registrado em <code>outputs/metrics/test_access_log.json</code>. Para repetir do zero é preciso apagar esse arquivo, sabendo que o teste está sendo reaberto.</p>')
    add(table(["Pasta", "Conteúdo"], [
        ["<code>original/</code>", "Scripts da monografia (sem correções), resultados originais e a metodologia original"],
        ["<code>data/</code>", "Bases brutas e previsões do teste"], ["<code>src/</code>", "data, targets, features, validation, models, calibration, evaluation, backtest, report"],
        ["<code>tests/</code>", "66 testes (dados, alvo, variáveis, divisão, modelos, calibração, backtest)"], ["<code>notebooks/</code>", "Três notebooks executados: dados, variáveis, resultados"],
        ["<code>outputs/</code>", "Figuras, tabelas, métricas (inclui <code>frozen_config.json</code> e <code>test_access_log.json</code>) e a planilha de conferência"], ["<code>docs/</code>", "Relatórios por fase, relatório geral, limitações e este relatório"]]))

    # ============================================================== apêndice
    add('<h2 id="apendice"><span class="n">Apêndice</span>Tabelas completas e glossário</h2>')
    add('<details><summary>Métricas por semente no teste, todas as regras (Fase 7)</summary><div>' + table(
        ["Modelo", "Procedimento", "Regra", "Limiar", "Precisão", "Recall", "F1", "AUC", "Brier", "Alerta"],
        [[r.model, r.procedure, r.rule, pt(r.threshold, 3), pt(r.precision, 3), pt(r.recall, 3), pt(r.f1, 3), pt(r.auc, 3), pt(r.brier, 3) if r.brier == r.brier else "–", pc(r.alert_rate, 0)] for r in p7.itertuples()],
        num=(3, 4, 5, 6, 7, 8, 9)) + "</div></details>")
    add('<details><summary>Backtest por semente (todas as regras e custos)</summary><div>' + table(
        ["Modelo", "Regra", "Custo (bps)", "Base 100 final", "Retorno anual", "Vol. anual", "Máx. drawdown", "Trocas", "Dias no CDI"],
        [[r.modelo, r.regra, str(int(r.custo_bps)), pt(r.final_base100, 2), pc(r.retorno_anual), pc(r.vol_anual, 2), pc(r.max_drawdown, 2), str(int(r.trocas)), pc(r.dias_no_cdi, 0)]
         for r in pd.read_csv(T / "phase9_backtest_by_seed.csv").itertuples()], num=(2, 3, 4, 5, 6, 7, 8)) + "</div></details>")
    add('<details><summary>Conferência da planilha: Excel (fórmulas) contra Python</summary><div>' + table(
        ["Item", "Fórmulas da planilha", "Python", "|Diferença|"], [[r.item, pt(r.excel_formulas, 6), pt(r.python, 6), f"{r.dif_abs:.1e}"] for r in xl.itertuples()], num=(1, 2, 3)) + "</div></details>")
    add('<details open><summary>Glossário</summary><div><ul>'
        '<li><b>IDA‑DI:</b> Índice de Debêntures ANBIMA indexadas ao DI (CDI); referência para o crédito privado corporativo.</li>'
        '<li><b>CDI:</b> taxa do Certificado de Depósito Interbancário; referência de baixo risco.</li>'
        '<li><b>AUC:</b> área sob a curva ROC; 0,5 é equivalente a um acaso, 1 é separação perfeita.</li>'
        '<li><b>Brier:</b> erro quadrático médio das probabilidades; menor é melhor.</li>'
        '<li><b>F1:</b> média harmônica de precisão e recall.</li>'
        '<li><b>Purge:</b> remoção de amostras cujo rótulo usa dados do bloco seguinte.</li>'
        '<li><b>Cross‑fit:</b> cada observação é calibrada por um calibrador ajustado sem ela.</li>'
        '<li><b>Vazamento (leakage):</b> uso, no treino ou na escolha de parâmetros, de informação que não estaria disponível na hora da decisão.</li>'
        '<li><b>Bootstrap em blocos:</b> reamostragem de trechos consecutivos de 20 dias, que preserva a autocorrelação.</li>'
        '<li><b>Ablação / decomposição:</b> aplicar as correções uma por vez para medir o efeito de cada uma.</li></ul></div></details>')
    add('<footer>Este relatório é uma simulação acadêmica e não constitui recomendação de investimento. Para citar um número, use as tabelas em <code>outputs/</code> e os documentos por fase em <code>docs/</code>.</footer>')
    add("</main></div>")

    page = ("<!doctype html><html lang=\"pt-BR\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Relatório completo da reconstrução da monografia</title><style>" + CSS + "</style></head><body>" + "".join(S) +
            "<button class=\"themebtn\" id=\"tb\" type=\"button\">Tema</button><script>(function(){var b=document.getElementById('tb'),r=document.documentElement;"
            "b.addEventListener('click',function(){var d=r.getAttribute('data-theme');var dark=d?d==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;r.setAttribute('data-theme',dark?'light':'dark');});})();</script></body></html>")
    out = ROOT / "docs" / "relatorio_completo.html"
    out.write_text(page, encoding="utf-8")
    print(out, round(len(page) / 1024), "KB")


if __name__ == "__main__":
    main()
