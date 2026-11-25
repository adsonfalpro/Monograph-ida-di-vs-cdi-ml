# Backtest (Fase 9)

Gerado por `src/backtest/figures_report.py`. Código: `src/backtest/`. Testes: `tests/test_backtest.py`. Planilha de conferência: `outputs/backtest_check.xlsx`.

## 1. Regras de execução

- **Período:** apenas o bloco de teste (815 dias; decisões de 2022‑08‑10 a 2025‑11‑05, retornos de 2022‑08‑11 a 2025‑11‑06).
- **Sinal:** a decisão em `t` usa informação até o fechamento de `t`; a posição vale para o retorno de `t+1`. Alerta → CDI; sem alerta → IDA. Reavaliado todo dia.
- **Custos:** 0, 10 e 25 bps por troca de posição (suposição minha, descontada no dia da troca). O IDA é uma carteira teórica, não negociável; esses custos são um proxy, não uma medida de mercado. Impostos não considerados.
- **Sinais avaliados (rotulados, nenhum escolhido depois de ver o resultado):** `principal` (cross‑fit + F1, pré‑declarada), `valor` (cross‑fit, regra de valor) e `estilo_original` (calibração e limiar na mesma validação, como na monografia). RF e LSTM: 5 sementes; Perceptron: 1 (extra, a monografia o descartou).
- **Retorno do IDA:** coluna `ida_ret` (decisão da Fase 2). Alternativa pelo nível do índice na tabela de benchmarks.

## 2. Resultado no teste (média das sementes, sem custos)

Benchmarks no mesmo intervalo:

| | base 100 final | retorno anual | vol. anual | máx. drawdown |
|---|---|---|---|---|
| IDA (comprar e manter) | 153,9 | 14,3 % | 0,75 % | -1,21 % |
| CDI 100 % | 147,6 | 12,8 % | 0,08 % | 0,00 % |
| IDA pelo nível do índice (alternativa) | 152,6 | 14,0 % | 0,83 % | -1,21 % |

Estratégias (custo 0):

| modelo | regra | base 100 final (mín.–máx. entre sementes) | trocas (média) | dias no CDI | excesso anual vs CDI [IC 95 %] | excesso anual vs IDA [IC 95 %] |
|---|---|---|---|---|---|---|
| LSTM | estilo_original | 155,2 (153,3–157,5) | 9,6 | 10% | +1.57 pp [+0.12; +2.85] | +0.27 pp [-0.08; +0.70] |
| LSTM | principal | 147,6 (147,6–147,6) | 1,0 | 100% | +0.00 pp [+0.00; +0.00] | -1.30 pp [-2.79; +0.48] |
| LSTM | valor | 153,8 (153,8–153,9) | 1,2 | 0% | +1.29 pp [-0.49; +2.78] | -0.01 pp [-0.02; -0.00] |
| Perceptron (extra) | estilo_original | 153,7 (153,7–153,7) | 24,0 | 22% | +1.26 pp [-0.09; +2.52] | -0.04 pp [-0.66; +0.76] |
| Perceptron (extra) | principal | 145,6 (145,6–145,6) | 10,0 | 66% | -0.41 pp [-1.50; +0.42] | -1.71 pp [-2.79; -0.64] |
| Perceptron (extra) | valor | 153,9 (153,9–153,9) | 0,0 | 0% | +1.30 pp [-0.48; +2.79] | +0.00 pp [+0.00; +0.00] |
| Random Forest | estilo_original | 151,0 (149,7–152,3) | 27,8 | 73% | +0.71 pp [+0.21; +1.26] | -0.58 pp [-1.86; +0.97] |
| Random Forest | principal | 147,6 (147,6–147,6) | 1,0 | 100% | +0.00 pp [+0.00; +0.00] | -1.30 pp [-2.79; +0.48] |
| Random Forest | valor | 153,9 (153,9–153,9) | 0,0 | 0% | +1.30 pp [-0.48; +2.79] | +0.00 pp [-0.00; +0.00] |

IC 95 % por bootstrap em blocos de 20 dias sobre o excesso diário médio entre sementes, anualizado (252 dias).

### Sensibilidade a custos (base 100 final, média das sementes)

| modelo | regra | 0 bps | 10 bps | 25 bps | trocas |
|---|---|---|---|---|---|
| LSTM | estilo_original | 155,2 | 153,7 | 151,5 | 9,6 |
| LSTM | principal | 147,6 | 147,4 | 147,2 | 1,0 |
| LSTM | valor | 153,8 | 153,6 | 153,4 | 1,2 |
| Perceptron (extra) | estilo_original | 153,7 | 150,0 | 144,7 | 24,0 |
| Perceptron (extra) | principal | 145,6 | 144,2 | 142,0 | 10,0 |
| Perceptron (extra) | valor | 153,9 | 153,9 | 153,9 | 0,0 |
| Random Forest | estilo_original | 151,0 | 146,9 | 140,9 | 27,8 |
| Random Forest | principal | 147,6 | 147,4 | 147,2 | 1,0 |
| Random Forest | valor | 153,9 | 153,9 | 153,9 | 0,0 |

Figuras: `outputs/figures/backtest_curves.png` e `backtest_original_split.png`.

## 3. O que o backtest mostra

1. **No período de teste o IDA rendeu mais que o CDI** (veja os benchmarks). Qualquer estratégia que fique no IDA 'supera o CDI' sem prever nada. Por isso a comparação relevante é com o IDA, não com o CDI.
2. **A regra principal fica 100 % no CDI** em RF e LSTM (um único alerta no início) e, por construção, perde para o IDA no teste.
3. **A regra de valor** praticamente nunca alerta e reproduz o IDA (no RF, nenhuma troca).
4. **O estilo original** troca de posição com frequência (dezenas de vezes no RF). Antes de custos o resultado sobre o IDA é pequeno (negativo no RF, positivo no LSTM) e o IC inclui zero; com 10 bps por troca o RF fica abaixo do IDA e o LSTM empata, e com 25 bps os dois ficam abaixo, principalmente o RF e o Perceptron.
5. **Nenhuma estratégia baseada em modelo supera o IDA de forma estatisticamente distinguível** no teste, com ou sem custos. É coerente com o AUC próximo de 0,5 da Fase 8.

## 4. O backtest original da monografia, decomposto

Recalculei a Fig. 7 a partir dos alertas salvos pelos scripts originais. A série reproduz **exatamente** as colunas do Excel (RF 570,9 e LSTM 572,8 na base 100; IDA 534,1; CDI 443,4). O gráfico da monografia cobre o período inteiro, que inclui treino e validação (usados para ajustar o modelo e o limiar).

| modelo | trecho | dias | estratégia | IDA | CDI | ganho vs IDA (pp) | taxa de alerta | trocas |
|---|---|---|---|---|---|---|---|---|
| Random Forest | período inteiro (como na monografia) | 4071 | 570,9 | 534,1 | 443,4 | +6.9 | 25% | 123 |
| Random Forest | treino (in-sample) | 2442 | 287,9 | 285,9 | 253,0 | +0.7 | 8% | 18 |
| Random Forest | validação (usada para limiar) | 814 | 128,4 | 121,4 | 118,8 | +5.8 | 48% | 46 |
| Random Forest | teste original (fora da amostra) | 815 | 154,4 | 153,9 | 147,6 | +0.4 | 54% | 59 |
| LSTM | período inteiro (como na monografia) | 4071 | 572,8 | 534,1 | 443,4 | +7.2 | 19% | 96 |
| LSTM | treino (in-sample) | 2442 | 282,9 | 285,9 | 253,0 | -1.1 | 20% | 57 |
| LSTM | validação (usada para limiar) | 814 | 128,2 | 121,4 | 118,8 | +5.6 | 28% | 34 |
| LSTM | teste original (fora da amostra) | 815 | 157,9 | 153,9 | 147,6 | +2.6 | 8% | 6 |

Leitura: o ganho sobre o IDA do período inteiro vem em boa parte da **validação** (2019‑2022, que contém a crise de 2020 e foi usada para escolher o limiar). No trecho de treino o ganho é próximo de zero (ou negativo, no LSTM) e no teste original é pequeno (RF) ou depende de poucas trocas (LSTM, 6 trocas, uma única execução). O ganho sobre o **CDI** existe em todos os trechos porque, nesses anos, o IDA rendeu mais que o CDI.

## 5. Conferência independente (planilha)

`outputs/backtest_check.xlsx` reproduz três estratégias (RF e LSTM no estilo original e LSTM de valor, semente 42, custo 10 bps) com fórmulas do Excel: troca, retorno, base 100, pico e drawdown, mais uma coluna de conferência contra o Python. O custo é editável em `Parametros!B1` (as conferências só valem com 10 bps).

Recalculei as fórmulas com um motor independente (biblioteca `formulas`, não o Excel): diferença máxima em relação ao Python = 6.5e-13 (base 100 final, trocas, drawdown e 815 de 815 conferências VERDADEIRO por estratégia). **O Excel em si não foi aberto aqui**; ao abrir a planilha, a coluna de conferência deve mostrar VERDADEIRO.

## 6. Limitações

- Custos de 0/10/25 bps são suposições; o IDA não é negociável e não há spread de debêntures real. Impostos e prazos de liquidação ignorados.
- Um único período de teste (815 dias), em que o IDA superou o CDI; o resultado dependeria do regime.
- A comparação 'original × corrigido' no mesmo teste usa pipelines diferentes (alvo e vazamentos), então o trecho de teste original da §4 não é diretamente comparável à §2.
