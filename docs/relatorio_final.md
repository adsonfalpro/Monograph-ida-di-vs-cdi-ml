# Relatório geral da reconstrução da monografia

*Modelos de Machine Learning aplicados a previsões no mercado de crédito brasileiro.* Fases 1 a 9, antes da organização para o GitHub.
Tudo o que está aqui foi calculado a partir dos arquivos da pasta de materiais originais; nenhum dado externo foi usado.

---

## 1. Resumo executivo

1. **A pergunta, os modelos, as variáveis, os horizontes e a divisão 60/20/20 foram mantidos.** Reconstruí o experimento em duas versões: a *original* (o que os scripts da monografia fazem) e a *corrigida* (mesmo desenho, sem vazamento de dados e com o alvo definido como no texto).
2. **O resultado principal muda.** Com o alvo e o método corrigidos, **nenhum dos três modelos se distingue do acaso no teste** (AUC ≈ 0,41 a 0,54, todos os intervalos de confiança de 95 % contêm 0,50) e **nenhuma estratégia baseada em modelo supera o IDA** no backtest. A monografia conclui o oposto (RF e LSTM superam IDA e CDI).
3. **A causa principal está no alvo.** O alvo usado nos scripts (`max` de 10 janelas retroativas) compartilhava 9 dos 10 dias com as variáveis de entrada. Só trocá-lo pelo alvo formal derruba o AUC do RF de 0,76 para 0,55, do LSTM de 0,78 para 0,61 e do Perceptron de 0,66 para 0,40. As demais correções (padronização, purge, calibração) movem o AUC bem menos, em magnitudes da ordem do desvio‑padrão entre sementes; não fiz um teste formal de cada passo.
4. **O backtest da monografia mistura treino e teste.** A série da Fig. 7 reproduz exatamente (RF 570,9, LSTM 572,8, IDA 534,1, CDI 443,4), mas cobre o período inteiro; o ganho sobre o IDA vem em boa parte da validação (usada para escolher o limiar), e no teste não se sustenta de forma distinguível.
5. **Isso é ausência de evidência, não prova de inutilidade.** O teste tem 169 eventos, as janelas são muito autocorrelacionadas, a taxa de eventos muda muito entre períodos e há só uma divisão cronológica. O poder estatístico é baixo.
6. **Limitações conhecidas:** a origem de uma figura da monografia (Perceptron, Fig. 4.1), o resultado exato do LSTM publicado e a conferência da planilha do *backtest* (veja a seção 7).

---

## 2. Como o trabalho foi organizado

Dez fases, cada uma com critério de aprovação. Nenhum resultado de teste foi visto antes de a Fase 7 congelar modelos, calibradores e limiares. Todas as decisões que alteram o experimento foram registradas.

| Fase | Entrega principal | Documento |
|---|---|---|
| 1 | Metodologia original reconstruída (com inconsistências) | `docs/original_methodology.md` |
| 2 | Base limpa, dicionário de dados e relatório de qualidade | `docs/data_quality_report.md`, `docs/data_dictionary.md` |
| 3 | Alvo formal e comparação com o original | `docs/target_definition.md` |
| 4 | Engenharia de variáveis sem vazamento | `docs/feature_engineering.md` |
| 5 | Divisão temporal, purge e proteção do teste | `docs/leakage_prevention.md` |
| 6 | Perceptron, Random Forest e LSTM reproduzíveis | `docs/model_specification.md` |
| 7 | Calibração e escolha do limiar; teste aberto uma vez por modelo congelado (a decomposição da Fase 8 mediu o teste de pipelines retreinados só como diagnóstico) | `docs/calibration_threshold.md` |
| 8 | Avaliação, intervalos de confiança e decomposição | `docs/evaluation_report.md` |
| 9 | Backtest, comparação com o original e planilha de conferência | `docs/backtest_report.md` |
| 10 | Organização para o GitHub | **pendente** |

---

## 3. O que cada fase encontrou e fez

### Fase 1 — o que a monografia realmente executa
- Os scripts finais são `LS.py` (RF, idêntico ao Apêndice A), `LST.py` (LSTM, Apêndice B) e `Perceptron.py` (Apêndice C). Vários outros arquivos da pasta (`Augusto.py`, `ARIMA.py`, `LSTM*.py`) são de outros projetos; `LSTM10.py` usa dados aleatórios.
- Todos leem `IDA_DI.xlsx`, que não existe na pasta; pelos arquivos de saída ele equivale a `Monografia - Dados.xlsx`.
- **O código do backtest não existe**: a Fig. 7 foi feita no Excel.

### Fase 2 — dados
Base primária: `Dados-IDA.xls` (4240 linhas, 2009‑01‑05 a 2025‑11‑19), superconjunto da base da monografia (4181 linhas). Erros encontrados:

| # | Erro | Tratamento |
|---|---|---|
| E1 | `BOVA` armazenado como texto em 18 linhas de 2009 (fora da amostra) | convertido |
| E2 | `IPCA Deb` em %, demais séries em decimal | convertido |
| E3 | `CDI_20_dias` da monografia usa janela de 19 dias (erro em todas as linhas); não afeta o alvo de 10 dias | recalculado |
| E4 | As colunas "Diferença" são excesso geométrico, não `IDA−CDI` (o sinal do alvo é o mesmo) | documentado |
| E5 | Scripts dividem `CDI_dia` por 100 sem necessidade | removido |
| E6 | Retorno do IDA diverge da variação do índice em 4 dias | mantido o retorno (como na monografia), dias marcados |
| E7 | `CDI_dia` muda em apenas 699 dias | conferido pelo autor do projeto |

### Fase 3 — alvo
- **Original:** `y = max(drawdown[t+1…t+10])`, com `drawdown` calculado sobre janelas retroativas de 10 dias. Cada janela tem 9 dias já presentes nas variáveis.
- **Formal (adotado):** `Y_t = 1` se o retorno composto do IDA nos 10 dias seguintes for menor que o do CDI; só retornos de `t+1…t+10` entram. Equivale à convenção do `M.py`.
- Os dois concordam em 93,9 % das janelas; o original é um superconjunto (247 casos só nele, nenhum só no novo).
- Taxa de positivos (treino / validação / teste): original 6,8 % / 24,8 % / 30,8 %; formal 3,3 % / 15,0 % / 20,7 %.

### Fase 4 — variáveis
- As mesmas 12 variáveis e fórmulas. Correções: sem `fillna(0)` (o aquecimento vem de 59 linhas anteriores), `ewm` com `min_periods`, tratamento do RSI sem perdas (não ocorre nesta amostra) e **padronização ajustada só no treino**.
- O vazamento do `StandardScaler` na base inteira era relevante: por exemplo, o desvio‑padrão de `macd` era 5,8 vezes maior na base inteira que no treino.
- Mantidos por fidelidade, mas questionáveis: RSI e MACD calculados sobre o retorno diário, não sobre o nível do índice.

### Fase 5 — divisão
- Cronológica, 60/20/20, com **purge** de 9 amostras no fim de treino e de validação (rótulos que olhavam para dentro do bloco seguinte). 4072 amostras, 4054 utilizáveis.
- Blocos: treino 2434 (80 positivos), validação 805 (122), teste 815 (169).
- **Teste protegido** (`TestSetGuard`): uma liberação por modelo e semente, registrada em arquivo.
- Decisão: a validação não foi dividida em duas metades (calibração + limiar), porque restariam 12 eventos para o limiar. A separação é feita por validação cruzada em blocos na Fase 7.

### Fase 6 — modelos
- Perceptron (camada única, do zero), Random Forest (800 árvores, mesmos parâmetros) e LSTM (arquitetura do Apêndice B, Keras 3, 5 sementes) com os parâmetros da monografia. Nenhum hiperparâmetro foi ajustado.
- Diagnóstico em validação: AUC do Perceptron 0,82, RF 0,58 ± 0,02, LSTM 0,68 ± 0,04. RF e LSTM sobreajustam muito (AUC em treino de 1,00 e 0,98).

### Fase 7 — calibração e limiar
- Calibração isotônica (RF) e Platt (LSTM; Perceptron como acréscimo) **fora da amostra**, por validação cruzada em blocos. Regra de alerta `p ≥ limiar`.
- Regra principal pré‑declarada: cross‑fit + F1. Com a calibração honesta, **o F1 degenera para "alertar sempre"** em RF e LSTM (F1 = 0,343, igual ao de alertar todo dia). O procedimento original parece melhor na validação (Brier ≈ 0,12) porque calibra e mede nos mesmos dados.
- Custos dos erros medidos em treino e validação: um falso negativo custa de 2× a 4× um falso positivo; o alerta só compensa com precisão acima de 19–32 %.

### Fase 8 — avaliação
Resultado no teste (815 amostras, 169 positivos; IC 95 % por bootstrap em blocos de 20 dias):

| modelo | AUC | Brier | F1 (regra principal) |
|---|---|---|---|
| Perceptron | 0,41 [0,23; 0,60] | 0,194 | 0,173 |
| Random Forest (5 sementes) | 0,54 [0,40; 0,69] | 0,168 | 0,343 |
| LSTM (5 sementes) | 0,54 [0,43; 0,64] | 0,172 | 0,343 |

Referências: AUC do acaso 0,50; Brier da previsão constante 0,164; F1 de alertar sempre 0,343. As diferenças entre os modelos são pequenas. A única com intervalo que exclui zero é RF − Perceptron (+0,133; IC 95 %: +0,009 a +0,269), e vem em boa parte de o Perceptron ficar abaixo de 0,50 no ponto; como são três comparações sem correção para múltiplas comparações, é uma evidência fraca. A LSTM não se distingue de nenhum dos dois.

Decomposição passo a passo (AUC médio):

| passo | RF | LSTM | Perceptron |
|---|---|---|---|
| S0 original (código literal) | 0,76 | 0,78 | 0,66 |
| S1 + alvo formal | 0,55 | 0,61 | 0,40 |
| S2…S5 (scaler, variáveis, purge, cross‑fit) | 0,54 a 0,58 | 0,51 a 0,61 | 0,40 a 0,41 |

O S5 não é idêntico ao pipeline final da Fase 7: retreina com 4071 janelas e mede o AUC sobre a probabilidade calibrada (RF 0,58), enquanto a Fase 7 usa 4072 janelas e o score bruto (RF 0,54). A distância entre os dois mostra o quanto o AUC varia com pequenas mudanças do pipeline.

### Fase 9 — backtest
- Regras: apenas o teste, decisão em `t` e posição em `t+1`, alerta → CDI, custos de 0, 10 e 25 bps por troca (suposição).
- No teste o IDA rendeu mais que o CDI (153,9 contra 147,6 em base 100). Regra principal: 100 % CDI (147,6). Regra de valor: reproduz o IDA. Estilo original: resultado pequeno e não significativo sem custos (negativo no RF, positivo no LSTM); com 10 bps o RF fica abaixo do IDA e com 25 bps os dois ficam abaixo (RF 140,9).
- Backtest original recalculado e decomposto: validação concentra o ganho sobre o IDA (+5,8 pp no RF, +5,6 pp no LSTM); treino ≈ 0 (LSTM −1,1 pp); teste original +0,4 pp (RF) e +2,6 pp (LSTM, 6 trocas).
- Planilha `outputs/backtest_check.xlsx` com fórmulas do Excel; as fórmulas foram recalculadas com um motor independente (diferença máxima 6,5e‑13).

---

## 4. Comparação com o que a monografia afirma

| Afirmação da monografia | Situação após a reconstrução |
|---|---|
| RF e LSTM superam buy‑and‑hold no IDA e CDI contínuo, preservando capital | **Não sustentada** fora da amostra; o ganho do período inteiro vem em boa parte da validação e depende de um alvo com sobreposição |
| LSTM tem desempenho "superior em equilíbrio entre precisão e recall", com "F1 elevado" | O F1 publicado é 0,389, menor que o do RF (0,562); o recall é 24 % |
| Perceptron descartado por baixo desempenho | Com o método revisado nenhum modelo se distingue do acaso e as diferenças entre eles são pequenas (só RF − Perceptron tem intervalo fora de zero, de forma marginal); o descarte não tem base estatística sólida |
| LSTM bem calibrado (Fig. 5.5) | A curva publicada fica bem acima da diagonal: o modelo subestima o risco |
| RF "moderadamente bem calibrado" (Fig. 6.4) | Curva com 6 pontos, dois em 1,0 e queda no último |
| Prioridade é maximizar o recall | O limiar foi escolhido por F1, não por recall |
| Brier Score reportado como métrica | Nunca aparece nos resultados do texto |
| Perceptron: 69 % / 56 % de acertos (p. 30) | A Fig. 4.2 mostra 78 % / 45 %; os dois não coincidem |
| Perceptron "Multilayer" | O código é de camada única |
| Backtest: CDI 436, base 100 em nov/2009 | O Excel mostra base em ago–set/2009 e CDI 443 até nov/2025; 436 deve ser um corte em set/2025 (não verificado) |
| Numeração de figuras (5.2 usada duas vezes etc.) | Precisa ser revista |

---

## 5. Decisões que tomei e que você pode contestar

1. Usei `Dados-IDA.xls` como base e assumi `IDA_DI.xlsx ≡ Monografia - Dados.xlsx`.
2. Mantive o retorno do IDA (`IDA_DI`) em vez do nível do índice (diferença acumulada de 0,2 %).
3. Dividi a validação com validação cruzada em blocos em vez de duas metades.
4. Critério principal de limiar: F1 com cross‑fit (você confirmou). Também reportei 0,50, regra de valor e estilo original.
5. LSTM com 5 sementes (3 na decomposição) em vez de uma execução.
6. Custos do backtest de 0, 10 e 25 bps por troca: suposição minha.
7. Incluí o Perceptron no backtest como extra, embora a monografia o descarte.

---

## 6. Limitações

- **Poder estatístico baixo:** 169 eventos no teste, 80 no treino, janelas que compartilham 99 de 100 dias.
- **Uma só divisão cronológica**; a taxa de eventos varia de 3,3 % a 20,7 % entre blocos.
- **Sem busca de hiperparâmetros, outros horizontes ou alvo contínuo** (para não misturar com a reconstrução fiel).
- **Backtest:** custos supostos; IDA não é negociável; um único período de teste em que o IDA superou o CDI.

---

## 7. Limitações conhecidas

1. **Origem da Fig. 4.1 (Perceptron):** tem 314 positivos no teste; o código gera 251; 168 variantes de alvo testadas, nenhuma dá 314.
2. **LSTM publicado:** é uma execução sem semente nem versão de TensorFlow registradas; só a reprodução qualitativa é possível.
3. **Planilha de conferência do *backtest*:** gerada localmente por `src/backtest/excel_check.py` e validada por um motor de fórmulas em Python (diferença máxima 6,5e-13); não foi aberta no Excel.
4. **Dados da ANBIMA:** não são redistribuídos neste repositório; devem ser obtidos diretamente na ANBIMA (fonte, período e colunas em `data/README.md`).

---

## 8. Inventário

**Código** (`src/`): `data/` (base limpa e verificação), `targets/`, `features/`, `validation/`, `models/`, `calibration/`, `evaluation/`, `backtest/`.
**Testes** (`tests/`): 66 testes (dados, alvo, variáveis, divisão, modelos, calibração, backtest), todos passando.
**Resultados:** `outputs/tables/`, `outputs/figures/`, `outputs/metrics/` (inclui `frozen_config.json` com hash e `test_access_log.json`), `outputs/backtest_check.xlsx`.
**Original preservado:** `original/scripts/` (scripts da monografia, sem alteração) e `original/results/`.
**Não versionável por ora:** `data/raw/`, `data/processed/` e `outputs/models/` (ignorados pelo git).
