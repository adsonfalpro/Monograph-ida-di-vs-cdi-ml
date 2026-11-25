# Metodologia original — reconstrução (Fase 1)

Status: **Fase 1, nenhuma correção aplicada.** Este documento descreve o que a monografia afirma, o que os scripts
realmente executam e o que foi verificado diretamente nos dados. Onde a evidência é insuficiente, está marcado
como **[NÃO VERIFICADO]**.

Fontes: `00098196_000595_monografia01.pdf`, scripts em `original/scripts/`, resultados em `original/results/`, e a
base `Monografia - Dados.xlsx` (pasta de materiais originais, somente leitura).

---

## 1. Escopo e arquivos que correspondem à monografia

| Papel | Arquivo | Evidência de vínculo |
|---|---|---|
| Random Forest final (Apêndice A) | `LS.py` | Idêntico ao Apêndice A; `resultados.csv` do RF (thr 0,2368; F1 0,562; AUC 0,737) bate com Fig. 6.x e com a matriz 319/245/57/194 |
| LSTM final (Apêndice B) | `LST.py` (pasta de saída `LSTM_10d_ajustada`) | Idêntico ao Apêndice B; `tabela_metricas.csv` (thr 0,2688; P 0,968; R 0,243; AUC 0,8104) bate com a matriz 562/2/190/61 |
| Perceptron final (Apêndice C) | `Perceptron.py` | Idêntico ao Apêndice C; **mas os resultados da Fig. 4.1 não reproduzem a partir deste script (ver §7)** |
| Matrizes normalizadas digitadas à mão | `IDA.py` (RF), `norm.py` (LSTM) | Valores fixos no código, não calculados |
| Versão anterior com outro alvo e outro split | `M.py` | LOOKBACK 100, horizontes 10/20/30/60, split 80/10/10, alvo diferente, custo FN=5/FP=1 |
| Versão anterior do RF | `S.py` | split 80/…, RF max_depth=10, threshold por recall mínimo |
| Não relacionados / outro projeto | `Augusto.py` (scraper de FIDC), `ARIMA.py`, `Regressao.py`, `LSTM.py`, `LSTM1/2/11/R.py` (usam `ativo_id`, lookback 22/30), `LSTM10.py` (**usa dados aleatórios simulados**) | Movidos para `original/scripts/other_projects_or_unrelated/`. Não entram na reprodução |

**Achados sobre os arquivos**
- Todos os scripts finais leem `IDA_DI.xlsx` (em um caminho local da pasta de materiais originais), que **não existe** na pasta. Pelos arquivos de
  saída (`alertas_modelo_base_completa*.xlsx`, 4181 linhas, colunas idênticas às de `Monografia - Dados.xlsx`),
  `IDA_DI.xlsx` é uma cópia/renomeação de `Monografia - Dados.xlsx`. **[Confirmar na Fase 2 por comparação célula a célula.]**
- **Não há código do backtest** (Fig. 7). As colunas `CDI`, `IDA`, `Modelo`, `Random Forest`, `LSTM` nos
  `alertas_*.xlsx` indicam que ele foi feito no **Excel**. O único backtest em Python
  (`backtest_estrategias.xlsx`) está quebrado (valores da ordem de 1e84, mistura de unidades) e **não** corresponde à Fig. 7.
- Não há código que gere as curvas ROC/PR/calibração do RF em arquivo isolado além de `LS.py` (truncado no gráfico de alertas); as imagens existem em `Results_modelo/RF_10d_otimizado`.

---

## 2. Dados

Base: `Monografia - Dados.xlsx`, aba `Planilha1`, **4181 linhas, 2009‑03‑31 a 2025‑11‑19**, ordenada, sem datas duplicadas, sem NaN.

| Coluna | Significado (verificado) |
|---|---|
| Data de Referência, Número Índice | data, nível do IDA‑DI |
| IDA DI | retorno diário do IDA‑DI, **em decimal** (média 0,000415) |
| Duration, IPCA Deb | duration do índice; IPCA Deb em **%** (média 0,044) — **escala diferente das demais** |
| BOVA | retorno diário do Ibovespa, em % (média 0,0004 ⇒ não é o mesmo padrão do IDA) **[verificar unidade]** |
| CDI_dia | CDI diário, em decimal (0,000417 em 2009) |
| `IDA/CDI_{10,20,30,60}_dias` | **retornos acumulados em janela *retroativa* terminando em t** (ver abaixo) |
| `Diferença CDIxIDA_*` | **Excesso geométrico** `(1+IDA)/(1+CDI)−1` (não `IDA − CDI`; corrigido na Fase 2) |

**Verificação feita:** `IDA_10_dias[t]` e `CDI_10_dias[t]` são exatamente `Π(1+r) − 1` sobre os 10 dias
**t‑9…t** (diferença máx. 1e‑15). Não são retornos futuros. A coluna `Diferença…10Dias` é o excesso geométrico (diferença de até 7,3e‑5 para a simples, mas o sinal é idêntico). **Erro adicional (Fase 2):** `CDI_20_dias` da monografia usa janela de 19 dias.

**Unidades:** os scripts dividem `CDI_dia` por 100 (`CDI_dia_decimal`). Como `CDI_dia` já é decimal, o
`CDI_acumulado` calculado em Python é ≈ 1,000004 no primeiro dia (errado por fator 100). O gráfico "Comparação" gerado
por `LST.py`/`LS.py` usa essa série e portanto **está errado**; a Fig. 7 da monografia veio do Excel (CDI final ≈ 443).

---

## 3. Variável‑alvo — o que é, de fato

**Texto (§3.1, §4.2, p. 28):** "valor 1 quando o excesso de retorno em 10 dias foi negativo", previsto "nos dez dias subsequentes".

**Código final (`LS.py`, `LST.py`, `Perceptron.py`):**
```
Excesso_Retorno[t] = IDA_10_dias[t] − CDI_10_dias[t]          # janela RETROATIVA de 10 dias terminando em t
drawdown[t]        = 1 se Excesso_Retorno[t] < 0
janela i:  features = linhas i … i+99
           y_i      = max( drawdown[i+100 … i+109] )          # "qualquer um dos 10 dias seguintes"
```
Consequências (verificadas por cálculo, não só por leitura):
1. `drawdown[t]` usa retornos dos dias t‑9…t. Logo `y_i` depende de retornos entre os dias **i+91 e i+109**.
   Os dias i+91…i+99 pertencem à **própria janela de features** (sobreposição entre alvo e features, sem ser vazamento de futuro, mas o alvo não é "10 dias à frente").
2. O alvo é "**algum** dos 10 dias seguintes tem excesso retroativo de 10 dias negativo", que **não é** "IDA < CDI acumulado nos próximos 10 dias". É uma definição de evento diferente da do texto, com taxa positiva bem maior.
3. Taxas positivas medidas: dia isolado `drawdown` = 8,9 %; alvo `y` por split — **treino 6,8 %, validação 24,8 %, teste 30,8 %**. A prevalência muda drasticamente entre períodos (regime de CDI alto em 2022‑25).

**`M.py` (versão anterior):** `y = (IDA_h < CDI_h)` na linha `i+100+h−1`, ou seja, o retorno retroativo de h dias terminando exatamente h dias após o fim da janela = **retorno dos h dias seguintes**. Esta é a definição mais próxima do texto e da Fase 3 do plano corrigido.

---

## 4. Variáveis explicativas

Lista final (idêntica em `LS.py`, `LST.py`, `Perceptron.py`): `IDA_DI, Duration, IPCA_Deb, BOVA, CDI_dia, Excesso_dia, vol_20, trend_20, ret_20, rsi_14, macd, macd_signal` (12 variáveis). `M.py` acrescenta `Número Índice`.

| Variável | Fórmula no código | Observação |
|---|---|---|
| `Excesso_dia` | `IDA_DI − CDI_dia` | |
| `vol_20` | `rolling(20).std()` de `Excesso_dia` | causal |
| `trend_20` | `rolling(20).mean()` de `IDA_DI` | causal; chamada de "tendência" |
| `ret_20` | `rolling(20).sum()` de `Excesso_dia` | soma simples, não composta |
| `rsi_14` | RSI com **média simples** (não Wilder) sobre `diff(IDA_DI)` | aplica‑se à diferença do **retorno diário**, não ao preço; denominador 0→NaN |
| `macd`, `macd_signal` | EMA12−EMA26 e EMA9, também sobre o **retorno diário** `IDA_DI` | idem |

Tratamento de ausentes: `fillna(0)` nas features (zera os primeiros ~25 dias de rolling **antes** da padronização; `M.py` usa `ffill` + `bfill`, o que traz informação **futura** para trás).

---

## 5. Janelas, agregação e divisão

- `LOOKBACK=100`, `HORIZON=10`, `RANDOM_STATE=42`. Nº de amostras `n = 4181 − 100 − 10 = 4071`.
- **RF e Perceptron:** cada janela 100×12 vira um vetor de 120 atributos (média, desvio, mín., máx., mediana, assimetria, curtose, P25, P75, último valor). O texto cita "percentis" genericamente; o código usa só P25 e P75.
- **LSTM:** recebe a sequência 100×12 padronizada.
- **Split cronológico 60/20/20 sobre as janelas** (sem purge, sem embargo):

| Conjunto | n | Datas (último dia da janela de features) | Taxa positiva |
|---|---|---|---|
| Treino | 2442 | 2009‑08‑21 → 2019‑05‑15 | 6,8 % |
| Validação | 814 | 2019‑05‑16 → 2022‑08‑08 | 24,8 % |
| Teste | 815 | 2022‑08‑09 → 2025‑11‑04 | 30,8 % (251 positivos) |

- **Vazamentos/ambiguidades identificados no código original:**
  1. `StandardScaler.fit_transform` ajustado em **todos** os dados antes do split (usa média/desvio de validação e teste).
  2. Sem purge: os rótulos das últimas 9 amostras de treino usam retornos de dias posteriores à primeira data de decisão da validação (idem na fronteira validação→teste). As janelas de features compartilham dias entre amostras vizinhas, o que não é vazamento por si só (são dados passados); o problema é o rótulo olhar para dentro do bloco seguinte. *(Redação corrigida na Fase 10; a versão anterior falava em "sobreposição ≥ 109 dias".)*
  3. O threshold e o calibrador são ajustados na validação, mas **LSTM**: `EarlyStopping`/`ReduceLROnPlateau` também usam a validação (válido), enquanto no **RF** o calibrador isotônico (`cv="prefit"`) é ajustado **na mesma validação** usada para escolher o threshold ⇒ threshold escolhido sobre probabilidades que o calibrador já "viu" (otimista).
  4. Backtest em Excel usa alertas para **todo o período** (treino + validação + teste), então ≈ 80 % do período é in‑sample.

---

## 6. Modelos — parâmetros

| | Perceptron | Random Forest | LSTM |
|---|---|---|---|
| Implementação | `PerceptronScratch` (numpy), camada única, degrau, `lr=0.01`, 100 épocas, pesos 0, shuffle por época com `np.random.seed(42)` | `RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1)` | `LSTM(64, return_sequences=True)` → `Dropout(0.3)` → `LSTM(32)` → `BatchNormalization` → `Dense(32, relu)` → `Dropout(0.2)` → `Dense(1, sigmoid)` |
| Treino | sem validação | treino 60 % | Adam lr 1e‑3, `binary_crossentropy`, 50 épocas máx., batch 32, `EarlyStopping(patience=5, restore_best_weights)`, `ReduceLROnPlateau(0,5; patience 3)`, `class_weight` balanceado |
| Calibração | nenhuma (saída binária) | `CalibratedClassifierCV(method="isotonic", cv="prefit")` na validação | Platt (`LogisticRegression`) na validação |
| Threshold | n/a | F1 máximo na validação → 0,2368 | F1 máximo na validação (calibrado) → 0,2688 |
| Sementes | `np.random.seed(42)` | 42 | `np.random.seed(42)`, `tf.random.set_seed(42)` (não garante determinismo em GPU/CPU multithread) |

Observações:
- O texto chama o modelo de **"Multilayer Perceptron"**, mas o código é um **perceptron de camada única**.
- O texto diz que o limiar é para "maximizar o recall" (§3.1), mas o código maximiza o **F1**.
- O texto diz que os modelos usam ReLU/BatchNorm etc. (confere), mas **não** informa `class_weight`, nº real de épocas, nem versões de bibliotecas **[NÃO VERIFICADO: versões]**.
- `precision_recall_curve` retorna `n+1` pontos mas `thr` tem `n`; `best_thresh = thr[argmax(f1)]` pode falhar se o argmax cair no último ponto (precision=1, recall=0). Não ocorreu nos resultados publicados.

---

## 7. Resultados publicados × o que os arquivos mostram

Teste = 815 amostras, 251 positivos (30,8 %).

| Modelo | Matriz (TN, FP, FN, TP) | Precisão | Recall | F1 | AUC | Brier | Origem |
|---|---|---|---|---|---|---|---|
| Perceptron (Fig. 4.1) | 390, 111, 174, 140 | 0,558 | 0,446 | 0,496 | — | — | **somente a imagem**; 314 positivos ≠ 251 |
| Random Forest | 319, 245, 57, 194 | 0,442 | 0,773 | 0,562 | 0,737 | 0,1755 | `RF_10d_otimizado/resultados.csv` |
| LSTM ajustada | 562, 2, 190, 61 | 0,968 | 0,243 | 0,389 | 0,810 | 0,198 | `LSTM_10d_ajustada/tabela_metricas.csv` |

**Inconsistências da monografia (a documentar, não a corrigir ainda):**
1. **Perceptron:** a matriz da Fig. 4.1 tem 314 casos positivos no teste, enquanto RF e LSTM têm 251 com o mesmo split. Logo a figura não foi gerada com o mesmo alvo/base que os outros dois, nem com `Perceptron.py` tal como está. Além disso, o **texto** (p. 30) diz 69 %/31 % e 56 %/44 %, mas a **Fig. 4.2** mostra 0,78/0,22 e 0,55/0,45 (que é o que 390/501 e 140/314 dão: 0,778 e 0,446 ⇒ a figura diz 0,45 para acerto de "Abaixo do CDI", o texto diz 56 %).
2. **Perceptron, curva PR:** construída a partir de predições **binárias** (`precision_recall_curve(y_test, perc_pred)`), não de scores; é uma linha de 3 pontos, sem significado de limiar.
3. **LSTM:** a matriz (2 FP, 190 FN) é coerente com o threshold 0,27, mas mostra um modelo **muito conservador (recall 24 %)** — resultado de calibração/threshold da validação (prevalência 24,8 %) aplicados a um teste com prevalência 30,8 % e regime diferente. O texto o descreve como "superior em equilíbrio entre precisão e recall" e com "F1 elevado" (F1 = 0,389 < RF 0,562).
4. **Curva de calibração (LSTM, Fig. 5.5):** pontos bem **acima** da diagonal (frequência observada > probabilidade prevista), ou seja, o modelo **subestima** o risco; o texto afirma que "segue de perto a linha ideal". O Brier não é reportado no texto, apesar de ser prometido em §3.1 e §3.5.
5. **Curva de calibração (RF, Fig. 6.4):** o texto diz "moderadamente bem calibrado", mas há 6 pontos, dois bins com frequência 1,0 e queda no último.
6. **Numeração de figuras** repetida/trocada (Fig. 5.2 usada duas vezes, "Fig. 5.3" para ROC que é 5.4, etc.).
7. **Backtest (Fig. 7):** texto: CDI 436, IDA 534, RF 565, LSTM 569, "base 100 em novembro de 2009". O Excel mostra base 100 em **2009‑08/09**, máximos de CDI 443, IDA 534, RF 571, LSTM 573 até 2025‑11‑19. As diferenças pequenas sugerem corte em set/2025 **[NÃO VERIFICADO]**.
8. **Backtest sem custos, sem defasagem explícita, 100 % do período** (a maior parte in‑sample). Não há código.
9. O texto afirma (cap. 3.1) que "os resultados mostraram que a estratégia superou ambas as alternativas" — claim feito antes de qualquer resultado e baseado em período majoritariamente in‑sample.

**Outras versões encontradas (não usadas na monografia, relevantes para rastreabilidade):**
- `LSTM_10d_otimizado`: thr 0,041; P 0,523; R 0,777; F1 0,625; AUC 0,810 (script não localizado).
- `LSTM_10d_otimizada`: thr 0,44; F1 0,189; AUC 0,606 (resultado da família `M.py`, alvo e split diferentes).
- `resultados_consolidados.csv` (`M.py`): RF/RNR/LSTM para h=10/20/30/60, split 80/10/10, alvo "retorno dos h dias seguintes"; AUC 10d: RF 0,27, LSTM 0,78. **O AUC de 0,27 (abaixo de 0,5) indica inversão de sinal entre validação e teste, não habilidade.**

---

## 8. Tabela: monografia × scripts × versão corrigida (plano)

| Item | Monografia diz | Script executa | Versão corrigida (plano) |
|---|---|---|---|
| Alvo | IDA < CDI nos 10 dias seguintes | max sobre 10 dias de "excesso retroativo de 10d < 0" (inclui dias já conhecidos) | `Y_t = 1[ΠIDA(t+1..t+10) < ΠCDI(t+1..t+10)]` (Fase 3) |
| Scaler | "padronizadas" | `fit` em todos os dados | `fit` só no treino |
| Ausentes | não descrito | `fillna(0)` | sem preenchimento com futuro; descartar warm‑up |
| Split | 60/20/20 | 60/20/20 sobre janelas, sem purge | cronológico + purge de 9 amostras por fronteira (horizonte − 1) |
| Calibração/threshold | calibra, F1 na validação | RF: mesma validação para ambos | calibração e threshold em blocos separados |
| Perceptron | "Multilayer", referência | camada única, avaliado em binário | manter algoritmo; score contínuo p/ ROC/PR |
| RF | 800 árvores etc. | idem | idem (documentado) |
| LSTM | arquitetura descrita | `LST.py` ≡ Apêndice B | reimplementar fiel; fixar sementes e versões |
| Backtest | 4 estratégias, base 100 | Excel, período inteiro, sem código | Python + conferência Excel, só OOS, sinal defasado, custos |
| Unidades CDI | — | `CDI_dia/100` indevido | convenção única (decimal) |

---

## 9. Perguntas abertas para a Fase 2
1. `IDA_DI.xlsx` ≡ `Monografia - Dados.xlsx`? (comparação célula a célula)
2. `Dados-IDA.xls` (início 2009‑01‑05, colunas diferentes) — fonte alternativa ou rascunho?
3. Unidade de `BOVA` e `IPCA_Deb` (% vs decimal) e diferença de 7,3e‑5 em `Diferença…10Dias`.
4. De onde veio a Fig. 4.1 do Perceptron (314 positivos)?
5. Versões de Python/TensorFlow/scikit‑learn usadas (monografia só diz Python 3.9).
6. Qual script gerou `LSTM_10d_otimizado` e `RF_10d_otimizado/*_com_perceptron.xlsx`.
