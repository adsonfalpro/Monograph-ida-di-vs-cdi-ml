# Engenharia de variáveis (Fase 4)

Gerado por `src/features/compare_features.py`. Código: `src/features/engineering.py`. Testes: `tests/test_features.py`.

## 1. Variáveis (mesma lista da monografia)

| Grupo | Variável (original → nova) | Fórmula | Disponível em t? |
|---|---|---|---|
| Crédito | `IDA_DI` → `ida_ret` | retorno diário do IDA‑DI | sim (fechamento de t) |
| Crédito | `Duration` → `duration` | duration média do índice | sim |
| Crédito | `IPCA_Deb` → `ipca_deb_ret` | retorno diário IDA‑IPCA (÷100) | sim |
| Mercado | `BOVA` → `ibov_ret` | retorno diário Ibovespa | sim |
| Mercado | `CDI_dia` → `cdi_ret` | CDI diário | sim |
| Excesso | `Excesso_dia` → `excesso_dia` | `ida_ret − cdi_ret` (forma simples, como no original) | sim |
| Volatilidade | `vol_20` | desvio‑padrão (ddof=1) de `excesso_dia` em 20 dias | sim |
| Tendência | `trend_20` | média de `ida_ret` em 20 dias | sim |
| Retorno | `ret_20` | soma de `excesso_dia` em 20 dias | sim |
| Momentum | `rsi_14` | RSI (média simples, 14 d.) da **variação diária do retorno** do IDA | sim |
| Técnica | `macd`, `macd_signal` | EMA12−EMA26 e EMA9 do `ida_ret` (`adjust=False`) | sim |

Todas só usam observações ≤ t (teste de perturbação em `tests/test_features.py`: alterar dados em t+1 em diante não muda nenhuma variável em t).

## 2. Mudanças em relação ao código original

| # | Original | Corrigido | Motivo |
|---|---|---|---|
| F1 | `fillna(0)` em todas as variáveis | nenhum preenchimento; aquecimento vem das 59 linhas anteriores à amostra | `fillna(0)` colocava 0 (média falsa) nas ~25 primeiras linhas; com a padronização isso virava um valor atípico |
| F2 | RSI: `loss=0 → NaN → fillna(0)` | RSI = 100 sem perdas (50 sem ganhos nem perdas) | correção defensiva: o original daria RSI 0 quando só há ganhos. **Nenhuma ocorrência nesta amostra** (0 janelas), então não altera os resultados |
| F3 | `StandardScaler` ajustado na base inteira | ajustado só nas linhas de treino (`fit_scaler`) | vazamento de média/desvio de validação e teste |
| F4 | `ewm` sem `min_periods` | `min_periods` = período da média | os primeiros valores do MACD eram baseados em 1–2 observações |
| F5 | `nan_to_num` em toda a matriz agregada | só assimetria/curtose de janela constante → 0; 874 casos de 97728 | evita esconder outros NaN; qualquer outro valor não finito levanta erro |

Não alterados (fiéis à monografia, mas questionáveis): RSI e MACD aplicados ao **retorno diário** e não ao nível do índice; `ret_20` é soma simples; `Excesso_dia` simples. Vão para `docs/limitations.md`.

## 3. Equivalência com o original

Variáveis ausentes na amostra original (4181 linhas): **0**; primeira linha com as 12 válidas: 59 (a amostra começa na linha 59, logo todas estão disponíveis sem preencher nada).

Diferença máxima absoluta nas linhas da amostra (alinhadas por data):

| variável | máx. total | máx. após 150 linhas | linhas com dif > 1e‑9 |
|---|---|---|---|
| IDA_DI | 0 | 0 | 0 |
| Duration | 0 | 0 | 0 |
| IPCA_Deb | 4.44e-16 | 4.44e-16 | 0 |
| BOVA | 0 | 0 | 0 |
| CDI_dia | 0 | 0 | 0 |
| Excesso_dia | 0 | 0 | 0 |
| vol_20 | 7.03e-18 | 7.03e-18 | 0 |
| trend_20 | 1.08e-19 | 0 | 0 |
| ret_20 | 8.67e-19 | 0 | 0 |
| rsi_14 | 2.7e-13 | 2.7e-13 | 0 |
| macd | 1.55e-05 | 2.19e-10 | 131 |
| macd_signal | 2.13e-05 | 3.22e-10 | 136 |

Leitura (comparação feita antes do `fillna(0)` do original, ignorando NaN): variáveis de nível, `vol_20`, `trend_20`, `ret_20` e RSI são idênticas onde o original tinha valor. Nas ~20–25 primeiras linhas o original tinha NaN e depois usava **0** (`fillna(0)`); aqui há valores reais, calculados com o aquecimento anterior à amostra. MACD e `macd_signal` diferem no início (até ~2e‑5) porque a média móvel aqui começa 59 linhas antes e converge; após 150 linhas a diferença é < 4e‑10.

## 4. Efeito do vazamento do scaler (fronteira de treino provisória)

Fronteira provisória: split 60% das 4072 janelas com alvo, última data de decisão de treino = 2019-05-16 (a divisão definitiva é a da Fase 5). Média e desvio‑padrão do treino versus da base inteira:

| variável | média treino | média base inteira | desloc. (dp treino) | razão dp |
|---|---|---|---|---|
| ida_ret | 0.0004337 | 0.0004146 | -0.14 | 3.03 |
| duration | 1.732 | 2.02 | +1.34 | 1.92 |
| ipca_deb_ret | 0.0005093 | 0.000444 | -0.03 | 1.17 |
| ibov_ret | 0.0004167 | 0.0004278 | +0.00 | 1.01 |
| cdi_ret | 0.0003796 | 0.0003664 | -0.15 | 1.40 |
| excesso_dia | 5.41e-05 | 4.828e-05 | -0.06 | 3.87 |
| vol_20 | 5.904e-05 | 0.0001427 | +1.12 | 4.03 |
| trend_20 | 0.0004348 | 0.0004146 | -0.19 | 2.59 |
| ret_20 | 0.001089 | 0.0009715 | -0.12 | 4.90 |
| rsi_14 | 49.89 | 49.97 | +0.01 | 1.08 |
| macd | -8.42e-07 | -1.297e-07 | +0.04 | 5.81 |
| macd_signal | -8.722e-07 | -1.681e-07 | +0.05 | 6.15 |

Quanto maior o deslocamento, mais informação do futuro (validação/teste) o original embutia nas variáveis padronizadas. Variáveis como `cdi_ret`, `duration` e `ida_ret` mudam de nível entre períodos (CDI alto em 2022‑25), então o efeito não é pequeno.

## 5. Janelas e agregação

- Janela de 100 linhas terminando na data de decisão t (`make_windows`); a janela de t não contém linhas > t.
- Janelas com alvo na amostra original: **4072** (o original tinha 4071: o laço `range(len−lookback−horizon)` descartava a última janela válida, cujo alvo usa o último retorno da base).
- Agregação RF/Perceptron: 120 atributos (10 estatísticas × 12 variáveis); estatísticas indefinidas trocadas por 0: 874.
- O LSTM usa a janela 100×12 padronizada, sem agregar.
