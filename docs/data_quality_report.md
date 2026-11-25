# Relatório de qualidade dos dados (Fase 2)

Gerado por `src/data/audit.py`. Fonte única: arquivos da pasta de materiais originais (nenhum dado externo).

## 1. Base oficial

**Primária: `Dados-IDA.xls`** (4240 linhas, 2009-01-05 a 2025-11-19). `Monografia - Dados.xlsx` (4181 linhas) é uma derivação dela: nas 4181 datas em comum, `Número Índice`, `IDA DI`, `Duration`, `IPCA Deb`, `CDI` e `BOVA` são **idênticos** (diferença máxima 0), e as datas da monografia coincidem exatamente com as de `data >= 2009-03-31` da base bruta (`True`).

Justificativa: (i) é a fonte bruta e um superconjunto; (ii) todas as colunas derivadas da monografia podem ser recalculadas a partir dela e uma delas está errada (§3); (iii) as 59 linhas extras servem de aquecimento causal para indicadores técnicos, sem preencher nada com informação futura. A amostra de modelagem fica marcada em `in_original_sample` (4181 linhas), idêntica à da monografia.

`LSTM.xlsx` (414 849 linhas por ativo, colunas `Ativo`, `taxa_indicativa`…) e `Mono.zip` (cópia dos mesmos arquivos) **não** fazem parte deste experimento. `IDA_DI.xlsx`, lido pelos scripts, não existe na pasta; as saídas `alertas_modelo_base_completa.xlsx` reproduzem `Monografia - Dados.xlsx` com diferença máxima 4e-16, então foi tratado como equivalente a ela.

## 2. Verificações estruturais

| Verificação | Resultado |
|---|---|
| Datas duplicadas | 0 |
| Ordem cronológica | True |
| Datas em fim de semana | 0 |
| Maior intervalo entre datas | 5 dias |
| `Duration` ausente | 59 linhas (até 2009-03-30); todas fora da amostra original |
| `BOVA` armazenado como texto (`'-2,69%'`) | 18 linhas (até 2009-03-30); fora da amostra original |
| `BOVA` = 0 (sem pregão: 24/12, 31/12, 25/01, 09/07, 20/11) | 64 dias |
| Demais colunas | sem valores ausentes |

Tabela completa: `outputs/tables/missing_values.csv`.

## 3. Ajustes realizados nos dados

| # | Problema | Evidência | Tratamento na base limpa |
|---|---|---|---|
| E1 | `BOVA` em texto com `%` e vírgula (18 linhas, mar/2009) | leitura como `object` | convertido para decimal; flag `ibov_from_text` |
| E2 | `IPCA Deb` em **%** enquanto IDA, CDI e BOVA estão em decimal | média 0,046 vs 0,0004 | dividido por 100 → `ipca_deb_ret` (decimal) |
| E3 | Coluna `CDI_20_dias` da monografia está **errada** | difere do composto de 20 dias do `CDI_dia` em 4181 linhas, erro máx. 5.57e-04; coincide com janela de **19** dias (erro máx. 1.3e-15) | recalculado a partir do retorno diário; `Diferença…20Dias` também corrigida. Não afeta o alvo de 10 dias |
| E4 | Colunas `Diferença CDIxIDA_*` e `Excesso` são **excesso geométrico** `(1+IDA)/(1+CDI)−1`, não `IDA−CDI` | diferença máx. vs geométrico 2.2e-16; vs simples 3.7e-06 | `excess_ret` e `excess_trail_h` usam a forma geométrica. O sinal de `IDA_10 < CDI_10` é idêntico nas duas formas (True), então o alvo original não muda por isso |
| E5 | `CDI_dia`: scripts dividem por 100 | `CDI_dia` já é decimal (≈ 13.6% a.a. em 2009) | **não** dividir; `cdi_ret` único, em decimal |
| E6 | Retorno diário do IDA diverge da variação do nível do índice em **4 dias** | ver `ida_return_vs_index_mismatch.csv`; nos demais dias a diferença máxima é 5.0e-07 | mantida `ida_ret` (base de todas as colunas acumuladas, igual à monografia), com `ida_index_ret` e a flag `ida_ret_index_mismatch`. **Decisão pendente para o backtest (Fase 9)** |
| E7 | `CDI_dia` muda em apenas 699 dias (igual ao dia anterior em 83.5%); anualizado (252 d.u.) vai de 13.6% (jan/2009) a 14.9% (nov/2025) | padrão de atualização em degraus | **conferido pelo autor do projeto** contra a fonte do CDI |

### E6 em detalhe

Correlação entre `ida_ret` e `ida_index_ret`: 0.9603. Nos demais dias a igualdade vale até arredondamento. Nos dias divergentes: em 2025-04-01 e 2025-09-16 o nível do índice **reverte o salto do dia anterior** (fim/início de mês) e a coluna de retorno não registra a reversão; em 2020-11-30 e 2022-05-30 o índice sobe 0,2 % e 0,5 % enquanto a coluna de retorno mostra apenas o carregamento normal. Os dois cálculos são mostrados e a escolha está registrada nas decisões. Efeito acumulado: índice 5.8142× vs composto dos retornos 5.8274× no período todo. Se o alvo usasse retornos derivados do índice, o sinal de `IDA_10 < CDI_10` mudaria em 14 dias (janelas de 10 dias que contêm algum dos 4 dias).

## 4. Reconciliação das colunas derivadas

`outputs/tables/reconciliation_raw_vs_monograph.csv` compara cada coluna da base limpa com a da monografia. Resumo (máx. |diferença|):

| coluna limpa | coluna monografia | máx. abs. dif. | linhas com dif. > 1e-9 |
|---|---|---|---|
| ida_index | ida_index_m | 0.00e+00 | 0 |
| ida_ret | ida_ret_m | 0.00e+00 | 0 |
| duration | duration_m | 0.00e+00 | 0 |
| ipca_deb_ret | ipca_deb_pct | 3.47e-18 | 0 |
| ibov_ret | ibov_ret_m | 0.00e+00 | 0 |
| cdi_ret | cdi_ret_m | 0.00e+00 | 0 |
| ida_trail_10 | ida_10 | 1.11e-15 | 0 |
| cdi_trail_10 | cdi_10 | 1.11e-15 | 0 |
| excess_trail_10 | excess_10 | 1.33e-15 | 0 |
| ida_trail_20 | ida_20 | 1.55e-15 | 0 |
| cdi_trail_20 | cdi_20 | 5.57e-04 | 4181 |
| excess_trail_20 | excess_20 | 5.55e-04 | 4181 |
| ida_trail_30 | ida_30 | 1.78e-15 | 0 |
| cdi_trail_30 | cdi_30 | 1.78e-15 | 0 |
| excess_trail_30 | excess_30 | 2.22e-15 | 0 |
| ida_trail_60 | ida_60 | 2.66e-15 | 0 |
| cdi_trail_60 | cdi_60 | 2.66e-15 | 0 |
| excess_trail_60 | excess_60 | 3.55e-15 | 0 |

Leitura: acumulados de 10, 30 e 60 dias do IDA e do CDI e o de 20 dias do IDA coincidem até 1e-15 (**janela retroativa terminando em t, inclusive**). As colunas `Diferença…` da monografia são excessos geométricos e coincidem com `excess_*` (exceto o de 20 dias, afetado por E3). `ipca_deb_ret` difere só por escala (E2).

## 5. Retornos e estatísticas

Ver `outputs/tables/descriptive_stats.csv`. Valores extremos: `BOVA` (até ±14 %) em 08/08/2011, 18/05/2017 e 09–25/03/2020; `IPCA Deb` (até ±3,4 %) em 24/09/2015, 18/05/2017, 12–18/03/2020, 24/04/2020 e 12/01/2023. Coincidem com episódios de estresse; **não foram removidos nem winsorizados** (a revisar na Fase 4 se afetarem a padronização).

## 6. Causalidade e dados futuros

- Nenhum valor ausente foi preenchido (`fillna`, `ffill`, `bfill` não são usados). As 59 primeiras linhas têm `Duration` ausente e ficam fora da amostra de modelagem; servem apenas de histórico para janelas móveis.
- Os acumulados `*_trail_*` usam apenas retornos até t. Indicadores técnicos (RSI, MACD, volatilidade, etc.) serão construídos na Fase 4 sobre esta base e testados quanto à causalidade.
- Nenhum `StandardScaler` é aplicado nesta fase (será ajustado só no treino, Fase 4/5).

## 7. Versão definitiva

`data/processed/ida_clean.csv` (não versionada: os dados da ANBIMA não são redistribuídos, veja `data/README.md`). Reproduzir com `python -I src/data/clean_base.py`. Dicionário em `docs/data_dictionary.md`.
