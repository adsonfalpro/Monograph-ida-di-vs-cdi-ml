# Dicionário de dados — `data/processed/ida_clean.csv`

Fonte: `Dados-IDA.xls` (ANBIMA, segundo a monografia). 4240 linhas, 2009‑01‑05 a 2025‑11‑19.
Convenção: retornos em **decimal** (0,01 = 1 %). Reproduzir: `python -I src/data/clean_base.py`.

| Coluna | Tipo | Descrição | Origem / transformação |
|---|---|---|---|
| `date` | data | dia útil | `Data de Referência` |
| `ida_index` | float | nível do IDA‑DI | `Número Índice` |
| `ida_ret` | float | retorno diário do IDA‑DI | `IDA DI` (sem alteração) |
| `duration` | float | duration média (anos) | `Duration`; **ausente nas 59 primeiras linhas, não preenchido** |
| `ipca_deb_ret` | float | retorno diário das debêntures IPCA | `IPCA Deb` ÷ 100 (a fonte está em %) |
| `ibov_ret` | float | retorno diário do Ibovespa; 0 = sem pregão | `BOVA`; 18 linhas de 2009 vinham como texto |
| `cdi_ret` | float | CDI diário | `CDI` (sem alteração; **não** dividir por 100) |
| `excess_ret` | float | excesso geométrico diário `(1+IDA)/(1+CDI)−1` | recalculado; = coluna `Excesso` |
| `ida_index_ret` | float | variação do nível do índice | `ida_index.pct_change()`; só verificação e backtest alternativo |
| `ida_trail_h` | float | retorno composto do IDA nos h dias **terminando em t** (h = 10, 20, 30, 60) | recalculado; **não é alvo** |
| `cdi_trail_h` | float | idem para o CDI | recalculado; corrige `CDI_20_dias` |
| `excess_trail_h` | float | `(1+ida_trail_h)/(1+cdi_trail_h)−1` | recalculado |
| `in_original_sample` | bool | `date ≥ 2009‑03‑31` (4181 linhas = base da monografia) | flag |
| `has_duration` | bool | `duration` disponível | flag |
| `ibov_from_text` | bool | BOVA convertido de texto | flag |
| `ibov_zero_no_trading` | bool | `ibov_ret == 0` | flag |
| `ida_ret_index_mismatch` | bool | `|ida_ret − ida_index_ret| > 1e‑4` (4 dias) | flag |

Mapeamento dos nomes originais: ver §3 de `data_quality_report.md`.
Nenhuma coluna de alvo, indicador técnico ou padronização existe nesta base (Fases 3 e 4).
