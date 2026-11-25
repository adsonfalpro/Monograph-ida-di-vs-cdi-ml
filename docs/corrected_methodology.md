# Metodologia revisada

Mesmo experimento da monografia, com os refinamentos abaixo. Cada item aponta para o documento da fase onde a decisão e a evidência estão.

## Pergunta e escopo (inalterados)
Prever, com até 10 dias de antecedência, se o IDA‑DI renderá menos que o CDI. Modelos: Perceptron, Random Forest e LSTM. Lookback de 100 dias, horizonte de 10 dias,
semente 42, divisão 60/20/20, 12 variáveis, calibração probabilística e limiar por F1.

## Pipeline

| Etapa | Versão revisada | Documento |
|---|---|---|
| Dados | Base bruta `Dados-IDA.xls`; BOVA em texto convertido, IPCA em decimal, `CDI_20_dias` recalculado; nada preenchido com valores futuros | `data_quality_report.md` |
| Alvo | `Y_t = 1` se o retorno composto do IDA nos dias `t+1…t+10` for menor que o do CDI (empate = 0) | `target_definition.md` |
| Variáveis | As mesmas 12; sem `fillna(0)`; aquecimento com 59 linhas anteriores; `StandardScaler` ajustado só com o treino | `feature_engineering.md` |
| Janelas | 100 dias terminando na data de decisão `t`; RF e Perceptron recebem 10 estatísticas por variável (120 atributos), o LSTM recebe a sequência 100 × 12 | `feature_engineering.md` |
| Divisão | Cronológica 60/20/20 com purge de 9 amostras no fim de treino e de validação; teste liberado uma vez por modelo (`TestSetGuard`) | `leakage_prevention.md` |
| Modelos | Parâmetros da monografia; LSTM e RF com 5 sementes; Perceptron com score contínuo para ROC/PR | `model_specification.md` |
| Calibração | Isotônica (RF) e Platt (LSTM e Perceptron) ajustadas por validação cruzada em blocos dentro da validação | `calibration_threshold.md` |
| Limiar | Alerta se `p ≥ limiar`; critério principal F1 sobre probabilidades fora da amostra; também 0,50, regra de valor e estilo original | `calibration_threshold.md` |
| Avaliação | AUC, Brier, precisão, recall, F1, curvas e matrizes; IC 95 % por bootstrap em blocos de 20 dias; decomposição passo a passo do original ao revisado | `evaluation_report.md` |
| Backtest | Só o teste; decisão em `t`, posição em `t+1`; custos de 0, 10 e 25 bps por troca (suposição); conferência em planilha | `backtest_report.md` |

## O que mudou em relação à versão original

| # | Original | Revisado | Efeito medido |
|---|---|---|---|
| 1 | Alvo = `max` de 10 janelas retroativas (9 dias já presentes nas variáveis) | Alvo formal dos 10 dias seguintes | Principal: AUC do RF 0,76 → 0,55 |
| 2 | `StandardScaler` ajustado na base inteira | Só no treino | Dentro da variação entre sementes |
| 3 | `fillna(0)` nas variáveis; `ewm` sem `min_periods` | Sem preenchimento; aquecimento causal | Dentro da variação entre sementes |
| 4 | Sem purge (9 rótulos por fronteira olhavam para o bloco seguinte) | Purge | Dentro da variação entre sementes |
| 5 | Calibração e limiar na mesma validação, regra `>` | Validação cruzada em blocos, regra `≥` | O F1 degenera para "alertar sempre" com a calibração honesta |
| 6 | Uma semente por modelo | 5 sementes (RF e LSTM) | Revela a dispersão do LSTM (AUC de 0,30 a 0,69) |
| 7 | Backtest no Excel em todo o período (≈ 80 % in‑sample), sem custos | Só o teste, com defasagem de 1 dia e custos | Ganho sobre o IDA não se sustenta |
| 8 | CDI dividido por 100 no código; coluna de 20 dias errada | Convenção única em decimal | Não afeta o alvo de 10 dias |

## O que não foi alterado de propósito (fidelidade)
RSI e MACD calculados sobre o retorno diário e não sobre o nível do índice; `ret_20` como soma simples; hiperparâmetros da monografia sem busca; horizonte único de 10 dias.
Essas escolhas são questionáveis e estão listadas em `limitations.md`; alterá‑las seria uma extensão, não uma reprodução.
