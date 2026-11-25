# Calibração e escolha do threshold (Fase 7)

Gerado por `src/calibration/report_phase7.py`. Código: `src/calibration/`. Testes: `tests/test_calibration.py`.

## 1. Procedimento

1. Modelo treinado em `fit` (Fase 6). 2. Pontuações brutas em `val`. 3. **Calibração em validação cruzada em blocos** (3 dobras contíguas de `val`, purge de 9 amostras de cada lado): cada observação é calibrada por um calibrador que não a viu. 4. Thresholds escolhidos sobre essas probabilidades fora da amostra. 5. Calibrador final ajustado em todo `val`. 6. Tudo congelado em `outputs/metrics/frozen_config.json` (SHA‑256 `85eb81f3f4a908a2…`). 7. Teste avaliado **uma vez por modelo/semente** (`TestSetGuard`; registro em `outputs/metrics/test_access_log.json`).

Calibradores (como no original): isotônica para o RF; Platt (`LogisticRegression`) para o LSTM e, **acréscimo**, para o Perceptron (sobre o score contínuo; o original não tinha probabilidades). Regra de alerta: `p ≥ limiar` (o original usava `>`).

Regras de threshold avaliadas: `fixed_0.5`; `f1` (máximo F1, como na monografia); `value` (maximiza o ganho acumulado de migrar para o CDI, isto é, a soma de −excesso dos dias com alerta). Procedimentos: `crossfit` (limpo) e `original_style` (calibra e escolhe o limiar na mesma validação). Perceptron: também `native` (degrau no score ≥ 0).

**Critério principal, declarado antes de abrir o teste:** `crossfit` + `f1`. Os demais são reportados sem escolha posterior.

## 2. Custos dos erros (medidos em `fit` e `val`, nunca no teste)

Retorno em excesso dos 10 dias seguintes (decimal). Falso positivo: migrar para o CDI quando o IDA ganharia (`cost_per_FP`). Verdadeiro positivo: perda evitada (`gain_per_TP`). Falso negativo: perda não evitada (= `gain_per_TP`). **Precisão de equilíbrio**: abaixo dela, alertar não compensa.

| bloco | positivos | negativos | ganho por TP / custo de FN | custo por FP | razão FN/FP | precisão de equilíbrio |
|---|---|---|---|---|---|---|
| fit | 80 | 2354 | 0.0012 | 0.0006 | 2.1 | 32.1% |
| val | 122 | 683 | 0.0065 | 0.0015 | 4.3 | 18.7% |

FN custa 2× (fit) a 4× (val) o FP em retorno esperado, mas um evento é raro; por isso o alerta só compensa quando a precisão supera 19–32 %. A prevalência no teste é 20.7%, ou seja, **alertar todo dia** já tem precisão 20.7% e F1 0.343.

## 3. Validação: efeito do procedimento (média das sementes)

| modelo | procedimento | Brier em val | F1 em val (regra f1) | taxa de alerta |
|---|---|---|---|---|
| Perceptron | crossfit | 0.156 | 0.284 | 81.1% |
| Perceptron | original_style | 0.118 | 0.541 | 12.4% |
| Random Forest | crossfit | 0.189 | 0.267 | 96.3% |
| Random Forest | original_style | 0.124 | 0.305 | 60.2% |
| LSTM | crossfit | 0.169 | 0.264 | 99.6% |
| LSTM | original_style | 0.123 | 0.374 | 38.8% |

Referências em `val`: Brier de prever só a prevalência = 0,129; F1 de alertar sempre = 0,263.

Leitura: com a calibração fora da amostra, o Brier de RF e LSTM em `val` é **pior** que o de uma constante, e a regra F1 vira "alertar quase sempre" (limiar ≈ 0). O procedimento original parece bem melhor (Brier ≈ 0,12; F1 0,3–0,5), mas porque calibra e mede nos mesmos dados. As dobras de `val` têm regimes muito diferentes (110 dos 122 eventos estão em 2019‑20), então o calibrador treinado nas outras dobras não generaliza: isso é a mudança de regime, e não um defeito do cross‑fit.

## 4. Teste (uma única avaliação por modelo/semente)

AUC e Brier não dependem do limiar. Média ± desvio entre sementes (Perceptron: 1 execução):

| modelo | AUC no teste | Brier no teste |
|---|---|---|
| Perceptron | 0.410 | 0.194 |
| Random Forest | 0.543 ± 0.063 | 0.168 ± 0.002 |
| LSTM | 0.544 ± 0.171 | 0.172 ± 0.008 |

Referências: AUC de um acaso = 0,500; Brier prevendo a prevalência = 0.164.

Por semente (LSTM e RF):

| modelo | AUC | Brier |
|---|---|---|
| perceptron_seed42 | 0.410 | 0.194 |
| rf_seed42 | 0.480 | 0.169 |
| rf_seed43 | 0.626 | 0.164 |
| rf_seed44 | 0.488 | 0.167 |
| rf_seed45 | 0.534 | 0.168 |
| rf_seed46 | 0.589 | 0.169 |
| lstm_seed42 | 0.694 | 0.162 |
| lstm_seed43 | 0.430 | 0.180 |
| lstm_seed44 | 0.303 | 0.178 |
| lstm_seed45 | 0.618 | 0.175 |
| lstm_seed46 | 0.675 | 0.163 |

### Critério principal (cross‑fit + F1) e demais regras no teste

Média entre sementes de precisão / recall / F1 / taxa de alerta:

| modelo | procedimento | regra | precisão | recall | F1 | alerta |
|---|---|---|---|---|---|---|
| Perceptron | crossfit | f1 **(principal)** | 0.114 | 0.361 | 0.173 | 65.6% |
| Perceptron | crossfit | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0.0% |
| Perceptron | crossfit | value | 0.000 | 0.000 | 0.000 | 0.0% |
| Perceptron | native | step_score>=0 | 0.192 | 0.272 | 0.225 | 29.3% |
| Perceptron | original_style | f1 | 0.210 | 0.225 | 0.217 | 22.2% |
| Perceptron | original_style | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0.0% |
| Perceptron | original_style | value | 0.210 | 0.225 | 0.217 | 22.2% |
| Random Forest | crossfit | f1 **(principal)** | 0.207 | 1.000 | 0.343 | 100.0% |
| Random Forest | crossfit | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0.0% |
| Random Forest | crossfit | value | 0.000 | 0.000 | 0.000 | 0.0% |
| Random Forest | original_style | f1 | 0.221 | 0.785 | 0.344 | 72.9% |
| Random Forest | original_style | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0.0% |
| Random Forest | original_style | value | 0.195 | 0.196 | 0.195 | 21.2% |
| LSTM | crossfit | f1 **(principal)** | 0.207 | 1.000 | 0.343 | 100.0% |
| LSTM | crossfit | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0.0% |
| LSTM | crossfit | value | 0.033 | 0.001 | 0.002 | 0.5% |
| LSTM | original_style | f1 | 0.449 | 0.179 | 0.209 | 9.6% |
| LSTM | original_style | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0.0% |
| LSTM | original_style | value | 0.456 | 0.174 | 0.205 | 12.8% |

Linha de base "alertar sempre": precisão 0.207, recall 1,000, F1 0.343, alerta 100 %.

Tabela completa por semente: `outputs/tables/phase7_test_metrics.csv`; previsões salvas em `data/processed/test_predictions.csv` (a Fase 8 não recalcula nada).

## 5. O que se pode e não se pode dizer

- O AUC no teste fica entre ~0,3 e ~0,7 conforme a semente, em média próximo de 0,5; o Brier médio de todos os modelos é pior ou igual ao de prever a prevalência do teste (0,164; uma linha de base que na prática não se conhece de antemão), e só 3 de 11 execuções ficam abaixo, por menos de 0,002. **Com este alvo, esta divisão e estes dados, não há evidência de capacidade preditiva fora da amostra** para nenhum dos três modelos.
- A regra principal (F1 com cross‑fit) resulta em alertar em quase todos os dias, ou seja, F1 ≈ o de "alertar sempre"; não é uma previsão, é uma carteira que fica no CDI. Isso é um achado, não foi "corrigido" depois de ver o teste.
- Esses números **não reproduzem** os da monografia (RF F1 0,56; LSTM AUC 0,81). O período de teste é o mesmo (as últimas 815 amostras); a Fase 8 decompõe a diferença passo a passo e mostra que ela vem principalmente da troca do alvo (o alvo original compartilhava 9 de 10 dias com as variáveis), e não das demais correções. Intervalos de confiança: bootstrap em blocos (Fase 8).
- Limitações: poucos eventos (fit 80, val 122, teste 169), regime instável e janelas sobrepostas; os testes de significância devem ser lidos com essas ressalvas.
