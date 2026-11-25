# Avaliação e comparação dos modelos (Fase 8)

Gerado por `src/evaluation/report_phase8.py`. Figuras em `outputs/figures/`. Tudo é calculado sobre as previsões do teste salvas na Fase 7 (`data/processed/test_predictions.csv`); nada foi retreinado nem reajustado. A decomposição da §4 retreina pipelines intermediários, mas **só mede o teste depois de cada pipeline estar definido**; ela não escolhe nada.

## 1. Resultado principal (pipeline corrigido, teste = 815 amostras, 169 positivos)

Critério principal pré‑declarado: calibração por cross‑fit + limiar de F1. Média entre sementes (RF e LSTM: 5; Perceptron: 1) e IC 95 % por bootstrap em blocos móveis de 20 dias (1000 reamostragens, mesmas datas para todas as sementes).

| modelo | AUC | Brier | F1 (regra principal) | P(AUC ≤ 0,5) |
|---|---|---|---|---|
| Perceptron | 0.410 [0.234; 0.605] | 0.194 [0.111; 0.286] | 0.173 [0.065; 0.283] | 0.86 |
| Random Forest | 0.543 [0.397; 0.688] | 0.168 [0.105; 0.238] | 0.343 [0.199; 0.478] | 0.27 |
| LSTM | 0.544 [0.426; 0.642] | 0.172 [0.097; 0.254] | 0.343 [0.199; 0.478] | 0.24 |

Referências no teste: AUC de um acaso 0,500; Brier prevendo a prevalência 0.164; F1 de alertar sempre 0.343 (precisão 0.207, recall 1). (A prevalência do teste só é conhecida depois; é uma linha de base ideal.)

Diferenças de AUC entre modelos:

| comparação | diferença [IC 95 %] |
|---|---|
| AUC Random Forest − Perceptron | +0.133 [+0.009; +0.269] |
| AUC LSTM − Perceptron | +0.134 [-0.030; +0.296] |
| AUC LSTM − Random Forest | +0.001 [-0.121; +0.115] |

**Leitura.** Todos os intervalos de AUC contêm 0,5, nenhum Brier é distinguível da previsão constante e o F1 de RF e LSTM coincide com o de alertar sempre (a regra principal degenera). **Nenhum modelo se distingue do acaso**. Entre os modelos, a única diferença com intervalo fora de zero é RF − Perceptron (+0,133; IC 95 %: +0,009 a +0,269), que vem em boa parte de o Perceptron ficar abaixo de 0,50 no ponto e é evidência fraca (três comparações, sem correção para múltiplas comparações); a LSTM não se distingue de nenhum dos dois. Tudo isso com este teste (169 eventos, janelas muito autocorrelacionadas). Isso é ausência de evidência, não evidência de que os modelos sejam inúteis; o poder do teste é baixo.

Figuras (mesmo conjunto de teste e critérios para os três modelos): `confusion_counts.png`, `confusion_normalized.png` (semente 42, escolhida de antemão), `roc_curve.png`, `pr_curve.png`, `calibration_curve.png` (todas as sementes), `real_vs_predicted.png`, `metrics_with_ci.png`.

## 2. Outras regras de decisão no teste (média entre sementes)

| modelo | procedimento | regra | precisão | recall | F1 | alerta |
|---|---|---|---|---|---|---|
| Perceptron | crossfit | f1 | 0.114 | 0.361 | 0.173 | 66% |
| Perceptron | crossfit | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0% |
| Perceptron | crossfit | value | 0.000 | 0.000 | 0.000 | 0% |
| Perceptron | native | step_score>=0 | 0.192 | 0.272 | 0.225 | 29% |
| Perceptron | original_style | f1 | 0.210 | 0.225 | 0.217 | 22% |
| Perceptron | original_style | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0% |
| Perceptron | original_style | value | 0.210 | 0.225 | 0.217 | 22% |
| Random Forest | crossfit | f1 | 0.207 | 1.000 | 0.343 | 100% |
| Random Forest | crossfit | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0% |
| Random Forest | crossfit | value | 0.000 | 0.000 | 0.000 | 0% |
| Random Forest | original_style | f1 | 0.221 | 0.785 | 0.344 | 73% |
| Random Forest | original_style | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0% |
| Random Forest | original_style | value | 0.195 | 0.196 | 0.195 | 21% |
| LSTM | crossfit | f1 | 0.207 | 1.000 | 0.343 | 100% |
| LSTM | crossfit | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0% |
| LSTM | crossfit | value | 0.033 | 0.001 | 0.002 | 0% |
| LSTM | original_style | f1 | 0.449 | 0.179 | 0.209 | 10% |
| LSTM | original_style | fixed_0.5 | 0.000 | 0.000 | 0.000 | 0% |
| LSTM | original_style | value | 0.456 | 0.174 | 0.205 | 13% |

O limiar 0,50 nunca emite alerta (probabilidades calibradas não chegam a 0,5 num problema com prevalência de 3–20 %).

## 3. Reconciliação com os números da monografia

O passo S0 executa o código original (literal) sobre `Monografia - Dados.xlsx`, para saber o que da monografia é reproduzível.

| item | monografia | S0 reproduzido |
|---|---|---|
| RF: matriz (VN/FP/FN/VP) | 319 / 245 / 57 / 194 | 387 / 177 / 85 / 166 (média de 5 sementes) |
| RF: F1 | 0,562 | 0.535 ± 0.121 |
| RF: AUC (prob. calibrada) | 0,737 | 0.757 ± 0.025 |
| RF: Brier | 0,175 | 0.177 ± 0.007 |
| RF: limiar F1 | 0,237 | 0.249 ± 0.029 |
| LSTM: matriz | 562 / 2 / 190 / 61 | 521 / 43 / 138 / 113 (média de 3 sementes) |
| LSTM: F1 | 0,389 | 0.528 ± 0.156 |
| LSTM: AUC | 0,810 | 0.778 ± 0.109 |
| LSTM: Brier | 0,198 | 0.182 ± 0.026 |
| LSTM: limiar F1 | 0,269 | 0.304 ± 0.121 |
| Perceptron: matriz | 390 / 111 / 174 / 140 (314 positivos) | 402 / 162 / 125 / 126 (251 positivos) |

- **RF: reproduzido aproximadamente** (mesma ordem de grandeza em F1, AUC e matriz, mesma prevalência de teste de 251 positivos). A diferença restante é compatível com a versão do scikit‑learn e com a base `IDA_DI.xlsx`, que não temos.
- **LSTM: reprodução apenas qualitativa.** O resultado da monografia (AUC 0,81, recall 24 %, 2 falsos positivos) é uma única execução (TensorFlow/semente não registrados); aqui o resultado varia entre sementes (ver a dispersão acima). O valor publicado não pode ser confirmado nem refutado.
- **Perceptron: não reproduzido.** A figura da monografia tem 314 positivos no teste; o código `Perceptron.py` com os dados disponíveis gera 251, igual ao RF e ao LSTM. Testei 168 combinações de coluna/horizonte/janela do alvo e nenhuma produz 314. A origem dessa figura permanece desconhecida e fica registrada como tal.

## 4. De onde vem a diferença: decomposição passo a passo

Cada linha acrescenta **uma** correção à anterior (mesmas 4071 janelas do original). Média entre sementes (RF: 5, LSTM: 3, Perceptron: 1) de AUC sobre a probabilidade calibrada (score bruto no Perceptron), Brier, F1 e prevalência no teste. Os passos S0–S4 usam a regra do original (calibração e limiar F1 na mesma validação, alerta se `p > limiar`); S5 usa cross‑fit.

### Perceptron

| passo | AUC | Brier | precisão | recall | F1 | positivos no teste |
|---|---|---|---|---|---|---|
| S0 original (literal) | 0.661 | – | 0.438 | 0.502 | 0.468 | 251 |
| S1 + alvo formal | 0.400 | – | 0.193 | 0.195 | 0.194 | 169 |
| S2 + scaler só no treino | 0.408 | – | 0.194 | 0.266 | 0.224 | 169 |
| S3 + variáveis corrigidas | 0.402 | – | 0.181 | 0.308 | 0.228 | 169 |
| S4 + purge | 0.410 | – | 0.190 | 0.308 | 0.235 | 169 |
| S5 + cross-fit (4071 janelas, retreinado) | 0.410 | – | 0.190 | 0.308 | 0.235 | 169 |

### Random Forest

| passo | AUC | Brier | precisão | recall | F1 | positivos no teste |
|---|---|---|---|---|---|---|
| S0 original (literal) | 0.757 ± 0.025 | 0.177 ± 0.007 | 0.564 ± 0.204 | 0.663 ± 0.267 | 0.535 ± 0.121 | 251 |
| S1 + alvo formal | 0.546 ± 0.056 | 0.167 ± 0.004 | 0.259 ± 0.092 | 0.459 ± 0.290 | 0.284 ± 0.100 | 169 |
| S2 + scaler só no treino | 0.548 ± 0.037 | 0.167 ± 0.003 | 0.245 ± 0.055 | 0.504 ± 0.312 | 0.302 ± 0.099 | 169 |
| S3 + variáveis corrigidas | 0.535 ± 0.032 | 0.167 ± 0.003 | 0.192 ± 0.039 | 0.328 ± 0.206 | 0.234 ± 0.083 | 169 |
| S4 + purge | 0.582 ± 0.031 | 0.163 ± 0.002 | 0.196 ± 0.112 | 0.299 ± 0.188 | 0.235 ± 0.138 | 169 |
| S5 + cross-fit (4071 janelas, retreinado) | 0.582 ± 0.031 | 0.163 ± 0.002 | 0.207 ± 0.000 | 1.000 ± 0.000 | 0.343 ± 0.000 | 169 |

### LSTM

| passo | AUC | Brier | precisão | recall | F1 | positivos no teste |
|---|---|---|---|---|---|---|
| S0 original (literal) | 0.778 ± 0.109 | 0.182 ± 0.026 | 0.808 ± 0.168 | 0.449 ± 0.247 | 0.528 ± 0.156 | 251 |
| S1 + alvo formal | 0.610 ± 0.144 | 0.168 ± 0.019 | 0.264 ± 0.256 | 0.219 ± 0.201 | 0.224 ± 0.194 | 169 |
| S2 + scaler só no treino | 0.514 ± 0.133 | 0.174 ± 0.012 | 0.283 ± 0.312 | 0.323 ± 0.457 | 0.191 ± 0.183 | 169 |
| S3 + variáveis corrigidas | 0.561 ± 0.180 | 0.172 ± 0.008 | 0.364 ± 0.310 | 0.280 ± 0.346 | 0.208 ± 0.174 | 169 |
| S4 + purge | 0.592 ± 0.086 | 0.172 ± 0.011 | 0.397 ± 0.157 | 0.231 ± 0.209 | 0.218 ± 0.107 | 169 |
| S5 + cross-fit (4071 janelas, retreinado) | 0.592 ± 0.086 | 0.172 ± 0.011 | 0.207 ± 0.000 | 1.000 ± 0.000 | 0.343 ± 0.000 | 169 |

Linha de base 'alertar sempre' em cada passo: F1 = 2p/(1+p) com p = prevalência do teste (0,308 → 0,471; 0,207 → 0,343).

## 5. Conclusões da Fase 8

1. **A maior parte da diferença vem do alvo.** Só trocar o alvo original pelo formal derruba o AUC de 0.76 para 0.55 no RF, de 0.78 para 0.61 no LSTM e de 0.66 para 0.40 no Perceptron. O alvo original marcava 1 se **qualquer** das 10 janelas deslizantes fosse negativa, e 9 dos 10 dias de cada janela já estavam nas variáveis de entrada (Fase 3): parte do rótulo era informação passada que o modelo já via. Isso, e não a previsão do futuro, explica a maior parte do desempenho da monografia.

2. **As demais correções (scaler só no treino, variáveis, purge, cross‑fit) movem o AUC bem menos que a troca do alvo**, e as mudanças (até cerca de 0,05 no RF e 0,10 no LSTM entre passos) são da ordem do desvio‑padrão entre sementes (RF ≈ 0,03; LSTM 0,09–0,18). Não fiz um teste formal de cada passo; a leitura é que nenhuma dessas correções, isolada, explica a queda. O cross‑fit muda a regra de decisão (F1 → alertar sempre), não a capacidade de ordenar.
   *Nota sobre S5:* não é idêntico ao pipeline final da Fase 7. O S5 retreina os modelos com 4071 janelas (as do original), e o AUC da tabela é medido sobre a probabilidade calibrada; a Fase 7 usa 4072 janelas e o score bruto (RF 0,54 e LSTM 0,54). A distância entre os dois (RF 0,58 contra 0,54) mostra o quanto o AUC varia com pequenas mudanças do pipeline, e reforça que as diferenças entre S2 e S5 estão dentro do ruído.
   *Nota sobre o teste:* o teste só foi aberto uma vez por modelo congelado (Fase 7). Os pipelines retreinados nesta decomposição também foram medidos no teste, apenas como diagnóstico; nenhuma escolha foi feita com base neles.

3. **Com o alvo formal, nenhum modelo se distingue do acaso** (§1). O Perceptron em particular fica abaixo de 0,5 no ponto, sem significância.

4. **O que isso muda para a monografia:** o resultado "RF e LSTM superam IDA e CDI" depende de um alvo cujo desempenho vem em grande parte da sobreposição com as variáveis e de um backtest majoritariamente in‑sample (Fase 9 mede isso). A conclusão correta, com os dados e o método corrigidos, é **ausência de evidência de previsibilidade** de um desempenho do IDA‑DI abaixo do CDI em 10 dias com estas variáveis.

5. **Limites desta conclusão:** teste com 169 eventos e janelas autocorrelacionadas (poder baixo); uma única divisão cronológica; hiperparâmetros da monografia sem busca; LSTM com 3–5 sementes; o CDI diário da base foi conferido pelo autor do projeto. Sugestões, não executadas aqui para não misturar com a reconstrução fiel: walk‑forward, outros horizontes, alvo contínuo (excesso de retorno).
