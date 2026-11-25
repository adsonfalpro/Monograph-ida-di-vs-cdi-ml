# Especificação dos modelos e treino (Fase 6)

Gerado por `src/models/report_phase6.py` a partir de `outputs/metrics/phase6_training_summary.json`. Código: `src/models/`. Testes: `tests/test_models.py`.

**Escopo desta fase:** treinar com o bloco `fit` e usar `val` apenas para desenvolvimento (parada antecipada do LSTM e diagnóstico). Nenhuma matriz de confusão, threshold ou métrica de teste foi calculada: o bloco `test` está bloqueado (`Dataset.release_test` + `TestSetGuard`). Calibração e threshold: Fase 7.

## 1. Dados de treino

`fit`: 2434 amostras, 80 positivos; `val`: 805 amostras, 122 positivos. Scaler ajustado nas linhas 59–2591 (até a última decisão do `fit`). Estatísticas de janela indefinidas fixadas em 0: 874.

| | Perceptron | Random Forest | LSTM |
|---|---|---|---|
| Entrada | 120 estatísticas agregadas da janela de 100 dias | idem | sequência 100 × 12 |
| Alvo | `Y_t` da Fase 3 | idem | idem |

## 2. Parâmetros

### Perceptron (`perceptron.py`)

Camada única, degrau (`≥ 0 → 1`), pesos e viés 0, `learning_rate = 0,01`, `n_epochs = 100`, passagem em ordem aleatória por época (gerador próprio com semente 42, sem alterar o estado global do NumPy), regra `w ← w + η (y − ŷ) x`, sem pesos de classe. É fiel ao Apêndice C. Acrescenta `decision_function` (score contínuo) para ROC/PR.

### Random Forest (`random_forest.py`)

| parâmetro | valor |
|---|---|
| n_estimators | 800 |
| max_depth | None |
| min_samples_leaf | 2 |
| max_features | sqrt |
| class_weight | balanced_subsample |
| random_state | 42 |
| n_jobs | -1 |

Idêntico ao Apêndice A / `LS.py`. A calibração isotônica do original (`CalibratedClassifierCV(cv='prefit')`) passa para a Fase 7.

### LSTM (`lstm_model.py`)

`Input(100,12)` → `LSTM(64, return_sequences=True)` → `Dropout(0,3)` → `LSTM(32)` → `BatchNormalization` → `Dense(32, relu)` → `Dropout(0,2)` → `Dense(1, sigmoid)`; 32 897 parâmetros. Adam (1e‑3), `binary_crossentropy`, até 50 épocas, batch 32, `class_weight` balanceado, `EarlyStopping(val_loss, paciência 5, restaura pesos)`, `ReduceLROnPlateau(0,5; paciência 3)`. Igual ao Apêndice B.

Diferenças de implementação (não metodológicas): `Input` explícito (Keras 3 não aceita `input_shape` na camada); semente via `keras.utils.set_random_seed` e `enable_op_determinism()` (CPU); **5 sementes (42–46)** em vez de uma só. Versões: TensorFlow 2.20, Keras 3.11, scikit‑learn 1.7.2, NumPy 2.3, pandas 2.3, Python 3.11 (a monografia só cita Python 3.9).

Observação: a parada antecipada usa `val` (como no original). Isso torna as pontuações do LSTM em `val` levemente otimistas; a Fase 7 considera isso ao calibrar.

## 3. Resultado do treino (diagnóstico, sem teste)

| modelo | AUC em `fit` | AUC em `val` | observações |
|---|---|---|---|
| Perceptron | 0.912 | 0.817 | erros/época: 117 → 64; prevê positivo em 22.5% de `val` |
| Random Forest (5 sementes) | 1.000 ± 0.000 (mín 1.000, máx 1.000) | 0.577 ± 0.023 (mín 0.549, máx 0.608) | |
| LSTM (5 sementes) | 0.980 ± 0.007 (mín 0.973, máx 0.988) | 0.675 ± 0.036 (mín 0.618, máx 0.702) | épocas executadas 8.200 ± 1.095 (mín 7.000, máx 10.000); melhor época 3.200 ± 1.095 (mín 2.000, máx 5.000) |

LSTM por semente:

| semente | épocas | melhor época | melhor val_loss | AUC fit | AUC val |
|---|---|---|---|---|---|
| 42 | 8 | 3 | 0.4758 | 0.978 | 0.701 |
| 43 | 7 | 2 | 0.5641 | 0.973 | 0.695 |
| 44 | 8 | 3 | 0.4683 | 0.986 | 0.662 |
| 45 | 10 | 5 | 0.4252 | 0.988 | 0.702 |
| 46 | 8 | 3 | 0.5045 | 0.977 | 0.618 |

## 4. Leitura (sem interpretar além do que os dados permitem)

- **Sobreajuste forte** em RF e LSTM: AUC em `fit` próxima de 1 e muito menor em `val`. É esperado com apenas 80 positivos de treino e 120 atributos (ou 1200 entradas por janela). O LSTM atinge o melhor `val_loss` nas primeiras épocas.
- A dispersão entre sementes do LSTM mede a incerteza de treino; os resultados da Fase 8 serão reportados com ela, não com uma semente escolhida.
- Os AUC em `val` **não** são a medida final (a validação foi usada para parada antecipada e para o diagnóstico). Nenhum hiperparâmetro foi ajustado.
- Não há razão para esperar que o ranking RF > Perceptron da monografia se mantenha; a comparação válida será feita no teste, na Fase 8.
