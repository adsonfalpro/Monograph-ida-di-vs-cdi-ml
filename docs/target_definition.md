# Definição da variável‑alvo (Fase 3)

Gerado por `src/targets/compare_targets.py`. Código: `src/targets/build_target.py`. Testes: `tests/test_target.py`.

## 1. Definição adotada

Com `t` = última data observada (data de decisão) e `h = 10` dias úteis:

```
R_IDA(t,t+h) = Π_{i=1..h}(1 + r_IDA[t+i]) − 1
R_CDI(t,t+h) = Π_{i=1..h}(1 + r_CDI[t+i]) − 1
Y_t = 1  se R_IDA(t,t+h) < R_CDI(t,t+h);   Y_t = 0  caso contrário (empate = 0)
```

- Só entram retornos de `t+1 … t+h`. As últimas `h` linhas não têm alvo (NaN, nunca preenchido).
- `r_IDA` = `ida_ret` (decisão: série da monografia) e `r_CDI` = `cdi_ret`, ambos em decimal.
- A coluna `target_end_idx` guarda a linha do último retorno usado pelo alvo, para o purge da Fase 5.
- É equivalente à convenção de `M.py` (retorno retroativo de 10 dias terminando em `t+10`): `True`.
- Usar a forma simples (`R_IDA < R_CDI`) ou o excesso geométrico `(1+R_IDA)/(1+R_CDI) − 1 < 0` dá o mesmo rótulo.

## 2. Alvo original (scripts finais)

```
drawdown[t] = 1[ IDA_10_dias[t] < CDI_10_dias[t] ]      # janela retroativa t-9..t
janela i = linhas i..i+99;   y_i = max(drawdown[i+100 .. i+109])
```

Os 10 rótulos `drawdown[i+100..i+109]` usam janelas retroativas que cobrem os dias `i+91 … i+109`. Portanto, **os dias `i+91 … i+99` (9 dias) já estão na janela de features** e só `i+100 … i+109` são futuros. Além disso, o `max` marca 1 se **qualquer** das 10 janelas deslizantes for negativa. Não é "retorno dos próximos 10 dias". Como cada janela contém dias do passado, o alvo original é um **superconjunto** do novo: nunca há `novo=1` com `original=0` (ver tabela).

## 3. Diferença medida (mesmas 4071 janelas, split 60/20/20 de referência)

Concordância entre os dois alvos: **93.9%**.

| original \ novo | 0 | 1 |
|---|---|---|
| 0 | 3453 | 0 |
| 1 | 247 | 371 |

Positivos só no original: 247; só no novo: 0; em ambos: 371.

Taxa de positivos por conjunto (split 60/20/20 apenas para comparação; a divisão definitiva é a da Fase 5):

| conjunto | n | original | novo |
|---|---|---|---|
| treino | 2442 | 6.8% | 3.3% |
| validacao | 814 | 24.8% | 15.0% |
| teste | 815 | 30.8% | 20.7% |

Taxa por ano:

| ano | original | novo |
|---|---|---|
| 2009 | 0.0% | 0.0% |
| 2010 | 4.0% | 0.4% |
| 2011 | 4.4% | 0.8% |
| 2012 | 6.8% | 2.4% |
| 2013 | 0.0% | 0.0% |
| 2014 | 0.0% | 0.0% |
| 2015 | 16.4% | 8.4% |
| 2016 | 16.3% | 12.7% |
| 2017 | 5.2% | 1.6% |
| 2018 | 7.6% | 4.0% |
| 2019 | 43.9% | 28.5% |
| 2020 | 31.1% | 17.5% |
| 2021 | 2.8% | 0.0% |
| 2022 | 8.4% | 4.8% |
| 2023 | 41.4% | 30.9% |
| 2024 | 30.0% | 17.4% |
| 2025 | 32.9% | 21.6% |

## 4. O que isto implica

- Os resultados da monografia (RF, LSTM, Perceptron) foram treinados e avaliados com o alvo original; **não são diretamente comparáveis** aos da versão corrigida. A diferença entre eles passa a ser parte da análise (Fase 8).
- A prevalência do alvo muda muito com o regime (veja a tabela por ano): é um problema de não estacionariedade que a divisão temporal e a escolha do threshold terão de tratar; não será "corrigido" ajustando o alvo.

## 5. Testes de alinhamento temporal (`tests/test_target.py`)

1. Alvo = laço explícito de referência em todas as linhas.
2. Alterar `r[t+1]` ou `r[t+h]` pode mudar `Y_t`; alterar `r[t]` ou `r[t+h+1]` **não** muda.
3. Últimas `h` linhas sem alvo; nenhuma outra linha sem alvo.
4. Empate ⇒ 0.
5. `target_end_idx = t + h`.
6. O alvo original é reproduzido pelo código de comparação e coincide com o do `drawdown` retroativo.
7. O alvo novo coincide com o rótulo `IDA_10 < CDI_10` da linha `t+10`.
