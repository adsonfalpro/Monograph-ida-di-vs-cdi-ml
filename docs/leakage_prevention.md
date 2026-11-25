# Divisão temporal e prevenção de vazamento (Fase 5)

Gerado por `src/validation/build_split.py`. Código: `src/validation/split.py`. Testes: `tests/test_split.py`. Índices por amostra: `data/processed/split_indices.csv`.

## 1. Unidade e blocos

Cada amostra tem data de decisão `t`: features = linhas `t−99..t` (100 dias); alvo = retornos `t+1..t+10`. A divisão é **cronológica, sem embaralhar**, na proporção da monografia (60/20/20):

```
| fit 60 % | val 20 % | test 20 % |
```

| bloco | uso |
|---|---|
| fit | treino dos modelos; ajuste do `StandardScaler` (linhas ≤ última decisão do fit) |
| val | parada antecipada do LSTM, calibração e threshold, **separados por validação cruzada em blocos** (§2b; Fase 7) |
| test | avaliação final, uma única vez por modelo (`TestSetGuard`) |

## 2. Purge

Uma amostra do bloco anterior é removida se o seu alvo usa retornos **posteriores** à primeira data de decisão do bloco seguinte (`target_end_row > t_próximo`). Com horizonte 10 isso remove **9 amostras** no fim de cada bloco seguido por outro (fit e val). No split original (sem purge) havia 9 rótulos por fronteira olhando para dentro do bloco seguinte: 9 em treino→validação e 9 em validação→teste (mais a fronteira interna que não existia).

### 2b. Por que `val` não é dividido em duas metades

Testado: com duas metades contíguas, a metade de calibração ficaria com 110 eventos (2019‑20) e a de threshold com apenas 12 (2021‑jul/2022), insuficiente para escolher um limiar. Em vez disso, `blocked_folds(k=3)` gera três dobras contíguas de `val`; o calibrador de cada dobra é ajustado nas outras duas (com purge de 9 amostras de cada lado) e o threshold é escolhido sobre as probabilidades assim obtidas. O calibrador final (para o teste) usa `val` inteiro. A Fase 7 compara com o procedimento original (mesma validação para tudo), para mostrar quanto isso muda.

Nota: as **features** de uma amostra de validação/teste incluem dias do período anterior. Isso não é vazamento (são dados passados, conhecidos em `t`); o risco está nos **rótulos** e nas estatísticas ajustadas, tratados aqui.

## 3. Resultado

| bloco | n | primeira decisão | última decisão | último dia usado no alvo | positivos | taxa |
|---|---|---|---|---|---|---|
| fit | 2434 | 2009-08-21 | 2019-05-03 | 2019-05-17 | 80 | 3.3% |
| val | 805 | 2019-05-17 | 2022-07-27 | 2022-08-10 | 122 | 15.2% |
| test | 815 | 2022-08-10 | 2025-11-05 | 2025-11-19 | 169 | 20.7% |
| purge | 18 | 2019-05-06 | 2022-08-09 | 2022-08-23 | 0 | 0.0% |

### Observações perdidas

| etapa | linhas |
|---|---|
| base bruta | 4240 |
| linhas antes da amostra original (Duration ausente; só aquecimento) | 59 |
| primeiras 99 linhas da amostra (completar a janela de 100 dias) | 99 |
| últimas 10 linhas (sem alvo) | 10 |
| amostras disponíveis | 4072 |
| removidas por purge | 18 |
| **amostras utilizáveis** | **4054** |

## 4. Vazamentos do original e situação agora

| # | Vazamento no original | Agora |
|---|---|---|
| L1 | `StandardScaler` ajustado na base inteira | ajustado só com linhas ≤ última decisão do `fit` (Fase 4) |
| L2 | Sem purge: 9 rótulos por fronteira usavam retornos do bloco seguinte | purge automático (18 amostras) |
| L3 | RF: calibrador isotônico e threshold na **mesma** validação | calibração e threshold por validação cruzada em blocos (Fase 7) |
| L4 | LSTM: parada antecipada, calibração e threshold na mesma validação | parada antecipada em `val`; calibração/threshold como em L3 |
| L5 | Backtest com alertas em todo o período (≈ 80 % in‑sample) | o backtest (Fase 9) usará só o bloco `test` |
| L6 | Nenhum controle de acesso ao teste | `TestSetGuard`: uma liberação por modelo, em `outputs/metrics/test_access_log.json` |

## 5. Riscos que permanecem (não são vazamento, mas limitam as conclusões)

- **Poucos positivos por bloco**: fit 80, val 122, test 169. As métricas do teste terão intervalos largos (Fase 8).
- **Mudança de regime**: a taxa de positivos vai de 3.3% (fit) a 20.7% (teste). Calibração e threshold aprendidos em um regime podem não valer no seguinte; é uma limitação do problema, não do código.
- **Autocorrelação**: janelas consecutivas compartilham 99 de 100 dias e rótulos se sobrepõem; o número efetivo de observações independentes é bem menor que n. Intervalos de confiança devem usar bootstrap em blocos (Fase 8).
- Uma única divisão: a Fase 8 poderá incluir avaliação em janelas expansivas (walk‑forward), separada e rotulada.
