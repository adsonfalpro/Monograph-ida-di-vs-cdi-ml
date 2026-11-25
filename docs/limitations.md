# Limitações (em construção)

Atualizado a cada fase. Itens mantidos por fidelidade à monografia ou impossíveis de resolver com os arquivos disponíveis.

## Dados (Fase 2)
- `CDI_dia` muda em apenas 699 dias (igual ao dia anterior em 83,5 %); a série foi conferida pelo autor do projeto.
- `IDA_DI` (retorno) diverge da variação do nível do índice em 4 dias; usada a coluna de retorno, como na monografia (diferença acumulada 0,2 %).
- Origem dos dados (ANBIMA) informada pela monografia; os dados não são redistribuídos neste repositório (veja `data/README.md`).

## Alvo (Fase 3)
- Prevalência muito instável: 3,3 % no treino provisório, 15,0 % na validação e 20,7 % no teste; anos sem nenhum positivo. Poucos eventos de treino (~80).

## Variáveis (Fase 4)
- RSI e MACD calculados sobre o **retorno diário** do IDA, não sobre o nível do índice (fiel ao original; não é o RSI/MACD usual).
- `ret_20` é soma simples de excessos diários; `excesso_dia` usa a forma simples.
- Valores extremos (mar/2020) mantidos; a padronização por média/desvio é sensível a eles.
- Mais de 99 % das variáveis agregadas têm valor; 0,9 % (assimetria/curtose de janelas constantes, em geral `cdi_ret`) foram fixadas em 0.

## Divisão (Fase 5)
- Eventos: fit 80, val 122 (110 deles em 2019-20), test 169. Calibração/threshold têm pouco suporte fora do regime 2019-20.
- Janelas consecutivas compartilham 99 de 100 dias; n efetivo bem menor que n. Usar bootstrap em blocos nos intervalos.
- Uma só divisão cronologica; walk-forward opcional na Fase 8.

## Calibração/threshold (Fase 7)
- Com calibração fora da amostra, RF e LSTM nao superam a previsao constante em Brier (val e teste); regra F1 degenera em "alertar sempre".
- Teste ja foi aberto (11 execucoes, ver test_access_log.json). Qualquer nova escolha de metodo apos isso e post-hoc e deve ser rotulada.
- Baseline de prevalencia do teste e um oraculo (nao conhecido ex-ante).

## Avaliação (Fase 8)
- Conclusao depende de 169 eventos de teste e uma unica divisao cronologica; poder estatistico baixo.
- Perceptron da Fig. 4.1 da monografia (314 positivos): origem desconhecida, nao reproduzido (168 variantes de alvo testadas).
- Reproducao do LSTM original apenas qualitativa (TF/semente nao registrados).

## Backtest (Fase 9)
- Custos de 0/10/25 bps sao suposicoes; IDA nao e negociavel.
- Um unico periodo de teste em que o IDA superou o CDI; conclusoes dependem do regime.
- Conferencia da planilha feita com motor de formulas em Python, nao com o Excel.
