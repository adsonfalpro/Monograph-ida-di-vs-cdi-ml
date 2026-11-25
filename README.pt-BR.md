# Machine learning no mercado de crédito brasileiro (resumo em português)

Código da monografia *Modelos de Machine Learning Aplicado para Previsões no Mercado de Crédito Brasileiro* (especialização em Engenharia Financeira, Escola Politécnica da USP).
Pergunta: é possível prever, com até 10 dias de antecedência, se o IDA-DI renderá menos que o CDI, com Perceptron, Random Forest e LSTM?

**Resultado:** nulo. No teste (815 amostras, 169 eventos), AUC de 0,41 (Perceptron), 0,54 (Random Forest) e 0,54 (LSTM), com intervalos de confiança que incluem 0,50.
Nenhum modelo se distingue do acaso e nenhuma estratégia baseada em modelo supera o IDA no *backtest*. A diferença para o resultado da primeira versão vem principalmente da definição do alvo
(a versão original usava janelas que se sobrepunham às variáveis de entrada). É ausência de evidência com poder estatístico baixo, não prova de que os modelos sejam inúteis.

- Instalação, estrutura e passo a passo para reproduzir: [`README.md`](README.md) (em inglês). Para rodar tudo: `bash scripts/run_all.sh`.
- **Os dados não estão neste repositório e a ANBIMA não é republicada aqui:** obtenha-os diretamente na ANBIMA (fonte, período e colunas em [`data/README.md`](data/README.md)).
- As pastas `original/` (scripts da monografia, sem alteração de lógica) e `outputs/` mantêm esses nomes.
- Relatório completo em português: [`docs/relatorio_completo.html`](docs/relatorio_completo.html) e [`docs/relatorio_final.md`](docs/relatorio_final.md).
- O texto da monografia não faz parte desta publicação.

Os resultados são uma simulação acadêmica, não recomendação de investimento. Licença MIT (código). Autor: Adson Protázio Falcão.
