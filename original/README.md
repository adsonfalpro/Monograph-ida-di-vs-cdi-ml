# Versão original (rastreabilidade)

Esta pasta guarda **o que foi feito na monografia**, sem alterações metodológicas.

- `scripts/mapped_to_monograph/`: scripts da monografia. `LS.py` (Random Forest, Apêndice A), `LST.py` (LSTM, Apêndice B), `Perceptron.py` (Apêndice C),
  `M.py` e `S.py` (versões anteriores, outro alvo e outro split), `IDA.py` e `norm.py` (matrizes normalizadas digitadas à mão).
  **Única alteração:** a linha `INPUT_PATH` foi anonimizada (era um caminho local do autor). `IDA_DI.xlsx` equivale a `data/raw/Monografia - Dados.xlsx`.
- `results/`: métricas publicadas pelos scripts e os alertas por data usados no backtest do Excel (`alertas_RF_original.xlsx`, `alertas_LSTM_original.xlsx`).
- `methodology.md`: reconstrução da metodologia original, com suas ambiguidades e inconsistências (cópia de `docs/original_methodology.md`).

Outros scripts que estavam na pasta de trabalho do autor (de outros projetos) não fazem parte deste repositório.
