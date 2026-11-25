#!/usr/bin/env bash
# Reproduz o pipeline completo, na ordem. Requer data/raw/Dados-IDA.xls (ver data/README.md).
# Tempo aproximado em CPU: treino 15 min, decomposição 25 min.
# evaluate_test.py abre o teste UMA vez por modelo e recusa repetir (outputs/metrics/test_access_log.json).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f data/raw/Dados-IDA.xls ]; then
  echo "Falta data/raw/Dados-IDA.xls. Veja data/README.md." >&2
  exit 1
fi

run() { echo; echo ">>> $*"; python -I "$@"; }

run src/data/audit.py                  # base limpa e relatório de qualidade
run src/targets/compare_targets.py     # alvo formal e comparação com o original
run src/features/compare_features.py   # variáveis e efeito do scaler
run src/validation/build_split.py      # divisão temporal com purge
run src/models/train_all.py            # Perceptron, Random Forest (5 sementes), LSTM (5 sementes)
run src/calibration/freeze.py          # congela calibradores e limiares (só fit/validação)
run src/calibration/evaluate_test.py   # abre o teste uma vez por modelo
run src/evaluation/evaluate.py         # figuras e intervalos de confiança
run src/evaluation/ablation.py         # decomposição passo a passo
run src/evaluation/report_phase8.py
run src/backtest/backtest.py
if ls original/results/alertas_*_original.xlsx >/dev/null 2>&1; then
  run src/backtest/original_backtest.py   # só se você tiver as planilhas originais
else
  echo "(pulando original_backtest.py: planilhas alertas_*_original.xlsx não distribuídas)"
fi
run src/backtest/excel_check.py
run src/backtest/figures_report.py

echo; echo ">>> testes"; python -m pytest -q tests
echo; echo "Tabelas: outputs/tables/   Figuras: outputs/figures/"
