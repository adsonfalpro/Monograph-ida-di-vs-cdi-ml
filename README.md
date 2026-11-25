# Machine learning in the Brazilian credit market: predicting IDA-DI returns below the CDI

Reproducible implementation and revised evaluation of the experiment in the monograph *Modelos de Machine Learning Aplicado para Previsões no Mercado de Crédito Brasileiro*
(specialization in Financial Engineering, Escola Politécnica da USP). The original question was kept: **can we predict, up to 10 days ahead,
whether the IDA-DI (ANBIMA debenture index) will earn less than the CDI?** So were the models (Perceptron, Random Forest and LSTM), the features, the horizon,
the 100-day lookback and the 60/20/20 chronological split.

> The results come from an academic simulation and are not investment advice. Portuguese summary: [`README.pt-BR.md`](README.pt-BR.md).

## Main result: a null result

On the test block (815 samples, 169 events) with the revised version:

| model | AUC [95% CI] | Brier | F1 (main rule) |
|---|---|---|---|
| Perceptron | 0.41 [0.23, 0.60] | 0.194 | 0.173 |
| Random Forest (5 seeds) | 0.54 [0.40, 0.69] | 0.168 | 0.343 |
| LSTM (5 seeds) | 0.54 [0.43, 0.64] | 0.172 | 0.343 |

References: AUC of chance 0.50, Brier of the constant prediction 0.164, F1 of "alert every day" 0.343. **No model is distinguishable from chance,**
and no model-based strategy beats buy-and-hold in the IDA in the backtest (IDA 153.9, CDI 147.6, base 100, test period).
The monograph reported a different outcome. The gap comes mainly from the target definition: in the original implementation the label was 1 if any
of the next 10 rolling windows was negative, and 9 of the 10 days of each window were already in the input features. Changing only the target
drops the Random Forest AUC from 0.76 to 0.55. This is absence of evidence with low statistical power (169 events, autocorrelated windows, one
chronological split), not proof that the models are useless. See [`docs/limitations.md`](docs/limitations.md).

Full write-up (in Portuguese): [`docs/relatorio_completo.html`](docs/relatorio_completo.html) (self-contained, opens in any browser) and
[`docs/relatorio_final.md`](docs/relatorio_final.md).

## Two versions of the code

| Folder | What it is |
|---|---|
| `original/` | The monograph's scripts as they were (only one local path was anonymized), their results and methodology notes. This is the "legacy" implementation. |
| `src/` | The revised version: target defined over the *next* 10 days, chronological split with purge, probability calibration, test set opened once per model, block-bootstrap intervals and a test-period backtest. |

The folders keep these names (`original/`, `outputs/`) on purpose, because documents and scripts refer to them.

## Layout

```
original/      monograph scripts, results and methodology (unchanged logic)
data/          only README.md: how to obtain the data (the data are NOT in this repository)
src/           data, targets, features, validation, models, calibration, evaluation, backtest, report
scripts/       run_all.sh (runs the whole pipeline in order)
tests/         66 tests (the ones that need the data are skipped when data/raw is empty)
notebooks/     01 data audit, 02 features, 03 results analysis
outputs/       figures/, tables/ and metrics/ (frozen_config.json, test_access_log.json); trained models are not versioned
docs/          reports, original and revised methodology, data dictionary, leakage, calibration, evaluation, backtest, limitations
```

## Install

Python 3.11. Versions used: TensorFlow 2.20, Keras 3.11, scikit-learn 1.7.2, NumPy 2.3, pandas 2.3 (pinned in `requirements.txt`).

```bash
python -m venv .venv && source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

1. **Get the data directly from ANBIMA** (the series are public, but they are not republished here): download the IDA series from
   <https://www.anbima.com.br/pt_br/informacoes/ida.htm> for 2009-01-05 to 2025-11-19 and save them as `data/raw/Dados-IDA.xls` with the 13 columns
   described in [`data/README.md`](data/README.md).
2. Run everything, or step by step (training takes about 15 minutes and the decomposition about 25, on CPU):

```bash
bash scripts/run_all.sh
```

```bash
python -I src/data/audit.py                  # cleaned base and data-quality report
python -I src/targets/compare_targets.py     # formal target and comparison with the original
python -I src/features/compare_features.py   # features and scaler effect
python -I src/validation/build_split.py      # chronological split with purge
python -I src/models/train_all.py            # Perceptron, Random Forest (5 seeds), LSTM (5 seeds)
python -I src/calibration/freeze.py          # freeze calibrators and thresholds using fit/validation only
python -I src/calibration/evaluate_test.py   # opens the test set ONCE per model (see note)
python -I src/evaluation/evaluate.py         # figures and confidence intervals
python -I src/evaluation/ablation.py         # step-by-step decomposition (cache in outputs/metrics/ablation_cache)
python -I src/evaluation/report_phase8.py
python -I src/backtest/backtest.py
python -I src/backtest/excel_check.py && python -I src/backtest/figures_report.py
python -m pytest -q tests
```

Figures are written to `outputs/figures/` and tables to `outputs/tables/`.
`src/backtest/original_backtest.py` needs `original/results/alertas_*_original.xlsx`, which contain the data series and are not distributed.

**Note on the test set:** `evaluate_test.py` logs every access in `outputs/metrics/test_access_log.json` and refuses to evaluate a model that is already logged.
To repeat the evaluation from scratch, delete that file, knowing that you are reopening the test set.

## Limitations

- The original LSTM result (one run, no seed or TensorFlow version recorded) cannot be reproduced exactly; only a qualitative reproduction is possible.
- The origin of the Perceptron figure in the monograph (314 positives in the test set) could not be established: the available code produces 251, the same as the other models.
- Backtest transaction costs (0, 10 and 25 bp per switch) are assumptions, and the IDA is a theoretical, non-tradable portfolio.
- `outputs/backtest_check.xlsx` (not distributed) was checked with a Python formula engine (maximum difference 6.5e-13), not opened in Excel.
- One chronological split and few events (80, 122 and 169 in fit, validation and test): the intervals are wide and a regime shift is visible between periods.

## Author and license

Author: Adson Protázio Falcão. Code and text are under the MIT license (`LICENSE`); the license does not cover third-party data.
To cite, see `CITATION.cff`.
