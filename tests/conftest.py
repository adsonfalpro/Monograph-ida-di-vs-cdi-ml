"""Pula os testes que dependem dos dados (que não são distribuídos com o repositório).

* Sem ``data/raw/Dados-IDA.xls``: pula os testes que reconstroem a base, o alvo, as variáveis, a divisão e os datasets.
* Sem ``data/processed/test_predictions.csv`` (gerado por ``src/calibration/evaluate_test.py``): pula o teste de alinhamento do backtest.
Com os dados e o pipeline executado, todos os testes rodam.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NEEDS_RAW = {"test_data.py", "test_features.py", "test_models.py", "test_split.py", "test_target.py"}
HAS_RAW = (ROOT / "data" / "raw" / "Dados-IDA.xls").exists()
HAS_PRED = (ROOT / "data" / "processed" / "test_predictions.csv").exists()


def pytest_collection_modifyitems(config, items):
    skip_raw = pytest.mark.skip(reason="requer data/raw/Dados-IDA.xls (veja data/README.md)")
    skip_pred = pytest.mark.skip(reason="requer data/processed/test_predictions.csv (rode o pipeline até evaluate_test.py)")
    for item in items:
        name = Path(str(item.fspath)).name
        if not HAS_RAW and name in NEEDS_RAW:
            item.add_marker(skip_raw)
        elif not HAS_PRED and name == "test_backtest.py" and "alignment" in item.name:
            item.add_marker(skip_pred)
