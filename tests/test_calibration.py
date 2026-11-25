import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("calibration", "validation"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from calibration import (  # noqa: E402
    IsotonicCalibrator, PlattCalibrator, confusion, cost_table, crossfit_calibrate, prf, select_threshold,
)
from evaluate_test import apply_calibration  # noqa: E402


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < s ** 2).astype(int)
    return s, y


@pytest.mark.parametrize("cls", [PlattCalibrator, IsotonicCalibrator])
def test_frozen_params_reproduce_calibrator(cls):
    s, y = toy()
    cal = cls().fit(s, y)
    q = np.linspace(-0.2, 1.2, 50)             # inclui pontos fora do intervalo de ajuste
    assert np.allclose(apply_calibration(cal.params(), q), cal.predict(q))


def test_threshold_uses_ge_and_matches_brute_force():
    s, y = toy(seed=1)
    p = np.round(s, 2)
    thr = select_threshold(y, p, "f1")
    grid = np.unique(p)
    f1 = np.array([prf(y, p, t)["f1"] for t in grid])
    assert prf(y, p, thr)["f1"] == pytest.approx(f1.max())
    assert thr in grid                           # um valor observado, usado com >=
    # empate: escolhe o maior limiar entre os máximos
    assert thr == grid[np.where(f1 >= f1.max() - 1e-12)[0].max()]


def test_confusion_and_prf_consistency():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.5, 0.1, 0.6, 0.5])
    tn, fp, fn, tp = confusion(y, p, 0.5)
    assert (tn, fp, fn, tp) == (1, 2, 1, 2)       # p == 0.5 conta como alerta
    m = prf(y, p, 0.5)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(2 / 3)


def test_fixed_and_value_rules():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.2])
    exc = np.array([-0.02, -0.01, 0.005, 0.005])  # migrar para o CDI vale -excesso
    assert select_threshold(y, p, "fixed_0.5") == 0.5
    assert select_threshold(y, p, "value", exc) == 0.8   # alerta só onde o excesso é negativo


def test_crossfit_never_uses_heldout_labels():
    rng = np.random.default_rng(3)
    n = 300
    rows = np.arange(n) + 100
    pos = np.arange(n)
    s = rng.uniform(size=n)
    y = (rng.uniform(size=n) < s).astype(int)
    base = crossfit_calibrate(s, y, pos, rows, 10, "isotonic", k=3)
    y2 = y.copy()
    y2[:100] = 1 - y2[:100]                       # inverte rótulos da 1ª dobra
    alt = crossfit_calibrate(s, y2, pos, rows, 10, "isotonic", k=3)
    # as previsões da 1ª dobra usam calibradores ajustados nas OUTRAS dobras -> não mudam
    assert np.array_equal(base[:100], alt[:100])
    assert not np.array_equal(base[100:], alt[100:])


def test_cost_table_breakeven():
    y = np.array([1, 1, 0, 0, 0])
    e = np.array([-0.02, -0.04, 0.01, 0.01, 0.01])
    c = cost_table(y, e)
    assert c["gain_per_TP"] == pytest.approx(0.03) and c["cost_per_FP"] == pytest.approx(0.01)
    assert c["breakeven_precision"] == pytest.approx(0.25)
