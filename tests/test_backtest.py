import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "backtest"))
from backtest import equity, max_drawdown, metrics, strategy_returns  # noqa: E402

IDA = np.array([0.003, -0.002, 0.004, 0.001, -0.005, 0.002])
CDI = np.full(6, 0.001)


def test_never_alert_is_ida_and_always_alert_is_cdi_with_one_switch():
    r, sw = strategy_returns(IDA, CDI, np.zeros(6, int), 25)
    assert np.allclose(r, IDA) and sw == 0
    r, sw = strategy_returns(IDA, CDI, np.ones(6, int), 0)
    assert np.allclose(r, CDI) and sw == 1


def test_cost_is_charged_once_per_switch_on_the_switch_day():
    alert = np.array([0, 1, 1, 0, 0, 1])
    r, sw = strategy_returns(IDA, CDI, alert, 100)  # 100 bps = 1 %
    assert sw == 3
    expected = np.where(alert == 1, CDI, IDA) - np.array([0, 1, 0, 1, 0, 1]) * 0.01
    assert np.allclose(r, expected)


def test_first_alert_counts_as_switch_from_initial_ida_position():
    r, sw = strategy_returns(IDA, CDI, np.array([1, 1, 0, 0, 0, 0]), 50)
    assert sw == 2 and r[0] == pytest.approx(CDI[0] - 0.005)


def test_signal_timing_alert_at_t_only_changes_return_of_next_day_row():
    """O vetor de retornos já está alinhado a t+1: mudar o alerta da linha k só muda a linha k (e custo em k, k+1)."""
    base = np.zeros(6, int)
    changed = base.copy(); changed[3] = 1
    ida = IDA + 0.0005   # garante IDA != CDI em todos os dias
    r0, _ = strategy_returns(ida, CDI, base, 0)
    r1, _ = strategy_returns(ida, CDI, changed, 0)
    assert (r0 != r1).tolist() == [False, False, False, True, False, False]


def test_equity_and_drawdown():
    r = np.array([0.10, -0.20, 0.05])
    c = equity(r)
    assert c == pytest.approx([110, 88, 92.4])
    assert max_drawdown(c) == pytest.approx(88 / 110 - 1)
    m = metrics(r, np.zeros(3), np.zeros(3), 0, 0.0)
    assert m["final_base100"] == pytest.approx(92.4) and m["n_dias"] == 3


def test_test_alignment_uses_next_day_returns():
    sys.path.insert(0, str(ROOT / "src" / "data"))
    from backtest import load_test_alignment, test_returns
    clean, pred, _ = load_test_alignment()
    g = pred[pred.model == "rf_seed42"].sort_values("date")
    ida, cdi, dates, _ = test_returns(clean, g["date"])
    first_decision = g["date"].iloc[0]
    row = int(np.where(clean["date"] == first_decision)[0][0])
    assert dates[0] == clean["date"].iloc[row + 1].to_datetime64()
    assert ida[0] == clean["ida_ret"].iloc[row + 1] and len(ida) == 815
