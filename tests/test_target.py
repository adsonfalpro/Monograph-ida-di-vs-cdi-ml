import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "targets"))
from build_target import HORIZON, build_target, forward_compound, original_target  # noqa: E402
from clean_base import build_clean_base, load_raw  # noqa: E402


def synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=n),
        "ida_ret": rng.normal(0.0004, 0.001, n),
        "cdi_ret": np.full(n, 0.0004),
    })


def reference_y(df, t, h=HORIZON):
    ida = np.prod(1 + df["ida_ret"].iloc[t + 1:t + h + 1].values) - 1
    cdi = np.prod(1 + df["cdi_ret"].iloc[t + 1:t + h + 1].values) - 1
    return float(ida < cdi)


def test_matches_explicit_loop():
    df = synthetic()
    tgt = build_target(df)
    for t in range(len(df) - HORIZON):
        assert tgt["y"].iloc[t] == reference_y(df, t)


def test_last_h_rows_have_no_target_and_nothing_else():
    tgt = build_target(synthetic())
    assert tgt["y"].iloc[-HORIZON:].isna().all()
    assert tgt["y"].iloc[:-HORIZON].notna().all()


def test_target_end_idx():
    tgt = build_target(synthetic(30))
    assert (tgt["target_end_idx"].iloc[:-HORIZON] == np.arange(30 - HORIZON) + HORIZON).all()


@pytest.mark.parametrize("shift,expected_change", [(0, False), (1, True), (HORIZON, True), (HORIZON + 1, False)])
def test_temporal_alignment(shift, expected_change):
    """Alterar r[t+shift] só pode afetar Y_t se 1 <= shift <= h."""
    df = synthetic(80, seed=1)
    t = 30
    base = build_target(df)
    df2 = df.copy()
    df2.loc[t + shift, "ida_ret"] += 0.05
    new = build_target(df2)
    # Passa de uma série sem sinal para uma com salto grande: Y_t muda quando o salto está dentro do horizonte.
    if expected_change:
        assert new["fwd_ida"].iloc[t] != base["fwd_ida"].iloc[t]
    else:
        assert new["fwd_ida"].iloc[t] == base["fwd_ida"].iloc[t]
        assert new["y"].iloc[t] == base["y"].iloc[t]


def test_tie_is_zero():
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=30),
                       "ida_ret": 0.0004, "cdi_ret": 0.0004})
    tgt = build_target(df)
    assert (tgt["y"].iloc[:-HORIZON] == 0).all()


def test_requires_sorted_dates():
    df = synthetic().iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        build_target(df)


def test_forward_compound_excludes_current_day():
    r = pd.Series([0.5] + [0.0] * 20)  # grande retorno só em t=0
    assert forward_compound(r, 10).iloc[0] == 0.0       # r[0] não entra em Y_0
    r2 = pd.Series([0.0, 0.5] + [0.0] * 20)
    assert forward_compound(r2, 10).iloc[0] == pytest.approx(0.5)


@pytest.fixture(scope="module")
def sample():
    c = build_clean_base(load_raw())
    return c.loc[c["in_original_sample"]].reset_index(drop=True)


def test_new_target_equals_trailing_label_ten_days_later(sample):
    tgt = build_target(sample)
    trailing = (sample["ida_trail_10"] < sample["cdi_trail_10"]).astype(float)
    valid = tgt["y"].notna()
    t = np.where(valid)[0]
    assert (tgt["y"].iloc[t].values == trailing.iloc[t + HORIZON].values).all()


def test_original_target_reproduction(sample):
    orig = original_target(sample, 100, HORIZON)
    assert len(orig) == len(sample) - 110 == 4071
    dd = (sample["ida_trail_10"] < sample["cdi_trail_10"]).astype(int).values
    for i in (0, 1000, 2442, 4070):
        assert orig["y_original"].iloc[i] == dd[i + 100:i + 110].max()
        assert orig["decision_idx"].iloc[i] == i + 99


def test_original_target_split_rates_match_phase1(sample):
    orig = original_target(sample, 100, HORIZON)
    y = orig["y_original"].values
    n = len(y)
    tr, va = int(n * .6), int(n * .8)
    assert round(y[:tr].mean(), 3) == 0.068
    assert round(y[tr:va].mean(), 3) == 0.248
    assert round(y[va:].mean(), 3) == 0.308
    assert y[va:].sum() == 251  # = positivos do teste de RF e LSTM na monografia
