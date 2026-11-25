import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data", "targets", "features"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from clean_base import build_clean_base, load_raw  # noqa: E402
from engineering import (  # noqa: E402
    FEATURE_COLUMNS, aggregate_windows, aggregated_names, build_features, first_valid_row, fit_scaler,
    make_windows, rsi, transform,
)


@pytest.fixture(scope="module")
def clean():
    return build_clean_base(load_raw())


@pytest.fixture(scope="module")
def feats(clean):
    return build_features(clean)


def test_twelve_features_no_nan_in_original_sample(clean, feats):
    assert len(FEATURE_COLUMNS) == 12
    start = int(np.where(clean["in_original_sample"])[0][0])
    assert first_valid_row(feats) <= start
    assert not feats.loc[start:, FEATURE_COLUMNS].isna().any().any()


def test_causality_perturbing_future_does_not_change_past(clean, feats):
    t = 2500
    raw = clean.copy()
    for col in ("ida_ret", "cdi_ret", "ibov_ret", "ipca_deb_ret", "duration"):
        raw.loc[t + 1:, col] = raw.loc[t + 1:, col] * 1.5 + 0.001
    f2 = build_features(raw)
    pd.testing.assert_frame_equal(feats.loc[:t, FEATURE_COLUMNS], f2.loc[:t, FEATURE_COLUMNS])
    assert not feats.loc[t + 1:t + 40, FEATURE_COLUMNS].equals(f2.loc[t + 1:t + 40, FEATURE_COLUMNS])


def test_rolling_features_match_manual_computation(clean, feats):
    t = 1234
    exc = (clean["ida_ret"] - clean["cdi_ret"]).iloc[t - 19:t + 1]
    assert feats.loc[t, "vol_20"] == pytest.approx(exc.std(ddof=1))
    assert feats.loc[t, "ret_20"] == pytest.approx(exc.sum())
    assert feats.loc[t, "trend_20"] == pytest.approx(clean["ida_ret"].iloc[t - 19:t + 1].mean())


def test_macd_uses_only_past():
    s = pd.Series(np.random.default_rng(0).normal(0.0004, 0.001, 200))
    clean = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=200), "ida_ret": s, "duration": 2.0,
                          "ipca_deb_ret": 0.0, "ibov_ret": 0.0, "cdi_ret": 0.0004})
    a = build_features(clean)
    clean2 = clean.copy()
    clean2.loc[150:, "ida_ret"] += 0.01
    b = build_features(clean2)
    pd.testing.assert_series_equal(a["macd"].iloc[:150], b["macd"].iloc[:150])


def test_rsi_bounds_and_no_loss_case():
    up = pd.Series(np.arange(40) * 0.001)  # variação diária sempre positiva
    r = rsi(up, 14)
    assert (r.dropna() == 100.0).all()
    flat = pd.Series(np.zeros(40))
    assert (rsi(flat, 14).dropna() == 50.0).all()
    rnd = rsi(pd.Series(np.random.default_rng(1).normal(size=300)), 14).dropna()
    assert rnd.between(0, 100).all()


def test_scaler_uses_only_training_rows(feats):
    end = 2540
    sc1 = fit_scaler(feats, end)
    f2 = feats.copy()
    f2.loc[end + 1:, FEATURE_COLUMNS] = f2.loc[end + 1:, FEATURE_COLUMNS] * 10 + 5
    sc2 = fit_scaler(f2, end)
    assert np.array_equal(sc1.mean_, sc2.mean_) and np.array_equal(sc1.scale_, sc2.scale_)
    z = transform(feats, sc1)[first_valid_row(feats):end + 1]
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9) and np.allclose(z.std(axis=0), 1, atol=1e-9)


def test_scaler_rejects_nan_rows(feats):
    with pytest.raises(ValueError):
        fit_scaler(feats, 2540, start_row=0)  # linhas de aquecimento têm NaN


def test_windows_end_at_decision_row(feats):
    x = feats[FEATURE_COLUMNS].values
    w = make_windows(x, np.array([200, 201]), 100)
    assert w.shape == (2, 100, 12)
    assert np.array_equal(w[0][-1], x[200]) and np.array_equal(w[0][0], x[101])
    assert np.array_equal(w[1][-1], x[201])
    with pytest.raises(ValueError):
        make_windows(x, np.array([50]), 100)


def test_window_does_not_depend_on_rows_after_decision(feats):
    x = feats[FEATURE_COLUMNS].values.copy()
    w1 = make_windows(x, np.array([300]), 100)
    x[301:] = 999.0
    w2 = make_windows(x, np.array([300]), 100)
    assert np.array_equal(w1, w2)


def test_aggregation_shape_names_and_constant_window():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(5, 100, 12))
    w[0, :, 3] = 0.7  # coluna constante na janela 0
    agg, undefined = aggregate_windows(w)
    assert agg.shape == (5, 120) and len(aggregated_names()) == 120
    assert np.isfinite(agg).all()
    assert undefined == 2  # assimetria e curtose da coluna constante
    i = aggregated_names().index("ida_ret__mean")
    assert agg[1, i] == pytest.approx(w[1, :, 0].mean())
    j = aggregated_names().index("ida_ret__last")
    assert agg[1, j] == w[1, -1, 0]
