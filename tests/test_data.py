import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
from clean_base import build_clean_base, load_monograph, load_raw  # noqa: E402


@pytest.fixture(scope="module")
def clean():
    return build_clean_base(load_raw())


@pytest.fixture(scope="module")
def mono():
    return load_monograph()


def test_structure(clean):
    assert len(clean) == 4240
    assert clean["date"].is_monotonic_increasing and not clean["date"].duplicated().any()
    assert (clean["date"].dt.dayofweek < 5).all()


def test_original_sample_matches_monograph_dates(clean, mono):
    dates = clean.loc[clean["in_original_sample"], "date"].reset_index(drop=True)
    assert len(dates) == 4181
    assert (dates == mono["date"]).all()


def test_no_missing_in_original_sample(clean):
    cols = [c for c in clean.columns if not c.startswith(("ida_trail", "cdi_trail", "excess_trail", "ida_index_ret"))]
    sub = clean.loc[clean["in_original_sample"], cols]
    assert not sub.isna().any().any()


def test_ibov_text_rows_parsed(clean):
    assert clean["ibov_from_text"].sum() == 18
    assert clean["ibov_ret"].dtype == float
    row = clean.loc[clean["date"] == "2009-03-05", "ibov_ret"].iloc[0]
    assert row == pytest.approx(-0.0269)


def test_units_are_decimal(clean):
    assert clean["cdi_ret"].between(5e-5, 1e-3).all()
    assert clean["ida_ret"].abs().max() < 0.01
    assert clean["ipca_deb_ret"].abs().max() < 0.05  # estava em % (até 3,4)


def test_shared_columns_identical_to_monograph(clean, mono):
    m = clean.merge(mono, on="date", suffixes=("", "_m"))
    for a, b in [("ida_index", "ida_index_m"), ("ida_ret", "ida_ret_m"), ("duration", "duration_m"),
                 ("ibov_ret", "ibov_ret_m"), ("cdi_ret", "cdi_ret_m")]:
        assert (m[a] - m[b]).abs().max() == 0
    assert (m["ipca_deb_ret"] - m["ipca_deb_pct"] / 100).abs().max() < 1e-15


@pytest.mark.parametrize("h", [10, 30, 60])
def test_trailing_windows_match_monograph(clean, mono, h):
    m = clean.merge(mono, on="date", suffixes=("", "_m"))
    assert (m[f"ida_trail_{h}"] - m[f"ida_{h}"]).abs().max() < 1e-12
    assert (m[f"cdi_trail_{h}"] - m[f"cdi_{h}"]).abs().max() < 1e-12
    assert (m[f"excess_trail_{h}"] - m[f"excess_{h}"]).abs().max() < 1e-12


def test_monograph_cdi20_is_a_19_day_window(clean, mono):
    m = clean.merge(mono, on="date", suffixes=("", "_m"))
    assert (m["cdi_trail_20"] - m["cdi_20"]).abs().max() > 1e-4  # erro da monografia
    cdi19 = (1 + clean["cdi_ret"]).rolling(19).apply(np.prod, raw=True) - 1
    mm = clean[["date"]].assign(c19=cdi19).merge(mono[["date", "cdi_20"]], on="date")
    assert (mm["c19"] - mm["cdi_20"]).abs().max() < 1e-12


def test_trailing_window_is_causal(clean):
    """Alterar o retorno em t+1 não pode mudar nenhum acumulado em t."""
    raw = load_raw()
    t = 2000
    raw2 = raw.copy()
    raw2.loc[t + 1, "ida_ret"] = raw2.loc[t + 1, "ida_ret"] + 0.01
    c2 = build_clean_base(raw2)
    cols = [c for c in clean.columns if c.startswith(("ida_trail", "excess_trail"))]
    assert clean.loc[:t, cols].equals(c2.loc[:t, cols])
    assert not clean.loc[t + 1, cols].equals(c2.loc[t + 1, cols])


def test_sign_of_target_invariant_to_excess_form(clean):
    ok = clean["ida_trail_10"].notna()
    simple = clean["ida_trail_10"] - clean["cdi_trail_10"] < 0
    geo = clean["excess_trail_10"] < 0
    assert (simple[ok] == geo[ok]).all()


def test_index_mismatch_days(clean):
    days = clean.loc[clean["ida_ret_index_mismatch"], "date"].dt.strftime("%Y-%m-%d").tolist()
    assert days == ["2020-11-30", "2022-05-30", "2025-04-01", "2025-09-16"]
