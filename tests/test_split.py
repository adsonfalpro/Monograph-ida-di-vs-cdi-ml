import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("data", "targets", "features", "validation"):
    sys.path.insert(0, str(ROOT / "src" / sub))
from build_split import HORIZON, LOOKBACK, load_samples  # noqa: E402
from split import BLOCKS, TestSetGuard, blocked_folds, make_split  # noqa: E402


@pytest.fixture(scope="module")
def data():
    clean, tgt, start, rows, y = load_samples()
    return clean, start, rows, y, make_split(rows, HORIZON)


def test_sample_count_and_alignment(data):
    clean, start, rows, y, _ = data
    assert len(rows) == 4072
    assert rows[0] == start + LOOKBACK - 1
    assert rows[-1] == len(clean) - 1 - HORIZON       # última decisão com alvo completo
    assert (np.diff(rows) == 1).all()


def test_blocks_disjoint_ordered_and_cover(data):
    _, _, rows, _, sp = data
    allpos = np.concatenate([sp.blocks[b] for b in BLOCKS] + [sp.purged[b] for b in BLOCKS])
    assert len(allpos) == len(set(allpos)) == len(rows)
    for a, b in zip(BLOCKS[:-1], BLOCKS[1:]):
        assert sp.rows(a).max() < sp.rows(b).min()      # cronológico, sem embaralhar
    for b in BLOCKS:
        assert (np.diff(sp.blocks[b]) > 0).all()


def test_no_label_crosses_into_next_block(data):
    _, _, rows, _, sp = data
    for a, b in zip(BLOCKS[:-1], BLOCKS[1:]):
        t_next = sp.rows(b).min()
        assert (sp.target_end_rows[sp.blocks[a]] <= t_next).all()
        assert len(sp.purged[a]) == 9                    # horizon-1 amostras por fronteira
        assert (sp.target_end_rows[sp.purged[a]] > t_next).all()
    assert len(sp.purged["test"]) == 0


def test_proportions_close_to_60_20_20(data):
    _, _, rows, _, sp = data
    n = len(rows)
    sizes = {b: len(sp.blocks[b]) + len(sp.purged[b]) for b in BLOCKS}
    assert sizes["fit"] == int(n * .6) and sizes["test"] == n - int(n * .8)
    assert abs(sizes["val"] / n - 0.2) < 0.001


def test_test_block_reaches_the_end_and_has_no_future_labels(data):
    clean, _, rows, _, sp = data
    assert sp.rows("test").max() == rows.max()
    assert sp.target_end_rows[sp.blocks["test"]].max() == len(clean) - 1


def test_scaler_limit_is_last_fit_decision(data):
    _, _, _, _, sp = data
    assert sp.last_fit_row() == sp.rows("fit").max() < sp.rows("val").min()


def test_make_split_requires_sorted_rows():
    with pytest.raises(ValueError):
        make_split(np.array([5, 4, 6]), 10)


def test_blocked_folds_purge_and_coverage(data):
    _, _, rows, _, sp = data
    folds = blocked_folds(sp.blocks["val"], rows, HORIZON, k=3)
    assert len(folds) == 3
    held_all = np.concatenate([h for _, h in folds])
    assert sorted(held_all) == sorted(sp.blocks["val"])
    for train, held in folds:
        a, b = rows[held].min(), rows[held].max()
        assert not set(train) & set(held)
        t = rows[train]
        # nenhum período de alvo do treino intersecta o período de alvo do bloco retido
        assert not ((t + HORIZON >= a + 1) & (t + 1 <= b + HORIZON)).any()
        assert len(held) + len(train) < len(sp.blocks["val"])  # houve remoção


def test_test_guard_allows_one_release_per_model(tmp_path):
    g = TestSetGuard(tmp_path / "log.json")
    g.release("rf", {"threshold": 0.24})
    with pytest.raises(PermissionError):
        g.release("rf", {"threshold": 0.30})
    g.release("lstm", {"threshold": 0.3})
    assert len(pd.read_json(tmp_path / "log.json")) == 2
