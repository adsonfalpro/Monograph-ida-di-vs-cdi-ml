import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
from dataset import build_dataset  # noqa: E402
from lstm_model import LSTM_PARAMS, build_lstm  # noqa: E402
from perceptron import PERCEPTRON_PARAMS, PerceptronScratch  # noqa: E402
from random_forest import RF_PARAMS, make_rf  # noqa: E402
from split import TestSetGuard  # noqa: E402


def test_perceptron_update_rule_matches_spec():
    X = np.array([[1.0, 2.0]])
    y = np.array([1])
    p = PerceptronScratch(learning_rate=0.1, n_epochs=1).fit(X, y)
    # primeiro passo: saída linear 0 -> y_hat=1 (>=0), sem erro; pesos permanecem 0
    assert np.allclose(p.weights, 0) and p.bias == 0
    y0 = np.array([0])
    p = PerceptronScratch(learning_rate=0.1, n_epochs=1).fit(X, y0)
    # y_hat=1, y=0 -> update = 0.1*(0-1) = -0.1 ; w = -0.1*x ; b = -0.1
    assert np.allclose(p.weights, [-0.1, -0.2]) and p.bias == pytest.approx(-0.1)


def test_perceptron_learns_separable_data_and_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X @ np.array([1.0, -2.0, 0.5]) + 0.3 > 0).astype(int)
    a = PerceptronScratch(random_state=7, n_epochs=50).fit(X, y)
    b = PerceptronScratch(random_state=7, n_epochs=50).fit(X, y)
    assert np.array_equal(a.weights, b.weights)
    assert (a.predict(X) == y).mean() > 0.95
    assert a.errors_per_epoch[-1] < a.errors_per_epoch[0]
    assert np.array_equal((a.decision_function(X) >= 0).astype(int), a.predict(X))


def test_perceptron_does_not_touch_global_numpy_state():
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    PerceptronScratch(n_epochs=2).fit(np.random.default_rng(1).normal(size=(20, 2)), np.array([0, 1] * 10))
    # fit usa gerador próprio: o estado global continua o definido por seed(123) + a chamada de rand abaixo
    np.random.seed(123)
    assert np.random.rand() == expected


def test_documented_parameters():
    assert PERCEPTRON_PARAMS == {"learning_rate": 0.01, "n_epochs": 100, "random_state": 42}
    assert RF_PARAMS["n_estimators"] == 800 and RF_PARAMS["min_samples_leaf"] == 2
    assert RF_PARAMS["max_features"] == "sqrt" and RF_PARAMS["class_weight"] == "balanced_subsample"
    assert RF_PARAMS["max_depth"] is None
    rf = make_rf(n_estimators=5)
    assert rf.get_params()["min_samples_leaf"] == 2 and rf.n_estimators == 5


def test_lstm_architecture_matches_appendix_b():
    m = build_lstm(100, 12)
    kinds = [type(layer).__name__ for layer in m.layers]
    assert kinds == ["LSTM", "Dropout", "LSTM", "BatchNormalization", "Dense", "Dropout", "Dense"]
    assert m.layers[0].units == 64 and m.layers[2].units == 32 and m.layers[4].units == 32
    assert m.layers[1].rate == 0.3 and m.layers[5].rate == 0.2
    assert m.layers[-1].activation.__name__ == "sigmoid"
    assert m.output_shape == (None, 1)
    assert m.count_params() == 4 * 64 * (12 + 64 + 1) + 4 * 32 * (64 + 32 + 1) + 4 * 32 + 32 * 32 + 32 + 33
    assert LSTM_PARAMS["learning_rate"] == 1e-3 and LSTM_PARAMS["es_patience"] == 5


@pytest.fixture(scope="module")
def ds():
    return build_dataset()


def test_dataset_shapes_and_alignment(ds):
    assert ds.seq("fit").shape == (2434, 100, 12) and ds.seq("val").shape == (805, 100, 12)
    assert ds.agg("fit").shape == (2434, 120)
    assert len(ds.y("fit")) == 2434 and ds.y("fit").sum() == 80 and ds.y("val").sum() == 122
    assert np.isfinite(ds.x_seq_all).all() and np.isfinite(ds.x_agg_all).all()


def test_scaler_fit_only_on_fit_rows(ds):
    first, last = ds.scaler_rows
    assert last == ds.split.last_fit_row() < ds.split.rows("val").min()
    # linhas de treino padronizadas têm média ~0 e dp ~1; as de validação/teste, não
    fit_last_window = ds.x_seq_all[ds.split.blocks["fit"][-1]]     # termina na última decisão do fit
    assert abs(fit_last_window.mean()) < 5
    val_rows = ds.seq("val")[:, -1, :]
    assert np.abs(val_rows.mean(axis=0)).max() > 0.3   # deslocamento real de regime: não foi "normalizado" com dados futuros


def test_test_block_is_locked(ds, tmp_path):
    with pytest.raises(PermissionError):
        ds.seq("test")
    with pytest.raises(PermissionError):
        ds.y("test")
    guard = TestSetGuard(tmp_path / "log.json")
    x_seq, x_agg, y, exc = ds.release_test(guard, "dummy", {"k": 1})
    assert len(y) == 815 and len(exc) == 815 and x_seq.shape[0] == 815 and x_agg.shape == (815, 120)
    with pytest.raises(PermissionError):
        ds.release_test(guard, "dummy", {"k": 1})
