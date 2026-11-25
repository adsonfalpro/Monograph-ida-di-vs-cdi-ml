"""Random Forest (parâmetros do Apêndice A / LS.py). A calibração e o threshold são da Fase 7."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = dict(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)


def make_rf(**overrides) -> RandomForestClassifier:
    return RandomForestClassifier(**{**RF_PARAMS, **overrides})
