"""Conjunto de dados para os três modelos (Fase 6).

Une base limpa, alvo (Fase 3), variáveis (Fase 4) e divisão (Fase 5). O scaler é ajustado só com linhas até a última
decisão do bloco ``fit``. Os blocos ``fit`` e ``val`` ficam disponíveis para desenvolvimento; o bloco ``test`` só é
entregue por ``Dataset.release_test`` (que passa pelo ``TestSetGuard``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _sub in ("data", "targets", "features", "validation"):
    sys.path.insert(0, str(ROOT / "src" / _sub))
from build_split import LOOKBACK, load_samples  # noqa: E402
from build_target import HORIZON  # noqa: E402
from engineering import (  # noqa: E402
    FEATURE_COLUMNS, aggregate_windows, build_features, fit_scaler, make_windows, transform,
)
from split import TemporalSplit, TestSetGuard, make_split  # noqa: E402


@dataclass
class Dataset:
    split: TemporalSplit
    y_all: np.ndarray
    x_seq_all: np.ndarray       # (n, 100, 12) padronizado
    x_agg_all: np.ndarray       # (n, 120)
    scaler_rows: tuple[int, int]
    undefined_stats: int
    excess_all: np.ndarray = None   # excesso geométrico dos próximos h dias (magnitude do alvo)

    def _pos(self, block: str) -> np.ndarray:
        return self.split.blocks[block]

    def seq(self, block: str) -> np.ndarray:
        self._check(block)
        return self.x_seq_all[self._pos(block)]

    def agg(self, block: str) -> np.ndarray:
        self._check(block)
        return self.x_agg_all[self._pos(block)]

    def y(self, block: str) -> np.ndarray:
        self._check(block)
        return self.y_all[self._pos(block)]

    def excess(self, block: str) -> np.ndarray:
        self._check(block)
        return self.excess_all[self._pos(block)]

    @staticmethod
    def _check(block: str) -> None:
        if block == "test":
            raise PermissionError("O bloco de teste só é liberado por Dataset.release_test(guard, modelo, config).")
        if block not in ("fit", "val"):
            raise KeyError(block)

    def release_test(self, guard: TestSetGuard, model_name: str, frozen_config: dict):
        guard.release(model_name, frozen_config)
        pos = self._pos("test")
        return self.x_seq_all[pos], self.x_agg_all[pos], self.y_all[pos], self.excess_all[pos]


def build_dataset() -> Dataset:
    clean, tgt, start, rows, y = load_samples()
    feats = build_features(clean)
    split = make_split(rows, HORIZON)
    last_fit = split.last_fit_row()
    scaler = fit_scaler(feats, last_fit)                    # só linhas <= última decisão do fit
    x = transform(feats, scaler)
    seq = make_windows(x, rows, LOOKBACK).astype(np.float32)
    agg, undefined = aggregate_windows(seq.astype(np.float64))
    first_row = int(feats[FEATURE_COLUMNS].notna().all(axis=1).values.argmax())
    return Dataset(split, y, seq, agg, (first_row, last_fit), undefined, tgt['fwd_excess'].iloc[rows].values)
