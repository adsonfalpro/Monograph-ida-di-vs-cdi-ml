"""Divisão temporal com purge e proteção do conjunto de teste (Fase 5).

Unidade de análise: uma amostra por data de decisão ``t`` (linha de ``clean``), com
  features = linhas t-99..t   |   alvo = retornos das linhas t+1..t+h   (``target_end_row = t+h``).

Blocos cronológicos (proporção da monografia 60/20/20):

    fit (60 %)  |  val (20 %)  |  test (20 %)

* ``val``: calibração probabilística, parada antecipada do LSTM e escolha do threshold. Como ``val`` tem poucos eventos
  (e concentrados em 2019-20), calibração e threshold NÃO são separados em dois blocos contíguos (testado: sobrariam 11
  eventos para o threshold). A separação é feita na Fase 7 por validação cruzada em blocos (``blocked_folds``), em que
  cada observação de ``val`` é calibrada por um calibrador que não a viu.
* Purge: dentro de cada bloco que é seguido por outro, removem-se as amostras cujo alvo usa retornos de uma data
  POSTERIOR à primeira data de decisão do bloco seguinte (``target_end_row > t_next_first``). Sem isso o bloco
  anterior "veria" retornos do bloco seguinte através dos seus rótulos.
* Nada é embaralhado. O teste só é liberado por ``TestSetGuard`` (uma liberação por modelo, registrada em arquivo).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BLOCKS = ("fit", "val", "test")


@dataclass
class TemporalSplit:
    decision_rows: np.ndarray          # todas as amostras (ordenadas)
    target_end_rows: np.ndarray
    blocks: dict[str, np.ndarray]      # posições (0..n-1) em ``decision_rows``, já purgadas
    purged: dict[str, np.ndarray]      # posições removidas por purge, por bloco
    nominal: dict[str, tuple[int, int]] = field(default_factory=dict)  # [início, fim) antes do purge

    def rows(self, block: str) -> np.ndarray:
        return self.decision_rows[self.blocks[block]]

    def last_fit_row(self) -> int:
        """Última data de decisão do treino (limite superior das linhas usadas para ajustar o scaler)."""
        return int(self.rows("fit").max())


def make_split(decision_rows: np.ndarray, horizon: int, fractions=(0.6, 0.2, 0.2)) -> TemporalSplit:
    decision_rows = np.asarray(decision_rows)
    if not (np.diff(decision_rows) > 0).all():
        raise ValueError("decision_rows deve ser estritamente crescente")
    n = len(decision_rows)
    target_end = decision_rows + horizon
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    nominal = {"fit": (0, a), "val": (a, b), "test": (b, n)}
    blocks, purged = {}, {}
    order = list(BLOCKS)
    for i, name in enumerate(order):
        lo, hi = nominal[name]
        idx = np.arange(lo, hi)
        if i + 1 < len(order):
            t_next_first = decision_rows[nominal[order[i + 1]][0]]
            bad = target_end[idx] > t_next_first
            purged[name] = idx[bad]
            idx = idx[~bad]
        else:
            purged[name] = np.array([], dtype=int)
        blocks[name] = idx
    return TemporalSplit(decision_rows, target_end, blocks, purged, nominal)


def blocked_folds(positions: np.ndarray, decision_rows: np.ndarray, horizon: int, k: int = 3):
    """Validação cruzada em blocos contíguos dentro de ``positions`` (posições em ``decision_rows``).

    Devolve [(treino, retido), ...]. Do treino de cada dobra são removidas as amostras cujo período de alvo
    (t+1..t+h) intersecta o período de alvo do bloco retido (t_ret_ini+1 .. t_ret_fim+h): 9 antes e 9 depois do bloco.
    Obs.: dobras posteriores entram no treino das anteriores (não é avaliação "walk-forward"); serve para
    calibrar/limiar sem que a observação seja vista pelo calibrador que a corrige.
    """
    positions = np.asarray(positions)
    chunks = np.array_split(positions, k)
    folds = []
    for j, held in enumerate(chunks):
        a, b = decision_rows[held].min(), decision_rows[held].max()
        others = np.concatenate([c for i, c in enumerate(chunks) if i != j])
        t = decision_rows[others]
        bad = (t + horizon >= a + 1) & (t + 1 <= b + horizon)
        folds.append((others[~bad], held))
    return folds


def split_table(split: TemporalSplit, dates: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """Uma linha por amostra com bloco (ou 'purge')."""
    label = np.full(len(split.decision_rows), "", dtype=object)
    for name in BLOCKS:
        label[split.blocks[name]] = name
        label[split.purged[name]] = "purge"
    return pd.DataFrame({
        "sample_id": np.arange(len(split.decision_rows)),
        "decision_row": split.decision_rows,
        "date": dates.iloc[split.decision_rows].values,
        "target_end_row": split.target_end_rows,
        "target_end_date": dates.iloc[split.target_end_rows].values,
        "y": y, "block": label,
    })


class TestSetGuard:
    __test__ = False  # evita coleta pelo pytest
    """Libera o teste uma única vez por modelo e registra o acesso (append-only)."""

    def __init__(self, log_path: Path | None = None):
        self.log_path = ROOT / "outputs" / "metrics" / "test_access_log.json" if log_path is None else Path(log_path)

    def _read(self) -> list[dict]:
        return json.loads(self.log_path.read_text(encoding="utf-8")) if self.log_path.exists() else []

    def release(self, model_name: str, frozen_config: dict) -> None:
        log = self._read()
        if any(e["model"] == model_name for e in log):
            raise PermissionError(f"O teste já foi usado para '{model_name}'. Nova avaliação exige novo registro explícito.")
        log.append({"model": model_name, "frozen_config": frozen_config,
                    "released_at": pd.Timestamp.now().isoformat(timespec="seconds")})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
