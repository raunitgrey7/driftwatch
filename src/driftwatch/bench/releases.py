"""Planted releases for the deployment-safety benchmark.

Each bad release reproduces a defect that reaches production in practice;
each good release is what a routine retrain produces. Defects in the
serving path wrap an otherwise healthy fresh retrain, so the only thing
wrong with the release is the defect:

* ``inverted_score``: the positive-class column is swapped, the score ranks
  applicants backwards;
* ``unit_bug``: the serving code converts income to thousands before scoring;
* ``miscalibrated``: the calibration step is dropped and the PD comes out
  2.5 times too high, so the cut-off rejects far more applicants;
* ``crash_5pct``: a parsing bug returns NaN for one row in twenty;
* ``no_bureau``: the bureau feed (score, delinquencies, utilisation) is
  missing at training time; calibrated and with the usual approval rate, so
  only outcome labels can show it ranks worse;
* ``small_sample``: trained on 1,200 rows, the same failure in a milder form.

Good releases: ``retrain_same`` (same recipe, fresh data) and
``retrain_more`` (twice the training data).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from driftwatch.deploy.rollout import bucket
from driftwatch.loop import ModelBundle
from driftwatch.model import CreditModel, Scorer, TrainConfig
from driftwatch.schema import Batch
from driftwatch.training import HistoryConfig, build_champion

BAD = ("inverted_score", "unit_bug", "miscalibrated", "crash_5pct", "no_bureau", "small_sample")
GOOD = ("retrain_same", "retrain_more")
BUREAU = ("bureau_score", "delinquencies_2y", "credit_utilization")


@dataclass
class InvertedScore:
    base: CreditModel

    @property
    def threshold(self) -> float:
        return self.base.threshold

    def predict_pd(self, batch: Batch) -> np.ndarray:
        return 1.0 - self.base.predict_pd(batch)


@dataclass
class IncomeInThousands:
    base: CreditModel

    @property
    def threshold(self) -> float:
        return self.base.threshold

    def predict_pd(self, batch: Batch) -> np.ndarray:
        scaled = Batch(batch.ids, dict(batch.numeric), batch.categorical, dict(batch._codes))
        scaled.numeric["monthly_income"] = batch.numeric["monthly_income"] / 1000.0
        return self.base.predict_pd(scaled)


@dataclass
class Uncalibrated:
    base: CreditModel
    factor: float = 2.5

    @property
    def threshold(self) -> float:
        return self.base.threshold

    def predict_pd(self, batch: Batch) -> np.ndarray:
        return np.minimum(self.base.predict_pd(batch) * self.factor, 0.999)


@dataclass
class Crashing:
    base: CreditModel
    fail_rate: float = 0.05

    @property
    def threshold(self) -> float:
        return self.base.threshold

    def predict_pd(self, batch: Batch) -> np.ndarray:
        pd = self.base.predict_pd(batch).copy()
        pd[bucket(batch.ids, "parse-bug") < self.fail_rate] = np.nan
        return pd


def _history(seed: int, train_batches: int = 40) -> HistoryConfig:
    return HistoryConfig(seed=seed, n_batches=train_batches + 20, train_batches=train_batches, calib_batches=10)


WRAPPERS: dict[str, Callable[[CreditModel], Scorer]] = {
    "inverted_score": InvertedScore,
    "unit_bug": IncomeInThousands,
    "miscalibrated": Uncalibrated,
    "crash_5pct": Crashing,
}


def build_release(kind: str, seed: int) -> ModelBundle:
    """Train one planted release on fresh data selected by ``seed`` and wrap it in its defect."""
    if kind not in (*BAD, *GOOD):
        raise ValueError(f"unknown release kind {kind!r}")
    if kind == "retrain_more":
        champion = build_champion(_history(seed, train_batches=80))
    elif kind == "no_bureau":
        champion = build_champion(_history(seed), train_config=TrainConfig(dropped=BUREAU))
    elif kind == "small_sample":
        champion = build_champion(HistoryConfig(seed=seed, n_batches=8, train_batches=3, calib_batches=2))
    else:
        champion = build_champion(_history(seed))
    wrap = WRAPPERS.get(kind)
    scorer: Scorer = wrap(champion.model) if wrap else champion.model
    return ModelBundle(scorer, champion.reference)
