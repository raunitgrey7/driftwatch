"""The credit model: histogram gradient boosting with Platt calibration.

Gradient boosting is the usual production choice for tabular credit risk
next to logistic scorecards. It handles missing values natively, which
matters here because data-quality incidents are part of the stream: a null
bureau score is routed down a learned branch instead of being imputed.
Categories enter as vocabulary codes (unknown values become missing): with
six or fewer levels the trees isolate them with ordinary splits, the
measured AUC matches native categorical support, and prediction skips the
library's per-call encoder, which was a third of the monitor's batch cost. Platt
scaling on a held-out calibration slice makes the probabilities calibrated on
the reference period, which the confidence-based performance estimator
relies on. The approval cut-off is fixed at training time as the PD quantile
that approves ``approval_rate`` of the calibration population, so every
version carries its own credit policy and a release cannot silently change it.

Every scorer used by the monitor and the rollout controller satisfies the
``Scorer`` protocol, which is also how the benchmark plants faulty releases.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier

from driftwatch.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, Batch

PD_FLOOR = 1e-6


class Scorer(Protocol):
    """Anything that turns a batch into default probabilities and decisions."""

    threshold: float

    def predict_pd(self, batch: Batch) -> np.ndarray: ...


def approve(pd: np.ndarray, threshold: float) -> np.ndarray:
    """The credit decision: approve when the probability of default is below the cut-off."""
    return pd < threshold


MODEL_FEATURES: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)


def encode(batch: Batch, dropped: tuple[str, ...] = ()) -> np.ndarray:
    """Feature matrix in schema order without ``dropped``; categories become codes, unknown or null NaN."""
    columns = []
    for name in MODEL_FEATURES:
        if name in dropped:
            continue
        if name in batch.numeric:
            columns.append(batch.numeric[name])
        else:
            codes = batch.codes(name)
            columns.append(np.where(codes >= 0, codes, np.nan))
    return np.column_stack(columns).astype(np.float64)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PD_FLOOR, 1.0 - PD_FLOOR)
    return np.log(p / (1.0 - p))


def fit_platt(raw: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit ``sigmoid(a * logit(raw) + b)`` by maximum likelihood (two parameters, no ties)."""
    x = _logit(raw)
    y = labels.astype(np.float64)

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        z = theta[0] * x + theta[1]
        nll = np.sum(np.logaddexp(0.0, z) - y * z)
        residual = 1.0 / (1.0 + np.exp(-z)) - y
        return float(nll), np.array([np.sum(residual * x), np.sum(residual)])

    result = minimize(loss, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    return float(result.x[0]), float(result.x[1])


@dataclass
class CreditModel:
    """A trained, calibrated model with its decision cut-off and training metadata."""

    booster: HistGradientBoostingClassifier
    platt: tuple[float, float]
    threshold: float
    dropped: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)

    def raw_pd(self, batch: Batch) -> np.ndarray:
        return self.booster.predict_proba(encode(batch, self.dropped))[:, 1]

    def predict_pd(self, batch: Batch) -> np.ndarray:
        a, b = self.platt
        return 1.0 / (1.0 + np.exp(-(a * _logit(self.raw_pd(batch)) + b)))


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters, fixed rather than tuned so retrained versions are comparable."""

    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    approval_rate: float = 0.80
    dropped: tuple[str, ...] = ()


def train_credit_model(
    train: Batch,
    y_train: np.ndarray,
    calib: Batch,
    y_calib: np.ndarray,
    *,
    seed: int = 0,
    config: TrainConfig | None = None,
) -> CreditModel:
    """Fit the booster on ``train``, calibrate and set the cut-off on ``calib``."""
    cfg = config or TrainConfig()
    started = time.perf_counter()
    booster = HistGradientBoostingClassifier(
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )
    booster.fit(encode(train, cfg.dropped), y_train)
    model = CreditModel(booster=booster, platt=(1.0, 0.0), threshold=1.0, dropped=cfg.dropped)
    model.platt = fit_platt(model.raw_pd(calib), y_calib)
    calibrated = model.predict_pd(calib)
    model.threshold = float(np.quantile(calibrated, cfg.approval_rate))
    model.info = {
        "seed": seed,
        "n_train": int(train.n_rows),
        "n_calib": int(calib.n_rows),
        "train_default_rate": round(float(np.mean(y_train)), 6),
        "iterations": int(booster.n_iter_),
        "approval_rate_target": cfg.approval_rate,
        "dropped_features": list(cfg.dropped),
        "train_seconds": round(time.perf_counter() - started, 3),
    }
    return model
