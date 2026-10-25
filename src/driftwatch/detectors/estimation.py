"""Label-free performance estimation from calibrated confidence (CBPE).

A credit default is observed months after the decision, so the monitor needs
an estimate of today's performance before the labels exist. If the model is
calibrated, row ``i`` defaults with probability ``p_i``, which turns every
label-dependent metric into an expectation over the model's own
probabilities: the expected AUC (``stats.expected_auc``), the expected
default rate (the mean PD), the expected accuracy of the decision and the
expected bad rate among approved applicants.

The assumption is that P(y | x) has not changed. That holds under covariate
shift and fails under concept drift and prior shift, which is why the
benchmark reports the estimation error per drift type rather than one number.
For prior shift there is a known correction: the EM procedure of Saerens,
Latinne and Decaestecker (2002) re-estimates the class prior from unlabelled
data, assuming P(x | y) is fixed. It is reported as a second default-rate
estimate; which of the two is right depends on a shift type that cannot be
identified without labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from driftwatch.alerts import Category, Signal
from driftwatch.stats import expected_auc


@dataclass(frozen=True)
class PerformanceEstimate:
    """Expected metrics of one batch under the calibration assumption."""

    auc: float
    default_rate: float
    accuracy: float
    approval_rate: float
    bad_rate_approved: float


def estimate_performance(pd: np.ndarray, threshold: float) -> PerformanceEstimate:
    """CBPE estimates for one batch of calibrated default probabilities."""
    approved = pd < threshold
    expected_correct = np.where(approved, 1.0 - pd, pd)
    return PerformanceEstimate(
        auc=expected_auc(pd, pd),
        default_rate=float(pd.mean()),
        accuracy=float(expected_correct.mean()),
        approval_rate=float(approved.mean()),
        bad_rate_approved=float(pd[approved].mean()) if approved.any() else math.nan,
    )


def em_prior(pd: np.ndarray, train_prior: float, max_iter: int = 200, tol: float = 1e-8) -> float:
    """Saerens-Latinne-Decaestecker EM estimate of the positive-class prior of a new batch."""
    p = np.clip(pd, 1e-9, 1.0 - 1e-9)
    prior = float(np.mean(p))
    for _ in range(max_iter):
        ratio_pos = prior / train_prior
        ratio_neg = (1.0 - prior) / (1.0 - train_prior)
        adjusted = ratio_pos * p / (ratio_pos * p + ratio_neg * (1.0 - p))
        updated = float(adjusted.mean())
        if abs(updated - prior) < tol:
            return updated
        prior = updated
    return prior


@dataclass
class CBPEMonitor:
    """Alarm when the estimated AUC leaves the band seen on clean reference batches."""

    ref_mean: float
    ref_sd: float
    ref_batch_size: int
    threshold: float
    n_sigma: float = 3.0

    @classmethod
    def fit(
        cls, reference_pd: np.ndarray, threshold: float, batch_size: int, *, n_boot: int = 300, seed: int = 0
    ) -> CBPEMonitor:
        """Bootstrap clean batches of ``batch_size`` rows to get the estimate's natural spread."""
        rng = np.random.default_rng(seed)
        draws = [
            expected_auc(s, s)
            for s in (reference_pd[rng.integers(0, len(reference_pd), batch_size)] for _ in range(n_boot))
        ]
        return cls(float(np.mean(draws)), float(np.std(draws, ddof=1)), batch_size, threshold)

    def lower_bound(self, n: int) -> float:
        """The band edge for a batch of ``n`` rows; the spread scales as one over root n."""
        return self.ref_mean - self.n_sigma * self.ref_sd * math.sqrt(self.ref_batch_size / max(n, 1))

    def evaluate(self, pd: np.ndarray) -> tuple[PerformanceEstimate, Signal]:
        estimate = estimate_performance(pd, self.threshold)
        bound = self.lower_bound(len(pd))
        fired = bool(estimate.auc < bound)
        message = f"estimated AUC {estimate.auc:.3f} below the reference band edge {bound:.3f}"
        signal = Signal("cbpe", Category.ESTIMATED_PERFORMANCE, "model", fired, estimate.auc, bound, None, message)
        return estimate, signal
