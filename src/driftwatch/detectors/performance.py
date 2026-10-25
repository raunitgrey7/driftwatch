"""Performance-drift detectors that need labels: ADWIN, DDM and a label-rate test.

Labels arrive ``label_delay`` batches after the decision, so these detectors
are slow by construction; their job is to confirm what the label-free side
suspected and to catch what it cannot see (concept drift).

* ``BatchedADWIN`` is ADWIN (Bifet and Gavalda 2007) with cut points at
  batch boundaries. The window keeps per-batch sums, so checking every split
  costs O(batches) instead of O(instances); the cut rule is the
  variance-aware bound used by the reference implementations. It watches
  the calibration residual ``y - p`` (mapped to [0, 1]), whose mean is zero
  for a calibrated model whatever the applicant mix, so a population change
  alone does not trip it, while concept drift and prior shift do.
* ``DDM`` (Gama et al. 2004) watches the error rate of the credit decision
  with the classic warning (2 sigma) and drift (3 sigma) levels over the
  running minimum, evaluated instance by instance in vectorised form. The
  minimum is only tracked after a burn-in of 4,000 instances (ten batches),
  because a minimum taken over the first few hundred rows is mostly noise
  and was the source of every early false alarm in the first benchmark run.
* ``LabelRate`` tests each labelled batch's default rate against the
  reference rate with a two-sided binomial test.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from driftwatch.alerts import Category, Signal
from driftwatch.stats import binomial_two_sided_p


@dataclass(frozen=True)
class AdwinResult:
    drift: bool
    mean_before: float
    mean_after: float
    width: int


@dataclass
class BatchedADWIN:
    """Adaptive window over batch summaries; a cut means the mean changed."""

    delta: float = 0.002
    max_batches: int = 300
    min_instances: int = 1000
    _window: deque[tuple[int, float, float]] = field(default_factory=deque)

    def _find_cut(self) -> tuple[int, float, float] | None:
        counts = np.array([w[0] for w in self._window], dtype=np.float64)
        sums = np.array([w[1] for w in self._window])
        squares = np.array([w[2] for w in self._window])
        n_total, s_total = counts.sum(), sums.sum()
        variance = max(squares.sum() / n_total - (s_total / n_total) ** 2, 0.0)
        log_term = math.log(2.0 * math.log(n_total) / self.delta)
        n0, s0 = np.cumsum(counts)[:-1], np.cumsum(sums)[:-1]
        n1, s1 = n_total - n0, s_total - s0
        ok = (n0 >= self.min_instances) & (n1 >= self.min_instances)
        if not ok.any():
            return None
        harmonic = 1.0 / n0 + 1.0 / n1
        epsilon = np.sqrt(2.0 * harmonic * variance * log_term) + (2.0 / 3.0) * log_term * harmonic
        gap = np.abs(s0 / n0 - s1 / n1)
        margin = np.where(ok, gap - epsilon, -np.inf)
        k = int(np.argmax(margin))
        if margin[k] <= 0:
            return None
        return k, float(s0[k] / n0[k]), float(s1[k] / n1[k])

    def update(self, values: np.ndarray) -> AdwinResult:
        """Add one batch of values in [0, 1]; drop the oldest batch while any split is significant."""
        values = np.asarray(values, dtype=np.float64)
        self._window.append((len(values), float(values.sum()), float(np.square(values).sum())))
        while len(self._window) > self.max_batches:
            self._window.popleft()
        drift, before, after = False, math.nan, math.nan
        while len(self._window) > 1 and (cut := self._find_cut()) is not None:
            if not drift:
                _, before, after = cut
            drift = True
            self._window.popleft()
        return AdwinResult(drift, before, after, sum(w[0] for w in self._window))


@dataclass(frozen=True)
class DDMResult:
    drift: bool
    warning: bool
    error_rate: float


@dataclass
class DDM:
    """Drift Detection Method on a 0/1 error stream, processed a batch at a time."""

    warning_level: float = 2.0
    drift_level: float = 3.0
    min_instances: int = 4000
    _n: int = 0
    _errors: float = 0.0
    _ps_min: float = math.inf
    _p_min: float = math.inf
    _s_min: float = math.inf

    def reset(self) -> None:
        self._n, self._errors = 0, 0.0
        self._ps_min = self._p_min = self._s_min = math.inf

    def update(self, errors: np.ndarray) -> DDMResult:
        """Feed errors in arrival order; on drift the detector resets and continues."""
        e = np.asarray(errors, dtype=np.float64)
        if len(e) == 0:
            return DDMResult(False, False, self._errors / max(self._n, 1))
        n = self._n + np.arange(1, len(e) + 1)
        p = (self._errors + np.cumsum(e)) / n
        s = np.sqrt(p * (1.0 - p) / n)
        ps = np.where(n >= self.min_instances, p + s, math.inf)
        running = np.minimum.accumulate(np.concatenate([[self._ps_min], ps]))[1:]
        is_new_min = np.isfinite(ps) & (ps <= running)
        last = np.maximum.accumulate(np.where(is_new_min, np.arange(len(e)), -1))
        p_min = np.where(last >= 0, p[np.maximum(last, 0)], self._p_min)
        s_min = np.where(last >= 0, s[np.maximum(last, 0)], self._s_min)
        drift_at = np.flatnonzero(np.isfinite(ps) & (ps > p_min + self.drift_level * s_min))
        if len(drift_at):
            j = int(drift_at[0])
            rate = float(p[j])
            self.reset()
            self.update(e[j + 1 :])
            return DDMResult(True, True, rate)
        self._n, self._errors = int(n[-1]), float(self._errors + e.sum())
        self._ps_min, self._p_min, self._s_min = float(running[-1]), float(p_min[-1]), float(s_min[-1])
        warning = bool(np.isfinite(ps[-1]) and ps[-1] > p_min[-1] + self.warning_level * s_min[-1])
        return DDMResult(False, warning, float(p[-1]))


@dataclass
class LabelRate:
    """Two-sided binomial test of a labelled batch's default rate against the reference rate."""

    reference_rate: float
    alpha: float = 0.001

    def evaluate(self, labels: np.ndarray) -> Signal:
        n, k = len(labels), int(np.sum(labels))
        p_value = binomial_two_sided_p(k, n, self.reference_rate)
        rate = k / max(n, 1)
        message = f"default rate {rate:.1%} against {self.reference_rate:.1%} in the reference"
        return Signal(
            "label_rate", Category.LABEL_DRIFT, "labels", p_value < self.alpha, rate, self.alpha, p_value, message
        )
