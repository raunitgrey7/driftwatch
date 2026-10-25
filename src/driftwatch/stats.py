"""Statistical primitives shared by the detectors, the guardrails and the benchmark.

They are implemented here instead of calling ``scipy.stats`` per batch because
the monitor runs every test on every batch and the reference side never
changes: sorting the reference once and reusing it keeps a batch at about a
millisecond. Each function is checked against its scipy counterpart (or a
brute-force definition) in the tests.

Two sequential tools live here as well. ``mixture_half_width`` is the
normal-mixture confidence sequence (Robbins; Howard et al. 2021): a bound on
a running sum that holds at every look simultaneously, so a guardrail can be
checked after every batch without inflating its false-alarm rate.
``delong_paired`` gives the variance of the difference of two correlated
AUCs on the same rows (DeLong et al. 1988, with the midrank algorithm of
Sun and Xu 2014).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special
from scipy.stats import rankdata


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC as the Mann-Whitney statistic with midranks; NaN when a class is absent."""
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_auc(scores: np.ndarray, probs: np.ndarray) -> float:
    """AUC expected when row ``i`` is positive with probability ``probs[i]``.

    Ratio of expectations: sum over ordered pairs i != j of
    p_i (1 - p_j) H(s_i - s_j) divided by the sum of p_i (1 - p_j), with H the
    step function and H(0) = 1/2. With ``probs`` in {0, 1} this is exactly
    the realised AUC. Sorting by score makes it O(n log n).
    """
    s = np.asarray(scores, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    order = np.argsort(s, kind="mergesort")
    s, p = s[order], p[order]
    q = 1.0 - p
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    group_q = np.add.reduceat(q, start)
    below = np.repeat(np.concatenate([[0.0], np.cumsum(group_q)[:-1]]), counts)
    tied = np.repeat(group_q, counts) - q
    numerator = float(np.sum(p * (below + 0.5 * tied)))
    denominator = float(p.sum() * q.sum() - np.sum(p * q))
    return numerator / denominator if denominator > 0 else math.nan


def ks_statistic(ref_sorted: np.ndarray, sample: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov D between a pre-sorted reference and a sample.

    Between two consecutive distinct sample values the sample ECDF is flat
    and the reference ECDF only rises, so the supremum is reached at a sample
    value (right limit) or just before the next one (left limit). Evaluating
    the reference ECDF at the sample values alone gives the exact D in
    O(n log m) instead of O((n + m) log m) over the pooled points.
    """
    if len(sample) == 0 or len(ref_sorted) == 0:
        raise ValueError("KS needs two non-empty samples")
    values, counts = np.unique(sample, return_counts=True)
    m = len(ref_sorted)
    after = np.cumsum(counts) / counts.sum()
    before = np.concatenate([[0.0], after[:-1]])
    ref_right = np.searchsorted(ref_sorted, values, side="right") / m
    ref_left = np.searchsorted(ref_sorted, values, side="left") / m
    return float(max(np.max(np.abs(ref_right - after)), np.max(np.abs(ref_left - before))))


def ks_pvalue(d: float, n: int, m: int) -> float:
    """Two-sided p-value from the Kolmogorov limit with Stephens' finite-sample correction."""
    en = math.sqrt(n * m / (n + m))
    return float(np.clip(special.kolmogorov((en + 0.12 + 0.11 / en) * d), 0.0, 1.0))


def chi2_homogeneity(ref_counts: np.ndarray, batch_counts: np.ndarray) -> tuple[float, int, float]:
    """Chi-square test that two count vectors share one distribution; empty cells are dropped."""
    table = np.vstack([ref_counts, batch_counts]).astype(np.float64)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or table[1].sum() == 0 or table[0].sum() == 0:
        return 0.0, 0, 1.0
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    statistic = float(np.sum((table - expected) ** 2 / expected))
    dof = table.shape[1] - 1
    return statistic, dof, float(special.chdtrc(dof, statistic))


def psi(ref_counts: np.ndarray, batch_counts: np.ndarray, smoothing: float = 0.5) -> float:
    """Population stability index with additive smoothing so empty bins stay finite."""
    k = len(ref_counts)
    p = (ref_counts + smoothing) / (ref_counts.sum() + smoothing * k)
    q = (batch_counts + smoothing) / (batch_counts.sum() + smoothing * k)
    return float(np.sum((q - p) * np.log(q / p)))


def binomial_upper_p(k: int, n: int, p0: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p0)."""
    if k <= 0:
        return 1.0
    return float(special.bdtrc(k - 1, n, p0))


def binomial_two_sided_p(k: int, n: int, p0: float) -> float:
    """Two-sided binomial p-value by doubling the smaller tail (capped at 1)."""
    if n == 0:
        return 1.0
    lower = float(special.bdtr(k, n, p0))
    return min(1.0, 2.0 * min(lower, binomial_upper_p(k, n, p0)))


@dataclass(frozen=True)
class PairedAUC:
    """Two AUCs on the same rows and the DeLong standard error of their difference."""

    auc_a: float
    auc_b: float
    diff: float
    se: float
    n_pos: int
    n_neg: int

    def z(self) -> float:
        if self.se > 0:
            return self.diff / self.se
        return 0.0 if self.diff == 0 else math.copysign(math.inf, self.diff)


def delong_paired(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> PairedAUC:
    """DeLong covariance of two correlated AUCs (structural components via midranks)."""
    y = np.asarray(labels).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    if m < 2 or n < 2:
        return PairedAUC(math.nan, math.nan, math.nan, math.nan, m, n)
    v10, v01, aucs = [], [], []
    for s in (np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)):
        pos, neg = s[y], s[~y]
        tx, ty = rankdata(pos), rankdata(neg)
        tz = rankdata(np.concatenate([pos, neg]))
        aucs.append(float((tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)))
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)
    cov = np.cov(np.vstack(v10)) / m + np.cov(np.vstack(v01)) / n
    variance = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    return PairedAUC(aucs[0], aucs[1], aucs[0] - aucs[1], math.sqrt(max(variance, 0.0)), m, n)


def mixture_half_width(variance_sum: float, rho: float, alpha: float) -> float:
    """Half-width for a running sum from the two-sided normal-mixture confidence sequence.

    With ``S_t`` a sum of independent mean-zero increments and ``V_t`` the sum
    of their variances, ``|S_t| <= sqrt((V_t + rho) (2 log(1/alpha) + log((V_t + rho) / rho)))``
    holds for all t at once with probability at least ``1 - alpha`` (Ville's
    inequality on the mixture martingale). ``rho`` sets where it is tightest.
    """
    v = max(variance_sum, 0.0) + rho
    return math.sqrt(v * (2.0 * math.log(1.0 / alpha) + math.log(v / rho)))


def gamma_sf(x: float, shape: float, scale: float) -> float:
    """Survival function of Gamma(shape, scale) via the regularised incomplete gamma function."""
    return float(special.gammaincc(shape, max(x, 0.0) / scale))


def gamma_fit(samples: np.ndarray) -> tuple[float, float]:
    """Method-of-moments Gamma(shape, scale), the classical approximation of an MMD null."""
    mean = float(np.mean(samples))
    var = float(np.var(samples, ddof=1))
    if mean <= 0 or var <= 0:
        raise ValueError("gamma fit needs positive mean and variance")
    return mean * mean / var, var / mean
