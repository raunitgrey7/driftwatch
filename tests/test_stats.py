"""Each statistic is checked against scipy, scikit-learn or its brute-force definition."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from driftwatch.stats import (
    auc,
    binomial_two_sided_p,
    binomial_upper_p,
    chi2_homogeneity,
    delong_paired,
    expected_auc,
    gamma_fit,
    gamma_sf,
    ks_pvalue,
    ks_statistic,
    mixture_half_width,
    psi,
)

scores = st.lists(st.integers(0, 6), min_size=4, max_size=40)


@settings(max_examples=80, deadline=None)
@given(scores, st.data())
def test_auc_matches_sklearn_with_ties(values, data):
    labels = data.draw(st.lists(st.booleans(), min_size=len(values), max_size=len(values)))
    if len(set(labels)) < 2:
        assert math.isnan(auc(np.array(values, float), np.array(labels)))
        return
    assert auc(np.array(values, float), np.array(labels)) == pytest.approx(roc_auc_score(labels, values))


def _brute_expected_auc(s, p):
    num = den = 0.0
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j:
                continue
            w = p[i] * (1 - p[j])
            den += w
            num += w * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
    return num / den


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 5), st.floats(0.01, 0.99)), min_size=3, max_size=25))
def test_expected_auc_matches_the_pairwise_definition(pairs):
    s = np.array([p[0] for p in pairs], float)
    p = np.array([p[1] for p in pairs])
    assert expected_auc(s, p) == pytest.approx(_brute_expected_auc(s, p), rel=1e-9)


@settings(max_examples=60, deadline=None)
@given(scores, st.data())
def test_expected_auc_with_certain_labels_is_the_realised_auc(values, data):
    labels = data.draw(st.lists(st.booleans(), min_size=len(values), max_size=len(values)))
    if len(set(labels)) < 2:
        return
    s, y = np.array(values, float), np.array(labels)
    assert expected_auc(s, y.astype(float)) == pytest.approx(auc(s, y))


@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(0, 8), min_size=1, max_size=60), st.lists(st.integers(0, 8), min_size=1, max_size=60))
def test_ks_statistic_is_exact_including_ties(a, b):
    a, b = np.array(a, float), np.array(b, float)
    assert ks_statistic(np.sort(a), b) == pytest.approx(sps.ks_2samp(a, b).statistic, abs=1e-12)


def test_ks_pvalue_tracks_the_exact_distribution(rng):
    ref = np.sort(rng.normal(size=4000))
    for shift in (0.0, 0.15, 0.3):
        x = rng.normal(shift, 1.0, 400)
        d = ks_statistic(ref, x)
        exact = sps.ks_2samp(ref, x, method="exact").pvalue
        assert ks_pvalue(d, 400, 4000) == pytest.approx(exact, rel=0.1, abs=1e-12)


def test_chi2_matches_scipy_contingency():
    ref, batch = np.array([500.0, 300.0, 200.0]), np.array([40.0, 45.0, 15.0])
    statistic, dof, p = chi2_homogeneity(ref, batch)
    expected = sps.chi2_contingency(np.vstack([ref, batch]), correction=False)
    assert statistic == pytest.approx(expected.statistic)
    assert dof == expected.dof
    assert p == pytest.approx(expected.pvalue)


def test_chi2_degenerate_tables_are_not_significant():
    assert chi2_homogeneity(np.array([10.0, 0.0]), np.array([0.0, 0.0]))[2] == 1.0
    assert chi2_homogeneity(np.array([10.0]), np.array([3.0]))[2] == 1.0


def test_psi_is_zero_for_identical_and_symmetric():
    p, q = np.array([100.0, 200.0, 700.0]), np.array([300.0, 300.0, 400.0])
    assert psi(p, p) == pytest.approx(0.0, abs=1e-12)
    assert psi(p, q) == pytest.approx(psi(q, p), rel=0.05)
    assert psi(p, q) > 0.1


def test_binomial_p_values_match_scipy():
    assert binomial_upper_p(15, 100, 0.08) == pytest.approx(sps.binom.sf(14, 100, 0.08))
    assert binomial_upper_p(0, 100, 0.08) == 1.0
    minlike = sps.binomtest(20, 100, 0.08).pvalue
    doubled = binomial_two_sided_p(20, 100, 0.08)
    assert minlike <= doubled <= 2.0 * minlike + 1e-15
    assert binomial_two_sided_p(8, 100, 0.08) == 1.0


def test_delong_standard_error_agrees_with_the_bootstrap(rng):
    n = 3000
    y = rng.random(n) < 0.2
    a = y * 1.0 + rng.normal(0, 1.2, n)
    b = a + rng.normal(0, 0.6, n)
    paired = delong_paired(y, a, b)
    assert paired.diff == pytest.approx(paired.auc_a - paired.auc_b)
    boots = []
    for _ in range(300):
        i = rng.integers(0, n, n)
        boots.append(auc(a[i], y[i]) - auc(b[i], y[i]))
    assert paired.se == pytest.approx(np.std(boots), rel=0.2)
    assert delong_paired(np.array([1, 0]), np.array([0.2, 0.1]), np.array([0.1, 0.2])).se != paired.se


def test_mixture_confidence_sequence_holds_at_every_look(rng):
    """Peeking after every step: the boundary is crossed in at most about alpha of the paths."""
    crossed = 0
    for _ in range(400):
        steps = rng.choice([-1.0, 1.0], size=2000)
        path = np.cumsum(steps)
        variance = np.arange(1, len(steps) + 1, dtype=float)
        bound = np.array([mixture_half_width(v, rho=50.0, alpha=0.05) for v in variance[::10]])
        crossed += bool(np.any(np.abs(path[::10]) > bound))
    assert crossed / 400 <= 0.07


def test_gamma_fit_and_survival_match_scipy(rng):
    samples = rng.gamma(3.0, 0.02, 20_000)
    shape, scale = gamma_fit(samples)
    assert shape == pytest.approx(3.0, rel=0.05)
    assert scale == pytest.approx(0.02, rel=0.05)
    assert gamma_sf(0.1, 3.0, 0.02) == pytest.approx(sps.gamma.sf(0.1, 3.0, scale=0.02))
    with pytest.raises(ValueError):
        gamma_fit(np.zeros(5))
