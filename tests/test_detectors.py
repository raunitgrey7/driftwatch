"""Detectors: quiet on clean data, loud on the mechanism they are for."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.alerts import Category
from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.detectors.drift import _merge_sparse, feature_drift
from driftwatch.detectors.estimation import CBPEMonitor, em_prior, estimate_performance
from driftwatch.detectors.performance import DDM, BatchedADWIN, LabelRate
from driftwatch.detectors.quality import QualityProfile, _unit_factor, quality_signals, schema_signals
from driftwatch.schema import SchemaIssue, coerce
from tests.conftest import stream_batch


def _fired(signals, detector=None):
    return {(s.detector, s.scope) for s in signals if s.fired and (detector is None or s.detector == detector)}


def test_univariate_drift_is_quiet_on_clean_batches(champion):
    drift = champion.reference.drift
    p_values = [r.p_value for i in range(5) for r in drift.evaluate(stream_batch("control", i, seed=77))]
    assert min(p_values) > 0.001 / 16


def test_univariate_drift_flags_the_shifted_features(champion):
    results = {r.feature: r for r in champion.reference.drift.evaluate(stream_batch("covariate_abrupt", 3, seed=77))}
    assert results["bureau_score"].p_value < 1e-6
    assert results["channel"].test == "chi2" and results["channel"].p_value < 1e-6
    assert results["bureau_score"].psi > 0.1


def test_feature_drift_needs_twenty_values(champion):
    profile = champion.reference.drift.profiles["age"]
    assert feature_drift(profile, np.array([30.0] * 19 + [np.nan] * 50)) is None


def test_categorical_bins_ignore_nulls_and_unseen_values(champion):
    profile = champion.reference.drift.profiles["employment_type"]
    counts = profile.bin_counts(np.array([0, 0, 1, -1, -2, 2]))
    assert counts.tolist() == [2.0, 1.0, 1.0]


def test_sparse_bins_are_pooled_for_the_chi_square():
    ref, batch = _merge_sparse(np.array([900.0, 90.0, 5.0, 5.0]), np.array([90.0, 9.0, 1.0, 0.0]), 100)
    assert ref.tolist() == [900.0, 90.0, 10.0]
    assert batch.tolist() == [90.0, 9.0, 1.0]


def test_mmd_is_calibrated_on_clean_batches_and_sensitive_to_shift(champion):
    mmd = champion.reference.mmd
    clean = [mmd.test(stream_batch("control", i, seed=91))[1] for i in range(20)]
    assert np.mean(np.array(clean) > 0.01) >= 0.85
    assert mmd.test(stream_batch("covariate_abrupt", 2, seed=91))[1] < 1e-4
    assert 400 in mmd._null


def test_score_drift_sees_a_shift_in_model_outputs(champion):
    stat, p = champion.reference.scores.evaluate(champion.model.predict_pd(stream_batch("covariate_abrupt", 1, seed=3)))
    assert p < 1e-4 and stat > 0.1


def _quality(champion, scenario):
    return quality_signals(stream_batch(scenario, 2, seed=13), champion.reference.quality)


def test_quality_checks_are_quiet_on_clean_data(champion):
    assert _fired(_quality(champion, "control")) == set()


def test_null_spike_and_unit_change_and_unseen_values_fire(champion):
    assert ("null_rate", "bureau_score") in _fired(_quality(champion, "dq_nulls"))
    unit = next(
        s for s in _quality(champion, "dq_unit_change") if s.detector == "unit_change" and s.scope == "monthly_income"
    )
    assert unit.fired and "x12" in unit.message
    assert ("range", "monthly_income") in _fired(_quality(champion, "dq_unit_change"))
    assert ("unseen_category", "employment_type") in _fired(_quality(champion, "dq_new_category"))


def test_unit_factor_picks_the_closest_conversion():
    assert _unit_factor(11.2, 0.15) == 12.0
    assert _unit_factor(9.6, 0.15) == 10.0
    assert _unit_factor(0.0011, 0.15) == pytest.approx(1 / 1000)
    assert _unit_factor(2.0, 0.15) is None


def test_small_ratio_changes_are_not_unit_changes(champion):
    batch = stream_batch("control", 2, seed=13)
    batch.numeric["monthly_income"] = batch.numeric["monthly_income"] * 1.1
    assert ("unit_change", "monthly_income") not in _fired(quality_signals(batch, champion.reference.quality))


def test_quality_profile_floors_non_negative_ranges(champion):
    lo, _ = champion.reference.quality.ranges["monthly_income"]
    assert lo == 0.0
    profile = QualityProfile.fit(champion.calib.batch)
    assert profile.null_rates["home_ownership"] > 0.0


def test_schema_issues_become_schema_signals():
    signals = schema_signals([SchemaIssue("missing_column", "age", "absent", 1.0)])
    assert signals[0].category is Category.SCHEMA and signals[0].fired


def test_cbpe_estimates_and_band(champion):
    pd = champion.model.predict_pd(champion.holdout.batch)
    estimate = estimate_performance(pd, champion.model.threshold)
    assert 0.75 < estimate.approval_rate < 0.85
    assert estimate.default_rate == pytest.approx(pd.mean())
    band = champion.reference.cbpe
    assert band.lower_bound(100) < band.lower_bound(400) < band.ref_mean
    _, signal = band.evaluate(np.full(400, 0.08))
    assert signal.fired


def test_cbpe_fit_is_reproducible(champion):
    pd = champion.model.predict_pd(champion.holdout.batch)
    a = CBPEMonitor.fit(pd, 0.1, 400, seed=3)
    b = CBPEMonitor.fit(pd, 0.1, 400, seed=3)
    assert (a.ref_mean, a.ref_sd) == (b.ref_mean, b.ref_sd)


def test_em_prior_recovers_a_label_shift(rng):
    """With exact posteriors and P(x | y) fixed, EM finds the new prior; mean PD does not."""
    n = 40_000
    y = rng.random(n) < 0.25
    x = np.where(y, rng.normal(1.0, 1.0, n), rng.normal(0.0, 1.0, n))
    train_prior = 0.10
    odds = np.exp(x - 0.5) * train_prior / (1 - train_prior)
    pd = odds / (1 + odds)
    assert em_prior(pd, train_prior) == pytest.approx(0.25, abs=0.02)
    assert pd.mean() < 0.2


def test_adwin_is_quiet_on_a_stationary_stream_and_cuts_on_a_shift(rng):
    adwin = BatchedADWIN()
    assert not any(adwin.update(rng.random(400) < 0.08).drift for _ in range(120))
    shifted = [adwin.update((rng.random(400) < 0.20).astype(float)) for _ in range(20)]
    drift = next(r for r in shifted if r.drift)
    assert drift.mean_after > drift.mean_before
    assert shifted[-1].width < 140 * 400


def test_ddm_waits_for_its_burn_in_and_detects_a_jump(rng):
    ddm = DDM()
    assert not ddm.update(np.ones(3000)).drift
    ddm.reset()
    stationary = [ddm.update(rng.random(400) < 0.19).drift for _ in range(40)]
    assert not any(stationary)
    results = [ddm.update(rng.random(400) < 0.30) for _ in range(30)]
    assert any(r.drift for r in results)


def test_label_rate_flags_a_default_rate_change():
    monitor = LabelRate(reference_rate=0.08)
    as_expected = np.array([1] * 32 + [0] * 368)
    doubled = np.array([1] * 80 + [0] * 320)
    assert not monitor.evaluate(as_expected).fired
    assert monitor.evaluate(doubled).fired


def test_stream_raw_columns_coerce_without_issues():
    raw = CreditStream(StreamConfig(seed=1, n_batches=1)).batch(0).raw
    assert coerce(raw)[1] == []
