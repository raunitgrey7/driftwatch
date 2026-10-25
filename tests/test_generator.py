"""The generator must change exactly one mechanism per scenario, at the stated batch."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

from driftwatch.data.generator import (
    BASELINE_NULL_RATES,
    Concept,
    Population,
    default_probability,
    lerp,
    monthly_instalment,
    sample_applicants,
)
from driftwatch.data.scenarios import (
    BASE_CONCEPT,
    NULL_INCIDENT_RATE,
    PRIOR_TARGET_MULTIPLIER,
    SCENARIOS,
    CreditStream,
    StreamConfig,
    base_default_rate,
    drift_weight,
)
from driftwatch.schema import ID_COLUMN
from driftwatch.stats import auc


def test_base_population_matches_the_documented_default_rate_and_power(rng):
    columns = sample_applicants(rng, 100_000, Population())
    pd = default_probability(columns, Concept())
    labels = rng.random(len(pd)) < pd
    assert 0.075 < labels.mean() < 0.085
    assert 0.78 < auc(pd, labels) < 0.82


def test_same_seed_and_index_give_the_same_batch():
    a = CreditStream(StreamConfig(seed=3, n_batches=5)).batch(4)
    b = CreditStream(StreamConfig(seed=3, n_batches=9)).batch(4)
    c = CreditStream(StreamConfig(seed=4, n_batches=5)).batch(4)
    np.testing.assert_array_equal(a.raw["monthly_income"], b.raw["monthly_income"])
    np.testing.assert_array_equal(a.labels, b.labels)
    assert not np.array_equal(a.raw["monthly_income"], c.raw["monthly_income"])


@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s != "control"])
def test_every_scenario_equals_control_before_its_onset(scenario):
    drifted = CreditStream(StreamConfig(scenario=scenario, seed=8, n_batches=12, onset=10)).batch(9)
    control = CreditStream(StreamConfig(seed=8, n_batches=12)).batch(9)
    np.testing.assert_array_equal(drifted.raw["bureau_score"], control.raw["bureau_score"])
    np.testing.assert_array_equal(drifted.labels, control.labels)


def test_drift_weight_steps_or_ramps_from_the_onset():
    abrupt = StreamConfig(scenario="covariate_abrupt", onset=10)
    gradual = StreamConfig(scenario="covariate_gradual", onset=10)
    subtle = StreamConfig(scenario="covariate_subtle", onset=10)
    assert drift_weight(abrupt, 9) == 0.0 and drift_weight(abrupt, 10) == 1.0
    assert drift_weight(gradual, 10) == pytest.approx(1 / 40)
    assert drift_weight(gradual, 49) == 1.0
    assert drift_weight(subtle, 30) == pytest.approx(0.3)


def test_concept_drift_leaves_the_applicants_unchanged_and_moves_the_risk():
    concept = CreditStream(StreamConfig(scenario="concept_abrupt", seed=8, n_batches=3, onset=0)).batch(2)
    control = CreditStream(StreamConfig(seed=8, n_batches=3)).batch(2)
    for name in ("bureau_score", "monthly_income", "employment_type"):
        np.testing.assert_array_equal(concept.raw[name], control.raw[name])
    assert not np.allclose(concept.true_pd, control.true_pd)


def test_covariate_shift_keeps_the_default_mechanism():
    batch = CreditStream(StreamConfig(scenario="covariate_abrupt", seed=8, n_batches=3, onset=0)).batch(2)
    clean = {k: v for k, v in batch.raw.items() if k != ID_COLUMN}
    complete = {k: v for k, v in clean.items() if k not in BASELINE_NULL_RATES}
    assert complete
    rows = (
        ~np.isnan(batch.raw["monthly_income"])
        & ~np.isnan(batch.raw["num_open_accounts"])
        & ~np.isnan(batch.raw["credit_history_months"])
    )
    rows &= np.array([v is not None for v in batch.raw["home_ownership"].tolist()])
    recomputed = default_probability({k: v[rows] for k, v in clean.items()}, BASE_CONCEPT)
    np.testing.assert_allclose(recomputed, batch.true_pd[rows])


def test_prior_shift_moves_the_default_rate_and_keeps_x_given_y():
    stream = CreditStream(StreamConfig(scenario="prior_shift", seed=21, n_batches=40, onset=0))
    control = CreditStream(StreamConfig(seed=22, n_batches=40))
    shifted = [stream.batch(i) for i in range(40)]
    base = [control.batch(i) for i in range(40)]
    rate = np.mean(np.concatenate([b.labels for b in shifted]))
    assert rate == pytest.approx(base_default_rate() * PRIOR_TARGET_MULTIPLIER, abs=0.01)

    def defaulters(batches):
        return np.concatenate([b.raw["bureau_score"][b.labels == 1] for b in batches])

    assert ks_2samp(defaulters(shifted), defaulters(base)).pvalue > 0.001
    true_mean = np.mean(np.concatenate([b.true_pd for b in shifted]))
    assert true_mean == pytest.approx(rate, abs=0.01)


def test_quality_incidents_corrupt_the_pipeline_not_the_world():
    def batch(name):
        return CreditStream(StreamConfig(scenario=name, seed=8, n_batches=3, onset=0)).batch(2)

    control = CreditStream(StreamConfig(seed=8, n_batches=3)).batch(2)
    nulls = batch("dq_nulls")
    assert abs(np.isnan(nulls.raw["bureau_score"]).mean() - NULL_INCIDENT_RATE) < 0.08
    np.testing.assert_allclose(batch("dq_unit_change").raw["monthly_income"], control.raw["monthly_income"] * 12.0)
    broken = batch("dq_schema_break").raw
    assert "credit_utilization" not in broken and "credit_utilization_pct" in broken
    assert "gig" in set(batch("dq_new_category").raw["employment_type"].tolist())
    np.testing.assert_array_equal(nulls.true_pd, control.true_pd)


def test_lerp_endpoints_and_valid_mixtures():
    a, b = Population(), Population(z_mean=-1.0, channel_probs=(0.2, 0.3, 0.5))
    assert lerp(a, b, 0.0) is a and lerp(a, b, 1.0) is b
    mid = lerp(a, b, 0.5)
    assert mid.z_mean == pytest.approx(-0.5)
    assert sum(mid.channel_probs) == pytest.approx(1.0)


def test_annuity_instalment_matches_the_closed_form():
    emi = monthly_instalment(np.array([100_000.0]), np.array([12.0]), annual_rate=0.12)
    assert emi[0] == pytest.approx(8884.88, abs=0.01)


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        StreamConfig(scenario="meteor_strike")
