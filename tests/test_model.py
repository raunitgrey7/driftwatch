"""The champion model: discrimination, calibration, its cut-off and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.model import MODEL_FEATURES, TrainConfig, encode, fit_platt
from driftwatch.stats import auc
from driftwatch.training import HistoryConfig, build_champion, fit_on


def test_champion_discriminates_and_is_calibrated_on_unseen_data(champion, clean_sample):
    pd = champion.model.predict_pd(clean_sample.batch)
    assert auc(pd, clean_sample.labels) > 0.75
    assert pd.mean() == pytest.approx(clean_sample.labels.mean(), abs=0.01)


def test_cut_off_approves_the_target_share_of_the_calibration_population(champion):
    pd = champion.model.predict_pd(champion.calib.batch)
    assert np.mean(pd < champion.model.threshold) == pytest.approx(0.80, abs=0.01)


def test_encode_maps_unknown_categories_to_missing(champion):
    batch = champion.holdout.batch.take(np.arange(3))
    batch.categorical["channel"] = np.array(["branch", "kiosk", None], dtype=object)
    batch._codes.pop("channel", None)
    x = encode(batch)
    column = MODEL_FEATURES.index("channel")
    assert x.shape == (3, len(MODEL_FEATURES))
    assert x[0, column] == 0.0 and np.isnan(x[1:, column]).all()
    assert encode(batch, dropped=("channel",)).shape[1] == len(MODEL_FEATURES) - 1


def test_training_is_deterministic_for_a_seed(champion):
    again = fit_on(champion.train, champion.calib, seed=champion.model.info["seed"])
    np.testing.assert_array_equal(
        again.predict_pd(champion.holdout.batch), champion.model.predict_pd(champion.holdout.batch)
    )


def test_dropping_the_bureau_feed_costs_discrimination(champion, clean_sample):
    weak = build_champion(
        HistoryConfig(seed=7),
        train_config=TrainConfig(dropped=("bureau_score", "delinquencies_2y", "credit_utilization")),
    ).model
    assert (
        auc(weak.predict_pd(clean_sample.batch), clean_sample.labels)
        < auc(champion.model.predict_pd(clean_sample.batch), clean_sample.labels) - 0.05
    )


def test_platt_recovers_known_parameters(rng):
    raw = rng.uniform(0.01, 0.6, 50_000)
    z = 1.3 * np.log(raw / (1 - raw)) - 0.4
    labels = (rng.random(len(raw)) < 1 / (1 + np.exp(-z))).astype(int)
    a, b = fit_platt(raw, labels)
    assert a == pytest.approx(1.3, abs=0.06)
    assert b == pytest.approx(-0.4, abs=0.08)
