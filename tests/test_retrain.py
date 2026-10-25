"""Retraining trigger rules and the champion-challenger gate."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.alerts import Alert, Category
from driftwatch.bench.releases import BUREAU
from driftwatch.deploy.retrain import RetrainConfig, RetrainTrigger, champion_challenger_gate, recalibrate, split_window
from driftwatch.model import TrainConfig
from driftwatch.stats import auc
from driftwatch.training import fit_on
from tests.conftest import stream_data


def alert(category: Category, at: int) -> Alert:
    return Alert(category, "model", at, "high", ("x",), "")


def test_label_alerts_record_the_judged_batch_as_evidence():
    trigger = RetrainTrigger()
    trigger.observe(100, [alert(Category.PERFORMANCE, 100)], labelled_index=70)
    assert trigger.evidence == 70
    trigger.observe(110, [], labelled_index=89)
    assert not trigger.due(110)
    trigger.observe(111, [], labelled_index=90)
    assert trigger.due(111)
    assert trigger.window() == (65, 90)


def test_pipeline_incidents_block_retraining():
    trigger = RetrainTrigger()
    trigger.observe(100, [alert(Category.PERFORMANCE, 100), alert(Category.DATA_QUALITY, 100)], labelled_index=95)
    trigger.observe(130, [], labelled_index=125)
    assert trigger.blocked(130) and not trigger.due(130)
    assert trigger.due(131)


def test_label_free_alerts_must_recur_to_trigger():
    trigger = RetrainTrigger()
    trigger.observe(10, [alert(Category.DATA_DRIFT, 10)])
    assert trigger.evidence is None
    trigger.observe(80, [alert(Category.DATA_DRIFT, 80)])
    assert trigger.evidence is None
    trigger.observe(90, [alert(Category.ESTIMATED_PERFORMANCE, 90)])
    assert trigger.evidence == 80


def test_a_rejected_challenger_backs_the_trigger_off():
    trigger = RetrainTrigger(RetrainConfig(backoff_batches=60))
    trigger.observe(50, [alert(Category.LABEL_DRIFT, 50)], labelled_index=20)
    trigger.reset(at=50, rejected=True)
    trigger.observe(60, [alert(Category.LABEL_DRIFT, 60)], labelled_index=60)
    trigger.observe(100, [], labelled_index=85)
    assert not trigger.due(100), "the wait is over but the backoff is not"
    assert trigger.due(110)


def test_split_window_is_time_ordered(champion):
    train, calib, holdout = split_window(champion.train)
    n = champion.train.batch.n_rows
    assert (train.batch.n_rows, calib.batch.n_rows, holdout.batch.n_rows) == (
        round(0.6 * n),
        round(0.15 * n),
        n - round(0.6 * n) - round(0.15 * n),
    )
    np.testing.assert_array_equal(holdout.batch.ids, champion.train.batch.ids[-holdout.batch.n_rows :])


@pytest.fixture(scope="module")
def recent():
    return stream_data("control", 0, 25, seed=808)


def test_the_gate_rejects_a_challenger_that_is_not_better(champion, recent):
    train, calib, holdout = split_window(recent)
    decision = champion_challenger_gate(champion.model, fit_on(train, calib, seed=1), holdout)
    assert not decision.promote


def test_the_gate_promotes_a_clearly_better_challenger(champion, recent):
    weak = fit_on(champion.train, champion.calib, seed=2, config=TrainConfig(dropped=BUREAU))
    _train, _calib, holdout = split_window(recent)
    decision = champion_challenger_gate(weak, champion.model, holdout)
    assert decision.promote and decision.auc_diff_lcb > 0.0
    assert decision.logloss_challenger < decision.logloss_champion


def test_recalibration_keeps_the_ranking_and_fixes_the_level(champion):
    shifted = stream_data("prior_shift", 0, 20, seed=909)
    fixed = recalibrate(champion.model, shifted, approval_rate=0.8)
    before, after = champion.model.predict_pd(shifted.batch), fixed.predict_pd(shifted.batch)
    assert auc(after, shifted.labels) == pytest.approx(auc(before, shifted.labels))
    assert abs(after.mean() - shifted.labels.mean()) < abs(before.mean() - shifted.labels.mean()) / 3
    assert np.mean(after < fixed.threshold) == pytest.approx(0.8, abs=0.01)
