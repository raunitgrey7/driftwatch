"""Rollout guardrails against planted defects, using the champion's own scores as ground truth."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtri

from driftwatch.bench.releases import Crashing, InvertedScore, Uncalibrated
from driftwatch.deploy.rollout import Rollout, RolloutConfig, RolloutStatus, bucket, failed_rows
from tests.conftest import stream_data


@pytest.fixture(scope="module")
def traffic():
    return [stream_data("control", i, i + 1, seed=314) for i in range(60)]


def drive(rollout, model, champion, traffic, delay=10):
    """Feed batches and delayed labels until the rollout decides or the traffic ends."""
    scored = []
    for t, data in enumerate(traffic):
        champ = champion.model.predict_pd(data.batch)
        cand = model.predict_pd(data.batch)
        outcome = rollout.observe_batch(data.batch.ids, champ, cand)
        scored.append((data.labels, champ, cand))
        if t >= delay and rollout.status is RolloutStatus.RUNNING:
            rollout.observe_labels(*scored[t - delay])
        if rollout.status is not RolloutStatus.RUNNING:
            return outcome
    return None


def make(champion, model, **overrides):
    return Rollout(2, champion.model.threshold, model.threshold, replace(RolloutConfig(), **overrides))


def test_routing_is_stable_and_uniform():
    ids = np.array([f"app-{i}" for i in range(20_000)], dtype=object)
    u = bucket(ids, "v2")
    np.testing.assert_array_equal(u, bucket(ids, "v2"))
    assert abs(np.mean(u < 0.25) - 0.25) < 0.01
    assert not np.array_equal(u, bucket(ids, "v3"))


def test_an_identical_candidate_is_promoted(champion, traffic):
    rollout = make(champion, champion.model, max_batches=60)
    drive(rollout, champion.model, champion, traffic)
    assert rollout.status is RolloutStatus.PROMOTED
    assert [h["reason"] for h in rollout.history if h["event"] == "advance"] == ["canary-5", "canary-25", "canary-50"]


@pytest.mark.parametrize(
    ("wrapper", "guardrail"),
    [(InvertedScore, "rank_agreement"), (Uncalibrated, "approval_rate"), (Crashing, "runtime")],
)
def test_gross_defects_are_rolled_back_in_shadow(champion, traffic, wrapper, guardrail):
    model = wrapper(champion.model)
    rollout = make(champion, model)
    drive(rollout, model, champion, traffic)
    assert rollout.status is RolloutStatus.ROLLED_BACK
    assert rollout.reason.startswith(guardrail)
    assert rollout.exposure == 0 and rollout.batches == 1


def test_failed_rows_fall_back_to_the_champion(champion, traffic):
    model = Crashing(champion.model, fail_rate=0.004)
    rollout = make(
        champion,
        model,
        steps=(RolloutConfig().steps[0], replace(RolloutConfig().steps[1], traffic=1.0)),
        error_rate_max=0.05,
    )
    data = traffic[0]
    champ, cand = champion.model.predict_pd(data.batch), model.predict_pd(data.batch)
    for _ in range(6):
        outcome = rollout.observe_batch(data.batch.ids, champ, cand)
    failed = failed_rows(cand)
    assert failed.any()
    assert not outcome.served_by_candidate[failed].any()
    np.testing.assert_array_equal(outcome.decisions[failed], champ[failed] < champion.model.threshold)


def test_traffic_share_follows_the_step(champion, traffic):
    rollout = make(champion, champion.model)
    shares = []
    for data in traffic[:8]:
        pd = champion.model.predict_pd(data.batch)
        shares.append((rollout.step_name, rollout.observe_batch(data.batch.ids, pd, pd).served_by_candidate.mean()))
    assert all(share == 0.0 for name, share in shares if name == "shadow")
    assert 0.0 < np.mean([share for name, share in shares if name == "canary-5"]) < 0.12


def test_traffic_above_five_percent_waits_for_label_evidence(champion, traffic):
    rollout = make(champion, champion.model)
    for data in traffic[:25]:
        pd = champion.model.predict_pd(data.batch)
        rollout.observe_batch(data.batch.ids, pd, pd)
    assert rollout.step_name == "canary-5" and not rollout.noninferior


def test_a_rollout_without_proof_times_out(champion, traffic):
    rollout = make(champion, champion.model, max_batches=12)
    drive(rollout, champion.model, champion, traffic, delay=50)
    assert rollout.status is RolloutStatus.ROLLED_BACK
    assert "non-inferiority not established" in rollout.reason


def test_look_threshold_spends_alpha_over_every_look():
    cfg = RolloutConfig(max_batches=150, look_every=5, alpha=0.05)
    assert cfg.z_look == pytest.approx(float(ndtri(1 - 0.05 / 60)))
