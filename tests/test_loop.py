"""The control loop end to end: the service's real code path, with a registry on disk."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.bench.releases import InvertedScore
from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.loop import ConflictError, ControlLoop, InvalidInputError, LoopConfig, ModelBundle, NotFoundError
from driftwatch.registry import ModelRegistry, Stage
from driftwatch.store import Store, pack, unpack
from tests.conftest import stream_batch


@pytest.fixture
def loop(tmp_path, champion):
    registry = ModelRegistry(tmp_path / "registry")
    first = registry.register(ModelBundle(champion.model, champion.reference), reason="champion")
    registry.transition(first.version, Stage.PRODUCTION, "test")
    store = Store()
    yield ControlLoop(registry, store, LoopConfig())
    store.close()
    registry.close()


def stream(loop, scenario, n, onset, delay=30, seed=17):
    results = []
    pending = {}
    for batch in CreditStream(StreamConfig(scenario=scenario, seed=seed, n_batches=n, onset=onset)):
        results.append(loop.ingest(batch.raw, f"b{batch.index}", batch.timestamp))
        pending[batch.index] = batch.labels
        if batch.index >= delay:
            results.append(loop.ingest_labels(f"b{batch.index - delay}", pending.pop(batch.index - delay)))
    return results


def test_bootstrap_registers_a_production_model(tmp_path):
    registry = ModelRegistry(tmp_path)
    loop = ControlLoop.bootstrap(registry, Store())
    record = registry.production()
    assert record is not None and record.version == loop.champion_version == 1
    assert record.metrics["holdout_auc"] > 0.75
    assert ControlLoop.bootstrap(registry, Store()).champion_version == 1
    registry.close()


def test_ingest_scores_routes_and_stores(loop):
    raw = CreditStream(StreamConfig(seed=1, n_batches=1)).batch(0).raw
    result = loop.ingest(raw, "first")
    assert result.decisions.shape == (400,) and 0.7 < result.decisions.mean() < 0.9
    assert loop.store.index_of("first") == 0
    with pytest.raises(ConflictError):
        loop.ingest(raw, "first")
    with pytest.raises(InvalidInputError):
        loop.ingest({"application_id": []}, "empty")


def test_label_delivery_errors(loop):
    raw = CreditStream(StreamConfig(seed=1, n_batches=1)).batch(0).raw
    loop.ingest(raw, "first")
    with pytest.raises(NotFoundError):
        loop.ingest_labels("nope", np.zeros(400))
    with pytest.raises(InvalidInputError):
        loop.ingest_labels("first", np.zeros(399))
    with pytest.raises(InvalidInputError):
        loop.ingest_labels("first", np.full(400, 2))
    loop.ingest_labels("first", np.zeros(400))
    with pytest.raises(ConflictError):
        loop.ingest_labels("first", np.zeros(400))


def test_concept_drift_is_retrained_rolled_out_and_promoted(loop):
    stream(loop, "concept_abrupt", n=200, onset=40)
    registry = loop.registry
    production = registry.production()
    assert production is not None and production.version >= 2
    assert production.parent_version == 1 and production.lineage["challenger"] in ("refit", "recalibrate")
    assert registry.get(1).stage is Stage.ARCHIVED
    stages = [e["to_stage"] for e in registry.events(production.version)]
    assert stages[:4] == ["registered", "shadow", "canary", "production"]


def test_a_bad_release_is_rolled_back_and_production_is_unchanged(loop, champion):
    bad = loop.registry.register(
        ModelBundle(InvertedScore(champion.model), champion.reference), parent=1, reason="planted"
    )
    loop.start_rollout(bad.version)
    stream(loop, "control", n=3, onset=10**9)
    assert loop.registry.get(bad.version).stage is Stage.ROLLED_BACK
    assert loop.registry.production().version == 1
    assert loop.store.rollouts()[0]["status"] == "rolled_back"
    with pytest.raises(ConflictError):
        loop.start_rollout(bad.version)


def test_a_restart_rolls_back_an_interrupted_rollout(loop, champion):
    candidate = loop.registry.register(ModelBundle(champion.model, champion.reference), parent=1, reason="restart test")
    loop.start_rollout(candidate.version)
    restarted = ControlLoop(loop.registry, loop.store, LoopConfig())
    assert loop.registry.get(candidate.version).stage is Stage.ROLLED_BACK
    assert restarted.rollout is None and restarted.champion_version == 1


def test_a_pipeline_incident_blocks_retraining(loop):
    stream(loop, "dq_unit_change", n=150, onset=40)
    assert loop.retrains == []
    assert loop.registry.production().version == 1


def test_store_payloads_round_trip_and_quality_batches_are_not_trained_on(loop):
    batch = stream_batch("dq_new_category", 0, seed=2)
    restored, extra = unpack(pack(batch, {"champion_pd": np.arange(400.0)}))
    np.testing.assert_array_equal(restored.numeric["age"], batch.numeric["age"])
    np.testing.assert_array_equal(restored.codes("employment_type"), batch.codes("employment_type"))
    np.testing.assert_array_equal(extra["champion_pd"], np.arange(400.0))
    stream(loop, "dq_nulls", n=50, onset=25, delay=5)
    window = loop.store.labelled_window(0, 49)
    assert window is not None and window.batch.n_rows == 25 * 400
