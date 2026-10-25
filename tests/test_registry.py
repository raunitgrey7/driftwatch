"""Registry: hashed artifacts, the stage machine, atomic promotion, rollback, lineage and audit."""

from __future__ import annotations

import pytest

from driftwatch.registry import ModelRegistry, RegistryError, Stage


@pytest.fixture
def registry(tmp_path):
    reg = ModelRegistry(tmp_path)
    yield reg
    reg.close()


def test_register_and_load_round_trip(registry):
    record = registry.register({"weights": [1, 2, 3]}, lineage={"window": [0, 9]}, metrics={"auc": 0.8}, reason="first")
    assert record.stage is Stage.REGISTERED and record.version == 1
    assert registry.load(1) == {"weights": [1, 2, 3]}
    assert registry.get(1).lineage == {"window": [0, 9]}


def test_a_tampered_artifact_is_refused(registry):
    record = registry.register({"weights": [1]})
    (registry.artifacts / record.artifact_path).write_bytes(b"not the model")
    with pytest.raises(RegistryError, match="hash"):
        registry.load(record.version)


def test_illegal_transitions_are_refused(registry):
    registry.register("a")
    registry.transition(1, Stage.REJECTED, "bad")
    with pytest.raises(RegistryError, match="not allowed"):
        registry.transition(1, Stage.SHADOW, "retry")


def test_promotion_keeps_exactly_one_production_version(registry):
    registry.register("v1")
    registry.transition(1, Stage.PRODUCTION, "bootstrap")
    registry.register("v2", parent=1)
    registry.transition(2, Stage.SHADOW, "rollout")
    registry.transition(2, Stage.CANARY, "advance")
    registry.promote(2, "non-inferior")
    assert [r.version for r in registry.in_stage(Stage.PRODUCTION)] == [2]
    assert registry.get(1).stage is Stage.ARCHIVED
    with pytest.raises(RegistryError, match="use promote"):
        registry.register("v3") and registry.transition(3, Stage.PRODUCTION, "second")


def test_rollback_of_a_canary_leaves_production_alone(registry):
    registry.register("v1")
    registry.transition(1, Stage.PRODUCTION, "bootstrap")
    registry.register("v2", parent=1)
    registry.transition(2, Stage.SHADOW, "rollout")
    registry.rollback(2, "guardrail")
    assert registry.get(2).stage is Stage.ROLLED_BACK
    assert registry.production().version == 1


def test_rollback_of_production_restores_its_parent(registry):
    registry.register("v1")
    registry.transition(1, Stage.PRODUCTION, "bootstrap")
    registry.register("v2", parent=1)
    registry.transition(2, Stage.SHADOW, "rollout")
    registry.transition(2, Stage.CANARY, "advance")
    registry.promote(2, "ok")
    registry.rollback(2, "regret")
    assert registry.production().version == 1
    with pytest.raises(RegistryError, match="no archived"):
        registry.rollback(1, "nothing to restore")


def test_lineage_and_the_audit_trail(registry):
    registry.register("v1")
    registry.register("v2", parent=1)
    registry.register("v3", parent=2)
    assert [r.version for r in registry.lineage(3)] == [3, 2, 1]
    registry.transition(3, Stage.REJECTED, "gate said no")
    events = registry.events(3)
    assert [(e["from_stage"], e["to_stage"]) for e in events] == [(None, "registered"), ("registered", "rejected")]
    assert events[-1]["reason"] == "gate said no"


def test_unknown_versions_and_parents_raise(registry):
    with pytest.raises(RegistryError, match="no model version"):
        registry.get(7)
    with pytest.raises(RegistryError):
        registry.register("orphan", parent=7)


def test_state_survives_reopening(tmp_path):
    first = ModelRegistry(tmp_path)
    first.register("v1")
    first.transition(1, Stage.PRODUCTION, "bootstrap")
    first.close()
    second = ModelRegistry(tmp_path)
    assert second.production().version == 1 and second.load(1) == "v1"
    second.close()
