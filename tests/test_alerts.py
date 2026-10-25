"""The alert policy: persistence, cooldowns, independent keys, suppression, advisory detectors."""

from __future__ import annotations

import pytest

from driftwatch.alerts import AlertPolicy, Category, Rule, Signal


def drift(scope: str, fired: bool, detector: str = "ks") -> Signal:
    return Signal(
        detector, Category.DATA_DRIFT, scope, fired, 1.0, 0.5, 1e-9 if fired else 0.5, f"{detector} on {scope}"
    )


def quality(scope: str) -> Signal:
    return Signal("null_rate", Category.DATA_QUALITY, scope, True, 0.4, 0.02, None, "nulls")


def test_two_of_three_persistence_needs_two_fired_evaluations():
    policy = AlertPolicy()
    assert policy.apply(0, [drift("age", True)]) == []
    assert policy.apply(1, [drift("age", False)]) == []
    alerts = policy.apply(2, [drift("age", True)])
    assert [(a.category, a.scope, a.batch_index) for a in alerts] == [(Category.DATA_DRIFT, "age", 2)]


def test_cooldown_suppresses_repeats_and_then_releases():
    policy = AlertPolicy(rules={**AlertPolicy().rules, Category.DATA_DRIFT: Rule(1, 1, 5)})
    opened = [t for t in range(12) if policy.apply(t, [drift("age", True)])]
    assert opened == [0, 5, 10]


def test_keys_are_independent_per_feature():
    policy = AlertPolicy(rules={**AlertPolicy().rules, Category.DATA_DRIFT: Rule(1, 1, 30)})
    assert len(policy.apply(0, [drift("age", True)])) == 1
    assert len(policy.apply(1, [drift("age", True), drift("bureau_score", True)])) == 1


def test_drift_explained_by_a_quality_incident_is_suppressed():
    policy = AlertPolicy(rules={**AlertPolicy().rules, Category.DATA_DRIFT: Rule(1, 1, 30)})
    alerts = policy.apply(0, [quality("monthly_income"), drift("monthly_income", True), drift("age", True)])
    assert {(a.category, a.scope) for a in alerts} == {
        (Category.DATA_QUALITY, "monthly_income"),
        (Category.DATA_DRIFT, "age"),
    }
    assert policy.suppressed == 1


def test_advisory_detectors_never_open_alerts():
    policy = AlertPolicy()
    ddm = Signal("ddm", Category.PERFORMANCE, "model", True, 0.2, 3.0)
    assert policy.apply(0, [ddm]) == []
    assert len(AlertPolicy(advisory=frozenset()).apply(0, [ddm])) == 1


def test_alerts_come_out_in_severity_order_with_joined_messages():
    policy = AlertPolicy(rules={**AlertPolicy().rules, Category.DATA_DRIFT: Rule(1, 1, 30)})
    schema = Signal("schema", Category.SCHEMA, "age", True, 1.0, 0.0, None, "missing")
    alerts = policy.apply(0, [drift("age", True, "psi"), drift("age", True, "ks"), schema])
    assert [a.category for a in alerts] == [Category.SCHEMA, Category.DATA_DRIFT]
    assert alerts[1].detectors == ("ks", "psi")
    assert "ks on age" in alerts[1].message and "psi on age" in alerts[1].message


def test_rule_validation():
    with pytest.raises(ValueError):
        Rule(3, 2, 0)
    with pytest.raises(ValueError):
        Rule(1, 1, -1)
