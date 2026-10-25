"""The monitor end to end on short streams: which alert, how soon."""

from __future__ import annotations

import json

import numpy as np

from driftwatch.alerts import Category
from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.monitor import Monitor
from driftwatch.schema import coerce


def run(champion, scenario, n=80, onset=40, delay=20, seed=33):
    monitor = Monitor(champion.model, champion.reference)
    stream = CreditStream(StreamConfig(scenario=scenario, seed=seed, n_batches=n, onset=onset))
    pending, alerts, reports = {}, [], []
    for batch in stream:
        report = monitor.observe(batch.index, batch.raw, batch.timestamp)
        reports.append(report)
        alerts += report.alerts
        pending[batch.index] = (batch.labels, report.scores)
        if batch.index >= delay:
            alerts += monitor.observe_labels(batch.index, batch.index - delay, *pending.pop(batch.index - delay)).alerts
    return alerts, reports


def first(alerts, category):
    return min((a.batch_index for a in alerts if a.category is category), default=None)


def test_a_clean_stream_raises_no_alerts(champion):
    alerts, _ = run(champion, "control", n=70, onset=10**9)
    assert alerts == []


def test_covariate_shift_is_a_data_drift_alert_within_two_batches(champion):
    alerts, _ = run(champion, "covariate_abrupt")
    assert first(alerts, Category.DATA_DRIFT) in (40, 41, 42)
    assert all(a.batch_index >= 40 for a in alerts)


def test_null_incident_alerts_on_its_first_batch(champion):
    alerts, _ = run(champion, "dq_nulls")
    assert first(alerts, Category.DATA_QUALITY) == 40


def test_schema_break_alerts_on_its_first_batch(champion):
    alerts, _ = run(champion, "dq_schema_break")
    scopes = {a.scope for a in alerts if a.category is Category.SCHEMA}
    assert first(alerts, Category.SCHEMA) == 40
    assert {"credit_utilization", "credit_utilization_pct"} <= scopes


def test_concept_drift_needs_the_labels(champion):
    alerts, _ = run(champion, "concept_abrupt", n=90)
    assert first(alerts, Category.DATA_DRIFT) is None
    assert 60 <= first(alerts, Category.PERFORMANCE) <= 70


def test_reports_serialise_to_finite_json(champion):
    _, reports = run(champion, "dq_unit_change", n=45)
    text = json.dumps([r.summary() for r in reports])
    assert "NaN" not in text and "Infinity" not in text


def test_coerced_batch_with_issues_equals_raw_input(champion):
    raw = CreditStream(StreamConfig(scenario="dq_schema_break", seed=4, n_batches=2, onset=0)).batch(1).raw
    batch, issues = coerce(raw)
    a = Monitor(champion.model, champion.reference).observe(0, raw)
    b = Monitor(champion.model, champion.reference).observe(0, batch, issues=issues)
    assert [(s.detector, s.scope, s.fired) for s in a.signals] == [(s.detector, s.scope, s.fired) for s in b.signals]
    np.testing.assert_array_equal(a.scores, b.scores)
