"""HTTP layer: validation, auth, status codes, metrics, request ids and the dashboard."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from driftwatch.api.app import create_app
from driftwatch.api.settings import Settings
from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.replay import ReplayConfig, replay, to_records


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(Settings(data_dir=tmp_path_factory.mktemp("api"), log_level="WARNING", max_rows=1000))
    with TestClient(app) as c:
        yield c


def rows(index=0, seed=1, scenario="control"):
    return to_records(
        CreditStream(StreamConfig(scenario=scenario, seed=seed, n_batches=index + 1, onset=0)).batch(index).raw
    )


def test_health_reports_the_bootstrapped_champion(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["champion_version"] == 1


def test_ingest_returns_a_decision_per_application(client):
    response = client.post("/v1/batches", json={"batch_id": "api-1", "rows": rows()})
    assert response.status_code == 201
    body = response.json()
    assert body["n_rows"] == 400 and len(body["decisions"]) == 400
    first = body["decisions"][0]
    assert first["application_id"].startswith("s1-b0-") and 0.0 < first["pd"] < 1.0 and first["decided_by"] == 1


def test_labels_by_application_id(client):
    client.post("/v1/batches", json={"batch_id": "api-2", "rows": rows(1), "include_rows": False})
    ids = [r["application_id"] for r in rows(1)]
    labels = [{"application_id": i, "defaulted": n % 9 == 0} for n, i in enumerate(ids)]
    assert client.post("/v1/batches/api-2/labels", json={"labels": labels[:-1]}).status_code == 422
    body = client.post("/v1/batches/api-2/labels", json={"labels": labels}).json()
    assert body["n_rows"] == 400 and body["default_rate"] == pytest.approx(np.mean([n % 9 == 0 for n in range(400)]))
    assert client.post("/v1/batches/api-2/labels", json={"labels": labels}).status_code == 409


def test_conflicts_and_missing_batches(client):
    assert client.post("/v1/batches", json={"batch_id": "api-1", "rows": rows()}).status_code == 409
    assert (
        client.post(
            "/v1/batches/none/labels", json={"labels": [{"application_id": "x", "defaulted": True}]}
        ).status_code
        == 404
    )


def test_validation_rejects_structures_but_accepts_broken_schemas(client):
    assert client.post("/v1/batches", json={"rows": [{"age": {"nested": 1}}]}).status_code == 422
    assert client.post("/v1/batches", json={"rows": []}).status_code == 422
    assert client.post("/v1/batches", json={"rows": [{"age": 1}], "extra": 1}).status_code == 422
    assert client.post("/v1/batches", json={"rows": [{"age": 1}] * 1001}).status_code == 422
    broken = client.post(
        "/v1/batches", json={"batch_id": "api-broken", "rows": [{"application_id": "z", "age": "forty"}]}
    )
    assert broken.status_code == 201
    assert any(a["category"] == "schema" for a in broken.json()["alerts"])


def test_request_ids_are_echoed_or_generated(client):
    assert client.get("/health", headers={"X-Request-ID": "trace-42"}).headers["X-Request-ID"] == "trace-42"
    generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces"}).headers["X-Request-ID"]
    assert generated != "bad id with spaces" and len(generated) == 32


def test_metrics_are_prometheus_text_and_count_requests(client):
    text = client.get("/metrics").text
    assert "# TYPE driftwatch_batches_total counter" in text
    assert 'driftwatch_http_requests_total{method="POST",path="/v1/batches",status="201"}' in text
    assert "driftwatch_champion_version 1" in text


def test_registry_and_rollout_endpoints(client):
    models = client.get("/v1/registry/models").json()
    assert models[0]["stage"] == "production"
    detail = client.get("/v1/registry/models/1").json()
    assert detail["lineage_chain"] == [1] and detail["events"][0]["to_stage"] == "registered"
    assert client.get("/v1/registry/models/99").status_code == 404
    assert client.post("/v1/rollout/1/start").status_code == 409
    assert client.post("/v1/rollout/abort", json={"reason": "nothing running"}).status_code == 409
    assert client.post("/v1/retrain", json={"reason": "too early"}).status_code == 409
    assert client.get("/v1/rollout").json()["champion_version"] == 1


def test_timeline_alerts_and_acknowledgement(client):
    timeline = client.get("/v1/timeline").json()
    assert timeline[0]["batch_id"] == "api-1" and "psi" in timeline[0]
    alerts = client.get("/v1/alerts?status=open").json()
    assert alerts and alerts[0]["status"] == "open"
    assert client.post(f"/v1/alerts/{alerts[0]['id']}/ack").json()["status"] == "acknowledged"
    assert client.post("/v1/alerts/9999/ack").status_code == 404
    assert client.get("/v1/alerts?status=bogus").status_code == 422


def test_dashboard_and_static_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "driftwatch" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert {s["name"] for s in client.get("/v1/scenarios").json()} >= {"control", "concept_abrupt"}


def test_writes_need_the_api_key_when_one_is_set(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_key="s3cret", log_level="WARNING"))
    with TestClient(app) as c:
        assert c.post("/v1/batches", json={"rows": rows()}).status_code == 401
        assert c.post("/v1/batches", json={"rows": rows()}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/v1/batches", json={"rows": rows()}, headers={"X-API-Key": "s3cret"}).status_code == 201
        assert c.get("/v1/timeline").status_code == 200


def test_replay_drives_the_service_and_the_alerts_follow_the_incident(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, log_level="WARNING"))
    lines = []
    with TestClient(app) as c:
        counts = replay(c, ReplayConfig(scenario="dq_nulls", batches=30, onset=20, label_delay=10), echo=lines.append)
        categories = {a["category"] for a in c.get("/v1/alerts").json()}
    assert counts["batches"] == 30 and counts["labelled"] == 20
    assert "data_quality" in categories
    assert any("alert data_quality:bureau_score" in line for line in lines)
