"""Version 1 routes: ingest, labels, timeline, alerts, registry, rollout and retraining."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from driftwatch.api.auth import require_api_key
from driftwatch.api.schemas import (
    IDENTIFIER,
    AlertOut,
    BatchIn,
    BatchOut,
    LabelsIn,
    LabelsOut,
    ModelDetail,
    ModelOut,
    ReasonIn,
    RetrainOut,
)
from driftwatch.api.service import Services, retrain_out
from driftwatch.data.scenarios import SCENARIOS

router = APIRouter(prefix="/v1")
write = [Depends(require_api_key)]


def services(request: Request) -> Services:
    return request.app.state.services


@router.post("/batches", response_model=BatchOut, dependencies=write, status_code=status.HTTP_201_CREATED)
def ingest_batch(body: BatchIn, svc: Services = Depends(services)) -> BatchOut:
    """Score, route and monitor one batch of applications."""
    return svc.ingest(body)


@router.post("/batches/{batch_id}/labels", response_model=LabelsOut, dependencies=write)
def ingest_labels(
    body: LabelsIn, batch_id: str = Path(pattern=IDENTIFIER), svc: Services = Depends(services)
) -> LabelsOut:
    """Outcomes for every application of an earlier batch."""
    return svc.ingest_labels(batch_id, body)


@router.get("/timeline")
def timeline(limit: int = Query(default=300, ge=1, le=5000), svc: Services = Depends(services)) -> list[dict[str, Any]]:
    """Per-batch drift statistics, estimates and realised metrics, oldest first."""
    with svc.lock:
        return svc.store.timeline(limit)


@router.get("/alerts", response_model=list[AlertOut])
def alerts(
    status_filter: str | None = Query(default=None, alias="status", pattern="^(open|acknowledged)$"),
    limit: int = Query(default=100, ge=1, le=1000),
    svc: Services = Depends(services),
) -> list[AlertOut]:
    with svc.lock:
        rows = svc.store.alerts(status_filter, limit)
    return [AlertOut(**{**r, "detectors": r["detectors"].split(",") if r["detectors"] else []}) for r in rows]


@router.post("/alerts/{alert_id}/ack", dependencies=write)
def acknowledge(alert_id: int = Path(ge=1), svc: Services = Depends(services)) -> dict[str, Any]:
    with svc.lock:
        if not svc.store.set_alert_status(alert_id, "acknowledged"):
            raise HTTPException(status_code=404, detail=f"no alert {alert_id}")
    svc.refresh_gauges()
    return {"id": alert_id, "status": "acknowledged"}


@router.get("/registry/models", response_model=list[ModelOut])
def models(svc: Services = Depends(services)) -> list[ModelOut]:
    return svc.models()


@router.get("/registry/models/{version}", response_model=ModelDetail)
def model(version: int = Path(ge=1), svc: Services = Depends(services)) -> ModelDetail:
    return svc.model_detail(version)


@router.get("/rollout")
def rollout(svc: Services = Depends(services)) -> dict[str, Any]:
    state = svc.state()
    with svc.lock:
        state["history"] = svc.store.rollouts(20)
    return state


@router.post("/rollout/{version}/start", dependencies=write)
def start_rollout(version: int = Path(ge=1), svc: Services = Depends(services)) -> dict[str, Any]:
    """Put a registered version into shadow against the champion."""
    with svc.lock:
        svc.loop.start_rollout(version)
    svc.refresh_gauges()
    return svc.state()


@router.post("/rollout/abort", dependencies=write)
def abort_rollout(body: ReasonIn, svc: Services = Depends(services)) -> dict[str, Any]:
    with svc.lock:
        svc.loop.abort_rollout(body.reason)
    svc.refresh_gauges()
    return svc.state()


@router.post("/retrain", response_model=RetrainOut, dependencies=write)
def retrain(body: ReasonIn, svc: Services = Depends(services)) -> RetrainOut:
    """Train challengers on the latest labelled window and gate them now."""
    with svc.lock:
        outcome = svc.loop.retrain(f"manual: {body.reason}")
    if outcome is None:
        raise HTTPException(status_code=409, detail="not enough labelled, quality-clean rows to retrain")
    svc.refresh_gauges()
    out = retrain_out(outcome)
    assert out is not None
    return out


@router.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    """The simulated scenarios ``driftwatch replay`` can stream."""
    return [{"name": s.name, "family": s.family, "description": s.description} for s in SCENARIOS.values()]
