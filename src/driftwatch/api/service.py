"""The service layer: one control loop behind a lock, plus metrics bookkeeping.

The control loop is stateful and not thread-safe, while FastAPI runs sync
endpoints on a thread pool, so every call that touches the loop holds one
lock. Translating loop results into response bodies and Prometheus counters
lives here so the route functions stay declarative.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np

from driftwatch.alerts import Alert
from driftwatch.api.metrics import MetricsRegistry
from driftwatch.api.schemas import (
    AlertOut,
    BatchIn,
    BatchOut,
    LabelsIn,
    LabelsOut,
    ModelDetail,
    ModelOut,
    RetrainOut,
    RowDecision,
)
from driftwatch.api.settings import Settings
from driftwatch.loop import ControlLoop, IngestResult, InvalidInputError, LoopConfig, NotFoundError, RetrainOutcome
from driftwatch.registry import ModelRecord, ModelRegistry
from driftwatch.schema import records_to_columns
from driftwatch.store import Store
from driftwatch.training import HistoryConfig


def _finite(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


def alert_out(alert: Alert) -> AlertOut:
    return AlertOut(
        batch_index=alert.batch_index,
        ts=alert.timestamp,
        category=alert.category.value,
        scope=alert.scope,
        severity=alert.severity,
        detectors=list(alert.detectors),
        message=alert.message,
    )


def retrain_out(outcome: RetrainOutcome | None) -> RetrainOut | None:
    if outcome is None:
        return None
    d = outcome.decision
    return RetrainOut(
        version=outcome.version,
        challenger=outcome.challenger,
        promote=d.promote,
        reason=d.reason,
        window=list(outcome.window),
        n_rows=outcome.n_rows,
        gate=d.to_dict(),
    )


def model_out(record: ModelRecord) -> ModelOut:
    return ModelOut(**{**record.to_dict(), "stage": record.stage.value})


class Services:
    """Registry, store and loop for one deployment slot."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.started_at = time.time()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry(settings.data_dir / "registry")
        self.store = Store(settings.data_dir / "driftwatch.db")
        config = LoopConfig(auto_retrain=settings.auto_retrain)
        self.loop = ControlLoop.bootstrap(self.registry, self.store, HistoryConfig(seed=settings.history_seed), config)
        self.metrics = MetricsRegistry()
        self.lock = threading.Lock()
        self.refresh_gauges()

    def close(self) -> None:
        self.store.close()
        self.registry.close()

    def refresh_gauges(self) -> None:
        state = self.loop.state()
        with self.metrics.lock:
            self.metrics.champion_version.set(state["champion_version"])
            self.metrics.open_alerts.set(state["open_alerts"])
            rollout = state["rollout"]
            self.metrics.rollout_traffic.set(rollout["traffic"] if rollout and rollout["status"] == "running" else 0.0)

    def _count(self, alerts: list[Alert], action: str, retrain: RetrainOutcome | None) -> None:
        with self.metrics.lock:
            for alert in alerts:
                self.metrics.alerts.inc(category=alert.category.value, severity=alert.severity)
            if action in ("promote", "rollback"):
                self.metrics.rollout_events.inc(event=action)
            if retrain is not None:
                self.metrics.retrains.inc(outcome="promoted_to_shadow" if retrain.decision.promote else "rejected")
                if retrain.decision.promote:
                    self.metrics.rollout_events.inc(event="start")

    def _batch_out(self, result: IngestResult, include_rows: bool) -> BatchOut:
        report = result.report
        decisions = None
        if include_rows:
            served = result.served_by_candidate
            decisions = [
                RowDecision(
                    application_id=str(i),
                    pd=_finite(float(p)),
                    approve=bool(a),
                    decided_by=int(
                        result.candidate_version if s and result.candidate_version else result.champion_version
                    ),
                )
                for i, p, a, s in zip(report_ids(result), result.pd, result.decisions, served, strict=True)
            ]
        return BatchOut(
            batch_index=result.index,
            batch_id=result.batch_id,
            n_rows=report.n_rows,
            champion_version=result.champion_version,
            candidate_version=result.candidate_version,
            approval_rate=float(np.mean(result.decisions)),
            estimated_auc=_finite(report.estimate.auc),
            estimated_default_rate=_finite(report.estimate.default_rate),
            alerts=[alert_out(a) for a in report.alerts],
            rollout_action=result.rollout_action,
            retrain=retrain_out(result.retrain),
            decisions=decisions,
        )

    def ingest(self, body: BatchIn) -> BatchOut:
        if len(body.rows) > self.settings.max_rows:
            raise InvalidInputError(f"at most {self.settings.max_rows} rows per batch")
        timestamp = body.timestamp or datetime.now(UTC).isoformat(timespec="seconds")
        with self.lock:
            result = self.loop.ingest(records_to_columns(body.rows), body.batch_id, timestamp)
            self._count(result.report.alerts, result.rollout_action, result.retrain)
            with self.metrics.lock:
                self.metrics.batches.inc()
                self.metrics.rows.inc(
                    result.report.n_rows - int(result.served_by_candidate.sum()), version=str(result.champion_version)
                )
                if result.candidate_version is not None:
                    self.metrics.rows.inc(int(result.served_by_candidate.sum()), version=str(result.candidate_version))
                if math.isfinite(result.report.estimate.auc):
                    self.metrics.estimated_auc.set(result.report.estimate.auc)
                self.metrics.approval_rate.set(float(np.mean(result.decisions)))
            self.refresh_gauges()
        return self._batch_out(result, body.include_rows)

    def ingest_labels(self, batch_id: str, body: LabelsIn) -> LabelsOut:
        with self.lock:
            index = self.store.index_of(batch_id)
            loaded = self.store.payload(index) if index is not None else None
            if index is None or loaded is None:
                raise NotFoundError(f"unknown batch {batch_id!r}")
            ids = [str(i) for i in loaded[0].ids.tolist()]
            given = {label.application_id: int(label.defaulted) for label in body.labels}
            missing = [i for i in ids if i not in given]
            if missing or len(given) != len(ids):
                raise InvalidInputError(
                    f"labels must cover exactly the {len(ids)} applications of the batch; {len(missing)} missing"
                )
            result = self.loop.ingest_labels(batch_id, np.array([given[i] for i in ids], dtype=np.int8))
            alerts = result.report.alerts if result.report else []
            self._count(alerts, result.rollout_action, result.retrain)
            with self.metrics.lock:
                self.metrics.label_batches.inc()
                if math.isfinite(result.realized_auc):
                    self.metrics.realized_auc.set(result.realized_auc)
            self.refresh_gauges()
        labels = np.array([given[i] for i in ids])
        return LabelsOut(
            batch_id=batch_id,
            batch_index=result.index,
            n_rows=len(ids),
            realized_auc=_finite(result.realized_auc),
            default_rate=float(labels.mean()),
            alerts=[alert_out(a) for a in alerts],
            rollout_action=result.rollout_action,
            retrain=retrain_out(result.retrain),
        )

    def model_detail(self, version: int) -> ModelDetail:
        with self.lock:
            record = self.registry.get(version)
            chain = [r.version for r in self.registry.lineage(version)]
            events = self.registry.events(version)
        return ModelDetail(**model_out(record).model_dump(), lineage_chain=chain, events=events)

    def models(self) -> list[ModelOut]:
        with self.lock:
            return [model_out(r) for r in self.registry.list()]

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self.loop.state()


def report_ids(result: IngestResult) -> list[str]:
    """Application ids in row order, as the monitor coerced them."""
    return [str(i) for i in result.ids.tolist()]
