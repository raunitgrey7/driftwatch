"""The control loop: score, monitor, roll out, retrain.

One ``ControlLoop`` drives one deployment slot: exactly one production model
and at most one candidate in rollout. The API and the deployment benchmarks
drive this same object, so the measured rollback and retraining behaviour is
the behaviour of the service.

* ``ingest`` scores a batch with the champion (and the candidate during a
  rollout), runs the monitor, applies the routed decisions and stores the
  rows for later retraining.
* ``ingest_labels`` joins outcomes to the scores used at decision time,
  feeds the label-based detectors and the rollout guardrails, and may
  promote or roll back the candidate.
* Either call may start a retrain when the trigger is due; a challenger that
  passes the gate goes to shadow automatically, one that fails is recorded
  as rejected, with its lineage, in the registry.

Detector windows live in memory and restart with the process; everything an
operator needs afterwards (registry, alerts, batch summaries, labels) is in
SQLite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from driftwatch.alerts import Category
from driftwatch.deploy.retrain import (
    GateDecision,
    RetrainConfig,
    RetrainTrigger,
    champion_challenger_gate,
    recalibrate,
    split_window,
)
from driftwatch.deploy.rollout import Guardrail, Rollout, RolloutConfig, RolloutStatus
from driftwatch.model import CreditModel, Scorer, approve
from driftwatch.monitor import BatchReport, LabelReport, Monitor, MonitorConfig, Reference
from driftwatch.registry import ModelRegistry, Stage
from driftwatch.schema import coerce, concat_batches
from driftwatch.stats import auc
from driftwatch.store import Store, pack
from driftwatch.training import HistoryConfig, LabeledData, build_champion, fit_on

log = logging.getLogger("driftwatch.loop")
QUALITY = (Category.SCHEMA, Category.DATA_QUALITY)


@dataclass
class ModelBundle:
    """What the registry stores per version: the scorer and its monitoring reference."""

    model: Scorer
    reference: Reference


@dataclass(frozen=True)
class LoopConfig:
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    auto_retrain: bool = True
    min_retrain_rows: int = 4000
    seed: int = 0


@dataclass(frozen=True)
class RetrainOutcome:
    version: int
    decision: GateDecision
    window: tuple[int, int]
    n_rows: int
    at: int
    challenger: str = "refit"


@dataclass
class _Scored:
    champion_version: int
    champion_pd: np.ndarray
    candidate_version: int | None
    candidate_pd: np.ndarray | None


@dataclass
class IngestResult:
    index: int
    batch_id: str
    champion_version: int
    candidate_version: int | None
    pd: np.ndarray
    champion_pd: np.ndarray
    decisions: np.ndarray
    served_by_candidate: np.ndarray
    report: BatchReport
    rollout_action: str
    retrain: RetrainOutcome | None


@dataclass
class LabelResult:
    index: int
    realized_auc: float
    report: LabelReport | None
    rollout_action: str
    retrain: RetrainOutcome | None


def safe_predict(model: Scorer, batch: Any) -> np.ndarray:
    """Candidate scores; an exception counts as every row failing, never as a crash of the service."""
    try:
        return np.asarray(model.predict_pd(batch), dtype=np.float64)
    except Exception:  # a faulty release must not take the decision path down
        log.exception("candidate failed to score a batch")
        return np.full(batch.n_rows, np.nan)


class ControlLoop:
    """Stateful runtime around a registry and a store."""

    def __init__(self, registry: ModelRegistry, store: Store, config: LoopConfig | None = None) -> None:
        self.registry, self.store = registry, store
        self.config = config or LoopConfig()
        production = registry.production()
        if production is None:
            raise RuntimeError("the registry has no production model; use ControlLoop.bootstrap")
        self._set_champion(production.version)
        self.rollout: Rollout | None = None
        self.candidate_version: int | None = None
        self.candidate: ModelBundle | None = None
        self.last_guardrails: list[Guardrail] = []
        self._rollout_id: int | None = None
        self.trigger = RetrainTrigger(self.config.retrain)
        self._scores: dict[int, _Scored] = {}
        self.now = store.next_index() - 1
        self.retrains: list[RetrainOutcome] = []

    @classmethod
    def bootstrap(
        cls,
        registry: ModelRegistry,
        store: Store,
        history: HistoryConfig | None = None,
        config: LoopConfig | None = None,
    ) -> ControlLoop:
        """Train and register the reference model when the registry is empty, then start."""
        cfg = config or LoopConfig()
        if registry.production() is None:
            h = history or HistoryConfig()
            champion = build_champion(h, cfg.monitor)
            lineage = {
                "source": "clean history",
                "history_seed": h.seed,
                "train_batches": h.train_batches,
                "rows": champion.train.batch.n_rows,
            }
            metrics = {
                "holdout_auc": round(auc(champion.model.predict_pd(champion.holdout.batch), champion.holdout.labels), 6)
            }
            record = registry.register(
                ModelBundle(champion.model, champion.reference),
                lineage=lineage,
                metrics=metrics,
                reason="reference model",
            )
            registry.transition(record.version, Stage.PRODUCTION, "bootstrap: first production model")
        return cls(registry, store, cfg)

    def _set_champion(self, version: int) -> None:
        bundle: ModelBundle = self.registry.load(version)
        self.champion_version, self.champion = version, bundle
        self.monitor = Monitor(bundle.model, bundle.reference, self.config.monitor)

    @property
    def rollout_active(self) -> bool:
        return self.rollout is not None and self.rollout.status is RolloutStatus.RUNNING

    def ingest(self, raw: dict[str, Any], batch_id: str | None = None, timestamp: str = "") -> IngestResult:
        """Score, route, monitor and store one batch of applications."""
        index = self.store.next_index()
        batch_id = batch_id or f"batch-{index}"
        if self.store.index_of(batch_id) is not None:
            raise ValueError(f"batch {batch_id!r} was already ingested")
        self.now = index
        batch, issues = coerce(raw)
        champ_pd = self.champion.model.predict_pd(batch)
        report = self.monitor.observe(index, batch, timestamp, scores=champ_pd, issues=issues)
        decisions, served = approve(champ_pd, self.champion.model.threshold), np.zeros(batch.n_rows, dtype=bool)
        cand_pd, action, cand_version = None, "none", self.candidate_version
        if self.rollout_active and self.candidate is not None and self.rollout is not None:
            cand_pd = safe_predict(self.candidate.model, batch)
            outcome = self.rollout.observe_batch(batch.ids, champ_pd, cand_pd)
            decisions, served, action = outcome.decisions, outcome.served_by_candidate, outcome.action
            self.last_guardrails = outcome.guardrails
            self._after_rollout(action)
        applied = np.where(served, cand_pd, champ_pd) if cand_pd is not None else champ_pd
        self.store.add_alerts(report.alerts)
        self.trigger.observe(index, report.alerts)
        quality_flag = any(s.fired and s.category in QUALITY for s in report.signals)
        arrays = {"champion_pd": champ_pd} | ({"candidate_pd": cand_pd} if cand_pd is not None else {})
        summary = report.summary() | {"rollout_action": action, "served_by_candidate": int(served.sum())}
        self.store.add_batch(
            index, batch_id, timestamp, self.champion_version, cand_version, quality_flag, summary, pack(batch, arrays)
        )
        self._scores[index] = _Scored(self.champion_version, champ_pd, cand_version, cand_pd)
        return IngestResult(
            index,
            batch_id,
            self.champion_version,
            cand_version,
            applied,
            champ_pd,
            decisions,
            served,
            report,
            action,
            self._maybe_retrain(),
        )

    def _scored(self, index: int) -> _Scored:
        cached = self._scores.pop(index, None)
        if cached is not None:
            return cached
        loaded = self.store.payload(index)
        if loaded is None:
            raise KeyError(index)
        champion, candidate = self.store.versions(index)
        arrays = loaded[1]
        return _Scored(champion, arrays["champion_pd"], candidate, arrays.get("candidate_pd"))

    def ingest_labels(self, batch_id: str, labels: np.ndarray) -> LabelResult:
        """Join outcomes to the batch's decision-time scores and run every label-based check."""
        index = self.store.index_of(batch_id)
        if index is None:
            raise KeyError(f"unknown batch {batch_id!r}")
        if self.store.has_labels(index):
            raise ValueError(f"labels for {batch_id!r} were already received")
        scored = self._scored(index)
        y = np.asarray(labels, dtype=np.int8)
        if len(y) != len(scored.champion_pd):
            raise ValueError(f"expected {len(scored.champion_pd)} labels, got {len(y)}")
        self.store.set_labels(index, y)
        report = None
        own = scored.champion_pd if scored.champion_version == self.champion_version else None
        if own is None and scored.candidate_version == self.champion_version:
            own = scored.candidate_pd
        realized = auc(scored.champion_pd, y)
        if own is not None:
            report = self.monitor.observe_labels(self.now, index, y, own)
            self.store.add_alerts(report.alerts)
        errors = approve(scored.champion_pd, self.champion.model.threshold) == (y == 1)
        brier = float(np.mean((scored.champion_pd - y) ** 2))
        self.store.add_label_report(index, self.now, len(y), realized, float(y.mean()), brier, float(errors.mean()))
        self.trigger.observe(self.now, report.alerts if report else [], labelled_index=index)
        action = "none"
        rollout = self.rollout
        if (
            self.rollout_active
            and rollout is not None
            and scored.candidate_pd is not None
            and scored.candidate_version == self.candidate_version
        ):
            self.last_guardrails, action = rollout.observe_labels(y, scored.champion_pd, scored.candidate_pd)
            self._after_rollout(action)
        return LabelResult(index, realized, report, action, self._maybe_retrain())

    def _after_rollout(self, action: str) -> None:
        rollout, version = self.rollout, self.candidate_version
        if rollout is None or version is None:
            return
        if action == "advance" and self.registry.get(version).stage is Stage.SHADOW:
            self.registry.transition(version, Stage.CANARY, f"advanced to {rollout.step_name}")
        elif action == "rollback":
            self.registry.rollback(version, rollout.reason)
            log.warning("rolled back", extra={"version": version, "reason": rollout.reason})
        elif action == "promote":
            self.registry.promote(version, rollout.reason)
            self._set_champion(version)
            self.trigger.reset()
            log.info("promoted", extra={"version": version, "reason": rollout.reason})
        self._rollout_id = self.store.save_rollout(
            self._rollout_id,
            version,
            self.champion_version,
            self.now,
            rollout.status.value,
            rollout.reason,
            rollout.exposure,
            rollout.history,
        )
        if rollout.status is not RolloutStatus.RUNNING:
            self.candidate, self.candidate_version, self._rollout_id = None, None, None

    def start_rollout(self, version: int) -> Rollout:
        """Put a registered version into shadow against the current champion."""
        if self.rollout_active:
            raise RuntimeError("a rollout is already running")
        bundle: ModelBundle = self.registry.load(version)
        self.registry.transition(version, Stage.SHADOW, "rollout started")
        self.candidate_version, self.candidate = version, bundle
        self.rollout = Rollout(version, self.champion.model.threshold, bundle.model.threshold, self.config.rollout)
        self._rollout_id = self.store.save_rollout(None, version, self.champion_version, self.now, "running", "", 0, [])
        return self.rollout

    def abort_rollout(self, reason: str) -> None:
        """Operator rollback of the running candidate."""
        if not self.rollout_active or self.rollout is None:
            raise RuntimeError("no rollout is running")
        self.rollout.status, self.rollout.reason = RolloutStatus.ROLLED_BACK, reason
        self._after_rollout("rollback")

    def _maybe_retrain(self) -> RetrainOutcome | None:
        if not self.config.auto_retrain or self.rollout_active or not self.trigger.due(self.now):
            return None
        return self.retrain(f"trigger evidence at batch {self.trigger.evidence}", self.trigger.window())

    def _challengers(self, train: LabeledData, calib: LabeledData) -> list[tuple[str, Scorer]]:
        """A full refit, plus the recalibrated champion when the champion is a booster."""
        cfg = self.config
        out: list[tuple[str, Scorer]] = [("refit", fit_on(train, calib, seed=cfg.seed + len(self.registry.list())))]
        if isinstance(self.champion.model, CreditModel):
            recent = LabeledData(
                concat_batches([train.batch, calib.batch]), np.concatenate([train.labels, calib.labels])
            )
            out.append(("recalibrate", recalibrate(self.champion.model, recent, cfg.retrain.approval_rate)))
        return out

    def retrain(self, reason: str, window: tuple[int, int] | None = None) -> RetrainOutcome | None:
        """Train challengers on a labelled window, gate them, register the best, roll it out if it passed."""
        cfg = self.config
        latest = self.trigger.latest_labelled
        start, end = window or (latest - cfg.retrain.window_batches + 1, latest)
        data = self.store.labelled_window(start, end)
        if data is None or data.batch.n_rows < cfg.min_retrain_rows:
            return None
        train, calib, holdout = split_window(data, cfg.retrain)
        challengers = self._challengers(train, calib)
        gate_config = replace(cfg.retrain, alpha=cfg.retrain.alpha / len(challengers))
        judged = [
            (kind, model, champion_challenger_gate(self.champion.model, model, holdout, gate_config))
            for kind, model in challengers
        ]
        passing = [j for j in judged if j[2].promote]
        kind, challenger, decision = min(passing or judged, key=lambda j: j[2].logloss_challenger)
        recent = LabeledData(
            concat_batches([calib.batch, holdout.batch]), np.concatenate([calib.labels, holdout.labels])
        )
        reference = Reference.build(challenger, recent.batch, recent.labels, train.batch, cfg.monitor, seed=cfg.seed)
        lineage = {
            "window": [start, end],
            "rows": data.batch.n_rows,
            "train_rows": train.batch.n_rows,
            "challenger": kind,
            "evidence": self.trigger.evidence,
            "trigger": reason,
            "alternatives": {k: d.to_dict() for k, _, d in judged},
        }
        record = self.registry.register(
            ModelBundle(challenger, reference),
            parent=self.champion_version,
            lineage=lineage,
            metrics=decision.to_dict(),
            reason=reason,
        )
        self.trigger.reset(self.now, rejected=not decision.promote)
        if decision.promote:
            self.start_rollout(record.version)
        else:
            self.registry.transition(record.version, Stage.REJECTED, decision.reason)
        outcome = RetrainOutcome(record.version, decision, (start, end), data.batch.n_rows, self.now, kind)
        self.retrains.append(outcome)
        return outcome

    def state(self) -> dict[str, Any]:
        """Snapshot for the API and dashboard."""
        rollout = None
        if self.rollout is not None:
            r = self.rollout
            rollout = {
                "candidate_version": r.candidate_version,
                "status": r.status.value,
                "step": r.step_name,
                "traffic": r.traffic,
                "batches": r.batches,
                "labelled_batches": r.labeled_batches,
                "exposure": r.exposure,
                "reason": r.reason,
                "guardrails": [g.__dict__ for g in self.last_guardrails],
                "history": r.history,
            }
        trigger = {
            "evidence": self.trigger.evidence,
            "latest_labelled": self.trigger.latest_labelled,
            "blocked": self.trigger.blocked(self.now),
        }
        return {
            "champion_version": self.champion_version,
            "now": self.now,
            "rollout": rollout,
            "retrain_trigger": trigger,
            **self.store.counts(),
        }
