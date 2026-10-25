"""Retraining benchmark: the trigger and the champion-challenger gate on drifted streams.

Each run is the full control loop with automatic retraining on one scenario
stream. Every gate decision is checked against ground truth: the expected
log loss and AUC of the champion and the challenger under the true PD, on a
fresh sample of ``truth_batches`` batches from the distribution in force at
decision time. The challenger should be promoted when it is truly better
(log loss lower by more than 0.001 without losing more than 0.005 AUC); a
promotion is acceptable when it is at least not worse (log loss within
0.0005). Control streams also get one forced retrain, which the gate should
reject because a challenger trained on less data of the same distribution
is not better.

Recovery is measured on the batches after the first retrained version
reaches production: the noise-free AUC and log loss of the scores actually
used for the decisions against the original champion's on the same rows.
AUC shows recovered ranking (concept drift); log loss also shows recovered
calibration, which is all a recalibration can give under prior shift.
"""

from __future__ import annotations

import math
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from driftwatch.bench.detection import champion_for
from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.deploy.retrain import row_log_loss
from driftwatch.loop import ControlLoop, LoopConfig, ModelBundle, RetrainOutcome
from driftwatch.registry import ModelRegistry, Stage
from driftwatch.schema import coerce
from driftwatch.stats import expected_auc
from driftwatch.store import Store
from driftwatch.training import HistoryConfig, LabeledData

SHOULD_GAIN = 0.001
ACCEPT_LOSS = 0.0005
AUC_SLACK = 0.005


@dataclass(frozen=True)
class RetrainTask:
    scenario: str
    seed: int
    n_batches: int
    onset: int
    batch_size: int
    label_delay: int
    history: HistoryConfig
    forced_at: int | None
    truth_batches: int


@dataclass(frozen=True)
class GateRecord:
    at: int
    forced: bool
    promote: bool
    champion_logloss: float
    challenger_logloss: float
    champion_auc: float
    challenger_auc: float
    should_promote: bool
    correct: bool


@dataclass
class RetrainSummary:
    task: RetrainTask
    gates: list[GateRecord] = field(default_factory=list)
    promoted_at: int | None = None
    first_alert_at: int | None = None
    deployed_auc: list[float] = field(default_factory=list)
    champion_auc: list[float] = field(default_factory=list)
    deployed_loss: list[float] = field(default_factory=list)
    champion_loss: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def track(self, deployed_pd: np.ndarray, champion_pd: np.ndarray, true_pd: np.ndarray) -> None:
        """Noise-free AUC and log loss of the deployed scores and of the original champion."""
        self.deployed_auc.append(expected_auc(deployed_pd, true_pd))
        self.champion_auc.append(expected_auc(champion_pd, true_pd))
        self.deployed_loss.append(_expected_loss(deployed_pd, true_pd))
        self.champion_loss.append(_expected_loss(champion_pd, true_pd))


def _truth(task: RetrainTask, at: int) -> LabeledData:
    config = StreamConfig(task.scenario, task.seed + 50_000, at + task.truth_batches, task.batch_size, task.onset)
    stream = CreditStream(config)
    return LabeledData.from_stream([stream.batch(i) for i in range(at, at + task.truth_batches)])


def _expected_loss(pd: np.ndarray, true_pd: np.ndarray) -> float:
    """Log loss expected under the true PD: q * loss(y=1) + (1 - q) * loss(y=0)."""
    return float(
        np.mean(true_pd * row_log_loss(pd, np.ones_like(pd)) + (1.0 - true_pd) * row_log_loss(pd, np.zeros_like(pd)))
    )


def judge_gate(task: RetrainTask, registry: ModelRegistry, outcome: RetrainOutcome, forced: bool) -> GateRecord:
    """Compare the gate's decision with the truly better model at decision time."""
    record = registry.get(outcome.version)
    champion: ModelBundle = registry.load(record.parent_version or 1)
    challenger: ModelBundle = registry.load(outcome.version)
    truth = _truth(task, outcome.at)
    p_champ, p_chall = champion.model.predict_pd(truth.batch), challenger.model.predict_pd(truth.batch)
    ll_champ, ll_chall = _expected_loss(p_champ, truth.true_pd), _expected_loss(p_chall, truth.true_pd)
    auc_champ, auc_chall = expected_auc(p_champ, truth.true_pd), expected_auc(p_chall, truth.true_pd)
    should = ll_chall < ll_champ - SHOULD_GAIN and auc_chall > auc_champ - AUC_SLACK
    acceptable = ll_chall < ll_champ + ACCEPT_LOSS and auc_chall > auc_champ - AUC_SLACK
    promote = outcome.decision.promote
    correct = acceptable if promote else not should
    return GateRecord(outcome.at, forced, promote, ll_champ, ll_chall, auc_champ, auc_chall, should, correct)


def run_retrain(task: RetrainTask) -> RetrainSummary:
    """The full loop with automatic retraining on one stream."""
    started = time.perf_counter()
    champion = champion_for(task.history)
    summary = RetrainSummary(task)
    with tempfile.TemporaryDirectory() as tmp:
        registry, store = ModelRegistry(Path(tmp)), Store()
        first = registry.register(ModelBundle(champion.model, champion.reference), reason="champion")
        registry.transition(first.version, Stage.PRODUCTION, "benchmark champion")
        loop = ControlLoop(registry, store, LoopConfig(auto_retrain=True))
        stream = CreditStream(StreamConfig(task.scenario, task.seed, task.n_batches, task.batch_size, task.onset))
        labels: dict[int, np.ndarray] = {}
        for batch in stream:
            result = loop.ingest(batch.raw, f"b{batch.index}", batch.timestamp)
            labels[batch.index] = batch.labels
            outcomes: list[tuple[RetrainOutcome | None, bool]] = [(result.retrain, False)]
            alerts = list(result.report.alerts)
            if batch.index >= task.label_delay:
                label_result = loop.ingest_labels(
                    f"b{batch.index - task.label_delay}", labels.pop(batch.index - task.label_delay)
                )
                outcomes.append((label_result.retrain, False))
                alerts += label_result.report.alerts if label_result.report else []
            if task.forced_at == batch.index and not loop.rollout_active:
                outcomes.append((loop.retrain("forced retrain for the gate benchmark"), True))
            if summary.first_alert_at is None and batch.index >= task.onset and alerts:
                summary.first_alert_at = batch.index
            if summary.promoted_at is None and loop.champion_version != first.version:
                summary.promoted_at = batch.index
            if summary.promoted_at is not None:
                summary.track(result.pd, champion.model.predict_pd(coerce(batch.raw)[0]), batch.true_pd)
            summary.gates += [judge_gate(task, registry, o, forced) for o, forced in outcomes if o is not None]
        registry.close()
        store.close()
    summary.seconds = time.perf_counter() - started
    return summary


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 6) if values else None


def retrain_metrics(summaries: list[RetrainSummary]) -> dict[str, object]:
    """Per scenario and pooled gate accuracy."""
    grouped: dict[str, list[RetrainSummary]] = defaultdict(list)
    for s in summaries:
        grouped[s.task.scenario].append(s)
    per_scenario: dict[str, dict[str, object]] = {}
    for name, items in sorted(grouped.items()):
        triggered = [g for s in items for g in s.gates if not g.forced]
        forced = [g for s in items for g in s.gates if g.forced]
        promoted = [s for s in items if s.promoted_at is not None]
        lifts = [float(np.mean(s.deployed_auc) - np.mean(s.champion_auc)) for s in promoted if s.deployed_auc]
        gains = [float(np.mean(s.champion_loss) - np.mean(s.deployed_loss)) for s in promoted if s.deployed_loss]
        per_scenario[name] = {
            "runs": len(items),
            "triggered_retrains_per_run": len(triggered) / len(items),
            "gate_promote_rate": sum(g.promote for g in triggered) / len(triggered) if triggered else None,
            "gate_correct_rate": sum(g.correct for g in triggered) / len(triggered) if triggered else None,
            "forced_retrains": len(forced),
            "forced_promote_rate": sum(g.promote for g in forced) / len(forced) if forced else None,
            "forced_correct_rate": sum(g.correct for g in forced) / len(forced) if forced else None,
            "runs_with_new_production": len(promoted),
            "batches_onset_to_production_median": float(np.median([s.promoted_at - s.task.onset for s in promoted]))
            if promoted
            else None,
            "true_auc_lift_after_promotion_mean": _mean(lifts),
            "true_logloss_gain_after_promotion_mean": _mean(gains),
        }
    gates = [g for s in summaries for g in s.gates]
    pooled = {
        "gate_decisions": len(gates),
        "gate_correct_rate": sum(g.correct for g in gates) / len(gates) if gates else math.nan,
        "gate_promotions": sum(g.promote for g in gates),
        "gate_should_promote": sum(g.should_promote for g in gates),
    }
    return {"per_scenario": per_scenario, "pooled": pooled}
