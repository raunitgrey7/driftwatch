"""Detection and estimation benchmark: every detector on control and drifted streams.

Scoring rules (docs/EVALUATION.md has the full definitions):

* a drift event is detected when an alert opens in ``[onset, onset + window)``;
  the delay is the alert batch minus the onset batch;
* an alert before the onset is a false alarm, and so is every alert on a
  control stream; precision per drift type is detected events over detected
  events plus the false alarms on that type's streams;
* the first alert's category is checked against the categories that name the
  mechanism (the diagnosis rate);
* raw detectors are scored the same way on their own fired signals, without
  persistence or cooldown, to show what the alert policy contributes;
* label-free estimates are compared per batch with the realised value (noisy
  at 400 rows) and with the value expected under the true PD (noise-free),
  so estimator bias is not hidden under sampling noise.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from driftwatch.alerts import Category
from driftwatch.data.scenarios import SCENARIOS, CreditStream, StreamBatch, StreamConfig
from driftwatch.detectors.estimation import em_prior
from driftwatch.monitor import BatchReport, Monitor
from driftwatch.stats import auc, expected_auc
from driftwatch.training import Champion, HistoryConfig, build_champion

EXPECTED_CATEGORIES: dict[str, frozenset[str]] = {
    "covariate": frozenset({Category.DATA_DRIFT, Category.PREDICTION_DRIFT, Category.ESTIMATED_PERFORMANCE}),
    "prior": frozenset({Category.LABEL_DRIFT, Category.PERFORMANCE, Category.PREDICTION_DRIFT}),
    "concept": frozenset({Category.PERFORMANCE, Category.LABEL_DRIFT}),
    "quality": frozenset({Category.SCHEMA, Category.DATA_QUALITY}),
}
NEVER = 10**9


@cache
def champion_for(history: HistoryConfig) -> Champion:
    """The reference model, trained once per worker process."""
    return build_champion(history)


@dataclass(frozen=True)
class StreamTask:
    scenario: str
    seed: int
    n_batches: int
    onset: int
    batch_size: int
    label_delay: int
    history: HistoryConfig


@dataclass
class StreamSummary:
    task: StreamTask
    alerts: list[tuple[int, str, str]] = field(default_factory=list)
    raw_first: dict[str, int] = field(default_factory=dict)
    raw_pre: dict[str, int] = field(default_factory=dict)
    errors: dict[str, list[float]] = field(default_factory=dict)
    seconds: float = 0.0

    def record_fired(self, index: int, detectors: set[str]) -> None:
        for name in detectors:
            if index < self.task.onset:
                self.raw_pre[name] = self.raw_pre.get(name, 0) + 1
            elif name not in self.raw_first:
                self.raw_first[name] = index

    def add_error(self, index: int, values: dict[str, float]) -> None:
        phase = "pre" if index < self.task.onset else "post"
        for key, value in values.items():
            if math.isfinite(value):
                slot = self.errors.setdefault(f"{phase}:{key}", [0.0, 0.0])
                slot[0] += value
                slot[1] += 1.0


@dataclass(frozen=True)
class Baselines:
    """Reference-period values the estimates are judged against."""

    naive_auc: float
    ref_cbpe: float
    ref_expected: float
    train_prior: float

    @classmethod
    def of(cls, champion: Champion) -> Baselines:
        reference = champion.calib.batch, champion.holdout.batch
        pd = np.concatenate([champion.model.predict_pd(b) for b in reference])
        true_pd = np.concatenate([champion.calib.true_pd, champion.holdout.true_pd])
        naive = auc(champion.model.predict_pd(champion.holdout.batch), champion.holdout.labels)
        return cls(naive, expected_auc(pd, pd), expected_auc(pd, true_pd), champion.reference.default_rate)


def batch_errors(report: BatchReport, batch: StreamBatch, base: Baselines) -> dict[str, float]:
    """Absolute, signed and change-tracking errors of the label-free estimates for one batch.

    The change-tracking error compares the estimated change of AUC since the
    reference period with the true change; a constant level bias of the
    estimator cancels in it, as it does in the alarm, which compares CBPE
    with its own reference band.
    """
    realized = auc(report.scores, batch.labels)
    expected = expected_auc(report.scores, batch.true_pd)
    cbpe = report.estimate.auc
    mean_pd = report.estimate.default_rate
    em = em_prior(report.scores, base.train_prior)
    rate, true_rate = float(batch.labels.mean()), float(batch.true_pd.mean())
    true_change = expected - base.ref_expected
    return {
        "auc_cbpe_vs_realized": abs(cbpe - realized),
        "auc_naive_vs_realized": abs(base.naive_auc - realized),
        "auc_cbpe_vs_expected": abs(cbpe - expected),
        "auc_naive_vs_expected": abs(base.naive_auc - expected),
        "auc_cbpe_bias": cbpe - expected,
        "auc_change_cbpe": abs((cbpe - base.ref_cbpe) - true_change),
        "auc_change_naive": abs(true_change),
        "rate_meanpd_vs_true": abs(mean_pd - true_rate),
        "rate_em_vs_true": abs(em - true_rate),
        "rate_meanpd_vs_realized": abs(mean_pd - rate),
        "rate_em_vs_realized": abs(em - rate),
    }


def run_stream(task: StreamTask) -> StreamSummary:
    """Monitor one stream end to end, delivering labels ``label_delay`` batches late."""
    started = time.perf_counter()
    champion = champion_for(task.history)
    base = Baselines.of(champion)
    monitor = Monitor(champion.model, champion.reference)
    stream = CreditStream(StreamConfig(task.scenario, task.seed, task.n_batches, task.batch_size, task.onset))
    summary = StreamSummary(task)
    pending: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for batch in stream:
        report = monitor.observe(batch.index, batch.raw, batch.timestamp)
        fired = {s.detector for s in report.signals if s.fired}
        alerts = list(report.alerts)
        pending[batch.index] = (batch.labels, report.scores)
        if batch.index >= task.label_delay:
            labels, scores = pending.pop(batch.index - task.label_delay)
            label_report = monitor.observe_labels(batch.index, batch.index - task.label_delay, labels, scores)
            fired |= {s.detector for s in label_report.signals if s.fired}
            alerts += label_report.alerts
        summary.record_fired(batch.index, fired)
        summary.alerts += [(a.batch_index, a.category.value, a.scope) for a in alerts]
        summary.add_error(batch.index, batch_errors(report, batch, base))
    summary.seconds = time.perf_counter() - started
    return summary


def _percentile(values: list[int], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _event_metrics(summaries: list[StreamSummary], window: int, family: str) -> dict[str, object]:
    """Policy-level recall, precision, delay and diagnosis for one drift type."""
    delays: list[int] = []
    false_alarms = 0
    diagnosed = 0
    for s in summaries:
        onset = s.task.onset
        false_alarms += sum(1 for a in s.alerts if a[0] < onset)
        after = sorted(a for a in s.alerts if onset <= a[0] < onset + window)
        if after:
            delays.append(after[0][0] - onset)
            diagnosed += after[0][1] in EXPECTED_CATEGORIES[family]
    n = len(summaries)
    detected = len(delays)
    return {
        "streams": n,
        "recall": detected / n if n else 0.0,
        "precision": detected / (detected + false_alarms) if detected + false_alarms else 0.0,
        "false_alarms_before_onset": false_alarms,
        "delay_median": _percentile(delays, 50),
        "delay_p90": _percentile(delays, 90),
        "diagnosis_rate": diagnosed / detected if detected else 0.0,
    }


def _raw_metrics(summaries: list[StreamSummary], window: int) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    names = sorted({k for s in summaries for k in (*s.raw_first, *s.raw_pre)})
    for name in names:
        delays = [
            s.raw_first[name] - s.task.onset for s in summaries if s.raw_first.get(name, NEVER) < s.task.onset + window
        ]
        out[name] = {"recall": len(delays) / len(summaries), "delay_median": _percentile(delays, 50)}
    return out


def _estimation(summaries: list[StreamSummary], phase: str) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for s in summaries:
        for key, (total, count) in s.errors.items():
            if key.startswith(f"{phase}:"):
                totals[key.split(":", 1)[1]][0] += total
                totals[key.split(":", 1)[1]][1] += count
    return {k: round(v[0] / v[1], 6) for k, v in sorted(totals.items()) if v[1]}


def control_metrics(summaries: list[StreamSummary]) -> dict[str, object]:
    """False alarms on drift-free streams, per 10,000 batches, for the policy and each raw detector."""
    batches = sum(s.task.n_batches for s in summaries)
    alerts = sum(len(s.alerts) for s in summaries)
    raw = defaultdict(int)
    for s in summaries:
        for name, count in s.raw_pre.items():
            raw[name] += count
    by_category: dict[str, int] = defaultdict(int)
    for s in summaries:
        for _, category, _ in s.alerts:
            by_category[category] += 1
    return {
        "streams": len(summaries),
        "batches": batches,
        "alerts": alerts,
        "false_alarms_per_10k": round(alerts / batches * 10_000, 3),
        "alerts_by_category": dict(sorted(by_category.items())),
        "raw_fires_per_10k": {k: round(v / batches * 10_000, 3) for k, v in sorted(raw.items())},
        "estimation": _estimation(summaries, "pre"),
    }


def drift_metrics(summaries: list[StreamSummary], window: int) -> dict[str, dict[str, object]]:
    """Per drift type: policy metrics, raw detector metrics and post-onset estimation errors."""
    grouped: dict[str, list[StreamSummary]] = defaultdict(list)
    for s in summaries:
        grouped[s.task.scenario].append(s)
    out: dict[str, dict[str, object]] = {}
    for name, items in sorted(grouped.items()):
        family = SCENARIOS[name].family
        out[name] = {
            "family": family,
            **_event_metrics(items, window, family),
            "raw": _raw_metrics(items, window),
            "estimation_post_onset": _estimation(items, "post"),
        }
    return out
