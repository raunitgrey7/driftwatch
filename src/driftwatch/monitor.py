"""The monitor: every detector on every batch, then the alert policy.

A ``Reference`` is built once per model version from that version's clean
reference data (and its own scores on it), so when a new version is
promoted the monitor is rebuilt from the new lineage rather than patched.

Two entry points mirror how credit data arrives:

* ``observe`` runs at decision time on the unlabelled batch: schema,
  data quality, data drift (KS or chi-square per feature with a Bonferroni
  correction, PSI, MMD), prediction drift and the CBPE estimate;
* ``observe_labels`` runs when the outcomes of an earlier batch arrive:
  realised AUC, ADWIN on calibration residuals, DDM on decision errors and
  the label-rate test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from driftwatch.alerts import Alert, AlertPolicy, Category, Signal
from driftwatch.detectors.drift import MMDDetector, ScoreDrift, UnivariateDrift
from driftwatch.detectors.estimation import CBPEMonitor, PerformanceEstimate
from driftwatch.detectors.performance import DDM, BatchedADWIN, LabelRate
from driftwatch.detectors.quality import QualityConfig, QualityProfile, quality_signals, schema_signals
from driftwatch.model import Scorer, approve
from driftwatch.schema import Batch, SchemaIssue, coerce
from driftwatch.stats import auc


@dataclass(frozen=True)
class MonitorConfig:
    """Alarm thresholds; each is a false-alarm budget, documented in docs/EVALUATION.md."""

    drift_alpha: float = 0.001
    psi_alarm: float = 0.25
    mmd_alpha: float = 0.001
    score_alpha: float = 0.001
    cbpe_sigma: float = 3.0
    label_alpha: float = 0.001
    adwin_delta: float = 0.002
    ddm_drift_level: float = 3.0
    batch_size: int = 400
    quality: QualityConfig = field(default_factory=QualityConfig)


@dataclass
class Reference:
    """Everything the detectors need from the clean period of one model version."""

    drift: UnivariateDrift
    mmd: MMDDetector
    scores: ScoreDrift
    quality: QualityProfile
    cbpe: CBPEMonitor
    default_rate: float

    @classmethod
    def build(
        cls,
        model: Scorer,
        reference: Batch,
        reference_labels: np.ndarray,
        pool: Batch,
        config: MonitorConfig | None = None,
        seed: int = 0,
    ) -> Reference:
        """Fit every profile; ``pool`` must be clean data disjoint from ``reference``."""
        cfg = config or MonitorConfig()
        reference_pd = model.predict_pd(reference)
        cbpe = CBPEMonitor.fit(reference_pd, model.threshold, cfg.batch_size, seed=seed)
        cbpe.n_sigma = cfg.cbpe_sigma
        return cls(
            drift=UnivariateDrift.fit(reference),
            mmd=MMDDetector.fit(reference, pool, seed=seed),
            scores=ScoreDrift.fit(reference_pd),
            quality=QualityProfile.fit(reference),
            cbpe=cbpe,
            default_rate=float(np.mean(reference_labels)),
        )


@dataclass
class BatchReport:
    """What the monitor saw in one unlabelled batch."""

    index: int
    timestamp: str
    n_rows: int
    signals: list[Signal]
    alerts: list[Alert]
    estimate: PerformanceEstimate
    psi: dict[str, float]
    drift_p: dict[str, float]
    mmd_p: float
    score_p: float
    scores: np.ndarray

    def summary(self) -> dict[str, Any]:
        """JSON-friendly per-batch record for the timeline."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "n_rows": self.n_rows,
            "estimated_auc": _finite(self.estimate.auc),
            "estimated_default_rate": _finite(self.estimate.default_rate),
            "approval_rate": _finite(self.estimate.approval_rate),
            "psi": {k: round(v, 5) for k, v in self.psi.items()},
            "drift_p": {k: float(f"{v:.3g}") for k, v in self.drift_p.items()},
            "mmd_p": float(f"{self.mmd_p:.3g}"),
            "score_p": float(f"{self.score_p:.3g}"),
            "fired": sorted({f"{s.category}:{s.scope}" for s in self.signals if s.fired}),
            "alerts": [a.category.value + ":" + a.scope for a in self.alerts],
        }


@dataclass
class LabelReport:
    """What the monitor saw when the labels of batch ``index`` arrived at batch ``at``."""

    index: int
    at: int
    n_rows: int
    realized_auc: float
    default_rate: float
    brier: float
    error_rate: float
    signals: list[Signal]
    alerts: list[Alert]


def _finite(value: float) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


class Monitor:
    """Stateful per-model monitor; feed it batches in time order."""

    def __init__(
        self,
        model: Scorer,
        reference: Reference,
        config: MonitorConfig | None = None,
        policy: AlertPolicy | None = None,
    ) -> None:
        self.model = model
        self.reference = reference
        self.config = config or MonitorConfig()
        self.policy = policy or AlertPolicy()
        self.adwin = BatchedADWIN(delta=self.config.adwin_delta)
        self.ddm = DDM(drift_level=self.config.ddm_drift_level)
        self.label_rate = LabelRate(reference.default_rate, self.config.label_alpha)

    def _drift_signals(self, batch: Batch) -> tuple[list[Signal], dict[str, float], dict[str, float], float]:
        cfg = self.config
        results = self.reference.drift.evaluate(batch)
        per_feature_alpha = cfg.drift_alpha / max(len(results), 1)
        signals: list[Signal] = []
        for r in results:
            signals.append(
                Signal(
                    r.test,
                    Category.DATA_DRIFT,
                    r.feature,
                    r.p_value < per_feature_alpha,
                    r.statistic,
                    per_feature_alpha,
                    r.p_value,
                    f"{r.test} p={r.p_value:.2g}, PSI {r.psi:.3f}",
                )
            )
            signals.append(
                Signal(
                    "psi",
                    Category.DATA_DRIFT,
                    r.feature,
                    r.psi >= cfg.psi_alarm,
                    r.psi,
                    cfg.psi_alarm,
                    None,
                    f"PSI {r.psi:.3f}",
                )
            )
        stat, mmd_p = self.reference.mmd.test(batch)
        signals.append(
            Signal(
                "mmd",
                Category.DATA_DRIFT,
                "multivariate",
                mmd_p < cfg.mmd_alpha,
                stat,
                cfg.mmd_alpha,
                mmd_p,
                f"MMD p={mmd_p:.2g}",
            )
        )
        return signals, {r.feature: r.psi for r in results}, {r.feature: r.p_value for r in results}, mmd_p

    def observe(
        self, index: int, data: dict[str, Any] | Batch, timestamp: str = "", scores: np.ndarray | None = None
    ) -> BatchReport:
        """Evaluate one unlabelled batch (raw columns or an already coerced batch)."""
        issues: list[SchemaIssue] = []
        batch = data if isinstance(data, Batch) else None
        if batch is None:
            batch, issues = coerce(data)
        signals = schema_signals(issues) + quality_signals(batch, self.reference.quality, self.config.quality)
        drift, psi_values, drift_p, mmd_p = self._drift_signals(batch)
        signals += drift
        pd = self.model.predict_pd(batch) if scores is None else scores
        d, score_p = self.reference.scores.evaluate(pd)
        alpha = self.config.score_alpha
        signals.append(
            Signal(
                "score_ks",
                Category.PREDICTION_DRIFT,
                "score",
                score_p < alpha,
                d,
                alpha,
                score_p,
                f"score KS D={d:.3f} p={score_p:.2g}",
            )
        )
        estimate, cbpe_signal = self.reference.cbpe.evaluate(pd)
        signals.append(cbpe_signal)
        alerts = self.policy.apply(index, signals, timestamp)
        return BatchReport(
            index, timestamp, batch.n_rows, signals, alerts, estimate, psi_values, drift_p, mmd_p, score_p, pd
        )

    def observe_labels(
        self, at: int, index: int, labels: np.ndarray, scores: np.ndarray, timestamp: str = ""
    ) -> LabelReport:
        """Evaluate the outcomes of batch ``index`` (scored at the time with ``scores``)."""
        y = np.asarray(labels, dtype=np.float64)
        p = np.asarray(scores, dtype=np.float64)
        errors = approve(p, self.model.threshold) == (y == 1)
        adwin = self.adwin.update((y - p + 1.0) / 2.0)
        ddm = self.ddm.update(errors)
        signals = [self.label_rate.evaluate(y)]
        if adwin.drift:
            before, after = 2.0 * adwin.mean_before - 1.0, 2.0 * adwin.mean_after - 1.0
            msg = f"calibration residual moved from {before:+.4f} to {after:+.4f}"
            signals.append(
                Signal("adwin", Category.PERFORMANCE, "model", True, after - before, self.config.adwin_delta, None, msg)
            )
        else:
            signals.append(Signal("adwin", Category.PERFORMANCE, "model", False, 0.0, self.config.adwin_delta))
        msg = (
            f"decision error rate {ddm.error_rate:.3f} above the running minimum by {self.config.ddm_drift_level} sigma"
        )
        signals.append(
            Signal(
                "ddm",
                Category.PERFORMANCE,
                "model",
                ddm.drift,
                ddm.error_rate,
                self.config.ddm_drift_level,
                None,
                msg if ddm.drift else "",
            )
        )
        alerts = self.policy.apply(at, signals, timestamp)
        brier = float(np.mean((p - y) ** 2))
        return LabelReport(index, at, len(y), auc(p, y), float(y.mean()), brier, float(errors.mean()), signals, alerts)
