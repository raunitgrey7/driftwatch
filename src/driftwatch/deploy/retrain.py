"""Retraining trigger and the champion-challenger gate.

Not every alert should retrain the model. Schema and data-quality alerts
mean the pipeline is broken: retraining on that data would teach the model
the bug, so they block retraining while they are active. Performance and
label alerts are direct evidence that the model is wrong for the current
world and trigger a retrain. Label-free alerts (data drift, prediction drift,
estimated performance) only trigger when they keep recurring, since a
population change alone does not make a calibrated model wrong.

The trigger waits ``wait_batches`` after the evidence so that labelled data
from after the change exists, then the training window starts just before
the evidence instead of mixing in months of the old regime.

Two challengers are built from the window: a full refit, and the champion
recalibrated on the recent rows (the right answer when only the default
level moved). Each is gated at half the error rate, Bonferroni for two
comparisons, and the passing one with the lower log loss is kept. After a
rejection the trigger backs off for ``backoff_batches`` so a persistent but
harmless shift does not retrain the model every few weeks.

The gate compares champion and challenger on the newest labelled slice,
which neither model was fitted on, with paired statistics on the same rows:
the challenger must be significantly better in log loss and not worse in
AUC by more than ``auc_margin`` (DeLong lower bound). A challenger that is
merely different is rejected, so the system does not churn models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtri

from driftwatch.alerts import Alert, Category
from driftwatch.model import CreditModel, Scorer, fit_platt
from driftwatch.stats import delong_paired
from driftwatch.training import LabeledData

TRIGGERS = (Category.PERFORMANCE, Category.LABEL_DRIFT)
SUSTAINED = (Category.DATA_DRIFT, Category.PREDICTION_DRIFT, Category.ESTIMATED_PERFORMANCE)
BLOCKERS = (Category.SCHEMA, Category.DATA_QUALITY)


@dataclass(frozen=True)
class RetrainConfig:
    wait_batches: int = 20
    window_batches: int = 40
    sustained_alerts: int = 2
    sustained_span: int = 60
    block_span: int = 30
    lead_batches: int = 5
    backoff_batches: int = 60
    train_fraction: float = 0.6
    calib_fraction: float = 0.15
    auc_margin: float = 0.005
    alpha: float = 0.05
    approval_rate: float = 0.80


@dataclass
class RetrainTrigger:
    """Collects alerts and says when a retrain is due and from which batch its data may start."""

    config: RetrainConfig = field(default_factory=RetrainConfig)
    evidence: int | None = None
    latest_labelled: int = -1
    _sustained: list[int] = field(default_factory=list)
    _blocked_until: int = -1
    _backoff_until: int = -1

    def observe(self, at: int, alerts: list[Alert], labelled_index: int | None = None) -> None:
        """Record alerts raised at batch ``at``; label alerts carry the index of the batch they judged."""
        cfg = self.config
        if labelled_index is not None:
            self.latest_labelled = max(self.latest_labelled, labelled_index)
        for alert in alerts:
            if alert.category in BLOCKERS:
                self._blocked_until = max(self._blocked_until, at + cfg.block_span)
            elif alert.category in TRIGGERS and self.evidence is None:
                self.evidence = at if labelled_index is None else labelled_index
            elif alert.category in SUSTAINED:
                self._sustained = [t for t in self._sustained if at - t < cfg.sustained_span] + [at]
                if len(self._sustained) >= cfg.sustained_alerts and self.evidence is None:
                    self.evidence = self._sustained[0]

    def blocked(self, at: int) -> bool:
        return at <= self._blocked_until

    def due(self, at: int) -> bool:
        """Evidence exists, enough labelled batches followed it, and nothing blocks or backs off."""
        if self.evidence is None or self.blocked(at) or at < self._backoff_until:
            return False
        return self.latest_labelled >= self.evidence + self.config.wait_batches

    def window(self) -> tuple[int, int]:
        """Inclusive range of labelled batch indices to train the challenger on."""
        latest = self.latest_labelled
        start = latest - self.config.window_batches + 1
        if self.evidence is not None:
            start = max(start, self.evidence - self.config.lead_batches)
        return start, latest

    def reset(self, at: int = -1, rejected: bool = False) -> None:
        """Forget the evidence; after a rejected challenger, wait ``backoff_batches`` before trying again."""
        self.evidence = None
        self._sustained.clear()
        if rejected:
            self._backoff_until = at + self.config.backoff_batches


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str
    n_rows: int
    auc_champion: float
    auc_challenger: float
    auc_diff_lcb: float
    logloss_champion: float
    logloss_challenger: float
    logloss_diff_ucb: float

    def to_dict(self) -> dict[str, object]:
        return {k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v) for k, v in self.__dict__.items()}


def row_log_loss(pd: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = np.clip(pd, 1e-6, 1.0 - 1e-6)
    return -(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))


def champion_challenger_gate(
    champion: Scorer, challenger: Scorer, holdout: LabeledData, config: RetrainConfig | None = None
) -> GateDecision:
    """Paired comparison on the same held-out rows; see the module docstring for the rule."""
    cfg = config or RetrainConfig()
    y = holdout.labels.astype(np.float64)
    champ, chall = champion.predict_pd(holdout.batch), challenger.predict_pd(holdout.batch)
    z = float(ndtri(1.0 - cfg.alpha / 2.0))
    paired = delong_paired(y, chall, champ)
    auc_lcb = paired.diff - z * paired.se
    loss_champ, loss_chall = row_log_loss(champ, y), row_log_loss(chall, y)
    delta = loss_chall - loss_champ
    loss_ucb = float(delta.mean() + z * delta.std(ddof=1) / math.sqrt(len(delta)))
    better = loss_ucb < 0.0
    not_worse = auc_lcb > -cfg.auc_margin
    if better and not_worse:
        reason = "challenger has lower log loss and non-inferior AUC"
    elif not better:
        reason = "log loss improvement not significant"
    else:
        reason = "AUC lower bound below the margin"
    return GateDecision(
        promote=better and not_worse,
        reason=reason,
        n_rows=len(y),
        auc_champion=paired.auc_b,
        auc_challenger=paired.auc_a,
        auc_diff_lcb=float(auc_lcb),
        logloss_champion=float(loss_champ.mean()),
        logloss_challenger=float(loss_chall.mean()),
        logloss_diff_ucb=loss_ucb,
    )


def recalibrate(model: CreditModel, data: LabeledData, approval_rate: float) -> CreditModel:
    """The champion's booster with its Platt scaling refitted on recent labelled rows.

    Under prior shift the ranking is still right and only the level is wrong,
    so two refitted parameters beat a full refit on a few thousand rows. The
    cut-off follows the same rule as training: approve ``approval_rate`` of
    the recent population.
    """
    platt = fit_platt(model.raw_pd(data.batch), data.labels)
    new = CreditModel(
        booster=model.booster,
        platt=platt,
        threshold=1.0,
        dropped=model.dropped,
        info={**model.info, "recalibrated_on": data.batch.n_rows},
    )
    new.threshold = float(np.quantile(new.predict_pd(data.batch), approval_rate))
    return new


def split_window(
    data: LabeledData, config: RetrainConfig | None = None
) -> tuple[LabeledData, LabeledData, LabeledData]:
    """Time-ordered train, calibration and gate-holdout slices of a labelled window."""
    cfg = config or RetrainConfig()
    train, rest = data.split(cfg.train_fraction)
    calib, holdout = rest.split(cfg.calib_fraction / (1.0 - cfg.train_fraction))
    return train, calib, holdout
