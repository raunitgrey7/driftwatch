"""Shadow and canary rollout with statistical guardrails and automatic rollback.

During a rollout both models score every application. The champion keeps
deciding for the control share; the candidate decides only for its canary
share, but its scores on every row are kept. Statistical power therefore
comes from all traffic while business exposure is limited to the canary
share, and in the shadow step the exposure is zero.

Guardrails, fastest first:

1. runtime: the candidate must return a finite probability in [0, 1]; more
   than ``error_rate_max`` failures in a batch is an immediate rollback, and
   failed rows fall back to the champion's decision;
2. rank agreement: Spearman correlation with the champion on the same rows
   below ``min_rank_corr`` means the candidate orders applicants unlike any
   plausible retrain (an inverted score, a broken feature transform);
3. approval rate: the paired difference of approval decisions on the same
   rows, bounded by a normal-mixture confidence sequence so it can be checked
   after every batch; rollback when the whole interval is beyond the tolerance;
4. calibration (labels): the candidate's mean residual ``y - p`` with the
   same kind of confidence sequence; rollback when it is surely beyond the
   tolerance;
5. discrimination (labels): DeLong paired AUC difference. Rollback when the
   candidate is significantly worse; promotion needs non-inferiority, a
   lower bound above ``-auc_margin``. Looks are Bonferroni-spent over the
   largest possible number of looks, so peeking does not inflate the error.

The ramp itself is evidence-gated: steps above 5 percent traffic open only
after a discrimination look has shown non-inferiority, so a subtly worse
model that only labels can expose serves at most 5 percent of applications
while the labels mature. A rollout that reaches ``max_batches`` without
proving non-inferiority is rolled back: the burden of proof is on the
candidate.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from driftwatch.model import approve
from driftwatch.stats import delong_paired, mixture_half_width


class RolloutStatus(StrEnum):
    RUNNING = "running"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class Step:
    """One rung of the ramp; ``needs_labels`` steps open only after non-inferiority is shown."""

    name: str
    traffic: float
    min_batches: int
    needs_labels: bool = False


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout plan and guardrail thresholds."""

    steps: tuple[Step, ...] = (
        Step("shadow", 0.0, 5),
        Step("canary-5", 0.05, 5),
        Step("canary-25", 0.25, 5, needs_labels=True),
        Step("canary-50", 0.50, 5, needs_labels=True),
    )
    max_batches: int = 150
    error_rate_max: float = 0.01
    min_rank_corr: float = 0.5
    approval_tolerance: float = 0.10
    calibration_tolerance: float = 0.02
    auc_margin: float = 0.01
    alpha: float = 0.05
    look_every: int = 5
    min_labeled_rows: int = 2000
    cs_rho: float = 50.0

    @property
    def z_look(self) -> float:
        """Two-sided critical value with alpha split over every possible look."""
        looks = math.ceil(self.max_batches / self.look_every)
        return float(ndtri(1.0 - self.alpha / (2.0 * looks)))


@dataclass(frozen=True)
class Guardrail:
    name: str
    passed: bool
    value: float
    bound: float
    detail: str


@dataclass
class _Running:
    """Running sum and variance sum for a confidence-sequence guardrail."""

    n: int = 0
    total: float = 0.0
    sq: float = 0.0

    def add(self, values: np.ndarray) -> None:
        self.n += len(values)
        self.total += float(values.sum())
        self.sq += float(np.square(values).sum())

    def interval(self, rho: float, alpha: float) -> tuple[float, float, float]:
        mean = self.total / max(self.n, 1)
        variance_sum = max(self.sq - self.n * mean * mean, 0.0)
        half = mixture_half_width(variance_sum, rho, alpha) / max(self.n, 1)
        return mean, mean - half, mean + half


def bucket(ids: np.ndarray, salt: str) -> np.ndarray:
    """Stable pseudo-uniform number in [0, 1) per application id, so routing is reproducible."""
    out = np.empty(len(ids))
    for i, value in enumerate(ids.tolist()):
        digest = hashlib.blake2b(f"{salt}:{value}".encode(), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "big") / 2.0**64
    return out


def failed_rows(pd: np.ndarray) -> np.ndarray:
    return ~np.isfinite(pd) | (pd < 0.0) | (pd > 1.0)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class BatchOutcome:
    """Decisions actually applied to one batch and what the guardrails said."""

    decisions: np.ndarray
    served_by_candidate: np.ndarray
    guardrails: list[Guardrail]
    action: str


@dataclass
class Rollout:
    """State machine for one candidate against the current champion."""

    candidate_version: int
    champion_threshold: float
    candidate_threshold: float
    config: RolloutConfig = field(default_factory=RolloutConfig)
    status: RolloutStatus = RolloutStatus.RUNNING
    step: int = 0
    batches_in_step: int = 0
    batches: int = 0
    exposure: int = 0
    reason: str = ""
    labeled_batches: int = 0
    noninferior: bool = False
    _approval: _Running = field(default_factory=_Running)
    _calibration: _Running = field(default_factory=_Running)
    _labels: list[np.ndarray] = field(default_factory=list)
    _champ: list[np.ndarray] = field(default_factory=list)
    _cand: list[np.ndarray] = field(default_factory=list)
    history: list[dict[str, object]] = field(default_factory=list)
    _last_look: str = ""

    @property
    def traffic(self) -> float:
        return self.config.steps[self.step].traffic if self.status is RolloutStatus.RUNNING else 0.0

    @property
    def step_name(self) -> str:
        return self.config.steps[self.step].name

    def _finish(self, status: RolloutStatus, reason: str) -> str:
        self.status, self.reason = status, reason
        self.history.append({"batch": self.batches, "event": status.value, "reason": reason})
        return "rollback" if status is RolloutStatus.ROLLED_BACK else "promote"

    def _label_free_guardrails(
        self, champ_pd: np.ndarray, cand_pd: np.ndarray, failures: np.ndarray
    ) -> list[Guardrail]:
        cfg = self.config
        error_rate = float(failures.mean()) if len(failures) else 0.0
        ok = ~failures
        rho = spearman(champ_pd[ok], cand_pd[ok]) if ok.sum() >= 30 else 1.0
        diff = approve(cand_pd[ok], self.candidate_threshold).astype(np.float64) - approve(
            champ_pd[ok], self.champion_threshold
        )
        self._approval.add(diff)
        mean, lo, hi = self._approval.interval(cfg.cs_rho, cfg.alpha)
        tol = cfg.approval_tolerance
        return [
            Guardrail(
                "runtime",
                error_rate <= cfg.error_rate_max,
                error_rate,
                cfg.error_rate_max,
                f"{error_rate:.2%} of rows failed",
            ),
            Guardrail(
                "rank_agreement",
                rho >= cfg.min_rank_corr,
                rho,
                cfg.min_rank_corr,
                f"Spearman {rho:.3f} with the champion",
            ),
            Guardrail(
                "approval_rate",
                not (lo > tol or hi < -tol),
                mean,
                tol,
                f"approval difference {mean:+.3f} in [{lo:+.3f}, {hi:+.3f}]",
            ),
        ]

    def observe_batch(self, ids: np.ndarray, champ_pd: np.ndarray, cand_pd: np.ndarray) -> BatchOutcome:
        """Route one batch, apply decisions and run the label-free guardrails."""
        champ_decision = approve(champ_pd, self.champion_threshold)
        if self.status is not RolloutStatus.RUNNING:
            return BatchOutcome(champ_decision, np.zeros(len(ids), dtype=bool), [], "none")
        failures = failed_rows(cand_pd)
        served = (bucket(ids, f"v{self.candidate_version}") < self.traffic) & ~failures
        safe_cand = np.where(failures, 1.0, cand_pd)
        decisions = np.where(served, approve(safe_cand, self.candidate_threshold), champ_decision)
        guardrails = self._label_free_guardrails(champ_pd, cand_pd, failures)
        self.batches += 1
        self.batches_in_step += 1
        self.exposure += int(served.sum())
        failed = [g for g in guardrails if not g.passed]
        if failed:
            action = self._finish(RolloutStatus.ROLLED_BACK, f"{failed[0].name}: {failed[0].detail}")
        else:
            action = self._advance()
        return BatchOutcome(decisions, served, guardrails, action)

    def _advance(self) -> str:
        """Move up the ramp when the step's minimum is served and the next step's evidence exists."""
        cfg = self.config
        if self.batches >= cfg.max_batches:
            return self._finish(RolloutStatus.ROLLED_BACK, "non-inferiority not established within the rollout budget")
        if self.batches_in_step < cfg.steps[self.step].min_batches:
            return "continue"
        if self.step == len(cfg.steps) - 1:
            if self.noninferior:
                return self._finish(
                    RolloutStatus.PROMOTED,
                    f"non-inferior after {self.labeled_batches} labelled batches: {self._last_look}",
                )
            return "continue"
        if cfg.steps[self.step + 1].needs_labels and not self.noninferior:
            return "continue"
        self.step += 1
        self.batches_in_step = 0
        self.history.append({"batch": self.batches, "event": "advance", "reason": self.step_name})
        return "advance"

    def _discrimination(self) -> Guardrail | None:
        cfg = self.config
        labels = np.concatenate(self._labels)
        if len(labels) < cfg.min_labeled_rows or self.labeled_batches % cfg.look_every:
            return None
        paired = delong_paired(labels, np.concatenate(self._cand), np.concatenate(self._champ))
        lo, hi = paired.diff - cfg.z_look * paired.se, paired.diff + cfg.z_look * paired.se
        detail = f"AUC difference {paired.diff:+.4f} in [{lo:+.4f}, {hi:+.4f}] on {len(labels)} rows"
        return Guardrail("discrimination", hi >= 0.0, lo, -cfg.auc_margin, detail)

    def observe_labels(
        self, labels: np.ndarray, champ_pd: np.ndarray, cand_pd: np.ndarray
    ) -> tuple[list[Guardrail], str]:
        """Feed the outcomes of a batch scored during the rollout; may roll back or promote."""
        if self.status is not RolloutStatus.RUNNING:
            return [], "none"
        cfg = self.config
        ok = ~failed_rows(cand_pd)
        y = np.asarray(labels, dtype=np.float64)[ok]
        self._labels.append(y)
        self._champ.append(champ_pd[ok])
        self._cand.append(cand_pd[ok])
        self.labeled_batches += 1
        self._calibration.add(y - cand_pd[ok])
        mean, lo, hi = self._calibration.interval(cfg.cs_rho, cfg.alpha)
        tol = cfg.calibration_tolerance
        guardrails = [
            Guardrail(
                "calibration",
                not (lo > tol or hi < -tol),
                mean,
                tol,
                f"mean residual {mean:+.4f} in [{lo:+.4f}, {hi:+.4f}]",
            )
        ]
        discrimination = self._discrimination()
        if discrimination is not None:
            guardrails.append(discrimination)
        failed = [g for g in guardrails if not g.passed]
        if failed:
            return guardrails, self._finish(RolloutStatus.ROLLED_BACK, f"{failed[0].name}: {failed[0].detail}")
        if discrimination is not None:
            self.noninferior = discrimination.value > -cfg.auc_margin
            self._last_look = discrimination.detail
        return guardrails, "continue"
