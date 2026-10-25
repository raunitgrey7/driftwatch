"""Signals, alerts and the policy that turns one into the other.

A detector emits a ``Signal`` for every evaluation, fired or not, so the
timeline can show statistics below the alarm line. The ``AlertPolicy``
decides which fired signals become alerts:

* Persistence (k of the last m evaluations of the same key fired) filters
  one-batch flukes of the statistical tests: with a per-batch false-positive
  rate ``a``, 2-of-3 persistence brings it to about ``3 a^2``.
* A cooldown per key stops one incident from raising an alert every batch.
* Schema and data-quality rules use persistence 1: a broken pipeline is
  certain the first time and waiting costs real credit decisions.
* A data-drift alert on a feature that has an active data-quality alert is
  suppressed, because the quality incident already explains it; drift on
  other features still alerts.

Keys are (category, scope), so drift on ``bureau_score`` does not hide drift
on ``monthly_income``.

DDM is advisory by default: its signals are kept for the timeline but do not
open alerts. On stationary streams its 3-sigma rule over a running minimum
fires falsely on a sizeable share of long streams, and in this benchmark it
detected nothing the other detectors missed (docs/EVALUATION.md has the
numbers); ADWIN carries the label-based performance alerts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum


class Category(StrEnum):
    """What kind of problem an alert reports; drives severity and the retraining trigger."""

    SCHEMA = "schema"
    DATA_QUALITY = "data_quality"
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    ESTIMATED_PERFORMANCE = "estimated_performance"
    LABEL_DRIFT = "label_drift"
    PERFORMANCE = "performance"


SEVERITY: dict[Category, str] = {
    Category.SCHEMA: "critical",
    Category.DATA_QUALITY: "high",
    Category.PERFORMANCE: "high",
    Category.LABEL_DRIFT: "high",
    Category.ESTIMATED_PERFORMANCE: "medium",
    Category.PREDICTION_DRIFT: "medium",
    Category.DATA_DRIFT: "medium",
}


@dataclass(frozen=True)
class Signal:
    """One detector evaluation on one batch (or one labelled batch)."""

    detector: str
    category: Category
    scope: str
    fired: bool
    statistic: float
    threshold: float
    p_value: float | None = None
    message: str = ""


@dataclass(frozen=True)
class Alert:
    """An alert opened by the policy."""

    category: Category
    scope: str
    batch_index: int
    severity: str
    detectors: tuple[str, ...]
    message: str
    timestamp: str = ""

    @property
    def key(self) -> tuple[Category, str]:
        return (self.category, self.scope)


@dataclass(frozen=True)
class Rule:
    """Persistence ``k`` of the last ``m`` evaluations, then ``cooldown`` batches of silence."""

    persistence: int = 1
    window: int = 1
    cooldown: int = 20

    def __post_init__(self) -> None:
        if not 1 <= self.persistence <= self.window:
            raise ValueError("need 1 <= persistence <= window")
        if self.cooldown < 0:
            raise ValueError("cooldown must be non-negative")


DEFAULT_RULES: dict[Category, Rule] = {
    Category.SCHEMA: Rule(1, 1, 10),
    Category.DATA_QUALITY: Rule(1, 1, 20),
    Category.DATA_DRIFT: Rule(2, 3, 30),
    Category.PREDICTION_DRIFT: Rule(2, 3, 30),
    Category.ESTIMATED_PERFORMANCE: Rule(2, 3, 30),
    Category.LABEL_DRIFT: Rule(2, 3, 30),
    Category.PERFORMANCE: Rule(1, 1, 30),
}


ADVISORY: frozenset[str] = frozenset({"ddm"})


@dataclass
class AlertPolicy:
    """Stateful filter from fired signals to alerts; ``advisory`` detectors are recorded but never page."""

    rules: dict[Category, Rule] = field(default_factory=lambda: dict(DEFAULT_RULES))
    suppress_explained_drift: bool = True
    advisory: frozenset[str] = ADVISORY
    _history: dict[tuple[Category, str], deque[bool]] = field(default_factory=dict)
    _last_alert: dict[tuple[Category, str], int] = field(default_factory=dict)
    suppressed: int = 0

    def _quality_active(self, scope: str, batch_index: int) -> bool:
        last = self._last_alert.get((Category.DATA_QUALITY, scope))
        return last is not None and batch_index - last < self.rules[Category.DATA_QUALITY].cooldown

    def apply(self, batch_index: int, signals: list[Signal], timestamp: str = "") -> list[Alert]:
        """Update the per-key history with this evaluation and return the alerts it opens."""
        grouped: dict[tuple[Category, str], list[Signal]] = {}
        for signal in signals:
            if signal.detector not in self.advisory:
                grouped.setdefault((signal.category, signal.scope), []).append(signal)
        alerts: list[Alert] = []
        for key in sorted(grouped, key=lambda k: (list(Category).index(k[0]), k[1])):
            category, scope = key
            rule = self.rules[category]
            fired = [s for s in grouped[key] if s.fired]
            history = self._history.setdefault(key, deque(maxlen=rule.window))
            history.append(bool(fired))
            if sum(history) < rule.persistence or not fired:
                continue
            last = self._last_alert.get(key)
            if last is not None and batch_index - last < rule.cooldown:
                continue
            if (
                category is Category.DATA_DRIFT
                and self.suppress_explained_drift
                and self._quality_active(scope, batch_index)
            ):
                self.suppressed += 1
                continue
            self._last_alert[key] = batch_index
            message = "; ".join(s.message for s in fired if s.message)
            detectors = tuple(sorted({s.detector for s in fired}))
            alerts.append(Alert(category, scope, batch_index, SEVERITY[category], detectors, message, timestamp))
        return alerts
