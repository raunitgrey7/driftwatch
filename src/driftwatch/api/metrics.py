"""Minimal Prometheus text exposition without the client library.

Three metric types cover the service, and a dependency-free implementation
keeps the image small. Label values are escaped per the exposition format and
``render`` output is sorted, so tests can assert on exact lines.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field

Labels = tuple[tuple[str, str], ...]


def _labels(**kwargs: str) -> Labels:
    return tuple(sorted((k, str(v)) for k, v in kwargs.items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_labels(labels: Labels) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in labels) + "}"


@dataclass
class Counter:
    name: str
    help: str
    _values: dict[Labels, float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _labels(**labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(_labels(**labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        return lines + [f"{self.name}{_fmt_labels(k)} {v:g}" for k, v in sorted(self._values.items())]


@dataclass
class Gauge:
    name: str
    help: str
    _values: dict[Labels, float] = field(default_factory=dict)

    def set(self, value: float, **labels: str) -> None:
        self._values[_labels(**labels)] = float(value)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        return lines + [f"{self.name}{_fmt_labels(k)} {v:g}" for k, v in sorted(self._values.items())]


@dataclass
class Histogram:
    name: str
    help: str
    buckets: Sequence[float] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
    _counts: dict[Labels, list[int]] = field(default_factory=dict)
    _sums: dict[Labels, float] = field(default_factory=dict)
    _totals: dict[Labels, int] = field(default_factory=dict)

    def observe(self, value: float, **labels: str) -> None:
        key = _labels(**labels)
        counts = self._counts.setdefault(key, [0] * len(self.buckets))
        for i in range(bisect_left(list(self.buckets), value), len(self.buckets)):
            counts[i] += 1
        self._sums[key] = self._sums.get(key, 0.0) + value
        self._totals[key] = self._totals.get(key, 0) + 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, counts in sorted(self._counts.items()):
            for bound, count in zip(self.buckets, counts, strict=True):
                lines.append(f"{self.name}_bucket{_fmt_labels((*key, ('le', f'{bound:g}')))} {count}")
            lines.append(f"{self.name}_bucket{_fmt_labels((*key, ('le', '+Inf')))} {self._totals[key]}")
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {self._sums[key]:g}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {self._totals[key]}")
        return lines


class MetricsRegistry:
    """The service's metrics, rendered in exposition format under one lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.http_requests = Counter(
            "driftwatch_http_requests_total", "HTTP requests by method, path template and status."
        )
        self.http_duration = Histogram(
            "driftwatch_http_request_duration_seconds", "HTTP request latency by method and path template."
        )
        self.batches = Counter("driftwatch_batches_total", "Batches ingested.")
        self.rows = Counter(
            "driftwatch_rows_scored_total", "Applications scored, by the model version that decided them."
        )
        self.label_batches = Counter("driftwatch_label_batches_total", "Batches whose outcome labels arrived.")
        self.alerts = Counter("driftwatch_alerts_opened_total", "Alerts opened, by category and severity.")
        self.rollout_events = Counter("driftwatch_rollout_events_total", "Rollout starts, promotions and rollbacks.")
        self.retrains = Counter("driftwatch_retrains_total", "Retraining attempts by gate outcome.")
        self.champion_version = Gauge("driftwatch_champion_version", "Model version in production.")
        self.rollout_traffic = Gauge("driftwatch_rollout_traffic_share", "Share of traffic decided by the candidate.")
        self.open_alerts = Gauge("driftwatch_open_alerts", "Alerts not yet acknowledged.")
        self.estimated_auc = Gauge("driftwatch_estimated_auc", "CBPE estimate of AUC on the latest batch.")
        self.realized_auc = Gauge("driftwatch_realized_auc", "AUC on the latest labelled batch.")
        self.approval_rate = Gauge("driftwatch_approval_rate", "Approval rate of the latest batch.")

    def render(self) -> str:
        with self.lock:
            metrics = [v for v in self.__dict__.values() if isinstance(v, Counter | Gauge | Histogram)]
            return "\n".join(line for m in metrics for line in m.render()) + "\n"
