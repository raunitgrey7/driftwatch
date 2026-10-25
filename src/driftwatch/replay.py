"""Stream a simulated scenario into a running driftwatch service.

The client plays the upstream systems: the origination pipeline posts each
day's batch of applications, and the collections system posts their outcomes
``label_delay`` batches later. It works against any ``httpx.Client``, so the
same code drives a live server or the in-process test client.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.schema import ID_COLUMN


@dataclass(frozen=True)
class ReplayConfig:
    scenario: str = "covariate_abrupt"
    seed: int = 11
    batches: int = 180
    batch_size: int = 400
    onset: int = 60
    label_delay: int = 30
    interval: float = 0.0
    prefix: str = ""

    def batch_id(self, index: int) -> str:
        return f"{self.prefix or f'{self.scenario}-s{self.seed}'}-{index}"


def to_records(raw: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Column arrays to JSON records; NaN becomes null, numpy scalars become Python values."""
    columns = list(raw)
    n = len(raw[ID_COLUMN])
    out: list[dict[str, Any]] = []
    for i in range(n):
        record: dict[str, Any] = {}
        for name in columns:
            value = raw[name][i]
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and math.isnan(value):
                value = None
            record[name] = value
        out.append(record)
    return out


def _events(body: dict[str, Any]) -> list[str]:
    events = [f"alert {a['category']}:{a['scope']}" for a in body.get("alerts", [])]
    if body.get("rollout_action") not in (None, "none", "continue"):
        events.append(f"rollout {body['rollout_action']}")
    if body.get("retrain"):
        r = body["retrain"]
        events.append(
            f"retrain v{r['version']} ({r['challenger']}) {'passed' if r['promote'] else 'rejected'}: {r['reason']}"
        )
    return events


def replay(client: httpx.Client, config: ReplayConfig, echo: Callable[[str], None] = print) -> dict[str, int]:
    """Post every batch, deliver labels late, and report what the service did."""
    stream = CreditStream(StreamConfig(config.scenario, config.seed, config.batches, config.batch_size, config.onset))
    pending: dict[int, tuple[list[str], np.ndarray]] = {}
    counts = {"batches": 0, "labelled": 0, "alerts": 0, "retrains": 0, "rollout_events": 0}
    for batch in stream:
        payload = {
            "batch_id": config.batch_id(batch.index),
            "timestamp": batch.timestamp,
            "rows": to_records(batch.raw),
            "include_rows": False,
        }
        response = client.post("/v1/batches", json=payload)
        response.raise_for_status()
        body = response.json()
        events = _events(body)
        counts["batches"] += 1
        pending[batch.index] = ([str(i) for i in batch.raw[ID_COLUMN].tolist()], batch.labels)
        if batch.index >= config.label_delay:
            ids, labels = pending.pop(batch.index - config.label_delay)
            labelled = [{"application_id": i, "defaulted": bool(y)} for i, y in zip(ids, labels.tolist(), strict=True)]
            response = client.post(
                f"/v1/batches/{config.batch_id(batch.index - config.label_delay)}/labels", json={"labels": labelled}
            )
            response.raise_for_status()
            events += _events(response.json())
            counts["labelled"] += 1
        counts["alerts"] += sum(e.startswith("alert") for e in events)
        counts["retrains"] += sum(e.startswith("retrain") for e in events)
        counts["rollout_events"] += sum(e.startswith("rollout") for e in events)
        if events or batch.index % 20 == 0:
            echo(
                f"batch {batch.index:4d} {batch.timestamp} drift weight {batch.drift_weight:.2f} | "
                + ("; ".join(events) or "quiet")
            )
        if config.interval > 0:
            time.sleep(config.interval)
    return counts
