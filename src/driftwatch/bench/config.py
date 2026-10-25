"""Benchmark configuration: one full preset for ``make bench`` and a quick one for tests.

The full preset is sized so false alarms are measured on at least 10,000
drift-free control batches and every drift type is replicated on ten
independent streams, while the whole run stays within a few minutes on two
worker processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from driftwatch.data.scenarios import SCENARIOS
from driftwatch.training import HistoryConfig

DRIFT_SCENARIOS: tuple[str, ...] = tuple(name for name in SCENARIOS if name != "control")
RETRAIN_SCENARIOS: tuple[str, ...] = (
    "control",
    "covariate_abrupt",
    "prior_shift",
    "concept_abrupt",
    "concept_gradual",
    "dq_unit_change",
)


@dataclass(frozen=True)
class BenchConfig:
    seed: int = 2026
    workers: int = 2
    batch_size: int = 400
    label_delay: int = 30
    history: HistoryConfig = field(default_factory=HistoryConfig)
    control_streams: int = 25
    control_batches: int = 400
    drift_seeds: int = 10
    drift_batches: int = 240
    onset_range: tuple[int, int] = (100, 140)
    detection_window: int = 80
    release_instances: int = 4
    release_streams: int = 3
    rollout_batches: int = 220
    retrain_seeds: int = 6
    retrain_batches: int = 300
    retrain_onset: int = 80
    forced_retrain_at: int = 150
    truth_batches: int = 20

    def quick(self) -> BenchConfig:
        """A small configuration with the same code paths, for tests and smoke runs."""
        return replace(
            self,
            control_streams=2,
            control_batches=120,
            drift_seeds=1,
            drift_batches=150,
            onset_range=(60, 70),
            detection_window=70,
            release_instances=1,
            release_streams=1,
            rollout_batches=200,
            retrain_seeds=1,
            retrain_batches=240,
            retrain_onset=60,
            forced_retrain_at=120,
            truth_batches=5,
        )
