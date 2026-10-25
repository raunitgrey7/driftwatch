"""From a clean labelled history to a champion model and its monitoring reference.

The history is split by time, never shuffled, because the model will be used
on the future: the first ``train_batches`` fit the booster, the next
``calib_batches`` fit the calibration and the cut-off, and everything after
the training slice is the drift reference. The MMD null pool is the training
slice, disjoint from the reference anchor, so the bootstrapped null is not
computed against the sample it is compared with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from driftwatch.data.scenarios import StreamBatch, clean_history
from driftwatch.model import CreditModel, TrainConfig, train_credit_model
from driftwatch.monitor import MonitorConfig, Reference
from driftwatch.schema import Batch, coerce, concat_batches


@dataclass(frozen=True)
class HistoryConfig:
    """Size and seed of the clean period the champion is trained on."""

    seed: int = 7
    n_batches: int = 60
    batch_size: int = 400
    train_batches: int = 40
    calib_batches: int = 10

    def __post_init__(self) -> None:
        if self.train_batches + self.calib_batches >= self.n_batches:
            raise ValueError("the history needs batches left over for the reference")


@dataclass
class LabeledData:
    """Coerced rows with their outcomes (and, in simulation, the true PD)."""

    batch: Batch
    labels: np.ndarray
    true_pd: np.ndarray = field(default_factory=lambda: np.empty(0))

    @classmethod
    def from_stream(cls, batches: list[StreamBatch]) -> LabeledData:
        return cls(
            batch=concat_batches([coerce(b.raw)[0] for b in batches]),
            labels=np.concatenate([b.labels for b in batches]),
            true_pd=np.concatenate([b.true_pd for b in batches]),
        )

    def split(self, fraction: float) -> tuple[LabeledData, LabeledData]:
        """Time-ordered split: the first ``fraction`` of rows, then the rest."""
        cut = round(self.batch.n_rows * fraction)
        head, tail = np.arange(cut), np.arange(cut, self.batch.n_rows)
        return self.take(head), self.take(tail)

    def take(self, index: np.ndarray) -> LabeledData:
        true_pd = self.true_pd[index] if len(self.true_pd) else self.true_pd
        return LabeledData(self.batch.take(index), self.labels[index], true_pd)


@dataclass
class Champion:
    """A trained model, its monitoring reference and the data both came from."""

    model: CreditModel
    reference: Reference
    train: LabeledData
    calib: LabeledData
    holdout: LabeledData


def fit_on(train: LabeledData, calib: LabeledData, seed: int, config: TrainConfig | None = None) -> CreditModel:
    return train_credit_model(train.batch, train.labels, calib.batch, calib.labels, seed=seed, config=config)


def build_champion(
    history: HistoryConfig | None = None,
    monitor: MonitorConfig | None = None,
    train_config: TrainConfig | None = None,
) -> Champion:
    """Generate the clean history, train the champion and fit every detector reference."""
    h = history or HistoryConfig()
    batches = clean_history(h.seed, h.n_batches, h.batch_size)
    train = LabeledData.from_stream(batches[: h.train_batches])
    calib = LabeledData.from_stream(batches[h.train_batches : h.train_batches + h.calib_batches])
    reference = LabeledData.from_stream(batches[h.train_batches :])
    holdout = LabeledData.from_stream(batches[h.train_batches + h.calib_batches :])
    model = fit_on(train, calib, seed=h.seed, config=train_config)
    model.info["history_seed"] = h.seed
    profile = Reference.build(model, reference.batch, reference.labels, train.batch, monitor, seed=h.seed)
    return Champion(model, profile, train, calib, holdout)
