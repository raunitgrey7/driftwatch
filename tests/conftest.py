"""Shared fixtures: the reference champion and streams are expensive, so they are built once."""

from __future__ import annotations

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.schema import Batch, coerce
from driftwatch.training import Champion, HistoryConfig, LabeledData, build_champion


@pytest.fixture(scope="session", autouse=True)
def _single_thread():
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="session")
def champion() -> Champion:
    return build_champion(HistoryConfig())


@pytest.fixture(scope="session")
def clean_sample() -> LabeledData:
    """A large clean sample from a seed the champion never saw."""
    return LabeledData.from_stream(list(CreditStream(StreamConfig(seed=4242, n_batches=30))))


def stream_batch(scenario: str, index: int, seed: int = 5, onset: int = 0) -> Batch:
    """One coerced batch of a scenario."""
    raw = CreditStream(StreamConfig(scenario=scenario, seed=seed, n_batches=index + 1, onset=onset)).batch(index).raw
    return coerce(raw)[0]


def stream_data(scenario: str, start: int, stop: int, seed: int = 5, onset: int = 0) -> LabeledData:
    stream = CreditStream(StreamConfig(scenario=scenario, seed=seed, n_batches=stop, onset=onset))
    return LabeledData.from_stream([stream.batch(i) for i in range(start, stop)])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
