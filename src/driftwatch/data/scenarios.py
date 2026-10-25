"""Scenario streams: daily batches of applications with one event at a known batch.

Each scenario changes exactly one mechanism so a detector's hits and misses
can be attributed:

* covariate shift changes P(x) and leaves P(y | x) alone (a partner campaign
  brings younger, thinner-file, lower-income applicants);
* prior shift changes P(y) and leaves P(x | y) alone, implemented by
  rejection sampling on the label so the class-conditional distributions are
  exactly preserved (a downturn raises the default rate);
* concept drift changes P(y | x) and leaves P(x) alone (a rate rise makes
  affordability matter more and the bureau score less);
* data-quality incidents leave the world unchanged and break the pipeline:
  nulls from a bureau outage, a unit change in income, a renamed column, a
  new category value.

Every batch is drawn from its own seed (``SeedSequence(seed, spawn_key=(i,))``),
so a scenario stream and a control stream with the same seed are identical
before the onset, and any batch can be regenerated on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta
from functools import cache

import numpy as np

from driftwatch.data.generator import (
    BASELINE_NULL_RATES,
    Concept,
    Population,
    apply_missingness,
    default_probability,
    lerp,
    sample_applicants,
)
from driftwatch.schema import ID_COLUMN

BASE_POPULATION = Population()
BASE_CONCEPT = Concept()

# Partner-channel campaign: younger, lower income, weaker and more homogeneous bureau profile.
COVARIATE_TARGET = replace(
    BASE_POPULATION,
    z_mean=-0.45,
    z_sd=0.80,
    age_scale=3.8,
    income_log_median=math.log(42_000.0) - 0.25,
    channel_probs=(0.22, 0.30, 0.48),
)
# Rate rise: affordability (FOIR) matters 2.5x more, the bureau score 60 percent less, and
# self-employed cash flow is squeezed. The risk ranking changes, not only the level.
CONCEPT_TARGET = replace(
    BASE_CONCEPT,
    intercept=BASE_CONCEPT.intercept + 0.10,
    bureau=BASE_CONCEPT.bureau * 0.4,
    foir=BASE_CONCEPT.foir * 2.5,
    self_employed=BASE_CONCEPT.self_employed + 0.70,
)
PRIOR_TARGET_MULTIPLIER = 1.75
NULL_INCIDENT_RATE = 0.35


@dataclass(frozen=True)
class Scenario:
    """One drift mechanism with its ground-truth family and shape."""

    name: str
    family: str
    ramp: int
    severity: float
    description: str


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("control", "none", 0, 0.0, "stationary stream, no event"),
        Scenario("covariate_abrupt", "covariate", 0, 1.0, "partner campaign changes the applicant mix overnight"),
        Scenario("covariate_gradual", "covariate", 40, 1.0, "the same campaign ramps up over 40 batches"),
        Scenario("covariate_subtle", "covariate", 0, 0.3, "30 percent of the campaign shift, overnight"),
        Scenario("prior_shift", "prior", 0, 1.0, "default rate rises 1.75x with P(x | y) unchanged"),
        Scenario("concept_abrupt", "concept", 0, 1.0, "rate rise changes P(y | x) overnight"),
        Scenario("concept_gradual", "concept", 40, 1.0, "the same rate rise over 40 batches"),
        Scenario("dq_nulls", "quality", 0, 1.0, "bureau outage: 35 percent of bureau_score values are null"),
        Scenario("dq_unit_change", "quality", 0, 1.0, "monthly_income starts arriving as annual income"),
        Scenario("dq_schema_break", "quality", 0, 1.0, "credit_utilization renamed and sent in percent"),
        Scenario("dq_new_category", "quality", 0, 1.0, "a new employment_type value 'gig' appears"),
    )
}


@dataclass(frozen=True)
class StreamConfig:
    """Shape of one simulated stream; ``onset`` is the first drifted batch."""

    scenario: str = "control"
    seed: int = 0
    n_batches: int = 240
    batch_size: int = 400
    onset: int = 10**9
    start: date = date(2026, 1, 1)

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario!r}")
        if self.batch_size < 1 or self.n_batches < 1:
            raise ValueError("batch_size and n_batches must be positive")


@dataclass
class StreamBatch:
    """What the pipeline delivers (``raw``) plus the ground truth only the benchmark sees."""

    index: int
    timestamp: str
    raw: dict[str, np.ndarray]
    labels: np.ndarray
    true_pd: np.ndarray
    drift_weight: float


@cache
def base_default_rate() -> float:
    """Mean true PD of the base population, by a fixed-seed Monte Carlo of 400k applicants."""
    rng = np.random.default_rng(20_261_001)
    columns = sample_applicants(rng, 400_000, BASE_POPULATION)
    return float(default_probability(columns, BASE_CONCEPT).mean())


def drift_weight(config: StreamConfig, index: int) -> float:
    """0 before the onset, then a linear ramp (or a step) up to the scenario severity."""
    scenario = SCENARIOS[config.scenario]
    if scenario.family == "none" or index < config.onset:
        return 0.0
    progress = 1.0 if scenario.ramp == 0 else min(1.0, (index - config.onset + 1) / scenario.ramp)
    return progress * scenario.severity


def _prior_shift_sample(
    rng: np.random.Generator, n: int, pop: Population, concept: Concept, target: float
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Rejection sampling on the label: keeps P(x | y) exact while moving P(y) to ``target``.

    The true PD of an accepted row is the posterior after acceptance,
    ``a1 q / (a1 q + a0 (1 - q))``, not the pre-shift ``q``.
    """
    base = base_default_rate()
    accept_neg = min(1.0, base * (1.0 - target) / (target * (1.0 - base)))
    accept_pos = min(1.0, target * (1.0 - base) / (base * (1.0 - target)))
    parts: list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
    kept = 0
    while kept < n:
        draw = 64 + int(1.25 * (n - kept) / (base * accept_pos + (1.0 - base) * accept_neg))
        columns = sample_applicants(rng, draw, pop)
        pd = default_probability(columns, concept)
        labels = (rng.random(draw) < pd).astype(np.int8)
        keep = rng.random(draw) < np.where(labels == 1, accept_pos, accept_neg)
        posterior = accept_pos * pd / (accept_pos * pd + accept_neg * (1.0 - pd))
        parts.append(({k: v[keep] for k, v in columns.items()}, labels[keep], posterior[keep]))
        kept += int(keep.sum())
    columns = {k: np.concatenate([p[0][k] for p in parts])[:n] for k in parts[0][0]}
    return columns, np.concatenate([p[1] for p in parts])[:n], np.concatenate([p[2] for p in parts])[:n]


def _apply_incident(rng: np.random.Generator, scenario: str, raw: dict[str, np.ndarray], weight: float) -> None:
    """Corrupt the delivered columns in place; the true PD was computed on clean data first."""
    if weight <= 0.0:
        return
    n = len(raw[ID_COLUMN])
    if scenario == "dq_nulls":
        column = raw["bureau_score"].copy()
        column[rng.random(n) < NULL_INCIDENT_RATE * weight] = np.nan
        raw["bureau_score"] = column
    elif scenario == "dq_unit_change":
        raw["monthly_income"] = raw["monthly_income"] * 12.0
    elif scenario == "dq_schema_break":
        raw["credit_utilization_pct"] = np.round(raw.pop("credit_utilization") * 100.0, 1)
    elif scenario == "dq_new_category":
        employment = raw["employment_type"].copy()
        switch = np.where(employment == "contract", 0.7, np.where(employment == "self_employed", 0.2, 0.0))
        employment[rng.random(n) < switch * weight] = "gig"
        raw["employment_type"] = employment


class CreditStream:
    """A deterministic, random-access stream of daily application batches."""

    def __init__(self, config: StreamConfig) -> None:
        self.config = config
        self.scenario = SCENARIOS[config.scenario]

    def __len__(self) -> int:
        return self.config.n_batches

    def __iter__(self):  # noqa: ANN204 - a generator of StreamBatch
        for index in range(self.config.n_batches):
            yield self.batch(index)

    def batch(self, index: int) -> StreamBatch:
        """Generate batch ``index``; the same config and index always give the same batch."""
        cfg = self.config
        rng = np.random.default_rng(np.random.SeedSequence(cfg.seed, spawn_key=(index,)))
        weight = drift_weight(cfg, index)
        family = self.scenario.family
        pop = lerp(BASE_POPULATION, COVARIATE_TARGET, weight) if family == "covariate" else BASE_POPULATION
        concept = lerp(BASE_CONCEPT, CONCEPT_TARGET, weight) if family == "concept" else BASE_CONCEPT
        if family == "prior" and weight > 0.0:
            base = base_default_rate()
            target = base * (1.0 + weight * (PRIOR_TARGET_MULTIPLIER - 1.0))
            columns, labels, pd = _prior_shift_sample(rng, cfg.batch_size, pop, concept, target)
        else:
            columns = sample_applicants(rng, cfg.batch_size, pop)
            pd = default_probability(columns, concept)
            labels = (rng.random(cfg.batch_size) < pd).astype(np.int8)
        raw = apply_missingness(rng, columns, BASELINE_NULL_RATES)
        raw[ID_COLUMN] = np.array([f"s{cfg.seed}-b{index}-{i}" for i in range(cfg.batch_size)], dtype=object)
        if family == "quality":
            _apply_incident(rng, cfg.scenario, raw, weight)
        timestamp = (cfg.start + timedelta(days=index)).isoformat()
        return StreamBatch(index, timestamp, raw, labels, pd, weight)


def clean_history(seed: int, n_batches: int, batch_size: int = 400) -> list[StreamBatch]:
    """The stationary period the reference model is trained and profiled on."""
    stream = CreditStream(StreamConfig(scenario="control", seed=seed, n_batches=n_batches, batch_size=batch_size))
    return list(stream)
