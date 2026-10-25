"""Seeded generator of credit applications with a known default mechanism.

Drift detection can only be scored against ground truth, so the generator
knows, for every row, the true probability of default, and for every batch
which mechanism changed. The structure is chosen so that the observed
features are correlated the way bureau and application data are:

* A latent creditworthiness ``z ~ N(z_mean, z_sd)`` per applicant drives the
  bureau score, revolving utilisation and past delinquencies. Shifting the
  distribution of ``z`` is a covariate shift: it moves several correlated
  features at once, as a change in the acquisition mix does.
* Age is ``21 + Gamma(k, theta)``, the right-skewed profile of retail loan
  applicants. Income is log-normal with a concave age profile (earnings peak
  in the late forties) and employment-type offsets.
* Counts (delinquencies, inquiries, open tradelines) are Poisson with
  log-linear rates, the standard model for event counts.
* The requested loan is a log-normal multiple of income; the instalment
  (EMI) follows the annuity formula at a 15 percent annual rate, and the
  fixed-obligation-to-income ratio (FOIR = existing obligations plus the new
  EMI, over income) is the affordability measure Indian lenders underwrite on.
* Default within 12 months is ``Bernoulli(sigmoid(eta(x)))``, with ``eta`` an
  explicit scorecard-like function of observed features plus one interaction
  (self-employed applicants with stretched FOIR) and one non-linearity (age).
  Because ``eta`` uses observed features only, ``P(default | x)`` is known
  exactly, which lets the benchmark separate estimation error from sampling
  noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np

from driftwatch.schema import FEATURE_INDEX, NUMERIC_FEATURES

ANNUAL_RATE = 0.15
TERMS = (12, 24, 36, 48, 60)
PURPOSE_LOAN_OFFSET = {
    "personal": 0.0,
    "consumer_durable": -0.6,
    "two_wheeler": 0.1,
    "education": 0.5,
    "medical": -0.2,
    "home_improvement": 0.4,
}


@dataclass(frozen=True)
class Population:
    """Parameters of the applicant distribution P(x)."""

    z_mean: float = 0.0
    z_sd: float = 1.0
    age_shape: float = 3.2
    age_scale: float = 5.0
    income_log_median: float = math.log(42_000.0)
    income_sigma: float = 0.55
    loan_log_multiple: float = math.log(3.0)
    employment_probs: tuple[float, ...] = (0.60, 0.25, 0.15)
    channel_probs: tuple[float, ...] = (0.40, 0.42, 0.18)
    purpose_probs: tuple[float, ...] = (0.34, 0.20, 0.16, 0.10, 0.08, 0.12)
    region_probs: tuple[float, ...] = (0.28, 0.30, 0.18, 0.24)
    term_probs: tuple[float, ...] = (0.14, 0.30, 0.30, 0.16, 0.10)


@dataclass(frozen=True)
class Concept:
    """Coefficients of the true default mechanism P(y | x); changing them is concept drift."""

    intercept: float = -3.49
    bureau: float = -0.60
    utilization: float = 0.30
    foir: float = 0.34
    delinquency: float = 0.41
    inquiries: float = 0.11
    income: float = -0.19
    age_curvature: float = 0.15
    contract: float = 0.26
    self_employed: float = 0.08
    self_employed_foir: float = 0.22
    thin_file: float = 0.26
    medical: float = 0.19
    partner_channel: float = 0.11


BASELINE_NULL_RATES: dict[str, float] = {
    "num_open_accounts": 0.012,
    "credit_history_months": 0.006,
    "monthly_income": 0.004,
    "home_ownership": 0.010,
}


def lerp[T: (Population, Concept)](start: T, end: T, weight: float) -> T:
    """Linear interpolation between two parameter sets; mixtures of probabilities stay valid."""
    if weight <= 0.0:
        return start
    if weight >= 1.0:
        return end
    changes: dict[str, object] = {}
    for f in fields(start):
        a, b = getattr(start, f.name), getattr(end, f.name)
        if isinstance(a, tuple):
            changes[f.name] = tuple(x + weight * (y - x) for x, y in zip(a, b, strict=True))
        else:
            changes[f.name] = a + weight * (b - a)
    return replace(start, **changes)


def _categorical(rng: np.random.Generator, n: int, labels: tuple[str, ...], probs: tuple[float, ...]) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    index = rng.choice(len(labels), size=n, p=p / p.sum())
    return np.asarray(labels, dtype=object)[index]


def _home_ownership(rng: np.random.Generator, age: np.ndarray) -> np.ndarray:
    """Housing depends on age (ownership rises, living with family falls); Gumbel-max sampling."""
    years = age - 21.0
    logits = np.stack(
        [np.zeros_like(years), -1.6 + 0.06 * years, -1.1 + 0.04 * years, 0.9 - 0.09 * years],
        axis=1,
    )
    choice = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)
    return np.asarray(FEATURE_INDEX["home_ownership"].vocabulary, dtype=object)[choice]


def _bureau_block(rng: np.random.Generator, z: np.ndarray, channel: np.ndarray) -> dict[str, np.ndarray]:
    """Features read from the credit bureau, all driven by the latent creditworthiness."""
    n = len(z)
    bureau = np.clip(np.round(700.0 + 60.0 * z + 20.0 * rng.standard_normal(n)), 300, 900)
    utilization = 1.0 / (1.0 + np.exp(0.4 + 0.8 * z - 0.6 * rng.standard_normal(n)))
    delinquencies = rng.poisson(np.exp(-1.8 - 0.9 * z))
    inquiry_rate = np.exp(0.05 - 0.30 * z + 0.20 * (channel == "online") + 0.35 * (channel == "partner"))
    return {
        "bureau_score": bureau,
        "credit_utilization": np.round(utilization, 3),
        "delinquencies_2y": delinquencies.astype(np.float64),
        "inquiries_6m": rng.poisson(inquiry_rate).astype(np.float64),
    }


def sample_applicants(rng: np.random.Generator, n: int, pop: Population) -> dict[str, np.ndarray]:
    """Draw ``n`` complete applications (no nulls yet) from the population ``pop``."""
    z = rng.normal(pop.z_mean, pop.z_sd, n)
    age = np.clip(np.round(21.0 + rng.gamma(pop.age_shape, pop.age_scale, n)), 21, 70)
    employment = _categorical(rng, n, FEATURE_INDEX["employment_type"].vocabulary, pop.employment_probs)
    channel = _categorical(rng, n, FEATURE_INDEX["channel"].vocabulary, pop.channel_probs)
    purpose = _categorical(rng, n, FEATURE_INDEX["loan_purpose"].vocabulary, pop.purpose_probs)
    emp_offset = np.select([employment == "self_employed", employment == "contract"], [0.05, -0.30], 0.0)
    log_income = (
        pop.income_log_median
        + 0.03 * (age - 35.0)
        - 0.0012 * (age - 35.0) ** 2
        + emp_offset
        + 0.12 * z
        + pop.income_sigma * rng.standard_normal(n)
    )
    income = np.maximum(np.round(np.exp(log_income) / 500.0) * 500.0, 8_000.0)
    loan_offset = np.array([PURPOSE_LOAN_OFFSET[p] for p in purpose.tolist()])
    loan = np.exp(np.log(income) + pop.loan_log_multiple + loan_offset + 0.45 * rng.standard_normal(n))
    bureau = _bureau_block(rng, z, channel)
    history = np.round((age - 20.0) * 12.0 * rng.beta(2.2, 2.0, n))
    open_accounts = rng.poisson(0.6 + history / 48.0 + 0.3 * (bureau["credit_utilization"] > 0.5))
    dti = rng.beta(2.0, 6.0, n) + 0.08 * bureau["credit_utilization"] + 0.02 * open_accounts
    columns: dict[str, np.ndarray] = {
        "age": age,
        "monthly_income": income,
        "loan_amount": np.clip(np.round(loan / 1000.0) * 1000.0, 10_000.0, 3_000_000.0),
        "debt_to_income": np.round(np.clip(dti, 0.0, 1.5), 3),
        "credit_history_months": history,
        "loan_term_months": np.asarray(TERMS, dtype=np.float64)[rng.choice(len(TERMS), size=n, p=pop.term_probs)],
        "num_open_accounts": open_accounts.astype(np.float64),
        "employment_type": employment,
        "home_ownership": _home_ownership(rng, age),
        "loan_purpose": purpose,
        "channel": channel,
        "region": _categorical(rng, n, FEATURE_INDEX["region"].vocabulary, pop.region_probs),
        **bureau,
    }
    return columns


def monthly_instalment(principal: np.ndarray, term_months: np.ndarray, annual_rate: float = ANNUAL_RATE) -> np.ndarray:
    """Annuity instalment: P r (1 + r)^T / ((1 + r)^T - 1) with monthly rate r."""
    r = annual_rate / 12.0
    growth = (1.0 + r) ** term_months
    return principal * r * growth / (growth - 1.0)


def foir(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Fixed obligations (existing plus the new instalment) over monthly income."""
    emi = monthly_instalment(columns["loan_amount"], columns["loan_term_months"])
    return columns["debt_to_income"] + emi / columns["monthly_income"]


def true_logit(columns: dict[str, np.ndarray], c: Concept) -> np.ndarray:
    """The log-odds of default for complete applications under concept ``c``."""
    obligations = foir(columns)
    self_employed = columns["employment_type"] == "self_employed"
    history = np.minimum(columns["credit_history_months"], 36.0)
    return (
        c.intercept
        + c.bureau * (columns["bureau_score"] - 700.0) / 60.0
        + c.utilization * (columns["credit_utilization"] - 0.40) / 0.20
        + c.foir * (obligations - 0.45) / 0.15
        + c.delinquency * np.log1p(columns["delinquencies_2y"])
        + c.inquiries * (columns["inquiries_6m"] - 1.0)
        + c.income * np.log(columns["monthly_income"] / 42_000.0)
        + c.age_curvature * ((columns["age"] - 40.0) / 12.0) ** 2
        + c.contract * (columns["employment_type"] == "contract")
        + c.self_employed * self_employed
        + c.self_employed_foir * self_employed * np.maximum(obligations - 0.5, 0.0) / 0.1
        + c.thin_file * (1.0 - history / 36.0)
        + c.medical * (columns["loan_purpose"] == "medical")
        + c.partner_channel * (columns["channel"] == "partner")
    )


def default_probability(columns: dict[str, np.ndarray], concept: Concept) -> np.ndarray:
    """True P(default within 12 months | x)."""
    return 1.0 / (1.0 + np.exp(-true_logit(columns, concept)))


def apply_missingness(
    rng: np.random.Generator, columns: dict[str, np.ndarray], null_rates: dict[str, float]
) -> dict[str, np.ndarray]:
    """Blank cells completely at random at the given per-column rates (a copy is returned)."""
    out = dict(columns)
    for name, rate in null_rates.items():
        mask = rng.random(len(out[name])) < rate
        if not mask.any():
            continue
        column = out[name].copy()
        column[mask] = np.nan if name in NUMERIC_FEATURES else None
        out[name] = column
    return out
