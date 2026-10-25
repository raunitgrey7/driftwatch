"""Schema and data-quality checks: is this batch what the pipeline promised?

These run before any drift statistic because a broken pipeline produces
enormous drift numbers that say nothing about the world. Each check is a
rule with a statistical floor and a practical floor, so that it fires on
incidents and not on sampling noise:

* null rate: one-sided binomial test against the reference null rate
  (p below 1e-6) and at least 2 percentage points above it;
* range: more than 1 percent of values outside a band around the reference
  0.05 and 99.95 percentiles widened by a quarter of their span;
* unit change: the batch median over the reference median within 15 percent
  of a known conversion factor (10, 12, 100, 1000 or their inverses), the
  signature of monthly against annual income or rupees against thousands;
* unseen category: at least 3 values and 1 percent of rows outside the
  vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from driftwatch.alerts import Category, Signal
from driftwatch.schema import FEATURES, NULL_CODE, UNSEEN_CODE, Batch, SchemaIssue
from driftwatch.stats import binomial_upper_p

UNIT_FACTORS = (10.0, 12.0, 100.0, 1000.0)


@dataclass(frozen=True)
class QualityConfig:
    null_p_value: float = 1e-6
    null_excess: float = 0.02
    range_fraction: float = 0.01
    unit_tolerance: float = 0.15
    unseen_fraction: float = 0.01
    unseen_min_count: int = 3
    min_values: int = 20


@dataclass(frozen=True)
class QualityProfile:
    """Reference null rates, medians and valid ranges per feature."""

    null_rates: dict[str, float]
    medians: dict[str, float]
    ranges: dict[str, tuple[float, float]]

    @classmethod
    def fit(cls, reference: Batch) -> QualityProfile:
        null_rates: dict[str, float] = {}
        medians: dict[str, float] = {}
        ranges: dict[str, tuple[float, float]] = {}
        for spec in FEATURES:
            if not spec.numeric:
                null_rates[spec.name] = float(np.mean(reference.codes(spec.name) == NULL_CODE))
                continue
            values = reference.numeric[spec.name]
            present = values[~np.isnan(values)]
            null_rates[spec.name] = float(np.isnan(values).mean())
            medians[spec.name] = float(np.median(present))
            lo, hi = np.quantile(present, [0.0005, 0.9995])
            span = max(float(hi - lo), 1e-9)
            ranges[spec.name] = (float(lo - 0.25 * span), float(hi + 0.25 * span))
        return cls(null_rates, medians, ranges)


def schema_signals(issues: list[SchemaIssue]) -> list[Signal]:
    """Every contract violation is an alert-worthy schema signal."""
    return [
        Signal(
            "schema",
            Category.SCHEMA,
            issue.column,
            True,
            issue.fraction,
            0.0,
            None,
            f"{issue.kind}: {issue.column} ({issue.detail})",
        )
        for issue in issues
    ]


def _null_signal(name: str, nulls: int, n: int, ref_rate: float, cfg: QualityConfig) -> Signal:
    rate = nulls / n
    p_value = binomial_upper_p(nulls, n, max(ref_rate, 1e-4))
    fired = p_value < cfg.null_p_value and rate - ref_rate > cfg.null_excess
    message = f"null rate {rate:.1%} against {ref_rate:.1%} in the reference"
    return Signal("null_rate", Category.DATA_QUALITY, name, fired, rate, ref_rate + cfg.null_excess, p_value, message)


def _unit_factor(ratio: float, tolerance: float) -> float | None:
    for factor in UNIT_FACTORS:
        for candidate in (factor, 1.0 / factor):
            if abs(math.log(ratio / candidate)) < math.log1p(tolerance):
                return candidate
    return None


def _numeric_signals(name: str, values: np.ndarray, profile: QualityProfile, cfg: QualityConfig) -> list[Signal]:
    present = values[~np.isnan(values)]
    if len(present) < cfg.min_values:
        return []
    lo, hi = profile.ranges[name]
    outside = float(np.mean((present < lo) | (present > hi)))
    signals = [
        Signal(
            "range",
            Category.DATA_QUALITY,
            name,
            outside > cfg.range_fraction,
            outside,
            cfg.range_fraction,
            None,
            f"{outside:.1%} of values outside [{lo:.4g}, {hi:.4g}]",
        )
    ]
    ref_median = profile.medians[name]
    batch_median = float(np.median(present))
    if ref_median > 0 and batch_median > 0:
        ratio = batch_median / ref_median
        factor = _unit_factor(ratio, cfg.unit_tolerance)
        message = f"median {batch_median:.4g} is {ratio:.3g}x the reference" + (
            f", consistent with a x{factor:.4g} unit change" if factor else ""
        )
        signals.append(
            Signal(
                "unit_change", Category.DATA_QUALITY, name, factor is not None, ratio, cfg.unit_tolerance, None, message
            )
        )
    return signals


def _unseen_signal(name: str, batch: Batch, cfg: QualityConfig) -> Signal:
    unseen_mask = batch.codes(name) == UNSEEN_CODE
    count = int(unseen_mask.sum())
    fraction = count / max(batch.n_rows, 1)
    fired = count >= cfg.unseen_min_count and fraction >= cfg.unseen_fraction
    examples = ", ".join(sorted({str(v) for v in batch.categorical[name][unseen_mask].tolist()})[:3])
    message = f"{count} values outside the vocabulary" + (f" (e.g. {examples})" if count else "")
    return Signal("unseen_category", Category.DATA_QUALITY, name, fired, fraction, cfg.unseen_fraction, None, message)


def quality_signals(batch: Batch, profile: QualityProfile, cfg: QualityConfig | None = None) -> list[Signal]:
    """All data-quality evaluations for one coerced batch."""
    cfg = cfg or QualityConfig()
    n = batch.n_rows
    if n == 0:
        return []
    signals: list[Signal] = []
    for spec in FEATURES:
        if spec.numeric:
            values = batch.numeric[spec.name]
            nulls = int(np.isnan(values).sum())
            signals.append(_null_signal(spec.name, nulls, n, profile.null_rates[spec.name], cfg))
            signals.extend(_numeric_signals(spec.name, values, profile, cfg))
        else:
            nulls = int(np.sum(batch.codes(spec.name) == NULL_CODE))
            signals.append(_null_signal(spec.name, nulls, n, profile.null_rates[spec.name], cfg))
            signals.append(_unseen_signal(spec.name, batch, cfg))
    return signals
