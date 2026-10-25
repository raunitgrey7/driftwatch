"""Feature schema of a credit application and the coercion of raw batches.

The schema is the contract between the upstream application pipeline and the
model. It lives in one place because three consumers depend on it: the model
encoder (column order and vocabularies), the schema and data-quality checks
(what a well-formed batch looks like) and the drift detectors (which test fits
which feature kind). Raw batches arrive as column dictionaries whose values
may be anything JSON can carry; ``coerce`` turns them into typed arrays and
reports every contract violation instead of raising, because a monitoring
system has to describe a broken batch, not refuse it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class Kind(StrEnum):
    """How a feature is distributed, which decides the drift test applied to it."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class FeatureSpec:
    """One input column of the credit model."""

    name: str
    kind: Kind
    description: str
    vocabulary: tuple[str, ...] = ()

    @property
    def numeric(self) -> bool:
        return self.kind is not Kind.CATEGORICAL


ID_COLUMN = "application_id"

FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("age", Kind.CONTINUOUS, "applicant age in whole years"),
    FeatureSpec("monthly_income", Kind.CONTINUOUS, "declared net monthly income, INR"),
    FeatureSpec("loan_amount", Kind.CONTINUOUS, "requested principal, INR"),
    FeatureSpec("debt_to_income", Kind.CONTINUOUS, "existing monthly obligations over monthly income"),
    FeatureSpec("credit_utilization", Kind.CONTINUOUS, "revolving balance over revolving limit, 0 to 1"),
    FeatureSpec("bureau_score", Kind.CONTINUOUS, "credit bureau score, 300 to 900"),
    FeatureSpec("credit_history_months", Kind.CONTINUOUS, "age of the oldest bureau tradeline in months"),
    FeatureSpec("loan_term_months", Kind.DISCRETE, "requested tenure: 12, 24, 36, 48 or 60"),
    FeatureSpec("num_open_accounts", Kind.DISCRETE, "open tradelines at the bureau"),
    FeatureSpec("delinquencies_2y", Kind.DISCRETE, "30+ days-past-due events in the last two years"),
    FeatureSpec("inquiries_6m", Kind.DISCRETE, "hard bureau inquiries in the last six months"),
    FeatureSpec("employment_type", Kind.CATEGORICAL, "employment", ("salaried", "self_employed", "contract")),
    FeatureSpec("home_ownership", Kind.CATEGORICAL, "housing", ("rent", "own", "mortgage", "family")),
    FeatureSpec(
        "loan_purpose",
        Kind.CATEGORICAL,
        "declared purpose",
        ("personal", "consumer_durable", "two_wheeler", "education", "medical", "home_improvement"),
    ),
    FeatureSpec("channel", Kind.CATEGORICAL, "acquisition channel", ("branch", "online", "partner")),
    FeatureSpec("region", Kind.CATEGORICAL, "branch region", ("north", "south", "east", "west")),
)

FEATURE_INDEX: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}
NUMERIC_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if f.numeric)
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(f.name for f in FEATURES if not f.numeric)


@dataclass(frozen=True)
class SchemaIssue:
    """A violation of the input contract, with the share of rows affected."""

    kind: str
    column: str
    detail: str
    fraction: float


UNSEEN_CODE = -1
NULL_CODE = -2


@dataclass
class Batch:
    """A coerced batch: float arrays (NaN for null) and object arrays of ``str | None``."""

    ids: np.ndarray
    numeric: dict[str, np.ndarray]
    categorical: dict[str, np.ndarray]
    _codes: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    @property
    def n_rows(self) -> int:
        return len(self.ids)

    def codes(self, name: str) -> np.ndarray:
        """Vocabulary index per row, ``UNSEEN_CODE`` outside it and ``NULL_CODE`` for null; cached."""
        cached = self._codes.get(name)
        if cached is None:
            lookup = {v: i for i, v in enumerate(FEATURE_INDEX[name].vocabulary)}
            values = self.categorical[name].tolist()
            cached = np.array([NULL_CODE if v is None else lookup.get(v, UNSEEN_CODE) for v in values], dtype=np.int64)
            self._codes[name] = cached
        return cached

    def take(self, index: np.ndarray) -> Batch:
        """Row subset, used to split a batch between champion and canary."""
        return Batch(
            ids=self.ids[index],
            numeric={k: v[index] for k, v in self.numeric.items()},
            categorical={k: v[index] for k, v in self.categorical.items()},
            _codes={k: v[index] for k, v in self._codes.items()},
        )

    def to_columns(self) -> dict[str, np.ndarray]:
        """Back to a raw column dictionary (what an upstream system would send)."""
        return {ID_COLUMN: self.ids, **self.numeric, **self.categorical}


def concat_batches(batches: Sequence[Batch]) -> Batch:
    """Stack batches row-wise; all batches share the schema by construction."""
    if not batches:
        raise ValueError("cannot concatenate zero batches")
    return Batch(
        ids=np.concatenate([b.ids for b in batches]),
        numeric={k: np.concatenate([b.numeric[k] for b in batches]) for k in NUMERIC_FEATURES},
        categorical={k: np.concatenate([b.categorical[k] for b in batches]) for k in CATEGORICAL_FEATURES},
    )


def records_to_columns(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Turn JSON records into columns; a key missing from a record becomes ``None``."""
    keys: dict[str, None] = {}
    for record in records:
        keys.update(dict.fromkeys(record))
    return {key: [record.get(key) for record in records] for key in keys}


def _to_float(value: Any) -> tuple[float, bool]:
    """Parse one numeric cell; returns (value, ok). ``None`` and NaN are nulls, not errors."""
    if value is None:
        return math.nan, True
    if isinstance(value, bool):
        return math.nan, False
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return (number, True) if math.isfinite(number) or math.isnan(number) else (math.nan, False)
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return math.nan, False
        return (number, True) if math.isfinite(number) else (math.nan, False)
    return math.nan, False


def _coerce_numeric(values: Any) -> tuple[np.ndarray, int]:
    """Numeric column to float64; returns the array and the count of unparseable cells."""
    array = np.asarray(values)
    if array.dtype.kind in "fiu":
        out = array.astype(np.float64)
        bad = ~np.isfinite(out) & ~np.isnan(out)
        out[bad] = np.nan
        return out, int(bad.sum())
    parsed = [_to_float(v) for v in array.tolist()]
    out = np.array([p[0] for p in parsed], dtype=np.float64)
    return out, sum(not p[1] for p in parsed)


def _coerce_categorical(values: Any) -> np.ndarray:
    """Categorical column to an object array of ``str | None``; numbers become their text."""
    items = np.asarray(values, dtype=object).tolist()
    return np.array(
        [None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in items], dtype=object
    )


def coerce(raw: Mapping[str, Any]) -> tuple[Batch, list[SchemaIssue]]:
    """Validate a raw column dictionary against the schema and return a typed batch.

    Missing columns become all-null columns, unexpected columns are dropped and
    unparseable numeric cells become null; each of these is reported as a
    ``SchemaIssue`` so the schema check can alert on it.
    """
    lengths = {len(v) for v in raw.values()}
    if len(lengths) > 1:
        raise ValueError(f"columns have different lengths: {sorted(lengths)}")
    n = lengths.pop() if lengths else 0
    issues: list[SchemaIssue] = []
    ids = raw.get(ID_COLUMN)
    ids_array = (
        np.array([str(i) for i in np.asarray(ids, dtype=object).tolist()], dtype=object)
        if ids is not None
        else np.array([f"row-{i}" for i in range(n)], dtype=object)
    )
    for column in raw:
        if column != ID_COLUMN and column not in FEATURE_INDEX:
            issues.append(SchemaIssue("unexpected_column", column, "column is not part of the schema", 1.0))
    numeric: dict[str, np.ndarray] = {}
    categorical: dict[str, np.ndarray] = {}
    for spec in FEATURES:
        if spec.name not in raw:
            issues.append(SchemaIssue("missing_column", spec.name, "required column is absent", 1.0))
            if spec.numeric:
                numeric[spec.name] = np.full(n, np.nan)
            else:
                categorical[spec.name] = np.full(n, None, dtype=object)
            continue
        if spec.numeric:
            numeric[spec.name], bad = _coerce_numeric(raw[spec.name])
            if bad:
                issues.append(SchemaIssue("type_mismatch", spec.name, f"{bad} values are not numbers", bad / max(n, 1)))
        else:
            categorical[spec.name] = _coerce_categorical(raw[spec.name])
    return Batch(ids=ids_array, numeric=numeric, categorical=categorical), issues
