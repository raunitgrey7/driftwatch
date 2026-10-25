"""Data-drift and prediction-drift detectors fitted once on a clean reference.

Per feature, the profile keeps what each test needs from the reference so a
batch never touches the full reference again:

* continuous features: the sorted reference for the two-sample KS test and
  decile edges for PSI;
* discrete features (tenure, counts): one bin per reference value with the
  upper tail above the 99th percentile pooled, used by both chi-square and PSI;
* categorical features: the schema vocabulary.

Nulls and values outside the vocabulary are excluded from these statistics on
purpose: they are data-quality findings with their own checks, and mixing
them in would make a drift alert fire for a pipeline fault.

The multivariate test is a kernel two-sample statistic (MMD) with an RBF
kernel approximated by random Fourier features (Rahimi and Recht 2007), so a
batch costs one small matrix product. Its null distribution at a given batch
size is bootstrapped from a clean pool disjoint from the anchor sample and
summarised by a Gamma fit (Gretton et al. 2009), which gives p-values far in
the tail without thousands of permutations per batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from driftwatch.schema import FEATURES, NUMERIC_FEATURES, Batch, FeatureSpec, Kind
from driftwatch.stats import chi2_homogeneity, gamma_fit, gamma_sf, ks_pvalue, ks_statistic, psi

MIN_EXPECTED = 5.0
LOG_SCALED = ("monthly_income", "loan_amount", "credit_history_months")


@dataclass(frozen=True)
class FeatureProfile:
    """Reference statistics of one feature."""

    spec: FeatureSpec
    edges: np.ndarray
    ref_counts: np.ndarray
    sorted_values: np.ndarray | None = None

    def bin_counts(self, values: np.ndarray) -> np.ndarray:
        """Counts per reference bin of the non-null, in-vocabulary values (codes for categoricals)."""
        if self.spec.kind is Kind.CATEGORICAL:
            return np.bincount(values[values >= 0], minlength=len(self.spec.vocabulary)).astype(np.float64)
        x = values[~np.isnan(values)]
        return np.bincount(np.searchsorted(self.edges, x, side="right"), minlength=len(self.edges) + 1).astype(
            np.float64
        )


def _numeric_edges(spec: FeatureSpec, x: np.ndarray) -> np.ndarray:
    if spec.kind is Kind.CONTINUOUS:
        return np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9)))
    values = np.unique(x[x <= np.quantile(x, 0.99)])
    return (values[:-1] + values[1:]) / 2.0


def fit_feature_profile(spec: FeatureSpec, values: np.ndarray) -> FeatureProfile:
    """Profile one reference column."""
    if spec.kind is Kind.CATEGORICAL:
        counts = np.bincount(values[values >= 0], minlength=len(spec.vocabulary)).astype(np.float64)
        return FeatureProfile(spec, np.empty(0), counts)
    x = values[~np.isnan(values)]
    edges = _numeric_edges(spec, x)
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1).astype(np.float64)
    sorted_values = np.sort(x) if spec.kind is Kind.CONTINUOUS else None
    return FeatureProfile(spec, edges, counts, sorted_values)


@dataclass(frozen=True)
class FeatureDrift:
    """The drift statistics of one feature in one batch."""

    feature: str
    test: str
    statistic: float
    p_value: float
    psi: float
    n: int


def _merge_sparse(ref: np.ndarray, batch: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pool bins whose expected batch count is under five so the chi-square approximation holds."""
    expected = ref / max(ref.sum(), 1.0) * n
    small = expected < MIN_EXPECTED
    if not small.any() or small.all():
        return ref, batch
    return (
        np.append(ref[~small], ref[small].sum()),
        np.append(batch[~small], batch[small].sum()),
    )


def feature_drift(profile: FeatureProfile, values: np.ndarray) -> FeatureDrift | None:
    """KS (continuous) or chi-square (discrete, categorical) plus PSI; None when too few values."""
    counts = profile.bin_counts(values)
    n = int(counts.sum())
    if n < 20:
        return None
    population_index = psi(profile.ref_counts, counts)
    if profile.sorted_values is not None:
        x = values[~np.isnan(values)]
        d = ks_statistic(profile.sorted_values, x)
        return FeatureDrift(
            profile.spec.name, "ks", d, ks_pvalue(d, len(x), len(profile.sorted_values)), population_index, n
        )
    ref, batch = _merge_sparse(profile.ref_counts, counts, n)
    statistic, _, p_value = chi2_homogeneity(ref, batch)
    return FeatureDrift(profile.spec.name, "chi2", statistic, p_value, population_index, n)


def batch_column(batch: Batch, name: str) -> np.ndarray:
    """Float values for numeric features, vocabulary codes for categorical ones."""
    return batch.numeric[name] if name in batch.numeric else batch.codes(name)


@dataclass
class UnivariateDrift:
    """All per-feature profiles of the reference."""

    profiles: dict[str, FeatureProfile]

    @classmethod
    def fit(cls, reference: Batch) -> UnivariateDrift:
        return cls({spec.name: fit_feature_profile(spec, batch_column(reference, spec.name)) for spec in FEATURES})

    def evaluate(self, batch: Batch) -> list[FeatureDrift]:
        results = (feature_drift(p, batch_column(batch, name)) for name, p in self.profiles.items())
        return [r for r in results if r is not None]


@dataclass
class ScoreDrift:
    """KS test of the model's output distribution against its reference outputs (prediction drift)."""

    reference_sorted: np.ndarray

    @classmethod
    def fit(cls, reference_scores: np.ndarray) -> ScoreDrift:
        return cls(np.sort(np.asarray(reference_scores, dtype=np.float64)))

    def evaluate(self, scores: np.ndarray) -> tuple[float, float]:
        d = ks_statistic(self.reference_sorted, scores)
        return d, ks_pvalue(d, len(scores), len(self.reference_sorted))


@dataclass
class MMDDetector:
    """Kernel two-sample test on standardised numeric features with a bootstrapped Gamma null."""

    center: np.ndarray
    scale: np.ndarray
    fill: np.ndarray
    omega: np.ndarray
    phase: np.ndarray
    anchor: np.ndarray
    pool: np.ndarray
    seed: int
    n_boot: int = 400
    _null: dict[int, tuple[float, float]] = field(default_factory=dict)

    def _standardise(self, batch: Batch) -> np.ndarray:
        x = self._matrix(batch)
        return (np.where(np.isnan(x), self.fill, x) - self.center) / self.scale

    def _embed(self, z: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.omega.shape[1]) * np.cos(z @ self.omega + self.phase)

    @staticmethod
    def _matrix(batch: Batch) -> np.ndarray:
        cols = [
            np.log1p(np.maximum(batch.numeric[n], 0.0)) if n in LOG_SCALED else batch.numeric[n]
            for n in NUMERIC_FEATURES
        ]
        return np.column_stack(cols)

    @classmethod
    def fit(cls, anchor: Batch, pool: Batch, *, n_features: int = 128, seed: int = 0) -> MMDDetector:
        """Standardise on the anchor, set the bandwidth by the median heuristic, embed both samples."""
        rng = np.random.default_rng(seed)
        x = cls._matrix(anchor)
        fill = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), fill, x)
        center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-9)
        z = (x - center) / scale
        sub = z[rng.choice(len(z), size=min(len(z), 1000), replace=False)]
        distances = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))
        bandwidth = float(np.median(distances[np.triu_indices(len(sub), k=1)]))
        omega = rng.standard_normal((z.shape[1], n_features)) / bandwidth
        phase = rng.uniform(0.0, 2.0 * np.pi, n_features)
        detector = cls(center, scale, fill, omega, phase, np.empty(0), np.empty((0, 0)), seed)
        detector.anchor = detector._embed(z).mean(axis=0)
        detector.pool = detector._standardise(pool).astype(np.float32)
        return detector

    def statistic(self, batch: Batch) -> float:
        diff = self._embed(self._standardise(batch)).mean(axis=0) - self.anchor
        return float(diff @ diff)

    def null(self, n: int) -> tuple[float, float]:
        """Gamma fit of the statistic for clean batches of size ``n``, cached per size.

        The pool is kept as standardised rows (small enough to ship inside a
        registry artifact) and embedded only when a new batch size is seen.
        """
        if n not in self._null:
            rng = np.random.default_rng([self.seed, n])
            embedded = self._embed(self.pool.astype(np.float64))
            stats = np.empty(self.n_boot)
            for start in range(0, self.n_boot, 25):
                draws = rng.integers(0, len(embedded), size=(min(25, self.n_boot - start), n))
                diffs = embedded[draws].mean(axis=1) - self.anchor
                stats[start : start + len(draws)] = np.einsum("ij,ij->i", diffs, diffs)
            self._null[n] = gamma_fit(stats)
        return self._null[n]

    def test(self, batch: Batch) -> tuple[float, float]:
        stat = self.statistic(batch)
        shape, scale = self.null(batch.n_rows)
        return stat, gamma_sf(stat, shape, scale)
