"""Deployment-safety benchmark: planted releases through the real control loop.

Each run registers the reference champion and one planted release in a fresh
registry, starts the release's rollout and streams stationary traffic with
labels arriving ``label_delay`` batches late, until the loop promotes or rolls
the release back. A run is correct when the registry ends in the right state:
a bad release ``rolled_back`` with the champion still in production, or a
good release in production with the champion archived.

Besides the decision, each run records how long it took, how many
applications the candidate decided (exposure) and the harm of those
decisions in the simulation's ground truth: defaulters approved by the
candidate that the champion would have rejected, net of the reverse.
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from driftwatch.bench.detection import champion_for
from driftwatch.bench.releases import BAD, GOOD, build_release
from driftwatch.data.scenarios import CreditStream, StreamConfig
from driftwatch.loop import ControlLoop, LoopConfig, ModelBundle
from driftwatch.model import approve
from driftwatch.registry import ModelRegistry, Stage
from driftwatch.stats import expected_auc
from driftwatch.store import Store
from driftwatch.training import HistoryConfig, LabeledData


@dataclass(frozen=True)
class RolloutTask:
    kind: str
    release_seed: int
    stream_seed: int
    n_batches: int
    batch_size: int
    label_delay: int
    history: HistoryConfig


@dataclass
class RolloutSummary:
    task: RolloutTask
    status: str
    batches: int
    exposure: int
    reason: str
    guardrail: str
    excess_defaults_approved: int
    registry_correct: bool
    release_true_auc: float
    seconds: float


release_for = cache(build_release)


@cache
def truth_sample(seed: int, batches: int, batch_size: int) -> LabeledData:
    """A large clean sample with true PDs, for the noise-free AUC of each release."""
    return LabeledData.from_stream(
        list(CreditStream(StreamConfig(seed=seed, n_batches=batches, batch_size=batch_size)))
    )


def _true_auc(bundle: ModelBundle, seed: int, batch_size: int) -> float:
    truth = truth_sample(seed, 20, batch_size)
    pd = np.asarray(bundle.model.predict_pd(truth.batch))
    ok = np.isfinite(pd)
    return expected_auc(pd[ok], truth.true_pd[ok])


def _registry_correct(registry: ModelRegistry, kind: str) -> bool:
    candidate, champion = registry.get(2).stage, registry.get(1).stage
    if kind in BAD:
        return candidate is Stage.ROLLED_BACK and champion is Stage.PRODUCTION
    return candidate is Stage.PRODUCTION and champion is Stage.ARCHIVED


def run_rollout(task: RolloutTask) -> RolloutSummary:
    """One planted release from shadow to promotion or rollback."""
    started = time.perf_counter()
    champion = champion_for(task.history)
    release = release_for(task.kind, task.release_seed)
    with tempfile.TemporaryDirectory() as tmp:
        registry, store = ModelRegistry(Path(tmp)), Store()
        first = registry.register(ModelBundle(champion.model, champion.reference), reason="champion")
        registry.transition(first.version, Stage.PRODUCTION, "benchmark champion")
        loop = ControlLoop(registry, store, LoopConfig(auto_retrain=False))
        candidate = registry.register(release, parent=first.version, reason=f"planted {task.kind}")
        loop.start_rollout(candidate.version)
        rollout = loop.rollout
        assert rollout is not None
        stream = CreditStream(StreamConfig(seed=task.stream_seed, n_batches=task.n_batches, batch_size=task.batch_size))
        labels: dict[int, np.ndarray] = {}
        excess = 0
        for batch in stream:
            result = loop.ingest(batch.raw, f"b{batch.index}", batch.timestamp)
            labels[batch.index] = batch.labels
            served = result.served_by_candidate
            if served.any():
                champ = approve(result.champion_pd, champion.model.threshold)
                excess += int(
                    np.sum(batch.labels[served] * (result.decisions[served].astype(int) - champ[served].astype(int)))
                )
            if batch.index >= task.label_delay:
                loop.ingest_labels(f"b{batch.index - task.label_delay}", labels.pop(batch.index - task.label_delay))
            if not loop.rollout_active:
                break
        correct = _registry_correct(registry, task.kind)
        registry.close()
        store.close()
    guardrail = rollout.reason.split(":", 1)[0] if rollout.status.value == "rolled_back" else "promoted"
    return RolloutSummary(
        task,
        rollout.status.value,
        rollout.batches,
        rollout.exposure,
        rollout.reason,
        guardrail,
        excess,
        correct,
        _true_auc(release, task.history.seed + 9_001, task.batch_size),
        time.perf_counter() - started,
    )


def rollout_metrics(summaries: list[RolloutSummary]) -> dict[str, object]:
    """Per release kind and pooled over bad and good releases."""
    grouped: dict[str, list[RolloutSummary]] = defaultdict(list)
    for s in summaries:
        grouped[s.task.kind].append(s)
    per_kind: dict[str, dict[str, object]] = {}
    for kind in (*BAD, *GOOD):
        items = grouped.get(kind, [])
        if not items:
            continue
        rolled = [s for s in items if s.status == "rolled_back"]
        per_kind[kind] = {
            "class": "bad" if kind in BAD else "good",
            "runs": len(items),
            "rolled_back_rate": len(rolled) / len(items),
            "promoted_rate": sum(s.status == "promoted" for s in items) / len(items),
            "registry_correct_rate": sum(s.registry_correct for s in items) / len(items),
            "batches_to_decision_median": float(np.median([s.batches for s in items])),
            "exposure_median": float(np.median([s.exposure for s in items])),
            "exposure_max": int(max(s.exposure for s in items)),
            "excess_defaults_approved_mean": float(np.mean([s.excess_defaults_approved for s in items])),
            "release_true_auc_mean": round(float(np.mean([s.release_true_auc for s in items])), 4),
            "guardrails": dict(Counter(s.guardrail for s in items)),
        }
    bad = [s for s in summaries if s.task.kind in BAD]
    good = [s for s in summaries if s.task.kind in GOOD]
    pooled = {
        "bad_runs": len(bad),
        "bad_rollback_rate": sum(s.status == "rolled_back" for s in bad) / max(len(bad), 1),
        "good_runs": len(good),
        "good_promotion_rate": sum(s.status == "promoted" for s in good) / max(len(good), 1),
        "false_rollback_rate": sum(s.status == "rolled_back" for s in good) / max(len(good), 1),
        "registry_correct_rate": sum(s.registry_correct for s in summaries) / max(len(summaries), 1),
    }
    return {"per_kind": per_kind, "pooled": pooled}
