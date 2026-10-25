"""Benchmark orchestration: build every task, run them on a small process pool, aggregate.

All three sections (detection, deployment safety, retraining) are independent
tasks with their own seeds, so they run in one pool of ``workers`` processes,
each limited to one BLAS and one OpenMP thread. Results are aggregated in task
order, which makes the output independent of scheduling: the same
configuration gives the same metrics with one worker or two.
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from driftwatch.bench.config import DRIFT_SCENARIOS, RETRAIN_SCENARIOS, BenchConfig
from driftwatch.bench.deployment import RolloutSummary, RolloutTask, rollout_metrics, run_rollout
from driftwatch.bench.detection import NEVER, StreamSummary, StreamTask, control_metrics, drift_metrics, run_stream
from driftwatch.bench.releases import BAD, GOOD
from driftwatch.bench.retraining import RetrainSummary, RetrainTask, retrain_metrics, run_retrain

Task = StreamTask | RolloutTask | RetrainTask
Summary = StreamSummary | RolloutSummary | RetrainSummary


def detection_tasks(cfg: BenchConfig) -> list[StreamTask]:
    """Control streams plus ``drift_seeds`` streams per drift type, each with a seeded onset."""
    rng = np.random.default_rng(cfg.seed)
    tasks = [
        StreamTask(
            "control", cfg.seed * 1000 + i, cfg.control_batches, NEVER, cfg.batch_size, cfg.label_delay, cfg.history
        )
        for i in range(cfg.control_streams)
    ]
    lo, hi = cfg.onset_range
    for s_index, scenario in enumerate(DRIFT_SCENARIOS):
        for k in range(cfg.drift_seeds):
            seed = cfg.seed * 1000 + 100 + s_index * 50 + k
            onset = int(rng.integers(lo, hi + 1))
            tasks.append(
                StreamTask(scenario, seed, cfg.drift_batches, onset, cfg.batch_size, cfg.label_delay, cfg.history)
            )
    return tasks


def rollout_tasks(cfg: BenchConfig) -> list[RolloutTask]:
    """Every release kind, ``release_instances`` trained copies, each on ``release_streams`` traffic streams."""
    tasks = []
    for k_index, kind in enumerate((*BAD, *GOOD)):
        for instance in range(cfg.release_instances):
            release_seed = cfg.seed * 10 + 700 + instance
            for stream in range(cfg.release_streams):
                stream_seed = cfg.seed * 1000 + 800 + k_index * 40 + instance * 10 + stream
                tasks.append(
                    RolloutTask(
                        kind,
                        release_seed,
                        stream_seed,
                        cfg.rollout_batches,
                        cfg.batch_size,
                        cfg.label_delay,
                        cfg.history,
                    )
                )
    return tasks


def retrain_tasks(cfg: BenchConfig) -> list[RetrainTask]:
    tasks = []
    for s_index, scenario in enumerate(RETRAIN_SCENARIOS):
        onset = NEVER if scenario == "control" else cfg.retrain_onset
        forced = cfg.forced_retrain_at if scenario == "control" else None
        for k in range(cfg.retrain_seeds):
            seed = cfg.seed * 1000 + 600 + s_index * 20 + k
            tasks.append(
                RetrainTask(
                    scenario,
                    seed,
                    cfg.retrain_batches,
                    onset,
                    cfg.batch_size,
                    cfg.label_delay,
                    cfg.history,
                    forced,
                    cfg.truth_batches,
                )
            )
    return tasks


def run_task(task: Task) -> Summary:
    if isinstance(task, StreamTask):
        return run_stream(task)
    if isinstance(task, RolloutTask):
        return run_rollout(task)
    return run_retrain(task)


def _init_worker() -> None:
    threadpool_limits(1)


def run_all(tasks: list[Task], workers: int) -> list[Summary]:
    """Run tasks in order; with more than one worker, on a spawn-context process pool."""
    if workers <= 1:
        with threadpool_limits(1):
            return [run_task(t) for t in tasks]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        return list(pool.map(run_task, tasks, chunksize=1))


def environment() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "commit": commit,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_benchmark(cfg: BenchConfig) -> dict[str, Any]:
    """Run every section and return the full results document."""
    started = time.perf_counter()
    tasks: list[Task] = [*retrain_tasks(cfg), *detection_tasks(cfg), *rollout_tasks(cfg)]
    summaries = run_all(tasks, cfg.workers)
    streams = [s for s in summaries if isinstance(s, StreamSummary)]
    rollouts = [s for s in summaries if isinstance(s, RolloutSummary)]
    retrains = [s for s in summaries if isinstance(s, RetrainSummary)]
    control = [s for s in streams if s.task.scenario == "control"]
    drifted = [s for s in streams if s.task.scenario != "control"]
    results: dict[str, Any] = {
        "config": asdict(cfg),
        "environment": environment(),
        "detection": {"control": control_metrics(control), "drift": drift_metrics(drifted, cfg.detection_window)},
        "deployment": rollout_metrics(rollouts),
        "retraining": retrain_metrics(retrains),
        "timing": {
            "total_seconds": round(time.perf_counter() - started, 1),
            "task_seconds": round(sum(s.seconds for s in summaries), 1),
            "batches_monitored": sum(s.task.n_batches for s in streams),
            "monitor_ms_per_batch": round(
                1000 * sum(s.seconds for s in streams) / max(sum(s.task.n_batches for s in streams), 1), 3
            ),
        },
    }
    results["headline"] = headline(results)
    return results


def headline(results: dict[str, Any]) -> dict[str, float | None]:
    """Flat ``section.name.metric`` view used by the threshold gate and the README table."""
    out: dict[str, float | None] = {}
    control = results["detection"]["control"]
    out["control.false_alarms_per_10k"] = control["false_alarms_per_10k"]
    out["control.batches"] = control["batches"]
    for name, metrics in results["detection"]["drift"].items():
        for key in ("recall", "precision", "delay_median", "delay_p90", "diagnosis_rate"):
            out[f"detection.{name}.{key}"] = metrics[key]
        for key, value in metrics["estimation_post_onset"].items():
            out[f"estimation.{name}.{key}"] = value
    for key, value in control["estimation"].items():
        out[f"estimation.control.{key}"] = value
    for key, value in results["deployment"]["pooled"].items():
        out[f"deployment.{key}"] = value
    for kind, metrics in results["deployment"]["per_kind"].items():
        for key in ("rolled_back_rate", "batches_to_decision_median", "exposure_median"):
            out[f"deployment.{kind}.{key}"] = metrics[key]
    out["retraining.gate_correct_rate"] = results["retraining"]["pooled"]["gate_correct_rate"]
    for name, metrics in results["retraining"]["per_scenario"].items():
        for key, value in metrics.items():
            out[f"retraining.{name}.{key}"] = value
    return out
