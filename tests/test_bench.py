"""The benchmark harness: deterministic, scheduling-independent, and its gate is strict."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from driftwatch.bench.__main__ import update_readme
from driftwatch.bench.check import Rule, evaluate, load_rules
from driftwatch.bench.config import BenchConfig
from driftwatch.bench.report import BEGIN, END, readme_table, results_markdown
from driftwatch.bench.run import detection_tasks, rollout_tasks, run_benchmark

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ("detection", "deployment", "retraining", "headline")


def tiny(workers: int) -> BenchConfig:
    return replace(
        BenchConfig().quick(),
        workers=workers,
        control_streams=1,
        control_batches=60,
        drift_batches=80,
        onset_range=(40, 45),
        detection_window=35,
        rollout_batches=40,
        retrain_batches=100,
        retrain_onset=40,
        forced_retrain_at=80,
        truth_batches=3,
    )


@pytest.fixture(scope="module")
def tiny_results():
    return run_benchmark(tiny(workers=1))


def test_the_benchmark_is_deterministic(tiny_results):
    again = run_benchmark(tiny(workers=1))
    for section in SECTIONS:
        assert again[section] == tiny_results[section], section


def test_results_do_not_depend_on_the_number_of_workers(tiny_results):
    parallel = run_benchmark(tiny(workers=2))
    for section in SECTIONS:
        assert parallel[section] == tiny_results[section], section


def test_the_full_preset_measures_false_alarms_on_ten_thousand_batches():
    cfg = BenchConfig()
    control = [t for t in detection_tasks(cfg) if t.scenario == "control"]
    assert sum(t.n_batches for t in control) >= 10_000
    assert len({t.seed for t in detection_tasks(cfg)}) == len(detection_tasks(cfg))
    assert len(rollout_tasks(cfg)) == 8 * cfg.release_instances * cfg.release_streams


def test_a_missing_metric_fails_the_gate():
    results = {"headline": {"a": 1.0, "b": None}}
    rules = [Rule("a", ">=", 0.5), Rule("b", ">=", 0.0), Rule("c", "<=", 1.0)]
    assert [o.passed for o in evaluate(results, rules)] == [True, False, False]
    with pytest.raises(ValueError):
        Rule.from_dict({"metric": "a", "op": "!=", "value": 1})


def test_every_gate_rule_names_a_measured_metric():
    results = json.loads((ROOT / "benchmarks/results.json").read_text())
    missing = [r.metric for r in load_rules(ROOT / "benchmarks/thresholds.json") if r.metric not in results["headline"]]
    assert missing == []


def test_committed_results_pass_their_gate():
    results = json.loads((ROOT / "benchmarks/results.json").read_text())
    failed = [o.line() for o in evaluate(results, load_rules(ROOT / "benchmarks/thresholds.json")) if not o.passed]
    assert failed == []


def test_readme_and_results_markdown_match_results_json():
    results = json.loads((ROOT / "benchmarks/results.json").read_text())
    readme = (ROOT / "README.md").read_text()
    block = readme.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    assert block == readme_table(results)
    assert (ROOT / "benchmarks/RESULTS.md").read_text() == results_markdown(results)


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{BEGIN}\nold\n{END}\noutro\n")
    assert update_readme(readme, "| new |")
    assert readme.read_text() == f"intro\n{BEGIN}\n| new |\n{END}\noutro\n"
    assert not update_readme(tmp_path / "absent.md", "x")
