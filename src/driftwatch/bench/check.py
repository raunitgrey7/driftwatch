"""Threshold gate over ``benchmarks/results.json``.

Rules live in ``benchmarks/thresholds.json`` as (metric, comparison, value)
over the flat ``headline`` view, so a change to a threshold is a reviewable
diff. A missing or null metric fails the rule: a gate must never pass
because something was not measured.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPS: dict[str, Callable[[float, float], bool]] = {">=": operator.ge, "<=": operator.le, "==": operator.eq}


@dataclass(frozen=True)
class Rule:
    metric: str
    op: str
    value: float
    why: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Rule:
        if raw["op"] not in OPS:
            raise ValueError(f"unsupported comparison {raw['op']!r}")
        return cls(str(raw["metric"]), str(raw["op"]), float(raw["value"]), str(raw.get("why", "")))


@dataclass(frozen=True)
class Outcome:
    rule: Rule
    actual: float | None
    passed: bool

    def line(self) -> str:
        shown = "missing" if self.actual is None else f"{self.actual:.6g}"
        return f"{'PASS' if self.passed else 'FAIL'} {self.rule.metric} = {shown} (want {self.rule.op} {self.rule.value:g})"


def load_rules(path: Path) -> list[Rule]:
    return [Rule.from_dict(r) for r in json.loads(path.read_text())["rules"]]


def evaluate(results: dict[str, Any], rules: list[Rule]) -> list[Outcome]:
    headline = results.get("headline", {})
    outcomes = []
    for rule in rules:
        actual = headline.get(rule.metric)
        ok = actual is not None and OPS[rule.op](float(actual), rule.value)
        outcomes.append(Outcome(rule, None if actual is None else float(actual), ok))
    return outcomes


def check(results_path: Path, thresholds_path: Path) -> bool:
    """Print one line per rule; True when every rule passes."""
    outcomes = evaluate(json.loads(results_path.read_text()), load_rules(thresholds_path))
    for outcome in outcomes:
        print(outcome.line())
    print(f"{sum(o.passed for o in outcomes)}/{len(outcomes)} threshold rules passed")
    return all(o.passed for o in outcomes)
