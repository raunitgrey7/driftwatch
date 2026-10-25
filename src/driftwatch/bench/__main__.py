"""Command line entry: ``python -m driftwatch.bench [--quick] [--check-only] [--no-check]``."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from driftwatch.bench.check import check
from driftwatch.bench.config import BenchConfig
from driftwatch.bench.report import BEGIN, END, readme_table, results_markdown
from driftwatch.bench.run import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="driftwatch.bench", description="Detection, estimation, rollout and retraining benchmark."
    )
    parser.add_argument("--out", type=Path, default=Path("benchmarks"))
    parser.add_argument("--thresholds", type=Path, default=Path("benchmarks/thresholds.json"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--workers", type=int, default=2, help="worker processes (each single-threaded)")
    parser.add_argument("--quick", action="store_true", help="small configuration with the same code paths")
    parser.add_argument("--check-only", action="store_true", help="only evaluate thresholds on existing results")
    parser.add_argument("--no-check", action="store_true", help="skip the threshold gate")
    return parser


def update_readme(readme: Path, table: str) -> bool:
    """Replace the README block between the RESULTS markers; False when the markers are absent."""
    if not readme.exists():
        return False
    text = readme.read_text()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(lambda _: f"{BEGIN}\n{table}\n{END}", text))
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_path = args.out / "results.json"
    if not args.check_only:
        config = replace(BenchConfig(), workers=args.workers)
        if args.quick:
            config = config.quick()
        results = run_benchmark(config)
        args.out.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2, sort_keys=False, default=str) + "\n")
        (args.out / "RESULTS.md").write_text(results_markdown(results))
        if not args.quick and update_readme(args.readme, readme_table(results)):
            print(f"updated the results block in {args.readme}")
        print(results_markdown(results))
        print(f"wrote {results_path} in {results['timing']['total_seconds']} s")
    if args.no_check or not args.thresholds.exists():
        return 0
    return 0 if check(results_path, args.thresholds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
