"""House rules enforced as tests: text style, function length, the CLI surface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from driftwatch.cli import build_parser, main

ROOT = Path(__file__).resolve().parent.parent


def test_no_dashes_and_no_long_functions():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_text.py")], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout


def test_cli_lists_scenarios(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert "concept_abrupt" in out and "dq_schema_break" in out


def test_cli_defaults_use_port_8774():
    args = build_parser().parse_args(["serve"])
    assert args.port == 8774
    replay = build_parser().parse_args(["replay", "--scenario", "prior_shift"])
    assert replay.api.endswith(":8774") and replay.scenario == "prior_shift"


def test_replay_reports_an_unreachable_service(capsys):
    assert main(["replay", "--api", "http://127.0.0.1:9", "--batches", "1"]) == 1
    assert "cannot reach" in capsys.readouterr().err
