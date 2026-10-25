"""Repository hygiene checks that ruff cannot express.

* No em dashes or en dashes anywhere in tracked text (the house style).
* Every function in ``src`` is at most 60 lines long.

Exits non-zero and prints each violation; run by ``make lint`` and the tests.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = {chr(0x2014): "em dash", chr(0x2013): "en dash"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yml", ".yaml", ".html", ".css", ".js", ".txt", ".cfg", ""}
MAX_FUNCTION_LINES = 60


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [ROOT / line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        skip = {".venv", ".git", "data", "artifacts", ".pytest_cache", ".ruff_cache", ".hypothesis"}
        return [p for p in ROOT.rglob("*") if p.is_file() and not skip & set(p.relative_to(ROOT).parts)]


def dash_violations(files: list[Path]) -> list[str]:
    problems = []
    for path in files:
        if path.suffix not in TEXT_SUFFIXES or not path.exists() or path.name == "uv.lock":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for char, name in FORBIDDEN.items():
                if char in line:
                    problems.append(f"{path.relative_to(ROOT)}:{number}: {name}")
    return problems


def long_functions(source_root: Path) -> list[str]:
    problems = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.end_lineno is not None:
                length = node.end_lineno - node.lineno + 1
                if length > MAX_FUNCTION_LINES:
                    problems.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.name} is {length} lines")
    return problems


def main() -> int:
    problems = dash_violations(tracked_files()) + long_functions(ROOT / "src")
    for problem in problems:
        print(problem)
    print(f"text checks: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
