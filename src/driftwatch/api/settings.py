"""Runtime configuration from environment variables, with safe defaults for local use."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _flag(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    auto_retrain: bool = True
    max_rows: int = 5000
    history_seed: int = 7

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            api_key=env.get("APP_API_KEY") or None,
            data_dir=Path(env.get("DRIFTWATCH_DATA_DIR", "data")),
            log_level=env.get("DRIFTWATCH_LOG_LEVEL", "INFO"),
            auto_retrain=_flag(env.get("DRIFTWATCH_AUTO_RETRAIN"), True),
            max_rows=int(env.get("DRIFTWATCH_MAX_ROWS", "5000")),
            history_seed=int(env.get("DRIFTWATCH_HISTORY_SEED", "7")),
        )
