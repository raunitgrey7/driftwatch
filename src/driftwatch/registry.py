"""Model registry: versions, lineage, stages and an audit trail in SQLite.

A directory of pickles is not enough for safe deployment. Rollback is only
safe if the registry knows exactly what was serving before, a rollout is
only auditable if every stage change is recorded with its reason, and a
retrained model is only trustworthy if it can be traced to its parent and
to the data window it was fitted on. Each of those is a column here.

Artifacts are files named by their SHA-256 and verified on every load, so a
truncated or modified artifact is refused instead of served. They are
pickles, which execute code when loaded: the registry only loads files it
wrote itself and whose hash matches its own record, and it is not meant for
models from untrusted sources.

Stage changes follow an explicit state machine; ``promote`` and ``rollback``
change two rows in one transaction so there is never zero or two production
versions.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class Stage(StrEnum):
    REGISTERED = "registered"
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


ALLOWED: dict[Stage, frozenset[Stage]] = {
    Stage.REGISTERED: frozenset({Stage.SHADOW, Stage.CANARY, Stage.PRODUCTION, Stage.REJECTED}),
    Stage.SHADOW: frozenset({Stage.CANARY, Stage.ROLLED_BACK}),
    Stage.CANARY: frozenset({Stage.PRODUCTION, Stage.ROLLED_BACK}),
    Stage.PRODUCTION: frozenset({Stage.ARCHIVED, Stage.ROLLED_BACK}),
    Stage.ARCHIVED: frozenset({Stage.PRODUCTION}),
    Stage.ROLLED_BACK: frozenset(),
    Stage.REJECTED: frozenset(),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_version INTEGER REFERENCES models(version),
    artifact_sha256 TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    lineage TEXT NOT NULL,
    metrics TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    version INTEGER NOT NULL REFERENCES models(version),
    from_stage TEXT,
    to_stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    actor TEXT NOT NULL
);
"""


class RegistryError(RuntimeError):
    """An illegal transition, a missing version or an artifact that fails verification."""


@dataclass(frozen=True)
class ModelRecord:
    version: int
    name: str
    stage: Stage
    created_at: str
    parent_version: int | None
    artifact_sha256: str
    artifact_path: str
    lineage: dict[str, Any]
    metrics: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "stage": self.stage.value}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ModelRegistry:
    """SQLite-backed registry rooted at ``root`` (``registry.db`` plus ``artifacts/``)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.root / "registry.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _record(row: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            version=row["version"],
            name=row["name"],
            stage=Stage(row["stage"]),
            created_at=row["created_at"],
            parent_version=row["parent_version"],
            artifact_sha256=row["artifact_sha256"],
            artifact_path=row["artifact_path"],
            lineage=json.loads(row["lineage"]),
            metrics=json.loads(row["metrics"]),
            reason=row["reason"],
        )

    def register(
        self,
        artifact: Any,
        *,
        name: str = "credit-pd",
        parent: int | None = None,
        lineage: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        reason: str = "",
    ) -> ModelRecord:
        """Store ``artifact`` by content hash and add a version in stage ``registered``."""
        payload = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(payload).hexdigest()
        path = self.artifacts / f"{digest}.pkl"
        if not path.exists():
            path.write_bytes(payload)
        if parent is not None:
            self.get(parent)
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO models (name, stage, created_at, parent_version, artifact_sha256, artifact_path, lineage, metrics, reason)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    Stage.REGISTERED.value,
                    _now(),
                    parent,
                    digest,
                    path.name,
                    json.dumps(lineage or {}, default=str),
                    json.dumps(metrics or {}, default=str),
                    reason,
                ),
            )
            version = int(cursor.lastrowid or 0)
            self._event(version, None, Stage.REGISTERED, reason or "registered", "system")
        return self.get(version)

    def _event(self, version: int, from_stage: Stage | None, to_stage: Stage, reason: str, actor: str) -> None:
        self._db.execute(
            "INSERT INTO events (ts, version, from_stage, to_stage, reason, actor) VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), version, from_stage.value if from_stage else None, to_stage.value, reason, actor),
        )

    def get(self, version: int) -> ModelRecord:
        row = self._db.execute("SELECT * FROM models WHERE version = ?", (version,)).fetchone()
        if row is None:
            raise RegistryError(f"no model version {version}")
        return self._record(row)

    def list(self) -> list[ModelRecord]:
        return [self._record(r) for r in self._db.execute("SELECT * FROM models ORDER BY version")]

    def in_stage(self, stage: Stage) -> list[ModelRecord]:
        rows = self._db.execute("SELECT * FROM models WHERE stage = ? ORDER BY version", (stage.value,))
        return [self._record(r) for r in rows]

    def production(self) -> ModelRecord | None:
        rows = self.in_stage(Stage.PRODUCTION)
        return rows[-1] if rows else None

    def load(self, version: int) -> Any:
        """Unpickle a version's artifact after checking its hash against the registry record."""
        record = self.get(version)
        payload = (self.artifacts / record.artifact_path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.artifact_sha256:
            raise RegistryError(f"artifact of version {version} does not match its recorded hash")
        return pickle.loads(payload)

    def _set_stage(self, record: ModelRecord, to: Stage, reason: str, actor: str) -> None:
        if to not in ALLOWED[record.stage]:
            raise RegistryError(f"version {record.version}: {record.stage} -> {to} is not allowed")
        self._db.execute("UPDATE models SET stage = ? WHERE version = ?", (to.value, record.version))
        self._event(record.version, record.stage, to, reason, actor)

    def transition(self, version: int, to: Stage, reason: str, actor: str = "system") -> ModelRecord:
        """Move one version to ``to``; production changes go through ``promote`` or ``rollback``."""
        if to is Stage.PRODUCTION and self.production() is not None:
            raise RegistryError("a production version exists: use promote()")
        with self._lock, self._db:
            self._set_stage(self.get(version), to, reason, actor)
        return self.get(version)

    def promote(self, version: int, reason: str, actor: str = "system") -> ModelRecord:
        """Make ``version`` production and archive the current one, atomically."""
        with self._lock, self._db:
            current = self.production()
            if current is not None and current.version != version:
                self._set_stage(current, Stage.ARCHIVED, f"replaced by v{version}: {reason}", actor)
            self._set_stage(self.get(version), Stage.PRODUCTION, reason, actor)
        return self.get(version)

    def rollback(self, version: int, reason: str, actor: str = "system") -> ModelRecord:
        """Retire ``version``; if it was production, restore its parent (or the latest archived)."""
        with self._lock, self._db:
            record = self.get(version)
            restore: ModelRecord | None = None
            if record.stage is Stage.PRODUCTION:
                archived = self.in_stage(Stage.ARCHIVED)
                parent = next((r for r in archived if r.version == record.parent_version), None)
                restore = parent or (archived[-1] if archived else None)
                if restore is None:
                    raise RegistryError("no archived version to restore")
            self._set_stage(record, Stage.ROLLED_BACK, reason, actor)
            if restore is not None:
                self._set_stage(restore, Stage.PRODUCTION, f"restored after rollback of v{version}", actor)
        return self.get(version)

    def lineage(self, version: int) -> list[ModelRecord]:
        """The chain from ``version`` back to the first model it descends from."""
        chain: list[ModelRecord] = []
        current: int | None = version
        while current is not None:
            record = self.get(current)
            chain.append(record)
            current = record.parent_version
        return chain

    def events(self, version: int | None = None) -> list[dict[str, Any]]:
        query, args = "SELECT * FROM events ORDER BY id", ()
        if version is not None:
            query, args = "SELECT * FROM events WHERE version = ? ORDER BY id", (version,)
        return [dict(r) for r in self._db.execute(query, args)]
