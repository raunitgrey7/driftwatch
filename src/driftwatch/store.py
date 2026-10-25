"""Operational state of the control loop in SQLite: batches, alerts, labels, rollouts.

Batches keep their coerced features as an ``npz`` payload (numbers and
vocabulary codes, never pickles) because retraining needs the rows once
their labels arrive, and labels arrive weeks after the batch. Batches with a
schema or data-quality finding are flagged so they are never used to train
a challenger. ``":memory:"`` gives the same store for the benchmark and
tests without touching disk.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from driftwatch.alerts import Alert
from driftwatch.schema import CATEGORICAL_FEATURES, FEATURE_INDEX, NUMERIC_FEATURES, Batch, concat_batches
from driftwatch.training import LabeledData

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_index INTEGER PRIMARY KEY,
    batch_id TEXT UNIQUE NOT NULL,
    ts TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    champion_version INTEGER NOT NULL,
    candidate_version INTEGER,
    quality_flag INTEGER NOT NULL,
    summary TEXT NOT NULL,
    payload BLOB NOT NULL,
    labels BLOB
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_index INTEGER NOT NULL,
    ts TEXT NOT NULL,
    category TEXT NOT NULL,
    scope TEXT NOT NULL,
    severity TEXT NOT NULL,
    detectors TEXT NOT NULL,
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS label_reports (
    batch_index INTEGER PRIMARY KEY,
    arrived_at INTEGER NOT NULL,
    n_rows INTEGER NOT NULL,
    realized_auc REAL,
    default_rate REAL NOT NULL,
    brier REAL NOT NULL,
    error_rate REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rollouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_version INTEGER NOT NULL,
    champion_version INTEGER NOT NULL,
    started_at INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    exposure INTEGER NOT NULL DEFAULT 0,
    history TEXT NOT NULL DEFAULT '[]'
);
"""


def pack(batch: Batch, arrays: dict[str, np.ndarray]) -> bytes:
    """Serialise a coerced batch (numeric values and category codes) plus extra arrays."""
    buffer = io.BytesIO()
    columns = {f"num_{k}": batch.numeric[k] for k in NUMERIC_FEATURES}
    columns |= {f"cat_{k}": batch.codes(k) for k in CATEGORICAL_FEATURES}
    np.savez_compressed(buffer, ids=batch.ids.astype(str), **columns, **{f"x_{k}": v for k, v in arrays.items()})
    return buffer.getvalue()


def unpack(payload: bytes) -> tuple[Batch, dict[str, np.ndarray]]:
    """Inverse of ``pack``; values outside the vocabulary come back as ``"<unseen>"``."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        numeric = {k: data[f"num_{k}"] for k in NUMERIC_FEATURES}
        categorical: dict[str, np.ndarray] = {}
        for k in CATEGORICAL_FEATURES:
            codes = data[f"cat_{k}"]
            vocab = FEATURE_INDEX[k].vocabulary
            categorical[k] = np.array(
                [vocab[c] if c >= 0 else (None if c == -2 else "<unseen>") for c in codes.tolist()], dtype=object
            )
        extra = {name[2:]: data[name] for name in data.files if name.startswith("x_")}
        ids = data["ids"].astype(object)
    return Batch(ids=ids, numeric=numeric, categorical=categorical), extra


class Store:
    """Thin SQLite layer; one connection guarded by a lock for the threaded API."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def next_index(self) -> int:
        row = self._db.execute("SELECT COALESCE(MAX(batch_index), -1) + 1 AS n FROM batches").fetchone()
        return int(row["n"])

    def index_of(self, batch_id: str) -> int | None:
        row = self._db.execute("SELECT batch_index FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        return None if row is None else int(row["batch_index"])

    def add_batch(
        self,
        index: int,
        batch_id: str,
        ts: str,
        champion: int,
        candidate: int | None,
        quality_flag: bool,
        summary: dict[str, Any],
        payload: bytes,
    ) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO batches (batch_index, batch_id, ts, n_rows, champion_version, candidate_version, quality_flag, summary, payload)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    index,
                    batch_id,
                    ts,
                    int(summary.get("n_rows", 0)),
                    champion,
                    candidate,
                    int(quality_flag),
                    json.dumps(summary),
                    payload,
                ),
            )

    def payload(self, index: int) -> tuple[Batch, dict[str, np.ndarray]] | None:
        row = self._db.execute("SELECT payload FROM batches WHERE batch_index = ?", (index,)).fetchone()
        return None if row is None else unpack(row["payload"])

    def versions(self, index: int) -> tuple[int, int | None]:
        """Champion and candidate versions that scored batch ``index``."""
        row = self._db.execute(
            "SELECT champion_version, candidate_version FROM batches WHERE batch_index = ?", (index,)
        ).fetchone()
        if row is None:
            raise KeyError(index)
        return int(row["champion_version"]), row["candidate_version"]

    def has_labels(self, index: int) -> bool:
        row = self._db.execute(
            "SELECT labels IS NOT NULL AS done FROM batches WHERE batch_index = ?", (index,)
        ).fetchone()
        return bool(row and row["done"])

    def set_labels(self, index: int, labels: np.ndarray) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE batches SET labels = ? WHERE batch_index = ?",
                (np.asarray(labels, dtype=np.int8).tobytes(), index),
            )

    def labelled_window(self, start: int, end: int) -> LabeledData | None:
        """Labelled, quality-clean batches with index in [start, end], in time order."""
        rows = self._db.execute(
            "SELECT payload, labels FROM batches WHERE batch_index BETWEEN ? AND ? AND labels IS NOT NULL AND quality_flag = 0 ORDER BY batch_index",
            (start, end),
        ).fetchall()
        if not rows:
            return None
        batches = [unpack(r["payload"])[0] for r in rows]
        labels = np.concatenate([np.frombuffer(r["labels"], dtype=np.int8) for r in rows])
        return LabeledData(concat_batches(batches), labels)

    def add_alerts(self, alerts: Iterable[Alert]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT INTO alerts (batch_index, ts, category, scope, severity, detectors, message) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        a.batch_index,
                        a.timestamp,
                        a.category.value,
                        a.scope,
                        a.severity,
                        ",".join(a.detectors),
                        a.message,
                    )
                    for a in alerts
                ],
            )

    def alerts(self, status: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT * FROM alerts" + (" WHERE status = ?" if status else "") + " ORDER BY id DESC LIMIT ?"
        args: tuple[Any, ...] = (status, limit) if status else (limit,)
        return [dict(r) for r in self._db.execute(query, args)]

    def set_alert_status(self, alert_id: int, status: str) -> bool:
        with self._lock, self._db:
            cursor = self._db.execute("UPDATE alerts SET status = ? WHERE id = ?", (status, alert_id))
        return cursor.rowcount == 1

    def add_label_report(
        self,
        index: int,
        arrived_at: int,
        n_rows: int,
        realized_auc: float,
        default_rate: float,
        brier: float,
        error_rate: float,
    ) -> None:
        auc_value = realized_auc if np.isfinite(realized_auc) else None
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO label_reports VALUES (?, ?, ?, ?, ?, ?, ?)",
                (index, arrived_at, n_rows, auc_value, default_rate, brier, error_rate),
            )

    def timeline(self, limit: int = 500) -> list[dict[str, Any]]:
        """Per-batch summaries joined with the realised metrics once labels are in."""
        rows = self._db.execute(
            "SELECT b.batch_index, b.batch_id, b.champion_version, b.candidate_version, b.summary, l.realized_auc, l.default_rate"
            " FROM batches b LEFT JOIN label_reports l ON l.batch_index = b.batch_index ORDER BY b.batch_index DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in reversed(rows):
            item = json.loads(r["summary"]) | {
                "batch_id": r["batch_id"],
                "champion_version": r["champion_version"],
                "candidate_version": r["candidate_version"],
            }
            item |= {"realized_auc": r["realized_auc"], "realized_default_rate": r["default_rate"]}
            out.append(item)
        return out

    def save_rollout(
        self,
        rollout_id: int | None,
        candidate: int,
        champion: int,
        started_at: int,
        status: str,
        reason: str,
        exposure: int,
        history: list[dict[str, Any]],
    ) -> int:
        with self._lock, self._db:
            if rollout_id is None:
                cursor = self._db.execute(
                    "INSERT INTO rollouts (candidate_version, champion_version, started_at, status, reason, exposure, history) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (candidate, champion, started_at, status, reason, exposure, json.dumps(history)),
                )
                return int(cursor.lastrowid or 0)
            self._db.execute(
                "UPDATE rollouts SET status = ?, reason = ?, exposure = ?, history = ? WHERE id = ?",
                (status, reason, exposure, json.dumps(history), rollout_id),
            )
            return rollout_id

    def rollouts(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT * FROM rollouts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) | {"history": json.loads(r["history"])} for r in rows]

    def counts(self) -> dict[str, int]:
        one = self._db.execute
        return {
            "batches": int(one("SELECT COUNT(*) FROM batches").fetchone()[0]),
            "labelled_batches": int(one("SELECT COUNT(*) FROM batches WHERE labels IS NOT NULL").fetchone()[0]),
            "open_alerts": int(one("SELECT COUNT(*) FROM alerts WHERE status = 'open'").fetchone()[0]),
        }
