"""Typed request and response bodies.

Rows are free-form records on purpose: the monitor exists to describe
batches that break the schema, so a renamed column or a string in a numeric
field must reach the schema check instead of being rejected here. What is
validated is what would hurt the service: sizes, value types (JSON scalars
only) and identifiers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

Scalar = StrictBool | StrictInt | StrictFloat | StrictStr | None
IDENTIFIER = r"^[A-Za-z0-9._:-]{1,64}$"


class BatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = Field(default=None, pattern=IDENTIFIER)
    timestamp: str | None = Field(default=None, max_length=40)
    rows: list[dict[str, Scalar]] = Field(min_length=1)
    include_rows: bool = True

    @field_validator("rows")
    @classmethod
    def bounded_rows(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            if len(row) > 64:
                raise ValueError("a row may have at most 64 fields")
            for key, value in row.items():
                if len(key) > 64:
                    raise ValueError("field names are limited to 64 characters")
                if isinstance(value, str) and len(value) > 128:
                    raise ValueError("string values are limited to 128 characters")
        return rows


class AlertOut(BaseModel):
    id: int | None = None
    batch_index: int
    ts: str
    category: str
    scope: str
    severity: str
    detectors: list[str]
    message: str
    status: str = "open"


class RowDecision(BaseModel):
    application_id: str
    pd: float | None
    approve: bool
    decided_by: int


class RetrainOut(BaseModel):
    version: int
    challenger: str
    promote: bool
    reason: str
    window: list[int]
    n_rows: int
    gate: dict[str, Any]


class BatchOut(BaseModel):
    batch_index: int
    batch_id: str
    n_rows: int
    champion_version: int
    candidate_version: int | None
    approval_rate: float
    estimated_auc: float | None
    estimated_default_rate: float | None
    alerts: list[AlertOut]
    rollout_action: str
    retrain: RetrainOut | None
    decisions: list[RowDecision] | None = None


class LabelIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_id: str = Field(min_length=1, max_length=128)
    defaulted: StrictBool


class LabelsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[LabelIn] = Field(min_length=1)


class LabelsOut(BaseModel):
    batch_id: str
    batch_index: int
    n_rows: int
    realized_auc: float | None
    default_rate: float
    alerts: list[AlertOut]
    rollout_action: str
    retrain: RetrainOut | None


class ModelOut(BaseModel):
    version: int
    name: str
    stage: str
    created_at: str
    parent_version: int | None
    artifact_sha256: str
    lineage: dict[str, Any]
    metrics: dict[str, Any]
    reason: str


class ModelDetail(ModelOut):
    lineage_chain: list[int]
    events: list[dict[str, Any]]


class ReasonIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=200)


class Health(BaseModel):
    status: str
    version: str
    uptime_s: float
    champion_version: int
    rollout_status: str | None
    batches: int
    open_alerts: int
