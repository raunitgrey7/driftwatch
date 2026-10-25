"""Structured JSON logging with a request id on every line.

Each request gets an id (from ``X-Request-ID`` when the caller sends a safe
one, otherwise generated) that is kept in a context variable, stamped on
every log record emitted while the request is handled and echoed back in the
response header, so an ingested batch can be traced from client to log line.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; fields passed with ``extra=`` are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True


async def request_logging_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Assign the request id, time the request and log one line for it."""
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if SAFE_ID.match(supplied) else uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logging.getLogger("driftwatch.http").info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response
