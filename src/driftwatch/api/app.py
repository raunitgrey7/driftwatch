"""Application factory: wiring, lifespan, error mapping, health, metrics and the dashboard."""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from driftwatch import __version__
from driftwatch.api.logging import configure_logging, request_logging_middleware
from driftwatch.api.routes import router
from driftwatch.api.schemas import Health
from driftwatch.api.service import Services
from driftwatch.api.settings import Settings
from driftwatch.loop import ConflictError, InvalidInputError, LoopError, NotFoundError
from driftwatch.registry import RegistryError

STATIC_DIR = Path(__file__).parent / "static"
STATUS: dict[type[LoopError], int] = {NotFoundError: 404, ConflictError: 409, InvalidInputError: 422}


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application; pass ``settings`` explicitly in tests."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = Services(settings)
        try:
            yield
        finally:
            app.state.services.close()

    app = FastAPI(title="driftwatch", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.middleware("http")(_metrics_middleware)
    app.middleware("http")(request_logging_middleware)
    app.add_exception_handler(RequestValidationError, _validation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(LoopError, _loop_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RegistryError, _registry_error_handler)  # type: ignore[arg-type]
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        svc: Services = request.app.state.services
        state = svc.state()
        rollout = state["rollout"]
        return Health(
            status="ok",
            version=__version__,
            uptime_s=round(time.time() - svc.started_at, 3),
            champion_version=state["champion_version"],
            rollout_status=rollout["status"] if rollout else None,
            batches=state["batches"],
            open_alerts=state["open_alerts"],
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> PlainTextResponse:
        svc: Services = request.app.state.services
        return PlainTextResponse(svc.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _finite(value: Any) -> Any:
    """Replace NaN and infinities so an error echo can always be serialised."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _finite(jsonable_encoder(exc.errors()))})


async def _loop_error_handler(_: Request, exc: LoopError) -> JSONResponse:
    return JSONResponse(status_code=STATUS.get(type(exc), 400), content={"detail": str(exc)})


async def _registry_error_handler(_: Request, exc: RegistryError) -> JSONResponse:
    code = 404 if str(exc).startswith("no model version") else 409
    return JSONResponse(status_code=code, content={"detail": str(exc)})


async def _metrics_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Count and time every request by route template so paths with ids do not explode the label set."""
    start = time.perf_counter()
    response = await call_next(request)
    svc: Services | None = getattr(request.app.state, "services", None)
    if svc is not None:
        route = request.scope.get("route")
        template = getattr(route, "path", "unmatched")
        with svc.metrics.lock:
            svc.metrics.http_requests.inc(method=request.method, path=template, status=str(response.status_code))
            svc.metrics.http_duration.observe(time.perf_counter() - start, method=request.method, path=template)
    return response
