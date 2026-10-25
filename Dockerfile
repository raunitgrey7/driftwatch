# Multi-stage build: uv resolves the locked dependencies in the builder, the
# runtime image carries only the virtual environment and runs as a non-root
# user. The service trains its reference model on first start (about a
# second) and keeps the registry and the store under /app/data.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim-bookworm AS runtime
RUN groupadd --system driftwatch && useradd --system --gid driftwatch --create-home driftwatch
WORKDIR /app
COPY --from=builder --chown=driftwatch:driftwatch /app/.venv /app/.venv
RUN mkdir -p /app/data && chown -R driftwatch:driftwatch /app
ENV PATH="/app/.venv/bin:$PATH" DRIFTWATCH_DATA_DIR=/app/data OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
USER driftwatch
EXPOSE 8774
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8774/health', timeout=2).status == 200 else 1)"]
CMD ["driftwatch", "serve", "--host", "0.0.0.0", "--port", "8774"]
