# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY sentinelx/ ./sentinelx/
COPY tasks/ ./tasks/
COPY data/ ./data/

# Install all dependencies into a prefix we'll copy into the runtime image
RUN pip install --no-cache-dir --prefix=/install \
        openenv-core>=0.2.0 \
        fastapi>=0.104.0 \
        "uvicorn[standard]>=0.24.0" \
        pydantic>=2.0.0 \
        openai>=1.0.0 \
        networkx>=3.0 \
    && pip install --no-cache-dir --prefix=/install --no-deps .

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY --from=builder /build/sentinelx ./sentinelx
COPY --from=builder /build/tasks     ./tasks
COPY --from=builder /build/data      ./data
COPY inference.py   ./inference.py
COPY openenv.yaml   ./openenv.yaml

RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces uses PORT env var (default 7860 externally, we bind to it)
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=2
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD ["sh", "-c", \
     "uvicorn sentinelx.server.app:app --host $HOST --port $PORT --workers $WORKERS"]
