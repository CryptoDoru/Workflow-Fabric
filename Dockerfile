# AI Workflow Fabric - Docker Image
# Multi-stage build for smaller production image

# =============================================================================
# Build Stage
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml README.md LICENSE ./
COPY awf/ ./awf/

# Install package with all extras
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api,cli]"

# =============================================================================
# Production Stage
# =============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY awf/ ./awf/
COPY spec/ ./spec/

# Create non-root user for security
RUN groupadd --gid 1000 awf && \
    useradd --uid 1000 --gid awf --shell /bin/bash --create-home awf && \
    chown -R awf:awf /app

USER awf

# Environment variables
ENV AWF_HOST=0.0.0.0
ENV AWF_PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Default command: start API server
CMD ["uvicorn", "awf.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Development Stage
# =============================================================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir "ai-workflow-fabric[dev]" 2>/dev/null || true

USER awf

# Override command for development with auto-reload
CMD ["uvicorn", "awf.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
