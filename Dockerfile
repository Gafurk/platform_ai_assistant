# syntax=docker/dockerfile:1

###############################################################################
# Stage 1 — builder: install Python dependencies into an isolated venv
###############################################################################
FROM python:3.11-slim AS builder

# Build tools needed by a few packages with C extensions (tiktoken, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment we can copy into the final image
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

###############################################################################
# Stage 2 — runtime: slim image with only the venv + application code
###############################################################################
FROM python:3.11-slim AS runtime

# Sensible Python defaults for containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy the application source
COPY . .

# Create data/log dirs the app writes to, and run as an unprivileged user
RUN mkdir -p /app/data/docs /app/data/lightrag /app/logs \
    && adduser --disabled-password --gecos "" --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Basic container healthcheck against the app's /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8001/health', timeout=4).status==200 else sys.exit(1)"

# No --reload in containers; bind to all interfaces so the port is reachable
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
