# Production Dockerfile for Context Store API
# This is the Python-only container for use with docker-compose
# where other services (Qdrant, Neo4j, Redis) run as separate containers

# Use pinned SHA256 digest for security (python:3.11-slim as of 2025-01-14)
# SHA verification ensures image integrity - Docker automatically verifies the digest
# during pull. To manually verify: docker inspect python:3.11-slim | grep -i sha256
FROM python:3.11-slim@sha256:2ec5a4a5c3e919570f57675471f081d6299668d909feabd8d4803c6c61af666c

# Build arguments
ARG ENVIRONMENT=production

# Security: Upgrade vulnerable packages to fix CVEs detected in base image
# CVE-2025-4802, CVE-2025-6965, CVE-2025-7458, CVE-2023-7104, CVE-2023-31484, CVE-2024-56406
RUN apt-get update && \
    apt-get upgrade -y \
        libc-bin \
        libc6 \
        libsqlite3-0 \
        perl-base && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install system dependencies (both build and runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libc6-dev \
        curl \
        netcat-openbsd \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user early
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Remove build dependencies to keep image smaller (but keep runtime deps)
RUN apt-get update && \
    apt-get remove -y gcc g++ libc6-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser schemas/ ./schemas/
COPY --chown=appuser:appuser contracts/ ./contracts/
# Use Docker-specific config file with service names instead of localhost
COPY --chown=appuser:appuser config/.ctxrc.docker.yaml ./.ctxrc.yaml

# Optional directories - commented out since they don't exist in current structure
# COPY --chown=appuser:appuser monitoring/ ./monitoring/
# COPY --chown=appuser:appuser secrets/ ./secrets/

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Environment variables (will be overridden by docker-compose)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV MCP_SERVER_PORT=8000
ENV LOG_LEVEL=info
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Start the application with FastAPI/Uvicorn for HTTP endpoints
CMD ["python", "-m", "uvicorn", "src.mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]