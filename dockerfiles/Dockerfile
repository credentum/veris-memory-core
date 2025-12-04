# Production Dockerfile for Context Store (MCP Server)
# Extends from veris-memory-core-base which has all ML dependencies
#
# Usage:
#   docker build -f dockerfiles/Dockerfile .
#   OR with local base: docker build --build-arg BASE_IMAGE=veris-base:local -f dockerfiles/Dockerfile .

ARG BASE_IMAGE=ghcr.io/credentum/veris-memory-core/base:latest
FROM ${BASE_IMAGE}

# Context Store specific configuration
ENV MCP_SERVER_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the MCP server
CMD ["python", "-m", "uvicorn", "src.mcp_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
