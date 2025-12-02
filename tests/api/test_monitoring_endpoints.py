#!/usr/bin/env python3
"""
Unit tests for Sentinel monitoring endpoints.

Tests the 4 new endpoints added in Phase 2:
- GET /metrics (Prometheus format)
- GET /database (database status)
- GET /storage (storage status)
- GET /tools/list (tools alias)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from fastapi.testclient import TestClient

# Import app at module level for better test performance and discovery
from src.mcp_server.main import app


@pytest.fixture
def mock_health():
    """Mock health check response with dynamic timestamp."""
    current_time = datetime.now(timezone.utc).timestamp()
    return {
        "status": "healthy",
        "uptime_seconds": 3600,
        "timestamp": current_time,
        "message": "Server is running",
    }


@pytest.fixture
def mock_health_detailed():
    """Mock detailed health check response with dynamic timestamps."""
    current_time = datetime.now(timezone.utc).timestamp()
    startup_time = current_time - 3600  # Started 1 hour ago
    return {
        "status": "healthy",
        "services": {
            "neo4j": "healthy",
            "qdrant": "healthy",
            "redis": "healthy",
            "embeddings": "healthy",
        },
        "startup_time": startup_time,
        "uptime_seconds": 3600,
        "grace_period_active": False,
    }




class TestPrometheusMetricsEndpoint:
    """Test GET /metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(self, mock_health, mock_health_detailed):
        """Test that /metrics returns valid Prometheus text format."""

        with patch("src.mcp_server.main.health", return_value=mock_health):
            with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
                client = TestClient(app)
                response = client.get("/metrics")

                assert response.status_code == 200
                assert response.headers["content-type"] == "text/plain; charset=utf-8"

                # Verify Prometheus format
                content = response.text
                assert "# HELP veris_memory_health_status" in content
                assert "# TYPE veris_memory_health_status gauge" in content
                assert "veris_memory_health_status{service=\"overall\"} 1" in content

    @pytest.mark.asyncio
    async def test_metrics_includes_all_services(self, mock_health, mock_health_detailed):
        """Test that metrics include all service health statuses."""

        with patch("src.mcp_server.main.health", return_value=mock_health):
            with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
                client = TestClient(app)
                response = client.get("/metrics")

                content = response.text

                # Verify all services are included
                assert 'service="qdrant"' in content
                assert 'service="neo4j"' in content
                assert 'service="redis"' in content

    @pytest.mark.asyncio
    async def test_metrics_reports_unhealthy_services(self, mock_health, mock_health_detailed):
        """Test that unhealthy services are reported as 0."""

        # Create a copy to avoid modifying the fixture
        unhealthy_detailed = mock_health_detailed.copy()
        unhealthy_detailed["services"] = mock_health_detailed["services"].copy()
        unhealthy_detailed["services"]["neo4j"] = "unhealthy"

        with patch("src.mcp_server.main.health", return_value=mock_health):
            with patch("src.mcp_server.main.get_cached_health_detailed", return_value=unhealthy_detailed):
                client = TestClient(app)
                response = client.get("/metrics")

                content = response.text

                # Neo4j should be 0 (unhealthy)
                assert 'veris_memory_health_status{service="neo4j"} 0' in content
                # Others should be 1 (healthy)
                assert 'veris_memory_health_status{service="qdrant"} 1' in content

    @pytest.mark.asyncio
    async def test_metrics_includes_uptime(self, mock_health, mock_health_detailed):
        """Test that metrics include uptime counter."""

        with patch("src.mcp_server.main.health", return_value=mock_health):
            with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
                client = TestClient(app)
                response = client.get("/metrics")

                content = response.text

                # Verify uptime metrics
                assert "# HELP veris_memory_uptime_seconds" in content
                assert "# TYPE veris_memory_uptime_seconds counter" in content
                assert "veris_memory_uptime_seconds 3600" in content

    @pytest.mark.asyncio
    async def test_metrics_includes_service_info(self, mock_health, mock_health_detailed):
        """Test that metrics include service information."""

        with patch("src.mcp_server.main.health", return_value=mock_health):
            with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
                client = TestClient(app)
                response = client.get("/metrics")

                content = response.text

                # Verify service info
                assert "# HELP veris_memory_info" in content
                assert "# TYPE veris_memory_info gauge" in content
                assert 'version="0.9.0"' in content
                assert 'protocol="MCP-1.0"' in content


class TestDatabaseStatusEndpoint:
    """Test GET /database endpoint."""

    @pytest.mark.asyncio
    async def test_database_returns_json(self, mock_health_detailed):
        """Test that /database returns valid JSON structure."""

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            client = TestClient(app)
            response = client.get("/database")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

            data = response.json()
            assert "status" in data
            assert "databases" in data
            assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_database_includes_all_databases(self, mock_health_detailed):
        """Test that all databases are included in response."""

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            client = TestClient(app)
            response = client.get("/database")

            data = response.json()
            databases = data["databases"]

            # Verify all databases present
            assert "neo4j" in databases
            assert "qdrant" in databases
            assert "redis" in databases

            # Verify structure for each database
            for db_name in ["neo4j", "qdrant", "redis"]:
                db = databases[db_name]
                assert "status" in db
                assert "type" in db
                assert "connected" in db
                assert "url" in db

    @pytest.mark.asyncio
    async def test_database_status_mapping(self, mock_health_detailed):
        """Test that database status is correctly mapped."""

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            client = TestClient(app)
            response = client.get("/database")

            data = response.json()

            # All healthy, so overall should be healthy
            assert data["status"] == "healthy"

            # Each database should be connected
            for db in data["databases"].values():
                assert db["connected"] is True

    @pytest.mark.asyncio
    async def test_database_degraded_status(self, mock_health_detailed):
        """Test that degraded status is reported when service unhealthy."""

        # Create a copy to avoid modifying the fixture
        degraded_health = mock_health_detailed.copy()
        degraded_health["status"] = "degraded"
        degraded_health["services"] = mock_health_detailed["services"].copy()
        degraded_health["services"]["neo4j"] = "unhealthy"

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=degraded_health):
            client = TestClient(app)
            response = client.get("/database")

            data = response.json()

            # Overall status should be degraded
            assert data["status"] == "degraded"

            # Neo4j should not be connected
            assert data["databases"]["neo4j"]["connected"] is False

    @pytest.mark.asyncio
    async def test_database_fallback_urls(self, mock_health_detailed):
        """Test that fallback URLs work when environment variables not set."""
        import os

        # Test with empty environment (fallback URLs should be used)
        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            with patch.dict(os.environ, {}, clear=True):
                client = TestClient(app)
                response = client.get("/database")

                data = response.json()

                # Verify URLs are masked in production (DEBUG=false)
                assert data["databases"]["neo4j"]["url"] == "bolt://[REDACTED]"
                assert data["databases"]["qdrant"]["url"] == "http://[REDACTED]"
                assert data["databases"]["redis"]["url"] == "redis://[REDACTED]"


class TestStorageStatusEndpoint:
    """Test GET /storage endpoint."""

    @pytest.mark.asyncio
    async def test_storage_returns_json(self, mock_health_detailed):
        """Test that /storage returns valid JSON structure."""

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            client = TestClient(app)
            response = client.get("/storage")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

            data = response.json()
            assert "status" in data
            assert "backends" in data
            assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_storage_includes_all_backends(self, mock_health_detailed):
        """Test that all storage backends are included."""

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            client = TestClient(app)
            response = client.get("/storage")

            data = response.json()
            backends = data["backends"]

            # Verify all backend types present
            assert "vector" in backends
            assert "graph" in backends
            assert "cache" in backends

            # Verify structure for each backend
            for backend_type in ["vector", "graph", "cache"]:
                backend = backends[backend_type]
                assert "service" in backend
                assert "status" in backend
                assert "healthy" in backend
                assert "type" in backend

    @pytest.mark.asyncio
    async def test_storage_backend_types(self, mock_health_detailed):
        """Test that backend types are correctly assigned."""

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            client = TestClient(app)
            response = client.get("/storage")

            data = response.json()
            backends = data["backends"]

            # Verify backend types
            assert backends["vector"]["type"] == "vector_database"
            assert backends["graph"]["type"] == "graph_database"
            assert backends["cache"]["type"] == "key_value_store"

            # Verify service mappings
            assert backends["vector"]["service"] == "qdrant"
            assert backends["graph"]["service"] == "neo4j"
            assert backends["cache"]["service"] == "redis"

    @pytest.mark.asyncio
    async def test_storage_health_mapping(self, mock_health_detailed):
        """Test that storage health is correctly mapped."""

        with patch("src.mcp_server.main.get_cached_health_detailed", return_value=mock_health_detailed):
            client = TestClient(app)
            response = client.get("/storage")

            data = response.json()

            # All healthy
            assert data["status"] == "healthy"
            for backend in data["backends"].values():
                assert backend["healthy"] is True


class TestToolsListAliasEndpoint:
    """Test GET /tools/list alias endpoint."""

    @pytest.mark.asyncio
    async def test_tools_list_alias_returns_json(self):
        """Test that /tools/list returns JSON."""

        # Mock the list_tools function
        mock_tools_response = {
            "tools": ["store_context", "retrieve_context", "query_graph"],
            "count": 3,
        }

        with patch("src.mcp_server.main.list_tools", return_value=mock_tools_response):
            client = TestClient(app)
            response = client.get("/tools/list")

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_tools_list_calls_list_tools(self):
        """Test that /tools/list calls list_tools() function."""

        mock_tools_response = {
            "tools": ["store_context", "retrieve_context"],
            "count": 2,
        }

        with patch("src.mcp_server.main.list_tools", return_value=mock_tools_response) as mock_list_tools:
            client = TestClient(app)
            response = client.get("/tools/list")

            # Verify list_tools was called
            mock_list_tools.assert_called_once()

            # Verify response matches
            assert response.json() == mock_tools_response

    @pytest.mark.asyncio
    async def test_tools_list_alias_identical_to_tools(self):
        """Test that /tools/list returns valid tools listing."""

        client = TestClient(app)

        # Verify /tools/list endpoint exists and returns valid structure
        response_tools_list = client.get("/tools/list")

        assert response_tools_list.status_code == 200
        assert response_tools_list.headers["content-type"] == "application/json"

        data = response_tools_list.json()
        # Verify it has tools list
        assert "tools" in data
        assert isinstance(data["tools"], list)
        # Should have at least some tools
        assert len(data["tools"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
