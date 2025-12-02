#!/usr/bin/env python3
"""
Final coverage push to reach 80%+ for server.py
"""

import os
from unittest.mock import Mock, patch

import pytest

from src.mcp_server import server
from src.mcp_server.server import get_health_status


class TestFinalCoverage:
    """Tests to push coverage above 80% for server.py"""

    @pytest.mark.asyncio
    async def test_health_status_timeout_errors(self):
        """Test health status with timeout errors."""
        # Test Neo4j timeout - ensure driver exists
        mock_neo4j = Mock()
        mock_neo4j.driver = Mock()  # Ensure driver exists
        mock_session = Mock()
        # session.run() returns a result, and .single() is called on that result
        mock_result = Mock()
        mock_result.single.side_effect = TimeoutError("Connection timeout")
        mock_session.run.return_value = mock_result

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_neo4j.driver.session = Mock(return_value=mock_context_manager)

        with (
            patch.object(server, "neo4j_client", mock_neo4j),
            patch.object(server, "qdrant_client", None),
            patch.object(server, "kv_store", None),
        ):
            status = await get_health_status()
            assert status["status"] == "degraded"
            assert status["services"]["neo4j"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_status_qdrant_timeout(self):
        """Test health status with Qdrant timeout."""
        mock_qdrant = Mock()
        mock_qdrant.client = Mock()
        mock_qdrant.client.get_collections.side_effect = TimeoutError("Qdrant timeout")

        with (
            patch.object(server, "neo4j_client", None),
            patch.object(server, "qdrant_client", mock_qdrant),
            patch.object(server, "kv_store", None),
        ):
            status = await get_health_status()
            assert status["status"] == "degraded"
            assert status["services"]["qdrant"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_status_redis_timeout(self):
        """Test health status with Redis timeout."""
        mock_kv = Mock()
        mock_kv.redis = Mock()
        mock_kv.redis.redis_client = Mock()
        mock_kv.redis.redis_client.ping.side_effect = TimeoutError("Redis timeout")

        with (
            patch.object(server, "neo4j_client", None),
            patch.object(server, "qdrant_client", None),
            patch.object(server, "kv_store", mock_kv),
        ):
            status = await get_health_status()
            assert status["status"] == "degraded"
            assert status["services"]["redis"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_status_generic_exceptions(self):
        """Test health status with generic exceptions."""
        # Test Neo4j with proper context manager setup
        mock_neo4j = Mock()
        mock_neo4j.driver = Mock()
        mock_session = Mock()
        # session.run() returns a result, and .single() is called on that result
        mock_result = Mock()
        mock_result.single.side_effect = RuntimeError("Generic Neo4j error")
        mock_session.run.return_value = mock_result

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_neo4j.driver.session = Mock(return_value=mock_context_manager)

        # Test Qdrant
        mock_qdrant = Mock()
        mock_qdrant.client = Mock()
        mock_qdrant.client.get_collections.side_effect = RuntimeError("Generic Qdrant error")

        # Test Redis/KV
        mock_kv = Mock()
        mock_kv.redis = Mock()
        mock_kv.redis.redis_client = Mock()
        mock_kv.redis.redis_client.ping.side_effect = RuntimeError("Generic Redis error")

        with (
            patch.object(server, "neo4j_client", mock_neo4j),
            patch.object(server, "qdrant_client", mock_qdrant),
            patch.object(server, "kv_store", mock_kv),
        ):
            status = await get_health_status()
            assert status["status"] == "degraded"
            assert status["services"]["neo4j"] == "unhealthy"
            assert status["services"]["qdrant"] == "unhealthy"
            assert status["services"]["redis"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_initialize_storage_missing_neo4j_user(self):
        """Test initialization without NEO4J_USER set."""
        from src.mcp_server.server import initialize_storage_clients

        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl_manager,
            patch("src.mcp_server.server.Neo4jInitializer") as mock_neo4j,
            patch.dict(os.environ, {"NEO4J_PASSWORD": "test"}, clear=True),
        ):
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_instance = Mock()  # Not AsyncMock
            ssl_manager_instance.validate_ssl_certificates.return_value = {"neo4j": True}
            ssl_manager_instance.get_neo4j_ssl_config.return_value = {"encrypted": True}
            mock_ssl_manager.return_value = ssl_manager_instance

            neo4j_instance = Mock()  # Not AsyncMock
            neo4j_instance.connect.return_value = True
            mock_neo4j.return_value = neo4j_instance

            await initialize_storage_clients()

            # Should call with default user "neo4j"
            neo4j_instance.connect.assert_called()
            call_args = neo4j_instance.connect.call_args[1]
            assert call_args["username"] == "neo4j"

    @pytest.mark.asyncio
    async def test_initialize_storage_no_redis_password(self):
        """Test Redis initialization without password."""
        from src.mcp_server.server import initialize_storage_clients

        with (
            patch("src.mcp_server.server.validate_all_configs") as mock_validate,
            patch("src.mcp_server.server.SSLConfigManager") as mock_ssl_manager,
            patch("src.mcp_server.server.ContextKV") as mock_kv,
            patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}, clear=True),
        ):
            mock_validate.return_value = {"valid": True, "config": {}}
            ssl_manager_instance = Mock()  # Not AsyncMock
            ssl_manager_instance.validate_ssl_certificates.return_value = {"redis": True}
            ssl_manager_instance.get_redis_ssl_config.return_value = {"ssl": False}
            mock_ssl_manager.return_value = ssl_manager_instance

            kv_instance = Mock()  # Not AsyncMock
            kv_instance.connect.return_value = True
            mock_kv.return_value = kv_instance

            await initialize_storage_clients()

            # Should call connect with redis_password=None
            kv_instance.connect.assert_called()
            call_args = kv_instance.connect.call_args[1]
            assert call_args["redis_password"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
