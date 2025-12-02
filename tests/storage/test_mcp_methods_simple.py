#!/usr/bin/env python3
"""
Simple functional tests for MCP protocol methods added for issue #1759.
Focus on verifying methods exist and handle basic error cases.
"""

from unittest.mock import Mock, patch

import pytest

from src.storage.neo4j_client import Neo4jInitializer
from src.storage.qdrant_client import VectorDBInitializer


class TestMCPMethodsExist:
    """Test that MCP protocol methods exist and are callable."""

    def test_neo4j_mcp_methods_exist(self):
        """Test Neo4j MCP methods exist with correct signatures."""
        client = Neo4jInitializer(test_mode=True)

        # Test methods exist
        assert hasattr(client, "create_node")
        assert hasattr(client, "query")
        assert hasattr(client, "create_relationship")

        # Test they're callable
        assert callable(client.create_node)
        assert callable(client.query)
        assert callable(client.create_relationship)

    def test_qdrant_mcp_methods_exist(self):
        """Test Qdrant MCP methods exist with correct signatures."""
        client = VectorDBInitializer(test_mode=True)

        # Test methods exist
        assert hasattr(client, "store_vector")
        assert hasattr(client, "search")
        assert hasattr(client, "get_collections")
        assert hasattr(client, "close")

        # Test they're callable
        assert callable(client.store_vector)
        assert callable(client.search)
        assert callable(client.get_collections)
        assert callable(client.close)


class TestMCPMethodsErrorHandling:
    """Test error handling when clients are not connected."""

    def test_neo4j_methods_require_connection(self):
        """Test Neo4j methods raise RuntimeError when not connected."""
        client = Neo4jInitializer(test_mode=True)
        client.driver = None  # Simulate no connection

        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            client.create_node(["Test"], {})

        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            client.query("RETURN 1")

        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            client.create_relationship("1", "2", "TEST")

    def test_qdrant_methods_require_connection(self):
        """Test Qdrant methods raise RuntimeError when not connected."""
        client = VectorDBInitializer(test_mode=True)
        client.client = None  # Simulate no connection

        with pytest.raises(RuntimeError, match="Not connected to Qdrant"):
            client.store_vector("test", [1, 2, 3])

        with pytest.raises(RuntimeError, match="Not connected to Qdrant"):
            client.search([1, 2, 3])

        with pytest.raises(RuntimeError, match="Not connected to Qdrant"):
            client.get_collections()

    def test_qdrant_close_method_works(self):
        """Test Qdrant close method doesn't raise errors."""
        client = VectorDBInitializer(test_mode=True)
        # Should not raise any exception
        client.close()
        assert True  # If we get here, close() worked


class TestMCPMethodsBasicFunctionality:
    """Test basic functionality with mocked dependencies."""

    @patch("src.storage.neo4j_client.yaml.safe_load")
    def test_neo4j_create_node_basic(self, mock_yaml):
        """Test Neo4j create_node with minimal mocking."""
        mock_yaml.return_value = {"neo4j": {"host": "localhost", "database": "test"}}

        client = Neo4jInitializer(test_mode=True)

        # Mock the session context manager
        mock_session = Mock()
        mock_session.run.return_value.single.return_value = {"node_id": "123"}

        with patch.object(client, "driver") as mock_driver:
            # Mock the context manager behavior
            mock_driver.session.return_value.__enter__.return_value = mock_session
            mock_driver.session.return_value.__exit__.return_value = None

            result = client.create_node(["Test"], {"data": "test"})

            assert result == "123"
            mock_session.run.assert_called_once()

    @patch("src.storage.qdrant_client.yaml.safe_load")
    def test_qdrant_store_vector_basic(self, mock_yaml):
        """Test Qdrant store_vector with minimal mocking."""
        mock_yaml.return_value = {"qdrant": {"host": "localhost", "collection_name": "test"}}

        client = VectorDBInitializer(test_mode=True)

        with patch.object(client, "client") as mock_client:
            mock_client.upsert.return_value = True

            result = client.store_vector("test_id", [1, 2, 3], {"meta": "data"})

            assert result == "test_id"
            mock_client.upsert.assert_called_once()


class TestHealthCheckMethods:
    """Test health check related functionality."""

    def test_health_check_retry_configuration(self):
        """Test that health check functions can be imported."""
        from src.mcp_server.main import _check_service_with_retries, _is_in_startup_grace_period

        # Test functions exist
        assert callable(_is_in_startup_grace_period)
        assert callable(_check_service_with_retries)

    def test_grace_period_basic_logic(self):
        """Test grace period basic functionality."""
        from src.mcp_server.main import _is_in_startup_grace_period

        # Test with very short grace period - should be outside
        with patch.dict("os.environ", {"HEALTH_CHECK_GRACE_PERIOD": "0"}):
            result = _is_in_startup_grace_period()
            assert result is False

    @pytest.mark.asyncio
    async def test_service_check_basic(self):
        """Test service check basic functionality."""
        from src.mcp_server.main import _check_service_with_retries

        # Mock a successful check
        def mock_check():
            return True

        with patch.dict("os.environ", {"HEALTH_CHECK_MAX_RETRIES": "1"}):
            status, error = await _check_service_with_retries("TestService", mock_check)

        assert status == "healthy"
        assert error == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
