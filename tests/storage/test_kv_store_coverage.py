#!/usr/bin/env python3
"""
Comprehensive tests for the KV store to increase coverage.

This test suite covers key-value storage operations and Redis client
to achieve high code coverage for the storage.kv_store module.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.storage.kv_store import ContextKV, RedisConnector  # noqa: E402


class TestRedisConnector:
    """Test suite for RedisConnector class."""

    def test_init_default(self):
        """Test RedisConnector initialization with defaults."""
        connector = RedisConnector()
        assert connector.redis_client is None
        assert connector.host == "localhost"
        assert connector.port == 6379
        assert connector.password is None

    def test_init_with_params(self):
        """Test RedisConnector initialization with custom parameters."""
        connector = RedisConnector(
            host="custom_host",
            port=9999,
            password="test_pass",
            ssl=True,
            ssl_cert_reqs="required",
        )
        assert connector.host == "custom_host"
        assert connector.port == 9999
        assert connector.password == "test_pass"

    @patch("storage.kv_store.redis.Redis")
    def test_connect_success(self, mock_redis):
        """Test successful Redis connection."""
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        result = connector.connect()

        assert result is True
        assert connector.redis_client == mock_redis_instance
        mock_redis.assert_called_once_with()

    @patch("storage.kv_store.redis.Redis")
    def test_connect_with_ssl(self, mock_redis):
        """Test Redis connection with SSL."""
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        result = connector.connect(ssl=True, ssl_cert_reqs="required")

        assert result is True
        mock_redis.assert_called_with(
            host="localhost",
            port=6379,
            password=None,
            ssl=True,
            ssl_cert_reqs="required",
        )

    @patch("storage.kv_store.redis.Redis")
    def test_connect_failure(self, mock_redis):
        """Test Redis connection failure."""
        mock_redis.side_effect = Exception("Connection failed")

        connector = RedisConnector()
        result = connector.connect()

        assert result is False
        assert connector.redis_client is None

    @patch("storage.kv_store.redis.Redis")
    def test_set_success(self, mock_redis):
        """Test successful Redis set operation."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.set("test_key", "test_value", ex=3600)

        assert result is True
        mock_redis_instance.set.assert_called_with("test_key", "test_value", ex=3600)

    def test_set_no_client(self):
        """Test Redis set operation without connected client."""
        connector = RedisConnector()
        result = connector.set("test_key", "test_value")

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_set_failure(self, mock_redis):
        """Test Redis set operation failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.side_effect = Exception("Set failed")
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.set("test_key", "test_value")

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_get_success(self, mock_redis):
        """Test successful Redis get operation."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = b"test_value"
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.get("test_key")

        assert result == "test_value"
        mock_redis_instance.get.assert_called_with("test_key")

    @patch("storage.kv_store.redis.Redis")
    def test_get_none_result(self, mock_redis):
        """Test Redis get operation returning None."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.get("test_key")

        assert result is None

    def test_get_no_client(self):
        """Test Redis get operation without connected client."""
        connector = RedisConnector()
        result = connector.get("test_key")

        assert result is None

    @patch("storage.kv_store.redis.Redis")
    def test_get_failure(self, mock_redis):
        """Test Redis get operation failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.side_effect = Exception("Get failed")
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.get("test_key")

        assert result is None

    @patch("storage.kv_store.redis.Redis")
    def test_delete_success(self, mock_redis):
        """Test successful Redis delete operation."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.delete.return_value = 1
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.delete("test_key")

        assert result is True
        mock_redis_instance.delete.assert_called_with("test_key")

    @patch("storage.kv_store.redis.Redis")
    def test_delete_not_found(self, mock_redis):
        """Test Redis delete operation for non-existent key."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.delete.return_value = 0
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.delete("test_key")

        assert result is False

    def test_delete_no_client(self):
        """Test Redis delete operation without connected client."""
        connector = RedisConnector()
        result = connector.delete("test_key")

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_delete_failure(self, mock_redis):
        """Test Redis delete operation failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.delete.side_effect = Exception("Delete failed")
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.delete("test_key")

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_exists_true(self, mock_redis):
        """Test Redis exists operation returning True."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.exists.return_value = 1
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.exists("test_key")

        assert result is True
        mock_redis_instance.exists.assert_called_with("test_key")

    @patch("storage.kv_store.redis.Redis")
    def test_exists_false(self, mock_redis):
        """Test Redis exists operation returning False."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.exists.return_value = 0
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.exists("test_key")

        assert result is False

    def test_exists_no_client(self):
        """Test Redis exists operation without connected client."""
        connector = RedisConnector()
        result = connector.exists("test_key")

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_exists_failure(self, mock_redis):
        """Test Redis exists operation failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.exists.side_effect = Exception("Exists failed")
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.exists("test_key")

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_ping_success(self, mock_redis):
        """Test successful Redis ping."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.ping()

        assert result is True
        mock_redis_instance.ping.assert_called_once_with()

    def test_ping_no_client(self):
        """Test Redis ping without connected client."""
        connector = RedisConnector()
        result = connector.ping()

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_ping_failure(self, mock_redis):
        """Test Redis ping failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.side_effect = Exception("Ping failed")
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        result = connector.ping()

        assert result is False

    @patch("storage.kv_store.redis.Redis")
    def test_close_success(self, mock_redis):
        """Test successful Redis connection close."""
        mock_redis_instance = AsyncMock()
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        connector.close()

        mock_redis_instance.close.assert_called_once_with()
        assert connector.redis_client is None

    def test_close_no_client(self):
        """Test Redis close without connected client."""
        connector = RedisConnector()
        connector.close()  # Should not raise exception

    @patch("storage.kv_store.redis.Redis")
    def test_close_failure(self, mock_redis):
        """Test Redis close with exception."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.close.side_effect = Exception("Close failed")
        mock_redis.return_value = mock_redis_instance

        connector = RedisConnector()
        connector.connect()
        connector.close()  # Should not raise exception

        assert connector.redis_client is None


class TestContextKV:
    """Test suite for ContextKV class."""

    @patch("storage.kv_store.RedisConnector")
    def test_init_default(self, mock_redis_connector):
        """Test ContextKV initialization with defaults."""
        kv = ContextKV()
        assert kv.redis is not None
        assert kv.default_ttl == 86400  # 24 hours

    @patch("storage.kv_store.RedisConnector")
    def test_init_with_params(self, mock_redis_connector):
        """Test ContextKV initialization with custom parameters."""
        kv = ContextKV(default_ttl=3600)
        assert kv.default_ttl == 3600

    @patch("storage.kv_store.RedisConnector")
    def test_connect_success(self, mock_redis_connector):
        """Test successful connection."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.connect.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.connect()

        assert result is True
        mock_redis_instance.connect.assert_called_once_with()

    @patch("storage.kv_store.RedisConnector")
    def test_connect_with_redis_password(self, mock_redis_connector):
        """Test connection with Redis password."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.connect.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.connect(redis_password="test_pass")

        assert result is True
        mock_redis_connector.assert_called_with(password="test_pass")

    @patch("storage.kv_store.RedisConnector")
    def test_connect_with_ssl(self, mock_redis_connector):
        """Test connection with SSL parameters."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.connect.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.connect(ssl=True, ssl_cert_reqs="required")

        assert result is True
        mock_redis_instance.connect.assert_called_with(ssl=True, ssl_cert_reqs="required")

    @patch("storage.kv_store.RedisConnector")
    def test_connect_failure(self, mock_redis_connector):
        """Test connection failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.connect.return_value = False
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.connect()

        assert result is False

    @patch("storage.kv_store.RedisConnector")
    def test_store_context_success(self, mock_redis_connector):
        """Test successful context storage."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        context_data = {"type": "test", "content": "sample content"}
        result = kv.store_context("test_id", context_data)

        assert result is True
        mock_redis_instance.set.assert_called_once_with()
        call_args = mock_redis_instance.set.call_args
        assert call_args[0][0] == "context:test_id"
        assert "test" in call_args[0][1]
        assert call_args[1]["ex"] == 86400

    @patch("storage.kv_store.RedisConnector")
    def test_store_context_with_ttl(self, mock_redis_connector):
        """Test context storage with custom TTL."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        context_data = {"type": "test", "content": "sample content"}
        result = kv.store_context("test_id", context_data, ttl=3600)

        assert result is True
        call_args = mock_redis_instance.set.call_args
        assert call_args[1]["ex"] == 3600

    @patch("storage.kv_store.RedisConnector")
    def test_store_context_failure(self, mock_redis_connector):
        """Test context storage failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = False
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.store_context("test_id", {"test": "data"})

        assert result is False

    @patch("storage.kv_store.RedisConnector")
    def test_retrieve_context_success(self, mock_redis_connector):
        """Test successful context retrieval."""
        stored_data = {"type": "test", "content": "sample content"}
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = json.dumps(stored_data)
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.retrieve_context("test_id")

        assert result == stored_data
        mock_redis_instance.get.assert_called_with("context:test_id")

    @patch("storage.kv_store.RedisConnector")
    def test_retrieve_context_not_found(self, mock_redis_connector):
        """Test context retrieval for non-existent key."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = None
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.retrieve_context("test_id")

        assert result is None

    @patch("storage.kv_store.RedisConnector")
    def test_retrieve_context_invalid_json(self, mock_redis_connector):
        """Test context retrieval with invalid JSON."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = "invalid json"
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.retrieve_context("test_id")

        assert result is None

    @patch("storage.kv_store.RedisConnector")
    def test_delete_context_success(self, mock_redis_connector):
        """Test successful context deletion."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.delete.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.delete_context("test_id")

        assert result is True
        mock_redis_instance.delete.assert_called_with("context:test_id")

    @patch("storage.kv_store.RedisConnector")
    def test_delete_context_failure(self, mock_redis_connector):
        """Test context deletion failure."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.delete.return_value = False
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.delete_context("test_id")

        assert result is False

    @patch("storage.kv_store.RedisConnector")
    def test_context_exists_true(self, mock_redis_connector):
        """Test context existence check returning True."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.exists.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.context_exists("test_id")

        assert result is True
        mock_redis_instance.exists.assert_called_with("context:test_id")

    @patch("storage.kv_store.RedisConnector")
    def test_context_exists_false(self, mock_redis_connector):
        """Test context existence check returning False."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.exists.return_value = False
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.context_exists("test_id")

        assert result is False

    @patch("storage.kv_store.RedisConnector")
    def test_get_health_status_healthy(self, mock_redis_connector):
        """Test health status when Redis is healthy."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.get_health_status()

        assert result["status"] == "healthy"
        assert result["redis"]["connected"] is True

    @patch("storage.kv_store.RedisConnector")
    def test_get_health_status_unhealthy(self, mock_redis_connector):
        """Test health status when Redis is unhealthy."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = False
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        result = kv.get_health_status()

        assert result["status"] == "unhealthy"
        assert result["redis"]["connected"] is False

    @patch("storage.kv_store.RedisConnector")
    def test_close(self, mock_redis_connector):
        """Test KV store close."""
        mock_redis_instance = AsyncMock()
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        kv.close()

        mock_redis_instance.close.assert_called_once_with()


class TestContextKVAdvanced:
    """Test suite for advanced ContextKV operations."""

    @patch("storage.kv_store.RedisConnector")
    def test_store_context_metadata_preservation(self, mock_redis_connector):
        """Test that context metadata is preserved during storage."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        context_data = {
            "type": "decision",
            "content": {"title": "API Design", "description": "REST API decision"},
            "metadata": {"author": "dev", "priority": "high"},
            "created_at": "2024-01-01T00:00:00Z",
        }

        result = kv.store_context("decision_001", context_data)

        assert result is True
        call_args = mock_redis_instance.set.call_args
        stored_json = call_args[0][1]
        stored_data = json.loads(stored_json)

        assert stored_data["type"] == "decision"
        assert stored_data["metadata"]["author"] == "dev"
        assert "created_at" in stored_data

    @patch("storage.kv_store.RedisConnector")
    def test_store_context_with_complex_data(self, mock_redis_connector):
        """Test storing context with complex nested data structures."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        complex_data = {
            "type": "trace",
            "content": {
                "execution_path": [
                    {"step": 1, "function": "validate_input", "duration": 0.01},
                    {"step": 2, "function": "process_data", "duration": 0.15},
                    {"step": 3, "function": "save_result", "duration": 0.03},
                ],
                "variables": {
                    "input_size": 1024,
                    "processing_time": 0.19,
                    "memory_usage": "2.5MB",
                },
            },
        }

        result = kv.store_context("trace_001", complex_data)

        assert result is True
        call_args = mock_redis_instance.set.call_args
        stored_json = call_args[0][1]
        stored_data = json.loads(stored_json)

        assert len(stored_data["content"]["execution_path"]) == 3
        assert stored_data["content"]["variables"]["input_size"] == 1024

    @patch("storage.kv_store.RedisConnector")
    def test_retrieve_context_data_integrity(self, mock_redis_connector):
        """Test that retrieved context maintains data integrity."""
        original_data = {
            "type": "log",
            "content": {"level": "ERROR", "message": "Database connection failed"},
            "metadata": {"service": "api", "environment": "production"},
            "timestamp": 1640995200,
        }

        mock_redis_instance = AsyncMock()
        mock_redis_instance.get.return_value = json.dumps(original_data)
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()
        retrieved_data = kv.retrieve_context("log_001")

        assert retrieved_data == original_data
        assert retrieved_data["timestamp"] == 1640995200
        assert retrieved_data["metadata"]["environment"] == "production"

    @patch("storage.kv_store.RedisConnector")
    def test_context_operations_with_special_characters(self, mock_redis_connector):
        """Test context operations with special characters and Unicode."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis_instance.get.return_value = json.dumps(
            {"content": "测试内容 with émojis 🚀 and symbols @#$%"}
        )
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()

        # Test storage with special characters in ID
        result = kv.store_context("test-id_with-symbols@123", {"content": "test"})
        assert result is True

        # Test retrieval with Unicode content
        retrieved = kv.retrieve_context("unicode_test")
        assert "测试内容" in retrieved["content"]
        assert "🚀" in retrieved["content"]

    @patch("storage.kv_store.RedisConnector")
    def test_ttl_edge_cases(self, mock_redis_connector):
        """Test TTL edge cases and validation."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()

        # Test with zero TTL (should use default)
        kv.store_context("test_id", {"test": "data"}, ttl=0)
        call_args = mock_redis_instance.set.call_args
        assert call_args[1]["ex"] == 86400  # Default TTL

        # Test with negative TTL (should use default)
        kv.store_context("test_id", {"test": "data"}, ttl=-100)
        call_args = mock_redis_instance.set.call_args
        assert call_args[1]["ex"] == 86400  # Default TTL

    @patch("storage.kv_store.RedisConnector")
    def test_concurrent_operations_simulation(self, mock_redis_connector):
        """Test simulation of concurrent operations."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.set.return_value = True
        mock_redis_instance.get.return_value = json.dumps({"version": 1})
        mock_redis_instance.exists.return_value = True
        mock_redis_connector.return_value = mock_redis_instance

        kv = ContextKV()

        # Simulate multiple rapid operations
        operations = []
        for i in range(10):
            operations.extend(
                [
                    kv.store_context(f"concurrent_{i}", {"data": i}),
                    kv.context_exists(f"concurrent_{i}"),
                    kv.retrieve_context(f"concurrent_{i}"),
                ]
            )

        # All operations should succeed
        assert all(op is not False for op in operations if op is not None)

    @patch("storage.kv_store.ContextKV")
    def test_main_function(mock_context_kv):
        """Test the main function."""
        from src.storage.kv_store import main

        mock_instance = AsyncMock()
        mock_instance.connect.return_value = True
        mock_instance.get_health_status.return_value = {"status": "healthy"}
        mock_context_kv.return_value = mock_instance

        # Test successful execution
        with patch("sys.argv", ["kv_store.py"]):
            main()

        mock_instance.connect.assert_called_once_with()
        mock_instance.get_health_status.assert_called_once_with()


@patch("storage.kv_store.ContextKV")
def test_main_function_connection_failure(mock_context_kv):
    """Test main function with connection failure."""
    from src.storage.kv_store import main

    mock_instance = AsyncMock()
    mock_instance.connect.return_value = False
    mock_context_kv.return_value = mock_instance

    # Test connection failure
    with patch("sys.argv", ["kv_store.py"]):
        with pytest.raises(SystemExit):
            main()


@patch("storage.kv_store.ContextKV")
def test_main_function_with_exception(mock_context_kv):
    """Test main function with exception during execution."""
    from src.storage.kv_store import main

    mock_context_kv.side_effect = Exception("Initialization error")

    # Test initialization with exception
    with patch("sys.argv", ["kv_store.py"]):
        with pytest.raises(SystemExit):
            main()
