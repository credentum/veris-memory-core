#!/usr/bin/env python3
"""
Comprehensive tests for KV Store to achieve 40% coverage.

This test suite covers:
- RedisConnector initialization and configuration
- Cache operations (set, get, delete)
- Session management
- Performance configuration loading
- Error handling and connection management
"""

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.storage.kv_store import CacheEntry, MetricEvent, RedisConnector


class TestMetricEventDataclass:
    """Test MetricEvent dataclass."""

    def test_metric_event_creation(self):
        """Test MetricEvent creation with all fields."""
        timestamp = datetime.utcnow()
        tags = {"env": "test", "component": "cache"}

        event = MetricEvent(
            timestamp=timestamp,
            metric_name="cache_hits",
            value=42.5,
            tags=tags,
            document_id="doc_123",
            agent_id="agent_456",
        )

        assert event.timestamp == timestamp
        assert event.metric_name == "cache_hits"
        assert event.value == 42.5
        assert event.tags == tags
        assert event.document_id == "doc_123"
        assert event.agent_id == "agent_456"

    def test_metric_event_optional_fields(self):
        """Test MetricEvent creation with only required fields."""
        timestamp = datetime.utcnow()
        tags = {"component": "test"}

        event = MetricEvent(timestamp=timestamp, metric_name="test_metric", value=100.0, tags=tags)

        assert event.document_id is None
        assert event.agent_id is None


class TestCacheEntryDataclass:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation with all fields."""
        created_at = datetime.utcnow()
        last_accessed = datetime.utcnow()

        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=created_at,
            ttl_seconds=3600,
            hit_count=5,
            last_accessed=last_accessed,
        )

        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.created_at == created_at
        assert entry.ttl_seconds == 3600
        assert entry.hit_count == 5
        assert entry.last_accessed == last_accessed

    def test_cache_entry_defaults(self):
        """Test CacheEntry creation with default values."""
        created_at = datetime.utcnow()

        entry = CacheEntry(
            key="test_key", value="test_value", created_at=created_at, ttl_seconds=1800
        )

        assert entry.hit_count == 0
        assert entry.last_accessed is None


class TestRedisConnectorInitialization:
    """Test RedisConnector initialization and configuration."""

    def test_redis_connector_init_with_config(self):
        """Test RedisConnector initialization with provided config."""
        config = {"redis": {"host": "localhost", "port": 6379, "database": 0}}

        connector = RedisConnector(config=config, test_mode=True, verbose=True)

        assert connector.config == config
        assert connector.test_mode is True
        assert connector.verbose is True
        assert connector.environment == "test"
        assert connector.redis_client is None

    def test_redis_connector_init_without_config(self):
        """Test RedisConnector initialization without config (uses parent class)."""
        with patch("src.storage.kv_store.DatabaseComponent.__init__") as mock_parent_init:
            mock_parent_init.return_value = None

            connector = RedisConnector(config_path="test.yaml", verbose=False)

            mock_parent_init.assert_called_once_with("test.yaml", False)

    def test_get_service_name(self):
        """Test _get_service_name returns correct service name."""
        config = {"redis": {}}
        connector = RedisConnector(config=config, test_mode=True)

        assert connector._get_service_name() == "redis"

    def test_load_performance_config_success(self):
        """Test loading performance configuration from file."""
        config = {"redis": {}}

        perf_data = {
            "kv_store": {
                "redis": {"connection_pool": {"max_size": 100}, "cache": {"ttl_seconds": 7200}}
            }
        }

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = ""
            with patch("yaml.safe_load", return_value=perf_data):
                connector = RedisConnector(config=config, test_mode=True)

                expected = {"connection_pool": {"max_size": 100}, "cache": {"ttl_seconds": 7200}}
                assert connector.perf_config == expected

    def test_load_performance_config_file_not_found(self):
        """Test handling of missing performance configuration file."""
        config = {"redis": {}}

        with patch("builtins.open", side_effect=FileNotFoundError):
            connector = RedisConnector(config=config, test_mode=True)

            assert connector.perf_config == {}

    def test_load_performance_config_invalid_structure(self):
        """Test handling of invalid performance configuration structure."""
        config = {"redis": {}}

        # Test when redis config is not a dict
        perf_data = {"kv_store": {"redis": "invalid"}}

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = ""
            with patch("yaml.safe_load", return_value=perf_data):
                connector = RedisConnector(config=config, test_mode=True)

                assert connector.perf_config == {}


class TestRedisConnectorConnection:
    """Test Redis connection functionality."""

    def test_connect_success(self):
        """Test successful Redis connection."""
        config = {"redis": {"host": "localhost", "port": 6379, "database": 0, "ssl": False}}

        connector = RedisConnector(config=config, test_mode=True)

        with patch("redis.ConnectionPool") as mock_pool_class:
            with patch("redis.Redis") as mock_redis_class:
                mock_pool = Mock()
                mock_pool_class.return_value = mock_pool

                mock_redis = Mock()
                mock_redis.ping.return_value = True
                mock_redis_class.return_value = mock_redis

                with patch.object(connector, "log_success") as mock_log:
                    result = connector.connect()

                    assert result is True
                    assert connector.is_connected is True
                    assert connector.redis_client == mock_redis
                    mock_log.assert_called_once()

    def test_connect_with_ssl(self):
        """Test Redis connection with SSL configuration."""
        config = {"redis": {"host": "redis.example.com", "port": 6380, "database": 1, "ssl": True}}

        connector = RedisConnector(config=config, test_mode=False)
        connector.environment = "production"

        with patch("redis.ConnectionPool") as mock_pool_class:
            with patch("redis.Redis") as mock_redis_class:
                mock_pool = Mock()
                mock_pool_class.return_value = mock_pool

                mock_redis = Mock()
                mock_redis.ping.return_value = True
                mock_redis_class.return_value = mock_redis

                result = connector.connect(password="test_password")

                # Verify SSL configuration was passed to pool
                call_args = mock_pool_class.call_args[1]
                assert call_args["ssl"] is True
                assert call_args["ssl_cert_reqs"] == "required"
                assert call_args["password"] == "test_password"

    def test_connect_failure(self):
        """Test Redis connection failure."""
        config = {"redis": {"host": "localhost", "port": 6379}}
        connector = RedisConnector(config=config, test_mode=True)

        with patch("redis.ConnectionPool", side_effect=Exception("Connection failed")):
            with patch.object(connector, "log_error") as mock_log:
                result = connector.connect()

                assert result is False
                assert connector.redis_client is None
                mock_log.assert_called_once()

    def test_connect_ping_failure(self):
        """Test Redis connection when ping fails."""
        config = {"redis": {"host": "localhost", "port": 6379}}
        connector = RedisConnector(config=config, test_mode=True)

        with patch("redis.ConnectionPool") as mock_pool_class:
            with patch("redis.Redis") as mock_redis_class:
                mock_pool = Mock()
                mock_pool_class.return_value = mock_pool

                mock_redis = Mock()
                mock_redis.ping.side_effect = Exception("Ping failed")
                mock_redis_class.return_value = mock_redis

                with patch.object(connector, "log_error") as mock_log:
                    result = connector.connect()

                    assert result is False
                    mock_log.assert_called_once()


class TestRedisConnectorCacheOperations:
    """Test cache operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"redis": {"prefixes": {"cache": "test_cache:"}}}
        self.connector = RedisConnector(config=self.config, test_mode=True)
        self.connector.redis_client = Mock()
        self.connector.is_connected = True

    def test_get_prefixed_key(self):
        """Test key prefixing functionality."""
        key = self.connector.get_prefixed_key("mykey", "cache")
        assert key == "test_cache:mykey"

        # Test default prefix when not configured
        key = self.connector.get_prefixed_key("mykey", "session")
        assert key == "session:mykey"

    def test_set_cache_success(self):
        """Test successful cache setting."""
        with patch("src.storage.kv_store.validate_redis_key", return_value=True):
            with patch.object(self.connector, "ensure_connected", return_value=True):
                with patch("src.storage.kv_store.datetime") as mock_datetime:
                    mock_now = datetime(2023, 1, 1, 12, 0, 0)
                    mock_datetime.utcnow.return_value = mock_now

                    result = self.connector.set_cache("test_key", {"data": "value"}, 3600)

                    assert result is True
                    self.connector.redis_client.setex.assert_called_once()

                    # Verify the call arguments
                    call_args = self.connector.redis_client.setex.call_args
                    assert call_args[0][0] == "test_cache:test_key"  # prefixed key
                    assert call_args[0][1] == 3600  # TTL

                    # Verify JSON data structure
                    json_data = call_args[0][2]
                    data = json.loads(json_data)
                    assert data["key"] == "test_key"
                    assert data["value"] == {"data": "value"}
                    assert data["ttl_seconds"] == 3600

    def test_set_cache_invalid_key(self):
        """Test cache setting with invalid key."""
        with patch("src.storage.kv_store.validate_redis_key", return_value=False):
            with patch.object(self.connector, "ensure_connected", return_value=True):
                with patch.object(self.connector, "log_error") as mock_log:
                    result = self.connector.set_cache("invalid key", "value")

                    assert result is False
                    mock_log.assert_called_once()

    def test_set_cache_not_connected(self):
        """Test cache setting when not connected."""
        with patch.object(self.connector, "ensure_connected", return_value=False):
            result = self.connector.set_cache("test_key", "value")

            assert result is False

    def test_set_cache_with_default_ttl(self):
        """Test cache setting with default TTL from performance config."""
        self.connector.perf_config = {"cache": {"ttl_seconds": 7200}}

        with patch("src.storage.kv_store.validate_redis_key", return_value=True):
            with patch.object(self.connector, "ensure_connected", return_value=True):
                with patch("src.storage.kv_store.datetime") as mock_datetime:
                    mock_datetime.utcnow.return_value = datetime.utcnow()

                    result = self.connector.set_cache("test_key", "value")

                    assert result is True
                    # Verify TTL was set to default
                    call_args = self.connector.redis_client.setex.call_args
                    assert call_args[0][1] == 7200

    def test_get_cache_success(self):
        """Test successful cache retrieval."""
        # Mock cache entry data
        cache_data = {
            "key": "test_key",
            "value": {"data": "cached_value"},
            "created_at": "2023-01-01T12:00:00",
            "ttl_seconds": 3600,
            "hit_count": 2,
        }

        self.connector.redis_client.get.return_value = json.dumps(cache_data)
        self.connector.redis_client.ttl.return_value = 1800  # 30 minutes left

        with patch.object(self.connector, "ensure_connected", return_value=True):
            with patch("src.storage.kv_store.datetime") as mock_datetime:
                mock_datetime.utcnow.return_value.isoformat.return_value = "2023-01-01T12:30:00"

                result = self.connector.get_cache("test_key")

                assert result == {"data": "cached_value"}

                # Verify hit count was incremented and last_accessed updated
                update_call = self.connector.redis_client.setex.call_args
                updated_data = json.loads(update_call[0][2])
                assert updated_data["hit_count"] == 3
                assert updated_data["last_accessed"] == "2023-01-01T12:30:00"

    def test_get_cache_not_found(self):
        """Test cache retrieval when key not found."""
        self.connector.redis_client.get.return_value = None

        with patch.object(self.connector, "ensure_connected", return_value=True):
            result = self.connector.get_cache("nonexistent_key")

            assert result is None

    def test_get_cache_not_connected(self):
        """Test cache retrieval when not connected."""
        with patch.object(self.connector, "ensure_connected", return_value=False):
            result = self.connector.get_cache("test_key")

            assert result is None

    def test_get_cache_exception(self):
        """Test cache retrieval with exception."""
        self.connector.redis_client.get.side_effect = Exception("Redis error")

        with patch.object(self.connector, "ensure_connected", return_value=True):
            with patch.object(self.connector, "log_error") as mock_log:
                result = self.connector.get_cache("test_key")

                assert result is None
                mock_log.assert_called_once()

    def test_delete_cache_success(self):
        """Test successful cache deletion."""
        self.connector.redis_client.scan_iter.return_value = iter(
            ["test_cache:pattern_key1", "test_cache:pattern_key2"]
        )
        self.connector.redis_client.delete.return_value = 2

        with patch.object(self.connector, "ensure_connected", return_value=True):
            result = self.connector.delete_cache("pattern_*")

            assert result == 2
            self.connector.redis_client.scan_iter.assert_called_once_with(
                match="test_cache:pattern_*"
            )

    def test_delete_cache_no_matches(self):
        """Test cache deletion with no matching keys."""
        self.connector.redis_client.scan_iter.return_value = iter([])

        with patch.object(self.connector, "ensure_connected", return_value=True):
            result = self.connector.delete_cache("nonexistent_*")

            assert result == 0

    def test_delete_cache_not_connected(self):
        """Test cache deletion when not connected."""
        with patch.object(self.connector, "ensure_connected", return_value=False):
            result = self.connector.delete_cache("pattern_*")

            assert result == 0


class TestRedisConnectorSessionOperations:
    """Test session management operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"redis": {"prefixes": {"session": "sess:"}}}
        self.connector = RedisConnector(config=self.config, test_mode=True)
        self.connector.redis_client = Mock()
        self.connector.is_connected = True

    def test_set_session_success(self):
        """Test successful session creation."""
        session_data = {"user_id": "123", "preferences": {"theme": "dark"}}

        with patch.object(self.connector, "ensure_connected", return_value=True):
            with patch("src.storage.kv_store.datetime") as mock_datetime:
                mock_now = datetime(2023, 1, 1, 12, 0, 0)
                mock_datetime.utcnow.return_value = mock_now

                result = self.connector.set_session("session_123", session_data, 7200)

                assert result is True

                # Verify Redis call
                call_args = self.connector.redis_client.setex.call_args
                assert call_args[0][0] == "sess:session_123"
                assert call_args[0][1] == 7200

                # Verify session data structure
                stored_data = json.loads(call_args[0][2])
                assert stored_data["id"] == "session_123"
                assert stored_data["data"] == session_data
                assert "created_at" in stored_data
                assert "last_activity" in stored_data

    def test_set_session_exception(self):
        """Test session creation with exception."""
        self.connector.redis_client.setex.side_effect = Exception("Redis error")

        with patch.object(self.connector, "ensure_connected", return_value=True):
            with patch.object(self.connector, "log_error") as mock_log:
                result = self.connector.set_session("session_123", {"data": "test"})

                assert result is False
                mock_log.assert_called_once()


class TestRedisConnectorErrorHandling:
    """Test error handling and edge cases."""

    def test_redis_client_none_handling(self):
        """Test operations when redis_client is None."""
        config = {"redis": {}}
        connector = RedisConnector(config=config, test_mode=True)
        connector.redis_client = None  # Simulate failed connection

        with patch.object(connector, "ensure_connected", return_value=True):
            # Should handle None client gracefully
            result = connector.get_cache("test_key")
            assert result is None

            result = connector.delete_cache("pattern_*")
            assert result == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
