#!/usr/bin/env python3
"""
Test suite for storage/kv_store.py - Key-Value store tests
"""
import pytest
import json
import hashlib
import time
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock, mock_open, call

# Mock dependencies to avoid import issues
import sys
from unittest.mock import MagicMock

# Mock duckdb module
duckdb_mock = MagicMock()
sys.modules['duckdb'] = duckdb_mock

# Mock redis module
redis_mock = MagicMock()
sys.modules['redis'] = redis_mock

# Mock base component
base_component_mock = MagicMock()
base_component_mock.DatabaseComponent = MagicMock
sys.modules['core.base_component'] = base_component_mock

# Mock validators
validators_mock = MagicMock()
validators_mock.sanitize_metric_name.return_value = "sanitized_metric"
validators_mock.validate_metric_event.return_value = True
validators_mock.validate_redis_key.return_value = True
validators_mock.validate_time_range.return_value = True
sys.modules['validators.kv_validators'] = validators_mock

# Now import the module
import importlib.util
import os

module_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'storage', 'kv_store.py')
spec = importlib.util.spec_from_file_location("kv_store", module_path)
kv_store_module = importlib.util.module_from_spec(spec)
sys.modules['kv_store'] = kv_store_module
spec.loader.exec_module(kv_store_module)

MetricEvent = kv_store_module.MetricEvent
CacheEntry = kv_store_module.CacheEntry
RedisConnector = kv_store_module.RedisConnector
DuckDBAnalytics = kv_store_module.DuckDBAnalytics
ContextKV = kv_store_module.ContextKV


class TestMetricEvent:
    """Test suite for MetricEvent dataclass"""

    def test_metric_event_creation(self):
        """Test MetricEvent creation with all fields"""
        timestamp = datetime.utcnow()
        tags = {"environment": "test", "version": "1.0"}
        
        event = MetricEvent(
            timestamp=timestamp,
            metric_name="api.requests",
            value=1.5,
            tags=tags,
            document_id="doc_123",
            agent_id="agent_456"
        )
        
        assert event.timestamp == timestamp
        assert event.metric_name == "api.requests"
        assert event.value == 1.5
        assert event.tags == tags
        assert event.document_id == "doc_123"
        assert event.agent_id == "agent_456"

    def test_metric_event_optional_fields(self):
        """Test MetricEvent creation with optional fields as None"""
        timestamp = datetime.utcnow()
        
        event = MetricEvent(
            timestamp=timestamp,
            metric_name="cpu.usage",
            value=0.75,
            tags={}
        )
        
        assert event.timestamp == timestamp
        assert event.metric_name == "cpu.usage"
        assert event.value == 0.75
        assert event.tags == {}
        assert event.document_id is None
        assert event.agent_id is None


class TestCacheEntry:
    """Test suite for CacheEntry dataclass"""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation with all fields"""
        created_at = datetime.utcnow()
        last_accessed = datetime.utcnow()
        
        entry = CacheEntry(
            key="test_key",
            value={"data": "test_value"},
            created_at=created_at,
            ttl_seconds=3600,
            hit_count=5,
            last_accessed=last_accessed
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test_value"}
        assert entry.created_at == created_at
        assert entry.ttl_seconds == 3600
        assert entry.hit_count == 5
        assert entry.last_accessed == last_accessed

    def test_cache_entry_defaults(self):
        """Test CacheEntry creation with default values"""
        created_at = datetime.utcnow()
        
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=created_at,
            ttl_seconds=1800
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.created_at == created_at
        assert entry.ttl_seconds == 1800
        assert entry.hit_count == 0  # Default
        assert entry.last_accessed is None  # Default


class TestRedisConnectorInit:
    """Test suite for RedisConnector initialization"""

    def test_init_with_config_dict(self):
        """Test initialization with provided config dictionary"""
        config = {
            "redis": {"host": "localhost", "port": 6379},
            "performance": {"cache": {"ttl_seconds": 7200}}
        }
        
        with patch("builtins.open", mock_open(read_data='{"redis": {"connection_pool": {"max_size": 100}}}')):
            with patch("yaml.safe_load", return_value={"kv_store": {"redis": {"connection_pool": {"max_size": 100}}}}):
                connector = RedisConnector(config=config, test_mode=True)
        
        assert connector.config == config
        assert connector.test_mode is True
        assert connector.redis_client is None
        assert connector.environment == "test"

    def test_init_without_config(self):
        """Test initialization without config (calls parent)"""
        mock_parent = MagicMock()
        
        with patch("builtins.open", mock_open(read_data='{}')):
            with patch("yaml.safe_load", return_value={}):
                with patch.object(base_component_mock.DatabaseComponent, "__init__", mock_parent):
                    connector = RedisConnector(config_path="test.yaml", verbose=True)
        
        # Should call parent init when no config provided
        mock_parent.assert_called_once_with("test.yaml", True)

    def test_init_test_mode(self):
        """Test initialization in test mode"""
        config = {"redis": {"host": "test_host"}}
        
        with patch("builtins.open", mock_open(read_data='{}')):
            with patch("yaml.safe_load", return_value={}):
                connector = RedisConnector(config=config, test_mode=True)
        
        assert connector.test_mode is True
        assert connector.environment == "test"

    def test_get_service_name(self):
        """Test service name getter"""
        connector = RedisConnector(config={}, test_mode=True)
        assert connector._get_service_name() == "redis"


class TestRedisConnectorPerformanceConfig:
    """Test suite for performance configuration loading"""

    def test_load_performance_config_success(self):
        """Test successful performance config loading"""
        perf_config = {
            "kv_store": {
                "redis": {
                    "connection_pool": {"max_size": 100},
                    "cache": {"ttl_seconds": 7200}
                }
            }
        }
        
        with patch("builtins.open", mock_open(read_data="dummy")):
            with patch("yaml.safe_load", return_value=perf_config):
                connector = RedisConnector(config={}, test_mode=True)
                result = connector._load_performance_config()
        
        expected = {"connection_pool": {"max_size": 100}, "cache": {"ttl_seconds": 7200}}
        assert result == expected

    def test_load_performance_config_file_not_found(self):
        """Test performance config loading when file doesn't exist"""
        with patch("builtins.open", side_effect=FileNotFoundError):
            connector = RedisConnector(config={}, test_mode=True)
            result = connector._load_performance_config()
        
        assert result == {}

    def test_load_performance_config_invalid_structure(self):
        """Test performance config loading with invalid structure"""
        perf_config = {"kv_store": "invalid_structure"}  # Not a dict
        
        with patch("builtins.open", mock_open(read_data="dummy")):
            with patch("yaml.safe_load", return_value=perf_config):
                connector = RedisConnector(config={}, test_mode=True)
                result = connector._load_performance_config()
        
        assert result == {}

    def test_load_performance_config_missing_sections(self):
        """Test performance config loading with missing sections"""
        perf_config = {"other_config": {"value": 123}}  # Missing kv_store
        
        with patch("builtins.open", mock_open(read_data="dummy")):
            with patch("yaml.safe_load", return_value=perf_config):
                connector = RedisConnector(config={}, test_mode=True)
                result = connector._load_performance_config()
        
        assert result == {}


class TestRedisConnectorConnection:
    """Test suite for Redis connection functionality"""

    def test_connect_success(self):
        """Test successful Redis connection"""
        config = {"redis": {"host": "localhost", "port": 6379, "database": 0, "ssl": False}}
        connector = RedisConnector(config=config, test_mode=True)
        
        # Mock Redis components
        mock_pool = MagicMock()
        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True
        
        with patch("redis.ConnectionPool", return_value=mock_pool):
            with patch("redis.Redis", return_value=mock_redis_client):
                with patch.object(connector, "log_success") as mock_log:
                    result = connector.connect(password="test_password")
        
        assert result is True
        assert connector.redis_client == mock_redis_client
        assert connector.is_connected is True
        mock_log.assert_called_once_with("Connected to Redis at localhost:6379")

    def test_connect_with_ssl(self):
        """Test Redis connection with SSL enabled"""
        config = {"redis": {"host": "secure.redis.com", "port": 6380, "ssl": True}}
        connector = RedisConnector(config=config, test_mode=False)  # Production mode
        connector.environment = "production"
        
        mock_pool = MagicMock()
        mock_redis_client = MagicMock()
        
        with patch("redis.ConnectionPool", return_value=mock_pool) as mock_pool_constructor:
            with patch("redis.Redis", return_value=mock_redis_client):
                with patch.object(connector, "log_success"):
                    result = connector.connect(password="secure_password")
        
        assert result is True
        # Check SSL configuration was passed
        call_kwargs = mock_pool_constructor.call_args[1]
        assert call_kwargs["ssl"] is True
        assert call_kwargs["ssl_cert_reqs"] == "required"

    def test_connect_with_ssl_development(self):
        """Test Redis connection with SSL in development (relaxed certs)"""
        config = {"redis": {"host": "dev.redis.com", "ssl": True}}
        connector = RedisConnector(config=config, test_mode=True)
        connector.environment = "development"
        
        mock_pool = MagicMock()
        mock_redis_client = MagicMock()
        
        with patch("redis.ConnectionPool", return_value=mock_pool) as mock_pool_constructor:
            with patch("redis.Redis", return_value=mock_redis_client):
                with patch.object(connector, "log_success"):
                    result = connector.connect()
        
        assert result is True
        # Check SSL cert requirements are relaxed for development
        call_kwargs = mock_pool_constructor.call_args[1]
        assert call_kwargs["ssl_cert_reqs"] == "none"

    def test_connect_failure(self):
        """Test Redis connection failure"""
        config = {"redis": {"host": "unreachable.redis.com"}}
        connector = RedisConnector(config=config, test_mode=True)
        
        with patch("redis.ConnectionPool", side_effect=Exception("Connection failed")):
            with patch.object(connector, "log_error") as mock_log_error:
                result = connector.connect(password="password")
        
        assert result is False
        mock_log_error.assert_called_once()

    def test_connect_no_redis_client(self):
        """Test connection when Redis client creation returns None"""
        config = {"redis": {"host": "localhost"}}
        connector = RedisConnector(config=config, test_mode=True)
        
        with patch("redis.ConnectionPool", return_value=MagicMock()):
            with patch("redis.Redis", return_value=None):
                result = connector.connect()
        
        assert result is False

    def test_connect_custom_pool_settings(self):
        """Test connection with custom pool settings"""
        config = {"redis": {"host": "localhost"}}
        connector = RedisConnector(config=config, test_mode=True)
        connector.perf_config = {"connection_pool": {"max_size": 200}}
        
        mock_pool = MagicMock()
        mock_redis_client = MagicMock()
        
        with patch("redis.ConnectionPool", return_value=mock_pool) as mock_pool_constructor:
            with patch("redis.Redis", return_value=mock_redis_client):
                with patch.object(connector, "log_success"):
                    result = connector.connect()
        
        assert result is True
        call_kwargs = mock_pool_constructor.call_args[1]
        assert call_kwargs["max_connections"] == 200


class TestRedisConnectorPrefixedKey:
    """Test suite for prefixed key functionality"""

    def test_get_prefixed_key_default(self):
        """Test getting prefixed key with default prefix"""
        config = {"redis": {}}
        connector = RedisConnector(config=config, test_mode=True)
        
        result = connector.get_prefixed_key("user_123", "cache")
        assert result == "cache:user_123"

    def test_get_prefixed_key_custom_prefix(self):
        """Test getting prefixed key with custom prefix"""
        config = {
            "redis": {
                "prefixes": {
                    "cache": "app_cache:",
                    "session": "app_session:",
                    "lock": "app_lock:"
                }
            }
        }
        connector = RedisConnector(config=config, test_mode=True)
        
        assert connector.get_prefixed_key("key1", "cache") == "app_cache:key1"
        assert connector.get_prefixed_key("key2", "session") == "app_session:key2"
        assert connector.get_prefixed_key("key3", "lock") == "app_lock:key3"

    def test_get_prefixed_key_missing_prefix_type(self):
        """Test getting prefixed key for undefined prefix type"""
        config = {"redis": {"prefixes": {"cache": "custom_cache:"}}}
        connector = RedisConnector(config=config, test_mode=True)
        
        result = connector.get_prefixed_key("key", "unknown")
        assert result == "unknown:key"  # Falls back to default format


class TestRedisConnectorCache:
    """Test suite for cache operations"""

    def test_set_cache_success(self):
        """Test successful cache setting"""
        config = {"redis": {}}
        connector = RedisConnector(config=config, test_mode=True)
        connector.is_connected = True
        
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_redis_key", return_value=True):
                result = connector.set_cache("test_key", {"data": "value"}, ttl_seconds=3600)
        
        assert result is True
        mock_redis_client.setex.assert_called_once()

    def test_set_cache_not_connected(self):
        """Test cache setting when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.set_cache("test_key", "value")
        
        assert result is False

    def test_set_cache_invalid_key(self):
        """Test cache setting with invalid key"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_redis_key", return_value=False):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.set_cache("invalid key", "value")
        
        assert result is False
        mock_log.assert_called_once_with("Invalid cache key: invalid key")

    def test_set_cache_default_ttl(self):
        """Test cache setting with default TTL from performance config"""
        config = {"redis": {}}
        connector = RedisConnector(config=config, test_mode=True)
        connector.perf_config = {"cache": {"ttl_seconds": 7200}}
        connector.redis_client = MagicMock()
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_redis_key", return_value=True):
                connector.set_cache("test_key", "value")  # No TTL specified
        
        # Should use TTL from performance config
        call_args = connector.redis_client.setex.call_args[0]
        assert call_args[1] == 7200  # TTL from perf config

    def test_set_cache_exception(self):
        """Test cache setting with exception"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_redis_key", return_value=True):
                with patch.object(connector, "get_prefixed_key", side_effect=Exception("Redis error")):
                    with patch.object(connector, "log_error") as mock_log:
                        result = connector.set_cache("test_key", "value")
        
        assert result is False
        mock_log.assert_called_once()

    def test_get_cache_success(self):
        """Test successful cache retrieval"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        # Mock cache entry data
        cache_data = {
            "key": "test_key",
            "value": {"data": "cached_value"},
            "hit_count": 5,
            "created_at": "2023-01-01T12:00:00",
            "ttl_seconds": 3600
        }
        
        mock_redis_client.get.return_value = json.dumps(cache_data)
        mock_redis_client.ttl.return_value = 1800  # 30 minutes remaining
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.get_cache("test_key")
        
        assert result == {"data": "cached_value"}
        # Verify hit count was incremented and cache was updated
        assert mock_redis_client.setex.call_count == 1

    def test_get_cache_not_found(self):
        """Test cache retrieval when key doesn't exist"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.get.return_value = None
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.get_cache("nonexistent_key")
        
        assert result is None

    def test_get_cache_not_connected(self):
        """Test cache retrieval when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.get_cache("test_key")
        
        assert result is None

    def test_get_cache_no_redis_client(self):
        """Test cache retrieval when redis client is None"""
        connector = RedisConnector(config={}, test_mode=True)
        connector.redis_client = None
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.get_cache("test_key")
        
        assert result is None

    def test_get_cache_exception(self):
        """Test cache retrieval with exception"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch.object(connector, "get_prefixed_key", side_effect=Exception("Redis error")):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.get_cache("test_key")
        
        assert result is None
        mock_log.assert_called_once()

    def test_delete_cache_success(self):
        """Test successful cache deletion"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        # Mock finding 3 matching keys
        mock_redis_client.scan_iter.return_value = ["cache:key1", "cache:key2", "cache:key3"]
        mock_redis_client.delete.return_value = 3
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.delete_cache("test_pattern*")
        
        assert result == 3
        mock_redis_client.delete.assert_called_once_with("cache:key1", "cache:key2", "cache:key3")

    def test_delete_cache_no_matches(self):
        """Test cache deletion when no keys match"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.scan_iter.return_value = []
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.delete_cache("nonexistent_pattern*")
        
        assert result == 0
        mock_redis_client.delete.assert_not_called()

    def test_delete_cache_not_connected(self):
        """Test cache deletion when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.delete_cache("pattern*")
        
        assert result == 0


class TestRedisConnectorSession:
    """Test suite for session operations"""

    def test_set_session_success(self):
        """Test successful session setting"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        session_data = {"user_id": "123", "username": "testuser"}
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.set_session("session_abc", session_data, ttl_seconds=7200)
        
        assert result is True
        mock_redis_client.setex.assert_called_once()
        # Verify the session data structure
        call_args = mock_redis_client.setex.call_args[0]
        stored_data = json.loads(call_args[2])
        assert stored_data["id"] == "session_abc"
        assert stored_data["data"] == session_data

    def test_set_session_not_connected(self):
        """Test session setting when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.set_session("session_123", {"user": "test"})
        
        assert result is False

    def test_set_session_exception(self):
        """Test session setting with exception"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch.object(connector, "get_prefixed_key", side_effect=Exception("Redis error")):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.set_session("session_123", {"user": "test"})
        
        assert result is False
        mock_log.assert_called_once()

    def test_get_session_success(self):
        """Test successful session retrieval"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        session_data = {
            "id": "session_123",
            "data": {"user_id": "456", "role": "admin"},
            "created_at": "2023-01-01T12:00:00",
            "last_activity": "2023-01-01T13:00:00"
        }
        
        mock_redis_client.get.return_value = json.dumps(session_data)
        mock_redis_client.ttl.return_value = 3600
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.get_session("session_123")
        
        assert result == {"user_id": "456", "role": "admin"}
        # Verify session was updated with new last_activity
        mock_redis_client.setex.assert_called_once()

    def test_get_session_not_found(self):
        """Test session retrieval when session doesn't exist"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.get.return_value = None
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.get_session("nonexistent_session")
        
        assert result is None

    def test_get_session_invalid_data_format(self):
        """Test session retrieval with invalid data format"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        session_data = {
            "id": "session_123",
            "data": "invalid_data_format",  # Should be dict
            "created_at": "2023-01-01T12:00:00"
        }
        
        mock_redis_client.get.return_value = json.dumps(session_data)
        mock_redis_client.ttl.return_value = 3600
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.get_session("session_123")
        
        assert result is None  # Invalid format returns None


class TestRedisConnectorLock:
    """Test suite for distributed lock operations"""

    def test_acquire_lock_success(self):
        """Test successful lock acquisition"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.set.return_value = True  # Lock acquired
        
        with patch.object(connector, "ensure_connected", return_value=True):
            lock_id = connector.acquire_lock("resource_123", timeout=60)
        
        assert lock_id is not None
        assert len(lock_id) == 16  # Should be 16 character hash
        mock_redis_client.set.assert_called_once()
        call_args = mock_redis_client.set.call_args
        assert call_args[1]["nx"] is True  # Only set if not exists
        assert call_args[1]["ex"] == 60   # Expiration timeout

    def test_acquire_lock_already_held(self):
        """Test lock acquisition when lock is already held"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.set.return_value = None  # Lock not acquired
        
        with patch.object(connector, "ensure_connected", return_value=True):
            lock_id = connector.acquire_lock("resource_123")
        
        assert lock_id is None

    def test_acquire_lock_not_connected(self):
        """Test lock acquisition when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=False):
            lock_id = connector.acquire_lock("resource_123")
        
        assert lock_id is None

    def test_acquire_lock_no_redis_client(self):
        """Test lock acquisition when redis client is None"""
        connector = RedisConnector(config={}, test_mode=True)
        connector.redis_client = None
        
        with patch.object(connector, "ensure_connected", return_value=True):
            lock_id = connector.acquire_lock("resource_123")
        
        assert lock_id is None

    def test_acquire_lock_exception(self):
        """Test lock acquisition with exception"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch.object(connector, "get_prefixed_key", side_effect=Exception("Redis error")):
                with patch.object(connector, "log_error") as mock_log:
                    lock_id = connector.acquire_lock("resource_123")
        
        assert lock_id is None
        mock_log.assert_called_once()

    def test_release_lock_success(self):
        """Test successful lock release"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.eval.return_value = 1  # Lock released
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.release_lock("resource_123", "lock_id_123")
        
        assert result is True
        mock_redis_client.eval.assert_called_once()

    def test_release_lock_not_owner(self):
        """Test lock release when not the owner"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.eval.return_value = 0  # Not released (not owner)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.release_lock("resource_123", "wrong_lock_id")
        
        assert result is False

    def test_release_lock_not_connected(self):
        """Test lock release when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.release_lock("resource_123", "lock_id")
        
        assert result is False

    def test_release_lock_lua_script_content(self):
        """Test that the Lua script is properly formed"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        with patch.object(connector, "ensure_connected", return_value=True):
            connector.release_lock("resource", "lock_id")
        
        # Verify Lua script was called with correct parameters
        call_args = mock_redis_client.eval.call_args[0]
        lua_script = call_args[0]
        assert "redis.call('get', KEYS[1])" in lua_script
        assert "redis.call('del', KEYS[1])" in lua_script


class TestRedisConnectorMetrics:
    """Test suite for metrics operations"""

    def test_record_metric_success(self):
        """Test successful metric recording"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        metric = MetricEvent(
            timestamp=datetime(2023, 1, 1, 12, 30),
            metric_name="api.requests",
            value=1.0,
            tags={"endpoint": "/api/users"},
            document_id="doc_123"
        )
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_metric_event", return_value=True):
                with patch("validators.kv_validators.sanitize_metric_name", return_value="api_requests"):
                    result = connector.record_metric(metric)
        
        assert result is True
        mock_redis_client.zadd.assert_called_once()
        mock_redis_client.expire.assert_called_once_with(mock_redis_client.zadd.call_args[0][0], 7 * 24 * 3600)

    def test_record_metric_not_connected(self):
        """Test metric recording when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        metric = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="test.metric",
            value=1.0,
            tags={}
        )
        
        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.record_metric(metric)
        
        assert result is False

    def test_record_metric_invalid_metric(self):
        """Test metric recording with invalid metric"""
        connector = RedisConnector(config={}, test_mode=True)
        
        metric = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="invalid.metric",
            value=1.0,
            tags={}
        )
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_metric_event", return_value=False):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.record_metric(metric)
        
        assert result is False
        mock_log.assert_called_once()

    def test_record_metric_exception(self):
        """Test metric recording with exception"""
        connector = RedisConnector(config={}, test_mode=True)
        
        metric = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="test.metric",
            value=1.0,
            tags={}
        )
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_metric_event", return_value=True):
                with patch.object(connector, "get_prefixed_key", side_effect=Exception("Redis error")):
                    with patch.object(connector, "log_error") as mock_log:
                        result = connector.record_metric(metric)
        
        assert result is False
        mock_log.assert_called_once()

    def test_get_metrics_success(self):
        """Test successful metrics retrieval"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        
        # Mock metric data
        metric_data = {
            "timestamp": "2023-01-01T12:00:00",
            "metric_name": "api_requests",
            "value": 1.0,
            "tags": {"endpoint": "/api/users"}
        }
        mock_redis_client.zrangebyscore.return_value = [json.dumps(metric_data)]
        
        start_time = datetime(2023, 1, 1, 10, 0)
        end_time = datetime(2023, 1, 1, 14, 0)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_time_range", return_value=True):
                with patch("validators.kv_validators.sanitize_metric_name", return_value="api_requests"):
                    result = connector.get_metrics("api.requests", start_time, end_time)
        
        assert len(result) > 0
        assert isinstance(result[0], MetricEvent)
        assert result[0].metric_name == "api_requests"

    def test_get_metrics_invalid_time_range(self):
        """Test metrics retrieval with invalid time range"""
        connector = RedisConnector(config={}, test_mode=True)
        
        start_time = datetime(2023, 1, 2)  # After end time
        end_time = datetime(2023, 1, 1)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_time_range", return_value=False):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.get_metrics("test.metric", start_time, end_time)
        
        assert result == []
        mock_log.assert_called_once()

    def test_get_metrics_not_connected(self):
        """Test metrics retrieval when not connected"""
        connector = RedisConnector(config={}, test_mode=True)
        
        start_time = datetime(2023, 1, 1, 10, 0)
        end_time = datetime(2023, 1, 1, 14, 0)
        
        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.get_metrics("test.metric", start_time, end_time)
        
        assert result == []

    def test_get_metrics_time_granularity(self):
        """Test metrics retrieval checks both hour and minute granularity"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        mock_redis_client.zrangebyscore.return_value = []
        
        start_time = datetime(2023, 1, 1, 10, 0)
        end_time = datetime(2023, 1, 1, 12, 0)
        
        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("validators.kv_validators.validate_time_range", return_value=True):
                with patch("validators.kv_validators.sanitize_metric_name", return_value="test_metric"):
                    result = connector.get_metrics("test.metric", start_time, end_time)
        
        # Should make multiple calls for different time formats and hours
        assert mock_redis_client.zrangebyscore.call_count > 1


class TestRedisConnectorClose:
    """Test suite for connection closure"""

    def test_close_with_client(self):
        """Test closing connection when client exists"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        connector.redis_client = mock_redis_client
        connector.is_connected = True
        
        connector.close()
        
        mock_redis_client.close.assert_called_once()
        assert connector.redis_client is None
        assert connector.is_connected is False

    def test_close_with_exception(self):
        """Test closing connection with exception"""
        connector = RedisConnector(config={}, test_mode=True)
        mock_redis_client = MagicMock()
        mock_redis_client.close.side_effect = Exception("Close error")
        connector.redis_client = mock_redis_client
        
        with patch.object(connector, "log_error") as mock_log:
            connector.close()
        
        mock_log.assert_called_once()
        assert connector.redis_client is None
        assert connector.is_connected is False

    def test_close_no_client(self):
        """Test closing when no client exists"""
        connector = RedisConnector(config={}, test_mode=True)
        connector.redis_client = None
        
        # Should not raise exception
        connector.close()


class TestDuckDBAnalyticsInit:
    """Test suite for DuckDBAnalytics initialization"""

    def test_init_basic(self):
        """Test basic DuckDB initialization"""
        with patch("builtins.open", mock_open(read_data='{}')):
            with patch("yaml.safe_load", return_value={}):
                analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        
        assert analytics.conn is None
        assert analytics.perf_config == {}

    def test_get_service_name(self):
        """Test service name getter"""
        with patch("builtins.open", mock_open(read_data='{}')):
            with patch("yaml.safe_load", return_value={}):
                analytics = DuckDBAnalytics()
        
        assert analytics._get_service_name() == "duckdb"

    def test_load_performance_config_success(self):
        """Test successful performance config loading for DuckDB"""
        perf_config = {
            "kv_store": {
                "duckdb": {
                    "query": {"timeout_seconds": 120},
                    "memory": {"limit": "4GB"}
                }
            }
        }
        
        with patch("builtins.open", mock_open(read_data="dummy")):
            with patch("yaml.safe_load", return_value=perf_config):
                analytics = DuckDBAnalytics()
                result = analytics._load_performance_config()
        
        expected = {"query": {"timeout_seconds": 120}, "memory": {"limit": "4GB"}}
        assert result == expected


class TestDuckDBAnalyticsConnection:
    """Test suite for DuckDB connection functionality"""

    def test_connect_success(self):
        """Test successful DuckDB connection"""
        config = {
            "duckdb": {
                "database_path": "/tmp/test.db",
                "memory_limit": "1GB",
                "threads": 2
            }
        }
        
        mock_conn = MagicMock()
        
        with patch("pathlib.Path.mkdir"):
            with patch("duckdb.connect", return_value=mock_conn):
                with patch.object(base_component_mock.DatabaseComponent, "__init__"):
                    analytics = DuckDBAnalytics()
                    analytics.config = config
                    with patch.object(analytics, "_initialize_tables"):
                        with patch.object(analytics, "log_success"):
                            result = analytics.connect()
        
        assert result is True
        assert analytics.conn == mock_conn
        assert analytics.is_connected is True
        
        # Verify configuration was set
        expected_calls = [
            call("SET memory_limit = '1GB'"),
            call("SET threads = 2")
        ]
        mock_conn.execute.assert_has_calls(expected_calls, any_order=True)

    def test_connect_directory_creation_error(self):
        """Test connection when directory creation fails"""
        config = {"duckdb": {"database_path": "/invalid/path/test.db"}}
        
        with patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied")):
            with patch.object(base_component_mock.DatabaseComponent, "__init__"):
                analytics = DuckDBAnalytics()
                analytics.config = config
                with patch.object(analytics, "log_error") as mock_log:
                    result = analytics.connect()
        
        assert result is False
        mock_log.assert_called_once()

    def test_connect_duckdb_error(self):
        """Test connection when DuckDB connection fails"""
        config = {"duckdb": {"database_path": "/tmp/test.db"}}
        
        with patch("pathlib.Path.mkdir"):
            with patch("duckdb.connect", side_effect=Exception("DuckDB error")):
                with patch.object(base_component_mock.DatabaseComponent, "__init__"):
                    analytics = DuckDBAnalytics()
                    analytics.config = config
                    with patch.object(analytics, "log_error") as mock_log:
                        result = analytics.connect()
        
        assert result is False
        mock_log.assert_called_once()

    def test_initialize_tables(self):
        """Test table initialization"""
        config = {
            "duckdb": {
                "tables": {
                    "metrics": "custom_metrics",
                    "events": "custom_events",
                    "summaries": "custom_summaries",
                    "trends": "custom_trends"
                }
            }
        }
        
        mock_conn = MagicMock()
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = config
            analytics.conn = mock_conn
            
            analytics._initialize_tables()
        
        # Should create multiple tables and indexes
        assert mock_conn.execute.call_count >= 7  # 4 tables + 3 indexes

    def test_initialize_tables_no_connection(self):
        """Test table initialization when not connected"""
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {"duckdb": {}}
            analytics.conn = None
            
            result = analytics._initialize_tables()
        
        assert result is False


class TestDuckDBAnalyticsQueries:
    """Test suite for DuckDB query operations"""

    def test_insert_metrics_success(self):
        """Test successful metrics insertion"""
        metrics = [
            MetricEvent(
                timestamp=datetime(2023, 1, 1, 12, 0),
                metric_name="cpu.usage",
                value=0.75,
                tags={"host": "server1"},
                document_id="doc_1"
            ),
            MetricEvent(
                timestamp=datetime(2023, 1, 1, 12, 1),
                metric_name="memory.usage",
                value=0.60,
                tags={"host": "server1"}
            )
        ]
        
        mock_conn = MagicMock()
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}
            analytics.conn = mock_conn
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.insert_metrics(metrics)
        
        assert result is True
        mock_conn.executemany.assert_called_once()

    def test_insert_metrics_not_connected(self):
        """Test metrics insertion when not connected"""
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            
            with patch.object(analytics, "ensure_connected", return_value=False):
                result = analytics.insert_metrics([])
        
        assert result is False

    def test_query_metrics_success(self):
        """Test successful metrics query"""
        mock_conn = MagicMock()
        mock_conn.description = [("metric_name",), ("value",), ("timestamp",)]
        mock_conn.execute.return_value.fetchall.return_value = [
            ("cpu.usage", 0.75, "2023-01-01 12:00:00"),
            ("memory.usage", 0.60, "2023-01-01 12:01:00")
        ]
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {}
            analytics.conn = mock_conn
            analytics.perf_config = {"query": {"timeout_seconds": 30}}
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.query_metrics("SELECT * FROM metrics")
        
        assert len(result) == 2
        assert result[0]["metric_name"] == "cpu.usage"
        assert result[0]["value"] == 0.75

    def test_query_metrics_with_parameters(self):
        """Test metrics query with parameters"""
        mock_conn = MagicMock()
        mock_conn.description = [("count",)]
        mock_conn.execute.return_value.fetchall.return_value = [(5,)]
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {}
            analytics.conn = mock_conn
            analytics.perf_config = {}
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.query_metrics(
                    "SELECT COUNT(*) as count FROM metrics WHERE metric_name = ?",
                    ["cpu.usage"]
                )
        
        assert len(result) == 1
        assert result[0]["count"] == 5

    def test_query_metrics_not_connected(self):
        """Test metrics query when not connected"""
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            
            with patch.object(analytics, "ensure_connected", return_value=False):
                result = analytics.query_metrics("SELECT * FROM metrics")
        
        assert result == []

    def test_aggregate_metrics_success(self):
        """Test successful metrics aggregation"""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (
            0.75,  # avg result
            100,   # count
            "2023-01-01 10:00:00",  # start_time
            "2023-01-01 14:00:00"   # end_time
        )
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}
            analytics.conn = mock_conn
            
            start_time = datetime(2023, 1, 1, 10, 0)
            end_time = datetime(2023, 1, 1, 14, 0)
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.aggregate_metrics("cpu.usage", start_time, end_time, "avg")
        
        assert result["aggregation"] == "avg"
        assert result["value"] == 0.75
        assert result["count"] == 100

    def test_aggregate_metrics_different_aggregations(self):
        """Test different aggregation types"""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = (1.0, 10, None, None)
        
        aggregations = ["avg", "sum", "min", "max", "count", "stddev"]
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}
            analytics.conn = mock_conn
            
            start_time = datetime(2023, 1, 1, 10, 0)
            end_time = datetime(2023, 1, 1, 14, 0)
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                for agg in aggregations:
                    result = analytics.aggregate_metrics("test.metric", start_time, end_time, agg)
                    assert result["aggregation"] == agg

    def test_generate_summary_daily(self):
        """Test daily summary generation"""
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            # Query results
            MagicMock(fetchall=MagicMock(return_value=[
                ("cpu.usage", 100, 0.75, 0.50, 0.90, 0.12),
                ("memory.usage", 80, 0.60, 0.40, 0.80, 0.10)
            ])),
            # Insert result
            MagicMock()
        ]
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {
                "duckdb": {
                    "tables": {
                        "metrics": "test_metrics",
                        "summaries": "test_summaries"
                    }
                }
            }
            analytics.conn = mock_conn
            
            summary_date = date(2023, 1, 1)
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.generate_summary(summary_date, "daily")
        
        assert result["summary_type"] == "daily"
        assert result["summary_date"] == "2023-01-01"
        assert "cpu.usage" in result["metrics"]
        assert "memory.usage" in result["metrics"]
        assert result["metrics"]["cpu.usage"]["count"] == 100
        assert result["metrics"]["cpu.usage"]["avg"] == 0.75

    def test_detect_trends_success(self):
        """Test successful trend detection"""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2023-01-01 10:00:00", 0.60, 10),
            ("2023-01-01 11:00:00", 0.65, 12),
            ("2023-01-01 12:00:00", 0.70, 15),
            ("2023-01-01 13:00:00", 0.75, 18),
            ("2023-01-01 14:00:00", 0.80, 20),
            ("2023-01-01 15:00:00", 0.85, 22)
        ]
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}
            analytics.conn = mock_conn
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.detect_trends("cpu.usage", period_days=1)
        
        assert result["metric_name"] == "cpu.usage"
        assert result["trend_direction"] == "increasing"
        assert result["data_points"] == 6
        assert "trend_strength" in result
        assert "confidence" in result

    def test_detect_trends_insufficient_data(self):
        """Test trend detection with insufficient data"""
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("2023-01-01 10:00:00", 0.60, 10)
        ]
        
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            analytics.config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}
            analytics.conn = mock_conn
            
            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.detect_trends("cpu.usage", period_days=1)
        
        assert result["trend"] == "insufficient_data"


class TestContextKV:
    """Test suite for ContextKV unified interface"""

    def test_init_test_mode(self):
        """Test ContextKV initialization in test mode"""
        config = {"redis": {"host": "localhost"}}
        
        kv = ContextKV(config=config, test_mode=True)
        
        assert kv.config == config
        assert kv.test_mode is True
        assert kv.redis is not None
        assert kv.duckdb is None  # Should be None in test mode

    def test_init_production_mode(self):
        """Test ContextKV initialization in production mode"""
        config = {"redis": {"host": "localhost"}}
        
        with patch.object(DuckDBAnalytics, "__init__", return_value=None):
            kv = ContextKV(config=config, test_mode=False)
        
        assert kv.test_mode is False
        assert kv.redis is not None
        # DuckDB creation is attempted but may be None due to exceptions

    def test_connect_success(self):
        """Test successful connection to both stores"""
        mock_redis = MagicMock()
        mock_redis.connect.return_value = True
        mock_duckdb = MagicMock()
        mock_duckdb.connect.return_value = True
        
        kv = ContextKV(test_mode=True)
        kv.redis = mock_redis
        kv.duckdb = mock_duckdb
        
        result = kv.connect(redis_password="test_pass")
        
        assert result is True
        mock_redis.connect.assert_called_once_with(password="test_pass")
        mock_duckdb.connect.assert_called_once()

    def test_connect_redis_failure(self):
        """Test connection when Redis fails"""
        mock_redis = MagicMock()
        mock_redis.connect.return_value = False
        mock_duckdb = MagicMock()
        mock_duckdb.connect.return_value = True
        
        kv = ContextKV(test_mode=True)
        kv.redis = mock_redis
        kv.duckdb = mock_duckdb
        
        result = kv.connect()
        
        assert result is False

    def test_record_event_success(self):
        """Test successful event recording"""
        mock_redis = MagicMock()
        mock_redis.record_metric.return_value = True
        mock_duckdb = MagicMock()
        mock_duckdb.insert_metrics.return_value = True
        
        kv = ContextKV(test_mode=True)
        kv.redis = mock_redis
        kv.duckdb = mock_duckdb
        
        result = kv.record_event(
            event_type="user_login",
            document_id="doc_123",
            agent_id="agent_456",
            data={"ip": "192.168.1.1"}
        )
        
        assert result is True
        mock_redis.record_metric.assert_called_once()
        mock_duckdb.insert_metrics.assert_called_once()

    def test_record_event_redis_failure(self):
        """Test event recording when Redis fails"""
        mock_redis = MagicMock()
        mock_redis.record_metric.return_value = False
        mock_duckdb = MagicMock()
        mock_duckdb.insert_metrics.return_value = True
        
        kv = ContextKV(test_mode=True)
        kv.redis = mock_redis
        kv.duckdb = mock_duckdb
        
        result = kv.record_event("test_event")
        
        assert result is False

    def test_get_recent_activity(self):
        """Test getting recent activity summary"""
        mock_duckdb = MagicMock()
        mock_duckdb.query_metrics.return_value = [
            {"metric_name": "event.user_login", "count": 50, "avg_value": 1.0},
            {"metric_name": "event.page_view", "count": 200, "avg_value": 1.0}
        ]
        
        kv = ContextKV(test_mode=True)
        kv.duckdb = mock_duckdb
        
        result = kv.get_recent_activity(hours=12)
        
        assert result["period_hours"] == 12
        assert len(result["metrics"]) == 2
        assert result["metrics"][0]["metric_name"] == "event.user_login"

    def test_close(self):
        """Test closing all connections"""
        mock_redis = MagicMock()
        mock_duckdb = MagicMock()
        
        kv = ContextKV(test_mode=True)
        kv.redis = mock_redis
        kv.duckdb = mock_duckdb
        
        kv.close()
        
        mock_redis.close.assert_called_once()
        mock_duckdb.close.assert_called_once()


class TestKVStoreIntegration:
    """Integration tests for KV store functionality"""

    def test_full_redis_workflow(self):
        """Test complete Redis workflow"""
        config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "prefixes": {"cache": "test_cache:", "session": "test_session:"}
            }
        }
        
        # Mock Redis client chain
        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.setex.return_value = True
        mock_redis_client.get.return_value = json.dumps({
            "key": "test_key",
            "value": "test_value",
            "hit_count": 0,
            "created_at": "2023-01-01T12:00:00"
        })
        mock_redis_client.ttl.return_value = 1800
        
        connector = RedisConnector(config=config, test_mode=True)
        
        with patch("redis.ConnectionPool"):
            with patch("redis.Redis", return_value=mock_redis_client):
                with patch.object(connector, "log_success"):
                    # Connect
                    connected = connector.connect(password="test_password")
                    assert connected is True
                    
                    # Set cache
                    with patch("validators.kv_validators.validate_redis_key", return_value=True):
                        cache_set = connector.set_cache("test_key", "test_value", ttl_seconds=3600)
                        assert cache_set is True
                    
                    # Get cache
                    cached_value = connector.get_cache("test_key")
                    assert cached_value == "test_value"
                    
                    # Close
                    connector.close()

    def test_full_context_kv_workflow(self):
        """Test complete ContextKV workflow"""
        mock_redis = MagicMock()
        mock_redis.connect.return_value = True
        mock_redis.record_metric.return_value = True
        
        mock_duckdb = MagicMock()
        mock_duckdb.connect.return_value = True
        mock_duckdb.insert_metrics.return_value = True
        mock_duckdb.query_metrics.return_value = [
            {"metric_name": "event.test", "count": 1, "avg_value": 1.0}
        ]
        
        kv = ContextKV(test_mode=True)
        kv.redis = mock_redis
        kv.duckdb = mock_duckdb
        
        # Connect
        connected = kv.connect()
        assert connected is True
        
        # Record event
        event_recorded = kv.record_event("test_event", data={"source": "test"})
        assert event_recorded is True
        
        # Get activity
        activity = kv.get_recent_activity(hours=24)
        assert activity["period_hours"] == 24
        assert len(activity["metrics"]) == 1
        
        # Close
        kv.close()

    def test_error_handling_robustness(self):
        """Test error handling across the module"""
        # Test Redis connector with various error conditions
        connector = RedisConnector(config={}, test_mode=True)
        
        # Connection errors
        with patch.object(connector, "ensure_connected", return_value=False):
            assert connector.set_cache("key", "value") is False
            assert connector.get_cache("key") is None
            assert connector.delete_cache("pattern") == 0
            assert connector.acquire_lock("resource") is None
            assert connector.release_lock("resource", "lock_id") is False
        
        # Analytics errors
        with patch.object(base_component_mock.DatabaseComponent, "__init__"):
            analytics = DuckDBAnalytics()
            
            with patch.object(analytics, "ensure_connected", return_value=False):
                assert analytics.insert_metrics([]) is False
                assert analytics.query_metrics("SELECT 1") == []
                assert analytics.aggregate_metrics("test", datetime.now(), datetime.now()) == {}

    def test_configuration_edge_cases(self):
        """Test various configuration edge cases"""
        # Empty configurations
        connector1 = RedisConnector(config={}, test_mode=True)
        assert connector1.get_prefixed_key("key", "cache") == "cache:key"
        
        # Partial configurations
        connector2 = RedisConnector(config={"redis": {"host": "custom"}}, test_mode=True)
        assert "custom" in str(connector2.config)
        
        # Invalid performance config structures
        with patch("builtins.open", mock_open(read_data="invalid")):
            with patch("yaml.safe_load", return_value="not_a_dict"):
                connector3 = RedisConnector(config={}, test_mode=True)
                assert connector3.perf_config == {}