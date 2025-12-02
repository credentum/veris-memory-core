#!/usr/bin/env python3
"""
Comprehensive tests for KV store components to achieve high coverage.

This test suite covers:
- RedisConnector connection management and operations
- DuckDBAnalytics database operations and queries
- ContextKV unified interface
- Cache operations, session management, distributed locking
- Metric recording and analytics
- Error handling and edge cases
"""

import json
from datetime import date, datetime
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from src.storage.kv_store import CacheEntry, ContextKV, DuckDBAnalytics, MetricEvent, RedisConnector


class TestMetricEvent:
    """Test MetricEvent dataclass."""

    def test_metric_event_creation(self):
        """Test MetricEvent creation with all fields."""
        timestamp = datetime.utcnow()
        event = MetricEvent(
            timestamp=timestamp,
            metric_name="test_metric",
            value=42.5,
            tags={"env": "test"},
            document_id="doc-123",
            agent_id="agent-456",
        )

        assert event.timestamp == timestamp
        assert event.metric_name == "test_metric"
        assert event.value == 42.5
        assert event.tags == {"env": "test"}
        assert event.document_id == "doc-123"
        assert event.agent_id == "agent-456"

    def test_metric_event_minimal(self):
        """Test MetricEvent creation with minimal fields."""
        timestamp = datetime.utcnow()
        event = MetricEvent(timestamp=timestamp, metric_name="minimal_metric", value=1.0, tags={})

        assert event.document_id is None
        assert event.agent_id is None


class TestCacheEntry:
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
        entry = CacheEntry(
            key="test_key", value="test_value", created_at=datetime.utcnow(), ttl_seconds=300
        )

        assert entry.hit_count == 0
        assert entry.last_accessed is None


class TestRedisConnector:
    """Test RedisConnector class."""

    def test_init_with_config_dict(self):
        """Test initialization with provided config dictionary."""
        config = {"redis": {"host": "test-host", "port": 6379, "database": 0}}

        connector = RedisConnector(config=config, test_mode=True)

        assert connector.config == config
        assert connector.test_mode is True
        assert connector.redis_client is None

    def test_init_with_parent_class(self):
        """Test initialization using parent class when no config provided."""
        with patch("storage.kv_store.DatabaseComponent.__init__") as mock_parent_init:
            with patch.object(RedisConnector, "_load_performance_config", return_value={}):
                connector = RedisConnector(config_path=".ctxrc.yaml", verbose=True)

                mock_parent_init.assert_called_once_with(".ctxrc.yaml", True)
                assert connector.redis_client is None

    def test_get_service_name(self):
        """Test service name for configuration."""
        connector = RedisConnector(config={}, test_mode=True)
        assert connector._get_service_name() == "redis"

    def test_load_performance_config_success(self):
        """Test loading performance configuration successfully."""
        perf_config = {"kv_store": {"redis": {"connection_pool": {"max_size": 100}}}}

        with patch("builtins.open", mock_open(read_data=yaml.dump(perf_config))):
            connector = RedisConnector(config={}, test_mode=True)
            result = connector._load_performance_config()

            assert result == {"connection_pool": {"max_size": 100}}

    def test_load_performance_config_file_not_found(self):
        """Test loading performance config when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            connector = RedisConnector(config={}, test_mode=True)
            result = connector._load_performance_config()

            assert result == {}

    def test_load_performance_config_invalid_structure(self):
        """Test loading performance config with invalid structure."""
        perf_config = {"kv_store": "not a dict"}

        with patch("builtins.open", mock_open(read_data=yaml.dump(perf_config))):
            connector = RedisConnector(config={}, test_mode=True)
            result = connector._load_performance_config()

            assert result == {}

    @patch("storage.kv_store.redis.Redis")
    @patch("storage.kv_store.redis.ConnectionPool")
    def test_connect_success_without_ssl(self, mock_pool_class, mock_redis_class):
        """Test successful Redis connection without SSL."""
        config = {"redis": {"host": "localhost", "port": 6379, "database": 0, "ssl": False}}

        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        connector = RedisConnector(config=config, test_mode=True)

        with patch.object(connector, "log_success") as mock_log:
            result = connector.connect()

            assert result is True
            assert connector.redis_client == mock_redis
            assert connector.is_connected is True
            mock_log.assert_called_with("Connected to Redis at localhost:6379")

    @patch("storage.kv_store.redis.Redis")
    @patch("storage.kv_store.redis.ConnectionPool")
    def test_connect_success_with_ssl(self, mock_pool_class, mock_redis_class):
        """Test successful Redis connection with SSL."""
        config = {"redis": {"host": "secure-host", "port": 6379, "database": 0, "ssl": True}}

        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        connector = RedisConnector(config=config, test_mode=True)
        connector.environment = "production"

        result = connector.connect(password="test_password")

        assert result is True
        # Should have SSL configuration
        pool_call = mock_pool_class.call_args
        assert pool_call[1]["ssl"] is True
        assert pool_call[1]["ssl_cert_reqs"] == "required"

    @patch("storage.kv_store.redis.Redis")
    @patch("storage.kv_store.redis.ConnectionPool")
    def test_connect_ssl_development_mode(self, mock_pool_class, mock_redis_class):
        """Test SSL connection in development mode."""
        config = {"redis": {"host": "localhost", "ssl": True}}

        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis

        connector = RedisConnector(config=config, test_mode=True)
        connector.environment = "development"

        result = connector.connect()

        assert result is True
        # Should use relaxed SSL for development
        pool_call = mock_pool_class.call_args
        assert pool_call[1]["ssl_cert_reqs"] == "none"

    @patch("storage.kv_store.redis.Redis")
    @patch("storage.kv_store.redis.ConnectionPool")
    def test_connect_failure(self, mock_pool_class, mock_redis_class):
        """Test Redis connection failure."""
        config = {"redis": {"host": "localhost"}}

        mock_pool_class.side_effect = Exception("Connection failed")

        connector = RedisConnector(config=config, test_mode=True)

        with patch.object(connector, "log_error") as mock_log:
            result = connector.connect(password="test_password")

            assert result is False
            assert connector.redis_client is None
            mock_log.assert_called_once()

    def test_get_prefixed_key(self):
        """Test key prefixing functionality."""
        config = {"redis": {"prefixes": {"cache": "app:cache:", "session": "app:session:"}}}

        connector = RedisConnector(config=config, test_mode=True)

        assert connector.get_prefixed_key("test_key", "cache") == "app:cache:test_key"
        assert connector.get_prefixed_key("test_key", "session") == "app:session:test_key"
        assert connector.get_prefixed_key("test_key", "unknown") == "unknown:test_key"

    def test_set_cache_not_connected(self):
        """Test setting cache when not connected."""
        connector = RedisConnector(config={}, test_mode=True)

        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.set_cache("test_key", "test_value")

            assert result is False

    def test_set_cache_invalid_key(self):
        """Test setting cache with invalid key."""
        connector = RedisConnector(config={}, test_mode=True)

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("storage.kv_store.validate_redis_key", return_value=False):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.set_cache("invalid:key", "value")

                    assert result is False
                    mock_log.assert_called_with("Invalid cache key: invalid:key")

    def test_set_cache_success(self):
        """Test successful cache setting."""
        config = {"redis": {"prefixes": {"cache": "test:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        mock_redis = Mock()
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("storage.kv_store.validate_redis_key", return_value=True):
                with patch("datetime.datetime") as mock_datetime:
                    mock_now = datetime(2025, 1, 1, 12, 0, 0)
                    mock_datetime.utcnow.return_value = mock_now

                    result = connector.set_cache("test_key", {"data": "test"}, ttl_seconds=300)

                    assert result is True
                    mock_redis.setex.assert_called_once()

    def test_get_cache_not_connected(self):
        """Test getting cache when not connected."""
        connector = RedisConnector(config={}, test_mode=True)

        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.get_cache("test_key")

            assert result is None

    def test_get_cache_not_found(self):
        """Test getting cache when key doesn't exist."""
        connector = RedisConnector(config={}, test_mode=True)

        mock_redis = Mock()
        mock_redis.get.return_value = None
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.get_cache("nonexistent_key")

            assert result is None

    def test_get_cache_success(self):
        """Test successful cache retrieval."""
        config = {"redis": {"prefixes": {"cache": "test:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        cache_data = {"key": "test_key", "value": {"data": "test"}, "hit_count": 5}

        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 300
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("datetime.datetime") as mock_datetime:
                mock_now = datetime(2025, 1, 1, 12, 0, 0)
                mock_datetime.utcnow.return_value = mock_now

                result = connector.get_cache("test_key")

                assert result == {"data": "test"}
                # Should update hit count and last accessed
                mock_redis.setex.assert_called_once()

    def test_delete_cache_not_connected(self):
        """Test deleting cache when not connected."""
        connector = RedisConnector(config={}, test_mode=True)

        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.delete_cache("test_*")

            assert result == 0

    def test_delete_cache_success(self):
        """Test successful cache deletion."""
        config = {"redis": {"prefixes": {"cache": "test:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        mock_redis = Mock()
        mock_redis.scan_iter.return_value = ["test:key1", "test:key2"]
        mock_redis.delete.return_value = 2
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.delete_cache("key*")

            assert result == 2
            mock_redis.delete.assert_called_with("test:key1", "test:key2")

    def test_set_session_success(self):
        """Test successful session setting."""
        config = {"redis": {"prefixes": {"session": "sess:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        mock_redis = Mock()
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("datetime.datetime") as mock_datetime:
                mock_now = datetime(2025, 1, 1, 12, 0, 0)
                mock_datetime.utcnow.return_value = mock_now

                result = connector.set_session(
                    "session123", {"user_id": "user456"}, ttl_seconds=1800
                )

                assert result is True
                mock_redis.setex.assert_called_once()

    def test_get_session_success(self):
        """Test successful session retrieval."""
        config = {"redis": {"prefixes": {"session": "sess:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        session_data = {
            "id": "session123",
            "data": {"user_id": "user456"},
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
        }

        mock_redis = Mock()
        mock_redis.get.return_value = json.dumps(session_data)
        mock_redis.ttl.return_value = 1800
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("datetime.datetime") as mock_datetime:
                mock_now = datetime(2025, 1, 1, 12, 0, 0)
                mock_datetime.utcnow.return_value = mock_now

                result = connector.get_session("session123")

                assert result == {"user_id": "user456"}
                # Should extend TTL
                mock_redis.setex.assert_called_once()

    def test_acquire_lock_success(self):
        """Test successful lock acquisition."""
        config = {"redis": {"prefixes": {"lock": "lock:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        mock_redis = Mock()
        mock_redis.set.return_value = True  # Lock acquired
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("time.time", return_value=1640995200):
                with patch("hashlib.sha256") as mock_hash:
                    mock_hash.return_value.hexdigest.return_value = "abcdef1234567890"

                    result = connector.acquire_lock("resource123", timeout=30)

                    assert result == "abcdef12"  # First 16 chars
                    mock_redis.set.assert_called_once()

    def test_acquire_lock_failed(self):
        """Test failed lock acquisition."""
        connector = RedisConnector(config={}, test_mode=True)

        mock_redis = Mock()
        mock_redis.set.return_value = False  # Lock not acquired
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.acquire_lock("resource123")

            assert result is None

    def test_release_lock_success(self):
        """Test successful lock release."""
        config = {"redis": {"prefixes": {"lock": "lock:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        mock_redis = Mock()
        mock_redis.eval.return_value = 1  # Lock released
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.release_lock("resource123", "lock_id_123")

            assert result is True
            mock_redis.eval.assert_called_once()

    def test_release_lock_wrong_owner(self):
        """Test lock release by wrong owner."""
        connector = RedisConnector(config={}, test_mode=True)

        mock_redis = Mock()
        mock_redis.eval.return_value = 0  # Lock not released (wrong owner)
        connector.redis_client = mock_redis

        with patch.object(connector, "ensure_connected", return_value=True):
            result = connector.release_lock("resource123", "wrong_lock_id")

            assert result is False

    def test_record_metric_not_connected(self):
        """Test recording metric when not connected."""
        connector = RedisConnector(config={}, test_mode=True)

        metric = MetricEvent(
            timestamp=datetime.utcnow(), metric_name="test_metric", value=1.0, tags={}
        )

        with patch.object(connector, "ensure_connected", return_value=False):
            result = connector.record_metric(metric)

            assert result is False

    def test_record_metric_invalid(self):
        """Test recording invalid metric."""
        connector = RedisConnector(config={}, test_mode=True)

        metric = MetricEvent(
            timestamp=datetime.utcnow(), metric_name="invalid metric name", value=1.0, tags={}
        )

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("storage.kv_store.validate_metric_event", return_value=False):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.record_metric(metric)

                    assert result is False
                    mock_log.assert_called_once()

    def test_record_metric_success(self):
        """Test successful metric recording."""
        config = {"redis": {"prefixes": {"metric": "metric:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        mock_redis = Mock()
        connector.redis_client = mock_redis

        metric = MetricEvent(
            timestamp=datetime(2025, 1, 1, 12, 30, 0),
            metric_name="test.metric",
            value=42.5,
            tags={"env": "test"},
        )

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("storage.kv_store.validate_metric_event", return_value=True):
                with patch("storage.kv_store.sanitize_metric_name", return_value="test_metric"):
                    result = connector.record_metric(metric)

                    assert result is True
                    mock_redis.zadd.assert_called_once()
                    mock_redis.expire.assert_called_once()

    def test_get_metrics_invalid_time_range(self):
        """Test getting metrics with invalid time range."""
        connector = RedisConnector(config={}, test_mode=True)

        start_time = datetime(2025, 1, 2)
        end_time = datetime(2025, 1, 1)  # End before start

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("storage.kv_store.validate_time_range", return_value=False):
                with patch.object(connector, "log_error") as mock_log:
                    result = connector.get_metrics("test_metric", start_time, end_time)

                    assert result == []
                    mock_log.assert_called_once()

    def test_get_metrics_success(self):
        """Test successful metrics retrieval."""
        config = {"redis": {"prefixes": {"metric": "metric:"}}}
        connector = RedisConnector(config=config, test_mode=True)

        mock_redis = Mock()
        metric_data = {
            "timestamp": "2025-01-01T12:00:00",
            "metric_name": "test_metric",
            "value": 10.0,
            "tags": {},
        }
        mock_redis.zrangebyscore.return_value = [json.dumps(metric_data)]
        connector.redis_client = mock_redis

        start_time = datetime(2025, 1, 1, 11, 0, 0)
        end_time = datetime(2025, 1, 1, 13, 0, 0)

        with patch.object(connector, "ensure_connected", return_value=True):
            with patch("storage.kv_store.validate_time_range", return_value=True):
                with patch("storage.kv_store.sanitize_metric_name", return_value="test_metric"):
                    result = connector.get_metrics("test_metric", start_time, end_time)

                    assert len(result) == 1
                    assert isinstance(result[0], MetricEvent)
                    assert result[0].metric_name == "test_metric"

    def test_close_connection(self):
        """Test closing Redis connection."""
        connector = RedisConnector(config={}, test_mode=True)

        mock_redis = Mock()
        connector.redis_client = mock_redis
        connector.is_connected = True

        connector.close()

        mock_redis.close.assert_called_once()
        assert connector.redis_client is None
        assert connector.is_connected is False

    def test_close_connection_with_error(self):
        """Test closing connection with error."""
        connector = RedisConnector(config={}, test_mode=True)

        mock_redis = Mock()
        mock_redis.close.side_effect = Exception("Close failed")
        connector.redis_client = mock_redis

        with patch.object(connector, "log_error") as mock_log:
            connector.close()

            mock_log.assert_called_once()
            assert connector.redis_client is None


class TestDuckDBAnalytics:
    """Test DuckDBAnalytics class."""

    def test_init(self):
        """Test DuckDBAnalytics initialization."""
        with patch("storage.kv_store.DatabaseComponent.__init__") as mock_parent_init:
            with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
                analytics = DuckDBAnalytics(".ctxrc.yaml", verbose=True)

                mock_parent_init.assert_called_once_with(".ctxrc.yaml", True)
                assert analytics.conn is None

    def test_get_service_name(self):
        """Test service name for configuration."""
        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            assert analytics._get_service_name() == "duckdb"

    @patch("storage.kv_store.duckdb.connect")
    @patch("pathlib.Path.mkdir")
    def test_connect_success(self, mock_mkdir, mock_duckdb_connect):
        """Test successful DuckDB connection."""
        config = {"duckdb": {"database_path": "/tmp/test.db", "memory_limit": "1GB", "threads": 2}}

        mock_conn = Mock()
        mock_duckdb_connect.return_value = mock_conn

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.config = config

            with patch.object(analytics, "_initialize_tables"):
                with patch.object(analytics, "log_success") as mock_log:
                    result = analytics.connect()

                    assert result is True
                    assert analytics.conn == mock_conn
                    assert analytics.is_connected is True
                    mock_log.assert_called_once()

    @patch("storage.kv_store.duckdb.connect")
    def test_connect_os_error(self, mock_duckdb_connect):
        """Test DuckDB connection with OS error."""
        mock_duckdb_connect.side_effect = OSError("Permission denied")

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()

            with patch.object(analytics, "log_error") as mock_log:
                result = analytics.connect()

                assert result is False
                mock_log.assert_called_once()

    def test_initialize_tables(self):
        """Test table initialization."""
        config = {
            "duckdb": {
                "tables": {
                    "metrics": "custom_metrics",
                    "events": "custom_events",
                    "summaries": "custom_summaries",
                    "trends": "custom_trends",
                }
            }
        }

        mock_conn = Mock()

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.config = config
            analytics.conn = mock_conn

            analytics._initialize_tables()

            # Should execute multiple CREATE TABLE statements
            assert mock_conn.execute.call_count >= 7  # Tables + indexes

    def test_insert_metrics_not_connected(self):
        """Test inserting metrics when not connected."""
        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()

            with patch.object(analytics, "ensure_connected", return_value=False):
                result = analytics.insert_metrics([])

                assert result is False

    def test_insert_metrics_success(self):
        """Test successful metrics insertion."""
        config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}

        mock_conn = Mock()

        metrics = [
            MetricEvent(
                timestamp=datetime(2025, 1, 1, 12, 0, 0),
                metric_name="test_metric",
                value=10.0,
                tags={"env": "test"},
            )
        ]

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.config = config
            analytics.conn = mock_conn

            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.insert_metrics(metrics)

                assert result is True
                mock_conn.executemany.assert_called_once()

    def test_query_metrics_not_connected(self):
        """Test querying metrics when not connected."""
        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()

            with patch.object(analytics, "ensure_connected", return_value=False):
                result = analytics.query_metrics("SELECT 1")

                assert result == []

    def test_query_metrics_success(self):
        """Test successful metrics query."""
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = [(1, "test"), (2, "data")]
        mock_conn.description = [("id", None), ("name", None)]

        with patch.object(
            DuckDBAnalytics,
            "_load_performance_config",
            return_value={"query": {"timeout_seconds": 30}},
        ):
            analytics = DuckDBAnalytics()
            analytics.conn = mock_conn

            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.query_metrics("SELECT id, name FROM test", ["param1"])

                assert result == [{"id": 1, "name": "test"}, {"id": 2, "name": "data"}]
                mock_conn.execute.assert_any_call("SET statement_timeout = '30s'")

    def test_aggregate_metrics_success(self):
        """Test successful metrics aggregation."""
        config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}

        mock_conn = Mock()
        mock_conn.execute.return_value.fetchone.return_value = (
            25.0,
            10,
            datetime(2025, 1, 1),
            datetime(2025, 1, 2),
        )

        start_time = datetime(2025, 1, 1)
        end_time = datetime(2025, 1, 2)

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.config = config
            analytics.conn = mock_conn

            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.aggregate_metrics("test_metric", start_time, end_time, "avg")

                expected = {
                    "aggregation": "avg",
                    "value": 25.0,
                    "count": 10,
                    "start_time": datetime(2025, 1, 1),
                    "end_time": datetime(2025, 1, 2),
                }
                assert result == expected

    def test_generate_summary_daily(self):
        """Test generating daily summary."""
        config = {"duckdb": {"tables": {"metrics": "test_metrics", "summaries": "test_summaries"}}}

        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ("test_metric", 5, 10.0, 5.0, 15.0, 3.0)
        ]

        summary_date = date(2025, 1, 1)

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.config = config
            analytics.conn = mock_conn

            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.generate_summary(summary_date, "daily")

                assert result["summary_type"] == "daily"
                assert "test_metric" in result["metrics"]
                assert result["metrics"]["test_metric"]["count"] == 5

    def test_detect_trends_insufficient_data(self):
        """Test trend detection with insufficient data."""
        config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}

        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = [
            (datetime(2025, 1, 1), 10.0, 1)
        ]  # Only 1 data point

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.config = config
            analytics.conn = mock_conn

            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.detect_trends("test_metric", 7)

                assert result == {"trend": "insufficient_data"}

    def test_detect_trends_success(self):
        """Test successful trend detection."""
        config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}

        mock_conn = Mock()
        # Mock time series data showing increasing trend
        mock_conn.execute.return_value.fetchall.return_value = [
            (datetime(2025, 1, 1, 0), 10.0, 1),
            (datetime(2025, 1, 1, 1), 12.0, 1),
            (datetime(2025, 1, 1, 2), 15.0, 1),
            (datetime(2025, 1, 1, 3), 18.0, 1),
        ]

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.config = config
            analytics.conn = mock_conn

            with patch.object(analytics, "ensure_connected", return_value=True):
                result = analytics.detect_trends("test_metric", 7)

                assert result["trend_direction"] == "increasing"
                assert result["data_points"] == 4
                assert "trend_strength" in result

    def test_close_connection(self):
        """Test closing DuckDB connection."""
        mock_conn = Mock()

        with patch.object(DuckDBAnalytics, "_load_performance_config", return_value={}):
            analytics = DuckDBAnalytics()
            analytics.conn = mock_conn
            analytics.is_connected = True

            analytics.close()

            mock_conn.close.assert_called_once()
            assert analytics.conn is None
            assert analytics.is_connected is False


class TestContextKV:
    """Test ContextKV unified interface."""

    def test_init_test_mode(self):
        """Test ContextKV initialization in test mode."""
        config = {"redis": {"host": "localhost"}}

        kv = ContextKV(config=config, test_mode=True)

        assert kv.config == config
        assert kv.test_mode is True
        assert kv.duckdb is None  # Should not create DuckDB in test mode

    def test_init_production_mode(self):
        """Test ContextKV initialization in production mode."""
        config = {"redis": {"host": "localhost"}}

        with patch("storage.kv_store.DuckDBAnalytics") as mock_duckdb_class:
            mock_duckdb = Mock()
            mock_duckdb_class.return_value = mock_duckdb

            kv = ContextKV(config=config, test_mode=False)

            assert kv.duckdb == mock_duckdb

    def test_init_production_mode_duckdb_error(self):
        """Test ContextKV initialization with DuckDB error."""
        config = {"redis": {"host": "localhost"}}

        with patch("storage.kv_store.DuckDBAnalytics", side_effect=Exception("DuckDB error")):
            kv = ContextKV(config=config, test_mode=False)

            assert kv.duckdb is None

    def test_connect_success(self):
        """Test successful connection to both stores."""
        kv = ContextKV(config={}, test_mode=True)

        with patch.object(kv.redis, "connect", return_value=True):
            with patch.object(kv, "duckdb", None):  # No DuckDB in test mode
                result = kv.connect()

                # Should succeed even without DuckDB
                assert result is True

    def test_connect_with_duckdb(self):
        """Test connection with DuckDB."""
        kv = ContextKV(config={}, test_mode=False)
        kv.duckdb = Mock()

        with patch.object(kv.redis, "connect", return_value=True):
            with patch.object(kv.duckdb, "connect", return_value=True):
                result = kv.connect(redis_password="password")

                assert result is True

    def test_record_event_success(self):
        """Test successful event recording."""
        kv = ContextKV(config={}, test_mode=True)
        kv.duckdb = Mock()

        with patch.object(kv.redis, "record_metric", return_value=True):
            with patch.object(kv.duckdb, "insert_metrics", return_value=True):
                result = kv.record_event("test_event", document_id="doc123", data={"key": "value"})

                assert result is True

    def test_get_recent_activity(self):
        """Test getting recent activity summary."""
        kv = ContextKV(config={}, test_mode=True)
        kv.duckdb = Mock()

        mock_metrics = [{"metric_name": "event.test", "count": 5, "avg_value": 1.0}]

        with patch.object(kv.duckdb, "query_metrics", return_value=mock_metrics):
            result = kv.get_recent_activity(hours=12)

            assert result["period_hours"] == 12
            assert result["metrics"] == mock_metrics

    def test_close_connections(self):
        """Test closing all connections."""
        kv = ContextKV(config={}, test_mode=True)
        kv.duckdb = Mock()

        with patch.object(kv.redis, "close") as mock_redis_close:
            with patch.object(kv.duckdb, "close") as mock_duckdb_close:
                kv.close()

                mock_redis_close.assert_called_once()
                mock_duckdb_close.assert_called_once()


class TestKVStoreCliCommands:
    """Test CLI command functionality."""

    @patch("storage.kv_store.ContextKV")
    def test_test_connection_success(self, mock_kv_class):
        """Test successful connection test command."""
        mock_kv = Mock()
        mock_kv.connect.return_value = True
        mock_kv.redis.set_cache.return_value = True
        mock_kv.redis.get_cache.return_value = {"test": True, "timestamp": "2025-01-01T12:00:00"}
        mock_kv.duckdb.query_metrics.return_value = [{"test": 1}]
        mock_kv.record_event.return_value = True

        mock_kv_class.return_value = mock_kv

        from click.testing import CliRunner

        from src.storage.kv_store import test_connection

        runner = CliRunner()
        result = runner.invoke(test_connection, ["--redis-pass", "password", "--verbose"])

        assert result.exit_code == 0
        assert "✓ All connections successful!" in result.output

    @patch("storage.kv_store.ContextKV")
    def test_record_metric_success(self, mock_kv_class):
        """Test successful metric recording command."""
        mock_kv = Mock()
        mock_kv.connect.return_value = True
        mock_kv.redis.record_metric.return_value = True
        mock_kv.duckdb.insert_metrics.return_value = True

        mock_kv_class.return_value = mock_kv

        from click.testing import CliRunner

        from src.storage.kv_store import record_metric

        runner = CliRunner()
        result = runner.invoke(
            record_metric,
            [
                "--metric",
                "test_metric",
                "--value",
                "42.5",
                "--document-id",
                "doc123",
                "--agent-id",
                "agent456",
            ],
        )

        assert result.exit_code == 0
        assert "✓ Recorded metric: test_metric = 42.5" in result.output

    @patch("storage.kv_store.ContextKV")
    def test_activity_summary_success(self, mock_kv_class):
        """Test successful activity summary command."""
        mock_kv = Mock()
        mock_kv.connect.return_value = True
        mock_kv.get_recent_activity.return_value = {
            "start_time": "2025-01-01T00:00:00",
            "end_time": "2025-01-01T12:00:00",
            "metrics": [{"metric_name": "test_metric", "count": 10, "avg_value": 5.5}],
        }

        mock_kv_class.return_value = mock_kv

        from click.testing import CliRunner

        from src.storage.kv_store import activity_summary

        runner = CliRunner()
        result = runner.invoke(activity_summary, ["--hours", "12"])

        assert result.exit_code == 0
        assert "Activity Summary (last 12 hours)" in result.output
        assert "test_metric: 10 events, avg value: 5.50" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
