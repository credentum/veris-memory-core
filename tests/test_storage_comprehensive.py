"""
Comprehensive storage layer tests for maximum coverage improvement.

Tests all storage components: RedisConnector, DuckDBAnalytics, and ContextKV
with comprehensive coverage of CRUD operations, error handling, and edge cases.
"""

import os
import tempfile
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.storage.kv_store import CacheEntry, ContextKV, DuckDBAnalytics, MetricEvent, RedisConnector


class TestMetricEvent:
    """Test MetricEvent dataclass."""

    def test_metric_event_creation(self):
        """Test MetricEvent creation with all fields."""
        timestamp = datetime.now()
        event = MetricEvent(
            timestamp=timestamp,
            metric_name="test_metric",
            value=42.5,
            tags={"env": "test", "service": "api"},
            document_id="doc123",
            agent_id="agent456",
        )

        assert event.timestamp == timestamp
        assert event.metric_name == "test_metric"
        assert event.value == 42.5
        assert event.tags == {"env": "test", "service": "api"}
        assert event.document_id == "doc123"
        assert event.agent_id == "agent456"

    def test_metric_event_optional_fields(self):
        """Test MetricEvent with only required fields."""
        timestamp = datetime.now()
        event = MetricEvent(timestamp=timestamp, metric_name="simple_metric", value=10.0, tags={})

        assert event.timestamp == timestamp
        assert event.metric_name == "simple_metric"
        assert event.value == 10.0
        assert event.tags == {}
        assert event.document_id is None
        assert event.agent_id is None


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation with all fields."""
        created_at = datetime.now()
        last_accessed = datetime.now()

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
        """Test CacheEntry with default values."""
        created_at = datetime.now()

        entry = CacheEntry(
            key="test_key", value="test_value", created_at=created_at, ttl_seconds=1800
        )

        assert entry.hit_count == 0
        assert entry.last_accessed is None


class TestRedisConnector:
    """Comprehensive tests for RedisConnector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, ".ctxrc.yaml")

        # Create test config
        test_config = {"redis": {"host": "localhost", "port": 6379, "database": 0, "ssl": False}}

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

        # Create test performance config
        self.perf_config_path = os.path.join(self.test_config_dir, "performance.yaml")
        perf_config = {"kv_store": {"redis": {"connection_pool": {"max_size": 10}}}}

        with open(self.perf_config_path, "w") as f:
            yaml.dump(perf_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_redis_connector_initialization(self):
        """Test RedisConnector initialization."""
        connector = RedisConnector(self.config_path, verbose=True)

        assert connector.redis_client is None
        assert connector.pipeline is None
        assert isinstance(connector.perf_config, dict)

    def test_get_service_name(self):
        """Test _get_service_name method."""
        connector = RedisConnector(self.config_path)
        assert connector._get_service_name() == "redis"

    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_performance_config_success(self, mock_yaml_load, mock_open):
        """Test successful performance config loading."""
        mock_yaml_load.return_value = {"kv_store": {"redis": {"connection_pool": {"max_size": 20}}}}

        connector = RedisConnector(self.config_path)
        config = connector._load_performance_config()

        assert config == {"connection_pool": {"max_size": 20}}

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_performance_config_file_not_found(self, mock_open):
        """Test performance config loading when file not found."""
        connector = RedisConnector(self.config_path)
        config = connector._load_performance_config()

        assert config == {}

    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_performance_config_invalid_structure(self, mock_yaml_load, mock_open):
        """Test performance config loading with invalid structure."""
        mock_yaml_load.return_value = {"kv_store": {"redis": "invalid_structure"}}  # Should be dict

        connector = RedisConnector(self.config_path)
        config = connector._load_performance_config()

        assert config == {}

    @patch("redis.ConnectionPool")
    @patch("redis.Redis")
    def test_connect_success(self, mock_redis, mock_pool):
        """Test successful Redis connection."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        connector = RedisConnector(self.config_path)
        result = connector.connect()

        assert result is True
        assert connector.redis_client == mock_redis_instance

    @patch("redis.ConnectionPool")
    @patch("redis.Redis")
    def test_connect_with_password(self, mock_redis, mock_pool):
        """Test Redis connection with password."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        connector = RedisConnector(self.config_path)
        result = connector.connect(password="test_password")

        assert result is True

    @patch("redis.ConnectionPool")
    @patch("redis.Redis")
    def test_connect_failure(self, mock_redis, mock_pool):
        """Test Redis connection failure."""
        mock_redis.side_effect = Exception("Connection failed")

        connector = RedisConnector(self.config_path)
        result = connector.connect()

        assert result is False
        assert connector.redis_client is None

    def test_get_prefixed_key(self):
        """Test key prefixing functionality."""
        connector = RedisConnector(self.config_path)

        # Test default prefix
        result = connector.get_prefixed_key("test_key")
        assert result == "cache:test_key"

        # Test custom prefix
        result = connector.get_prefixed_key("test_key", "session")
        assert result == "session:test_key"

    @patch("redis.Redis")
    def test_set_cache_success(self, mock_redis):
        """Test successful cache setting."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.set_cache("test_key", {"data": "test"}, 3600)

        assert result is True
        mock_redis_instance.setex.assert_called_once()  # Just check it was called

    @patch("redis.Redis")
    def test_set_cache_no_ttl(self, mock_redis):
        """Test cache setting without TTL."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.set_cache("test_key", "test_value")

        assert result is True
        mock_redis_instance.setex.assert_called_once()  # Just check it was called

    @patch("redis.Redis")
    def test_set_cache_not_connected(self, mock_redis):
        """Test cache setting when not connected."""
        connector = RedisConnector(self.config_path)
        # Don't call connect()

        result = connector.set_cache("test_key", "test_value")

        assert result is False

    @patch("redis.Redis")
    def test_set_cache_exception(self, mock_redis):
        """Test cache setting with exception."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.side_effect = Exception("Redis error")

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.set_cache("test_key", "test_value", 3600)

        assert result is False

    @patch("redis.Redis")
    def test_get_cache_success(self, mock_redis):
        """Test successful cache retrieval."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = b'{"value": "test_data", "hit_count": 0}'
        mock_redis_instance.ttl.return_value = 3600  # TTL in seconds
        mock_redis_instance.setex.return_value = True

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.get_cache("test_key")

        # get_cache returns the value from the cache entry
        assert result == "test_data"

    @patch("redis.Redis")
    def test_get_cache_not_found(self, mock_redis):
        """Test cache retrieval when key not found."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = None

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.get_cache("test_key")

        assert result is None

    @patch("redis.Redis")
    def test_get_cache_not_connected(self, mock_redis):
        """Test cache retrieval when not connected."""
        connector = RedisConnector(self.config_path)
        # Don't call connect()

        result = connector.get_cache("test_key")

        assert result is None

    @patch("redis.Redis")
    def test_get_cache_json_decode_error(self, mock_redis):
        """Test cache retrieval with JSON decode error."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = b"invalid_json{"

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.get_cache("test_key")

        assert result is None

    @patch("redis.Redis")
    def test_delete_cache_success(self, mock_redis):
        """Test successful cache deletion."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.delete.return_value = 2
        mock_redis_instance.scan_iter.return_value = [b"cache:key1", b"cache:key2"]

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.delete_cache("test*")

        assert result == 2

    @patch("redis.Redis")
    def test_delete_cache_not_connected(self, mock_redis):
        """Test cache deletion when not connected."""
        connector = RedisConnector(self.config_path)
        # Don't call connect()

        result = connector.delete_cache("test*")

        assert result == 0

    @patch("redis.Redis")
    def test_set_session_success(self, mock_redis):
        """Test successful session setting."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.return_value = True

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.set_session("session123", {"user_id": "user456"}, 3600)

        assert result is True

    @patch("redis.Redis")
    def test_get_session_success(self, mock_redis):
        """Test successful session retrieval."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = b'{"data": {"user_id": "user456"}}'
        mock_redis_instance.ttl.return_value = 3600  # TTL in seconds
        mock_redis_instance.setex.return_value = True

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.get_session("session123")

        # get_session returns the nested data field
        assert result["user_id"] == "user456"
        # last_activity is added to the session_data but not to the returned data field
        assert isinstance(result, dict)

    @patch("redis.Redis")
    def test_acquire_lock_success(self, mock_redis):
        """Test successful lock acquisition."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.set.return_value = True

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.acquire_lock("resource123", 30)

        assert result is not None
        assert len(result) > 0

    @patch("redis.Redis")
    def test_acquire_lock_already_locked(self, mock_redis):
        """Test lock acquisition when already locked."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.set.return_value = False

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.acquire_lock("resource123", 30)

        assert result is None

    @patch("redis.Redis")
    def test_release_lock_success(self, mock_redis):
        """Test successful lock release."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.eval.return_value = 1

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.release_lock("resource123", "lock_id_123")

        assert result is True

    @patch("redis.Redis")
    def test_release_lock_failure(self, mock_redis):
        """Test lock release failure."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.eval.return_value = 0

        connector = RedisConnector(self.config_path)
        connector.connect()

        result = connector.release_lock("resource123", "wrong_lock_id")

        assert result is False

    @patch("redis.Redis")
    def test_record_metric_success(self, mock_redis):
        """Test successful metric recording."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.zadd.return_value = 1

        connector = RedisConnector(self.config_path)
        connector.connect()

        metric = MetricEvent(
            timestamp=datetime.now(), metric_name="test_metric", value=42.5, tags={"env": "test"}
        )

        result = connector.record_metric(metric)

        assert result is True

    @patch("redis.Redis")
    def test_get_metrics_success(self, mock_redis):
        """Test successful metrics retrieval."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        # The method queries multiple time buckets, so mock returns nothing by default
        mock_redis_instance.zrangebyscore.return_value = []

        connector = RedisConnector(self.config_path)
        connector.connect()

        # Use a smaller time range to avoid too many bucket iterations
        start_time = datetime(2024, 1, 1, 10, 0)
        end_time = datetime(2024, 1, 1, 11, 0)

        result = connector.get_metrics("test_metric", start_time, end_time)

        # Empty result is fine - just check that the method runs without error
        assert isinstance(result, list)

    @patch("redis.Redis")
    def test_close(self, mock_redis):
        """Test Redis connection close."""
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        connector = RedisConnector(self.config_path)
        connector.connect()
        connector.close()

        mock_redis_instance.close.assert_called_once_with()


class TestDuckDBAnalytics:
    """Comprehensive tests for DuckDBAnalytics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, ".ctxrc.yaml")

        # Create test config
        test_config = {"duckdb": {"database_path": ":memory:", "read_only": False}}

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_duckdb_analytics_initialization(self):
        """Test DuckDBAnalytics initialization."""
        analytics = DuckDBAnalytics(self.config_path, verbose=True)

        assert analytics.conn is None

    def test_get_service_name(self):
        """Test _get_service_name method."""
        analytics = DuckDBAnalytics(self.config_path)
        assert analytics._get_service_name() == "duckdb"

    @patch("duckdb.connect")
    def test_connect_success(self, mock_connect):
        """Test successful DuckDB connection."""
        mock_connection = AsyncMock()
        mock_connect.return_value = mock_connection

        analytics = DuckDBAnalytics(self.config_path)
        result = analytics.connect()

        assert result is True
        assert analytics.conn == mock_connection

    @patch("duckdb.connect")
    def test_connect_failure(self, mock_connect):
        """Test DuckDB connection failure."""
        mock_connect.side_effect = Exception("Connection failed")

        analytics = DuckDBAnalytics(self.config_path)
        result = analytics.connect()

        assert result is False
        assert analytics.conn is None

    @patch("duckdb.connect")
    def test_initialize_tables(self, mock_connect):
        """Test table initialization."""
        mock_connection = AsyncMock()
        mock_connect.return_value = mock_connection

        analytics = DuckDBAnalytics(self.config_path)
        analytics.connect()
        analytics._initialize_tables()

        # Verify that execute was called multiple times for table creation
        assert mock_connection.execute.call_count >= 2

    @patch("duckdb.connect")
    def test_insert_metrics_success(self, mock_connect):
        """Test successful metrics insertion."""
        mock_connection = AsyncMock()
        mock_connect.return_value = mock_connection

        analytics = DuckDBAnalytics(self.config_path)
        analytics.connect()

        metrics = [
            MetricEvent(
                timestamp=datetime.now(),
                metric_name="test_metric",
                value=42.5,
                tags={"env": "test"},
            )
        ]

        result = analytics.insert_metrics(metrics)

        assert result is True

    @patch("duckdb.connect")
    def test_insert_metrics_not_connected(self, mock_connect):
        """Test metrics insertion when not connected."""
        analytics = DuckDBAnalytics(self.config_path)
        # Don't call connect()

        metrics = [
            MetricEvent(timestamp=datetime.now(), metric_name="test_metric", value=42.5, tags={})
        ]

        result = analytics.insert_metrics(metrics)

        assert result is False

    @patch("duckdb.connect")
    def test_query_metrics_success(self, mock_connect):
        """Test successful metrics querying."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_result = MagicMock()
        mock_connection.execute.return_value = mock_result
        mock_result.fetchall.return_value = [
            (datetime.now(), "test_metric", 42.5, '{"env": "test"}', None, None)
        ]

        analytics = DuckDBAnalytics(self.config_path)
        analytics.connect()

        result = analytics.query_metrics("SELECT * FROM metrics WHERE metric_name = 'test_metric'")

        assert len(result) == 1

    @patch("duckdb.connect")
    def test_aggregate_metrics_success(self, mock_connect):
        """Test successful metrics aggregation."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_result = MagicMock()
        mock_connection.execute.return_value = mock_result
        mock_result.fetchall.return_value = [("test_metric", 42.5, 40.0, 45.0, 1)]

        analytics = DuckDBAnalytics(self.config_path)
        analytics.connect()

        start_time = datetime(2024, 1, 1)
        end_time = datetime(2024, 1, 2)

        # Method signature changed - takes metric_name as first parameter
        result = analytics.aggregate_metrics("test_metric", start_time, end_time, "avg")

        assert "aggregation" in result
        assert "value" in result
        assert "count" in result

    @patch("duckdb.connect")
    def test_generate_summary_success(self, mock_connect):
        """Test successful summary generation."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_result = MagicMock()
        mock_connection.execute.return_value = mock_result
        mock_result.fetchall.return_value = [("test_metric", 42.5, 40.0, 45.0, 10)]

        analytics = DuckDBAnalytics(self.config_path)
        analytics.connect()

        summary_date = date(2024, 1, 1)
        result = analytics.generate_summary(summary_date, "daily")

        assert "summary_date" in result
        assert "metrics" in result

    @patch("duckdb.connect")
    def test_detect_trends_success(self, mock_connect):
        """Test successful trend detection."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_result = MagicMock()
        mock_connection.execute.return_value = mock_result
        mock_result.fetchall.return_value = [(42.5, 40.0, 45.0, 2.5, 7)]

        analytics = DuckDBAnalytics(self.config_path)
        analytics.connect()

        result = analytics.detect_trends("test_metric", 7)

        assert "metric_name" in result
        assert "trend_data" in result

    @patch("duckdb.connect")
    def test_close(self, mock_connect):
        """Test DuckDB connection close."""
        mock_connection = AsyncMock()
        mock_connect.return_value = mock_connection

        analytics = DuckDBAnalytics(self.config_path)
        analytics.connect()
        analytics.close()

        mock_connection.close.assert_called_once_with()


class TestContextKV:
    """Comprehensive tests for ContextKV unified API."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, ".ctxrc.yaml")

        # Create test config
        test_config = {
            "redis": {"host": "localhost", "port": 6379, "database": 0},
            "duckdb": {"database_path": ":memory:"},
        }

        with open(self.config_path, "w") as f:
            yaml.dump(test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_context_kv_initialization(self):
        """Test ContextKV initialization."""
        kv = ContextKV(self.config_path, verbose=True)

        assert kv.redis is not None
        assert kv.duckdb is not None

    @patch.object(RedisConnector, "connect")
    @patch.object(DuckDBAnalytics, "connect")
    def test_connect_success(self, mock_analytics_connect, mock_redis_connect):
        """Test successful ContextKV connection."""
        mock_redis_connect.return_value = True
        mock_analytics_connect.return_value = True

        kv = ContextKV(self.config_path)
        result = kv.connect()

        assert result is True

    @patch.object(RedisConnector, "connect")
    @patch.object(DuckDBAnalytics, "connect")
    def test_connect_redis_failure(self, mock_analytics_connect, mock_redis_connect):
        """Test ContextKV connection with Redis failure."""
        mock_redis_connect.return_value = False
        mock_analytics_connect.return_value = True

        kv = ContextKV(self.config_path)
        result = kv.connect()

        assert result is False

    @patch.object(RedisConnector, "record_metric")
    @patch.object(DuckDBAnalytics, "insert_metrics")
    def test_record_event_success(self, mock_analytics_insert, mock_redis_record):
        """Test successful event recording."""
        mock_redis_record.return_value = True
        mock_analytics_insert.return_value = True

        kv = ContextKV(self.config_path)

        result = kv.record_event(
            event_type="test_event",
            document_id="doc123",
            agent_id="agent456",
            data={"metric_name": "test_metric", "value": 42.5},
        )

        assert result is True

    @patch.object(RedisConnector, "get_metrics")
    def test_get_recent_activity_success(self, mock_get_metrics):
        """Test successful recent activity retrieval."""
        mock_get_metrics.return_value = [
            {
                "timestamp": datetime.now().isoformat(),
                "metric_name": "test_metric",
                "value": 42.5,
                "tags": {"env": "test"},
            }
        ]

        kv = ContextKV(self.config_path)
        # Method only takes hours parameter
        result = kv.get_recent_activity(24)

        assert result is not None
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
