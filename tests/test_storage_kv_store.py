#!/usr/bin/env python3
"""
Comprehensive tests for Storage KV Store - Phase 8 Coverage

This test module provides comprehensive coverage for the key-value storage system
including Redis connector, DuckDB analytics, and unified KV operations.
"""
import pytest
import tempfile
import json
import time
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock, MagicMock, mock_open
from typing import Dict, Any, List, Optional

# Import KV store components
try:
    from src.storage.kv_store import (
        MetricEvent, CacheEntry, RedisConnector, DuckDBAnalytics, ContextKV
    )
    KV_STORE_AVAILABLE = True
except ImportError:
    KV_STORE_AVAILABLE = False


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestKVStoreDataModels:
    """Test KV store data models"""
    
    def test_metric_event_creation(self):
        """Test MetricEvent dataclass creation"""
        now = datetime.now(timezone.utc)
        event = MetricEvent(
            timestamp=now,
            metric_name="test_metric",
            value=42.5,
            tags={"environment": "test", "service": "kv_store"},
            document_id="doc123",
            agent_id="agent456"
        )
        
        assert event.timestamp == now
        assert event.metric_name == "test_metric"
        assert event.value == 42.5
        assert event.tags == {"environment": "test", "service": "kv_store"}
        assert event.document_id == "doc123"
        assert event.agent_id == "agent456"
    
    def test_metric_event_defaults(self):
        """Test MetricEvent default values"""
        now = datetime.now(timezone.utc)
        event = MetricEvent(
            timestamp=now,
            metric_name="simple_metric",
            value=1.0,
            tags={}
        )
        
        assert event.document_id is None
        assert event.agent_id is None
    
    def test_cache_entry_creation(self):
        """Test CacheEntry dataclass creation"""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            key="test_key",
            value={"data": "test_value", "number": 123},
            created_at=now,
            ttl_seconds=3600,
            hit_count=5,
            last_accessed=now
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test_value", "number": 123}
        assert entry.created_at == now
        assert entry.ttl_seconds == 3600
        assert entry.hit_count == 5
        assert entry.last_accessed == now
    
    def test_cache_entry_defaults(self):
        """Test CacheEntry default values"""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            key="simple_key",
            value="simple_value",
            created_at=now,
            ttl_seconds=1800
        )
        
        assert entry.hit_count == 0
        assert entry.last_accessed is None


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestRedisConnector:
    """Test Redis connector functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "database": 0,
                "ssl": False,
                "prefixes": {
                    "cache": "test_cache:",
                    "session": "test_session:",
                    "metrics": "test_metrics:"
                }
            }
        }
        
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.connector = RedisConnector(config=self.test_config, test_mode=True)
    
    def test_redis_connector_initialization(self):
        """Test Redis connector initialization"""
        assert self.connector is not None
        assert self.connector.config == self.test_config
        assert self.connector.test_mode is True
        assert self.connector.redis_client is None
        assert hasattr(self.connector, 'perf_config')
    
    def test_get_service_name(self):
        """Test service name retrieval"""
        service_name = self.connector._get_service_name()
        assert service_name == "redis"
    
    @patch('src.storage.kv_store.yaml.safe_load')
    @patch('builtins.open', mock_open(read_data='kv_store:\n  redis:\n    connection_pool:\n      max_size: 100'))
    def test_load_performance_config_success(self, mock_yaml):
        """Test successful performance config loading"""
        mock_yaml.return_value = {
            "kv_store": {
                "redis": {
                    "connection_pool": {"max_size": 100},
                    "cache": {"ttl_seconds": 7200}
                }
            }
        }
        
        config = self.connector._load_performance_config()
        assert config == {
            "connection_pool": {"max_size": 100},
            "cache": {"ttl_seconds": 7200}
        }
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_performance_config_file_not_found(self, mock_open):
        """Test performance config loading when file not found"""
        config = self.connector._load_performance_config()
        assert config == {}
    
    def test_get_prefixed_key(self):
        """Test key prefixing"""
        cache_key = self.connector.get_prefixed_key("user123", "cache")
        assert cache_key == "test_cache:user123"
        
        session_key = self.connector.get_prefixed_key("session456", "session")
        assert session_key == "test_session:session456"
        
        # Test default prefix for unknown type
        unknown_key = self.connector.get_prefixed_key("data", "unknown")
        assert unknown_key == "unknown:data"
    
    @patch('src.storage.kv_store.redis.Redis')
    @patch('src.storage.kv_store.redis.ConnectionPool')
    def test_connect_success(self, mock_pool_class, mock_redis_class):
        """Test successful Redis connection"""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis
        
        result = self.connector.connect()
        
        assert result is True
        assert self.connector.is_connected is True
        assert self.connector.redis_client == mock_redis
        mock_redis.ping.assert_called_once()
    
    @patch('src.storage.kv_store.redis.Redis')
    @patch('src.storage.kv_store.redis.ConnectionPool')
    def test_connect_with_ssl(self, mock_pool_class, mock_redis_class):
        """Test Redis connection with SSL"""
        ssl_config = self.test_config.copy()
        ssl_config["redis"]["ssl"] = True
        
        with patch('src.storage.kv_store.yaml.safe_load'):
            ssl_connector = RedisConnector(config=ssl_config, test_mode=True)
        
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_class.return_value = mock_redis
        
        result = ssl_connector.connect()
        
        assert result is True
        # Verify SSL parameters were passed
        pool_call_args = mock_pool_class.call_args[1]
        assert pool_call_args["ssl"] is True
        assert "ssl_cert_reqs" in pool_call_args
    
    @patch('src.storage.kv_store.redis.Redis', side_effect=Exception("Connection failed"))
    def test_connect_failure(self, mock_redis_class):
        """Test Redis connection failure"""
        result = self.connector.connect()
        
        assert result is False
        assert self.connector.redis_client is None
    
    @patch('src.storage.kv_store.validate_redis_key')
    def test_set_cache_success(self, mock_validate):
        """Test successful cache set operation"""
        mock_validate.return_value = True
        
        mock_redis = MagicMock()
        self.connector.redis_client = mock_redis
        self.connector.is_connected = True
        
        # Mock ensure_connected to return True
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.set_cache("test_key", {"data": "value"}, 1800)
        
        assert result is True
        mock_validate.assert_called_once_with("test_key")
    
    @patch('src.storage.kv_store.validate_redis_key')
    def test_set_cache_invalid_key(self, mock_validate):
        """Test cache set with invalid key"""
        mock_validate.return_value = False
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.set_cache("invalid key!", "value")
        
        assert result is False
    
    def test_set_cache_not_connected(self):
        """Test cache set when not connected"""
        with patch.object(self.connector, 'ensure_connected', return_value=False):
            result = self.connector.set_cache("test_key", "value")
        
        assert result is False


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestRedisConnectorCacheOperations:
    """Test Redis connector cache operations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "database": 0,
                "prefixes": {"cache": "test_cache:"}
            }
        }
        
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.connector = RedisConnector(config=self.test_config, test_mode=True)
        
        # Mock Redis client
        self.mock_redis = MagicMock()
        self.connector.redis_client = self.mock_redis
        self.connector.is_connected = True
    
    @patch('src.storage.kv_store.validate_redis_key')
    def test_get_cache_success(self, mock_validate):
        """Test successful cache retrieval"""
        mock_validate.return_value = True
        
        # Mock Redis response
        cache_data = {
            "key": "test_key",
            "value": {"data": "test_value"},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": 3600,
            "hit_count": 1
        }
        self.mock_redis.get.return_value = json.dumps(cache_data)
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.get_cache("test_key")
        
        assert result == {"data": "test_value"}
        mock_validate.assert_called_once_with("test_key")
    
    @patch('src.storage.kv_store.validate_redis_key')
    def test_get_cache_not_found(self, mock_validate):
        """Test cache retrieval when key not found"""
        mock_validate.return_value = True
        self.mock_redis.get.return_value = None
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.get_cache("nonexistent_key")
        
        assert result is None
    
    @patch('src.storage.kv_store.validate_redis_key')
    def test_delete_cache_success(self, mock_validate):
        """Test successful cache deletion"""
        mock_validate.return_value = True
        self.mock_redis.delete.return_value = 1  # Number of keys deleted
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.delete_cache("test_key")
        
        assert result is True
        self.mock_redis.delete.assert_called_once_with("test_cache:test_key")
    
    @patch('src.storage.kv_store.validate_redis_key')
    def test_delete_cache_not_found(self, mock_validate):
        """Test cache deletion when key not found"""
        mock_validate.return_value = True
        self.mock_redis.delete.return_value = 0  # No keys deleted
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.delete_cache("nonexistent_key")
        
        assert result is False
    
    def test_list_cache_keys(self):
        """Test listing cache keys"""
        self.mock_redis.keys.return_value = [
            b"test_cache:key1",
            b"test_cache:key2", 
            b"test_cache:key3"
        ]
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            keys = self.connector.list_cache_keys()
        
        expected_keys = ["key1", "key2", "key3"]
        assert keys == expected_keys
    
    def test_cache_exists(self):
        """Test checking if cache key exists"""
        self.mock_redis.exists.return_value = 1
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.cache_exists("test_key")
        
        assert result is True
        self.mock_redis.exists.assert_called_once_with("test_cache:test_key")
    
    def test_get_cache_ttl(self):
        """Test getting cache TTL"""
        self.mock_redis.ttl.return_value = 1800  # 30 minutes
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            ttl = self.connector.get_cache_ttl("test_key")
        
        assert ttl == 1800
        self.mock_redis.ttl.assert_called_once_with("test_cache:test_key")


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestRedisConnectorSessionOperations:
    """Test Redis connector session operations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {
                "prefixes": {"session": "test_session:"}
            }
        }
        
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.connector = RedisConnector(config=self.test_config, test_mode=True)
        
        self.mock_redis = MagicMock()
        self.connector.redis_client = self.mock_redis
        self.connector.is_connected = True
    
    def test_create_session(self):
        """Test session creation"""
        session_data = {
            "user_id": "user123",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.create_session("session123", session_data, 7200)
        
        assert result is True
        self.mock_redis.setex.assert_called_once()
    
    def test_get_session(self):
        """Test session retrieval"""
        session_data = {
            "user_id": "user123",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.mock_redis.get.return_value = json.dumps(session_data)
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.get_session("session123")
        
        assert result == session_data
    
    def test_update_session(self):
        """Test session update"""
        updated_data = {
            "user_id": "user123",
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "new_field": "new_value"
        }
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.update_session("session123", updated_data)
        
        assert result is True
    
    def test_delete_session(self):
        """Test session deletion"""
        self.mock_redis.delete.return_value = 1
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.delete_session("session123")
        
        assert result is True
    
    def test_list_active_sessions(self):
        """Test listing active sessions"""
        self.mock_redis.keys.return_value = [
            b"test_session:session1",
            b"test_session:session2"
        ]
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            sessions = self.connector.list_active_sessions()
        
        assert sessions == ["session1", "session2"]


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestRedisConnectorMetricsOperations:
    """Test Redis connector metrics operations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {
                "prefixes": {"metrics": "test_metrics:"}
            }
        }
        
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.connector = RedisConnector(config=self.test_config, test_mode=True)
        
        self.mock_redis = MagicMock()
        self.connector.redis_client = self.mock_redis
        self.connector.is_connected = True
    
    @patch('src.storage.kv_store.validate_metric_event')
    def test_record_metric(self, mock_validate):
        """Test metric recording"""
        mock_validate.return_value = True
        
        metric = MetricEvent(
            timestamp=datetime.now(timezone.utc),
            metric_name="api_requests",
            value=1.0,
            tags={"endpoint": "/health", "method": "GET"}
        )
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.record_metric(metric)
        
        assert result is True
        mock_validate.assert_called_once_with(metric)
    
    @patch('src.storage.kv_store.validate_metric_event')
    def test_record_invalid_metric(self, mock_validate):
        """Test recording invalid metric"""
        mock_validate.return_value = False
        
        metric = MetricEvent(
            timestamp=datetime.now(timezone.utc),
            metric_name="",  # Invalid empty name
            value=1.0,
            tags={}
        )
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            result = self.connector.record_metric(metric)
        
        assert result is False
    
    def test_get_metric_keys(self):
        """Test getting metric keys"""
        self.mock_redis.keys.return_value = [
            b"test_metrics:api_requests:2025-01-15",
            b"test_metrics:db_queries:2025-01-15"
        ]
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            keys = self.connector.get_metric_keys("2025-01-15")
        
        assert "api_requests:2025-01-15" in keys
        assert "db_queries:2025-01-15" in keys
    
    def test_get_metric_summary(self):
        """Test getting metric summary"""
        metric_data = [
            json.dumps({"value": 10.0, "timestamp": "2025-01-15T10:00:00Z"}),
            json.dumps({"value": 15.0, "timestamp": "2025-01-15T11:00:00Z"}),
            json.dumps({"value": 20.0, "timestamp": "2025-01-15T12:00:00Z"})
        ]
        self.mock_redis.lrange.return_value = metric_data
        
        with patch.object(self.connector, 'ensure_connected', return_value=True):
            summary = self.connector.get_metric_summary("api_requests", "2025-01-15")
        
        assert summary["count"] == 3
        assert summary["total"] == 45.0
        assert summary["average"] == 15.0
        assert summary["min"] == 10.0
        assert summary["max"] == 20.0


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestDuckDBAnalytics:
    """Test DuckDB analytics functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "duckdb": {
                "database_path": ":memory:",
                "performance": {
                    "threads": 4,
                    "memory_limit": "1GB"
                }
            }
        }
        
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.analytics = DuckDBAnalytics(config=self.test_config, test_mode=True)
    
    def test_duckdb_analytics_initialization(self):
        """Test DuckDB analytics initialization"""
        assert self.analytics is not None
        assert self.analytics.config == self.test_config
        assert self.analytics.test_mode is True
        assert hasattr(self.analytics, 'db_connection')
    
    def test_get_service_name(self):
        """Test service name retrieval"""
        service_name = self.analytics._get_service_name()
        assert service_name == "duckdb"
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_connect_success(self, mock_connect):
        """Test successful DuckDB connection"""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        
        result = self.analytics.connect()
        
        assert result is True
        assert self.analytics.is_connected is True
        assert self.analytics.db_connection == mock_connection
    
    @patch('src.storage.kv_store.duckdb.connect', side_effect=Exception("Connection failed"))
    def test_connect_failure(self, mock_connect):
        """Test DuckDB connection failure"""
        result = self.analytics.connect()
        
        assert result is False
        assert self.analytics.db_connection is None
    
    def test_create_metrics_table(self):
        """Test metrics table creation"""
        mock_connection = MagicMock()
        self.analytics.db_connection = mock_connection
        self.analytics.is_connected = True
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            result = self.analytics.create_metrics_table()
        
        assert result is True
        mock_connection.execute.assert_called()
    
    def test_insert_metric_data(self):
        """Test inserting metric data"""
        mock_connection = MagicMock()
        self.analytics.db_connection = mock_connection
        self.analytics.is_connected = True
        
        metric_data = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "metric_name": "api_requests",
                "value": 100.0,
                "tags": {"endpoint": "/health"}
            }
        ]
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            result = self.analytics.insert_metric_data(metric_data)
        
        assert result is True
        mock_connection.execute.assert_called()
    
    def test_query_metrics_by_time_range(self):
        """Test querying metrics by time range"""
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("2025-01-15T10:00:00Z", "api_requests", 100.0, "{}"),
            ("2025-01-15T11:00:00Z", "api_requests", 150.0, "{}")
        ]
        mock_connection.execute.return_value = mock_result
        
        self.analytics.db_connection = mock_connection
        self.analytics.is_connected = True
        
        start_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            results = self.analytics.query_metrics_by_time_range(
                "api_requests", start_time, end_time
            )
        
        assert len(results) == 2
        assert results[0][1] == "api_requests"
        assert results[0][2] == 100.0
    
    def test_get_aggregated_metrics(self):
        """Test getting aggregated metrics"""
        mock_connection = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (250.0, 125.0, 100.0, 150.0, 2)
        mock_connection.execute.return_value = mock_result
        
        self.analytics.db_connection = mock_connection
        self.analytics.is_connected = True
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            aggregation = self.analytics.get_aggregated_metrics("api_requests", "1h")
        
        assert aggregation["sum"] == 250.0
        assert aggregation["avg"] == 125.0
        assert aggregation["min"] == 100.0
        assert aggregation["max"] == 150.0
        assert aggregation["count"] == 2


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestDuckDBAnalyticsAdvanced:
    """Test advanced DuckDB analytics operations"""
    
    def setup_method(self):
        """Setup test environment"""
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.analytics = DuckDBAnalytics(test_mode=True)
        
        self.mock_connection = MagicMock()
        self.analytics.db_connection = self.mock_connection
        self.analytics.is_connected = True
    
    def test_create_time_series_analysis(self):
        """Test time series analysis creation"""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("2025-01-15T10:00:00Z", 100.0),
            ("2025-01-15T11:00:00Z", 120.0),
            ("2025-01-15T12:00:00Z", 110.0)
        ]
        self.mock_connection.execute.return_value = mock_result
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            results = self.analytics.create_time_series_analysis(
                "api_requests", "1h", hours_back=3
            )
        
        assert len(results) == 3
        assert results[0][0] == "2025-01-15T10:00:00Z"
        assert results[0][1] == 100.0
    
    def test_detect_anomalies(self):
        """Test anomaly detection"""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("2025-01-15T10:00:00Z", 1000.0, 100.0, 3.16),  # Anomaly: z-score > 3
            ("2025-01-15T11:00:00Z", 120.0, 100.0, 0.63),   # Normal
        ]
        self.mock_connection.execute.return_value = mock_result
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            anomalies = self.analytics.detect_anomalies("api_requests", threshold=2.0)
        
        assert len(anomalies) >= 1
        # First result should be the anomaly
        anomaly = anomalies[0] if anomalies else None
        assert anomaly is not None
    
    def test_generate_performance_report(self):
        """Test performance report generation"""
        # Mock multiple query results for different metrics
        mock_results = [
            MagicMock(),  # API requests
            MagicMock(),  # Response times
            MagicMock(),  # Error rates
        ]
        
        mock_results[0].fetchall.return_value = [("api_requests", 1000.0, 50.0, 10.0, 200.0)]
        mock_results[1].fetchall.return_value = [("response_time", 150.0, 75.0, 50.0, 300.0)]
        mock_results[2].fetchall.return_value = [("error_rate", 5.0, 2.5, 0.0, 10.0)]
        
        self.mock_connection.execute.side_effect = mock_results
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            report = self.analytics.generate_performance_report(hours_back=24)
        
        assert "summary" in report
        assert "metrics" in report
        assert len(report["metrics"]) >= 1
    
    def test_cleanup_old_data(self):
        """Test cleanup of old data"""
        self.mock_connection.execute.return_value.rowcount = 150
        
        with patch.object(self.analytics, 'ensure_connected', return_value=True):
            deleted_count = self.analytics.cleanup_old_data(days_to_keep=7)
        
        assert deleted_count == 150
        self.mock_connection.execute.assert_called()
    
    def test_export_metrics_csv(self):
        """Test exporting metrics to CSV"""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("2025-01-15T10:00:00Z", "api_requests", 100.0, "{}"),
            ("2025-01-15T11:00:00Z", "api_requests", 150.0, "{}")
        ]
        self.mock_connection.execute.return_value = mock_result
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            output_path = tmp_file.name
        
        try:
            with patch.object(self.analytics, 'ensure_connected', return_value=True):
                result = self.analytics.export_metrics_csv(
                    "api_requests", output_path, hours_back=2
                )
            
            assert result is True
            
            # Verify CSV file was created and has content
            with open(output_path, 'r') as f:
                content = f.read()
                assert "timestamp" in content
                assert "metric_name" in content
                assert "api_requests" in content
        finally:
            os.unlink(output_path)


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestContextKV:
    """Test unified ContextKV interface"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "database": 0
            },
            "duckdb": {
                "database_path": ":memory:"
            }
        }
        
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.context_kv = ContextKV(config=self.test_config, test_mode=True)
    
    def test_context_kv_initialization(self):
        """Test ContextKV initialization"""
        assert self.context_kv is not None
        assert hasattr(self.context_kv, 'redis')
        assert hasattr(self.context_kv, 'analytics')
        assert self.context_kv.test_mode is True
    
    def test_initialize_connections(self):
        """Test connection initialization"""
        with patch.object(self.context_kv.redis, 'connect', return_value=True) as mock_redis_connect:
            with patch.object(self.context_kv.analytics, 'connect', return_value=True) as mock_analytics_connect:
                result = self.context_kv.initialize()
        
        assert result is True
        mock_redis_connect.assert_called_once()
        mock_analytics_connect.assert_called_once()
    
    def test_initialize_partial_failure(self):
        """Test initialization with partial connection failure"""
        with patch.object(self.context_kv.redis, 'connect', return_value=True):
            with patch.object(self.context_kv.analytics, 'connect', return_value=False):
                result = self.context_kv.initialize()
        
        # Should still return True if Redis connects (analytics is optional)
        assert result is True
    
    def test_store_context_data(self):
        """Test storing context data"""
        context_data = {
            "agent_id": "agent123",
            "session_id": "session456",
            "context": {"key": "value", "number": 42}
        }
        
        with patch.object(self.context_kv.redis, 'set_cache', return_value=True) as mock_set_cache:
            result = self.context_kv.store_context("context_key", context_data, ttl=3600)
        
        assert result is True
        mock_set_cache.assert_called_once_with("context_key", context_data, 3600)
    
    def test_retrieve_context_data(self):
        """Test retrieving context data"""
        expected_data = {
            "agent_id": "agent123",
            "context": {"key": "value"}
        }
        
        with patch.object(self.context_kv.redis, 'get_cache', return_value=expected_data) as mock_get_cache:
            result = self.context_kv.retrieve_context("context_key")
        
        assert result == expected_data
        mock_get_cache.assert_called_once_with("context_key")
    
    def test_delete_context_data(self):
        """Test deleting context data"""
        with patch.object(self.context_kv.redis, 'delete_cache', return_value=True) as mock_delete_cache:
            result = self.context_kv.delete_context("context_key")
        
        assert result is True
        mock_delete_cache.assert_called_once_with("context_key")
    
    def test_record_context_metric(self):
        """Test recording context metrics"""
        with patch.object(self.context_kv.redis, 'record_metric', return_value=True) as mock_record_metric:
            metric = MetricEvent(
                timestamp=datetime.now(timezone.utc),
                metric_name="context_operations",
                value=1.0,
                tags={"operation": "store", "agent_id": "agent123"}
            )
            
            result = self.context_kv.record_metric(metric)
        
        assert result is True
        mock_record_metric.assert_called_once_with(metric)
    
    def test_get_context_analytics(self):
        """Test getting context analytics"""
        expected_analytics = {
            "total_contexts": 100,
            "active_sessions": 25,
            "avg_context_size": 1024
        }
        
        with patch.object(self.context_kv.analytics, 'get_aggregated_metrics', return_value=expected_analytics):
            result = self.context_kv.get_analytics_summary("1h")
        
        assert result == expected_analytics
    
    def test_cleanup_expired_contexts(self):
        """Test cleanup of expired contexts"""
        expired_keys = ["expired_key1", "expired_key2", "expired_key3"]
        
        with patch.object(self.context_kv.redis, 'list_cache_keys', return_value=expired_keys):
            with patch.object(self.context_kv.redis, 'get_cache_ttl', return_value=-1):  # Expired
                with patch.object(self.context_kv.redis, 'delete_cache', return_value=True) as mock_delete:
                    cleaned_count = self.context_kv.cleanup_expired_contexts()
        
        assert cleaned_count == len(expired_keys)
        assert mock_delete.call_count == len(expired_keys)


@pytest.mark.skipif(not KV_STORE_AVAILABLE, reason="KV store not available")
class TestContextKVIntegrationScenarios:
    """Test ContextKV integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        with patch('src.storage.kv_store.yaml.safe_load'):
            self.context_kv = ContextKV(test_mode=True)
    
    def test_complete_context_lifecycle(self):
        """Test complete context lifecycle"""
        # Mock all dependencies
        with patch.object(self.context_kv.redis, 'connect', return_value=True):
            with patch.object(self.context_kv.analytics, 'connect', return_value=True):
                with patch.object(self.context_kv.redis, 'set_cache', return_value=True):
                    with patch.object(self.context_kv.redis, 'get_cache') as mock_get:
                        with patch.object(self.context_kv.redis, 'delete_cache', return_value=True):
                            
                            # Initialize
                            init_result = self.context_kv.initialize()
                            assert init_result is True
                            
                            # Store context
                            context_data = {"agent_id": "agent123", "data": "test"}
                            store_result = self.context_kv.store_context("test_key", context_data)
                            assert store_result is True
                            
                            # Retrieve context
                            mock_get.return_value = context_data
                            retrieved_data = self.context_kv.retrieve_context("test_key")
                            assert retrieved_data == context_data
                            
                            # Delete context
                            delete_result = self.context_kv.delete_context("test_key")
                            assert delete_result is True
    
    def test_high_throughput_operations(self):
        """Test high throughput operations"""
        # Mock successful operations
        with patch.object(self.context_kv.redis, 'set_cache', return_value=True):
            with patch.object(self.context_kv.redis, 'get_cache', return_value={"data": "test"}):
                
                # Simulate high throughput
                operations_count = 100
                successful_operations = 0
                
                for i in range(operations_count):
                    # Store operation
                    store_result = self.context_kv.store_context(f"key_{i}", {"data": f"value_{i}"})
                    if store_result:
                        successful_operations += 1
                    
                    # Retrieve operation
                    retrieve_result = self.context_kv.retrieve_context(f"key_{i}")
                    if retrieve_result:
                        successful_operations += 1
                
                # Should handle all operations successfully
                assert successful_operations == operations_count * 2
    
    def test_error_recovery_scenarios(self):
        """Test error recovery scenarios"""
        # Test Redis connection failure recovery
        with patch.object(self.context_kv.redis, 'connect', side_effect=[False, True]):
            with patch.object(self.context_kv.analytics, 'connect', return_value=True):
                
                # First attempt fails
                result1 = self.context_kv.initialize()
                
                # Retry should succeed
                result2 = self.context_kv.initialize()
                assert result2 is True
    
    def test_metrics_integration(self):
        """Test metrics integration with analytics"""
        metrics_data = []
        
        def mock_record_metric(metric):
            metrics_data.append(metric)
            return True
        
        with patch.object(self.context_kv.redis, 'record_metric', side_effect=mock_record_metric):
            with patch.object(self.context_kv.analytics, 'insert_metric_data', return_value=True):
                
                # Record multiple metrics
                for i in range(5):
                    metric = MetricEvent(
                        timestamp=datetime.now(timezone.utc),
                        metric_name="test_metric",
                        value=float(i),
                        tags={"iteration": str(i)}
                    )
                    
                    result = self.context_kv.record_metric(metric)
                    assert result is True
                
                # Verify metrics were recorded
                assert len(metrics_data) == 5
                assert all(m.metric_name == "test_metric" for m in metrics_data)
    
    def test_concurrent_access_simulation(self):
        """Test concurrent access simulation"""
        # Mock thread-safe operations
        with patch.object(self.context_kv.redis, 'set_cache', return_value=True):
            with patch.object(self.context_kv.redis, 'get_cache') as mock_get:
                
                # Simulate concurrent access patterns
                context_data = {"shared_data": "value", "access_count": 0}
                mock_get.return_value = context_data
                
                # Multiple "threads" accessing same context
                for thread_id in range(10):
                    # Read
                    data = self.context_kv.retrieve_context("shared_context")
                    assert data == context_data
                    
                    # Modify
                    data["access_count"] = thread_id
                    
                    # Write back
                    result = self.context_kv.store_context("shared_context", data)
                    assert result is True
                    
                    # Update mock for next iteration
                    context_data["access_count"] = thread_id
                    mock_get.return_value = context_data
    
    def test_data_persistence_verification(self):
        """Test data persistence verification"""
        # Mock persistent storage
        storage_data = {}
        
        def mock_set_cache(key, value, ttl=None):
            storage_data[key] = {"value": value, "ttl": ttl}
            return True
        
        def mock_get_cache(key):
            return storage_data.get(key, {}).get("value")
        
        with patch.object(self.context_kv.redis, 'set_cache', side_effect=mock_set_cache):
            with patch.object(self.context_kv.redis, 'get_cache', side_effect=mock_get_cache):
                
                # Store data
                test_data = {"persistent": True, "value": 12345}
                self.context_kv.store_context("persistent_key", test_data, ttl=7200)
                
                # Verify storage
                assert "persistent_key" in storage_data
                assert storage_data["persistent_key"]["value"] == test_data
                assert storage_data["persistent_key"]["ttl"] == 7200
                
                # Retrieve and verify
                retrieved_data = self.context_kv.retrieve_context("persistent_key")
                assert retrieved_data == test_data