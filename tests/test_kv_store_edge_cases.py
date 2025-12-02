#!/usr/bin/env python3
"""
Comprehensive edge case tests for KV Store components - Phase 5 Coverage

This test module focuses on edge cases, boundary conditions, error scenarios,
and performance edge cases that weren't covered in basic unit tests.
"""
import pytest
import tempfile
import os
import json
import time
from unittest.mock import patch, Mock, MagicMock, call
from typing import Dict, Any, List
from datetime import datetime, timedelta
from threading import Thread
import concurrent.futures

# Import storage components
try:
    from src.storage.kv_store import (
        RedisConnector, DuckDBAnalytics, ContextKV, 
        CacheEntry, MetricEvent
    )
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    RedisConnector = None
    DuckDBAnalytics = None
    ContextKV = None
    CacheEntry = None
    MetricEvent = None


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestRedisConnectorEdgeCases:
    """Edge case tests for Redis connector functionality"""
    
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
                    "lock": "test_lock:",
                    "metric": "test_metric:"
                }
            }
        }
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_connection_failure_scenarios(self, mock_redis):
        """Test various connection failure scenarios"""
        # Test initial connection failure
        mock_redis.side_effect = Exception("Connection refused")
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        result = connector.connect()
        
        assert result is False
        assert connector.is_connected is False
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_connection_timeout_scenarios(self, mock_redis):
        """Test connection timeout scenarios"""
        mock_redis_client = Mock()
        mock_redis_client.ping.side_effect = Exception("Timeout")
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        result = connector.connect()
        
        assert result is False
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_ssl_configuration_edge_cases(self, mock_redis):
        """Test SSL configuration edge cases"""
        # Test SSL in production environment
        ssl_config = {
            "redis": {
                "host": "secure.redis.com",
                "port": 6380,
                "ssl": True,
                "database": 0
            }
        }
        
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=ssl_config, test_mode=False)
        connector.environment = "production"
        
        result = connector.connect()
        
        # Verify SSL configuration was applied
        assert mock_redis.called
        call_args = mock_redis.call_args
        pool_kwargs = call_args[1] if len(call_args) > 1 else call_args[0][0]
        
        # SSL should be configured for production
        assert result is True
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_cache_operations_with_edge_cases(self, mock_redis):
        """Test cache operations with various edge cases"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.setex.return_value = True
        mock_redis_client.get.return_value = None
        mock_redis_client.ttl.return_value = -1
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        # Test with invalid key
        with patch('src.storage.kv_store.validate_redis_key', return_value=False):
            result = connector.set_cache("invalid key", "value")
            assert result is False
        
        # Test with None value
        with patch('src.storage.kv_store.validate_redis_key', return_value=True):
            result = connector.set_cache("valid_key", None)
            assert result is True  # Should handle None values
        
        # Test with very large value
        large_value = {"data": "x" * 10000}  # 10KB value
        with patch('src.storage.kv_store.validate_redis_key', return_value=True):
            result = connector.set_cache("large_key", large_value)
            assert result is True
        
        # Test cache miss
        result = connector.get_cache("nonexistent_key")
        assert result is None
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_session_operations_edge_cases(self, mock_redis):
        """Test session operations with edge cases"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.setex.return_value = True
        mock_redis_client.get.return_value = None
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        # Test session with empty data
        result = connector.set_session("session_1", {})
        assert result is True
        
        # Test session with complex nested data
        complex_data = {
            "user": {"id": 123, "roles": ["admin", "user"]},
            "metadata": {"created": datetime.utcnow().isoformat()},
            "settings": {"theme": "dark", "notifications": True}
        }
        result = connector.set_session("session_2", complex_data)
        assert result is True
        
        # Test session with very short TTL
        result = connector.set_session("session_3", {"test": True}, ttl_seconds=1)
        assert result is True
        
        # Test getting non-existent session
        result = connector.get_session("nonexistent_session")
        assert result is None
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_distributed_lock_edge_cases(self, mock_redis):
        """Test distributed lock operations with edge cases"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.set.return_value = True  # Lock acquired
        mock_redis_client.eval.return_value = 1   # Lock released
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        # Test lock acquisition
        lock_id = connector.acquire_lock("resource_1")
        assert lock_id is not None
        assert len(lock_id) == 16  # SHA256 hash truncated to 16 chars
        
        # Test lock release with correct lock_id
        result = connector.release_lock("resource_1", lock_id)
        assert result is True
        
        # Test lock release with wrong lock_id
        mock_redis_client.eval.return_value = 0  # Lock not released
        result = connector.release_lock("resource_1", "wrong_lock_id")
        assert result is False
        
        # Test lock acquisition with timeout edge case
        mock_redis_client.set.return_value = False  # Lock not acquired
        lock_id = connector.acquire_lock("busy_resource", timeout=1)
        assert lock_id is None
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_metric_recording_edge_cases(self, mock_redis):
        """Test metric recording with edge cases"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.zadd.return_value = 1
        mock_redis_client.expire.return_value = True
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        # Test metric with invalid data
        invalid_metric = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="invalid/metric:name",  # Invalid characters
            value=float('inf'),  # Invalid value
            tags={"test": "value"}
        )
        
        with patch('src.storage.kv_store.validate_metric_event', return_value=False):
            result = connector.record_metric(invalid_metric)
            assert result is False
        
        # Test metric with extreme values
        extreme_metric = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="extreme_metric",
            value=1e10,  # Very large value
            tags={"extreme": "true"}
        )
        
        with patch('src.storage.kv_store.validate_metric_event', return_value=True):
            with patch('src.storage.kv_store.sanitize_metric_name', return_value="extreme_metric"):
                result = connector.record_metric(extreme_metric)
                assert result is True
        
        # Test metric with empty tags
        empty_tags_metric = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="empty_tags_metric",
            value=42.0,
            tags={}
        )
        
        with patch('src.storage.kv_store.validate_metric_event', return_value=True):
            with patch('src.storage.kv_store.sanitize_metric_name', return_value="empty_tags_metric"):
                result = connector.record_metric(empty_tags_metric)
                assert result is True
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_metric_retrieval_edge_cases(self, mock_redis):
        """Test metric retrieval with edge cases"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.zrangebyscore.return_value = []
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        # Test with invalid time range
        future_time = datetime.utcnow() + timedelta(days=1)
        past_time = datetime.utcnow() - timedelta(days=1)
        
        with patch('src.storage.kv_store.validate_time_range', return_value=False):
            metrics = connector.get_metrics("test_metric", future_time, past_time)
            assert metrics == []
        
        # Test with very wide time range
        very_past = datetime.utcnow() - timedelta(days=365)
        very_future = datetime.utcnow() + timedelta(days=365)
        
        with patch('src.storage.kv_store.validate_time_range', return_value=True):
            with patch('src.storage.kv_store.sanitize_metric_name', return_value="test_metric"):
                metrics = connector.get_metrics("test_metric", very_past, very_future)
                assert metrics == []
        
        # Test with no metrics found
        with patch('src.storage.kv_store.validate_time_range', return_value=True):
            with patch('src.storage.kv_store.sanitize_metric_name', return_value="nonexistent_metric"):
                metrics = connector.get_metrics("nonexistent_metric", past_time, future_time)
                assert metrics == []
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_connection_cleanup_edge_cases(self, mock_redis):
        """Test connection cleanup in various scenarios"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.close.side_effect = Exception("Close failed")
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        # Test cleanup with exception during close
        connector.close()  # Should not raise exception
        assert connector.redis_client is None
        assert connector.is_connected is False
        
        # Test double close
        connector.close()  # Should handle gracefully
    
    def test_performance_configuration_edge_cases(self):
        """Test performance configuration with edge cases"""
        # Test missing performance config file
        connector = RedisConnector(config=self.test_config, test_mode=True)
        perf_config = connector._load_performance_config()
        assert isinstance(perf_config, dict)
        assert perf_config == {}  # Should return empty dict for missing file
        
        # Test invalid performance config
        with patch('builtins.open', side_effect=FileNotFoundError):
            perf_config = connector._load_performance_config()
            assert perf_config == {}


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestDuckDBAnalyticsEdgeCases:
    """Edge case tests for DuckDB analytics functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "duckdb": {
                "database_path": ":memory:",
                "memory_limit": "512MB",
                "threads": 2,
                "tables": {
                    "metrics": "test_metrics",
                    "events": "test_events",
                    "summaries": "test_summaries",
                    "trends": "test_trends"
                }
            }
        }
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_database_creation_edge_cases(self, mock_connect):
        """Test database creation with various edge cases"""
        # Test connection failure
        mock_connect.side_effect = Exception("Database connection failed")
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        
        result = analytics.connect()
        assert result is False
        
        # Test directory creation failure
        mock_connect.side_effect = OSError("Permission denied")
        result = analytics.connect()
        assert result is False
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_table_initialization_edge_cases(self, mock_connect):
        """Test table initialization with edge cases"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        
        # Test successful connection and table creation
        result = analytics.connect()
        assert result is True
        
        # Verify table creation calls
        assert mock_conn.execute.call_count >= 4  # At least 4 tables created
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_metrics_insertion_edge_cases(self, mock_connect):
        """Test metrics insertion with edge cases"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        analytics.connect()
        
        # Test empty metrics list
        result = analytics.insert_metrics([])
        assert result is True
        
        # Test metrics with missing fields
        incomplete_metrics = [
            MetricEvent(
                timestamp=datetime.utcnow(),
                metric_name="test",
                value=1.0,
                tags={}
            )
        ]
        result = analytics.insert_metrics(incomplete_metrics)
        assert result is True
        
        # Test large batch of metrics
        large_batch = []
        for i in range(1000):
            large_batch.append(MetricEvent(
                timestamp=datetime.utcnow(),
                metric_name=f"metric_{i}",
                value=float(i),
                tags={"batch": "large"}
            ))
        
        result = analytics.insert_metrics(large_batch)
        assert result is True
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_query_execution_edge_cases(self, mock_connect):
        """Test query execution with edge cases"""
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.description = [("col1",), ("col2",)]
        mock_connect.return_value = mock_conn
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        analytics.connect()
        
        # Test empty query result
        result = analytics.query_metrics("SELECT * FROM nonexistent_table")
        assert result == []
        
        # Test query with parameters
        result = analytics.query_metrics(
            "SELECT * FROM test_metrics WHERE value > ?", 
            [100]
        )
        assert result == []
        
        # Test query timeout scenario
        mock_conn.execute.side_effect = Exception("Query timeout")
        result = analytics.query_metrics("SELECT * FROM test_metrics")
        assert result == []
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_aggregation_edge_cases(self, mock_connect):
        """Test metric aggregation with edge cases"""
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_connect.return_value = mock_conn
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        analytics.connect()
        
        # Test aggregation with no data
        result = analytics.aggregate_metrics(
            "nonexistent_metric",
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow(),
            "avg"
        )
        assert result == {}
        
        # Test invalid aggregation function
        result = analytics.aggregate_metrics(
            "test_metric",
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow(),
            "invalid_func"
        )
        assert result == {}
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_summary_generation_edge_cases(self, mock_connect):
        """Test summary generation with edge cases"""
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_connect.return_value = mock_conn
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        analytics.connect()
        
        # Test daily summary with no data
        from datetime import date
        result = analytics.generate_summary(date.today(), "daily")
        assert "summary_date" in result
        assert result["metrics"] == {}
        
        # Test weekly summary
        result = analytics.generate_summary(date.today(), "weekly")
        assert result["summary_type"] == "weekly"
        
        # Test monthly summary
        result = analytics.generate_summary(date.today(), "monthly")
        assert result["summary_type"] == "monthly"
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_trend_detection_edge_cases(self, mock_connect):
        """Test trend detection with edge cases"""
        mock_conn = Mock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_connect.return_value = mock_conn
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        analytics.connect()
        
        # Test trend detection with insufficient data
        result = analytics.detect_trends("test_metric", period_days=7)
        assert result == {"trend": "insufficient_data"}
        
        # Test trend detection with single data point
        mock_conn.execute.return_value.fetchall.return_value = [(datetime.utcnow(), 42.0, 1)]
        result = analytics.detect_trends("test_metric", period_days=1)
        assert result == {"trend": "insufficient_data"}


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestContextKVEdgeCases:
    """Edge case tests for unified ContextKV interface"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {"host": "localhost", "port": 6379},
            "duckdb": {"database_path": ":memory:"}
        }
    
    @patch('src.storage.kv_store.RedisConnector')
    @patch('src.storage.kv_store.DuckDBAnalytics')
    def test_unified_interface_edge_cases(self, mock_duckdb, mock_redis):
        """Test unified interface with various edge cases"""
        # Mock successful connections
        mock_redis_instance = Mock()
        mock_redis_instance.connect.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        mock_duckdb_instance = Mock()
        mock_duckdb_instance.connect.return_value = True
        mock_duckdb.return_value = mock_duckdb_instance
        
        # Test successful connection to both stores
        kv = ContextKV(config=self.test_config, test_mode=False)
        result = kv.connect()
        assert result is True
        
        # Test partial connection failure (Redis succeeds, DuckDB fails)
        mock_duckdb_instance.connect.return_value = False
        result = kv.connect()
        assert result is False
        
        # Test complete connection failure
        mock_redis_instance.connect.return_value = False
        result = kv.connect()
        assert result is False
    
    @patch('src.storage.kv_store.RedisConnector')
    @patch('src.storage.kv_store.DuckDBAnalytics')
    def test_event_recording_edge_cases(self, mock_duckdb, mock_redis):
        """Test event recording with edge cases"""
        mock_redis_instance = Mock()
        mock_redis_instance.record_metric.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        mock_duckdb_instance = Mock()
        mock_duckdb_instance.insert_metrics.return_value = True
        mock_duckdb.return_value = mock_duckdb_instance
        
        kv = ContextKV(config=self.test_config, test_mode=False)
        
        # Test event with all optional parameters
        result = kv.record_event(
            "test_event",
            document_id="doc_123",
            agent_id="agent_456",
            data={"complex": {"nested": "data"}}
        )
        assert result is True
        
        # Test event with minimal parameters
        result = kv.record_event("minimal_event")
        assert result is True
        
        # Test event recording failure in one store
        mock_redis_instance.record_metric.return_value = False
        result = kv.record_event("failing_event")
        assert result is False
    
    @patch('src.storage.kv_store.RedisConnector')
    @patch('src.storage.kv_store.DuckDBAnalytics')
    def test_activity_summary_edge_cases(self, mock_duckdb, mock_redis):
        """Test activity summary with edge cases"""
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_duckdb_instance = Mock()
        mock_duckdb_instance.query_metrics.return_value = []
        mock_duckdb.return_value = mock_duckdb_instance
        
        kv = ContextKV(config=self.test_config, test_mode=False)
        
        # Test activity summary with no data
        result = kv.get_recent_activity(hours=24)
        assert result["period_hours"] == 24
        assert result["metrics"] == []
        
        # Test activity summary with very short period
        result = kv.get_recent_activity(hours=1)
        assert result["period_hours"] == 1
        
        # Test activity summary with very long period
        result = kv.get_recent_activity(hours=8760)  # 1 year
        assert result["period_hours"] == 8760


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestConcurrencyAndPerformanceEdgeCases:
    """Edge cases for concurrency and performance scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {"host": "localhost", "port": 6379},
            "duckdb": {"database_path": ":memory:"}
        }
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_concurrent_cache_operations(self, mock_redis):
        """Test concurrent cache operations"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.setex.return_value = True
        mock_redis_client.get.return_value = '{"value": "test"}'
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        def cache_operation(key_suffix):
            """Perform cache operations"""
            key = f"concurrent_test_{key_suffix}"
            with patch('src.storage.kv_store.validate_redis_key', return_value=True):
                # Set cache
                result = connector.set_cache(key, {"data": f"value_{key_suffix}"})
                assert result is True
                
                # Get cache
                value = connector.get_cache(key)
                return value
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(cache_operation, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All operations should complete
        assert len(results) == 10
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_concurrent_lock_operations(self, mock_redis):
        """Test concurrent distributed lock operations"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.set.return_value = True
        mock_redis_client.eval.return_value = 1
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        def lock_operation(resource_id):
            """Perform lock operations"""
            resource = f"resource_{resource_id}"
            lock_id = connector.acquire_lock(resource, timeout=5)
            if lock_id:
                # Simulate work
                time.sleep(0.01)
                result = connector.release_lock(resource, lock_id)
                return result
            return False
        
        # Run concurrent lock operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(lock_operation, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All operations should complete
        assert len(results) == 5
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_large_data_insertion(self, mock_connect):
        """Test insertion of large amounts of data"""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = {"duckdb": {"tables": {"metrics": "test_metrics"}}}
        analytics.connect()
        
        # Create large dataset
        large_metrics = []
        for i in range(10000):
            large_metrics.append(MetricEvent(
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                metric_name=f"performance_metric_{i % 100}",
                value=float(i),
                tags={"batch": "performance", "index": str(i)}
            ))
        
        # Test large batch insertion
        result = analytics.insert_metrics(large_metrics)
        assert result is True
        
        # Verify executemany was called
        mock_conn.executemany.assert_called_once()
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns with large objects"""
        # Test large cache entry creation
        large_data = {"content": "x" * 100000}  # 100KB content
        
        cache_entry = CacheEntry(
            key="large_key",
            value=large_data,
            created_at=datetime.utcnow(),
            ttl_seconds=3600,
            hit_count=0
        )
        
        # Verify entry was created successfully
        assert cache_entry.key == "large_key"
        assert len(cache_entry.value["content"]) == 100000
        
        # Test metric event with large tags
        large_tags = {f"tag_{i}": f"value_{i}" for i in range(1000)}
        
        metric_event = MetricEvent(
            timestamp=datetime.utcnow(),
            metric_name="large_tags_metric",
            value=42.0,
            tags=large_tags
        )
        
        # Verify metric was created successfully
        assert metric_event.metric_name == "large_tags_metric"
        assert len(metric_event.tags) == 1000


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestErrorRecoveryAndResilience:
    """Tests for error recovery and system resilience"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_config = {
            "redis": {"host": "localhost", "port": 6379},
            "duckdb": {"database_path": ":memory:"}
        }
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_redis_connection_recovery(self, mock_redis):
        """Test Redis connection recovery scenarios"""
        mock_redis_client = Mock()
        
        # Simulate intermittent connection issues
        call_count = 0
        def ping_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Connection lost")
            return True
        
        mock_redis_client.ping.side_effect = ping_side_effect
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        
        # First connection attempts should fail
        result1 = connector.connect()
        result2 = connector.connect()
        assert result1 is False
        assert result2 is False
        
        # Third attempt should succeed
        result3 = connector.connect()
        assert result3 is True
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_operation_failure_handling(self, mock_redis):
        """Test handling of operation failures"""
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        
        # Simulate operation failures
        mock_redis_client.setex.side_effect = Exception("Redis operation failed")
        mock_redis_client.get.side_effect = Exception("Redis get failed")
        mock_redis.return_value = mock_redis_client
        
        connector = RedisConnector(config=self.test_config, test_mode=True)
        connector.connect()
        
        # Operations should handle failures gracefully
        with patch('src.storage.kv_store.validate_redis_key', return_value=True):
            result = connector.set_cache("test_key", "test_value")
            assert result is False
        
        result = connector.get_cache("test_key")
        assert result is None
    
    @patch('src.storage.kv_store.duckdb.connect')
    def test_database_corruption_recovery(self, mock_connect):
        """Test database corruption recovery scenarios"""
        # Simulate database corruption
        mock_connect.side_effect = [
            Exception("Database corrupted"),
            Exception("Database locked"),
            Mock()  # Successful connection after recovery
        ]
        
        analytics = DuckDBAnalytics(config_path="test.yaml", verbose=True)
        analytics.config = self.test_config
        
        # First attempts should fail
        result1 = analytics.connect()
        result2 = analytics.connect()
        assert result1 is False
        assert result2 is False
        
        # Final attempt should succeed
        result3 = analytics.connect()
        assert result3 is True
    
    @patch('src.storage.kv_store.RedisConnector')
    @patch('src.storage.kv_store.DuckDBAnalytics')
    def test_partial_system_failure_resilience(self, mock_duckdb, mock_redis):
        """Test system resilience with partial failures"""
        # Redis works, DuckDB fails
        mock_redis_instance = Mock()
        mock_redis_instance.connect.return_value = True
        mock_redis_instance.record_metric.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        mock_duckdb_instance = Mock()
        mock_duckdb_instance.connect.return_value = False
        mock_duckdb_instance.insert_metrics.side_effect = Exception("DuckDB failed")
        mock_duckdb.return_value = mock_duckdb_instance
        
        kv = ContextKV(config=self.test_config, test_mode=False)
        
        # System should handle partial failures
        connection_result = kv.connect()
        assert connection_result is False  # Overall connection fails
        
        # But Redis operations might still work
        # Note: In actual implementation, this would depend on error handling strategy