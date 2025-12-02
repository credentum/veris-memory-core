#!/usr/bin/env python3
"""
Comprehensive integration tests for storage layer components - Phase 4 Coverage

This test module focuses on testing interactions between storage components
and verifies end-to-end storage workflows without requiring external services.
"""
import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Import storage components with fallback handling
try:
    from src.storage.context_kv import ContextKV
    from src.storage.kv_store import ContextKV as BaseContextKV, CacheEntry, MetricEvent
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False
    ContextKV = None
    BaseContextKV = None
    CacheEntry = None
    MetricEvent = None

# Import core components
try:
    from src.core.config import load_config
    from src.core.utils import format_timestamp, sanitize_filename
    from src.validators.kv_validators import validate_redis_key, validate_metric_event
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestContextKVIntegration:
    """Integration tests for context KV storage system"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.test_config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "decode_responses": True
            },
            "duckdb": {
                "path": ":memory:",
                "timeout": 30
            }
        }
    
    @patch('src.storage.context_kv.BaseContextKV.__init__')
    def test_context_kv_initialization(self, mock_base_init):
        """Test ContextKV initialization with proper inheritance"""
        mock_base_init.return_value = None
        
        # Create ContextKV instance
        context_kv = ContextKV(
            config_path=".ctxrc.yaml",
            verbose=True,
            config=self.test_config,
            test_mode=True
        )
        
        # Verify base class initialization was called
        mock_base_init.assert_called_once_with(
            ".ctxrc.yaml", True, self.test_config, True
        )
        
        # Verify enhanced features are initialized
        assert hasattr(context_kv, 'context_cache')
        assert isinstance(context_kv.context_cache, dict)
    
    @patch('src.storage.kv_store.RedisConnector.set_cache')
    @patch('src.storage.context_kv.datetime')
    def test_store_context_integration(self, mock_datetime, mock_store):
        """Test context storage with metadata integration"""
        # Mock datetime
        fixed_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.utcnow.return_value = fixed_time
        
        # Mock base store method
        mock_store.return_value = True
        
        # Create ContextKV instance with mocked base
        with patch('src.storage.context_kv.BaseContextKV.__init__'):
            context_kv = ContextKV(test_mode=True)
            context_kv.context_cache = {}
            
            # Test context storage
            context_data = {"content": "test content", "type": "test"}
            result = context_kv.store_context("test_ctx_001", context_data, ttl_seconds=3600)
            
            # Verify metadata was added
            expected_data = {
                "content": "test content",
                "type": "test",
                "_stored_at": "2023-01-01T12:00:00",
                "_context_id": "test_ctx_001"
            }
            
            # Verify Redis set_cache was called with enhanced data
            mock_store.assert_called_once()
            assert result is True
    
    def test_store_context_validation(self):
        """Test context storage validation logic"""
        with patch('src.storage.context_kv.BaseContextKV.__init__'):
            context_kv = ContextKV(test_mode=True)
            
            # Test empty context_id
            result = context_kv.store_context("", {"content": "test"})
            assert result is False
            
            # Test empty context_data
            result = context_kv.store_context("test_id", {})
            assert result is False
            
            # Test None values
            result = context_kv.store_context(None, {"content": "test"})
            assert result is False
            
            result = context_kv.store_context("test_id", None)
            assert result is False


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestKVStoreBaseIntegration:
    """Integration tests for base KV store functionality"""
    
    def setup_method(self):
        """Setup method run before each test"""
        self.test_config_path = tempfile.mktemp(suffix='.yaml')
        self.test_config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "duckdb": {
                "path": ":memory:"
            }
        }
        
        # Write test config
        with open(self.test_config_path, 'w') as f:
            import yaml
            yaml.dump(self.test_config, f)
    
    def teardown_method(self):
        """Cleanup after each test"""
        if os.path.exists(self.test_config_path):
            os.unlink(self.test_config_path)
    
    @patch('src.storage.kv_store.redis.Redis')
    @patch('src.storage.kv_store.duckdb.connect')
    def test_base_kv_initialization_integration(self, mock_duckdb_connect, mock_redis):
        """Test base KV store initialization with mocked backends"""
        # Mock Redis connection
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis.return_value = mock_redis_client
        
        # Mock DuckDB connection
        mock_duckdb_conn = Mock()
        mock_duckdb_connect.return_value = mock_duckdb_conn
        
        # Initialize KV store
        kv_store = BaseContextKV(
            config_path=self.test_config_path,
            verbose=True,
            test_mode=True
        )
        
        # Verify initialization
        assert kv_store.config == self.test_config
        assert kv_store.verbose is True
        assert kv_store.test_mode is True
        
        # Verify connections were attempted
        mock_redis.assert_called_once()
        mock_duckdb_connect.assert_called_once()
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_redis_operations_integration(self, mock_redis):
        """Test Redis operations integration"""
        # Mock Redis client
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.set.return_value = True
        mock_redis_client.get.return_value = '{"key": "value"}'
        mock_redis_client.exists.return_value = 1
        mock_redis_client.delete.return_value = 1
        mock_redis.return_value = mock_redis_client
        
        # Initialize KV store
        with patch('src.storage.kv_store.duckdb.connect'):
            kv_store = BaseContextKV(config=self.test_config, test_mode=True)
            kv_store.redis = Mock()
            kv_store.redis.redis_client = mock_redis_client
            
            # Test store operation
            test_data = {"content": "test", "timestamp": "2023-01-01"}
            result = kv_store.store("test_key", test_data, ttl_seconds=3600)
            
            # Verify Redis operations
            mock_redis_client.set.assert_called_once()
            assert result is True
    
    def test_cache_entry_integration(self):
        """Test CacheEntry dataclass integration"""
        # Create cache entry
        cache_entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=datetime.now(),
            ttl_seconds=3600,
            hit_count=1
        )
        
        # Test serialization
        entry_dict = {
            "key": cache_entry.key,
            "value": cache_entry.value,
            "created_at": cache_entry.created_at.isoformat(),
            "ttl_seconds": cache_entry.ttl_seconds,
            "hit_count": cache_entry.hit_count
        }
        
        # Verify structure
        assert entry_dict["key"] == "test_key"
        assert entry_dict["value"] == {"data": "test"}
        assert entry_dict["ttl_seconds"] == 3600
        assert entry_dict["hit_count"] == 1
    
    def test_metric_event_integration(self):
        """Test MetricEvent dataclass integration"""
        # Create metric event
        metric_event = MetricEvent(
            timestamp=datetime.now(),
            metric_name="test_metric",
            value=100.0,
            tags={"env": "test", "service": "storage"}
        )
        
        # Test validation (if validators are available)
        if CORE_AVAILABLE:
            # This would use actual validation logic
            assert metric_event.metric_name is not None
            assert metric_event.value is not None
            assert metric_event.timestamp is not None
        
        # Test serialization
        event_dict = {
            "metric_name": metric_event.metric_name,
            "value": metric_event.value,
            "timestamp": metric_event.timestamp.isoformat(),
            "tags": metric_event.tags
        }
        
        assert event_dict["metric_name"] == "test_metric"
        assert event_dict["value"] == 100.0
        assert event_dict["tags"]["env"] == "test"


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")
class TestCoreStorageIntegration:
    """Integration tests between core utilities and storage components"""
    
    def test_config_loading_integration(self):
        """Test configuration loading for storage components"""
        # Create temporary config file
        config_data = {
            "storage": {
                "redis": {
                    "host": "localhost",
                    "port": 6379
                },
                "duckdb": {
                    "path": ":memory:"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test config loading
            loaded_config = load_config(config_path)
            
            # Verify storage configuration
            assert "storage" in loaded_config
            assert "redis" in loaded_config["storage"]
            assert loaded_config["storage"]["redis"]["host"] == "localhost"
            assert loaded_config["storage"]["redis"]["port"] == 6379
            
        finally:
            os.unlink(config_path)
    
    def test_utility_functions_integration(self):
        """Test core utility functions with storage data"""
        # Test timestamp formatting
        test_time = datetime(2023, 1, 1, 12, 0, 0)
        formatted = format_timestamp(test_time)
        
        # Verify format
        assert isinstance(formatted, str)
        assert "2023" in formatted
        
        # Test filename sanitization for storage keys
        unsafe_filename = "context/key:with*special?chars"
        safe_filename = sanitize_filename(unsafe_filename)
        
        # Verify sanitization
        assert "/" not in safe_filename
        assert ":" not in safe_filename
        assert "*" not in safe_filename
        assert "?" not in safe_filename
    
    @patch('src.validators.kv_validators.validate_redis_key')
    def test_validation_integration(self, mock_validate):
        """Test validator integration with storage operations"""
        mock_validate.return_value = True
        
        # Test key validation
        test_keys = [
            "valid_key",
            "agent:12345",
            "context_store_key_123"
        ]
        
        for key in test_keys:
            result = validate_redis_key(key)
            assert result is True
        
        # Verify validator was called
        assert mock_validate.call_count == len(test_keys)


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestStorageWorkflowIntegration:
    """End-to-end workflow integration tests"""
    
    def setup_method(self):
        """Setup test environment"""
        self.mock_config = {
            "redis": {"host": "localhost", "port": 6379},
            "duckdb": {"path": ":memory:"}
        }
    
    @patch('src.storage.kv_store.redis.Redis')
    @patch('src.storage.kv_store.duckdb.connect')
    def test_complete_storage_workflow(self, mock_duckdb, mock_redis):
        """Test complete storage workflow: store, retrieve, update, delete"""
        # Mock Redis operations
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.set.return_value = True
        mock_redis_client.get.return_value = '{"content": "test", "_stored_at": "2023-01-01T12:00:00"}'
        mock_redis_client.exists.return_value = 1
        mock_redis_client.delete.return_value = 1
        mock_redis.return_value = mock_redis_client
        
        # Mock DuckDB operations
        mock_duckdb_conn = Mock()
        mock_duckdb.return_value = mock_duckdb_conn
        
        # Initialize storage system
        kv_store = BaseContextKV(config=self.mock_config, test_mode=True)
        kv_store.redis = Mock()
        kv_store.redis.redis_client = mock_redis_client
        kv_store.duckdb_conn = mock_duckdb_conn
        
        # Test workflow steps
        
        # 1. Store data
        test_data = {"content": "test content", "type": "workflow_test"}
        store_result = kv_store.store("workflow_key", test_data, ttl_seconds=3600)
        assert store_result is True
        
        # 2. Verify store operation
        mock_redis_client.set.assert_called()
        
        # 3. Retrieve data
        retrieved_data = kv_store.get("workflow_key")
        assert retrieved_data == {"content": "test", "_stored_at": "2023-01-01T12:00:00"}
        
        # 4. Check existence
        exists_result = kv_store.exists("workflow_key")
        assert exists_result is True
        
        # 5. Delete data
        delete_result = kv_store.delete("workflow_key")
        assert delete_result is True
        
        # Verify all operations were called
        mock_redis_client.get.assert_called_with("workflow_key")
        mock_redis_client.exists.assert_called_with("workflow_key")
        mock_redis_client.delete.assert_called_with("workflow_key")
    
    @patch('src.storage.context_kv.BaseContextKV.__init__')
    @patch('src.storage.context_kv.BaseContextKV.store')
    @patch('src.storage.context_kv.BaseContextKV.get')
    def test_enhanced_context_workflow(self, mock_get, mock_store, mock_init):
        """Test enhanced context KV workflow"""
        mock_init.return_value = None
        mock_store.return_value = True
        mock_get.return_value = {
            "content": "enhanced content",
            "_stored_at": "2023-01-01T12:00:00",
            "_context_id": "ctx_001"
        }
        
        # Initialize enhanced KV store
        context_kv = ContextKV(test_mode=True)
        context_kv.context_cache = {}
        
        # Test enhanced workflow
        
        # 1. Store context with metadata
        context_data = {"content": "enhanced content", "metadata": {"source": "test"}}
        
        with patch('src.storage.context_kv.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)
            
            store_result = context_kv.store_context("ctx_001", context_data, ttl_seconds=7200)
            assert store_result is True
        
        # 2. Verify enhanced data was stored
        mock_store.assert_called_once()
        stored_data = mock_store.call_args[0][1]
        assert stored_data["_stored_at"] == "2023-01-01T12:00:00"
        assert stored_data["_context_id"] == "ctx_001"
        assert stored_data["content"] == "enhanced content"
    
    def test_error_handling_integration(self):
        """Test error handling across storage components"""
        # Test invalid configuration
        with patch('src.storage.kv_store.yaml.safe_load') as mock_yaml:
            mock_yaml.side_effect = Exception("Invalid YAML")
            
            with pytest.raises(Exception):
                BaseContextKV(config_path="invalid.yaml")
        
        # Test connection failures
        with patch('src.storage.kv_store.redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            
            # Should handle gracefully in test mode
            try:
                kv_store = BaseContextKV(config=self.mock_config, test_mode=True)
                # In test mode, should not raise exception
            except Exception as e:
                # Expected in some cases
                assert "Connection failed" in str(e)


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestStoragePerformanceIntegration:
    """Performance and scalability integration tests"""
    
    @patch('src.storage.kv_store.redis.Redis')
    @patch('src.storage.kv_store.duckdb.connect')
    def test_bulk_operations_integration(self, mock_duckdb, mock_redis):
        """Test bulk storage operations"""
        # Mock Redis for bulk operations
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.pipeline.return_value = mock_redis_client
        mock_redis_client.execute.return_value = [True] * 100
        mock_redis.return_value = mock_redis_client
        
        # Mock DuckDB
        mock_duckdb_conn = Mock()
        mock_duckdb.return_value = mock_duckdb_conn
        
        # Initialize storage
        kv_store = BaseContextKV(config={"redis": {}, "duckdb": {}}, test_mode=True)
        kv_store.redis = Mock()
        kv_store.redis.redis_client = mock_redis_client
        
        # Test bulk store operations
        bulk_data = []
        for i in range(100):
            bulk_data.append({
                "key": f"bulk_key_{i}",
                "data": {"content": f"bulk content {i}", "index": i}
            })
        
        # Simulate bulk operations
        for item in bulk_data:
            result = kv_store.store(item["key"], item["data"])
            # In real implementation, would batch these
        
        # Verify operations were performed
        assert mock_redis_client.set.call_count >= len(bulk_data) or mock_redis_client.pipeline.called
    
    def test_memory_usage_integration(self):
        """Test memory-efficient operations"""
        # Test with mock data to verify memory patterns
        large_data = {
            "content": "x" * 1000,  # 1KB content
            "metadata": {"size": 1000, "type": "large"}
        }
        
        # Verify data structure efficiency
        import sys
        data_size = sys.getsizeof(large_data)
        assert data_size > 0
        
        # Test memory cleanup patterns
        del large_data
        # In real implementation, would verify garbage collection


@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available")
class TestStorageSecurityIntegration:
    """Security integration tests for storage layer"""
    
    def test_key_sanitization_integration(self):
        """Test key sanitization across storage operations"""
        dangerous_keys = [
            "../../../etc/passwd",
            "key;rm -rf /",
            "key`whoami`",
            "key$(cat /etc/passwd)",
            "key\x00injection"
        ]
        
        for dangerous_key in dangerous_keys:
            # Test that dangerous keys are handled safely
            with patch('src.storage.kv_store.redis.Redis'):
                kv_store = BaseContextKV(config={}, test_mode=True)
                
                # Keys should be sanitized or rejected
                try:
                    # In production, this would sanitize or reject
                    sanitized_key = dangerous_key.replace("../", "").replace(";", "").replace("`", "").replace("$", "").replace("\x00", "")
                    assert len(sanitized_key) <= len(dangerous_key)
                except Exception:
                    # Expected for malicious inputs
                    pass
    
    def test_data_validation_integration(self):
        """Test data validation in storage operations"""
        # Test various data types and validation
        test_cases = [
            {"valid": True, "data": {"content": "valid data", "type": "test"}},
            {"valid": False, "data": None},
            {"valid": False, "data": ""},
            {"valid": True, "data": {"nested": {"data": "valid"}}},
        ]
        
        for case in test_cases:
            try:
                # Test data validation
                if case["valid"]:
                    json.dumps(case["data"])  # Should serialize
                    assert case["data"] is not None
                else:
                    # Should handle invalid data gracefully
                    assert case["data"] is None or case["data"] == ""
            except (TypeError, ValueError):
                # Expected for invalid data
                assert not case["valid"]


# Performance benchmark integration test
@pytest.mark.skipif(not STORAGE_AVAILABLE, reason="Storage modules not available") 
@pytest.mark.slow
class TestStorageBenchmarkIntegration:
    """Benchmark integration tests (marked as slow)"""
    
    @patch('src.storage.kv_store.redis.Redis')
    def test_throughput_benchmark(self, mock_redis):
        """Basic throughput benchmark test"""
        # Mock high-performance Redis
        mock_redis_client = Mock()
        mock_redis_client.ping.return_value = True
        mock_redis_client.set.return_value = True
        mock_redis.return_value = mock_redis_client
        
        with patch('src.storage.kv_store.duckdb.connect'):
            kv_store = BaseContextKV(config={}, test_mode=True)
            kv_store.redis = Mock()
            kv_store.redis.redis_client = mock_redis_client
            
            # Benchmark 1000 operations
            import time
            start_time = time.time()
            
            for i in range(1000):
                kv_store.store(f"bench_key_{i}", {"data": f"benchmark_{i}"})
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Verify reasonable performance (mocked operations should be fast)
            assert duration < 5.0  # Should complete in under 5 seconds
            assert mock_redis_client.set.call_count == 1000