#!/usr/bin/env python3
"""
Isolated tests for storage.context_kv module to achieve high coverage.

This test suite avoids the full import cascade that causes duckdb issues
by directly importing and mocking the ContextKV class.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

# Mock the problematic imports before importing the module
with patch.dict("sys.modules", {"duckdb": Mock(), "storage.kv_store": Mock()}):
    # Create mock classes for the base imports
    mock_base_context_kv = Mock()
    mock_metric_event = Mock()
    mock_cache_entry = Mock()

    # Mock the module structure
    mock_kv_store = Mock()
    mock_kv_store.ContextKV = mock_base_context_kv
    mock_kv_store.MetricEvent = mock_metric_event
    mock_kv_store.CacheEntry = mock_cache_entry

    with patch.dict("sys.modules", {"storage.kv_store": mock_kv_store}):
        from src.storage.context_kv import ContextKV


class MockMetricEvent:
    """Mock MetricEvent for testing."""

    def __init__(self, timestamp, metric_name, value, tags, document_id=None, agent_id=None):
        self.timestamp = timestamp
        self.metric_name = metric_name
        self.value = value
        self.tags = tags
        self.document_id = document_id
        self.agent_id = agent_id


class TestContextKVIsolated:
    """Test ContextKV class in isolation."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("storage.context_kv.BaseContextKV.__init__"):
            self.context_kv = ContextKV()
            self.context_kv.redis = Mock()

    def test_context_kv_initialization(self):
        """Test ContextKV initialization."""
        with patch("storage.context_kv.BaseContextKV.__init__") as mock_super_init:
            mock_super_init.return_value = None

            context_kv = ContextKV()

            mock_super_init.assert_called_once_with(".ctxrc.yaml", False, None, False)
            assert hasattr(context_kv, "context_cache")
            assert context_kv.context_cache == {}

    def test_store_context_success(self):
        """Test successful context storage."""
        context_id = "test_context_123"
        context_data = {
            "title": "Test Context",
            "content": "Test content",
            "metadata": {"author": "test_user"},
        }

        self.context_kv.redis.set_cache.return_value = True
        self.context_kv.redis.record_metric.return_value = None

        with patch("storage.context_kv.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)

            with patch("storage.context_kv.MetricEvent", MockMetricEvent):
                result = self.context_kv.store_context(context_id, context_data)

                assert result is True

                # Verify set_cache was called with enhanced data
                call_args = self.context_kv.redis.set_cache.call_args
                assert call_args[0][0] == f"context:{context_id}"
                stored_data = call_args[0][1]
                assert stored_data["title"] == "Test Context"
                assert stored_data["_stored_at"] == "2023-01-01T12:00:00"
                assert stored_data["_context_id"] == context_id
                assert call_args[0][2] is None  # ttl_seconds

                # Verify metric was recorded
                self.context_kv.redis.record_metric.assert_called_once()

    def test_store_context_with_ttl(self):
        """Test context storage with TTL."""
        context_id = "ttl_context"
        context_data = {"data": "expires"}
        ttl_seconds = 3600

        self.context_kv.redis.set_cache.return_value = True
        self.context_kv.redis.record_metric.return_value = None

        with patch("storage.context_kv.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)

            with patch("storage.context_kv.MetricEvent", MockMetricEvent):
                result = self.context_kv.store_context(context_id, context_data, ttl_seconds)

                assert result is True

                # Verify TTL was passed
                call_args = self.context_kv.redis.set_cache.call_args
                assert call_args[0][2] == ttl_seconds

    def test_store_context_empty_inputs(self):
        """Test context storage with empty inputs."""
        # Empty context ID
        result = self.context_kv.store_context("", {"data": "test"})
        assert result is False

        result = self.context_kv.store_context(None, {"data": "test"})
        assert result is False

        # Empty context data
        result = self.context_kv.store_context("test_id", {})
        assert result is False

        result = self.context_kv.store_context("test_id", None)
        assert result is False

    def test_store_context_redis_failure(self):
        """Test context storage when Redis fails."""
        context_id = "fail_context"
        context_data = {"data": "test"}

        self.context_kv.redis.set_cache.return_value = False

        with patch("storage.context_kv.datetime"):
            result = self.context_kv.store_context(context_id, context_data)

            assert result is False
            # Metric should not be recorded on failure
            self.context_kv.redis.record_metric.assert_not_called()

    def test_get_context_success(self):
        """Test successful context retrieval."""
        context_id = "test_context"
        cached_data = {
            "title": "Cached Context",
            "_stored_at": "2023-01-01T12:00:00",
            "_context_id": context_id,
        }

        self.context_kv.redis.get_cache.return_value = cached_data
        self.context_kv.redis.record_metric.return_value = None

        with patch("storage.context_kv.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 30, 0)

            with patch("storage.context_kv.MetricEvent", MockMetricEvent):
                result = self.context_kv.get_context(context_id)

                assert result == cached_data

                # Verify cache was checked
                self.context_kv.redis.get_cache.assert_called_once_with(f"context:{context_id}")

                # Verify hit metric was recorded
                self.context_kv.redis.record_metric.assert_called_once()

    def test_get_context_not_found(self):
        """Test context retrieval when context not found."""
        context_id = "missing_context"

        self.context_kv.redis.get_cache.return_value = None

        result = self.context_kv.get_context(context_id)

        assert result is None
        # No metric should be recorded on miss
        self.context_kv.redis.record_metric.assert_not_called()

    def test_get_context_empty_context_id(self):
        """Test context retrieval with empty context ID."""
        result = self.context_kv.get_context("")
        assert result is None

        result = self.context_kv.get_context(None)
        assert result is None

        # Redis should not be called
        self.context_kv.redis.get_cache.assert_not_called()

    def test_delete_context_success(self):
        """Test successful context deletion."""
        context_id = "delete_context"

        self.context_kv.redis.delete_cache.return_value = 1
        self.context_kv.redis.record_metric.return_value = None

        with patch("storage.context_kv.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)

            with patch("storage.context_kv.MetricEvent", MockMetricEvent):
                result = self.context_kv.delete_context(context_id)

                assert result is True

                # Verify delete was called
                self.context_kv.redis.delete_cache.assert_called_once_with(f"context:{context_id}")

                # Verify deletion metric was recorded
                self.context_kv.redis.record_metric.assert_called_once()

    def test_delete_context_not_found(self):
        """Test context deletion when context doesn't exist."""
        context_id = "missing_context"

        self.context_kv.redis.delete_cache.return_value = 0

        result = self.context_kv.delete_context(context_id)

        assert result is False
        # No metric should be recorded when nothing deleted
        self.context_kv.redis.record_metric.assert_not_called()

    def test_delete_context_multiple_deleted(self):
        """Test context deletion when multiple items deleted."""
        context_id = "multi_delete_context"

        self.context_kv.redis.delete_cache.return_value = 3
        self.context_kv.redis.record_metric.return_value = None

        with patch("storage.context_kv.datetime"):
            with patch("storage.context_kv.MetricEvent", MockMetricEvent) as mock_metric:
                result = self.context_kv.delete_context(context_id)

                assert result is True

                # Verify metric reflects actual count deleted
                metric_args = mock_metric.call_args[1]
                assert metric_args["value"] == 3.0

    def test_list_contexts(self):
        """Test context listing functionality."""
        # Default pattern
        result = self.context_kv.list_contexts()
        assert result == []
        assert isinstance(result, list)

        # Custom pattern
        result = self.context_kv.list_contexts("test_*")
        assert result == []
        assert isinstance(result, list)

        # Empty pattern
        result = self.context_kv.list_contexts("")
        assert result == []
        assert isinstance(result, list)

    def test_update_context_merge_success(self):
        """Test successful context update with merge."""
        context_id = "update_context"
        existing_data = {
            "title": "Original Title",
            "content": "Original content",
            "_stored_at": "2023-01-01T10:00:00",
            "_context_id": context_id,
        }
        updates = {"content": "Updated content", "new_field": "new value"}

        # Mock get_context to return existing data
        with patch.object(self.context_kv, "get_context", return_value=existing_data):
            # Mock store_context to return success
            with patch.object(self.context_kv, "store_context", return_value=True) as mock_store:
                with patch("storage.context_kv.datetime") as mock_datetime:
                    mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)

                    result = self.context_kv.update_context(context_id, updates, merge=True)

                    assert result is True

                    # Verify store_context was called with merged data
                    mock_store.assert_called_once()
                    call_args = mock_store.call_args[0]
                    assert call_args[0] == context_id

                    merged_data = call_args[1]
                    assert merged_data["title"] == "Original Title"  # Preserved
                    assert merged_data["content"] == "Updated content"  # Updated
                    assert merged_data["new_field"] == "new value"  # Added
                    assert merged_data["_updated_at"] == "2023-01-01T12:00:00"
                    assert merged_data["_context_id"] == context_id

    def test_update_context_replace_success(self):
        """Test successful context update with replace."""
        context_id = "replace_context"
        existing_data = {
            "title": "Original Title",
            "content": "Original content",
            "_stored_at": "2023-01-01T10:00:00",
            "_context_id": context_id,
        }
        updates = {"title": "New Title", "description": "New description"}

        # Mock get_context to return existing data
        with patch.object(self.context_kv, "get_context", return_value=existing_data):
            # Mock store_context to return success
            with patch.object(self.context_kv, "store_context", return_value=True) as mock_store:
                with patch("storage.context_kv.datetime") as mock_datetime:
                    mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 12, 0, 0)

                    result = self.context_kv.update_context(context_id, updates, merge=False)

                    assert result is True

                    # Verify store_context was called with replacement data
                    call_args = mock_store.call_args[0]
                    replaced_data = call_args[1]
                    assert replaced_data["title"] == "New Title"
                    assert replaced_data["description"] == "New description"
                    assert "content" not in replaced_data  # Original field removed
                    assert replaced_data["_updated_at"] == "2023-01-01T12:00:00"
                    assert replaced_data["_context_id"] == context_id

    def test_update_context_not_found(self):
        """Test context update when context doesn't exist."""
        context_id = "missing_context"
        updates = {"field": "value"}

        # Mock get_context to return None
        with patch.object(self.context_kv, "get_context", return_value=None):
            result = self.context_kv.update_context(context_id, updates)

            assert result is False

    def test_update_context_empty_inputs(self):
        """Test context update with empty inputs."""
        # Empty context ID
        result = self.context_kv.update_context("", {"field": "value"})
        assert result is False

        result = self.context_kv.update_context(None, {"field": "value"})
        assert result is False

        # Empty updates
        result = self.context_kv.update_context("test_id", {})
        assert result is False

        result = self.context_kv.update_context("test_id", None)
        assert result is False

    def test_get_context_metrics_success(self):
        """Test successful context metrics retrieval."""
        context_id = "metrics_context"
        hours = 24

        # Create mock metrics
        mock_metrics = [
            MockMetricEvent(
                timestamp=datetime(2023, 1, 1, 10, 0, 0),
                metric_name="context.store",
                value=1.0,
                tags={"context_id": context_id},
                document_id="doc_123",
                agent_id="agent_456",
            ),
            MockMetricEvent(
                timestamp=datetime(2023, 1, 1, 11, 0, 0),
                metric_name="context.get",
                value=1.0,
                tags={"context_id": context_id, "cache_hit": "true"},
                document_id=None,
                agent_id=None,
            ),
        ]

        self.context_kv.redis.get_metrics.return_value = mock_metrics

        with patch("storage.context_kv.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 2, 10, 0, 0)

            result = self.context_kv.get_context_metrics(context_id, hours)

            assert result["context_id"] == context_id
            assert result["period_hours"] == hours
            assert result["metrics_count"] == 2
            assert len(result["metrics"]) == 2

            # Verify metrics data structure
            metric1 = result["metrics"][0]
            assert metric1["timestamp"] == "2023-01-01T10:00:00"
            assert metric1["metric_name"] == "context.store"
            assert metric1["value"] == 1.0
            assert metric1["tags"]["context_id"] == context_id

            metric2 = result["metrics"][1]
            assert metric2["timestamp"] == "2023-01-01T11:00:00"
            assert metric2["metric_name"] == "context.get"
            assert metric2["tags"]["cache_hit"] == "true"

    def test_get_context_metrics_no_metrics(self):
        """Test context metrics when no metrics found."""
        context_id = "no_metrics_context"

        self.context_kv.redis.get_metrics.return_value = []

        with patch("storage.context_kv.datetime"):
            result = self.context_kv.get_context_metrics(context_id)

            assert result["context_id"] == context_id
            assert result["metrics_count"] == 0
            assert result["metrics"] == []

    def test_context_lifecycle_integration(self):
        """Test complete context lifecycle: store, get, update, delete."""
        context_id = "lifecycle_context"
        initial_data = {"title": "Initial", "version": 1}

        # 1. Store context
        self.context_kv.redis.set_cache.return_value = True
        self.context_kv.redis.record_metric.return_value = None

        with patch("storage.context_kv.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2023, 1, 1, 10, 0, 0)

            with patch("storage.context_kv.MetricEvent", MockMetricEvent):
                store_result = self.context_kv.store_context(context_id, initial_data)
                assert store_result is True

        # 2. Get context
        stored_data = {
            **initial_data,
            "_stored_at": "2023-01-01T10:00:00",
            "_context_id": context_id,
        }
        self.context_kv.redis.get_cache.return_value = stored_data

        with patch("storage.context_kv.datetime"):
            with patch("storage.context_kv.MetricEvent", MockMetricEvent):
                get_result = self.context_kv.get_context(context_id)
                assert get_result == stored_data

        # 3. Update context
        updates = {"version": 2, "status": "updated"}
        with patch.object(self.context_kv, "get_context", return_value=stored_data):
            with patch.object(self.context_kv, "store_context", return_value=True):
                update_result = self.context_kv.update_context(context_id, updates)
                assert update_result is True

        # 4. Delete context
        self.context_kv.redis.delete_cache.return_value = 1

        with patch("storage.context_kv.datetime"):
            with patch("storage.context_kv.MetricEvent", MockMetricEvent):
                delete_result = self.context_kv.delete_context(context_id)
                assert delete_result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
