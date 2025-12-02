#!/usr/bin/env python3
"""
Comprehensive tests for dashboard background collection functionality.

Tests cover:
- Collection loop lifecycle (start/stop)
- Service client integration
- Error handling and retry logic
- Metric collection intervals
- Health check execution paths
"""

import asyncio
import pytest
import time
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.monitoring.dashboard import UnifiedDashboard, ServiceMetrics


@pytest.fixture
def mock_service_clients():
    """Create mock service clients for testing."""
    redis_client = Mock()
    redis_client.ping = Mock(return_value=True)
    redis_client.keys = Mock(return_value=[b'scratchpad:agent1:key1', b'scratchpad:agent2:key2'])
    
    neo4j_client = Mock()
    neo4j_client.query = Mock(return_value=[{'test': 1}])
    
    qdrant_client = Mock()
    mock_collection = Mock()
    mock_collection.vectors_count = 100
    mock_collections = Mock()
    mock_collections.collections = [mock_collection]
    qdrant_client.get_collections = Mock(return_value=mock_collections)
    
    return {
        'redis': redis_client,
        'neo4j': neo4j_client,
        'qdrant': qdrant_client
    }


@pytest.fixture
async def dashboard():
    """Create dashboard instance for testing."""
    config = {
        'collection_interval_seconds': 0.1,  # Fast for testing
        'cache_duration_seconds': 1
    }
    dashboard = UnifiedDashboard(config)
    
    # Ensure clean state
    dashboard._collection_running = False
    dashboard._collection_task = None
    
    yield dashboard
    
    # Cleanup after test
    try:
        if dashboard._collection_running:
            await dashboard.stop_collection_loop()
    except Exception:
        pass  # Ignore cleanup errors


class TestDashboardCollectionLoop:
    """Test collection loop lifecycle and functionality."""
    
    @pytest.mark.asyncio
    async def test_start_collection_loop(self, dashboard):
        """Test collection loop starts properly."""
        assert not dashboard._collection_running
        assert dashboard._collection_task is None
        
        await dashboard.start_collection_loop()
        
        assert dashboard._collection_running
        assert dashboard._collection_task is not None
        assert not dashboard._collection_task.done()
        
        # Cleanup
        await dashboard.stop_collection_loop()
    
    @pytest.mark.asyncio
    async def test_stop_collection_loop(self, dashboard):
        """Test collection loop stops cleanly."""
        await dashboard.start_collection_loop()
        assert dashboard._collection_running
        
        await dashboard.stop_collection_loop()
        
        assert not dashboard._collection_running
        assert dashboard._collection_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_start_collection_loop_idempotent(self, dashboard):
        """Test starting collection loop multiple times is safe."""
        await dashboard.start_collection_loop()
        first_task = dashboard._collection_task
        
        # Starting again should not create new task
        await dashboard.start_collection_loop()
        assert dashboard._collection_task is first_task
        
        # Cleanup
        await dashboard.stop_collection_loop()
    
    @pytest.mark.asyncio
    async def test_collection_loop_updates_metrics(self, dashboard, mock_service_clients):
        """Test collection loop actually updates metrics."""
        dashboard.set_service_clients(**mock_service_clients)
        
        # Mock collect_all_metrics to avoid full collection complexity
        dashboard.collect_all_metrics = AsyncMock(return_value={'test': 'data'})
        
        await dashboard.start_collection_loop()
        
        # Wait for at least one collection cycle
        await asyncio.sleep(0.2)
        
        # Verify collect_all_metrics was called
        assert dashboard.collect_all_metrics.called
        assert dashboard.collect_all_metrics.call_count >= 1
        
        # Cleanup
        await dashboard.stop_collection_loop()
    
    @pytest.mark.asyncio
    async def test_collection_loop_interval_timing(self, dashboard):
        """Test collection loop respects configured intervals."""
        dashboard.config['collection_interval_seconds'] = 0.05
        
        call_times = []
        original_collect = dashboard.collect_all_metrics
        
        async def mock_collect(*args, **kwargs):
            call_times.append(time.time())
            return {'test': 'data'}
        
        dashboard.collect_all_metrics = mock_collect
        
        await dashboard.start_collection_loop()
        await asyncio.sleep(0.15)  # Allow ~3 collection cycles
        await dashboard.stop_collection_loop()
        
        # Should have at least 2 calls with proper timing
        assert len(call_times) >= 2
        if len(call_times) >= 2:
            interval = call_times[1] - call_times[0]
            assert 0.04 <= interval <= 0.08  # Allow some timing variance
        
        dashboard.collect_all_metrics = original_collect


class TestCollectionErrorHandling:
    """Test error handling and retry logic in collection loop."""
    
    @pytest.mark.asyncio
    async def test_collection_loop_handles_connection_errors(self, dashboard):
        """Test collection loop handles connection errors gracefully."""
        failure_count = 0
        
        async def failing_collect(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Fail first 2 times
                raise ConnectionError("Service unavailable")
            return {'test': 'data'}
        
        dashboard.collect_all_metrics = failing_collect
        
        await dashboard.start_collection_loop()
        await asyncio.sleep(0.3)  # Allow multiple collection attempts
        await dashboard.stop_collection_loop()
        
        # Should have recovered after failures
        assert failure_count >= 3  # 2 failures + 1 success
    
    @pytest.mark.asyncio
    async def test_collection_loop_stops_after_max_failures(self, dashboard):
        """Test collection loop stops after too many consecutive failures."""
        dashboard.config['collection_interval_seconds'] = 0.01  # Very fast for testing
        
        async def always_failing_collect(*args, **kwargs):
            raise ConnectionError("Persistent failure")
        
        dashboard.collect_all_metrics = always_failing_collect
        
        await dashboard.start_collection_loop()
        
        # Wait for collection loop to give up
        start_time = time.time()
        while dashboard._collection_running and (time.time() - start_time) < 2:
            await asyncio.sleep(0.01)
        
        # Collection should have stopped due to too many failures
        assert not dashboard._collection_running
    
    @pytest.mark.asyncio
    async def test_collection_loop_exponential_backoff(self, dashboard):
        """Test collection loop implements exponential backoff on failures."""
        dashboard.config['collection_interval_seconds'] = 0.01
        
        call_times = []
        failure_count = 0
        
        async def intermittent_failing_collect(*args, **kwargs):
            nonlocal failure_count
            call_times.append(time.time())
            failure_count += 1
            if failure_count <= 3:  # Fail first 3 times
                raise ConnectionError("Temporary failure")
            return {'test': 'data'}
        
        dashboard.collect_all_metrics = intermittent_failing_collect
        
        await dashboard.start_collection_loop()
        await asyncio.sleep(0.5)  # Allow backoff cycles
        await dashboard.stop_collection_loop()
        
        # Should show increasing intervals between failed attempts
        assert len(call_times) >= 3
        if len(call_times) >= 3:
            # Second interval should be longer than first (backoff)
            interval1 = call_times[1] - call_times[0]
            interval2 = call_times[2] - call_times[1]
            assert interval2 > interval1
    
    @pytest.mark.asyncio
    async def test_collection_loop_cancellation_handling(self, dashboard):
        """Test collection loop handles cancellation properly."""
        # Mock a long-running collection that can be cancelled
        async def slow_collect(*args, **kwargs):
            await asyncio.sleep(1)  # Long operation
            return {'test': 'data'}
        
        dashboard.collect_all_metrics = slow_collect
        
        await dashboard.start_collection_loop()
        
        # Start the loop and immediately cancel
        await asyncio.sleep(0.01)  # Let it start
        await dashboard.stop_collection_loop()
        
        # Should complete without hanging
        assert not dashboard._collection_running
        assert dashboard._collection_task.cancelled()


class TestServiceClientIntegration:
    """Test service client integration and health checks."""
    
    def test_set_service_clients(self, dashboard, mock_service_clients):
        """Test setting service clients."""
        dashboard.set_service_clients(**mock_service_clients)
        
        assert dashboard.service_clients['redis'] is mock_service_clients['redis']
        assert dashboard.service_clients['neo4j'] is mock_service_clients['neo4j']
        assert dashboard.service_clients['qdrant'] is mock_service_clients['qdrant']
    
    def test_set_service_clients_partial(self, dashboard, mock_service_clients):
        """Test setting only some service clients."""
        dashboard.set_service_clients(redis_client=mock_service_clients['redis'])
        
        assert dashboard.service_clients['redis'] is mock_service_clients['redis']
        assert dashboard.service_clients['neo4j'] is None
        assert dashboard.service_clients['qdrant'] is None
    
    @pytest.mark.asyncio
    async def test_collect_service_metrics_with_clients(self, dashboard, mock_service_clients):
        """Test service metrics collection with real clients."""
        dashboard.set_service_clients(**mock_service_clients)
        
        services = await dashboard._collect_service_metrics()
        
        # Should have 5 services
        assert len(services) == 5
        
        # Find specific services and check their status
        service_by_name = {s.name: s for s in services}
        
        assert service_by_name["MCP Server"].status == "healthy"
        assert service_by_name["Redis"].status == "healthy"
        assert service_by_name["Neo4j HTTP"].status == "healthy"
        assert service_by_name["Neo4j Bolt"].status == "healthy"
        assert service_by_name["Qdrant"].status == "healthy"
    
    @pytest.mark.asyncio
    async def test_collect_service_metrics_redis_failure(self, dashboard, mock_service_clients):
        """Test service metrics when Redis fails."""
        mock_service_clients['redis'].ping.side_effect = ConnectionError("Redis down")
        dashboard.set_service_clients(**mock_service_clients)
        
        services = await dashboard._collect_service_metrics()
        service_by_name = {s.name: s for s in services}
        
        assert service_by_name["Redis"].status == "unhealthy"
        assert service_by_name["Neo4j HTTP"].status == "healthy"  # Others still work
    
    @pytest.mark.asyncio
    async def test_collect_service_metrics_neo4j_failure(self, dashboard, mock_service_clients):
        """Test service metrics when Neo4j fails."""
        mock_service_clients['neo4j'].query.side_effect = Exception("Neo4j down")
        dashboard.set_service_clients(**mock_service_clients)
        
        services = await dashboard._collect_service_metrics()
        service_by_name = {s.name: s for s in services}
        
        assert service_by_name["Neo4j HTTP"].status == "unhealthy"
        assert service_by_name["Neo4j Bolt"].status == "unhealthy"
        assert service_by_name["Redis"].status == "healthy"  # Others still work
    
    @pytest.mark.asyncio
    async def test_collect_service_metrics_qdrant_failure(self, dashboard, mock_service_clients):
        """Test service metrics when Qdrant fails."""
        mock_service_clients['qdrant'].get_collections.side_effect = Exception("Qdrant down")
        dashboard.set_service_clients(**mock_service_clients)
        
        services = await dashboard._collect_service_metrics()
        service_by_name = {s.name: s for s in services}
        
        assert service_by_name["Qdrant"].status == "unhealthy"
        assert service_by_name["Redis"].status == "healthy"  # Others still work
    
    @pytest.mark.asyncio
    async def test_collect_service_metrics_no_clients(self, dashboard):
        """Test service metrics without any clients set."""
        services = await dashboard._collect_service_metrics()
        service_by_name = {s.name: s for s in services}
        
        # All services except MCP Server should be unknown
        assert service_by_name["MCP Server"].status == "healthy"
        assert service_by_name["Redis"].status == "unknown"
        assert service_by_name["Neo4j HTTP"].status == "unknown"
        assert service_by_name["Neo4j Bolt"].status == "unknown"
        assert service_by_name["Qdrant"].status == "unknown"


class TestMetricsCollectionIntegration:
    """Test full metrics collection with service integration."""
    
    @pytest.mark.asyncio
    async def test_collect_all_metrics_integration(self, dashboard, mock_service_clients):
        """Test full metrics collection with service clients."""
        dashboard.set_service_clients(**mock_service_clients)
        
        metrics = await dashboard.collect_all_metrics(force_refresh=True)
        
        # Verify structure
        assert 'timestamp' in metrics
        assert 'system' in metrics
        assert 'services' in metrics
        assert 'veris' in metrics
        assert 'security' in metrics
        assert 'backups' in metrics
        
        # Verify services have real status
        services = metrics['services']
        service_by_name = {s['name']: s for s in services}
        assert service_by_name["Redis"]['status'] == "healthy"
        assert service_by_name["Neo4j HTTP"]['status'] == "healthy"
        assert service_by_name["Qdrant"]['status'] == "healthy"
    
    @pytest.mark.asyncio
    async def test_collect_all_metrics_caching(self, dashboard, mock_service_clients):
        """Test metrics collection respects caching."""
        dashboard.set_service_clients(**mock_service_clients)
        
        # First collection
        metrics1 = await dashboard.collect_all_metrics()
        timestamp1 = metrics1['timestamp']
        
        # Second collection (should use cache)
        metrics2 = await dashboard.collect_all_metrics()
        timestamp2 = metrics2['timestamp']
        
        assert timestamp1 == timestamp2  # Same timestamp = from cache
        
        # Force refresh should get new timestamp
        metrics3 = await dashboard.collect_all_metrics(force_refresh=True)
        timestamp3 = metrics3['timestamp']
        
        assert timestamp3 != timestamp1  # Different timestamp = fresh collection


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])