#!/usr/bin/env python3
"""
Comprehensive tests for service health check functionality.

Tests cover:
- Redis health check execution paths
- Neo4j health check execution paths  
- Qdrant health check execution paths
- Error handling for each service type
- Async execution and executor usage
- Health status propagation
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

# Add src to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.monitoring.dashboard import UnifiedDashboard, ServiceMetrics


@pytest.fixture
def dashboard():
    """Create dashboard instance for testing."""
    return UnifiedDashboard()


class TestRedisHealthChecks:
    """Test Redis service health check functionality."""
    
    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, dashboard):
        """Test successful Redis health check."""
        redis_client = Mock()
        redis_client.ping = Mock(return_value=True)
        
        dashboard.set_service_clients(redis_client=redis_client)
        services = await dashboard._collect_service_metrics()
        
        redis_service = next(s for s in services if s.name == "Redis")
        assert redis_service.status == "healthy"
        redis_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_health_check_connection_error(self, dashboard):
        """Test Redis health check with connection error."""
        redis_client = Mock()
        redis_client.ping = Mock(side_effect=ConnectionError("Connection refused"))
        
        dashboard.set_service_clients(redis_client=redis_client)
        services = await dashboard._collect_service_metrics()
        
        redis_service = next(s for s in services if s.name == "Redis")
        assert redis_service.status == "unhealthy"
        redis_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_health_check_timeout(self, dashboard):
        """Test Redis health check with timeout."""
        redis_client = Mock()
        redis_client.ping = Mock(side_effect=TimeoutError("Operation timed out"))
        
        dashboard.set_service_clients(redis_client=redis_client)
        services = await dashboard._collect_service_metrics()
        
        redis_service = next(s for s in services if s.name == "Redis")
        assert redis_service.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_redis_health_check_generic_exception(self, dashboard):
        """Test Redis health check with generic exception."""
        redis_client = Mock()
        redis_client.ping = Mock(side_effect=Exception("Unexpected error"))
        
        dashboard.set_service_clients(redis_client=redis_client)
        services = await dashboard._collect_service_metrics()
        
        redis_service = next(s for s in services if s.name == "Redis")
        assert redis_service.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_redis_health_check_no_client(self, dashboard):
        """Test Redis health check without client set."""
        # Don't set any Redis client
        services = await dashboard._collect_service_metrics()
        
        redis_service = next(s for s in services if s.name == "Redis")
        assert redis_service.status == "unknown"


class TestNeo4jHealthChecks:
    """Test Neo4j service health check functionality."""
    
    @pytest.mark.asyncio
    async def test_neo4j_health_check_success(self, dashboard):
        """Test successful Neo4j health check."""
        neo4j_client = Mock()
        neo4j_client.query = Mock(return_value=[{'test': 1}])
        
        dashboard.set_service_clients(neo4j_client=neo4j_client)
        services = await dashboard._collect_service_metrics()
        
        neo4j_http = next(s for s in services if s.name == "Neo4j HTTP")
        neo4j_bolt = next(s for s in services if s.name == "Neo4j Bolt")
        
        assert neo4j_http.status == "healthy"
        assert neo4j_bolt.status == "healthy"
        neo4j_client.query.assert_called_once_with("RETURN 1 as test")
    
    @pytest.mark.asyncio
    async def test_neo4j_health_check_empty_result(self, dashboard):
        """Test Neo4j health check with empty result."""
        neo4j_client = Mock()
        neo4j_client.query = Mock(return_value=[])  # Empty result
        
        dashboard.set_service_clients(neo4j_client=neo4j_client)
        services = await dashboard._collect_service_metrics()
        
        neo4j_http = next(s for s in services if s.name == "Neo4j HTTP")
        neo4j_bolt = next(s for s in services if s.name == "Neo4j Bolt")
        
        assert neo4j_http.status == "unhealthy"
        assert neo4j_bolt.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_neo4j_health_check_none_result(self, dashboard):
        """Test Neo4j health check with None result."""
        neo4j_client = Mock()
        neo4j_client.query = Mock(return_value=None)
        
        dashboard.set_service_clients(neo4j_client=neo4j_client)
        services = await dashboard._collect_service_metrics()
        
        neo4j_http = next(s for s in services if s.name == "Neo4j HTTP")
        assert neo4j_http.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_neo4j_health_check_connection_error(self, dashboard):
        """Test Neo4j health check with connection error."""
        neo4j_client = Mock()
        neo4j_client.query = Mock(side_effect=ConnectionError("Neo4j unavailable"))
        
        dashboard.set_service_clients(neo4j_client=neo4j_client)
        services = await dashboard._collect_service_metrics()
        
        neo4j_http = next(s for s in services if s.name == "Neo4j HTTP")
        assert neo4j_http.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_neo4j_health_check_authentication_error(self, dashboard):
        """Test Neo4j health check with authentication error."""
        neo4j_client = Mock()
        neo4j_client.query = Mock(side_effect=Exception("Authentication failed"))
        
        dashboard.set_service_clients(neo4j_client=neo4j_client)
        services = await dashboard._collect_service_metrics()
        
        neo4j_http = next(s for s in services if s.name == "Neo4j HTTP")
        assert neo4j_http.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_neo4j_health_check_no_client(self, dashboard):
        """Test Neo4j health check without client set."""
        services = await dashboard._collect_service_metrics()
        
        neo4j_http = next(s for s in services if s.name == "Neo4j HTTP")
        neo4j_bolt = next(s for s in services if s.name == "Neo4j Bolt")
        
        assert neo4j_http.status == "unknown"
        assert neo4j_bolt.status == "unknown"


class TestQdrantHealthChecks:
    """Test Qdrant service health check functionality."""
    
    @pytest.mark.asyncio
    async def test_qdrant_health_check_success(self, dashboard):
        """Test successful Qdrant health check."""
        qdrant_client = Mock()
        mock_collections = Mock()
        qdrant_client.get_collections = Mock(return_value=mock_collections)
        
        dashboard.set_service_clients(qdrant_client=qdrant_client)
        services = await dashboard._collect_service_metrics()
        
        qdrant_service = next(s for s in services if s.name == "Qdrant")
        assert qdrant_service.status == "healthy"
        qdrant_client.get_collections.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_qdrant_health_check_empty_collections(self, dashboard):
        """Test Qdrant health check with empty collections."""
        qdrant_client = Mock()
        qdrant_client.get_collections = Mock(return_value=None)
        
        dashboard.set_service_clients(qdrant_client=qdrant_client)
        services = await dashboard._collect_service_metrics()
        
        qdrant_service = next(s for s in services if s.name == "Qdrant")
        assert qdrant_service.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_qdrant_health_check_connection_error(self, dashboard):
        """Test Qdrant health check with connection error."""
        qdrant_client = Mock()
        qdrant_client.get_collections = Mock(side_effect=ConnectionError("Qdrant down"))
        
        dashboard.set_service_clients(qdrant_client=qdrant_client)
        services = await dashboard._collect_service_metrics()
        
        qdrant_service = next(s for s in services if s.name == "Qdrant")
        assert qdrant_service.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_qdrant_health_check_api_error(self, dashboard):
        """Test Qdrant health check with API error."""
        qdrant_client = Mock()
        qdrant_client.get_collections = Mock(side_effect=Exception("API error"))
        
        dashboard.set_service_clients(qdrant_client=qdrant_client)
        services = await dashboard._collect_service_metrics()
        
        qdrant_service = next(s for s in services if s.name == "Qdrant")
        assert qdrant_service.status == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_qdrant_health_check_no_client(self, dashboard):
        """Test Qdrant health check without client set."""
        services = await dashboard._collect_service_metrics()
        
        qdrant_service = next(s for s in services if s.name == "Qdrant")
        assert qdrant_service.status == "unknown"


class TestAsyncExecutorUsage:
    """Test async executor usage in health checks."""
    
    @pytest.mark.asyncio
    async def test_health_checks_use_executor(self, dashboard):
        """Test that health checks properly use thread executor."""
        redis_client = Mock()
        neo4j_client = Mock()
        qdrant_client = Mock()
        
        # Mock successful responses
        redis_client.ping = Mock(return_value=True)
        neo4j_client.query = Mock(return_value=[{'test': 1}])
        qdrant_client.get_collections = Mock(return_value=Mock())
        
        dashboard.set_service_clients(
            redis_client=redis_client,
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client
        )
        
        # Use patch to verify run_in_executor is called
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor = AsyncMock(side_effect=[
                True,  # Redis ping
                [{'test': 1}],  # Neo4j query  
                Mock(),  # Qdrant collections
            ])
            
            services = await dashboard._collect_service_metrics()
            
            # Verify executor was used for each service
            assert mock_loop.run_in_executor.call_count == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, dashboard):
        """Test that health checks can run concurrently."""
        import time
        
        # Create slow clients to test concurrency
        redis_client = Mock()
        neo4j_client = Mock()
        qdrant_client = Mock()
        
        def slow_ping():
            time.sleep(0.1)
            return True
            
        def slow_query(query):
            time.sleep(0.1)
            return [{'test': 1}]
            
        def slow_collections():
            time.sleep(0.1)
            return Mock()
        
        redis_client.ping = slow_ping
        neo4j_client.query = slow_query
        qdrant_client.get_collections = slow_collections
        
        dashboard.set_service_clients(
            redis_client=redis_client,
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client
        )
        
        start_time = time.time()
        services = await dashboard._collect_service_metrics()
        end_time = time.time()
        
        # If running concurrently, should take ~0.1s instead of ~0.3s
        elapsed = end_time - start_time
        assert elapsed < 0.2  # Should be much faster than sequential


class TestMixedServiceStates:
    """Test health checks with mixed service states."""
    
    @pytest.mark.asyncio
    async def test_mixed_service_health_states(self, dashboard):
        """Test health checks with some services healthy and others not."""
        redis_client = Mock()
        neo4j_client = Mock()
        qdrant_client = Mock()
        
        # Redis healthy, Neo4j unhealthy, Qdrant healthy
        redis_client.ping = Mock(return_value=True)
        neo4j_client.query = Mock(side_effect=ConnectionError("Neo4j down"))
        qdrant_client.get_collections = Mock(return_value=Mock())
        
        dashboard.set_service_clients(
            redis_client=redis_client,
            neo4j_client=neo4j_client,
            qdrant_client=qdrant_client
        )
        
        services = await dashboard._collect_service_metrics()
        service_by_name = {s.name: s for s in services}
        
        assert service_by_name["MCP Server"].status == "healthy"
        assert service_by_name["Redis"].status == "healthy"
        assert service_by_name["Neo4j HTTP"].status == "unhealthy"
        assert service_by_name["Neo4j Bolt"].status == "unhealthy"
        assert service_by_name["Qdrant"].status == "healthy"
    
    @pytest.mark.asyncio
    async def test_partial_client_configuration(self, dashboard):
        """Test health checks with only some clients configured."""
        redis_client = Mock()
        redis_client.ping = Mock(return_value=True)
        
        # Only set Redis client, leave others None
        dashboard.set_service_clients(redis_client=redis_client)
        
        services = await dashboard._collect_service_metrics()
        service_by_name = {s.name: s for s in services}
        
        assert service_by_name["MCP Server"].status == "healthy"
        assert service_by_name["Redis"].status == "healthy"
        assert service_by_name["Neo4j HTTP"].status == "unknown"
        assert service_by_name["Neo4j Bolt"].status == "unknown"
        assert service_by_name["Qdrant"].status == "unknown"


class TestServiceMetricsDataStructure:
    """Test service metrics data structure and consistency."""
    
    @pytest.mark.asyncio
    async def test_service_metrics_structure(self, dashboard):
        """Test that service metrics have consistent structure."""
        services = await dashboard._collect_service_metrics()
        
        # Should always return 5 services
        assert len(services) == 5
        
        expected_services = {
            "MCP Server", "Redis", "Neo4j HTTP", "Neo4j Bolt", "Qdrant"
        }
        actual_services = {s.name for s in services}
        assert actual_services == expected_services
        
        # Each service should have required fields
        for service in services:
            assert isinstance(service, ServiceMetrics)
            assert hasattr(service, 'name')
            assert hasattr(service, 'status')
            assert hasattr(service, 'port')
            assert service.status in ['healthy', 'unhealthy', 'unknown']
            assert isinstance(service.port, int)
    
    @pytest.mark.asyncio
    async def test_service_port_assignments(self, dashboard):
        """Test that services have correct port assignments."""
        services = await dashboard._collect_service_metrics()
        service_by_name = {s.name: s for s in services}
        
        assert service_by_name["MCP Server"].port == 8000
        assert service_by_name["Redis"].port == 6379
        assert service_by_name["Neo4j HTTP"].port == 7474
        assert service_by_name["Neo4j Bolt"].port == 7687
        assert service_by_name["Qdrant"].port == 6333


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])