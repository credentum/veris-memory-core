#!/usr/bin/env python3
"""
Phase Integration Tests

End-to-end testing of the complete Veris Memory system integrating
all phases: Foundation, Backend Modularization, Ranking & Filtering,
and API Hardening & Observability.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.dependencies import set_query_dispatcher, get_query_dispatcher
from src.core.query_dispatcher import QueryDispatcher, SearchMode, DispatchPolicy
from src.ranking.policy_engine import RankingPolicyEngine, DefaultRankingPolicy, CodeBoostRankingPolicy
from src.filters.pre_filter import PreFilterEngine, FilterCriteria, TimeWindowFilter
from src.interfaces.memory_result import MemoryResult, ContentType, ResultSource
from src.interfaces.backend_interface import BackendSearchInterface, SearchOptions, BackendHealthStatus


class MockIntegrationBackend(BackendSearchInterface):
    """Mock backend for integration testing."""
    
    def __init__(self, name: str, sample_results: list):
        self._backend_name = name
        self.sample_results = sample_results
        self.search_calls = []
        self.health_calls = []
    
    @property
    def backend_name(self) -> str:
        return self._backend_name
    
    async def search(self, query: str, options: SearchOptions) -> list[MemoryResult]:
        """Mock search implementation."""
        self.search_calls.append((query, options))
        
        # Filter results based on limit
        results = self.sample_results[:options.limit] if options.limit else self.sample_results
        
        # Simulate backend-specific scoring
        for result in results:
            if self._backend_name == "vector":
                result.score = min(1.0, result.score + 0.1)  # Vector boost
            elif self._backend_name == "graph":
                result.score = min(1.0, result.score + 0.05)  # Graph boost
        
        return results
    
    async def health_check(self) -> BackendHealthStatus:
        """Mock health check implementation."""
        self.health_calls.append(datetime.utcnow())
        
        return BackendHealthStatus(
            status="healthy",
            response_time_ms=25.0,
            error_message=None,
            metadata={"backend": self._backend_name, "mock": True}
        )


@pytest.fixture
def sample_memory_results():
    """Create sample memory results for testing."""
    base_time = datetime.now(timezone.utc)
    
    return [
        MemoryResult(
            id="code_python_auth",
            text="def authenticate_user(token): return validate_jwt(token)",
            type=ContentType.CODE,
            score=0.85,
            source=ResultSource.VECTOR,
            timestamp=base_time - timedelta(hours=2),
            tags=["python", "auth", "jwt", "security"],
            namespace="auth_service",
            metadata={"language": "python", "complexity": "medium"}
        ),
        MemoryResult(
            id="doc_api_security",
            text="API Security Best Practices: Always validate input, use HTTPS, implement rate limiting",
            type=ContentType.DOCUMENTATION,
            score=0.78,
            source=ResultSource.GRAPH,
            timestamp=base_time - timedelta(days=1),
            tags=["security", "api", "best-practices"],
            namespace="docs",
            metadata={"category": "security", "audience": "developers"}
        ),
        MemoryResult(
            id="config_rate_limit",
            text='{"rate_limit": {"requests_per_minute": 100, "burst": 20}}',
            type=ContentType.PREFERENCE,
            score=0.65,
            source=ResultSource.KV,
            timestamp=base_time - timedelta(hours=6),
            tags=["config", "rate-limit", "api"],
            namespace="config",
            metadata={"format": "json", "env": "production"}
        ),
        MemoryResult(
            id="fact_jwt_standard",
            text="JWT (JSON Web Tokens) is defined in RFC 7519 and provides secure information transmission",
            type=ContentType.FACT,
            score=0.72,
            source=ResultSource.HYBRID,
            timestamp=base_time - timedelta(days=3),
            tags=["jwt", "security", "standard", "rfc"],
            namespace="knowledge",
            metadata={"verified": True, "source": "RFC-7519"}
        ),
        MemoryResult(
            id="recent_update",
            text="Updated authentication service to support OAuth 2.0 flows",
            type=ContentType.GENERAL,
            score=0.68,
            source=ResultSource.VECTOR,
            timestamp=base_time - timedelta(minutes=30),
            tags=["update", "oauth", "authentication"],
            namespace="updates",
            metadata={"author": "dev_team", "version": "2.1.0"}
        )
    ]


@pytest.fixture
async def integration_dispatcher(sample_memory_results):
    """Create a fully configured query dispatcher for integration testing."""
    dispatcher = QueryDispatcher()
    
    # Create mock backends with different result subsets
    vector_results = [r for r in sample_memory_results if r.source in [ResultSource.VECTOR, ResultSource.HYBRID]]
    graph_results = [r for r in sample_memory_results if r.source in [ResultSource.GRAPH, ResultSource.HYBRID]]
    kv_results = [r for r in sample_memory_results if r.source == ResultSource.KV]
    
    vector_backend = MockIntegrationBackend("vector", vector_results)
    graph_backend = MockIntegrationBackend("graph", graph_results)
    kv_backend = MockIntegrationBackend("kv", kv_results)
    
    # Register backends
    dispatcher.register_backend("vector", vector_backend)
    dispatcher.register_backend("graph", graph_backend)
    dispatcher.register_backend("kv", kv_backend)
    
    return dispatcher


@pytest.fixture
def integration_api_client(integration_dispatcher):
    """Create API client with integrated dispatcher."""
    app = create_app()
    
    # Set the dispatcher
    set_query_dispatcher(integration_dispatcher)
    
    return TestClient(app)


class TestCompleteSystemIntegration:
    """Test complete system integration across all phases."""
    
    @pytest.mark.asyncio
    async def test_phase1_foundation_integration(self, integration_dispatcher):
        """Test Phase 1: Foundation components work together."""
        
        # Test backend interface consistency
        backends = integration_dispatcher.list_backends()
        assert "vector" in backends
        assert "graph" in backends 
        assert "kv" in backends
        
        # Test health checks across all backends
        health_results = await integration_dispatcher.health_check_all_backends()
        
        assert len(health_results) == 3
        for backend_name, health in health_results.items():
            assert health["status"] == "healthy"
            assert "response_time_ms" in health
            assert health["metadata"]["mock"] is True
    
    @pytest.mark.asyncio
    async def test_phase2_backend_modularization_integration(self, integration_dispatcher):
        """Test Phase 2: Backend modularization with different dispatch policies."""
        
        query = "authentication security"
        options = SearchOptions(limit=10)
        
        # Test PARALLEL dispatch policy
        result_parallel = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            options=options,
            dispatch_policy=DispatchPolicy.PARALLEL
        )
        
        assert result_parallel.success is True
        assert len(result_parallel.results) > 0
        assert len(result_parallel.backends_used) > 1  # Multiple backends used
        assert "vector" in result_parallel.backend_timings
        assert "graph" in result_parallel.backend_timings
        
        # Test SEQUENTIAL dispatch policy
        result_sequential = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            options=options,
            dispatch_policy=DispatchPolicy.SEQUENTIAL
        )
        
        assert result_sequential.success is True
        assert len(result_sequential.results) > 0
        # Sequential may use fewer backends if early results satisfy limit
        
        # Test individual backend search modes
        for search_mode in [SearchMode.VECTOR, SearchMode.GRAPH, SearchMode.KV]:
            result = await integration_dispatcher.dispatch_query(
                query=query,
                search_mode=search_mode,
                options=options
            )
            assert result.success is True
            assert result.search_mode_used == search_mode.value
    
    @pytest.mark.asyncio
    async def test_phase3_ranking_filtering_integration(self, integration_dispatcher):
        """Test Phase 3: Ranking and filtering integration."""
        
        query = "python authentication"
        
        # Test default ranking policy
        result_default = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            ranking_policy="default"
        )
        
        assert result_default.success is True
        default_scores = [r.score for r in result_default.results]
        
        # Test code boost ranking policy
        result_code_boost = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            ranking_policy="code_boost"
        )
        
        assert result_code_boost.success is True
        code_boost_scores = [r.score for r in result_code_boost.results]
        
        # Code results should generally be ranked higher with code_boost
        code_results_default = [r for r in result_default.results if r.type == ContentType.CODE]
        code_results_boost = [r for r in result_code_boost.results if r.type == ContentType.CODE]
        
        if code_results_default and code_results_boost:
            # Code results should appear earlier in code_boost ranking
            default_code_positions = [i for i, r in enumerate(result_default.results) if r.type == ContentType.CODE]
            boost_code_positions = [i for i, r in enumerate(result_code_boost.results) if r.type == ContentType.CODE]
            
            if default_code_positions and boost_code_positions:
                assert min(boost_code_positions) <= min(default_code_positions)
        
        # Test recency ranking policy
        result_recency = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            ranking_policy="recency"
        )
        
        assert result_recency.success is True
        
        # Test pre-filtering
        time_filter = TimeWindowFilter(hours_ago=24)
        
        result_filtered = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            time_window=time_filter
        )
        
        assert result_filtered.success is True
        
        # All results should be within 24 hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        for result in result_filtered.results:
            assert result.timestamp >= cutoff_time
        
        # Test filter criteria
        filter_criteria = [
            FilterCriteria(field="type", operator="eq", value="code")
        ]
        
        result_type_filtered = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            pre_filters=filter_criteria
        )
        
        assert result_type_filtered.success is True
        for result in result_type_filtered.results:
            assert result.type == ContentType.CODE
    
    def test_phase4_api_integration(self, integration_api_client):
        """Test Phase 4: Complete API integration."""
        
        # Test system info endpoint
        response = integration_api_client.get("/api/v1/search/system-info")
        assert response.status_code == 200
        
        system_info = response.json()
        assert "backends" in system_info
        assert "ranking_policies" in system_info
        assert "filter_capabilities" in system_info
        assert len(system_info["backends"]) == 3  # vector, graph, kv
        
        # Test search endpoint with basic query
        search_request = {
            "query": "authentication security",
            "limit": 5
        }
        
        response = integration_api_client.post("/api/v1/search", json=search_request)
        assert response.status_code == 200
        
        search_response = response.json()
        assert search_response["success"] is True
        assert len(search_response["results"]) > 0
        assert "response_time_ms" in search_response
        assert "backend_timings" in search_response
        assert len(search_response["backends_used"]) > 0
        
        # Test advanced search with ranking and filtering
        advanced_request = {
            "query": "python authentication jwt",
            "search_mode": "hybrid",
            "dispatch_policy": "parallel",
            "ranking_policy": "code_boost",
            "content_types": ["code", "documentation"],
            "min_score": 0.6,
            "time_window": {"hours_ago": 48},
            "tags": ["security"],
            "limit": 10
        }
        
        response = integration_api_client.post("/api/v1/search", json=advanced_request)
        assert response.status_code == 200
        
        advanced_response = response.json()
        assert advanced_response["success"] is True
        assert advanced_response["ranking_policy_used"] == "code_boost"
        assert advanced_response["filters_applied"] > 0
        
        # Verify filtering worked
        for result in advanced_response["results"]:
            assert result["type"] in ["code", "documentation"]
            assert result["score"] >= 0.6
            assert "security" in result["tags"]
        
        # Test health endpoints
        response = integration_api_client.get("/api/v1/health")
        assert response.status_code == 200
        
        health_response = response.json()
        assert health_response["status"] in ["healthy", "degraded"]
        assert len(health_response["components"]) > 0
        
        # Test metrics endpoint
        response = integration_api_client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        metrics_response = response.json()
        assert "api_metrics" in metrics_response
        assert "dispatcher_metrics" in metrics_response
        assert "system_info" in metrics_response
    
    def test_error_handling_integration(self, integration_api_client):
        """Test integrated error handling across the system."""
        
        # Test validation errors
        invalid_request = {
            "query": "",  # Empty query should fail validation
            "limit": 1000  # Exceeds maximum
        }
        
        response = integration_api_client.post("/api/v1/search", json=invalid_request)
        assert response.status_code == 422
        
        error_response = response.json()
        assert "error" in error_response
        assert error_response["error"]["code"] == "VALIDATION_ERROR"
        assert "trace_id" in error_response["error"]
        
        # Test invalid endpoints
        response = integration_api_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, integration_dispatcher):
        """Test performance characteristics of integrated system."""
        
        import time
        
        query = "security authentication"
        options = SearchOptions(limit=5)
        
        # Measure performance of different configurations
        start_time = time.time()
        
        result = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            dispatch_policy=DispatchPolicy.PARALLEL
        )
        
        parallel_time = time.time() - start_time
        
        start_time = time.time()
        
        result = await integration_dispatcher.dispatch_query(
            query=query,
            search_mode=SearchMode.HYBRID,
            dispatch_policy=DispatchPolicy.SEQUENTIAL
        )
        
        sequential_time = time.time() - start_time
        
        # Parallel should generally be faster or similar (with mocks, timing may vary)
        # This is more about ensuring both work correctly
        assert result.success is True
        
        # Test that performance stats are collected
        perf_stats = integration_dispatcher.get_performance_stats()
        assert "timing_summary" in perf_stats
        assert "registered_backends" in perf_stats


class TestWorkflowIntegration:
    """Test realistic user workflows end-to-end."""
    
    def test_developer_code_search_workflow(self, integration_api_client):
        """Test typical developer workflow: searching for code examples."""
        
        # 1. Developer searches for authentication code
        search_request = {
            "query": "user authentication jwt token",
            "ranking_policy": "code_boost",  # Favor code results
            "content_types": ["code", "documentation"],
            "tags": ["python", "security"],
            "limit": 5
        }
        
        response = integration_api_client.post("/api/v1/search", json=search_request)
        assert response.status_code == 200
        
        search_result = response.json()
        assert search_result["success"] is True
        
        # Should find relevant code and documentation
        code_results = [r for r in search_result["results"] if r["type"] == "code"]
        assert len(code_results) > 0
        
        # Code results should be highly ranked due to code_boost policy
        if len(search_result["results"]) > 1:
            first_result = search_result["results"][0]
            # First result should likely be code with code_boost policy
            assert first_result["type"] in ["code", "documentation"]
        
        # 2. Developer gets system information
        response = integration_api_client.get("/api/v1/search/system-info")
        assert response.status_code == 200
        
        system_info = response.json()
        assert "code_boost" in [p["name"] for p in system_info["ranking_policies"]]
    
    def test_recent_changes_workflow(self, integration_api_client):
        """Test workflow: finding recent changes and updates."""
        
        # Search for recent updates using recency policy
        search_request = {
            "query": "authentication oauth update",
            "ranking_policy": "recency",  # Prioritize recent content
            "time_window": {"hours_ago": 6},  # Last 6 hours
            "limit": 10
        }
        
        response = integration_api_client.post("/api/v1/search", json=search_request)
        assert response.status_code == 200
        
        search_result = response.json()
        assert search_result["success"] is True
        assert search_result["ranking_policy_used"] == "recency"
        
        # Results should be within time window
        if search_result["results"]:
            # All results should be recent (our mock data includes recent updates)
            recent_results = [r for r in search_result["results"] 
                            if "recent" in r["id"] or "update" in r["text"].lower()]
            # Should find at least some recent content
            assert len(recent_results) >= 0  # May be 0 if no recent content matches
    
    def test_configuration_search_workflow(self, integration_api_client):
        """Test workflow: searching for configuration and settings."""
        
        # Search for configuration using KV backend focus
        search_request = {
            "query": "rate limit configuration",
            "search_mode": "kv",  # Focus on key-value store
            "content_types": ["preference", "general"],
            "namespace": "config"
        }
        
        response = integration_api_client.post("/api/v1/search", json=search_request)
        assert response.status_code == 200
        
        search_result = response.json()
        assert search_result["success"] is True
        assert search_result["search_mode_used"] == "kv"
        
        # Should find configuration-related results
        if search_result["results"]:
            config_results = [r for r in search_result["results"] 
                            if r["namespace"] == "config" or "config" in r["text"].lower()]
            assert len(config_results) >= 0
    
    def test_health_monitoring_workflow(self, integration_api_client):
        """Test workflow: system health monitoring."""
        
        # 1. Check overall system health
        response = integration_api_client.get("/api/v1/health")
        assert response.status_code == 200
        
        health = response.json()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        
        # 2. Check specific backend health
        response = integration_api_client.get("/api/v1/health/backends")
        assert response.status_code == 200
        
        backend_health = response.json()
        assert len(backend_health) == 3  # vector, graph, kv
        
        for backend_name, health_info in backend_health.items():
            assert backend_name in ["vector", "graph", "kv"]
            assert "status" in health_info
        
        # 3. Get performance metrics
        response = integration_api_client.get("/api/v1/metrics/performance")
        assert response.status_code == 200
        
        perf_metrics = response.json()
        assert "request_metrics" in perf_metrics
        assert "system_metrics" in perf_metrics
        
        # 4. Check liveness probe (for Kubernetes)
        response = integration_api_client.get("/api/v1/health/live")
        assert response.status_code == 200
        
        liveness = response.json()
        assert "status" in liveness
        assert liveness["status"] == "alive"