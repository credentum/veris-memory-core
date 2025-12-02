#!/usr/bin/env python3
"""
Tests for QueryDispatcher implementation.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from typing import List, Dict, Any

from src.core.query_dispatcher import QueryDispatcher, SearchMode, DispatchPolicy
from src.interfaces.backend_interface import BackendSearchInterface, SearchOptions, BackendHealthStatus
from src.interfaces.memory_result import MemoryResult, ResultSource, ContentType


class MockBackend(BackendSearchInterface):
    """Mock backend for testing."""
    
    def __init__(self, name: str, results: List[MemoryResult] = None):
        self._name = name
        self._results = results or []
        self.search_called_count = 0
        self.last_query = None
        self.last_options = None
    
    @property
    def backend_name(self) -> str:
        return self._name
    
    async def search(self, query: str, options: SearchOptions) -> List[MemoryResult]:
        self.search_called_count += 1
        self.last_query = query
        self.last_options = options
        return self._results.copy()
    
    async def health_check(self) -> BackendHealthStatus:
        return BackendHealthStatus(status="healthy", response_time_ms=5.0)


@pytest.fixture
def mock_vector_results():
    """Create mock vector search results."""
    return [
        MemoryResult(
            id="vec_1",
            text="Vector result 1",
            score=0.9,
            source=ResultSource.VECTOR,
            type=ContentType.GENERAL
        ),
        MemoryResult(
            id="vec_2",
            text="Vector result 2", 
            score=0.8,
            source=ResultSource.VECTOR,
            type=ContentType.CODE
        )
    ]


@pytest.fixture
def mock_graph_results():
    """Create mock graph search results."""
    return [
        MemoryResult(
            id="graph_1",
            text="Graph result 1",
            score=1.0,
            source=ResultSource.GRAPH,
            type=ContentType.FACT
        )
    ]


@pytest.fixture
def mock_kv_results():
    """Create mock KV search results."""
    return [
        MemoryResult(
            id="kv_1",
            text="KV result 1",
            score=0.7,
            source=ResultSource.KV,
            type=ContentType.PERSONAL_INFO
        )
    ]


@pytest.fixture
def mock_text_results():
    """Create mock text search results (Issue #311)."""
    return [
        MemoryResult(
            id="text_1",
            text="Text result 1",
            score=0.85,
            source=ResultSource.TEXT,
            type=ContentType.DOCUMENTATION
        ),
        MemoryResult(
            id="text_2",
            text="Text result 2",
            score=0.75,
            source=ResultSource.TEXT,
            type=ContentType.CODE
        )
    ]


class TestQueryDispatcher:
    """Test QueryDispatcher functionality."""
    
    def test_initialization(self):
        """Test dispatcher initialization."""
        dispatcher = QueryDispatcher()
        
        assert len(dispatcher.backends) == 0
        assert dispatcher.list_backends() == []
        assert dispatcher.default_policy == DispatchPolicy.PARALLEL
    
    def test_backend_registration(self, mock_vector_results):
        """Test backend registration and management."""
        dispatcher = QueryDispatcher()
        vector_backend = MockBackend("vector", mock_vector_results)
        
        # Register backend
        dispatcher.register_backend("vector", vector_backend)
        
        assert "vector" in dispatcher.backends
        assert dispatcher.get_backend("vector") == vector_backend
        assert dispatcher.list_backends() == ["vector"]
    
    def test_backend_registration_name_mismatch(self, mock_vector_results):
        """Test warning when registered name doesn't match backend name."""
        dispatcher = QueryDispatcher()
        vector_backend = MockBackend("vector", mock_vector_results)
        
        # This should work but log a warning
        dispatcher.register_backend("wrong_name", vector_backend)
        assert "wrong_name" in dispatcher.backends
    
    def test_backend_unregistration(self, mock_vector_results):
        """Test backend unregistration."""
        dispatcher = QueryDispatcher()
        vector_backend = MockBackend("vector", mock_vector_results)
        
        dispatcher.register_backend("vector", vector_backend)
        assert len(dispatcher.backends) == 1
        
        # Unregister existing backend
        result = dispatcher.unregister_backend("vector")
        assert result is True
        assert len(dispatcher.backends) == 0
        
        # Try to unregister non-existent backend
        result = dispatcher.unregister_backend("nonexistent")
        assert result is False
    
    def test_invalid_backend_registration(self):
        """Test error handling for invalid backend registration."""
        dispatcher = QueryDispatcher()
        
        with pytest.raises(ValueError, match="must implement BackendSearchInterface"):
            dispatcher.register_backend("invalid", "not_a_backend")
    
    @pytest.mark.asyncio
    async def test_single_backend_search(self, mock_vector_results):
        """Test search with single backend."""
        dispatcher = QueryDispatcher()
        vector_backend = MockBackend("vector", mock_vector_results)
        dispatcher.register_backend("vector", vector_backend)
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.VECTOR
        )
        
        assert response.success is True
        assert len(response.results) == 2
        assert response.search_mode_used == "vector"
        assert response.total_count == 2
        assert "vector" in response.backends_used
        assert "vector" in response.backend_timings
        
        # Check that backend was called correctly
        assert vector_backend.search_called_count == 1
        assert vector_backend.last_query == "test query"
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, mock_vector_results, mock_graph_results, mock_kv_results):
        """Test hybrid search across multiple backends."""
        dispatcher = QueryDispatcher()
        
        # Register all backends
        vector_backend = MockBackend("vector", mock_vector_results)
        graph_backend = MockBackend("graph", mock_graph_results)
        kv_backend = MockBackend("kv", mock_kv_results)
        
        dispatcher.register_backend("vector", vector_backend)
        dispatcher.register_backend("graph", graph_backend)
        dispatcher.register_backend("kv", kv_backend)
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.HYBRID
        )
        
        assert response.success is True
        assert len(response.results) == 4  # 2 + 1 + 1
        assert response.search_mode_used == "hybrid"
        assert len(response.backends_used) == 3
        assert set(response.backends_used) == {"vector", "graph", "kv"}
        
        # All backends should have been called
        assert vector_backend.search_called_count == 1
        assert graph_backend.search_called_count == 1
        assert kv_backend.search_called_count == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_source_breakdown(
        self, mock_vector_results, mock_graph_results, mock_kv_results, mock_text_results
    ):
        """Test that source_breakdown field correctly counts results from each backend (Issue #311)."""
        dispatcher = QueryDispatcher()

        # Register all 4 backends (vector, graph, text, kv)
        vector_backend = MockBackend("vector", mock_vector_results)
        graph_backend = MockBackend("graph", mock_graph_results)
        text_backend = MockBackend("text", mock_text_results)
        kv_backend = MockBackend("kv", mock_kv_results)

        dispatcher.register_backend("vector", vector_backend)
        dispatcher.register_backend("graph", graph_backend)
        dispatcher.register_backend("text", text_backend)
        dispatcher.register_backend("kv", kv_backend)

        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.HYBRID
        )

        assert response.success is True
        assert response.search_mode_used == "hybrid"

        # Verify source_breakdown field exists
        assert hasattr(response, "source_breakdown")
        assert response.source_breakdown is not None
        assert isinstance(response.source_breakdown, dict)

        # Verify source counts match actual results
        expected_breakdown = {
            "vector": 2,  # mock_vector_results has 2
            "graph": 1,   # mock_graph_results has 1
            "text": 2,    # mock_text_results has 2
            "kv": 1       # mock_kv_results has 1
        }

        assert response.source_breakdown == expected_breakdown

        # Verify sum of breakdown matches total results
        total_from_breakdown = sum(response.source_breakdown.values())
        assert total_from_breakdown == len(response.results)

        # Verify each source is represented in results
        result_sources = [r.source for r in response.results]
        for source, count in response.source_breakdown.items():
            assert result_sources.count(source) == count

    @pytest.mark.asyncio
    async def test_search_with_options(self, mock_vector_results):
        """Test search with custom options."""
        dispatcher = QueryDispatcher()
        vector_backend = MockBackend("vector", mock_vector_results)
        dispatcher.register_backend("vector", vector_backend)
        
        options = SearchOptions(
            limit=1,
            score_threshold=0.5,
            namespace="test_namespace"
        )
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.VECTOR,
            options=options
        )
        
        assert response.success is True
        assert len(response.results) <= 1  # Limited by options
        
        # Check that options were passed to backend
        assert vector_backend.last_options.limit == 1
        assert vector_backend.last_options.score_threshold == 0.5
        assert vector_backend.last_options.namespace == "test_namespace"
    
    @pytest.mark.asyncio
    async def test_no_backends_available(self):
        """Test behavior when no backends are available."""
        dispatcher = QueryDispatcher()
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.VECTOR
        )
        
        assert response.success is False
        assert len(response.results) == 0
        assert "No backends available" in response.message
    
    @pytest.mark.asyncio
    async def test_backend_failure_handling(self):
        """Test handling of backend failures."""
        dispatcher = QueryDispatcher()
        
        # Create a backend that will fail
        failing_backend = MockBackend("vector", [])
        failing_backend.search = AsyncMock(side_effect=Exception("Backend failed"))
        
        dispatcher.register_backend("vector", failing_backend)
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.VECTOR
        )
        
        # Backend failures in parallel mode are gracefully handled
        assert response.success is True
        assert len(response.results) == 0
        assert response.backend_timings["vector"] == 0.0  # Failed backend has 0 timing
    
    @pytest.mark.asyncio
    async def test_sequential_dispatch_policy(self, mock_vector_results, mock_graph_results):
        """Test sequential dispatch policy."""
        dispatcher = QueryDispatcher()
        
        vector_backend = MockBackend("vector", mock_vector_results)
        graph_backend = MockBackend("graph", mock_graph_results)
        
        dispatcher.register_backend("vector", vector_backend)
        dispatcher.register_backend("graph", graph_backend)
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.HYBRID,
            dispatch_policy=DispatchPolicy.SEQUENTIAL,
            options=SearchOptions(limit=2)  # Should stop after getting enough results
        )
        
        assert response.success is True
        assert len(response.results) <= 2
        
        # Vector has higher priority, should be called first
        assert vector_backend.search_called_count == 1
        # Graph might not be called if vector provides enough results
    
    @pytest.mark.asyncio
    async def test_fallback_dispatch_policy(self, mock_graph_results):
        """Test fallback dispatch policy."""
        dispatcher = QueryDispatcher()
        
        # Create a failing backend and a working one
        failing_backend = MockBackend("vector", [])
        failing_backend.search = AsyncMock(side_effect=Exception("Failed"))
        
        graph_backend = MockBackend("graph", mock_graph_results)
        
        dispatcher.register_backend("vector", failing_backend)
        dispatcher.register_backend("graph", graph_backend)
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.HYBRID,
            dispatch_policy=DispatchPolicy.FALLBACK
        )
        
        assert response.success is True
        assert len(response.results) == 1
        assert response.results[0].source == ResultSource.GRAPH
    
    @pytest.mark.asyncio
    async def test_auto_search_mode_key_query(self, mock_kv_results):
        """Test AUTO mode with key-based query."""
        dispatcher = QueryDispatcher()
        
        vector_backend = MockBackend("vector", [])
        kv_backend = MockBackend("kv", mock_kv_results)
        
        dispatcher.register_backend("vector", vector_backend)
        dispatcher.register_backend("kv", kv_backend)
        
        # Key-based query should prefer KV backend
        response = await dispatcher.dispatch_query(
            "state:user_123",
            search_mode=SearchMode.AUTO
        )
        
        assert response.success is True
        # Should have used KV backend
        assert kv_backend.search_called_count == 1
    
    @pytest.mark.asyncio
    async def test_auto_search_mode_semantic_query(self, mock_vector_results):
        """Test AUTO mode with semantic query."""
        dispatcher = QueryDispatcher()
        
        vector_backend = MockBackend("vector", mock_vector_results)
        kv_backend = MockBackend("kv", [])
        
        dispatcher.register_backend("vector", vector_backend)
        dispatcher.register_backend("kv", kv_backend)
        
        # Multi-word semantic query should prefer vector backend
        response = await dispatcher.dispatch_query(
            "find user preferences about colors",
            search_mode=SearchMode.AUTO
        )
        
        assert response.success is True
        # Should have used vector backend
        assert vector_backend.search_called_count == 1
    
    @pytest.mark.asyncio
    async def test_health_check_all_backends(self, mock_vector_results):
        """Test health checking all backends."""
        dispatcher = QueryDispatcher()
        
        vector_backend = MockBackend("vector", mock_vector_results)
        dispatcher.register_backend("vector", vector_backend)
        
        health_results = await dispatcher.health_check_all_backends()
        
        assert "vector" in health_results
        assert health_results["vector"]["status"] == "healthy"
        assert health_results["vector"]["response_time_ms"] == 5.0
    
    @pytest.mark.asyncio 
    async def test_health_check_with_failing_backend(self):
        """Test health check with failing backend."""
        dispatcher = QueryDispatcher()
        
        failing_backend = MockBackend("failing", [])
        failing_backend.health_check = AsyncMock(side_effect=Exception("Health check failed"))
        
        dispatcher.register_backend("failing", failing_backend)
        
        health_results = await dispatcher.health_check_all_backends()
        
        assert "failing" in health_results
        assert health_results["failing"]["status"] == "error"
        assert "Health check failed" in health_results["failing"]["error_message"]
    
    def test_performance_stats(self, mock_vector_results):
        """Test getting performance statistics."""
        dispatcher = QueryDispatcher()
        vector_backend = MockBackend("vector", mock_vector_results)
        dispatcher.register_backend("vector", vector_backend)
        
        stats = dispatcher.get_performance_stats()
        
        assert "timing_summary" in stats
        assert "registered_backends" in stats
        assert "backend_priorities" in stats
        assert "default_policy" in stats
        
        assert stats["registered_backends"] == ["vector"]
        assert stats["default_policy"] == "parallel"
    
    @pytest.mark.asyncio
    async def test_result_ranking(self, mock_vector_results, mock_graph_results):
        """Test result ranking functionality."""
        dispatcher = QueryDispatcher()
        
        # Modify scores to test ranking
        mock_vector_results[0].score = 0.6  # Lower than graph
        mock_graph_results[0].score = 0.9   # Higher than vector
        
        vector_backend = MockBackend("vector", mock_vector_results)
        graph_backend = MockBackend("graph", mock_graph_results)
        
        dispatcher.register_backend("vector", vector_backend)
        dispatcher.register_backend("graph", graph_backend)
        
        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.HYBRID
        )
        
        assert response.success is True
        assert len(response.results) >= 2
        
        # Results should be sorted by score (highest first)
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_duplicate_result_handling(self, mock_vector_results):
        """Test handling of duplicate results from multiple backends."""
        dispatcher = QueryDispatcher()

        # Create backends that return overlapping results
        duplicate_result = MemoryResult(
            id="duplicate",
            text="Duplicate result",
            score=0.8,
            source=ResultSource.VECTOR,
            type=ContentType.GENERAL
        )

        backend1 = MockBackend("vector", [duplicate_result])
        backend2 = MockBackend("graph", [duplicate_result])  # Same ID

        dispatcher.register_backend("vector", backend1)
        dispatcher.register_backend("graph", backend2)

        response = await dispatcher.dispatch_query(
            "test query",
            search_mode=SearchMode.HYBRID
        )

        assert response.success is True
        # Should have deduplicated results
        result_ids = [r.id for r in response.results]
        assert len(set(result_ids)) == len(result_ids)  # No duplicates


class TestQueryDispatcherSearchByEmbedding:
    """Test suite for search_by_embedding method (HyDE support)."""

    @pytest.fixture
    def mock_vector_results(self):
        """Create mock vector search results."""
        return [
            MemoryResult(
                id="hyde_1",
                text="Result from HyDE embedding search",
                score=0.92,
                source=ResultSource.VECTOR,
                type=ContentType.DOCUMENTATION
            ),
            MemoryResult(
                id="hyde_2",
                text="Another HyDE result",
                score=0.85,
                source=ResultSource.VECTOR,
                type=ContentType.GENERAL
            )
        ]

    @pytest.mark.asyncio
    async def test_search_by_embedding_success(self, mock_vector_results):
        """Test successful search by embedding."""
        dispatcher = QueryDispatcher()

        # Create a mock vector backend with search_by_embedding support
        vector_backend = MockBackend("vector", mock_vector_results)
        vector_backend.search_by_embedding = AsyncMock(return_value=mock_vector_results)

        dispatcher.register_backend("vector", vector_backend)

        embedding = [0.1, 0.2, 0.3] * 512
        options = SearchOptions(limit=10)

        response = await dispatcher.search_by_embedding(
            embedding=embedding,
            options=options,
            search_mode=SearchMode.VECTOR
        )

        assert response.success is True
        assert len(response.results) == 2
        assert response.search_mode_used == "hyde"
        assert "vector" in response.backends_used
        assert "vector_hyde" in response.backend_timings
        vector_backend.search_by_embedding.assert_called_once_with(embedding, options)

    @pytest.mark.asyncio
    async def test_search_by_embedding_no_vector_backend(self):
        """Test search by embedding when vector backend is not available."""
        dispatcher = QueryDispatcher()

        # Register only graph backend
        graph_backend = MockBackend("graph", [])
        dispatcher.register_backend("graph", graph_backend)

        embedding = [0.1, 0.2, 0.3]
        options = SearchOptions(limit=10)

        response = await dispatcher.search_by_embedding(
            embedding=embedding,
            options=options,
            search_mode=SearchMode.VECTOR
        )

        assert response.success is False
        assert len(response.results) == 0
        assert "Vector backend not available" in response.message

    @pytest.mark.asyncio
    async def test_search_by_embedding_error_handling(self):
        """Test error handling in search by embedding."""
        dispatcher = QueryDispatcher()

        # Create a vector backend that fails
        vector_backend = MockBackend("vector", [])
        vector_backend.search_by_embedding = AsyncMock(
            side_effect=Exception("Embedding search failed")
        )

        dispatcher.register_backend("vector", vector_backend)

        embedding = [0.1, 0.2, 0.3]
        options = SearchOptions(limit=10)

        response = await dispatcher.search_by_embedding(
            embedding=embedding,
            options=options,
            search_mode=SearchMode.VECTOR
        )

        assert response.success is False
        assert len(response.results) == 0
        assert "failed" in response.message.lower()

    @pytest.mark.asyncio
    async def test_search_by_embedding_empty_results(self):
        """Test search by embedding with no results."""
        dispatcher = QueryDispatcher()

        vector_backend = MockBackend("vector", [])
        vector_backend.search_by_embedding = AsyncMock(return_value=[])

        dispatcher.register_backend("vector", vector_backend)

        embedding = [0.1, 0.2, 0.3]
        options = SearchOptions(limit=10)

        response = await dispatcher.search_by_embedding(
            embedding=embedding,
            options=options,
            search_mode=SearchMode.VECTOR
        )

        assert response.success is True
        assert len(response.results) == 0
        assert response.total_count == 0

    @pytest.mark.asyncio
    async def test_search_by_embedding_with_options(self, mock_vector_results):
        """Test search by embedding respects search options."""
        dispatcher = QueryDispatcher()

        vector_backend = MockBackend("vector", mock_vector_results)
        vector_backend.search_by_embedding = AsyncMock(return_value=mock_vector_results)

        dispatcher.register_backend("vector", vector_backend)

        embedding = [0.1, 0.2, 0.3]
        options = SearchOptions(
            limit=1,
            score_threshold=0.9,
            namespace="test_ns"
        )

        await dispatcher.search_by_embedding(
            embedding=embedding,
            options=options,
            search_mode=SearchMode.VECTOR
        )

        # Verify options were passed to backend
        call_args = vector_backend.search_by_embedding.call_args
        assert call_args[0][1].limit == 1
        assert call_args[0][1].score_threshold == 0.9
        assert call_args[0][1].namespace == "test_ns"