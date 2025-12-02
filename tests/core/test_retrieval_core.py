#!/usr/bin/env python3
"""
Comprehensive tests for RetrievalCore module.

Tests cover search modes, backend failures, health checks, and error handling
to ensure unified retrieval works correctly across API and MCP interfaces.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from src.core.retrieval_core import RetrievalCore, get_retrieval_core, set_retrieval_core, initialize_retrieval_core
from src.core.query_dispatcher import QueryDispatcher, SearchMode
from src.interfaces.backend_interface import SearchOptions, BackendHealthStatus
from src.interfaces.memory_result import (
    SearchResultResponse, MemoryResult, ContentType, ResultSource
)


class TestRetrievalCore:
    """Test cases for RetrievalCore functionality."""

    @pytest.fixture
    def mock_query_dispatcher(self):
        """Create a mock query dispatcher for testing."""
        dispatcher = AsyncMock(spec=QueryDispatcher)
        dispatcher.backends = {
            "vector": AsyncMock(),
            "graph": AsyncMock(), 
            "kv": AsyncMock()
        }
        return dispatcher

    @pytest.fixture
    def retrieval_core(self, mock_query_dispatcher):
        """Create a RetrievalCore instance for testing."""
        return RetrievalCore(mock_query_dispatcher)

    @pytest.mark.asyncio
    async def test_search_basic_functionality(self, retrieval_core, mock_query_dispatcher):
        """Test basic search functionality with successful response."""
        # Setup mock response
        mock_results = [
            MemoryResult(
                id="test-1",
                text="Test result 1",
                type=ContentType.GENERAL,
                score=0.95,
                source=ResultSource.VECTOR,
                tags=["test"],
                metadata={"key": "value1"}
            ),
            MemoryResult(
                id="test-2", 
                text="Test result 2",
                type=ContentType.CODE,
                score=0.85,
                source=ResultSource.GRAPH,
                tags=["test", "code"],
                metadata={"key": "value2"}
            )
        ]
        
        mock_response = SearchResultResponse(
            results=mock_results,
            total_count=2,
            backend_timings={"vector": 15.2, "graph": 8.5},
            backends_used=["vector", "graph"],
            trace_id="test-trace-123"
        )
        
        mock_query_dispatcher.search.return_value = mock_response
        
        # Execute search
        result = await retrieval_core.search(
            query="test query",
            limit=10,
            search_mode="hybrid"
        )
        
        # Verify results
        assert isinstance(result, SearchResultResponse)
        assert len(result.results) == 2
        assert result.total_count == 2
        assert result.backend_timings == {"vector": 15.2, "graph": 8.5}
        assert result.backends_used == ["vector", "graph"]
        assert result.trace_id == "test-trace-123"
        
        # Verify dispatcher was called correctly
        mock_query_dispatcher.search.assert_called_once()
        call_args = mock_query_dispatcher.search.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["options"].limit == 10
        assert call_args[1]["search_mode"] == SearchMode.HYBRID

    @pytest.mark.asyncio
    async def test_search_mode_validation(self, retrieval_core, mock_query_dispatcher):
        """Test that invalid search modes are handled gracefully."""
        mock_response = SearchResultResponse(results=[], total_count=0, backend_timings={}, backends_used=[])
        mock_query_dispatcher.search.return_value = mock_response
        
        # Test with invalid search mode
        result = await retrieval_core.search(
            query="test",
            search_mode="invalid_mode"  # This should default to hybrid
        )
        
        # Should have defaulted to hybrid mode
        call_args = mock_query_dispatcher.search.call_args
        assert call_args[1]["search_mode"] == SearchMode.HYBRID

    @pytest.mark.asyncio
    async def test_search_with_filters(self, retrieval_core, mock_query_dispatcher):
        """Test search with metadata filters and context type."""
        mock_response = SearchResultResponse(results=[], total_count=0, backend_timings={}, backends_used=[])
        mock_query_dispatcher.search.return_value = mock_response
        
        await retrieval_core.search(
            query="filtered query",
            limit=5,
            search_mode="vector",
            context_type="code",
            metadata_filters={"author": "test_user", "language": "python"},
            score_threshold=0.7
        )
        
        # Verify filters were applied correctly
        call_args = mock_query_dispatcher.search.call_args
        search_options = call_args[1]["options"]
        
        assert search_options.limit == 5
        assert search_options.score_threshold == 0.7
        assert search_options.filters["type"] == "code"
        assert search_options.filters["author"] == "test_user"
        assert search_options.filters["language"] == "python"

    @pytest.mark.asyncio
    async def test_search_error_handling(self, retrieval_core, mock_query_dispatcher):
        """Test that search errors are properly propagated."""
        # Setup dispatcher to raise an exception
        mock_query_dispatcher.search.side_effect = RuntimeError("Backend failure")
        
        # Verify exception is propagated
        with pytest.raises(RuntimeError, match="Backend failure"):
            await retrieval_core.search("test query")

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, retrieval_core):
        """Test health check when all backends are healthy."""
        # Setup mock backends with healthy status
        for backend_name, backend in retrieval_core.dispatcher.backends.items():
            backend.health_check.return_value = BackendHealthStatus(
                status="healthy",
                response_time_ms=10.0,
                error_message=None,
                metadata={"connection": "ok"}
            )
        
        health = await retrieval_core.health_check()
        
        assert health["overall_status"] == "healthy"
        assert health["healthy_backends"] == 3
        assert health["total_backends"] == 3
        
        # Check individual backend health
        assert health["backends"]["vector"]["status"] == "healthy"
        assert health["backends"]["graph"]["status"] == "healthy" 
        assert health["backends"]["kv"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_partial_failure(self, retrieval_core):
        """Test health check when some backends are unhealthy."""
        # Setup mixed health statuses
        retrieval_core.dispatcher.backends["vector"].health_check.return_value = BackendHealthStatus(
            status="healthy", response_time_ms=5.0
        )
        retrieval_core.dispatcher.backends["graph"].health_check.return_value = BackendHealthStatus(
            status="unhealthy", response_time_ms=-1, error_message="Connection failed"
        )
        retrieval_core.dispatcher.backends["kv"].health_check.return_value = BackendHealthStatus(
            status="degraded", response_time_ms=50.0
        )
        
        health = await retrieval_core.health_check()
        
        assert health["overall_status"] == "degraded"  # At least one healthy backend
        assert health["healthy_backends"] == 1
        assert health["total_backends"] == 3
        assert health["backends"]["vector"]["status"] == "healthy"
        assert health["backends"]["graph"]["status"] == "unhealthy"
        assert health["backends"]["kv"]["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_all_failed(self, retrieval_core):
        """Test health check when all backends fail."""
        # Setup all backends to fail health checks
        for backend in retrieval_core.dispatcher.backends.values():
            backend.health_check.side_effect = RuntimeError("Health check failed")
        
        health = await retrieval_core.health_check()
        
        assert health["overall_status"] == "unhealthy"
        assert health["healthy_backends"] == 0
        assert health["total_backends"] == 3
        
        # All backends should show as unhealthy
        for backend_name in ["vector", "graph", "kv"]:
            assert health["backends"][backend_name]["status"] == "unhealthy"
            assert "Health check failed" in health["backends"][backend_name]["error_message"]

    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, retrieval_core):
        """Test health check handles unexpected exceptions gracefully."""
        # Make the dispatcher itself unavailable
        retrieval_core.dispatcher = None
        
        health = await retrieval_core.health_check()
        
        assert health["overall_status"] == "unhealthy"
        assert health["healthy_backends"] == 0
        assert health["total_backends"] == 0
        assert "error" in health


class TestRetrievalCoreGlobalState:
    """Test the global state management for RetrievalCore."""

    def test_global_instance_management(self):
        """Test setting and getting global RetrievalCore instance."""
        # Initially should be None
        assert get_retrieval_core() is None
        
        # Create and set instance
        mock_dispatcher = MagicMock()
        core = RetrievalCore(mock_dispatcher)
        set_retrieval_core(core)
        
        # Should now return the instance
        assert get_retrieval_core() is core
        
        # Clear for other tests
        set_retrieval_core(None)

    def test_initialize_retrieval_core(self):
        """Test the initialization helper function."""
        mock_dispatcher = MagicMock()
        
        core = initialize_retrieval_core(mock_dispatcher)
        
        assert isinstance(core, RetrievalCore)
        assert core.dispatcher is mock_dispatcher
        assert get_retrieval_core() is core
        
        # Clear for other tests
        set_retrieval_core(None)


class TestSearchOptions:
    """Test SearchOptions construction and validation."""
    
    def test_search_options_defaults(self):
        """Test default SearchOptions values."""
        options = SearchOptions()
        
        assert options.limit == 10
        assert options.score_threshold == 0.0
        assert options.namespace is None
        assert options.filters == {}

    def test_search_options_custom_values(self):
        """Test SearchOptions with custom values."""
        options = SearchOptions(
            limit=25,
            score_threshold=0.5,
            namespace="test_ns",
            filters={"type": "code", "author": "user"}
        )
        
        assert options.limit == 25
        assert options.score_threshold == 0.5
        assert options.namespace == "test_ns"
        assert options.filters == {"type": "code", "author": "user"}


class TestSearchModeEnum:
    """Test SearchMode enum handling."""
    
    def test_search_mode_values(self):
        """Test that all expected search modes exist."""
        assert SearchMode.VECTOR == "vector"
        assert SearchMode.GRAPH == "graph" 
        assert SearchMode.KV == "kv"
        assert SearchMode.HYBRID == "hybrid"
        assert SearchMode.AUTO == "auto"

    def test_search_mode_from_string(self):
        """Test creating SearchMode from string."""
        mode = SearchMode("hybrid")
        assert mode == SearchMode.HYBRID
        
        # Test invalid mode raises ValueError
        with pytest.raises(ValueError):
            SearchMode("invalid")


@pytest.mark.integration
class TestRetrievalCoreIntegration:
    """Integration tests with real components (when available)."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_simulation(self):
        """Simulate end-to-end search with realistic data flow."""
        # This test simulates realistic usage without requiring actual backends
        mock_dispatcher = AsyncMock()
        
        # Create realistic search response
        mock_results = [
            MemoryResult(
                id="doc-123",
                text="This is a test document about machine learning algorithms",
                type=ContentType.DOCUMENTATION,
                score=0.92,
                source=ResultSource.VECTOR,
                tags=["ml", "algorithms", "documentation"],
                metadata={
                    "created_date": "2025-08-26",
                    "author": "researcher",
                    "word_count": 1500
                },
                title="Machine Learning Algorithms Guide",
                user_id="user-456"
            )
        ]
        
        mock_response = SearchResultResponse(
            results=mock_results,
            total_count=1,
            backend_timings={"vector": 12.5, "graph": 3.2},
            backends_used=["vector", "graph"],
            trace_id="integration-test-789"
        )
        
        mock_dispatcher.search.return_value = mock_response
        
        core = RetrievalCore(mock_dispatcher)
        
        # Execute realistic search
        result = await core.search(
            query="machine learning documentation",
            limit=20,
            search_mode="hybrid",
            metadata_filters={"author": "researcher"},
            score_threshold=0.8
        )
        
        # Verify realistic response structure
        assert len(result.results) == 1
        assert result.results[0].title == "Machine Learning Algorithms Guide"
        assert result.results[0].score == 0.92
        assert result.results[0].source == ResultSource.VECTOR
        assert "ml" in result.results[0].tags
        assert result.backend_timings["vector"] > 0
        assert "vector" in result.backends_used