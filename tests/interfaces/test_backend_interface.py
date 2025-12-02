#!/usr/bin/env python3
"""
Tests for backend interface definitions.
"""

import pytest
from unittest.mock import AsyncMock
from typing import List, Dict, Any

from src.interfaces.backend_interface import (
    BackendSearchInterface,
    BackendSearchError,
    SearchOptions,
    BackendHealthStatus
)


class MockBackend(BackendSearchInterface):
    """Mock backend for testing interface compliance."""
    
    @property
    def backend_name(self) -> str:
        return "mock"
    
    async def search(self, query: str, options: SearchOptions) -> List[Dict[str, Any]]:
        return [
            {
                "id": "test_1",
                "text": f"Result for query: {query}",
                "score": 0.95,
                "metadata": {"source": "mock"}
            }
        ]
    
    async def health_check(self) -> BackendHealthStatus:
        return BackendHealthStatus(
            status="healthy",
            response_time_ms=5.2,
            metadata={"connections": 10}
        )


class TestSearchOptions:
    """Test SearchOptions model."""
    
    def test_default_values(self):
        options = SearchOptions()
        assert options.limit == 10
        assert options.filters == {}
        assert options.score_threshold == 0.0
        assert options.include_metadata is True
        assert options.namespace is None
    
    def test_custom_values(self):
        options = SearchOptions(
            limit=50,
            filters={"type": "code"},
            score_threshold=0.5,
            include_metadata=False,
            namespace="test_namespace"
        )
        assert options.limit == 50
        assert options.filters == {"type": "code"}
        assert options.score_threshold == 0.5
        assert options.include_metadata is False
        assert options.namespace == "test_namespace"
    
    def test_limit_validation(self):
        with pytest.raises(ValueError):
            SearchOptions(limit=0)  # Too low
        
        with pytest.raises(ValueError):
            SearchOptions(limit=1001)  # Too high
    
    def test_score_threshold_validation(self):
        with pytest.raises(ValueError):
            SearchOptions(score_threshold=-0.1)  # Too low
        
        with pytest.raises(ValueError):
            SearchOptions(score_threshold=1.1)  # Too high


class TestBackendHealthStatus:
    """Test BackendHealthStatus model."""
    
    def test_basic_health_status(self):
        status = BackendHealthStatus(status="healthy")
        assert status.status == "healthy"
        assert status.response_time_ms is None
        assert status.error_message is None
        assert status.metadata == {}
    
    def test_complete_health_status(self):
        status = BackendHealthStatus(
            status="degraded",
            response_time_ms=125.5,
            error_message="Connection timeout",
            metadata={"retry_count": 3}
        )
        assert status.status == "degraded"
        assert status.response_time_ms == 125.5
        assert status.error_message == "Connection timeout"
        assert status.metadata["retry_count"] == 3


class TestBackendSearchError:
    """Test BackendSearchError exception."""
    
    def test_basic_error(self):
        error = BackendSearchError("test_backend", "Connection failed")
        assert error.backend_name == "test_backend"
        assert error.message == "Connection failed"
        assert error.original_error is None
        assert "Backend 'test_backend' search failed: Connection failed" in str(error)
    
    def test_error_with_original(self):
        original = ValueError("Invalid parameter")
        error = BackendSearchError("test_backend", "Validation error", original)
        assert error.backend_name == "test_backend"
        assert error.message == "Validation error"
        assert error.original_error == original


class TestBackendSearchInterface:
    """Test the abstract backend interface."""
    
    @pytest.fixture
    def mock_backend(self):
        return MockBackend()
    
    @pytest.mark.asyncio
    async def test_interface_compliance(self, mock_backend):
        """Test that mock backend implements interface correctly."""
        assert mock_backend.backend_name == "mock"
        
        # Test search method
        options = SearchOptions(limit=5)
        results = await mock_backend.search("test query", options)
        
        assert len(results) == 1
        assert results[0]["id"] == "test_1"
        assert "test query" in results[0]["text"]
        assert results[0]["score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_backend):
        """Test health check functionality."""
        health = await mock_backend.health_check()
        
        assert health.status == "healthy"
        assert health.response_time_ms == 5.2
        assert health.metadata["connections"] == 10
    
    @pytest.mark.asyncio
    async def test_optional_methods(self, mock_backend):
        """Test optional initialize and cleanup methods."""
        # Should not raise errors
        await mock_backend.initialize()
        await mock_backend.cleanup()


class TestInterfaceValidation:
    """Test interface validation and error handling."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that the abstract interface cannot be instantiated."""
        with pytest.raises(TypeError):
            BackendSearchInterface()
    
    @pytest.mark.asyncio
    async def test_search_with_various_options(self):
        """Test search with different option combinations."""
        backend = MockBackend()
        
        # Test minimal options
        results = await backend.search("simple", SearchOptions())
        assert len(results) == 1
        
        # Test with filters
        options = SearchOptions(filters={"type": "code", "tags": ["python"]})
        results = await backend.search("filtered", options)
        assert len(results) == 1
        
        # Test with high score threshold
        options = SearchOptions(score_threshold=0.9)
        results = await backend.search("high threshold", options)
        assert len(results) == 1
    
    def test_search_options_json_schema(self):
        """Test that SearchOptions can be serialized/deserialized."""
        options = SearchOptions(
            limit=25,
            filters={"type": "documentation"},
            score_threshold=0.7,
            namespace="agent_123"
        )
        
        # Test JSON serialization
        json_data = options.json()
        assert "limit" in json_data
        assert "25" in json_data
        
        # Test deserialization
        restored = SearchOptions.parse_raw(json_data)
        assert restored.limit == 25
        assert restored.filters == {"type": "documentation"}
        assert restored.score_threshold == 0.7
        assert restored.namespace == "agent_123"