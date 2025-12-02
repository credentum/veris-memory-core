#!/usr/bin/env python3
"""
Tests for normalized memory result schema.
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from src.interfaces.memory_result import (
    MemoryResult,
    SearchResultResponse,
    ResultSource,
    ContentType,
    merge_results,
    sort_results_by_score,
    filter_results_by_threshold
)


class TestMemoryResult:
    """Test MemoryResult model validation and functionality."""
    
    def test_minimal_memory_result(self):
        """Test creating MemoryResult with minimal required fields."""
        result = MemoryResult(
            id="test_123",
            text="Test content",
            source=ResultSource.VECTOR
        )
        
        assert result.id == "test_123"
        assert result.text == "Test content"
        assert result.type == ContentType.GENERAL
        assert result.score == 1.0
        assert result.source == ResultSource.VECTOR
        assert result.tags == []
        assert result.metadata == {}
        assert isinstance(result.timestamp, datetime)
    
    def test_complete_memory_result(self):
        """Test creating MemoryResult with all fields."""
        timestamp = datetime.now(timezone.utc)
        result = MemoryResult(
            id="complete_456",
            text="Complete test content with all fields",
            type=ContentType.CODE,
            score=0.85,
            timestamp=timestamp,
            source=ResultSource.HYBRID,
            tags=["python", "function", "test"],
            metadata={"confidence": 0.9, "method": "semantic_search"},
            namespace="agent_789",
            title="Test Function",
            user_id="user_123"
        )
        
        assert result.id == "complete_456"
        assert result.text == "Complete test content with all fields"
        assert result.type == ContentType.CODE
        assert result.score == 0.85
        assert result.timestamp == timestamp
        assert result.source == ResultSource.HYBRID
        assert result.tags == ["python", "function", "test"]
        assert result.metadata["confidence"] == 0.9
        assert result.namespace == "agent_789"
        assert result.title == "Test Function"
        assert result.user_id == "user_123"
    
    def test_text_validation(self):
        """Test text field validation."""
        # Empty text should fail
        with pytest.raises(ValueError, match="Text field cannot be empty"):
            MemoryResult(
                id="test",
                text="",
                source=ResultSource.VECTOR
            )
        
        # Whitespace-only text should fail
        with pytest.raises(ValueError, match="Text field cannot be empty"):
            MemoryResult(
                id="test",
                text="   ",
                source=ResultSource.VECTOR
            )
        
        # Text with content should pass and be stripped
        result = MemoryResult(
            id="test",
            text="  valid content  ",
            source=ResultSource.VECTOR
        )
        assert result.text == "valid content"
    
    def test_score_validation(self):
        """Test score field validation."""
        # Valid scores should pass
        for score in [0.0, 0.5, 1.0]:
            result = MemoryResult(
                id="test",
                text="test content",
                score=score,
                source=ResultSource.VECTOR
            )
            assert result.score == score
        
        # Invalid scores should fail
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            MemoryResult(
                id="test",
                text="test content",
                score=-0.1,
                source=ResultSource.VECTOR
            )
        
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            MemoryResult(
                id="test",
                text="test content",
                score=1.1,
                source=ResultSource.VECTOR
            )
    
    def test_tags_normalization(self):
        """Test tags normalization and deduplication."""
        result = MemoryResult(
            id="test",
            text="test content",
            source=ResultSource.VECTOR,
            tags=["Python", "FUNCTION", "python", "  test  ", "", "function"]
        )
        
        # Tags should be lowercase, stripped, and deduplicated
        expected_tags = ["python", "function", "test"]
        assert sorted(result.tags) == sorted(expected_tags)
    
    def test_enum_values(self):
        """Test enum value handling."""
        result = MemoryResult(
            id="test",
            text="test content",
            type=ContentType.DOCUMENTATION,
            source=ResultSource.GRAPH
        )
        
        # Enums should be stored as their values
        assert result.type == "documentation"
        assert result.source == "graph"
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        original = MemoryResult(
            id="json_test",
            text="JSON serialization test",
            type=ContentType.FACT,
            score=0.75,
            source=ResultSource.KV,
            tags=["serialization", "json"],
            metadata={"test": True}
        )
        
        # Serialize to JSON
        json_str = original.json()
        assert "json_test" in json_str
        assert "serialization" in json_str
        
        # Deserialize from JSON
        restored = MemoryResult.parse_raw(json_str)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.type == original.type
        assert restored.score == original.score
        assert restored.source == original.source
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata


class TestSearchResultResponse:
    """Test SearchResultResponse model."""
    
    def test_minimal_response(self):
        """Test minimal search response."""
        response = SearchResultResponse(
            success=True,
            search_mode_used="vector"
        )
        
        assert response.success is True
        assert response.results == []
        assert response.total_count == 0
        assert response.search_mode_used == "vector"
        assert response.message == ""
    
    def test_complete_response(self):
        """Test complete search response with results."""
        results = [
            MemoryResult(
                id="result_1",
                text="First result",
                source=ResultSource.VECTOR,
                score=0.9
            ),
            MemoryResult(
                id="result_2", 
                text="Second result",
                source=ResultSource.GRAPH,
                score=0.8
            )
        ]
        
        response = SearchResultResponse(
            success=True,
            results=results,
            total_count=2,
            search_mode_used="hybrid",
            message="Found 2 results",
            response_time_ms=45.2,
            trace_id="trace_abc_123",
            backend_timings={"vector": 25.1, "graph": 20.1},
            backends_used=["vector", "graph"]
        )
        
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_count == 2
        assert response.search_mode_used == "hybrid"
        assert response.response_time_ms == 45.2
        assert response.trace_id == "trace_abc_123"
        assert response.backend_timings["vector"] == 25.1
        assert response.backends_used == ["vector", "graph"]
    
    def test_total_count_validation(self):
        """Test total_count validation against actual results."""
        results = [
            MemoryResult(id="1", text="test", source=ResultSource.VECTOR),
            MemoryResult(id="2", text="test", source=ResultSource.VECTOR)
        ]
        
        # total_count equal to results should pass
        response = SearchResultResponse(
            success=True,
            results=results,
            total_count=2,
            search_mode_used="vector"
        )
        assert response.total_count == 2
        
        # total_count higher than results should pass (pagination)
        response = SearchResultResponse(
            success=True,
            results=results,
            total_count=10,
            search_mode_used="vector"
        )
        assert response.total_count == 10
        
        # total_count lower than results should fail
        with pytest.raises(ValueError, match="total_count cannot be less than actual results count"):
            SearchResultResponse(
                success=True,
                results=results,
                total_count=1,
                search_mode_used="vector"
            )


class TestUtilityFunctions:
    """Test utility functions for result manipulation."""
    
    def create_test_results(self):
        """Create test results for utility function testing."""
        return [
            MemoryResult(id="1", text="First", source=ResultSource.VECTOR, score=0.9),
            MemoryResult(id="2", text="Second", source=ResultSource.GRAPH, score=0.8),
            MemoryResult(id="3", text="Third", source=ResultSource.KV, score=0.7),
            MemoryResult(id="1", text="Duplicate", source=ResultSource.VECTOR, score=0.95),  # Duplicate ID
        ]
    
    def test_merge_results(self):
        """Test merging result lists with duplicate removal."""
        results1 = [
            MemoryResult(id="1", text="First", source=ResultSource.VECTOR, score=0.9),
            MemoryResult(id="2", text="Second", source=ResultSource.GRAPH, score=0.8)
        ]
        
        results2 = [
            MemoryResult(id="1", text="Duplicate", source=ResultSource.VECTOR, score=0.95),
            MemoryResult(id="3", text="Third", source=ResultSource.KV, score=0.7)
        ]
        
        merged = merge_results(results1, results2)
        
        # Should have 3 unique results (first occurrence wins for duplicates)
        assert len(merged) == 3
        ids = [r.id for r in merged]
        assert ids == ["1", "2", "3"]
        
        # First occurrence should win for duplicate ID "1"
        first_result = next(r for r in merged if r.id == "1")
        assert first_result.text == "First"  # From results1
        assert first_result.score == 0.9
    
    def test_sort_results_by_score(self):
        """Test sorting results by score."""
        results = self.create_test_results()[:3]  # Avoid duplicate for this test
        
        # Sort descending (default)
        sorted_desc = sort_results_by_score(results)
        scores_desc = [r.score for r in sorted_desc]
        assert scores_desc == [0.9, 0.8, 0.7]
        
        # Sort ascending
        sorted_asc = sort_results_by_score(results, descending=False)
        scores_asc = [r.score for r in sorted_asc]
        assert scores_asc == [0.7, 0.8, 0.9]
    
    def test_filter_results_by_threshold(self):
        """Test filtering results by score threshold."""
        results = self.create_test_results()[:3]  # Avoid duplicate for this test
        
        # Filter with threshold 0.75
        filtered = filter_results_by_threshold(results, 0.75)
        assert len(filtered) == 2  # 0.9 and 0.8 should pass, 0.7 should not
        
        filtered_scores = [r.score for r in filtered]
        assert all(score >= 0.75 for score in filtered_scores)
        
        # Filter with threshold 1.0
        filtered_high = filter_results_by_threshold(results, 1.0)
        assert len(filtered_high) == 0  # No results should pass
        
        # Filter with threshold 0.0
        filtered_low = filter_results_by_threshold(results, 0.0)
        assert len(filtered_low) == 3  # All results should pass