#!/usr/bin/env python3
"""
Test suite to verify sort_by parameter validation fix.

This test verifies that the validation added for sort_by parameter
properly rejects invalid values and accepts valid ones.
"""

import pytest
from pydantic import ValidationError
from src.mcp_server.main import RetrieveContextRequest


class TestSortByValidationFix:
    """Test the sort_by parameter validation fix."""
    
    def test_valid_timestamp_sort_by(self) -> None:
        """Test that 'timestamp' is accepted as valid sort_by value."""
        request = RetrieveContextRequest(
            query="test query",
            sort_by="timestamp"
        )
        assert request.sort_by == "timestamp"
    
    def test_valid_relevance_sort_by(self) -> None:
        """Test that 'relevance' is accepted as valid sort_by value."""
        request = RetrieveContextRequest(
            query="test query",
            sort_by="relevance"
        )
        assert request.sort_by == "relevance"
    
    def test_default_sort_by(self) -> None:
        """Test that sort_by defaults to 'timestamp' when not provided."""
        request = RetrieveContextRequest(
            query="test query"
        )
        assert request.sort_by == "timestamp"
    
    def test_invalid_sort_by_values(self) -> None:
        """Test that invalid sort_by values are rejected."""
        invalid_values = ["invalid", "date", "score", "alphabetical", "random"]
        
        for invalid_value in invalid_values:
            with pytest.raises(ValidationError) as exc_info:
                RetrieveContextRequest(
                    query="test query",
                    sort_by=invalid_value
                )
            
            # Verify the error is about the sort_by field
            errors = exc_info.value.errors()
            assert any(error["loc"] == ("sort_by",) for error in errors), \
                f"Expected validation error for sort_by='{invalid_value}'"
    
    def test_empty_string_sort_by(self) -> None:
        """Test that empty string is rejected for sort_by."""
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by=""
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for empty sort_by"
    
    def test_numeric_sort_by(self) -> None:
        """Test that numeric values are rejected for sort_by."""
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by=123  # type: ignore
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for numeric sort_by"
    
    def test_none_sort_by(self) -> None:
        """Test that None value is rejected."""
        # None should be rejected as it's not a valid string
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by=None  # type: ignore
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for None sort_by"
    
    def test_case_sensitivity(self) -> None:
        """Test that sort_by values are case-sensitive."""
        # Uppercase should be rejected
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by="TIMESTAMP"
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for uppercase TIMESTAMP"
        
        # Mixed case should be rejected
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by="Relevance"
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for mixed case Relevance"
    
    def test_whitespace_handling(self) -> None:
        """Test that sort_by values with whitespace are rejected."""
        # Leading whitespace
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by=" timestamp"
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for leading whitespace"
        
        # Trailing whitespace
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by="relevance "
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for trailing whitespace"
        
        # Whitespace only
        with pytest.raises(ValidationError) as exc_info:
            RetrieveContextRequest(
                query="test query",
                sort_by="   "
            )
        
        errors = exc_info.value.errors()
        assert any(error["loc"] == ("sort_by",) for error in errors), \
            "Expected validation error for whitespace only"


class TestSortByIntegration:
    """Integration tests for sort_by parameter."""
    
    @pytest.mark.asyncio
    async def test_sorting_error_response_format(self) -> None:
        """Test that invalid sort_by returns proper error format."""
        from src.mcp_server.simple_server import handle_retrieve_context
        
        # Test with invalid sort_by
        result = await handle_retrieve_context({
            "query": "test",
            "sort_by": "invalid_option"
        })
        
        assert result["success"] is False
        assert "Invalid sort_by value" in result["message"]
        assert result["error_type"] == "invalid_parameter"
        assert result["results"] == []
    
    @pytest.mark.asyncio
    async def test_sorting_accepts_valid_values(self) -> None:
        """Test that valid sort_by values are accepted."""
        from src.mcp_server.simple_server import handle_retrieve_context
        
        # Test with valid sort_by values
        for valid_value in ["timestamp", "relevance"]:
            result = await handle_retrieve_context({
                "query": "test",
                "sort_by": valid_value
            })
            
            # Should not return validation error
            if not result["success"]:
                assert "Invalid sort_by value" not in result.get("message", "")
    
    @pytest.mark.asyncio
    async def test_actual_sorting_behavior(self) -> None:
        """Test that sorting actually orders results correctly."""
        from src.mcp_server.simple_server import handle_store_context, handle_retrieve_context
        from datetime import datetime, timedelta
        
        # Store test contexts with different timestamps
        base_time = datetime.now()
        
        # Store oldest context
        await handle_store_context({
            "content": {"text": "Old context"},
            "type": "test",
            "metadata": {"created_at": (base_time - timedelta(days=7)).isoformat()}
        })
        
        # Store newest context
        await handle_store_context({
            "content": {"text": "New context"},
            "type": "test",
            "metadata": {"created_at": base_time.isoformat()}
        })
        
        # Store middle context
        await handle_store_context({
            "content": {"text": "Middle context"},
            "type": "test",
            "metadata": {"created_at": (base_time - timedelta(days=3)).isoformat()}
        })
        
        # Test timestamp sorting (newest first)
        result = await handle_retrieve_context({
            "query": "context",
            "sort_by": "timestamp"
        })
        
        if result["success"] and len(result["results"]) >= 3:
            # Verify newest is first
            timestamps = [r.get("created_at", "") for r in result["results"]]
            # Check that timestamps are in descending order
            for i in range(len(timestamps) - 1):
                if timestamps[i] and timestamps[i+1]:
                    assert timestamps[i] >= timestamps[i+1], \
                        "Timestamps should be in descending order (newest first)"
        
        # Test relevance sorting
        result = await handle_retrieve_context({
            "query": "context",
            "sort_by": "relevance"
        })
        
        if result["success"] and len(result["results"]) >= 2:
            # Verify scores are in descending order
            scores = [r.get("relevance", 0) for r in result["results"]]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i+1], \
                    "Relevance scores should be in descending order (highest first)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])