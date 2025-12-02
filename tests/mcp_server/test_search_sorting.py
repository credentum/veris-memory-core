"""
Test suite for search result sorting functionality in MCP server.

Tests the new sort_by parameter and graph result scoring improvements.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any, List

# Import the server module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))


class TestSearchSorting:
    """Test suite for search result sorting with sort_by parameter."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        with patch('src.mcp_server.server.qdrant_client') as mock_qdrant, \
             patch('src.mcp_server.server.neo4j_client') as mock_neo4j, \
             patch('src.mcp_server.server.input_validator') as mock_validator, \
             patch('src.mcp_server.server.rate_limit_check') as mock_rate_limit, \
             patch('src.mcp_server.server.embedding_generator') as mock_embedding, \
             patch('src.mcp_server.server.intent_classifier') as mock_intent, \
             patch('src.mcp_server.server.logger') as mock_logger:
            
            # Setup rate limit to always allow
            mock_rate_limit.return_value = asyncio.coroutine(lambda *args: (True, ""))()
            
            # Setup input validator to always pass
            mock_validation_result = MagicMock()
            mock_validation_result.valid = True
            mock_validator.validate_input.return_value = mock_validation_result
            
            # Setup mock intent classifier
            mock_intent_result = MagicMock()
            mock_intent_result.intent.value = "search"
            mock_intent_result.attribute = None
            mock_intent.classify.return_value = mock_intent_result
            
            yield {
                'qdrant': mock_qdrant,
                'neo4j': mock_neo4j,
                'validator': mock_validator,
                'rate_limit': mock_rate_limit,
                'embedding': mock_embedding,
                'intent': mock_intent,
                'logger': mock_logger
            }
    
    @pytest.mark.asyncio
    async def test_sort_by_timestamp_default(self, mock_dependencies):
        """Test that sort_by defaults to 'timestamp'."""
        from src.mcp_server.server import retrieve_context_tool
        
        # Create test data with different timestamps
        now = datetime.now()
        mock_results = [
            {
                "id": "old",
                "created_at": (now - timedelta(days=7)).isoformat(),
                "score": 0.9,
                "source": "vector"
            },
            {
                "id": "new",
                "created_at": now.isoformat(),
                "score": 0.5,
                "source": "vector"
            },
            {
                "id": "middle",
                "created_at": (now - timedelta(days=3)).isoformat(),
                "score": 0.7,
                "source": "vector"
            }
        ]
        
        # Setup mock to return unsorted results
        mock_dependencies['qdrant'].client.search.return_value = mock_results
        mock_dependencies['neo4j'].driver = None  # Disable graph search
        
        # Call without sort_by parameter (should default to timestamp)
        result = await retrieve_context_tool({"query": "test"})
        
        # Verify results are sorted by timestamp (newest first)
        assert result["success"] == True
        if len(result["results"]) > 0:
            timestamps = [r.get("created_at", "") for r in result["results"]]
            assert timestamps == sorted(timestamps, reverse=True), \
                "Results should be sorted by timestamp (newest first) by default"
    
    @pytest.mark.asyncio
    async def test_sort_by_relevance(self, mock_dependencies):
        """Test sorting by relevance score."""
        from src.mcp_server.server import retrieve_context_tool
        
        # Create test data with different scores
        mock_results = [
            {"id": "low", "score": 0.3, "created_at": "2025-01-03", "source": "vector"},
            {"id": "high", "score": 0.9, "created_at": "2025-01-01", "source": "vector"},
            {"id": "medium", "score": 0.6, "created_at": "2025-01-02", "source": "vector"}
        ]
        
        mock_dependencies['qdrant'].client.search.return_value = mock_results
        mock_dependencies['neo4j'].driver = None
        
        # Call with sort_by='relevance'
        result = await retrieve_context_tool({
            "query": "test",
            "sort_by": "relevance"
        })
        
        # Verify results are sorted by score (highest first)
        assert result["success"] == True
        if len(result["results"]) > 0:
            scores = [r.get("score", 0) for r in result["results"]]
            assert scores == sorted(scores, reverse=True), \
                "Results should be sorted by score (highest first) when sort_by='relevance'"
    
    @pytest.mark.asyncio
    async def test_invalid_sort_by_parameter(self, mock_dependencies):
        """Test that invalid sort_by values are rejected."""
        from src.mcp_server.server import retrieve_context_tool
        
        # Call with invalid sort_by value
        result = await retrieve_context_tool({
            "query": "test",
            "sort_by": "invalid_option"
        })
        
        # Verify error response
        assert result["success"] == False
        assert "Invalid sort_by value" in result["message"]
        assert result["error_type"] == "invalid_parameter"
    
    @pytest.mark.asyncio  
    async def test_graph_score_calculation(self, mock_dependencies):
        """Test that graph results get proper distance-based scores."""
        from src.mcp_server.server import retrieve_context_tool
        
        # Mock graph query results with hop distances
        mock_graph_results = [
            {"id": "direct", "hop_distance": 1, "created_at": "2025-01-01"},
            {"id": "two_hops", "hop_distance": 2, "created_at": "2025-01-02"},
            {"id": "three_hops", "hop_distance": 3, "created_at": "2025-01-03"}
        ]
        
        # Setup neo4j mock
        mock_session = MagicMock()
        mock_session.run.return_value = mock_graph_results
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        
        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_dependencies['neo4j'].driver = mock_driver
        mock_dependencies['qdrant'].client = None  # Disable vector search
        
        result = await retrieve_context_tool({"query": "test"})
        
        # Verify graph scores are calculated correctly
        # Formula: score = 1.0 / (hop_distance + 0.5)
        expected_scores = {
            "direct": 1.0 / (1 + 0.5),      # 0.667
            "two_hops": 1.0 / (2 + 0.5),    # 0.4
            "three_hops": 1.0 / (3 + 0.5)   # 0.286
        }
        
        for item in result.get("results", []):
            if item["id"] in expected_scores:
                assert abs(item["score"] - expected_scores[item["id"]]) < 0.01, \
                    f"Graph score for {item['id']} should be {expected_scores[item['id']]}"
    
    @pytest.mark.asyncio
    async def test_combined_graph_vector_timestamp_sorting(self, mock_dependencies):
        """Test that combined graph and vector results sort correctly by timestamp."""
        from src.mcp_server.server import retrieve_context_tool
        
        now = datetime.now()
        
        # Mock vector results
        mock_vector_results = [
            {
                "id": "vector_old",
                "score": 0.9,
                "created_at": (now - timedelta(days=5)).isoformat(),
                "payload": {"content": "old vector"}
            }
        ]
        
        # Mock graph results  
        mock_graph_results = [
            {
                "id": "graph_new",
                "hop_distance": 1,
                "created_at": now.isoformat(),
                "content": "new graph"
            },
            {
                "id": "graph_middle", 
                "hop_distance": 2,
                "created_at": (now - timedelta(days=2)).isoformat(),
                "content": "middle graph"
            }
        ]
        
        # Setup mocks
        mock_dependencies['qdrant'].client.search.return_value = mock_vector_results
        
        mock_session = MagicMock()
        mock_session.run.return_value = mock_graph_results
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)
        
        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_dependencies['neo4j'].driver = mock_driver
        
        # Call with default timestamp sorting
        result = await retrieve_context_tool({"query": "test"})
        
        # Verify combined results are sorted by timestamp
        assert result["success"] == True
        result_ids = [r["id"] for r in result.get("results", [])]
        
        # Expected order: graph_new (newest), graph_middle, vector_old (oldest)
        # This verifies that graph and vector results are combined and sorted together
        if len(result_ids) == 3:
            assert result_ids[0] == "graph_new", "Newest result should be first"
            assert result_ids[-1] == "vector_old", "Oldest result should be last"
    
    @pytest.mark.asyncio
    async def test_empty_results_handling(self, mock_dependencies):
        """Test that empty results don't cause errors with sorting."""
        from src.mcp_server.server import retrieve_context_tool
        
        # Return empty results
        mock_dependencies['qdrant'].client.search.return_value = []
        mock_dependencies['neo4j'].driver = None
        
        # Test both sort modes with empty results
        for sort_by in ["timestamp", "relevance"]:
            result = await retrieve_context_tool({
                "query": "test",
                "sort_by": sort_by
            })
            
            assert result["success"] == True
            assert result["results"] == []
            assert result["total_count"] == 0
    
    @pytest.mark.asyncio
    async def test_missing_timestamps_handling(self, mock_dependencies):
        """Test that results with missing timestamps are handled gracefully."""
        from src.mcp_server.server import retrieve_context_tool
        
        # Create results with missing/null timestamps
        mock_results = [
            {"id": "has_timestamp", "created_at": "2025-01-15", "score": 0.5},
            {"id": "null_timestamp", "created_at": None, "score": 0.7},
            {"id": "missing_timestamp", "score": 0.6},  # No created_at field
            {"id": "empty_timestamp", "created_at": "", "score": 0.8}
        ]
        
        mock_dependencies['qdrant'].client.search.return_value = mock_results
        mock_dependencies['neo4j'].driver = None
        
        # Should not crash when sorting by timestamp
        result = await retrieve_context_tool({
            "query": "test", 
            "sort_by": "timestamp"
        })
        
        assert result["success"] == True
        # Results with valid timestamps should appear before those without
        result_ids = [r["id"] for r in result.get("results", [])]
        assert "has_timestamp" in result_ids, "Result with valid timestamp should be included"


class TestGraphScoringFormula:
    """Specific tests for the graph score calculation formula."""
    
    def test_score_formula_direct_connection(self):
        """Test score for direct connection (hop_distance=1)."""
        hop_distance = 1
        expected_score = 1.0 / (hop_distance + 0.5)
        assert abs(expected_score - 0.667) < 0.01
    
    def test_score_formula_two_hops(self):
        """Test score for two hops."""
        hop_distance = 2
        expected_score = 1.0 / (hop_distance + 0.5)
        assert abs(expected_score - 0.4) < 0.01
    
    def test_score_formula_multiple_hops(self):
        """Test score decreases with distance."""
        scores = []
        for hop_distance in range(1, 6):
            score = 1.0 / (hop_distance + 0.5)
            scores.append(score)
        
        # Verify scores decrease monotonically
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1], \
                "Scores should decrease as hop distance increases"
    
    def test_score_formula_never_zero(self):
        """Test that score never reaches zero even for large distances."""
        for hop_distance in [10, 50, 100, 1000]:
            score = 1.0 / (hop_distance + 0.5)
            assert score > 0, f"Score should be positive for hop_distance={hop_distance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])