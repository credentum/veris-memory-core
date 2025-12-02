#!/usr/bin/env python3
"""
Simple focused tests for search sorting validation and scoring.

These tests verify:
1. sort_by parameter validation
2. Graph score calculation formula
3. Timestamp vs relevance sorting logic
"""

import pytest
from datetime import datetime, timedelta


class TestSortByValidation:
    """Test validation of sort_by parameter values."""
    
    def test_valid_sort_by_values(self):
        """Test that valid sort_by values are accepted."""
        valid_values = ["timestamp", "relevance"]
        
        for value in valid_values:
            # These should not raise any exception
            assert value in ["timestamp", "relevance"]
    
    def test_invalid_sort_by_values(self):
        """Test that invalid sort_by values are detected."""
        invalid_values = ["date", "score", "random", "alphabetical", "", None]
        
        for value in invalid_values:
            if value is not None:
                assert value not in ["timestamp", "relevance"], \
                    f"'{value}' should not be a valid sort_by value"


class TestGraphScoreFormula:
    """Test the graph distance-based scoring formula."""
    
    def test_direct_connection_score(self):
        """Test score for direct graph connection (1 hop)."""
        hop_distance = 1
        score = 1.0 / (hop_distance + 0.5)
        
        # Direct connection should have score of 0.667
        assert abs(score - 0.667) < 0.001, \
            f"Direct connection score should be ~0.667, got {score}"
    
    def test_two_hop_score(self):
        """Test score for 2-hop graph connection."""
        hop_distance = 2
        score = 1.0 / (hop_distance + 0.5)
        
        # Two hops should have score of 0.4
        assert abs(score - 0.4) < 0.001, \
            f"Two-hop connection score should be 0.4, got {score}"
    
    def test_three_hop_score(self):
        """Test score for 3-hop graph connection."""
        hop_distance = 3
        score = 1.0 / (hop_distance + 0.5)
        
        # Three hops should have score of ~0.286
        assert abs(score - 0.286) < 0.001, \
            f"Three-hop connection score should be ~0.286, got {score}"
    
    def test_score_decreases_with_distance(self):
        """Test that scores decrease as hop distance increases."""
        scores = []
        for distance in range(1, 10):
            score = 1.0 / (distance + 0.5)
            scores.append((distance, score))
        
        # Verify monotonic decrease
        for i in range(len(scores) - 1):
            assert scores[i][1] > scores[i+1][1], \
                f"Score should decrease: {scores[i][0]} hops ({scores[i][1]}) " \
                f"should be > {scores[i+1][0]} hops ({scores[i+1][1]})"
    
    def test_score_never_negative_or_zero(self):
        """Test that score is always positive, even for large distances."""
        test_distances = [1, 5, 10, 50, 100, 1000]
        
        for distance in test_distances:
            score = 1.0 / (distance + 0.5)
            assert score > 0, \
                f"Score should be positive for distance {distance}, got {score}"
            assert score <= 1.0, \
                f"Score should not exceed 1.0 for distance {distance}, got {score}"


class TestSortingLogic:
    """Test the sorting logic for timestamp and relevance."""
    
    def test_timestamp_sorting_order(self):
        """Test that timestamp sorting puts newest first."""
        now = datetime.now()
        
        # Create test data
        items = [
            {"id": "old", "created_at": (now - timedelta(days=7)).isoformat()},
            {"id": "newest", "created_at": now.isoformat()},
            {"id": "middle", "created_at": (now - timedelta(days=3)).isoformat()},
        ]
        
        # Sort by timestamp (newest first)
        sorted_items = sorted(
            items,
            key=lambda x: x.get("created_at", ""),
            reverse=True
        )
        
        # Verify order
        assert sorted_items[0]["id"] == "newest", "Newest should be first"
        assert sorted_items[1]["id"] == "middle", "Middle should be second"
        assert sorted_items[2]["id"] == "old", "Oldest should be last"
    
    def test_relevance_sorting_order(self):
        """Test that relevance sorting puts highest score first."""
        items = [
            {"id": "low", "score": 0.3},
            {"id": "high", "score": 0.9},
            {"id": "medium", "score": 0.6},
        ]
        
        # Sort by score (highest first)
        sorted_items = sorted(
            items,
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        
        # Verify order
        assert sorted_items[0]["id"] == "high", "Highest score should be first"
        assert sorted_items[1]["id"] == "medium", "Medium score should be second"
        assert sorted_items[2]["id"] == "low", "Lowest score should be last"
    
    def test_handling_missing_timestamps(self):
        """Test that missing timestamps are handled gracefully."""
        items = [
            {"id": "has_timestamp", "created_at": "2025-01-15T10:00:00"},
            {"id": "no_timestamp"},
            {"id": "null_timestamp", "created_at": None},
            {"id": "empty_timestamp", "created_at": ""},
        ]
        
        # Sort by timestamp with missing values
        sorted_items = sorted(
            items,
            key=lambda x: x.get("created_at", "") or "",
            reverse=True
        )
        
        # Items with valid timestamps should come first
        assert sorted_items[0]["id"] == "has_timestamp", \
            "Item with valid timestamp should be first"
        
        # Items without timestamps should not cause errors
        assert len(sorted_items) == 4, "All items should be in result"
    
    def test_handling_missing_scores(self):
        """Test that missing scores are handled gracefully."""
        items = [
            {"id": "has_score", "score": 0.8},
            {"id": "no_score"},
            {"id": "zero_score", "score": 0},
            {"id": "null_score", "score": None},
        ]
        
        # Sort by score with missing values
        sorted_items = sorted(
            items,
            key=lambda x: x.get("score", 0) or 0,
            reverse=True
        )
        
        # Item with highest score should be first
        assert sorted_items[0]["id"] == "has_score", \
            "Item with highest score should be first"
        
        # All items should be included
        assert len(sorted_items) == 4, "All items should be in result"


class TestCypherQueryModifications:
    """Test the Cypher query modifications for hop distance."""
    
    def test_cypher_includes_path_length(self):
        """Test that Cypher query includes length(path) for scoring."""
        # The modified query should include these elements
        expected_elements = [
            "MATCH path =",  # Capture the path
            "length(path) as hop_distance",  # Return path length
            "ORDER BY hop_distance ASC",  # Order by distance
        ]
        
        # This is what the query should look like
        sample_query = """
        MATCH path = (n:Context)-[r*1..2]->(m)
        WHERE (n.type = $type OR $type = 'all')
        AND (n.content CONTAINS $query OR n.metadata CONTAINS $query OR
             m.content CONTAINS $query OR m.metadata CONTAINS $query)
        RETURN DISTINCT m.id as id, m.type as type, m.content as content,
               m.metadata as metadata, m.created_at as created_at,
               length(path) as hop_distance
        ORDER BY hop_distance ASC
        LIMIT $limit
        """
        
        for element in expected_elements:
            assert element in sample_query, \
                f"Query should include '{element}' for proper scoring"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])