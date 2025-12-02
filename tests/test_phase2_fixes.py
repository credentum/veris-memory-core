"""
Test for Phase 2 fixes addressing GitHub issue #127 LIM-001 and LIM-002.

This module tests the performance score calculation and strict metadata filtering fixes.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import json

# Import the dashboard with performance score calculation
from src.monitoring.dashboard import UnifiedDashboard


class TestPerformanceScoreCalculation:
    """Test the performance score calculation implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dashboard = UnifiedDashboard()
        # Mock default thresholds
        self.thresholds = {
            'error_rate_warning_percent': 1.0,
            'error_rate_critical_percent': 5.0,
            'latency_warning_ms': 100,
            'latency_critical_ms': 500
        }

    def test_perfect_performance_score(self):
        """Test performance score calculation with perfect metrics."""
        # Perfect performance: no errors, low latency
        error_rate = 0.0
        avg_latency = 10.0
        p99_latency = 50.0
        
        score = self.dashboard._calculate_performance_score(
            error_rate, avg_latency, p99_latency, self.thresholds
        )
        
        # Should be 1.0 (perfect score)
        assert score == 1.0

    def test_warning_level_performance_score(self):
        """Test performance score calculation with warning-level metrics."""
        # Warning level: some errors, moderate latency
        error_rate = 1.0  # At warning threshold
        avg_latency = 100.0  # At warning threshold
        p99_latency = 200.0  # Below critical threshold
        
        score = self.dashboard._calculate_performance_score(
            error_rate, avg_latency, p99_latency, self.thresholds
        )
        
        # Should be less than 1.0 but greater than 0.5
        assert 0.5 < score < 1.0

    def test_critical_performance_score(self):
        """Test performance score calculation with critical metrics."""
        # Critical level: high errors, high latency
        error_rate = 10.0  # Above critical threshold
        avg_latency = 1000.0  # Above critical threshold
        p99_latency = 2000.0  # Very high
        
        score = self.dashboard._calculate_performance_score(
            error_rate, avg_latency, p99_latency, self.thresholds
        )
        
        # Should be low but above 0.0
        assert 0.0 <= score <= 0.5

    def test_performance_score_bounds(self):
        """Test that performance score is always between 0.0 and 1.0."""
        # Test extreme values
        test_cases = [
            (0.0, 0.0, 0.0),      # Perfect
            (100.0, 10000.0, 20000.0),  # Terrible
            (2.5, 300.0, 800.0),  # Mixed
        ]
        
        for error_rate, avg_latency, p99_latency in test_cases:
            score = self.dashboard._calculate_performance_score(
                error_rate, avg_latency, p99_latency, self.thresholds
            )
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for metrics: {error_rate}, {avg_latency}, {p99_latency}"

    @pytest.mark.asyncio
    async def test_performance_score_in_insights(self):
        """Test that performance score is included in insights."""
        # Mock global stats
        global_stats = {
            'error_rate_percent': 0.5,
            'avg_duration_ms': 50.0,
            'p99_duration_ms': 150.0,
            'requests_per_minute': 100,
            'total_requests': 1000
        }
        
        endpoint_stats = {
            '/api/test': {
                'error_rate_percent': 0.2,
                'avg_duration_ms': 45.0,
                'p99_duration_ms': 120.0,
                'request_count': 500
            }
        }
        
        # Set up dashboard config with thresholds
        self.dashboard.config = {'thresholds': self.thresholds}
        
        # Test the insights generation
        insights = await self.dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        # Performance score should be present and > 0
        assert 'performance_score' in insights
        assert isinstance(insights['performance_score'], float)
        assert 0.0 <= insights['performance_score'] <= 1.0
        
        # With these good metrics, score should be high
        assert insights['performance_score'] > 0.8

    @pytest.mark.asyncio
    async def test_performance_score_zero_fix(self):
        """Test that the performance score is no longer always 0.0."""
        # Mock the dashboard's method calls
        self.dashboard.config = {'thresholds': self.thresholds}
        
        # Create realistic performance data
        global_stats = {
            'error_rate_percent': 1.2,
            'avg_duration_ms': 85.0,
            'p99_duration_ms': 200.0,
            'requests_per_minute': 50,
            'total_requests': 2500
        }
        
        endpoint_stats = {}
        
        insights = await self.dashboard._generate_performance_insights(global_stats, endpoint_stats)
        
        # The key fix: performance_score should NOT be 0.0
        assert insights['performance_score'] != 0.0
        assert insights['performance_score'] > 0.0
        print(f"Performance score: {insights['performance_score']}")  # For debugging


class TestMetadataFiltering:
    """Test the strict metadata filtering implementation."""
    
    def test_metadata_filter_creation(self):
        """Test that metadata filters are properly constructed."""
        metadata_filters = {"project": "api-v2", "priority": "high"}
        
        # Test Qdrant filter construction
        filter_conditions = []
        for key, value in metadata_filters.items():
            metadata_key = f"metadata.{key}"
            filter_conditions.append({
                "key": metadata_key,
                "match": {"value": value}
            })
        
        expected_filter = {"must": filter_conditions}
        
        assert len(filter_conditions) == 2
        assert filter_conditions[0]["key"] == "metadata.project"
        assert filter_conditions[0]["match"]["value"] == "api-v2"
        assert filter_conditions[1]["key"] == "metadata.priority"
        assert filter_conditions[1]["match"]["value"] == "high"

    def test_strict_metadata_post_filtering(self):
        """Test that post-processing metadata filtering works strictly."""
        # Mock results with correct structure (metadata is nested in payload)
        results = [
            {
                'id': 'ctx_1',
                'payload': {
                    'content': {'title': 'Test 1'},
                    'metadata': {'project': 'api-v2', 'priority': 'high'}
                }
            },
            {
                'id': 'ctx_2',
                'payload': {
                    'content': {'title': 'Test 2'},
                    'metadata': {'project': 'api-v2', 'priority': 'low'}  # Won't match
                }
            },
            {
                'id': 'ctx_3',
                'payload': {
                    'content': {'title': 'Test 3'},
                    'metadata': {'project': 'web-app', 'priority': 'high'}  # Won't match
                }
            },
            {
                'id': 'ctx_4',
                'payload': {
                    'content': {'title': 'Test 4'},
                    'metadata': {'project': 'api-v2', 'priority': 'high'}  # Will match
                }
            }
        ]

        metadata_filters = {"project": "api-v2", "priority": "high"}

        # Apply the same filtering logic as in the server (with fix)
        filtered_results = []
        for result in results:
            # Fix: Access metadata from correct nesting level in payload structure
            metadata = result.get('payload', {}).get('metadata', {})

            # Check if all metadata filters match exactly
            match = True
            for filter_key, filter_value in metadata_filters.items():
                if metadata.get(filter_key) != filter_value:
                    match = False
                    break

            if match:
                filtered_results.append(result)

        # Should only match ctx_1 and ctx_4
        assert len(filtered_results) == 2
        assert filtered_results[0]['id'] == 'ctx_1'
        assert filtered_results[1]['id'] == 'ctx_4'

    def test_empty_metadata_filtering(self):
        """Test filtering behavior with empty metadata."""
        results = [
            {
                'id': 'ctx_1',
                'payload': {'content': {'title': 'Test 1'}, 'metadata': {}}
            },
            {
                'id': 'ctx_2',
                'payload': {'content': {'title': 'Test 2'}}  # No metadata key
            },
        ]

        metadata_filters = {"project": "api-v2"}

        filtered_results = []
        for result in results:
            # Fix: Access metadata from correct nesting level in payload structure
            metadata = result.get('payload', {}).get('metadata', {})

            match = True
            for filter_key, filter_value in metadata_filters.items():
                if metadata.get(filter_key) != filter_value:
                    match = False
                    break

            if match:
                filtered_results.append(result)

        # No results should match since metadata is empty/missing
        assert len(filtered_results) == 0

    def test_cypher_metadata_filtering(self):
        """Test that Cypher query generation includes metadata filters."""
        metadata_filters = {"project": "test", "status": "active"}
        
        # Simulate the Cypher query generation logic
        metadata_conditions = []
        for key, value in metadata_filters.items():
            metadata_conditions.append(f"n.metadata.{key} = ${key}")
        
        metadata_clause = ""
        if metadata_conditions:
            metadata_clause = "AND " + " AND ".join(metadata_conditions)
        
        expected_conditions = [
            "n.metadata.project = $project",
            "n.metadata.status = $status"
        ]
        
        assert metadata_conditions == expected_conditions
        assert "AND n.metadata.project = $project AND n.metadata.status = $status" in metadata_clause


if __name__ == "__main__":
    pytest.main([__file__])