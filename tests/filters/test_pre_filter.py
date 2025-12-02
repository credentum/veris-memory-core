#!/usr/bin/env python3
"""
Tests for pre-filter engine.
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List

from src.filters.pre_filter import (
    PreFilterEngine,
    TimeWindowFilter,
    FilterCriteria,
    FilterOperator,
    filter_by_tags,
    filter_by_time_window,
    filter_by_content_type,
    filter_by_score_threshold,
    pre_filter_engine
)
from src.interfaces.memory_result import MemoryResult, ResultSource, ContentType


@pytest.fixture
def sample_results():
    """Create sample results for filtering tests."""
    base_time = datetime.now(timezone.utc)
    
    return [
        MemoryResult(
            id="code_python",
            text="def python_hello_world(): print('Hello')",
            type=ContentType.CODE,
            score=0.9,
            source=ResultSource.VECTOR,
            timestamp=base_time,
            tags=["python", "function", "code"],
            namespace="dev_project",
            metadata={"language": "python", "complexity": "simple"}
        ),
        MemoryResult(
            id="doc_javascript",
            text="JavaScript function documentation",
            type=ContentType.DOCUMENTATION,
            score=0.8,
            source=ResultSource.GRAPH,
            timestamp=base_time - timedelta(days=5),
            tags=["javascript", "documentation", "functions"],
            namespace="web_project",
            metadata={"language": "javascript", "type": "api_doc"}
        ),
        MemoryResult(
            id="general_old",
            text="General programming concepts",
            type=ContentType.GENERAL,
            score=0.6,
            source=ResultSource.KV,
            timestamp=base_time - timedelta(days=30),
            tags=["programming", "concepts"],
            namespace="dev_project",
            metadata={"category": "tutorial"}
        ),
        MemoryResult(
            id="fact_python",
            text="Python was created by Guido van Rossum",
            type=ContentType.FACT,
            score=0.7,
            source=ResultSource.HYBRID,
            timestamp=base_time - timedelta(days=2),
            tags=["python", "history", "fact"],
            namespace="knowledge_base",
            metadata={"verified": True, "source": "wikipedia"}
        )
    ]


class TestTimeWindowFilter:
    """Test TimeWindowFilter data class."""
    
    def test_absolute_time_range(self):
        """Test filtering with absolute start/end times."""
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 12, 31, tzinfo=timezone.utc)
        
        time_filter = TimeWindowFilter(
            start_time=start_time,
            end_time=end_time
        )
        
        filter_start, filter_end = time_filter.get_time_range()
        
        assert filter_start == start_time
        assert filter_end == end_time
    
    def test_relative_hours(self):
        """Test filtering with relative hours."""
        reference_time = datetime.now(timezone.utc)
        
        time_filter = TimeWindowFilter(relative_hours=24)
        filter_start, filter_end = time_filter.get_time_range(reference_time)
        
        expected_start = reference_time - timedelta(hours=24)
        assert abs((filter_start - expected_start).total_seconds()) < 1  # Within 1 second
        assert filter_end == reference_time
    
    def test_relative_days(self):
        """Test filtering with relative days."""
        reference_time = datetime.now(timezone.utc)
        
        time_filter = TimeWindowFilter(relative_days=7)
        filter_start, filter_end = time_filter.get_time_range(reference_time)
        
        expected_start = reference_time - timedelta(days=7)
        assert abs((filter_start - expected_start).total_seconds()) < 1
        assert filter_end == reference_time
    
    def test_default_time_range(self):
        """Test default time range behavior."""
        time_filter = TimeWindowFilter()
        reference_time = datetime.now(timezone.utc)
        
        filter_start, filter_end = time_filter.get_time_range(reference_time)
        
        # Should use min time to now
        assert filter_start == datetime.min.replace(tzinfo=timezone.utc)
        assert filter_end == reference_time


class TestFilterCriteria:
    """Test FilterCriteria data class."""
    
    def test_basic_criteria(self):
        """Test creating basic filter criteria."""
        criteria = FilterCriteria(
            field="score",
            operator=FilterOperator.GREATER_THAN,
            value=0.5
        )
        
        assert criteria.field == "score"
        assert criteria.operator == FilterOperator.GREATER_THAN
        assert criteria.value == 0.5
        assert criteria.case_sensitive is False
    
    def test_string_operator_conversion(self):
        """Test automatic string to operator conversion."""
        criteria = FilterCriteria(
            field="text",
            operator="contains",  # String instead of enum
            value="test"
        )
        
        assert criteria.operator == FilterOperator.CONTAINS
    
    def test_case_sensitive_criteria(self):
        """Test case-sensitive criteria."""
        criteria = FilterCriteria(
            field="text",
            operator=FilterOperator.EQUALS,
            value="Test",
            case_sensitive=True
        )
        
        assert criteria.case_sensitive is True


class TestPreFilterEngine:
    """Test PreFilterEngine functionality."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = PreFilterEngine()
        assert engine.custom_filters == {}
    
    def test_tag_filter_any_mode(self, sample_results):
        """Test tag filtering with 'any' match mode."""
        engine = PreFilterEngine()
        
        # Filter for Python-related tags
        filtered = engine.apply_tag_filter(sample_results, ["python", "javascript"], "any")
        
        # Should include results with either Python or JavaScript tags
        assert len(filtered) == 3  # code_python, doc_javascript, fact_python
        
        filtered_ids = {r.id for r in filtered}
        assert "code_python" in filtered_ids
        assert "doc_javascript" in filtered_ids
        assert "fact_python" in filtered_ids
        assert "general_old" not in filtered_ids  # Doesn't have target tags
    
    def test_tag_filter_all_mode(self, sample_results):
        """Test tag filtering with 'all' match mode."""
        engine = PreFilterEngine()
        
        # Filter for results with both 'python' AND 'function' tags
        filtered = engine.apply_tag_filter(sample_results, ["python", "function"], "all")
        
        # Only code_python has both tags
        assert len(filtered) == 1
        assert filtered[0].id == "code_python"
    
    def test_tag_filter_exact_mode(self, sample_results):
        """Test tag filtering with 'exact' match mode."""
        engine = PreFilterEngine()
        
        # Filter for exact tag set
        filtered = engine.apply_tag_filter(sample_results, ["python", "function", "code"], "exact")
        
        # Only code_python has exactly these tags
        assert len(filtered) == 1
        assert filtered[0].id == "code_python"
    
    def test_tag_filter_single_tag(self, sample_results):
        """Test tag filtering with single tag string."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_tag_filter(sample_results, "python", "any")
        
        # Should include code_python and fact_python
        assert len(filtered) == 2
        filtered_ids = {r.id for r in filtered}
        assert "code_python" in filtered_ids
        assert "fact_python" in filtered_ids
    
    def test_time_window_filter(self, sample_results):
        """Test time window filtering."""
        engine = PreFilterEngine()
        
        # Filter for results from last 7 days
        time_filter = TimeWindowFilter(relative_days=7)
        filtered = engine.apply_time_window(sample_results, time_filter)
        
        # Should include recent results (code_python, doc_javascript, fact_python)
        # but exclude general_old (30 days old)
        assert len(filtered) == 3
        filtered_ids = {r.id for r in filtered}
        assert "general_old" not in filtered_ids
    
    def test_time_window_filter_absolute(self, sample_results):
        """Test time window filtering with absolute times."""
        engine = PreFilterEngine()
        
        base_time = datetime.now(timezone.utc)
        time_filter = TimeWindowFilter(
            start_time=base_time - timedelta(days=10),
            end_time=base_time
        )
        
        filtered = engine.apply_time_window(sample_results, time_filter)
        
        # Should exclude general_old (30 days old)
        assert len(filtered) == 3
        filtered_ids = {r.id for r in filtered}
        assert "general_old" not in filtered_ids
    
    def test_content_type_filter_include(self, sample_results):
        """Test content type filtering (include mode)."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_content_type_filter(
            sample_results, 
            [ContentType.CODE, ContentType.DOCUMENTATION]
        )
        
        # Should include code and documentation results
        assert len(filtered) == 2
        types = {r.type for r in filtered}
        assert ContentType.CODE in types
        assert ContentType.DOCUMENTATION in types
    
    def test_content_type_filter_exclude(self, sample_results):
        """Test content type filtering (exclude mode)."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_content_type_filter(
            sample_results,
            ContentType.GENERAL,
            exclude=True
        )
        
        # Should exclude general content
        assert len(filtered) == 3
        types = {r.type for r in filtered}
        assert ContentType.GENERAL not in types
    
    def test_source_filter(self, sample_results):
        """Test source filtering."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_source_filter(
            sample_results,
            [ResultSource.VECTOR, ResultSource.GRAPH]
        )
        
        # Should include vector and graph results
        assert len(filtered) == 2
        sources = {r.source for r in filtered}
        assert ResultSource.VECTOR in sources
        assert ResultSource.GRAPH in sources
        assert ResultSource.KV not in sources
    
    def test_score_filter(self, sample_results):
        """Test score range filtering."""
        engine = PreFilterEngine()
        
        # Filter for scores between 0.7 and 1.0
        filtered = engine.apply_score_filter(sample_results, min_score=0.7, max_score=1.0)
        
        # Should include high-scoring results
        assert len(filtered) == 3  # code_python (0.9), doc_javascript (0.8), fact_python (0.7)
        scores = [r.score for r in filtered]
        assert all(0.7 <= score <= 1.0 for score in scores)
    
    def test_score_filter_min_only(self, sample_results):
        """Test score filtering with minimum only."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_score_filter(sample_results, min_score=0.75)
        
        # Should include results with score >= 0.75
        assert len(filtered) == 2  # code_python (0.9), doc_javascript (0.8)
        scores = [r.score for r in filtered]
        assert all(score >= 0.75 for score in scores)
    
    def test_namespace_filter(self, sample_results):
        """Test namespace filtering."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_namespace_filter(sample_results, "dev_project")
        
        # Should include results from dev_project namespace
        assert len(filtered) == 2  # code_python and general_old
        namespaces = {r.namespace for r in filtered}
        assert namespaces == {"dev_project"}
    
    def test_namespace_filter_multiple(self, sample_results):
        """Test namespace filtering with multiple namespaces."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_namespace_filter(
            sample_results, 
            ["dev_project", "web_project"]
        )
        
        # Should include results from both namespaces
        assert len(filtered) == 3  # All except knowledge_base
        namespaces = {r.namespace for r in filtered}
        assert "knowledge_base" not in namespaces
    
    def test_text_filter_contains(self, sample_results):
        """Test text filtering with contains operator."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_text_filter(
            sample_results,
            "python",
            FilterOperator.CONTAINS,
            case_sensitive=False
        )
        
        # Should include results containing "python" (case-insensitive)
        assert len(filtered) == 2  # code_python, fact_python
        
        for result in filtered:
            assert "python" in result.text.lower()
    
    def test_text_filter_regex(self, sample_results):
        """Test text filtering with regex operator."""
        engine = PreFilterEngine()
        
        # Search for function definitions
        filtered = engine.apply_text_filter(
            sample_results,
            r"def\s+\w+",
            FilterOperator.REGEX
        )
        
        # Should match the Python function definition
        assert len(filtered) == 1
        assert filtered[0].id == "code_python"
    
    def test_text_filter_case_sensitive(self, sample_results):
        """Test case-sensitive text filtering."""
        engine = PreFilterEngine()
        
        filtered = engine.apply_text_filter(
            sample_results,
            "Python",  # Capital P
            FilterOperator.CONTAINS,
            case_sensitive=True
        )
        
        # Should only match exact case
        assert len(filtered) == 1
        assert filtered[0].id == "fact_python"  # Only this has "Python" with capital P
    
    def test_custom_filter_registration(self, sample_results):
        """Test registering and using custom filters."""
        engine = PreFilterEngine()
        
        # Register a custom filter that keeps only high-complexity items
        def high_complexity_filter(results: List[MemoryResult], **kwargs) -> List[MemoryResult]:
            return [r for r in results if r.metadata.get("complexity") == "high"]
        
        engine.register_custom_filter("high_complexity", high_complexity_filter)
        
        assert "high_complexity" in engine.custom_filters
        
        # Add a high-complexity result
        high_complex_result = MemoryResult(
            id="complex_code",
            text="Complex algorithm implementation",
            score=0.8,
            source=ResultSource.VECTOR,
            metadata={"complexity": "high"}
        )
        
        test_results = sample_results + [high_complex_result]
        filtered = engine.apply_custom_filter(test_results, "high_complexity")
        
        assert len(filtered) == 1
        assert filtered[0].id == "complex_code"
    
    def test_custom_filter_not_registered(self, sample_results):
        """Test error handling for unregistered custom filters."""
        engine = PreFilterEngine()
        
        with pytest.raises(ValueError, match="Custom filter 'nonexistent' not registered"):
            engine.apply_custom_filter(sample_results, "nonexistent")
    
    def test_criteria_filter_single(self, sample_results):
        """Test filtering with single criterion."""
        engine = PreFilterEngine()
        
        criteria = [FilterCriteria(
            field="score",
            operator=FilterOperator.GREATER_THAN,
            value=0.75
        )]
        
        filtered = engine.apply_criteria_filter(sample_results, criteria)
        
        # Should include results with score > 0.75
        assert len(filtered) == 2
        scores = [r.score for r in filtered]
        assert all(score > 0.75 for score in scores)
    
    def test_criteria_filter_multiple(self, sample_results):
        """Test filtering with multiple criteria."""
        engine = PreFilterEngine()
        
        criteria = [
            FilterCriteria(
                field="score",
                operator=FilterOperator.GREATER_EQUAL,
                value=0.7
            ),
            FilterCriteria(
                field="type",
                operator=FilterOperator.IN,
                value=[ContentType.CODE, ContentType.FACT]
            )
        ]
        
        filtered = engine.apply_criteria_filter(sample_results, criteria)
        
        # Should include results with score >= 0.7 AND type in [CODE, FACT]
        assert len(filtered) == 2  # code_python and fact_python
        
        for result in filtered:
            assert result.score >= 0.7
            assert result.type in [ContentType.CODE, ContentType.FACT]
    
    def test_criteria_filter_nested_field(self, sample_results):
        """Test filtering with nested field access."""
        engine = PreFilterEngine()
        
        criteria = [FilterCriteria(
            field="metadata.verified",
            operator=FilterOperator.EQUALS,
            value=True
        )]
        
        filtered = engine.apply_criteria_filter(sample_results, criteria)
        
        # Should include only fact_python (has verified: True)
        assert len(filtered) == 1
        assert filtered[0].id == "fact_python"
    
    def test_criteria_filter_range_operator(self, sample_results):
        """Test filtering with range operator."""
        engine = PreFilterEngine()
        
        criteria = [FilterCriteria(
            field="score",
            operator=FilterOperator.RANGE,
            value=[0.7, 0.85]
        )]
        
        filtered = engine.apply_criteria_filter(sample_results, criteria)
        
        # Should include results with 0.7 <= score <= 0.85
        assert len(filtered) == 2  # doc_javascript (0.8) and fact_python (0.7)
        
        for result in filtered:
            assert 0.7 <= result.score <= 0.85


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_filter_by_tags_function(self, sample_results):
        """Test filter_by_tags convenience function."""
        filtered = filter_by_tags(sample_results, ["python"], "any")
        
        assert len(filtered) == 2
        filtered_ids = {r.id for r in filtered}
        assert "code_python" in filtered_ids
        assert "fact_python" in filtered_ids
    
    def test_filter_by_time_window_function(self, sample_results):
        """Test filter_by_time_window convenience function."""
        filtered = filter_by_time_window(sample_results, days=10)
        
        # Should exclude old results
        assert len(filtered) == 3
        filtered_ids = {r.id for r in filtered}
        assert "general_old" not in filtered_ids
    
    def test_filter_by_content_type_function(self, sample_results):
        """Test filter_by_content_type convenience function."""
        filtered = filter_by_content_type(sample_results, ContentType.CODE)
        
        assert len(filtered) == 1
        assert filtered[0].type == ContentType.CODE
    
    def test_filter_by_score_threshold_function(self, sample_results):
        """Test filter_by_score_threshold convenience function."""
        filtered = filter_by_score_threshold(sample_results, 0.75)
        
        assert len(filtered) == 2
        scores = [r.score for r in filtered]
        assert all(score >= 0.75 for score in scores)


class TestGlobalFilterEngine:
    """Test the global filter engine instance."""
    
    def test_global_instance(self):
        """Test that global filter engine is properly initialized."""
        assert pre_filter_engine.custom_filters == {}
    
    def test_global_instance_functionality(self, sample_results):
        """Test that global instance works correctly."""
        filtered = pre_filter_engine.apply_tag_filter(sample_results, "python", "any")
        
        assert len(filtered) == 2
        filtered_ids = {r.id for r in filtered}
        assert "code_python" in filtered_ids
        assert "fact_python" in filtered_ids


class TestFilterErrorHandling:
    """Test error handling in filters."""
    
    def test_empty_results_handling(self):
        """Test handling of empty result lists."""
        engine = PreFilterEngine()
        
        # All filters should handle empty results gracefully
        assert engine.apply_tag_filter([], ["test"]) == []
        assert engine.apply_content_type_filter([], ContentType.CODE) == []
        assert engine.apply_score_filter([], min_score=0.5) == []
    
    def test_invalid_regex_pattern(self, sample_results):
        """Test handling of invalid regex patterns."""
        engine = PreFilterEngine()
        
        # Should handle invalid regex gracefully
        filtered = engine.apply_text_filter(
            sample_results,
            "[invalid regex",  # Unclosed bracket
            FilterOperator.REGEX
        )
        
        # Should return empty results and not crash
        assert isinstance(filtered, list)
    
    def test_missing_field_in_criteria(self, sample_results):
        """Test handling of missing fields in criteria filtering."""
        engine = PreFilterEngine()
        
        criteria = [FilterCriteria(
            field="nonexistent_field",
            operator=FilterOperator.EQUALS,
            value="test"
        )]
        
        filtered = engine.apply_criteria_filter(sample_results, criteria)
        
        # Should return empty results (no items have the field)
        assert filtered == []
    
    def test_type_mismatch_in_comparison(self, sample_results):
        """Test handling of type mismatches in comparisons."""
        engine = PreFilterEngine()
        
        # Try to compare string field with number
        criteria = [FilterCriteria(
            field="text",
            operator=FilterOperator.GREATER_THAN,
            value=5  # Comparing text to number
        )]
        
        # Should handle gracefully and not crash
        filtered = engine.apply_criteria_filter(sample_results, criteria)
        assert isinstance(filtered, list)