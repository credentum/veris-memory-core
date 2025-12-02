#!/usr/bin/env python3
"""
Comprehensive tests for the advanced filtering system.

Tests cover all filter types, combination modes, and edge cases
for the enhanced filtering capabilities.
"""

import pytest
import re
from datetime import datetime, timedelta
from typing import List

from src.filters.advanced_filters import (
    AdvancedFilterEngine, FilterCriteria, FilterType, CombineMode,
    DateRangeFilter, NumericRangeFilter, FuzzyMatcher,
    advanced_filter_engine
)
from src.interfaces.memory_result import MemoryResult


@pytest.fixture
def sample_results():
    """Create sample MemoryResult objects for testing."""
    base_time = datetime(2024, 1, 15, 12, 0, 0)
    
    return [
        MemoryResult(
            id="result1",
            text="Python programming tutorial for beginners",
            content_type="tutorial",
            source="text",
            score=0.85,
            timestamp=base_time,
            tags=["python", "programming", "tutorial", "beginner"],
            metadata={"author": "John Doe", "difficulty": "easy", "views": 1000}
        ),
        MemoryResult(
            id="result2", 
            text="Advanced machine learning algorithms",
            content_type="article",
            source="vector",
            score=0.92,
            timestamp=base_time + timedelta(days=1),
            tags=["machine-learning", "algorithms", "advanced"],
            metadata={"author": "Jane Smith", "difficulty": "hard", "views": 2500}
        ),
        MemoryResult(
            id="result3",
            text="JavaScript fundamentals and best practices",
            content_type="guide", 
            source="text",
            score=0.78,
            timestamp=base_time + timedelta(days=2),
            tags=["javascript", "fundamentals", "best-practices"],
            metadata={"author": "Bob Wilson", "difficulty": "medium", "views": 1500}
        ),
        MemoryResult(
            id="result4",
            text="Database design patterns",
            content_type="reference",
            source="graph",
            score=0.88,
            timestamp=base_time - timedelta(days=1),
            tags=["database", "design", "patterns"],
            metadata={"author": "Alice Johnson", "difficulty": "medium", "views": 800}
        )
    ]


class TestFilterCriteria:
    """Test filter criteria data structure."""
    
    def test_filter_criteria_creation(self):
        """Test basic filter criteria creation."""
        criteria = FilterCriteria(
            field="text",
            filter_type=FilterType.CONTAINS,
            value="python",
            case_sensitive=True,
            description="Python content filter"
        )
        
        assert criteria.field == "text"
        assert criteria.filter_type == FilterType.CONTAINS
        assert criteria.value == "python"
        assert criteria.case_sensitive is True
        assert criteria.combine_mode == CombineMode.AND  # Default
        assert criteria.description == "Python content filter"
    
    def test_filter_criteria_defaults(self):
        """Test filter criteria with default values."""
        criteria = FilterCriteria(
            field="score",
            filter_type=FilterType.NUMERIC_RANGE,
            value={"min": 0.8}
        )
        
        assert criteria.combine_mode == CombineMode.AND
        assert criteria.case_sensitive is False
        assert criteria.description is None


class TestDateRangeFilter:
    """Test date range filter helper."""
    
    def test_date_range_filter_absolute(self):
        """Test date range filter with absolute dates."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        
        date_filter = DateRangeFilter(start_date=start, end_date=end)
        criteria = date_filter.to_criteria()
        
        assert criteria.field == "timestamp"
        assert criteria.filter_type == FilterType.DATE_RANGE
        assert criteria.value["start"] == start
        assert criteria.value["end"] == end
    
    def test_date_range_filter_relative(self):
        """Test date range filter with relative days."""
        date_filter = DateRangeFilter(relative_days=7, field="custom_date")
        criteria = date_filter.to_criteria()
        
        assert criteria.field == "custom_date"
        assert criteria.filter_type == FilterType.DATE_RANGE
        assert criteria.value["start"] is not None
        assert criteria.value["end"] is None
        assert "Last 7 days" in criteria.description


class TestNumericRangeFilter:
    """Test numeric range filter helper."""
    
    def test_numeric_range_filter(self):
        """Test numeric range filter creation."""
        num_filter = NumericRangeFilter(field="score", min_value=0.5, max_value=0.9)
        criteria = num_filter.to_criteria()
        
        assert criteria.field == "score"
        assert criteria.filter_type == FilterType.NUMERIC_RANGE
        assert criteria.value["min"] == 0.5
        assert criteria.value["max"] == 0.9


class TestFuzzyMatcher:
    """Test fuzzy string matching utilities."""
    
    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        # Identical strings
        assert FuzzyMatcher.levenshtein_distance("hello", "hello") == 0
        
        # Single substitution
        assert FuzzyMatcher.levenshtein_distance("hello", "hallo") == 1
        
        # Single insertion
        assert FuzzyMatcher.levenshtein_distance("hello", "hellos") == 1
        
        # Single deletion
        assert FuzzyMatcher.levenshtein_distance("hellos", "hello") == 1
        
        # Multiple operations
        assert FuzzyMatcher.levenshtein_distance("kitten", "sitting") == 3
        
        # Empty strings
        assert FuzzyMatcher.levenshtein_distance("", "test") == 4
        assert FuzzyMatcher.levenshtein_distance("test", "") == 4
        assert FuzzyMatcher.levenshtein_distance("", "") == 0
    
    def test_similarity_ratio(self):
        """Test similarity ratio calculation."""
        # Identical strings
        assert FuzzyMatcher.similarity_ratio("test", "test") == 1.0
        
        # Completely different
        ratio = FuzzyMatcher.similarity_ratio("abc", "xyz")
        assert 0 <= ratio < 0.5
        
        # Similar strings
        ratio = FuzzyMatcher.similarity_ratio("programming", "programmer")
        assert ratio > 0.8
        
        # Empty strings
        assert FuzzyMatcher.similarity_ratio("", "") == 1.0
    
    def test_fuzzy_match(self):
        """Test fuzzy matching with thresholds."""
        # High similarity
        assert FuzzyMatcher.fuzzy_match("python", "python", threshold=1.0)
        assert FuzzyMatcher.fuzzy_match("python", "Python", threshold=0.9)
        
        # Medium similarity
        assert FuzzyMatcher.fuzzy_match("programming", "programmer", threshold=0.8)
        
        # Low similarity
        assert not FuzzyMatcher.fuzzy_match("python", "javascript", threshold=0.5)
        
        # Case insensitive
        assert FuzzyMatcher.fuzzy_match("PYTHON", "python", threshold=1.0)


class TestAdvancedFilterEngine:
    """Test the main advanced filter engine."""
    
    @pytest.fixture
    def engine(self):
        """Create fresh filter engine for each test."""
        return AdvancedFilterEngine()
    
    def test_engine_initialization(self, engine):
        """Test filter engine initialization."""
        assert len(engine.custom_filters) == 0
        assert "text" in engine._field_extractors
        assert "score" in engine._field_extractors
        assert "tags" in engine._field_extractors
    
    def test_field_extraction(self, engine, sample_results):
        """Test field value extraction."""
        result = sample_results[0]
        
        # Direct field access
        assert engine.extract_field_value(result, "text") == result.text
        assert engine.extract_field_value(result, "score") == result.score
        
        # Nested field access
        assert engine.extract_field_value(result, "metadata.author") == "John Doe"
        assert engine.extract_field_value(result, "metadata.views") == 1000
        
        # Non-existent fields
        assert engine.extract_field_value(result, "nonexistent") is None
        assert engine.extract_field_value(result, "metadata.missing") is None
    
    def test_custom_filter_registration(self, engine):
        """Test custom filter registration."""
        def high_score_filter(result, params):
            threshold = params.get("threshold", 0.9)
            return result.score >= threshold
        
        engine.register_custom_filter("high_score", high_score_filter)
        
        assert "high_score" in engine.custom_filters
        assert engine.custom_filters["high_score"] == high_score_filter
    
    def test_custom_field_extractor(self, engine, sample_results):
        """Test custom field extractor registration."""
        def word_count_extractor(result):
            return len(result.text.split())
        
        engine.register_field_extractor("word_count", word_count_extractor)
        
        result = sample_results[0]
        word_count = engine.extract_field_value(result, "word_count")
        expected_count = len(result.text.split())
        assert word_count == expected_count


class TestSingleFilterApplication:
    """Test applying individual filters."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedFilterEngine()
    
    def test_exact_match_filter(self, engine, sample_results):
        """Test exact match filtering."""
        criteria = FilterCriteria(
            field="content_type",
            filter_type=FilterType.EXACT_MATCH,
            value="tutorial"
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        assert len(filtered) == 1
        assert filtered[0].id == "result1"
        assert filtered[0].content_type == "tutorial"
    
    def test_contains_filter(self, engine, sample_results):
        """Test contains filtering."""
        criteria = FilterCriteria(
            field="text",
            filter_type=FilterType.CONTAINS,
            value="programming"
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        # Should match "Python programming tutorial" and no others
        assert len(filtered) == 1
        assert "programming" in filtered[0].text
    
    def test_contains_filter_case_sensitive(self, engine, sample_results):
        """Test case-sensitive contains filtering."""
        criteria = FilterCriteria(
            field="text",
            filter_type=FilterType.CONTAINS,
            value="Python",
            case_sensitive=True
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        assert len(filtered) == 1
        
        # Test case insensitive (default)
        criteria.case_sensitive = False
        filtered = engine.apply_single_filter(sample_results, criteria)
        assert len(filtered) == 1
    
    def test_regex_filter(self, engine, sample_results):
        """Test regex filtering."""
        criteria = FilterCriteria(
            field="text",
            filter_type=FilterType.REGEX,
            value=r"\b\w+ing\b"  # Words ending in 'ing'
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        # Should match "programming" and "learning"
        assert len(filtered) >= 1
        for result in filtered:
            assert re.search(r"\b\w+ing\b", result.text)
    
    def test_date_range_filter(self, engine, sample_results):
        """Test date range filtering."""
        base_date = datetime(2024, 1, 15, 12, 0, 0)
        
        criteria = FilterCriteria(
            field="timestamp",
            filter_type=FilterType.DATE_RANGE,
            value={
                "start": base_date,
                "end": base_date + timedelta(days=1, hours=12)
            }
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        # Should match results within the date range
        for result in filtered:
            assert base_date <= result.timestamp <= base_date + timedelta(days=1, hours=12)
    
    def test_numeric_range_filter(self, engine, sample_results):
        """Test numeric range filtering."""
        criteria = FilterCriteria(
            field="score",
            filter_type=FilterType.NUMERIC_RANGE,
            value={"min": 0.8, "max": 0.9}
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        for result in filtered:
            assert 0.8 <= result.score <= 0.9
    
    def test_tag_filter_required(self, engine, sample_results):
        """Test tag filtering with required tags."""
        criteria = FilterCriteria(
            field="tags",
            filter_type=FilterType.TAG_FILTER,
            value={"required": ["python"], "mode": "all"}
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        assert len(filtered) == 1
        assert "python" in filtered[0].tags
    
    def test_tag_filter_excluded(self, engine, sample_results):
        """Test tag filtering with excluded tags."""
        criteria = FilterCriteria(
            field="tags",
            filter_type=FilterType.TAG_FILTER,
            value={"excluded": ["python"]}
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        # Should exclude the Python tutorial
        assert len(filtered) == 3
        for result in filtered:
            assert "python" not in result.tags
    
    def test_source_filter(self, engine, sample_results):
        """Test source filtering."""
        criteria = FilterCriteria(
            field="source",
            filter_type=FilterType.SOURCE_FILTER,
            value=["text", "vector"]
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        for result in filtered:
            assert result.source in ["text", "vector"]
    
    def test_score_threshold_filter(self, engine, sample_results):
        """Test score threshold filtering."""
        criteria = FilterCriteria(
            field="score",
            filter_type=FilterType.SCORE_THRESHOLD,
            value=0.85
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        for result in filtered:
            assert result.score >= 0.85
    
    def test_fuzzy_match_filter(self, engine, sample_results):
        """Test fuzzy match filtering."""
        criteria = FilterCriteria(
            field="text",
            filter_type=FilterType.FUZZY_MATCH,
            value={"pattern": "programing", "threshold": 0.8}  # Misspelled
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        # Should match "programming" despite misspelling
        assert len(filtered) == 1
        assert "programming" in filtered[0].text
    
    def test_custom_filter(self, engine, sample_results):
        """Test custom filter application."""
        def view_count_filter(result, params):
            min_views = params.get("min_views", 1000)
            views = result.metadata.get("views", 0)
            return views >= min_views
        
        engine.register_custom_filter("min_views", view_count_filter)
        
        criteria = FilterCriteria(
            field="custom",  # Field doesn't matter for custom filters
            filter_type=FilterType.CUSTOM,
            value={"name": "min_views", "params": {"min_views": 1200}}
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        
        # Should match results with >= 1200 views
        for result in filtered:
            assert result.metadata["views"] >= 1200


class TestMultipleFilterCombination:
    """Test combining multiple filters."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedFilterEngine()
    
    def test_and_combination(self, engine, sample_results):
        """Test AND combination of filters."""
        criteria_list = [
            FilterCriteria(
                field="source",
                filter_type=FilterType.EXACT_MATCH,
                value="text"
            ),
            FilterCriteria(
                field="score",
                filter_type=FilterType.NUMERIC_RANGE,
                value={"min": 0.8}
            )
        ]
        
        filtered = engine.apply_multiple_filters(
            sample_results, 
            criteria_list, 
            global_combine_mode=CombineMode.AND
        )
        
        # Should match only results that satisfy ALL criteria
        for result in filtered:
            assert result.source == "text"
            assert result.score >= 0.8
    
    def test_or_combination(self, engine, sample_results):
        """Test OR combination of filters."""
        criteria_list = [
            FilterCriteria(
                field="content_type",
                filter_type=FilterType.EXACT_MATCH,
                value="tutorial"
            ),
            FilterCriteria(
                field="content_type", 
                filter_type=FilterType.EXACT_MATCH,
                value="article"
            )
        ]
        
        filtered = engine.apply_multiple_filters(
            sample_results,
            criteria_list,
            global_combine_mode=CombineMode.OR
        )
        
        # Should match results that satisfy ANY criteria
        for result in filtered:
            assert result.content_type in ["tutorial", "article"]
    
    def test_not_combination(self, engine, sample_results):
        """Test NOT combination of filters."""
        criteria_list = [
            FilterCriteria(
                field="content_type",
                filter_type=FilterType.EXACT_MATCH,
                value="tutorial"
            )
        ]
        
        filtered = engine.apply_multiple_filters(
            sample_results,
            criteria_list,
            global_combine_mode=CombineMode.NOT
        )
        
        # Should exclude results matching criteria
        for result in filtered:
            assert result.content_type != "tutorial"


class TestHelperMethods:
    """Test helper methods and utilities."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedFilterEngine()
    
    def test_date_range_filter_helper(self, engine, sample_results):
        """Test date range filter helper method."""
        date_filter = DateRangeFilter(
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 1, 16),
            field="timestamp"
        )
        
        filtered = engine.apply_date_range_filter(sample_results, date_filter)
        
        for result in filtered:
            assert datetime(2024, 1, 15) <= result.timestamp <= datetime(2024, 1, 16)
    
    def test_tag_filters_helper(self, engine, sample_results):
        """Test tag filters helper method."""
        # Required tags
        filtered = engine.apply_tag_filters(
            sample_results,
            required_tags=["programming"]
        )
        
        for result in filtered:
            assert "programming" in result.tags
        
        # Excluded tags
        filtered = engine.apply_tag_filters(
            sample_results,
            excluded_tags=["advanced"]
        )
        
        for result in filtered:
            assert "advanced" not in result.tags
    
    def test_content_search_helper(self, engine, sample_results):
        """Test content search helper method."""
        # Contains search
        filtered = engine.apply_content_search(
            sample_results,
            "machine",
            match_type=FilterType.CONTAINS
        )
        
        assert len(filtered) == 1
        assert "machine" in filtered[0].text
        
        # Regex search
        filtered = engine.apply_content_search(
            sample_results,
            r"\bPython\b",
            match_type=FilterType.REGEX,
            case_sensitive=True
        )
        
        for result in filtered:
            assert re.search(r"\bPython\b", result.text)
    
    def test_filter_statistics(self, engine, sample_results):
        """Test filter statistics generation."""
        stats = engine.get_filter_statistics(sample_results)
        
        assert stats["total_results"] == len(sample_results)
        assert "sources" in stats
        assert "content_types" in stats
        assert "tag_frequency" in stats
        assert "score_range" in stats
        assert "date_range" in stats
        
        # Verify structure
        assert stats["score_range"]["min"] <= stats["score_range"]["max"]
        assert stats["date_range"]["earliest"] <= stats["date_range"]["latest"]
        
        # Verify content
        assert "text" in stats["sources"]
        assert "tutorial" in stats["content_types"]
        assert "python" in stats["tag_frequency"]


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def engine(self):
        return AdvancedFilterEngine()
    
    def test_empty_results_list(self, engine):
        """Test filtering empty results list."""
        criteria = FilterCriteria(
            field="text",
            filter_type=FilterType.CONTAINS,
            value="anything"
        )
        
        filtered = engine.apply_single_filter([], criteria)
        assert len(filtered) == 0
    
    def test_empty_criteria_list(self, engine, sample_results):
        """Test filtering with empty criteria list."""
        filtered = engine.apply_multiple_filters(sample_results, [])
        assert len(filtered) == len(sample_results)
    
    def test_invalid_field_access(self, engine, sample_results):
        """Test filtering with invalid field names."""
        criteria = FilterCriteria(
            field="nonexistent.field",
            filter_type=FilterType.EXACT_MATCH,
            value="anything"
        )
        
        # Should not crash, should return empty results
        filtered = engine.apply_single_filter(sample_results, criteria)
        assert len(filtered) == 0
    
    def test_invalid_filter_type(self, engine, sample_results):
        """Test with invalid filter type."""
        # This would need to be tested at runtime since enum validation happens earlier
        pass
    
    def test_malformed_filter_values(self, engine, sample_results):
        """Test with malformed filter values."""
        # Date range with invalid dates
        criteria = FilterCriteria(
            field="timestamp",
            filter_type=FilterType.DATE_RANGE,
            value={"start": "not_a_date"}  # Invalid date
        )
        
        # Should not crash
        filtered = engine.apply_single_filter(sample_results, criteria)
        # Behavior depends on implementation - may return all or none
    
    def test_regex_compilation_error(self, engine, sample_results):
        """Test with invalid regex patterns."""
        criteria = FilterCriteria(
            field="text",
            filter_type=FilterType.REGEX,
            value="[invalid regex"  # Unclosed bracket
        )
        
        # Should handle regex compilation errors gracefully
        filtered = engine.apply_single_filter(sample_results, criteria)
        # Should not crash - may return all results or none
    
    def test_numeric_range_with_non_numeric(self, engine, sample_results):
        """Test numeric range filter on non-numeric field."""
        criteria = FilterCriteria(
            field="text",  # Non-numeric field
            filter_type=FilterType.NUMERIC_RANGE,
            value={"min": 0.5, "max": 1.0}
        )
        
        filtered = engine.apply_single_filter(sample_results, criteria)
        # Should return empty results since text can't be converted to numbers
        assert len(filtered) == 0


class TestGlobalFilterEngine:
    """Test global filter engine instance."""
    
    def test_global_instance_access(self):
        """Test accessing global filter engine instance."""
        # Global instance should be available
        assert advanced_filter_engine is not None
        assert isinstance(advanced_filter_engine, AdvancedFilterEngine)
    
    def test_global_instance_modification(self):
        """Test modifying global filter engine."""
        def test_filter(result, params):
            return True
        
        # Register filter on global instance
        advanced_filter_engine.register_custom_filter("test_global", test_filter)
        
        # Should be available
        assert "test_global" in advanced_filter_engine.custom_filters
        
        # Clean up
        del advanced_filter_engine.custom_filters["test_global"]