#!/usr/bin/env python3
"""
Pre-filter engine for advanced search result filtering.

This module provides sophisticated filtering capabilities that can be applied
before or after search operations to constrain result sets based on various criteria.
"""

import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set, Union, Callable
from dataclasses import dataclass
from enum import Enum

from ..interfaces.memory_result import MemoryResult, ContentType, ResultSource
from ..utils.logging_middleware import search_logger


class FilterOperator(str, Enum):
    """Available filter operators."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_EQUAL = "gte"
    LESS_EQUAL = "lte"
    RANGE = "range"


class TagFilterMode(str, Enum):
    """Tag filtering modes for multiple tag matching."""
    ANY = "any"      # Match if result has ANY of the specified tags
    ALL = "all"      # Match if result has ALL of the specified tags  
    EXACT = "exact"  # Match if result has EXACTLY the specified tags


@dataclass
class FilterCriteria:
    """Criteria for filtering results."""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False
    
    def __post_init__(self):
        if isinstance(self.operator, str):
            self.operator = FilterOperator(self.operator)


@dataclass
class TimeWindowFilter:
    """Time-based filtering configuration."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    relative_hours: Optional[int] = None
    relative_days: Optional[int] = None
    
    def get_time_range(self, reference_time: Optional[datetime] = None) -> tuple[datetime, datetime]:
        """Get effective time range for filtering."""
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        
        # Handle relative time specifications
        if self.relative_hours is not None:
            start_time = reference_time - timedelta(hours=self.relative_hours)
            end_time = reference_time
        elif self.relative_days is not None:
            start_time = reference_time - timedelta(days=self.relative_days)
            end_time = reference_time
        else:
            # Use absolute times or defaults
            start_time = self.start_time or datetime.min.replace(tzinfo=timezone.utc)
            end_time = self.end_time or reference_time
        
        return start_time, end_time


class PreFilterEngine:
    """
    Advanced pre-filtering engine for search results.
    
    Provides sophisticated filtering capabilities including tag matching,
    time windows, content type filtering, and custom field filtering.
    """
    
    def __init__(self):
        self.custom_filters: Dict[str, Callable] = {}
    
    def apply_tag_filter(
        self,
        results: List[MemoryResult],
        tags: Union[str, List[str]],
        match_mode: str = "any"
    ) -> List[MemoryResult]:
        """
        Filter results by tags.
        
        Args:
            results: Results to filter
            tags: Tag(s) to match
            match_mode: "any" (match any tag), "all" (match all tags), "exact" (exact tag set)
            
        Returns:
            Filtered results
        """
        if not tags:
            return results
        
        if isinstance(tags, str):
            tags = [tags]
        
        filter_tags = set(tag.lower().strip() for tag in tags)
        filtered_results = []
        
        for result in results:
            result_tags = set(tag.lower() for tag in result.tags)
            
            if match_mode == "any":
                if filter_tags.intersection(result_tags):
                    filtered_results.append(result)
            elif match_mode == "all":
                if filter_tags.issubset(result_tags):
                    filtered_results.append(result)
            elif match_mode == "exact":
                if filter_tags == result_tags:
                    filtered_results.append(result)
        
        search_logger.debug(
            f"Tag filter applied",
            input_count=len(results),
            output_count=len(filtered_results),
            tags=list(filter_tags),
            match_mode=match_mode
        )
        
        return filtered_results
    
    def apply_time_window(
        self,
        results: List[MemoryResult],
        time_filter: TimeWindowFilter,
        reference_time: Optional[datetime] = None
    ) -> List[MemoryResult]:
        """
        Filter results by time window.
        
        Args:
            results: Results to filter
            time_filter: Time window configuration
            reference_time: Reference time for relative calculations
            
        Returns:
            Results within the time window
        """
        if not results:
            return results
        
        start_time, end_time = time_filter.get_time_range(reference_time)
        filtered_results = []
        
        for result in results:
            result_time = result.timestamp
            
            # Handle different timestamp formats
            if isinstance(result_time, str):
                try:
                    # Try parsing ISO format
                    result_time = datetime.fromisoformat(result_time.replace('Z', '+00:00'))
                except ValueError:
                    # Skip results with unparseable timestamps
                    search_logger.warning(f"Could not parse timestamp: {result_time}")
                    continue
            elif not isinstance(result_time, datetime):
                # Skip non-datetime timestamps
                continue
            
            # Ensure timezone awareness
            if result_time.tzinfo is None:
                result_time = result_time.replace(tzinfo=timezone.utc)
            
            # Check if within time window
            if start_time <= result_time <= end_time:
                filtered_results.append(result)
        
        search_logger.debug(
            f"Time window filter applied",
            input_count=len(results),
            output_count=len(filtered_results),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat()
        )
        
        return filtered_results
    
    def apply_content_type_filter(
        self,
        results: List[MemoryResult],
        content_types: Union[ContentType, List[ContentType]],
        exclude: bool = False
    ) -> List[MemoryResult]:
        """
        Filter results by content type.
        
        Args:
            results: Results to filter
            content_types: Content type(s) to match
            exclude: If True, exclude matching types instead of including
            
        Returns:
            Filtered results
        """
        if not content_types:
            return results
        
        if isinstance(content_types, ContentType):
            content_types = [content_types]
        
        target_types = set(content_types)
        filtered_results = []
        
        for result in results:
            type_match = result.type in target_types
            
            if (type_match and not exclude) or (not type_match and exclude):
                filtered_results.append(result)
        
        search_logger.debug(
            f"Content type filter applied",
            input_count=len(results),
            output_count=len(filtered_results),
            types=[t.value for t in target_types],
            exclude=exclude
        )
        
        return filtered_results
    
    def apply_source_filter(
        self,
        results: List[MemoryResult],
        sources: Union[ResultSource, List[ResultSource]],
        exclude: bool = False
    ) -> List[MemoryResult]:
        """
        Filter results by source backend.
        
        Args:
            results: Results to filter
            sources: Source(s) to match
            exclude: If True, exclude matching sources instead of including
            
        Returns:
            Filtered results
        """
        if not sources:
            return results
        
        if isinstance(sources, ResultSource):
            sources = [sources]
        
        target_sources = set(sources)
        filtered_results = []
        
        for result in results:
            source_match = result.source in target_sources
            
            if (source_match and not exclude) or (not source_match and exclude):
                filtered_results.append(result)
        
        return filtered_results
    
    def apply_score_filter(
        self,
        results: List[MemoryResult],
        min_score: Optional[float] = None,
        max_score: Optional[float] = None
    ) -> List[MemoryResult]:
        """
        Filter results by score range.
        
        Args:
            results: Results to filter
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            
        Returns:
            Results within score range
        """
        if min_score is None and max_score is None:
            return results
        
        filtered_results = []
        
        for result in results:
            score = result.score
            
            if min_score is not None and score < min_score:
                continue
            
            if max_score is not None and score > max_score:
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def apply_namespace_filter(
        self,
        results: List[MemoryResult],
        namespaces: Union[str, List[str]],
        exclude: bool = False
    ) -> List[MemoryResult]:
        """
        Filter results by namespace.
        
        Args:
            results: Results to filter
            namespaces: Namespace(s) to match
            exclude: If True, exclude matching namespaces
            
        Returns:
            Filtered results
        """
        if not namespaces:
            return results
        
        if isinstance(namespaces, str):
            namespaces = [namespaces]
        
        target_namespaces = set(namespaces)
        filtered_results = []
        
        for result in results:
            result_namespace = result.namespace
            namespace_match = result_namespace in target_namespaces
            
            if (namespace_match and not exclude) or (not namespace_match and exclude):
                filtered_results.append(result)
        
        return filtered_results
    
    def apply_text_filter(
        self,
        results: List[MemoryResult],
        pattern: str,
        operator: FilterOperator = FilterOperator.CONTAINS,
        case_sensitive: bool = False
    ) -> List[MemoryResult]:
        """
        Filter results by text content.
        
        Args:
            results: Results to filter
            pattern: Text pattern to match
            operator: Matching operator
            case_sensitive: Whether matching should be case sensitive
            
        Returns:
            Filtered results
        """
        if not pattern:
            return results
        
        filtered_results = []
        
        for result in results:
            text = result.text
            if not case_sensitive:
                text = text.lower()
                pattern = pattern.lower()
            
            match = False
            
            if operator == FilterOperator.EQUALS:
                match = text == pattern
            elif operator == FilterOperator.NOT_EQUALS:
                match = text != pattern
            elif operator == FilterOperator.CONTAINS:
                match = pattern in text
            elif operator == FilterOperator.NOT_CONTAINS:
                match = pattern not in text
            elif operator == FilterOperator.STARTS_WITH:
                match = text.startswith(pattern)
            elif operator == FilterOperator.ENDS_WITH:
                match = text.endswith(pattern)
            elif operator == FilterOperator.REGEX:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    match = bool(re.search(pattern, result.text, flags))
                except re.error:
                    search_logger.warning(f"Invalid regex pattern: {pattern}")
                    continue
            
            if match:
                filtered_results.append(result)
        
        return filtered_results
    
    def apply_custom_filter(
        self,
        results: List[MemoryResult],
        filter_name: str,
        **kwargs
    ) -> List[MemoryResult]:
        """
        Apply a custom filter function.
        
        Args:
            results: Results to filter
            filter_name: Name of registered custom filter
            **kwargs: Arguments to pass to the filter function
            
        Returns:
            Filtered results
        """
        if filter_name not in self.custom_filters:
            raise ValueError(f"Custom filter '{filter_name}' not registered")
        
        filter_func = self.custom_filters[filter_name]
        return filter_func(results, **kwargs)
    
    def register_custom_filter(self, name: str, filter_func: Callable) -> None:
        """
        Register a custom filter function.
        
        Args:
            name: Name for the filter
            filter_func: Function that takes (results, **kwargs) and returns filtered results
        """
        self.custom_filters[name] = filter_func
        search_logger.info(f"Registered custom filter: {name}")
    
    def apply_criteria_filter(
        self,
        results: List[MemoryResult],
        criteria: List[FilterCriteria]
    ) -> List[MemoryResult]:
        """
        Apply multiple filter criteria.
        
        Args:
            results: Results to filter
            criteria: List of filter criteria to apply
            
        Returns:
            Results matching all criteria
        """
        if not criteria:
            return results
        
        filtered_results = results
        
        for criterion in criteria:
            filtered_results = self._apply_single_criterion(filtered_results, criterion)
        
        return filtered_results
    
    def _apply_single_criterion(
        self,
        results: List[MemoryResult],
        criterion: FilterCriteria
    ) -> List[MemoryResult]:
        """Apply a single filter criterion."""
        filtered_results = []
        
        for result in results:
            # Get field value
            field_value = self._get_field_value(result, criterion.field)
            if field_value is None:
                continue  # Skip results without the field
            
            # Apply operator
            if self._evaluate_criterion(field_value, criterion):
                filtered_results.append(result)
        
        return filtered_results
    
    def _get_field_value(self, result: MemoryResult, field: str) -> Any:
        """Extract field value from result."""
        # Handle nested field access (e.g., "metadata.confidence")
        if '.' in field:
            parts = field.split('.')
            value = result
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        
        # Handle direct field access
        if hasattr(result, field):
            return getattr(result, field)
        elif field in result.metadata:
            return result.metadata[field]
        
        return None
    
    def _evaluate_criterion(self, field_value: Any, criterion: FilterCriteria) -> bool:
        """Evaluate if field value matches criterion."""
        value = criterion.value
        operator = criterion.operator
        
        # Handle string comparisons
        if isinstance(field_value, str) and isinstance(value, str):
            if not criterion.case_sensitive:
                field_value = field_value.lower()
                value = value.lower()
        
        try:
            if operator == FilterOperator.EQUALS:
                return field_value == value
            elif operator == FilterOperator.NOT_EQUALS:
                return field_value != value
            elif operator == FilterOperator.CONTAINS:
                return str(value) in str(field_value)
            elif operator == FilterOperator.NOT_CONTAINS:
                return str(value) not in str(field_value)
            elif operator == FilterOperator.STARTS_WITH:
                return str(field_value).startswith(str(value))
            elif operator == FilterOperator.ENDS_WITH:
                return str(field_value).endswith(str(value))
            elif operator == FilterOperator.REGEX:
                flags = 0 if criterion.case_sensitive else re.IGNORECASE
                return bool(re.search(str(value), str(field_value), flags))
            elif operator == FilterOperator.IN:
                return field_value in value if hasattr(value, '__contains__') else False
            elif operator == FilterOperator.NOT_IN:
                return field_value not in value if hasattr(value, '__contains__') else True
            elif operator == FilterOperator.GREATER_THAN:
                return field_value > value
            elif operator == FilterOperator.LESS_THAN:
                return field_value < value
            elif operator == FilterOperator.GREATER_EQUAL:
                return field_value >= value
            elif operator == FilterOperator.LESS_EQUAL:
                return field_value <= value
            elif operator == FilterOperator.RANGE:
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return value[0] <= field_value <= value[1]
                return False
            
        except (TypeError, ValueError) as e:
            search_logger.warning(f"Filter evaluation error: {e}")
            return False
        
        return False


# Convenience functions for common filtering operations

def filter_by_tags(
    results: List[MemoryResult],
    tags: Union[str, List[str]],
    match_mode: str = "any"
) -> List[MemoryResult]:
    """Convenience function for tag filtering."""
    engine = PreFilterEngine()
    return engine.apply_tag_filter(results, tags, match_mode)


def filter_by_time_window(
    results: List[MemoryResult],
    hours: Optional[int] = None,
    days: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[MemoryResult]:
    """Convenience function for time window filtering."""
    engine = PreFilterEngine()
    time_filter = TimeWindowFilter(
        start_time=start_time,
        end_time=end_time,
        relative_hours=hours,
        relative_days=days
    )
    return engine.apply_time_window(results, time_filter)


def filter_by_content_type(
    results: List[MemoryResult],
    content_types: Union[ContentType, List[ContentType]],
    exclude: bool = False
) -> List[MemoryResult]:
    """Convenience function for content type filtering."""
    engine = PreFilterEngine()
    return engine.apply_content_type_filter(results, content_types, exclude)


def filter_by_score_threshold(
    results: List[MemoryResult],
    min_score: float
) -> List[MemoryResult]:
    """Convenience function for score threshold filtering."""
    engine = PreFilterEngine()
    return engine.apply_score_filter(results, min_score=min_score)


# Global filter engine instance
pre_filter_engine = PreFilterEngine()