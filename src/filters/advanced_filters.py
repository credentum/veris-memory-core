#!/usr/bin/env python3
"""
Advanced filtering system for enhanced search capabilities.

This module extends the basic filtering capabilities with advanced filters
including date ranges, content matching, fuzzy matching, and combined filters.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Callable, Set
from dataclasses import dataclass
from enum import Enum

from ..interfaces.memory_result import MemoryResult
from ..utils.logging_middleware import search_logger


class FilterType(str, Enum):
    """Types of filters available."""
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    REGEX = "regex"
    DATE_RANGE = "date_range"
    NUMERIC_RANGE = "numeric_range"
    TAG_FILTER = "tag_filter"
    SOURCE_FILTER = "source_filter"
    CONTENT_TYPE = "content_type"
    SCORE_THRESHOLD = "score_threshold"
    FUZZY_MATCH = "fuzzy_match"
    CUSTOM = "custom"


class CombineMode(str, Enum):
    """How to combine multiple filter criteria."""
    AND = "and"  # All filters must match
    OR = "or"    # Any filter can match
    NOT = "not"  # Exclude matching results


@dataclass
class FilterCriteria:
    """Advanced filter criteria specification."""
    field: str  # Field name to filter on (e.g., "text", "tags", "metadata.author")
    filter_type: FilterType
    value: Any  # Filter value or criteria
    combine_mode: CombineMode = CombineMode.AND
    case_sensitive: bool = False
    description: Optional[str] = None


@dataclass
class DateRangeFilter:
    """Date range filtering specification."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    relative_days: Optional[int] = None  # Filter to last N days
    field: str = "timestamp"
    
    def to_criteria(self) -> FilterCriteria:
        """Convert to FilterCriteria."""
        if self.relative_days:
            cutoff = datetime.now() - timedelta(days=self.relative_days)
            return FilterCriteria(
                field=self.field,
                filter_type=FilterType.DATE_RANGE,
                value={"start": cutoff, "end": None},
                description=f"Last {self.relative_days} days"
            )
        else:
            return FilterCriteria(
                field=self.field,
                filter_type=FilterType.DATE_RANGE,
                value={"start": self.start_date, "end": self.end_date},
                description=f"Date range: {self.start_date} to {self.end_date}"
            )


@dataclass
class NumericRangeFilter:
    """Numeric range filtering specification."""
    field: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def to_criteria(self) -> FilterCriteria:
        """Convert to FilterCriteria."""
        return FilterCriteria(
            field=self.field,
            filter_type=FilterType.NUMERIC_RANGE,
            value={"min": self.min_value, "max": self.max_value},
            description=f"Range: {self.min_value} to {self.max_value}"
        )


class FuzzyMatcher:
    """Fuzzy string matching utilities."""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return FuzzyMatcher.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def similarity_ratio(s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings (0-1)."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = FuzzyMatcher.levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)
    
    @staticmethod
    def fuzzy_match(text: str, pattern: str, threshold: float = 0.8) -> bool:
        """Check if text fuzzy matches pattern with given threshold."""
        return FuzzyMatcher.similarity_ratio(text.lower(), pattern.lower()) >= threshold


class AdvancedFilterEngine:
    """
    Advanced filtering engine with support for multiple filter types and combinations.
    """
    
    def __init__(self):
        """Initialize the advanced filter engine."""
        self.custom_filters: Dict[str, Callable] = {}
        self._field_extractors: Dict[str, Callable] = {
            "text": lambda r: r.text,
            "content_type": lambda r: r.content_type,
            "source": lambda r: r.source,
            "score": lambda r: r.score,
            "timestamp": lambda r: r.timestamp,
            "tags": lambda r: r.tags,
            "metadata": lambda r: r.metadata,
            "id": lambda r: r.id
        }
    
    def register_custom_filter(self, name: str, filter_func: Callable[[MemoryResult, Any], bool]) -> None:
        """
        Register a custom filter function.
        
        Args:
            name: Filter name
            filter_func: Function that takes (MemoryResult, criteria) and returns bool
        """
        self.custom_filters[name] = filter_func
        search_logger.info(f"Registered custom filter: {name}")
    
    def register_field_extractor(self, field_name: str, extractor_func: Callable[[MemoryResult], Any]) -> None:
        """
        Register a custom field extractor function.
        
        Args:
            field_name: Field name to extract
            extractor_func: Function that takes MemoryResult and returns field value
        """
        self._field_extractors[field_name] = extractor_func
        search_logger.info(f"Registered field extractor: {field_name}")
    
    def extract_field_value(self, result: MemoryResult, field_path: str) -> Any:
        """
        Extract field value from MemoryResult using dot notation.
        
        Args:
            result: MemoryResult to extract from
            field_path: Field path (e.g., "metadata.author", "tags")
            
        Returns:
            Field value or None if not found
        """
        try:
            # Handle simple field names
            if "." not in field_path:
                if field_path in self._field_extractors:
                    return self._field_extractors[field_path](result)
                else:
                    return getattr(result, field_path, None)
            
            # Handle nested field paths (e.g., "metadata.author")
            parts = field_path.split(".")
            value = result
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            
            return value
            
        except Exception as e:
            search_logger.warning(f"Failed to extract field '{field_path}': {e}")
            return None
    
    def apply_single_filter(self, results: List[MemoryResult], criteria: FilterCriteria) -> List[MemoryResult]:
        """
        Apply a single filter criteria to results.
        
        Args:
            results: List of results to filter
            criteria: Filter criteria to apply
            
        Returns:
            Filtered list of results
        """
        if not results:
            return results
        
        try:
            filtered_results = []
            
            for result in results:
                if self._matches_criteria(result, criteria):
                    filtered_results.append(result)
            
            search_logger.debug(
                f"Applied {criteria.filter_type.value} filter",
                field=criteria.field,
                input_count=len(results),
                output_count=len(filtered_results),
                description=criteria.description
            )
            
            return filtered_results
            
        except Exception as e:
            search_logger.error(f"Filter application failed: {e}")
            return results  # Return unfiltered results on error
    
    def apply_multiple_filters(
        self, 
        results: List[MemoryResult], 
        criteria_list: List[FilterCriteria],
        global_combine_mode: CombineMode = CombineMode.AND
    ) -> List[MemoryResult]:
        """
        Apply multiple filter criteria with combination logic.
        
        Args:
            results: List of results to filter
            criteria_list: List of filter criteria
            global_combine_mode: How to combine all criteria
            
        Returns:
            Filtered list of results
        """
        if not results or not criteria_list:
            return results
        
        try:
            if global_combine_mode == CombineMode.AND:
                # All filters must match - apply sequentially
                filtered = results
                for criteria in criteria_list:
                    filtered = self.apply_single_filter(filtered, criteria)
                return filtered
                
            elif global_combine_mode == CombineMode.OR:
                # Any filter can match - collect all matches
                matched_ids = set()
                all_matches = []
                
                for criteria in criteria_list:
                    matches = self.apply_single_filter(results, criteria)
                    for match in matches:
                        if match.id not in matched_ids:
                            matched_ids.add(match.id)
                            all_matches.append(match)
                
                return all_matches
                
            elif global_combine_mode == CombineMode.NOT:
                # Exclude all matches
                exclude_ids = set()
                
                for criteria in criteria_list:
                    matches = self.apply_single_filter(results, criteria)
                    exclude_ids.update(match.id for match in matches)
                
                return [r for r in results if r.id not in exclude_ids]
            
            else:
                search_logger.warning(f"Unknown combine mode: {global_combine_mode}")
                return results
                
        except Exception as e:
            search_logger.error(f"Multiple filter application failed: {e}")
            return results
    
    def apply_date_range_filter(
        self, 
        results: List[MemoryResult], 
        date_filter: DateRangeFilter
    ) -> List[MemoryResult]:
        """
        Apply date range filtering.
        
        Args:
            results: List of results to filter
            date_filter: Date range filter specification
            
        Returns:
            Filtered list of results
        """
        criteria = date_filter.to_criteria()
        return self.apply_single_filter(results, criteria)
    
    def apply_tag_filters(
        self, 
        results: List[MemoryResult], 
        required_tags: Optional[List[str]] = None,
        excluded_tags: Optional[List[str]] = None
    ) -> List[MemoryResult]:
        """
        Apply tag-based filtering.
        
        Args:
            results: List of results to filter
            required_tags: Tags that must be present (AND logic)
            excluded_tags: Tags that must not be present
            
        Returns:
            Filtered list of results
        """
        filtered = results
        
        if required_tags:
            criteria = FilterCriteria(
                field="tags",
                filter_type=FilterType.TAG_FILTER,
                value={"required": required_tags, "mode": "all"},
                description=f"Required tags: {required_tags}"
            )
            filtered = self.apply_single_filter(filtered, criteria)
        
        if excluded_tags:
            criteria = FilterCriteria(
                field="tags",
                filter_type=FilterType.TAG_FILTER,
                value={"excluded": excluded_tags},
                combine_mode=CombineMode.NOT,
                description=f"Excluded tags: {excluded_tags}"
            )
            filtered = self.apply_single_filter(filtered, criteria)
        
        return filtered
    
    def apply_content_search(
        self, 
        results: List[MemoryResult], 
        search_text: str,
        match_type: FilterType = FilterType.CONTAINS,
        case_sensitive: bool = False
    ) -> List[MemoryResult]:
        """
        Apply content-based text search filtering.
        
        Args:
            results: List of results to filter
            search_text: Text to search for
            match_type: Type of text matching
            case_sensitive: Whether matching is case sensitive
            
        Returns:
            Filtered list of results
        """
        criteria = FilterCriteria(
            field="text",
            filter_type=match_type,
            value=search_text,
            case_sensitive=case_sensitive,
            description=f"Content search: '{search_text}'"
        )
        
        return self.apply_single_filter(results, criteria)
    
    def get_filter_statistics(self, results: List[MemoryResult]) -> Dict[str, Any]:
        """
        Get statistics about filterable fields in results.
        
        Args:
            results: Results to analyze
            
        Returns:
            Dictionary with field statistics
        """
        if not results:
            return {}
        
        stats = {
            "total_results": len(results),
            "sources": {},
            "content_types": {},
            "tag_frequency": {},
            "score_range": {"min": float("inf"), "max": float("-inf")},
            "date_range": {"earliest": None, "latest": None}
        }
        
        for result in results:
            # Source distribution
            source = result.source
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            # Content type distribution
            content_type = result.content_type
            stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            
            # Tag frequency
            for tag in result.tags:
                stats["tag_frequency"][tag] = stats["tag_frequency"].get(tag, 0) + 1
            
            # Score range
            stats["score_range"]["min"] = min(stats["score_range"]["min"], result.score)
            stats["score_range"]["max"] = max(stats["score_range"]["max"], result.score)
            
            # Date range
            if result.timestamp:
                if stats["date_range"]["earliest"] is None or result.timestamp < stats["date_range"]["earliest"]:
                    stats["date_range"]["earliest"] = result.timestamp
                if stats["date_range"]["latest"] is None or result.timestamp > stats["date_range"]["latest"]:
                    stats["date_range"]["latest"] = result.timestamp
        
        # Sort by frequency
        stats["sources"] = dict(sorted(stats["sources"].items(), key=lambda x: x[1], reverse=True))
        stats["content_types"] = dict(sorted(stats["content_types"].items(), key=lambda x: x[1], reverse=True))
        stats["tag_frequency"] = dict(sorted(stats["tag_frequency"].items(), key=lambda x: x[1], reverse=True)[:20])  # Top 20 tags
        
        return stats
    
    # Private methods
    
    def _matches_criteria(self, result: MemoryResult, criteria: FilterCriteria) -> bool:
        """Check if a result matches the given criteria."""
        try:
            field_value = self.extract_field_value(result, criteria.field)
            
            if field_value is None:
                return False
            
            return self._evaluate_filter(field_value, criteria)
            
        except Exception as e:
            search_logger.warning(f"Criteria evaluation failed for {criteria.field}: {e}")
            return False
    
    def _evaluate_filter(self, field_value: Any, criteria: FilterCriteria) -> bool:
        """Evaluate if field value matches filter criteria."""
        filter_type = criteria.filter_type
        target_value = criteria.value
        case_sensitive = criteria.case_sensitive
        
        if filter_type == FilterType.EXACT_MATCH:
            if case_sensitive:
                return str(field_value) == str(target_value)
            else:
                return str(field_value).lower() == str(target_value).lower()
        
        elif filter_type == FilterType.CONTAINS:
            text = str(field_value)
            pattern = str(target_value)
            if not case_sensitive:
                text = text.lower()
                pattern = pattern.lower()
            return pattern in text
        
        elif filter_type == FilterType.REGEX:
            flags = 0 if case_sensitive else re.IGNORECASE
            return bool(re.search(str(target_value), str(field_value), flags))
        
        elif filter_type == FilterType.DATE_RANGE:
            if not isinstance(field_value, datetime):
                return False
            start_date = target_value.get("start")
            end_date = target_value.get("end")
            
            if start_date and field_value < start_date:
                return False
            if end_date and field_value > end_date:
                return False
            return True
        
        elif filter_type == FilterType.NUMERIC_RANGE:
            try:
                num_value = float(field_value)
                min_val = target_value.get("min")
                max_val = target_value.get("max")
                
                if min_val is not None and num_value < min_val:
                    return False
                if max_val is not None and num_value > max_val:
                    return False
                return True
            except (ValueError, TypeError):
                return False
        
        elif filter_type == FilterType.TAG_FILTER:
            if not isinstance(field_value, list):
                return False
            
            tags = set(field_value)
            required = target_value.get("required", [])
            excluded = target_value.get("excluded", [])
            mode = target_value.get("mode", "any")  # "any" or "all"
            
            # Check excluded tags
            if excluded and any(tag in tags for tag in excluded):
                return False
            
            # Check required tags
            if required:
                if mode == "all":
                    return all(tag in tags for tag in required)
                else:  # mode == "any"
                    return any(tag in tags for tag in required)
            
            return True
        
        elif filter_type == FilterType.SOURCE_FILTER:
            return str(field_value) in (target_value if isinstance(target_value, list) else [target_value])
        
        elif filter_type == FilterType.CONTENT_TYPE:
            return str(field_value) in (target_value if isinstance(target_value, list) else [target_value])
        
        elif filter_type == FilterType.SCORE_THRESHOLD:
            try:
                return float(field_value) >= float(target_value)
            except (ValueError, TypeError):
                return False
        
        elif filter_type == FilterType.FUZZY_MATCH:
            threshold = target_value.get("threshold", 0.8)
            pattern = target_value.get("pattern", "")
            return FuzzyMatcher.fuzzy_match(str(field_value), pattern, threshold)
        
        elif filter_type == FilterType.CUSTOM:
            filter_name = target_value.get("name")
            if filter_name in self.custom_filters:
                return self.custom_filters[filter_name](result, target_value.get("params", {}))
            return False
        
        else:
            search_logger.warning(f"Unknown filter type: {filter_type}")
            return False


# Global instance
advanced_filter_engine = AdvancedFilterEngine()