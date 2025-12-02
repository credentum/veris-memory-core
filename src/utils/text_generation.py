#!/usr/bin/env python3
"""
Text generation utilities for creating searchable representations of contexts.

This module provides functions to generate unified searchable text from context data,
enabling search across both standard fields and custom properties.
"""

from typing import Dict, Any

# System fields that should not be included in searchable text
SYSTEM_FIELDS = {
    'id', 'type', 'created_at', 'updated_at', 'author',
    'author_type', 'timestamp', 'vector_id', 'graph_id',
    'namespace', 'tags', 'metadata', 'embedding_status',
    'relationships_created', 'session_id', 'source',
    'test_type', 'sentinel', 'bug_report', 'reporter',
    'priority', 'duration_ms'
}


def generate_searchable_text(data: Dict[str, Any]) -> str:
    """
    Generate unified searchable text representation from context data.

    This function creates a comprehensive searchable text field that includes:
    - Standard content fields (title, description, keyword, user_input, bot_response)
    - Custom properties formatted as searchable phrases
    - Multiple representations of each property for better search recall

    Args:
        data: Context data dictionary containing both standard and custom fields

    Returns:
        Searchable text string combining all searchable content

    Example:
        >>> data = {"title": "User Info", "food": "spicy", "name": "Matt"}
        >>> generate_searchable_text(data)
        'User Info food is spicy food: spicy spicy name is Matt name: Matt Matt'
    """
    text_parts = []

    # Include standard content fields first (maintain existing search behavior)
    standard_fields = ['title', 'description', 'keyword', 'user_input', 'bot_response']
    for field in standard_fields:
        value = data.get(field)
        if value and isinstance(value, str) and value.strip():
            text_parts.append(value.strip())

    # Include custom properties as searchable phrases
    custom_properties = []
    for key, value in data.items():
        # Skip system fields and None values
        if key in SYSTEM_FIELDS or value is None:
            continue

        # Skip standard fields already processed
        if key in standard_fields:
            continue

        # Only process simple types (not nested objects/lists)
        if isinstance(value, (str, int, float, bool)):
            # Convert to string and create searchable phrases
            value_str = str(value).strip()
            if value_str:
                # Multiple representations for better search recall
                custom_properties.append(f"{key} is {value_str}")
                custom_properties.append(f"{key}: {value_str}")
                custom_properties.append(value_str)

    # Combine standard fields and custom properties
    text_parts.extend(custom_properties)

    # Join with spaces and return
    searchable_text = " ".join(text_parts)

    return searchable_text


def extract_custom_properties(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract custom properties from context data (non-system, non-standard fields).

    Args:
        data: Context data dictionary

    Returns:
        Dictionary containing only custom properties

    Example:
        >>> data = {"id": "123", "food": "spicy", "title": "Info"}
        >>> extract_custom_properties(data)
        {'food': 'spicy'}
    """
    standard_fields = {'title', 'description', 'keyword', 'user_input', 'bot_response'}
    custom_props = {}

    for key, value in data.items():
        if key not in SYSTEM_FIELDS and key not in standard_fields and value is not None:
            if isinstance(value, (str, int, float, bool)):
                custom_props[key] = value

    return custom_props


def count_searchable_properties(data: Dict[str, Any]) -> int:
    """
    Count the number of searchable properties in context data.

    Args:
        data: Context data dictionary

    Returns:
        Count of searchable properties (standard + custom)
    """
    count = 0

    # Count standard fields
    standard_fields = ['title', 'description', 'keyword', 'user_input', 'bot_response']
    for field in standard_fields:
        if data.get(field):
            count += 1

    # Count custom properties
    custom_props = extract_custom_properties(data)
    count += len(custom_props)

    return count
