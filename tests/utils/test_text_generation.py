#!/usr/bin/env python3
"""
Unit tests for text generation utilities.

Tests the generate_searchable_text function and related utilities.
"""

import pytest
from src.utils.text_generation import (
    generate_searchable_text,
    extract_custom_properties,
    count_searchable_properties,
    SYSTEM_FIELDS
)


class TestGenerateSearchableText:
    """Test suite for generate_searchable_text function."""

    def test_standard_fields_only(self):
        """Test searchable text generation with only standard fields."""
        data = {
            "title": "User Profile",
            "description": "Profile information",
            "keyword": "user_data"
        }

        result = generate_searchable_text(data)

        assert "User Profile" in result
        assert "Profile information" in result
        assert "user_data" in result

    def test_custom_properties_only(self):
        """Test searchable text generation with only custom properties."""
        data = {
            "food": "spicy",
            "name": "Matt",
            "location": "San Francisco"
        }

        result = generate_searchable_text(data)

        # Should contain multiple representations
        assert "food is spicy" in result
        assert "food: spicy" in result
        assert "spicy" in result

        assert "name is Matt" in result
        assert "name: Matt" in result
        assert "Matt" in result

        assert "location is San Francisco" in result
        assert "location: San Francisco" in result
        assert "San Francisco" in result

    def test_mixed_standard_and_custom(self):
        """Test with both standard fields and custom properties."""
        data = {
            "title": "User Facts",
            "description": "User preferences",
            "food": "spicy",
            "name": "Matt"
        }

        result = generate_searchable_text(data)

        # Standard fields should come first
        assert result.startswith("User Facts")

        # Custom properties should be included
        assert "food is spicy" in result
        assert "name is Matt" in result

    def test_system_fields_excluded(self):
        """Test that system fields are not included in searchable text."""
        data = {
            "id": "test-id-123",
            "type": "log",
            "created_at": "2025-11-25",
            "author": "test_user",
            "food": "spicy"
        }

        result = generate_searchable_text(data)

        # System fields should not appear
        assert "test-id-123" not in result
        assert "log" not in result
        assert "test_user" not in result

        # Custom properties should appear
        assert "food is spicy" in result

    def test_empty_data(self):
        """Test with empty dictionary."""
        data = {}
        result = generate_searchable_text(data)
        assert result == ""

    def test_none_values_skipped(self):
        """Test that None values are not included."""
        data = {
            "title": "Test",
            "description": None,
            "food": None,
            "name": "Matt"
        }

        result = generate_searchable_text(data)

        assert "Test" in result
        assert "None" not in result
        assert "name is Matt" in result

    def test_empty_strings_skipped(self):
        """Test that empty strings are not included."""
        data = {
            "title": "",
            "description": "   ",
            "food": "spicy"
        }

        result = generate_searchable_text(data)

        # Empty/whitespace strings should not appear
        assert not result.startswith(" ")

        # Valid content should appear
        assert "food is spicy" in result

    def test_numeric_values(self):
        """Test with numeric custom properties."""
        data = {
            "age": 25,
            "score": 98.5,
            "active": True
        }

        result = generate_searchable_text(data)

        assert "age is 25" in result
        assert "age: 25" in result
        assert "25" in result

        assert "score is 98.5" in result
        assert "score: 98.5" in result

        assert "active is True" in result

    def test_voice_bot_context(self):
        """Test with voice bot conversation context."""
        data = {
            "user_input": "You are my sunshine",
            "bot_response": "That's a lovely song!",
            "session_id": "voice_123"
        }

        result = generate_searchable_text(data)

        # Standard fields should be included
        assert "You are my sunshine" in result
        assert "That's a lovely song!" in result

        # System field should be excluded
        assert "voice_123" not in result

    def test_fact_context(self):
        """Test with user fact context (VoiceBot use case)."""
        data = {
            "content_type": "fact",
            "food": "spicy",
            "name": "Matt",
            "location": "San Francisco",
            "test_type": "golden_recall"
        }

        result = generate_searchable_text(data)

        # All three facts should be searchable
        assert "food is spicy" in result
        assert "name is Matt" in result
        assert "location is San Francisco" in result

        # Each value should appear standalone
        assert "spicy" in result
        assert "Matt" in result
        assert "San Francisco" in result

    def test_whitespace_handling(self):
        """Test proper whitespace handling in generated text."""
        data = {
            "title": "  Test Title  ",
            "food": " spicy "
        }

        result = generate_searchable_text(data)

        # Should not have double spaces
        assert "  " not in result

        # Should have trimmed values
        assert "Test Title" in result
        assert "spicy" in result

    def test_nested_objects_skipped(self):
        """Test that nested objects/lists are not included."""
        data = {
            "title": "Test",
            "metadata": {"key": "value"},  # Should be skipped
            "tags": ["tag1", "tag2"],      # Should be skipped
            "food": "spicy"
        }

        result = generate_searchable_text(data)

        assert "Test" in result
        assert "food is spicy" in result

        # Nested structures should not appear
        assert "key" not in result
        assert "value" not in result
        assert "tag1" not in result


class TestExtractCustomProperties:
    """Test suite for extract_custom_properties function."""

    def test_extract_only_custom(self):
        """Test extracting only custom properties."""
        data = {
            "id": "123",
            "title": "Test",
            "food": "spicy",
            "name": "Matt"
        }

        custom = extract_custom_properties(data)

        assert custom == {"food": "spicy", "name": "Matt"}
        assert "id" not in custom
        assert "title" not in custom

    def test_no_custom_properties(self):
        """Test when there are no custom properties."""
        data = {
            "id": "123",
            "title": "Test",
            "description": "Desc"
        }

        custom = extract_custom_properties(data)

        assert custom == {}

    def test_skip_complex_types(self):
        """Test that complex types are not extracted."""
        data = {
            "food": "spicy",
            "metadata": {"key": "value"},
            "tags": ["tag1"]
        }

        custom = extract_custom_properties(data)

        assert custom == {"food": "spicy"}


class TestCountSearchableProperties:
    """Test suite for count_searchable_properties function."""

    def test_count_standard_fields(self):
        """Test counting standard fields."""
        data = {
            "title": "Test",
            "description": "Desc",
            "keyword": "key"
        }

        count = count_searchable_properties(data)
        assert count == 3

    def test_count_custom_properties(self):
        """Test counting custom properties."""
        data = {
            "food": "spicy",
            "name": "Matt"
        }

        count = count_searchable_properties(data)
        assert count == 2

    def test_count_mixed(self):
        """Test counting mixed standard and custom."""
        data = {
            "title": "Test",
            "food": "spicy",
            "name": "Matt"
        }

        count = count_searchable_properties(data)
        assert count == 3

    def test_count_excludes_system_fields(self):
        """Test that system fields are not counted."""
        data = {
            "id": "123",
            "type": "log",
            "title": "Test",
            "food": "spicy"
        }

        count = count_searchable_properties(data)
        assert count == 2  # Only title and food


class TestSystemFieldsConstant:
    """Test suite for SYSTEM_FIELDS constant."""

    def test_system_fields_contains_core_fields(self):
        """Test that SYSTEM_FIELDS includes all expected system fields."""
        required_fields = {
            'id', 'type', 'created_at', 'author', 'author_type',
            'timestamp', 'vector_id', 'graph_id'
        }

        assert required_fields.issubset(SYSTEM_FIELDS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
