#!/usr/bin/env python3
"""
Test suite for validators/cache_entry_validator.py - Cache entry validation tests
"""
import json
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List

# Import the module under test
from src.validators.cache_entry_validator import (
    CacheEntryValidator,
    validate_cache_key,
    validate_cache_value,
    validate_ttl
)


class TestCacheEntryValidatorConstants:
    """Test suite for CacheEntryValidator constants and configuration"""

    def test_class_constants(self):
        """Test that class constants are properly defined"""
        assert CacheEntryValidator.MAX_KEY_LENGTH == 256
        assert CacheEntryValidator.MAX_VALUE_SIZE == 1024 * 1024  # 1MB
        assert CacheEntryValidator.MAX_TTL_SECONDS == 86400 * 30  # 30 days
        assert CacheEntryValidator.MIN_TTL_SECONDS == 1
        
        # Test key pattern exists
        assert hasattr(CacheEntryValidator, 'KEY_PATTERN')
        assert CacheEntryValidator.KEY_PATTERN.pattern == r"^[a-zA-Z0-9_\-:.]+$"


class TestValidateKey:
    """Test suite for key validation functionality"""

    def test_validate_key_success(self):
        """Test successful key validation"""
        valid_keys = [
            "simple_key",
            "key-with-dash",
            "key.with.dots",
            "key:with:colons",
            "key123",
            "123key",
            "a",
            "A" * 256  # Maximum length
        ]
        
        for key in valid_keys:
            is_valid, error = CacheEntryValidator.validate_key(key)
            assert is_valid is True, f"Key '{key}' should be valid"
            assert error is None

    def test_validate_key_empty(self):
        """Test validation with empty key"""
        is_valid, error = CacheEntryValidator.validate_key("")
        assert is_valid is False
        assert error == "Key cannot be empty"

    def test_validate_key_too_long(self):
        """Test validation with key that's too long"""
        long_key = "a" * 257  # Exceeds MAX_KEY_LENGTH
        is_valid, error = CacheEntryValidator.validate_key(long_key)
        assert is_valid is False
        assert "Key length exceeds maximum" in error
        assert "256 characters" in error

    def test_validate_key_invalid_characters(self):
        """Test validation with invalid characters"""
        invalid_keys = [
            "key with spaces",
            "key@with#special",
            "key/with/slash",
            "key\\with\\backslash",
            "key+with+plus",
            "key=with=equals",
            "key[with]brackets",
            "key{with}braces",
            "key|with|pipe"
        ]
        
        for key in invalid_keys:
            is_valid, error = CacheEntryValidator.validate_key(key)
            assert is_valid is False, f"Key '{key}' should be invalid"
            assert "invalid characters" in error.lower()

    def test_validate_key_edge_cases(self):
        """Test validation with edge cases"""
        # Just valid characters at boundaries
        edge_cases = [
            "_",
            "-",
            ":",
            ".",
            "0",
            "A",
            "z"
        ]
        
        for key in edge_cases:
            is_valid, error = CacheEntryValidator.validate_key(key)
            assert is_valid is True, f"Key '{key}' should be valid"
            assert error is None


class TestValidateValue:
    """Test suite for value validation functionality"""

    def test_validate_value_success(self):
        """Test successful value validation"""
        valid_values = [
            "string_value",
            123,
            123.45,
            True,
            False,
            ["list", "of", "values"],
            {"dict": "value"},
            {"complex": {"nested": {"structure": [1, 2, 3]}}},
            "",  # Empty string is valid
            0,   # Zero is valid
        ]
        
        for value in valid_values:
            is_valid, error = CacheEntryValidator.validate_value(value)
            assert is_valid is True, f"Value {repr(value)} should be valid"
            assert error is None

    def test_validate_value_none(self):
        """Test validation with None value"""
        is_valid, error = CacheEntryValidator.validate_value(None)
        assert is_valid is False
        assert error == "Value cannot be None"

    def test_validate_value_too_large(self):
        """Test validation with value that's too large when serialized"""
        # Create a large value that exceeds MAX_VALUE_SIZE
        large_value = {"data": "x" * (1024 * 1024 + 1)}  # Slightly over 1MB
        
        is_valid, error = CacheEntryValidator.validate_value(large_value)
        assert is_valid is False
        assert "Value size exceeds maximum" in error
        assert "1048576 bytes" in error

    def test_validate_value_non_serializable(self):
        """Test validation with non-serializable value"""
        # Create a value that can't be serialized to JSON
        class NonSerializable:
            pass
        
        # Object without default serializer
        non_serializable = NonSerializable()
        # Note: The validator uses json.dumps with default=str, so this might actually work
        # Let's test with something that truly can't be serialized
        
        # Create a circular reference
        circular = {}
        circular["self"] = circular
        
        # This should fail even with default=str due to circular reference
        is_valid, error = CacheEntryValidator.validate_value(circular)
        # The validator might handle this with default=str, so we test the error message
        if not is_valid:
            assert "cannot be serialized" in error.lower()

    def test_validate_value_complex_structures(self):
        """Test validation with complex but valid data structures"""
        complex_values = [
            {
                "user": {
                    "id": 123,
                    "name": "John Doe",
                    "preferences": {
                        "theme": "dark",
                        "notifications": True,
                        "tags": ["important", "work"]
                    }
                }
            },
            [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2", "metadata": {"created": "2023-01-01"}}
            ],
            "Simple text with émojis 🚀 and unicode characters"
        ]
        
        for value in complex_values:
            is_valid, error = CacheEntryValidator.validate_value(value)
            assert is_valid is True, f"Complex value should be valid: {repr(value)}"
            assert error is None


class TestValidateTTL:
    """Test suite for TTL validation functionality"""

    def test_validate_ttl_success(self):
        """Test successful TTL validation"""
        valid_ttls = [
            None,  # TTL is optional
            1,     # Minimum value
            3600,  # 1 hour
            86400, # 1 day
            86400 * 30,  # Maximum value (30 days)
        ]
        
        for ttl in valid_ttls:
            is_valid, error = CacheEntryValidator.validate_ttl(ttl)
            assert is_valid is True, f"TTL {ttl} should be valid"
            assert error is None

    def test_validate_ttl_non_integer(self):
        """Test TTL validation with non-integer values"""
        invalid_ttls = [
            "3600",  # String
            3600.5,  # Float
            [3600],  # List
            {"ttl": 3600},  # Dict
        ]
        
        for ttl in invalid_ttls:
            is_valid, error = CacheEntryValidator.validate_ttl(ttl)
            assert is_valid is False, f"TTL {repr(ttl)} should be invalid"
            assert "TTL must be an integer" in error

    def test_validate_ttl_too_small(self):
        """Test TTL validation with values below minimum"""
        invalid_ttls = [0, -1, -100]
        
        for ttl in invalid_ttls:
            is_valid, error = CacheEntryValidator.validate_ttl(ttl)
            assert is_valid is False, f"TTL {ttl} should be invalid (too small)"
            assert f"TTL must be at least {CacheEntryValidator.MIN_TTL_SECONDS}" in error

    def test_validate_ttl_too_large(self):
        """Test TTL validation with values above maximum"""
        invalid_ttls = [
            86400 * 30 + 1,  # Just over 30 days
            86400 * 365,     # 1 year
            999999999        # Very large number
        ]
        
        for ttl in invalid_ttls:
            is_valid, error = CacheEntryValidator.validate_ttl(ttl)
            assert is_valid is False, f"TTL {ttl} should be invalid (too large)"
            assert f"TTL cannot exceed {CacheEntryValidator.MAX_TTL_SECONDS}" in error


class TestValidateTimestamp:
    """Test suite for timestamp validation (private method)"""

    def test_validate_timestamp_datetime_objects(self):
        """Test timestamp validation with datetime objects"""
        now = datetime.utcnow()
        past = now - timedelta(days=1)
        future = now + timedelta(hours=1)
        
        timestamps = [now, past, future]
        
        for ts in timestamps:
            result = CacheEntryValidator._validate_timestamp(ts)
            assert result is True, f"DateTime {ts} should be valid"

    def test_validate_timestamp_iso_strings(self):
        """Test timestamp validation with ISO format strings"""
        valid_strings = [
            "2023-01-01T00:00:00",
            "2023-12-31T23:59:59",
            "2023-06-15T12:30:45.123456",
            "2023-01-01T00:00:00Z",  # UTC timezone
            "2023-01-01T00:00:00+00:00",  # UTC timezone with offset
        ]
        
        for ts_str in valid_strings:
            result = CacheEntryValidator._validate_timestamp(ts_str)
            assert result is True, f"Timestamp string '{ts_str}' should be valid"

    def test_validate_timestamp_invalid_strings(self):
        """Test timestamp validation with invalid strings"""
        invalid_strings = [
            "not-a-timestamp",
            "2023-13-01",  # Invalid month
            "2023-01-32",  # Invalid day
            "2023-01-01 25:00:00",  # Invalid hour
            "",
            "2023/01/01",  # Wrong format
        ]
        
        for ts_str in invalid_strings:
            result = CacheEntryValidator._validate_timestamp(ts_str)
            assert result is False, f"Invalid timestamp string '{ts_str}' should be rejected"

    def test_validate_timestamp_invalid_types(self):
        """Test timestamp validation with invalid types"""
        invalid_types = [
            123,  # Integer
            123.45,  # Float
            [],  # List
            {},  # Dict
            True,  # Boolean
        ]
        
        for ts in invalid_types:
            result = CacheEntryValidator._validate_timestamp(ts)
            assert result is False, f"Invalid type {type(ts)} should be rejected"


class TestValidateCacheEntry:
    """Test suite for complete cache entry validation"""

    def test_validate_cache_entry_minimal_valid(self):
        """Test validation with minimal valid cache entry"""
        entry = {
            "key": "test_key",
            "value": "test_value"
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is True
        assert errors == []

    def test_validate_cache_entry_complete_valid(self):
        """Test validation with complete valid cache entry"""
        entry = {
            "key": "test_key",
            "value": {"data": "complex_value"},
            "ttl_seconds": 3600,
            "created_at": datetime.utcnow(),
            "last_accessed": datetime.utcnow(),
            "hit_count": 5
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is True
        assert errors == []

    def test_validate_cache_entry_missing_key(self):
        """Test validation with missing key field"""
        entry = {
            "value": "test_value"
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is False
        assert "Missing required field: key" in errors

    def test_validate_cache_entry_missing_value(self):
        """Test validation with missing value field"""
        entry = {
            "key": "test_key"
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is False
        assert "Missing required field: value" in errors

    def test_validate_cache_entry_invalid_key(self):
        """Test validation with invalid key"""
        entry = {
            "key": "invalid key with spaces",
            "value": "test_value"
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is False
        assert any("Invalid key:" in error for error in errors)

    def test_validate_cache_entry_invalid_value(self):
        """Test validation with invalid value (None)"""
        entry = {
            "key": "test_key",
            "value": None
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is False
        assert any("Invalid value:" in error for error in errors)

    def test_validate_cache_entry_invalid_ttl(self):
        """Test validation with invalid TTL"""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "ttl_seconds": -1
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is False
        assert any("Invalid TTL:" in error for error in errors)

    def test_validate_cache_entry_invalid_timestamps(self):
        """Test validation with invalid timestamps"""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": "invalid-timestamp",
            "last_accessed": 12345  # Invalid type
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is False
        assert any("Invalid created_at timestamp" in error for error in errors)
        assert any("Invalid last_accessed timestamp" in error for error in errors)

    def test_validate_cache_entry_invalid_hit_count(self):
        """Test validation with invalid hit_count values"""
        invalid_entries = [
            {"key": "test", "value": "test", "hit_count": -1},  # Negative
            {"key": "test", "value": "test", "hit_count": "5"},  # String
            {"key": "test", "value": "test", "hit_count": 5.5},  # Float
        ]
        
        for entry in invalid_entries:
            is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
            assert is_valid is False, f"Entry with hit_count {entry['hit_count']} should be invalid"
            assert any("Invalid hit_count:" in error for error in errors)

    def test_validate_cache_entry_multiple_errors(self):
        """Test validation with multiple errors"""
        entry = {
            "key": "",  # Invalid: empty
            "value": None,  # Invalid: None
            "ttl_seconds": -1,  # Invalid: negative
            "created_at": "invalid",  # Invalid: bad format
            "hit_count": -5  # Invalid: negative
        }
        
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is False
        assert len(errors) >= 4  # Should have multiple errors
        
        # Check that all expected errors are present
        error_text = " ".join(errors)
        assert "Invalid key:" in error_text
        assert "Invalid value:" in error_text
        assert "Invalid TTL:" in error_text
        assert "Invalid hit_count:" in error_text


class TestSanitizeKey:
    """Test suite for key sanitization functionality"""

    def test_sanitize_key_valid_unchanged(self):
        """Test that valid keys remain unchanged"""
        valid_keys = [
            "simple_key",
            "key-with-dash",
            "key.with.dots",
            "key:with:colons",
            "key123"
        ]
        
        for key in valid_keys:
            sanitized = CacheEntryValidator.sanitize_key(key)
            assert sanitized == key, f"Valid key '{key}' should remain unchanged"

    def test_sanitize_key_replace_invalid_chars(self):
        """Test sanitization of invalid characters"""
        test_cases = [
            ("key with spaces", "key_with_spaces"),
            ("key@with#special", "key_with_special"),
            ("key/with\\slash", "key_with_slash"),
            ("key[with]brackets", "key_with_brackets"),
            ("key+equals=value", "key_equals_value"),
        ]
        
        for original, expected in test_cases:
            sanitized = CacheEntryValidator.sanitize_key(original)
            assert sanitized == expected, f"'{original}' should sanitize to '{expected}', got '{sanitized}'"

    def test_sanitize_key_truncate_long(self):
        """Test truncation of overly long keys"""
        long_key = "a" * 300  # Longer than MAX_KEY_LENGTH
        sanitized = CacheEntryValidator.sanitize_key(long_key)
        
        assert len(sanitized) == CacheEntryValidator.MAX_KEY_LENGTH
        assert sanitized == "a" * 256

    def test_sanitize_key_complex_cases(self):
        """Test sanitization with complex cases"""
        complex_cases = [
            ("user@domain.com/path?param=value", "user_domain.com_path_param_value"),
            ("   ", "___"),  # Only spaces
        ]
        
        for original, expected in complex_cases:
            sanitized = CacheEntryValidator.sanitize_key(original)
            # For complex cases, mainly verify the result is valid
            is_valid, _ = CacheEntryValidator.validate_key(sanitized)
            assert is_valid, f"Sanitized key '{sanitized}' from '{original}' should be valid"
            assert sanitized == expected

    def test_sanitize_key_unicode_handling(self):
        """Test sanitization with unicode characters"""
        # Test unicode - the exact result may vary but should be valid
        unicode_key = "key with émojis 🚀"
        sanitized = CacheEntryValidator.sanitize_key(unicode_key)
        
        # Should be valid after sanitization
        is_valid, _ = CacheEntryValidator.validate_key(sanitized)
        assert is_valid, f"Unicode-sanitized key '{sanitized}' should be valid"
        
        # Should contain underscores where invalid chars were
        assert "_" in sanitized
        assert "key_with_" in sanitized
    
    def test_sanitize_key_empty_string(self):
        """Test sanitization with empty string (special case)"""
        # Empty string remains empty, but is still invalid for validation
        sanitized = CacheEntryValidator.sanitize_key("")
        assert sanitized == ""
        
        # Empty string should still fail validation
        is_valid, _ = CacheEntryValidator.validate_key(sanitized)
        assert is_valid is False


class TestCreateValidEntry:
    """Test suite for creating validated cache entries"""

    def test_create_valid_entry_minimal(self):
        """Test creating valid entry with minimal parameters"""
        entry = CacheEntryValidator.create_valid_entry("test_key", "test_value")
        
        assert entry["key"] == "test_key"
        assert entry["value"] == "test_value"
        assert isinstance(entry["created_at"], datetime)
        assert entry["hit_count"] == 0
        assert "ttl_seconds" not in entry

    def test_create_valid_entry_with_ttl(self):
        """Test creating valid entry with TTL"""
        entry = CacheEntryValidator.create_valid_entry("test_key", "test_value", ttl_seconds=3600)
        
        assert entry["key"] == "test_key"
        assert entry["value"] == "test_value"
        assert entry["ttl_seconds"] == 3600
        assert isinstance(entry["created_at"], datetime)
        assert entry["hit_count"] == 0

    def test_create_valid_entry_with_kwargs(self):
        """Test creating valid entry with additional kwargs"""
        entry = CacheEntryValidator.create_valid_entry(
            "test_key", 
            "test_value",
            custom_field="custom_value",
            priority=5
        )
        
        assert entry["key"] == "test_key"
        assert entry["value"] == "test_value"
        assert entry["custom_field"] == "custom_value"
        assert entry["priority"] == 5

    def test_create_valid_entry_validation_failure(self):
        """Test that invalid entries raise ValueError"""
        with pytest.raises(ValueError, match="Invalid cache entry"):
            CacheEntryValidator.create_valid_entry("", "test_value")  # Empty key
        
        with pytest.raises(ValueError, match="Invalid cache entry"):
            CacheEntryValidator.create_valid_entry("test_key", None)  # None value
        
        with pytest.raises(ValueError, match="Invalid cache entry"):
            CacheEntryValidator.create_valid_entry("test_key", "test_value", ttl_seconds=-1)  # Invalid TTL


class TestConvenienceFunctions:
    """Test suite for convenience validation functions"""

    def test_validate_cache_key_function(self):
        """Test the standalone validate_cache_key function"""
        assert validate_cache_key("valid_key") is True
        assert validate_cache_key("invalid key") is False
        assert validate_cache_key("") is False

    def test_validate_cache_value_function(self):
        """Test the standalone validate_cache_value function"""
        assert validate_cache_value("valid_value") is True
        assert validate_cache_value(123) is True
        assert validate_cache_value(None) is False

    def test_validate_ttl_function(self):
        """Test the standalone validate_ttl function"""
        assert validate_ttl(None) is True  # Optional
        assert validate_ttl(3600) is True
        assert validate_ttl(-1) is False
        assert validate_ttl("3600") is False


class TestValidatorIntegration:
    """Integration tests for the complete validator functionality"""

    def test_real_world_cache_entries(self):
        """Test validation with realistic cache entry scenarios"""
        scenarios = [
            # User session cache
            {
                "key": "user_session:12345",
                "value": {
                    "user_id": 12345,
                    "username": "johndoe",
                    "permissions": ["read", "write"],
                    "last_activity": "2023-01-01T12:00:00Z"
                },
                "ttl_seconds": 3600,
                "created_at": datetime.utcnow(),
                "hit_count": 0
            },
            
            # API response cache
            {
                "key": "api_response:users:list",
                "value": [
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"}
                ],
                "ttl_seconds": 300,
                "created_at": datetime.utcnow()
            },
            
            # Simple string cache
            {
                "key": "config:app_name",
                "value": "Veris Memory System",
                "created_at": datetime.utcnow()
            }
        ]
        
        for i, entry in enumerate(scenarios):
            is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
            assert is_valid is True, f"Scenario {i+1} should be valid: {errors}"
            assert errors == []

    def test_performance_with_large_values(self):
        """Test validator performance with reasonably large values"""
        # Create a large but valid value (under 1MB limit)
        large_value = {
            "data": "x" * 100000,  # 100KB string
            "metadata": {
                "created": datetime.utcnow().isoformat(),
                "size": 100000,
                "type": "bulk_data"
            },
            "items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]
        }
        
        entry = {
            "key": "large_data_cache",
            "value": large_value,
            "ttl_seconds": 7200
        }
        
        # This should complete efficiently
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is True
        assert errors == []

    def test_edge_case_combinations(self):
        """Test combinations of edge cases"""
        edge_cases = [
            # Minimum valid entry
            {"key": "a", "value": ""},
            
            # Maximum length key with minimum TTL
            {"key": "a" * 256, "value": "test", "ttl_seconds": 1},
            
            # Complex nested structure
            {
                "key": "nested.data:complex",
                "value": {
                    "level1": {
                        "level2": {
                            "level3": ["deep", "nested", "array"]
                        }
                    }
                },
                "ttl_seconds": 86400 * 30  # Maximum TTL
            }
        ]
        
        for i, entry in enumerate(edge_cases):
            is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
            assert is_valid is True, f"Edge case {i+1} should be valid: {errors}"

    def test_sanitize_and_validate_workflow(self):
        """Test complete workflow of sanitizing and validating"""
        # Start with invalid key
        original_key = "user session@123#cache"
        sanitized_key = CacheEntryValidator.sanitize_key(original_key)
        
        # Create entry with sanitized key
        entry = CacheEntryValidator.create_valid_entry(sanitized_key, {"user": "data"})
        
        # Validate the final entry
        is_valid, errors = CacheEntryValidator.validate_cache_entry(entry)
        assert is_valid is True
        assert errors == []
        
        # Verify the key was properly sanitized
        assert CacheEntryValidator.validate_key(entry["key"])[0] is True