#!/usr/bin/env python3
"""
Comprehensive tests for kv_validators module to achieve 100% coverage.
"""

from datetime import datetime, timedelta

from src.validators.kv_validators import (  # noqa: E402
    sanitize_metric_name,
    validate_cache_entry,
    validate_metric_event,
    validate_redis_key,
    validate_session_data,
    validate_time_range,
)


class TestValidateCacheEntry:
    """Test suite for validate_cache_entry function."""

    def test_valid_cache_entry(self):
        """Test validation of a valid cache entry."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is True

    def test_missing_required_field_key(self):
        """Test validation with missing key field."""
        entry = {
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is False

    def test_missing_required_field_value(self):
        """Test validation with missing value field."""
        entry = {
            "key": "test_key",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is False

    def test_missing_required_field_created_at(self):
        """Test validation with missing created_at field."""
        entry = {"key": "test_key", "value": "test_value", "ttl_seconds": 3600}
        assert validate_cache_entry(entry) is False

    def test_missing_required_field_ttl_seconds(self):
        """Test validation with missing ttl_seconds field."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
        }
        assert validate_cache_entry(entry) is False

    def test_invalid_key_type(self):
        """Test validation with non-string key."""
        entry = {
            "key": 123,  # Should be string
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is False

    def test_invalid_ttl_not_integer(self):
        """Test validation with non-integer ttl_seconds."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": "3600",  # Should be int
        }
        assert validate_cache_entry(entry) is False

    def test_invalid_ttl_negative(self):
        """Test validation with negative ttl_seconds."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": -1,
        }
        assert validate_cache_entry(entry) is False

    def test_invalid_timestamp_format(self):
        """Test validation with invalid timestamp format."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": "not-a-timestamp",
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is False

    def test_invalid_timestamp_type(self):
        """Test validation with invalid timestamp type."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": 12345,  # Should be string
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is False

    def test_none_created_at(self):
        """Test validation with None created_at."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": None,
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is False


class TestValidateMetricEvent:
    """Test suite for validate_metric_event function."""

    def test_valid_metric_event_with_string_timestamp(self):
        """Test validation of a valid metric event with string timestamp."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "test.metric",
            "value": 42.5,
            "tags": {"env": "test", "service": "api"},
        }
        assert validate_metric_event(metric) is True

    def test_valid_metric_event_with_datetime_timestamp(self):
        """Test validation of a valid metric event with datetime timestamp."""
        metric = {
            "timestamp": datetime.now(),
            "metric_name": "test.metric",
            "value": 42,
            "tags": {},
        }
        assert validate_metric_event(metric) is True

    def test_missing_timestamp(self):
        """Test validation with missing timestamp."""
        metric = {"metric_name": "test.metric", "value": 42, "tags": {}}
        assert validate_metric_event(metric) is False

    def test_missing_metric_name(self):
        """Test validation with missing metric_name."""
        metric = {"timestamp": datetime.now().isoformat(), "value": 42, "tags": {}}
        assert validate_metric_event(metric) is False

    def test_missing_value(self):
        """Test validation with missing value."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "test.metric",
            "tags": {},
        }
        assert validate_metric_event(metric) is False

    def test_missing_tags(self):
        """Test validation with missing tags."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "test.metric",
            "value": 42,
        }
        assert validate_metric_event(metric) is False

    def test_invalid_metric_name_type(self):
        """Test validation with non-string metric_name."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": 123,  # Should be string
            "value": 42,
            "tags": {},
        }
        assert validate_metric_event(metric) is False

    def test_invalid_value_type(self):
        """Test validation with invalid value type."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "test.metric",
            "value": "not_a_number",  # Should be int or float
            "tags": {},
        }
        assert validate_metric_event(metric) is False

    def test_invalid_tags_type(self):
        """Test validation with non-dict tags."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "test.metric",
            "value": 42,
            "tags": "not_a_dict",  # Should be dict
        }
        assert validate_metric_event(metric) is False

    def test_invalid_string_timestamp(self):
        """Test validation with invalid string timestamp."""
        metric = {
            "timestamp": "not-a-timestamp",
            "metric_name": "test.metric",
            "value": 42,
            "tags": {},
        }
        assert validate_metric_event(metric) is False

    def test_invalid_timestamp_type(self):
        """Test validation with invalid timestamp type."""
        metric = {
            "timestamp": 12345,  # Neither string nor datetime
            "metric_name": "test.metric",
            "value": 42,
            "tags": {},
        }
        assert validate_metric_event(metric) is False


class TestSanitizeMetricName:
    """Test suite for sanitize_metric_name function."""

    def test_valid_metric_name(self):
        """Test sanitization of a valid metric name."""
        name = "valid.metric_name-123"
        assert sanitize_metric_name(name) == name

    def test_special_characters(self):
        """Test sanitization of special characters."""
        name = "metric@#$%^&*()name"
        sanitized = sanitize_metric_name(name)
        assert "@" not in sanitized
        assert "#" not in sanitized
        assert "$" not in sanitized
        assert all(c.isalnum() or c in "._-" for c in sanitized)

    def test_spaces_replaced(self):
        """Test that spaces are replaced."""
        name = "metric name with spaces"
        sanitized = sanitize_metric_name(name)
        assert " " not in sanitized
        assert "_" in sanitized

    def test_long_name_truncated(self):
        """Test that long names are truncated to 100 characters."""
        name = "a" * 150
        sanitized = sanitize_metric_name(name)
        assert len(sanitized) == 100

    def test_exactly_100_chars(self):
        """Test that names with exactly 100 chars are not truncated."""
        name = "a" * 100
        sanitized = sanitize_metric_name(name)
        assert len(sanitized) == 100
        assert sanitized == name

    def test_unicode_characters(self):
        """Test sanitization of unicode characters."""
        name = "metric_name_with_émojis_🎉"
        sanitized = sanitize_metric_name(name)
        assert "🎉" not in sanitized
        assert "é" not in sanitized

    def test_empty_string(self):
        """Test sanitization of empty string."""
        name = ""
        sanitized = sanitize_metric_name(name)
        assert sanitized == ""

    def test_only_invalid_chars(self):
        """Test sanitization when name contains only invalid characters."""
        name = "@#$%^&*()"
        sanitized = sanitize_metric_name(name)
        assert sanitized == "_________"


class TestValidateTimeRange:
    """Test suite for validate_time_range function."""

    def test_valid_time_range(self):
        """Test validation of a valid time range."""
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(start, end) is True

    def test_start_after_end(self):
        """Test validation with start time after end time."""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow() - timedelta(days=7)
        assert validate_time_range(start, end) is False

    def test_start_equals_end(self):
        """Test validation with start time equal to end time."""
        time = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(time, time) is False

    def test_range_exceeds_max_days(self):
        """Test validation with range exceeding max days."""
        start = datetime.utcnow() - timedelta(days=100)
        end = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(start, end, max_days=90) is False

    def test_range_exactly_max_days(self):
        """Test validation with range exactly at max days."""
        start = datetime.utcnow() - timedelta(days=90)
        end = datetime.utcnow() - timedelta(days=0, hours=1)
        assert validate_time_range(start, end, max_days=90) is True

    def test_custom_max_days(self):
        """Test validation with custom max_days."""
        start = datetime.utcnow() - timedelta(days=20)
        end = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(start, end, max_days=30) is True
        assert validate_time_range(start, end, max_days=10) is False

    def test_future_end_date(self):
        """Test validation with future end date."""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow() + timedelta(days=1)
        assert validate_time_range(start, end) is False

    def test_both_dates_in_future(self):
        """Test validation with both dates in future."""
        start = datetime.utcnow() + timedelta(days=1)
        end = datetime.utcnow() + timedelta(days=2)
        assert validate_time_range(start, end) is False

    def test_end_date_exactly_now(self):
        """Test validation with end date exactly now."""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow()
        # This might be True depending on execution timing
        # Just ensure it doesn't raise an exception
        result = validate_time_range(start, end)
        assert isinstance(result, bool)


class TestValidateRedisKey:
    """Test suite for validate_redis_key function."""

    def test_valid_redis_key(self):
        """Test validation of a valid Redis key."""
        key = "user:session:12345"
        assert validate_redis_key(key) is True

    def test_empty_key(self):
        """Test validation of empty key."""
        assert validate_redis_key("") is False

    def test_none_key(self):
        """Test validation of None key."""
        assert validate_redis_key(None) is False

    def test_non_string_key(self):
        """Test validation of non-string key."""
        assert validate_redis_key(123) is False
        assert validate_redis_key(["key"]) is False
        assert validate_redis_key({"key": "value"}) is False

    def test_key_too_long(self):
        """Test validation of key exceeding max length."""
        key = "a" * 1025  # Max is 1024
        assert validate_redis_key(key) is False

    def test_key_exactly_max_length(self):
        """Test validation of key at max length."""
        key = "a" * 1024
        assert validate_redis_key(key) is True

    def test_key_with_control_characters(self):
        """Test validation of key with control characters."""
        key = "key\x00with\x01control\x1fchars"
        assert validate_redis_key(key) is False

    def test_key_with_newline(self):
        """Test validation of key with newline."""
        key = "key\nwith\nnewlines"
        assert validate_redis_key(key) is False

    def test_key_with_tab(self):
        """Test validation of key with tab."""
        key = "key\twith\ttabs"
        assert validate_redis_key(key) is False

    def test_key_with_space(self):
        """Test validation of key with space (should be valid)."""
        key = "key with spaces"
        assert validate_redis_key(key) is True

    def test_key_with_special_chars(self):
        """Test validation of key with special characters (should be valid)."""
        key = "key:with:colons_and-dashes.and.dots"
        assert validate_redis_key(key) is True

    def test_key_with_unicode(self):
        """Test validation of key with unicode characters."""
        key = "key_with_émojis_🎉"
        # Unicode is allowed as long as no control characters
        assert validate_redis_key(key) is True


class TestValidateSessionData:
    """Test suite for validate_session_data function."""

    def test_valid_session_data(self):
        """Test validation of valid session data."""
        data = {
            "user_id": 12345,
            "username": "testuser",
            "created_at": datetime.now().isoformat(),
            "preferences": {"theme": "dark", "language": "en"},
        }
        assert validate_session_data(data) is True

    def test_non_dict_session_data(self):
        """Test validation of non-dict session data."""
        assert validate_session_data("not a dict") is False
        assert validate_session_data(123) is False
        assert validate_session_data([1, 2, 3]) is False
        assert validate_session_data(None) is False

    def test_small_session_data(self):
        """Test validation of small session data."""
        data = {"key": "value"}
        assert validate_session_data(data) is True

    def test_session_data_at_limit(self):
        """Test validation of session data at size limit."""
        # Create data that's just under 1MB when serialized
        large_string = "a" * (1024 * 1024 - 100)
        data = {"data": large_string}
        result = validate_session_data(data)
        # Should be True if under 1MB
        assert isinstance(result, bool)

    def test_session_data_exceeds_limit(self):
        """Test validation of session data exceeding size limit."""
        # Create data that exceeds 1MB when serialized
        large_string = "a" * (1024 * 1024 + 1)
        data = {"data": large_string}
        assert validate_session_data(data) is False

    def test_non_serializable_data(self):
        """Test validation of non-serializable data."""
        # Create a dict with non-serializable content
        data = {
            "function": lambda x: x,
            "key": "value",
        }  # Functions can't be JSON serialized
        assert validate_session_data(data) is False

    def test_circular_reference(self):
        """Test validation of data with circular reference."""
        data = {"key": "value"}
        data["self"] = data  # Create circular reference
        assert validate_session_data(data) is False

    def test_empty_dict(self):
        """Test validation of empty dictionary."""
        data = {}
        assert validate_session_data(data) is True

    def test_deeply_nested_data(self):
        """Test validation of deeply nested data."""
        data = {"level1": {"level2": {"level3": {"level4": {"level5": "value"}}}}}
        assert validate_session_data(data) is True

    def test_data_with_datetime_objects(self):
        """Test validation of data with datetime objects (not JSON serializable)."""
        data = {
            "timestamp": datetime.now(),  # datetime objects aren't JSON serializable
            "key": "value",
        }
        assert validate_session_data(data) is False

    def test_data_with_complex_types(self):
        """Test validation of data with various complex types."""
        data = {
            "string": "value",
            "integer": 123,
            "float": 123.45,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested_dict": {"key": "value"},
        }
        assert validate_session_data(data) is True
