#!/usr/bin/env python3
"""
Comprehensive tests for kv_validators.py to achieve high coverage.

This test suite covers:
- Cache entry validation with all scenarios
- Metric event validation and edge cases
- Metric name sanitization
- Time range validation
- Redis key validation
- Session data validation
- Error handling and boundary conditions
"""

from datetime import datetime, timedelta

import pytest

from src.validators.kv_validators import (
    sanitize_metric_name,
    validate_cache_entry,
    validate_metric_event,
    validate_redis_key,
    validate_session_data,
    validate_time_range,
)


class TestValidateCacheEntry:
    """Test validate_cache_entry function."""

    @pytest.fixture
    def valid_cache_entry(self):
        """Valid cache entry for testing."""
        return {
            "key": "test_key",
            "value": {"data": "test_value"},
            "created_at": "2024-01-01T10:00:00",
            "ttl_seconds": 3600,
        }

    def test_validate_cache_entry_success(self, valid_cache_entry):
        """Test successful cache entry validation."""
        assert validate_cache_entry(valid_cache_entry) is True

    def test_validate_cache_entry_non_dict(self):
        """Test cache entry validation fails for non-dict."""
        invalid_entries = ["not a dict", 123, ["list", "instead"], None, True]

        for entry in invalid_entries:
            assert validate_cache_entry(entry) is False

    def test_validate_cache_entry_missing_required_fields(self, valid_cache_entry):
        """Test cache entry validation fails for missing required fields."""
        required_fields = ["key", "value", "created_at", "ttl_seconds"]

        for field in required_fields:
            incomplete_entry = valid_cache_entry.copy()
            del incomplete_entry[field]

            assert validate_cache_entry(incomplete_entry) is False

    def test_validate_cache_entry_invalid_key_type(self, valid_cache_entry):
        """Test cache entry validation fails for invalid key type."""
        invalid_keys = [123, None, [], {}, True]

        for key in invalid_keys:
            entry = valid_cache_entry.copy()
            entry["key"] = key

            assert validate_cache_entry(entry) is False

    def test_validate_cache_entry_invalid_ttl_type(self, valid_cache_entry):
        """Test cache entry validation fails for invalid TTL type."""
        invalid_ttls = ["not_int", None, [], {}, 3.14]  # Remove True as it's valid

        for ttl in invalid_ttls:
            entry = valid_cache_entry.copy()
            entry["ttl_seconds"] = ttl

            assert validate_cache_entry(entry) is False

    def test_validate_cache_entry_boolean_ttl(self, valid_cache_entry):
        """Test cache entry validation with boolean TTL."""
        # True is treated as 1 in Python, False as 0
        entry_true = valid_cache_entry.copy()
        entry_true["ttl_seconds"] = True
        assert validate_cache_entry(entry_true) is True  # True is valid (treated as 1)

        entry_false = valid_cache_entry.copy()
        entry_false["ttl_seconds"] = False
        assert validate_cache_entry(entry_false) is True  # False is valid (treated as 0)

    def test_validate_cache_entry_negative_ttl(self, valid_cache_entry):
        """Test cache entry validation fails for negative TTL."""
        entry = valid_cache_entry.copy()
        entry["ttl_seconds"] = -1

        assert validate_cache_entry(entry) is False

    def test_validate_cache_entry_zero_ttl(self, valid_cache_entry):
        """Test cache entry validation passes for zero TTL."""
        entry = valid_cache_entry.copy()
        entry["ttl_seconds"] = 0

        assert validate_cache_entry(entry) is True

    def test_validate_cache_entry_invalid_timestamp_format(self, valid_cache_entry):
        """Test cache entry validation fails for invalid timestamp format."""
        invalid_timestamps = [
            "not-a-timestamp",
            "2024-13-01T10:00:00",  # Invalid month
            "2024/01/01 10:00:00",  # Wrong format
            123456789,  # Integer
            None,
            [],
        ]

        for timestamp in invalid_timestamps:
            entry = valid_cache_entry.copy()
            entry["created_at"] = timestamp

            assert validate_cache_entry(entry) is False

    def test_validate_cache_entry_valid_timestamp_formats(self, valid_cache_entry):
        """Test cache entry validation passes for valid timestamp formats."""
        valid_timestamps = [
            "2024-01-01T10:00:00",
            "2024-01-01T10:00:00.123456",
            "2024-01-01T10:00:00+00:00",
            "2024-01-01T10:00:00.123456+05:30",
        ]

        for timestamp in valid_timestamps:
            entry = valid_cache_entry.copy()
            entry["created_at"] = timestamp

            assert validate_cache_entry(entry) is True

    def test_validate_cache_entry_value_can_be_any_type(self, valid_cache_entry):
        """Test cache entry validation allows any value type."""
        valid_values = [
            "string_value",
            123,
            3.14,
            {"nested": "dict"},
            ["list", "of", "items"],
            None,
            True,
            False,
        ]

        for value in valid_values:
            entry = valid_cache_entry.copy()
            entry["value"] = value

            assert validate_cache_entry(entry) is True


class TestValidateMetricEvent:
    """Test validate_metric_event function."""

    @pytest.fixture
    def valid_metric_event(self):
        """Valid metric event for testing."""
        return {
            "timestamp": "2024-01-01T10:00:00",
            "metric_name": "api_requests_total",
            "value": 42,
            "tags": {"service": "api", "endpoint": "/users"},
        }

    def test_validate_metric_event_success(self, valid_metric_event):
        """Test successful metric event validation."""
        assert validate_metric_event(valid_metric_event) is True

    def test_validate_metric_event_non_dict(self):
        """Test metric event validation fails for non-dict."""
        invalid_events = ["not a dict", 123, ["list", "instead"], None, True]

        for event in invalid_events:
            assert validate_metric_event(event) is False

    def test_validate_metric_event_missing_required_fields(self, valid_metric_event):
        """Test metric event validation fails for missing required fields."""
        required_fields = ["timestamp", "metric_name", "value", "tags"]

        for field in required_fields:
            incomplete_event = valid_metric_event.copy()
            del incomplete_event[field]

            assert validate_metric_event(incomplete_event) is False

    def test_validate_metric_event_invalid_metric_name_type(self, valid_metric_event):
        """Test metric event validation fails for invalid metric name type."""
        invalid_names = [123, None, [], {}, True]

        for name in invalid_names:
            event = valid_metric_event.copy()
            event["metric_name"] = name

            assert validate_metric_event(event) is False

    def test_validate_metric_event_invalid_value_type(self, valid_metric_event):
        """Test metric event validation fails for invalid value type."""
        invalid_values = ["not_number", None, [], {}]  # Remove True as it's valid

        for value in invalid_values:
            event = valid_metric_event.copy()
            event["value"] = value

            assert validate_metric_event(event) is False

    def test_validate_metric_event_boolean_value(self, valid_metric_event):
        """Test metric event validation with boolean value."""
        # True and False are valid numeric values in Python
        event_true = valid_metric_event.copy()
        event_true["value"] = True
        assert validate_metric_event(event_true) is True  # True is valid (treated as 1)

        event_false = valid_metric_event.copy()
        event_false["value"] = False
        assert validate_metric_event(event_false) is True  # False is valid (treated as 0)

    def test_validate_metric_event_valid_numeric_values(self, valid_metric_event):
        """Test metric event validation passes for valid numeric values."""
        valid_values = [42, -10, 0, 3.14, -2.5, 0.0]

        for value in valid_values:
            event = valid_metric_event.copy()
            event["value"] = value

            assert validate_metric_event(event) is True

    def test_validate_metric_event_invalid_tags_type(self, valid_metric_event):
        """Test metric event validation fails for invalid tags type."""
        invalid_tags = ["not_dict", 123, ["list"], None, True]

        for tags in invalid_tags:
            event = valid_metric_event.copy()
            event["tags"] = tags

            assert validate_metric_event(event) is False

    def test_validate_metric_event_empty_tags_dict(self, valid_metric_event):
        """Test metric event validation passes for empty tags dict."""
        event = valid_metric_event.copy()
        event["tags"] = {}

        assert validate_metric_event(event) is True

    def test_validate_metric_event_string_timestamp(self, valid_metric_event):
        """Test metric event validation with string timestamp."""
        valid_timestamps = [
            "2024-01-01T10:00:00",
            "2024-01-01T10:00:00.123456",
            "2024-01-01T10:00:00+00:00",
        ]

        for timestamp in valid_timestamps:
            event = valid_metric_event.copy()
            event["timestamp"] = timestamp

            assert validate_metric_event(event) is True

    def test_validate_metric_event_datetime_timestamp(self, valid_metric_event):
        """Test metric event validation with datetime timestamp."""
        event = valid_metric_event.copy()
        event["timestamp"] = datetime(2024, 1, 1, 10, 0, 0)

        assert validate_metric_event(event) is True

    def test_validate_metric_event_invalid_string_timestamp(self, valid_metric_event):
        """Test metric event validation fails for invalid string timestamp."""
        invalid_timestamps = [
            "not-a-timestamp",
            "2024-13-01T10:00:00",  # Invalid month
            "2024/01/01 10:00:00",  # Wrong format
        ]

        for timestamp in invalid_timestamps:
            event = valid_metric_event.copy()
            event["timestamp"] = timestamp

            assert validate_metric_event(event) is False

    def test_validate_metric_event_invalid_timestamp_type(self, valid_metric_event):
        """Test metric event validation fails for invalid timestamp type."""
        invalid_timestamps = [123456789, None, [], {}, True]

        for timestamp in invalid_timestamps:
            event = valid_metric_event.copy()
            event["timestamp"] = timestamp

            assert validate_metric_event(event) is False


class TestSanitizeMetricName:
    """Test sanitize_metric_name function."""

    def test_sanitize_metric_name_clean(self):
        """Test sanitizing already clean metric names."""
        clean_names = [
            "api_requests_total",
            "cpu.usage.percent",
            "memory-used-bytes",
            "disk_io_read_ops",
            "network.bytes.received",
        ]

        for name in clean_names:
            assert sanitize_metric_name(name) == name

    def test_sanitize_metric_name_with_spaces(self):
        """Test sanitizing metric names with spaces."""
        assert sanitize_metric_name("api requests total") == "api_requests_total"
        assert sanitize_metric_name("  spaced  name  ") == "__spaced__name__"

    def test_sanitize_metric_name_with_special_chars(self):
        """Test sanitizing metric names with special characters."""
        test_cases = [
            ("api@requests#total", "api_requests_total"),
            ("cpu%usage&percent", "cpu_usage_percent"),
            ("memory(used)bytes", "memory_used_bytes"),
            ("network/bytes\\received", "network_bytes_received"),
        ]

        for input_name, expected in test_cases:
            assert sanitize_metric_name(input_name) == expected

    def test_sanitize_metric_name_preserve_allowed_chars(self):
        """Test that allowed characters are preserved."""
        name_with_allowed = "api.requests_total-v2"
        assert sanitize_metric_name(name_with_allowed) == name_with_allowed

    def test_sanitize_metric_name_long_name(self):
        """Test sanitizing very long metric names."""
        long_name = "a" * 150
        sanitized = sanitize_metric_name(long_name)

        assert len(sanitized) == 100
        assert sanitized == "a" * 100

    def test_sanitize_metric_name_exactly_100_chars(self):
        """Test sanitizing metric name with exactly 100 characters."""
        name_100 = "a" * 100
        sanitized = sanitize_metric_name(name_100)

        assert len(sanitized) == 100
        assert sanitized == name_100

    def test_sanitize_metric_name_empty_string(self):
        """Test sanitizing empty string."""
        assert sanitize_metric_name("") == ""

    def test_sanitize_metric_name_only_special_chars(self):
        """Test sanitizing string with only special characters."""
        assert sanitize_metric_name("@#$%^&*()") == "_________"  # 9 characters, not 10

    def test_sanitize_metric_name_unicode_chars(self):
        """Test sanitizing metric names with unicode characters."""
        unicode_names = [
            ("api_requests_café", "api_requests_caf_"),
            ("metric_名前_test", "metric____test"),
            ("测试_metric", "___metric"),
        ]

        for input_name, expected in unicode_names:
            assert sanitize_metric_name(input_name) == expected


class TestValidateTimeRange:
    """Test validate_time_range function."""

    def test_validate_time_range_success(self):
        """Test successful time range validation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 2, 10, 0, 0)

        assert validate_time_range(start, end) is True

    def test_validate_time_range_same_time(self):
        """Test time range validation fails when start equals end."""
        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        assert validate_time_range(timestamp, timestamp) is False

    def test_validate_time_range_start_after_end(self):
        """Test time range validation fails when start is after end."""
        start = datetime(2024, 1, 2, 10, 0, 0)
        end = datetime(2024, 1, 1, 10, 0, 0)

        assert validate_time_range(start, end) is False

    def test_validate_time_range_exceeds_max_days_default(self):
        """Test time range validation fails when exceeding default max days."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 4, 1, 10, 0, 0)  # More than 90 days

        assert validate_time_range(start, end) is False

    def test_validate_time_range_exactly_max_days_default(self):
        """Test time range validation at exactly default max days."""
        # Use past dates to avoid future date validation issues
        start = datetime(2023, 1, 1, 10, 0, 0)

        # Calculate exact 90 days - should PASS (delta.days == max_days is allowed)
        end_90_days = start + timedelta(days=90)
        assert validate_time_range(start, end_90_days) is True  # Exactly 90 days should pass

        # Calculate 91 days - should FAIL (delta.days > max_days)
        end_91_days = start + timedelta(days=91)
        assert validate_time_range(start, end_91_days) is False  # 91 days should fail

    def test_validate_time_range_custom_max_days(self):
        """Test time range validation with custom max days."""
        # Use past dates to avoid future date validation issues
        start = datetime(2023, 1, 1, 10, 0, 0)
        end_7_days = start + timedelta(days=7)  # Exactly 7 days
        end_8_days = start + timedelta(days=8)  # 8 days

        assert (
            validate_time_range(start, end_7_days, max_days=7) is True
        )  # Exactly 7 days should pass
        assert validate_time_range(start, end_8_days, max_days=7) is False  # 8 days should fail
        assert (
            validate_time_range(start, end_7_days, max_days=8) is True
        )  # 7 days with max 8 should pass

    def test_validate_time_range_future_end_time(self):
        """Test time range validation fails for future end time."""
        now = datetime.utcnow()
        start = now - timedelta(hours=1)
        future_end = now + timedelta(hours=1)

        assert validate_time_range(start, future_end) is False

    def test_validate_time_range_end_time_now(self):
        """Test time range validation with end time as current time."""
        now = datetime.utcnow()
        start = now - timedelta(hours=1)

        # This might be flaky if there's a timing issue, but should generally pass
        # since we allow end_time <= now
        assert validate_time_range(start, now) is True

    def test_validate_time_range_very_recent(self):
        """Test time range validation with very recent times."""
        now = datetime.utcnow()
        start = now - timedelta(seconds=30)
        end = now - timedelta(seconds=10)

        assert validate_time_range(start, end) is True


class TestValidateRedisKey:
    """Test validate_redis_key function."""

    def test_validate_redis_key_success(self):
        """Test successful Redis key validation."""
        valid_keys = [
            "simple_key",
            "user:123:profile",
            "cache/api/users/123",
            "session-abc-def-123",
            "metric.cpu.usage",
            "a" * 1024,  # Maximum length
        ]

        for key in valid_keys:
            assert validate_redis_key(key) is True

    def test_validate_redis_key_empty_or_none(self):
        """Test Redis key validation fails for empty or None keys."""
        invalid_keys = [None, "", " "]  # Note: single space is valid

        for key in invalid_keys[:-1]:  # Exclude single space
            assert validate_redis_key(key) is False

        # Single space should be valid
        assert validate_redis_key(" ") is True

    def test_validate_redis_key_non_string(self):
        """Test Redis key validation fails for non-string types."""
        invalid_keys = [123, [], {}, True, 3.14]

        for key in invalid_keys:
            assert validate_redis_key(key) is False

    def test_validate_redis_key_too_long(self):
        """Test Redis key validation fails for keys that are too long."""
        too_long_key = "a" * 1025  # Over the 1024 limit

        assert validate_redis_key(too_long_key) is False

    def test_validate_redis_key_control_characters(self):
        """Test Redis key validation fails for keys with control characters."""
        keys_with_control_chars = [
            "key\x00with\x01null",  # Null and control chars
            "key\nwith\nnewlines",  # Newlines
            "key\twith\ttabs",  # Tabs
            "key\rwith\rcarriage",  # Carriage returns
            "key\x1fwith\x1funit",  # Unit separator
        ]

        for key in keys_with_control_chars:
            assert validate_redis_key(key) is False

    def test_validate_redis_key_printable_chars(self):
        """Test Redis key validation passes for printable characters."""
        printable_keys = [
            "key with spaces",
            "key!@#$%^&*()_+-=[]{}|;:,.<>?",
            "key~`",
            "key'\"",
            "key/\\",
        ]

        for key in printable_keys:
            assert validate_redis_key(key) is True

    def test_validate_redis_key_unicode(self):
        """Test Redis key validation with unicode characters."""
        unicode_keys = [
            "key_with_café",
            "键_with_chinese",
            "🔑_with_emoji",
            "key_with_数字_123",
        ]

        for key in unicode_keys:
            assert validate_redis_key(key) is True


class TestValidateSessionData:
    """Test validate_session_data function."""

    def test_validate_session_data_success(self):
        """Test successful session data validation."""
        valid_data = {
            "user_id": "123",
            "username": "testuser",
            "login_time": "2024-01-01T10:00:00",
            "preferences": {"theme": "dark", "language": "en"},
            "cart": ["item1", "item2"],
        }

        assert validate_session_data(valid_data) is True

    def test_validate_session_data_non_dict(self):
        """Test session data validation fails for non-dict."""
        invalid_data = ["not a dict", 123, ["list", "instead"], None, True]

        for data in invalid_data:
            assert validate_session_data(data) is False

    def test_validate_session_data_empty_dict(self):
        """Test session data validation passes for empty dict."""
        assert validate_session_data({}) is True

    def test_validate_session_data_too_large(self):
        """Test session data validation fails for data that's too large."""
        # Create data that exceeds 1MB when serialized
        large_data = {"large_field": "x" * (1024 * 1024 + 1)}

        assert validate_session_data(large_data) is False

    def test_validate_session_data_exactly_1mb(self):
        """Test session data validation at exactly 1MB limit."""
        # Create data that's close to but under 1MB
        # Account for JSON overhead (quotes, braces, etc.)
        field_size = 1024 * 1024 - 100  # Leave room for JSON overhead
        data_near_limit = {"field": "x" * field_size}

        # This should pass (under 1MB)
        assert validate_session_data(data_near_limit) is True

    def test_validate_session_data_non_serializable(self):
        """Test session data validation fails for non-JSON-serializable data."""

        class NonSerializable:
            pass

        non_serializable_data = [
            {"object": NonSerializable()},
            {"function": lambda x: x},
            {"set": {1, 2, 3}},  # Sets are not JSON serializable
        ]

        for data in non_serializable_data:
            assert validate_session_data(data) is False

    def test_validate_session_data_complex_nested(self):
        """Test session data validation with complex nested structures."""
        complex_data = {
            "user": {
                "id": 123,
                "profile": {
                    "name": "Test User",
                    "settings": {"notifications": {"email": True, "sms": False, "push": True}},
                },
            },
            "session": {
                "created": "2024-01-01T10:00:00",
                "last_activity": "2024-01-01T10:05:00",
                "data": {
                    "cart": [
                        {"id": 1, "name": "Item 1", "price": 10.99},
                        {"id": 2, "name": "Item 2", "price": 25.50},
                    ]
                },
            },
        }

        assert validate_session_data(complex_data) is True

    def test_validate_session_data_with_various_types(self):
        """Test session data validation with various JSON-compatible types."""
        data_with_types = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean_true": True,
            "boolean_false": False,
            "null_value": None,
            "list": [1, 2, 3, "four"],
            "nested_dict": {"key": "value"},
        }

        assert validate_session_data(data_with_types) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
