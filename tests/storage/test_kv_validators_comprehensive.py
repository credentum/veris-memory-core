#!/usr/bin/env python3
"""
Comprehensive tests for src/validators/kv_validators.py
Critical domain tests to boost coverage above 78.5% threshold
"""

import json
from datetime import datetime, timedelta
from typing import Any

from src.validators.kv_validators import (  # noqa: E402
    sanitize_metric_name,
    validate_cache_entry,
    validate_metric_event,
    validate_redis_key,
    validate_session_data,
    validate_time_range,
)


class TestValidateCacheEntry:
    """Test cache entry validation for KV store operations"""

    def test_valid_cache_entry(self) -> None:
        """Test valid cache entry structure"""
        valid_entry = {
            "key": "test_key",
            "value": {"data": "test"},
            "created_at": "2025-07-16T10:00:00",
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(valid_entry) is True

    def test_missing_required_fields(self) -> None:
        """Test validation fails when required fields are missing"""
        required_fields = ["key", "value", "created_at", "ttl_seconds"]

        for missing_field in required_fields:
            entry = {
                "key": "test_key",
                "value": "test_value",
                "created_at": "2025-07-16T10:00:00",
                "ttl_seconds": 3600,
            }
            del entry[missing_field]
            assert validate_cache_entry(entry) is False

    def test_invalid_key_type(self) -> None:
        """Test validation fails for non-string keys"""
        invalid_entries: list[dict[str, Any]] = [
            {
                "key": 123,
                "value": "test",
                "created_at": "2025-07-16T10:00:00",
                "ttl_seconds": 3600,
            },
            {
                "key": None,
                "value": "test",
                "created_at": "2025-07-16T10:00:00",
                "ttl_seconds": 3600,
            },
            {
                "key": [],
                "value": "test",
                "created_at": "2025-07-16T10:00:00",
                "ttl_seconds": 3600,
            },
        ]

        for entry in invalid_entries:
            assert validate_cache_entry(entry) is False

    def test_invalid_ttl_values(self) -> None:
        """Test validation fails for invalid TTL values"""
        invalid_ttls = [-1, "3600", None, 3.14, []]

        for ttl in invalid_ttls:
            entry = {
                "key": "test_key",
                "value": "test_value",
                "created_at": "2025-07-16T10:00:00",
                "ttl_seconds": ttl,
            }
            assert validate_cache_entry(entry) is False

    def test_valid_zero_ttl(self) -> None:
        """Test that TTL of 0 is valid"""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": "2025-07-16T10:00:00",
            "ttl_seconds": 0,
        }
        assert validate_cache_entry(entry) is True

    def test_invalid_timestamp_formats(self) -> None:
        """Test validation fails for invalid timestamp formats"""
        invalid_timestamps = [
            "invalid-date",
            "2025-13-01T10:00:00",  # Invalid month
            "2025-07-32T10:00:00",  # Invalid day
            123456789,
            None,
            [],
        ]

        for timestamp in invalid_timestamps:
            entry = {
                "key": "test_key",
                "value": "test_value",
                "created_at": timestamp,
                "ttl_seconds": 3600,
            }
            assert validate_cache_entry(entry) is False

    def test_valid_timestamp_formats(self) -> None:
        """Test various valid ISO timestamp formats"""
        valid_timestamps = [
            "2025-07-16T10:00:00",
            "2025-07-16T10:00:00.123",
            "2025-07-16T10:00:00+00:00",
            "2025-07-16T10:00:00.123456",
        ]

        for timestamp in valid_timestamps:
            entry = {
                "key": "test_key",
                "value": "test_value",
                "created_at": timestamp,
                "ttl_seconds": 3600,
            }
            assert validate_cache_entry(entry) is True


class TestValidateMetricEvent:
    """Test metric event validation"""

    def test_valid_metric_event_string_timestamp(self) -> None:
        """Test valid metric event with string timestamp"""
        valid_metric = {
            "timestamp": "2025-07-16T10:00:00",
            "metric_name": "cpu_usage",
            "value": 75.5,
            "tags": {"host": "server1", "env": "prod"},
        }
        assert validate_metric_event(valid_metric) is True

    def test_valid_metric_event_datetime_timestamp(self) -> None:
        """Test valid metric event with datetime timestamp"""
        valid_metric = {
            "timestamp": datetime.now(),
            "metric_name": "memory_usage",
            "value": 80,
            "tags": {"service": "api"},
        }
        assert validate_metric_event(valid_metric) is True

    def test_missing_required_fields(self) -> None:
        """Test validation fails when required fields are missing"""
        required_fields = ["timestamp", "metric_name", "value", "tags"]

        for missing_field in required_fields:
            metric = {
                "timestamp": "2025-07-16T10:00:00",
                "metric_name": "test_metric",
                "value": 100,
                "tags": {"env": "test"},
            }
            del metric[missing_field]
            assert validate_metric_event(metric) is False

    def test_invalid_metric_name_types(self) -> None:
        """Test validation fails for non-string metric names"""
        invalid_names: list[Any] = [123, None, [], {"name": "test"}]

        for name in invalid_names:
            metric = {
                "timestamp": "2025-07-16T10:00:00",
                "metric_name": name,
                "value": 100,
                "tags": {"env": "test"},
            }
            assert validate_metric_event(metric) is False

    def test_invalid_value_types(self) -> None:
        """Test validation fails for invalid value types"""
        invalid_values: list[Any] = ["100", None, [], {"value": 100}]

        for value in invalid_values:
            metric = {
                "timestamp": "2025-07-16T10:00:00",
                "metric_name": "test_metric",
                "value": value,
                "tags": {"env": "test"},
            }
            assert validate_metric_event(metric) is False

    def test_valid_numeric_values(self) -> None:
        """Test various valid numeric value types"""
        valid_values = [100, 75.5, 0, -50, 3.14159]

        for value in valid_values:
            metric = {
                "timestamp": "2025-07-16T10:00:00",
                "metric_name": "test_metric",
                "value": value,
                "tags": {"env": "test"},
            }
            assert validate_metric_event(metric) is True

    def test_invalid_tags_types(self) -> None:
        """Test validation fails for non-dict tags"""
        invalid_tags = ["tags", 123, None, []]

        for tags in invalid_tags:
            metric = {
                "timestamp": "2025-07-16T10:00:00",
                "metric_name": "test_metric",
                "value": 100,
                "tags": tags,
            }
            assert validate_metric_event(metric) is False

    def test_invalid_timestamp_string(self) -> None:
        """Test validation fails for invalid timestamp strings"""
        invalid_timestamps = [
            "invalid-date",
            "2025-13-01T10:00:00",
            "not-a-date",
        ]

        for timestamp in invalid_timestamps:
            metric = {
                "timestamp": timestamp,
                "metric_name": "test_metric",
                "value": 100,
                "tags": {"env": "test"},
            }
            assert validate_metric_event(metric) is False

    def test_invalid_timestamp_types(self) -> None:
        """Test validation fails for invalid timestamp types"""
        invalid_timestamps: list[Any] = [123, None, []]

        for timestamp in invalid_timestamps:
            metric = {
                "timestamp": timestamp,
                "metric_name": "test_metric",
                "value": 100,
                "tags": {"env": "test"},
            }
            assert validate_metric_event(metric) is False


class TestSanitizeMetricName:
    """Test metric name sanitization"""

    def test_valid_metric_names(self) -> None:
        """Test that valid metric names are preserved"""
        valid_names = [
            "cpu_usage",
            "memory.free",
            "network-latency",
            "disk_io_123",
            "api.requests.per_second",
        ]

        for name in valid_names:
            assert sanitize_metric_name(name) == name

    def test_invalid_characters_replaced(self) -> None:
        """Test that invalid characters are replaced with underscores"""
        test_cases = [
            ("cpu usage", "cpu_usage"),
            ("memory@free", "memory_free"),
            ("network/latency", "network_latency"),
            ("disk#io", "disk_io"),
            ("api$requests", "api_requests"),
            ("metric!name", "metric_name"),
        ]

        for input_name, expected in test_cases:
            assert sanitize_metric_name(input_name) == expected

    def test_multiple_invalid_characters(self) -> None:
        """Test handling of multiple invalid characters"""
        input_name = "cpu@#$%^&*()usage"
        expected = "cpu_________usage"
        assert sanitize_metric_name(input_name) == expected

    def test_length_limitation(self) -> None:
        """Test that long metric names are truncated"""
        long_name = "a" * 150  # 150 characters
        sanitized = sanitize_metric_name(long_name)
        assert len(sanitized) == 100
        assert sanitized == "a" * 100

    def test_unicode_characters(self) -> None:
        """Test handling of unicode characters"""
        unicode_name = "cpu_usage_测试"
        sanitized = sanitize_metric_name(unicode_name)
        assert "测试" not in sanitized  # Unicode should be replaced
        assert sanitized == "cpu_usage___"

    def test_empty_string(self) -> None:
        """Test handling of empty string"""
        assert sanitize_metric_name("") == ""

    def test_only_invalid_characters(self) -> None:
        """Test string with only invalid characters"""
        assert sanitize_metric_name("@#$%^") == "_____"


class TestValidateTimeRange:
    """Test time range validation"""

    def test_valid_time_range(self) -> None:
        """Test valid time range within limits"""
        start = datetime(2025, 7, 1, 10, 0, 0)
        end = datetime(2025, 7, 2, 10, 0, 0)
        assert validate_time_range(start, end) is True

    def test_start_after_end(self) -> None:
        """Test validation fails when start time is after end time"""
        start = datetime(2025, 7, 2, 10, 0, 0)
        end = datetime(2025, 7, 1, 10, 0, 0)
        assert validate_time_range(start, end) is False

    def test_start_equals_end(self) -> None:
        """Test validation fails when start equals end"""
        timestamp = datetime(2025, 7, 1, 10, 0, 0)
        assert validate_time_range(timestamp, timestamp) is False

    def test_range_exceeds_max_days(self) -> None:
        """Test validation fails when range exceeds maximum days"""
        start = datetime(2025, 1, 1, 10, 0, 0)
        end = datetime(2025, 12, 31, 10, 0, 0)  # About 364 days
        assert validate_time_range(start, end, max_days=90) is False

    def test_custom_max_days(self) -> None:
        """Test validation with custom maximum days"""
        start = datetime(2025, 7, 1, 10, 0, 0)
        end = datetime(2025, 7, 8, 10, 0, 0)  # 7 days

        assert validate_time_range(start, end, max_days=10) is True
        assert validate_time_range(start, end, max_days=5) is False

    def test_future_end_date(self) -> None:
        """Test validation fails for future end dates"""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow() + timedelta(days=1)  # Future date
        assert validate_time_range(start, end) is False

    def test_exactly_max_days(self) -> None:
        """Test validation at exactly the maximum day limit"""
        start = datetime(2025, 7, 1, 10, 0, 0)
        end = datetime(2025, 9, 29, 10, 0, 0)  # Exactly 90 days
        assert validate_time_range(start, end, max_days=90) is False  # > max_days

    def test_within_max_days(self) -> None:
        """Test validation just under the maximum day limit"""
        start = datetime(2025, 4, 1, 10, 0, 0)  # Past date
        end = datetime(2025, 6, 25, 10, 0, 0)  # 85 days later, past date
        assert validate_time_range(start, end, max_days=90) is True


class TestValidateRedisKey:
    """Test Redis key validation"""

    def test_valid_redis_keys(self) -> None:
        """Test various valid Redis key formats"""
        valid_keys = [
            "simple_key",
            "key:with:colons",
            "key-with-hyphens",
            "key.with.dots",
            "key_123_numbers",
            "MixedCaseKey",
            "key/with/slashes",
        ]

        for key in valid_keys:
            assert validate_redis_key(key) is True

    def test_empty_key(self) -> None:
        """Test that empty keys are invalid"""
        assert validate_redis_key("") is False
        assert validate_redis_key(None) is False  # type: ignore[arg-type]

    def test_non_string_keys(self) -> None:
        """Test that non-string keys are invalid"""
        invalid_keys: list[Any] = [123, [], {}, 3.14, True]

        for key in invalid_keys:
            assert validate_redis_key(key) is False

    def test_key_length_limit(self) -> None:
        """Test key length validation"""
        # Valid length
        valid_key = "a" * 1024
        assert validate_redis_key(valid_key) is True

        # Too long
        long_key = "a" * 1025
        assert validate_redis_key(long_key) is False

    def test_control_characters(self) -> None:
        """Test that keys with control characters are invalid"""
        control_chars = [
            "key\x00with\x00nulls",
            "key\twith\ttabs",
            "key\nwith\nnewlines",
            "key\rwith\rreturns",
            "key\x01with\x01control",
        ]

        for key in control_chars:
            assert validate_redis_key(key) is False

    def test_boundary_control_characters(self) -> None:
        """Test characters at the boundary of control character range"""
        # Character 31 (0x1F) is a control character
        assert validate_redis_key("key\x1ftest") is False

        # Character 32 (space) is the first printable character
        assert validate_redis_key("key test") is True

    def test_unicode_keys(self) -> None:
        """Test that unicode keys are valid"""
        unicode_keys = [
            "key_with_émojis",
            "key_测试",
            "key_🔑",
        ]

        for key in unicode_keys:
            assert validate_redis_key(key) is True


class TestValidateSessionData:
    """Test session data validation"""

    def test_valid_session_data(self) -> None:
        """Test valid session data structures"""
        valid_sessions = [
            {"user_id": "123", "username": "testuser"},
            {"preferences": {"theme": "dark"}, "last_login": "2025-07-16"},
            {},  # Empty dict is valid
            {"complex": {"nested": {"data": [1, 2, 3]}}},
        ]

        for session in valid_sessions:
            assert validate_session_data(session) is True

    def test_non_dict_session_data(self) -> None:
        """Test that non-dict data is invalid"""
        invalid_data = [
            "string",
            123,
            [],
            None,
            True,
        ]

        for data in invalid_data:
            assert validate_session_data(data) is False

    def test_session_data_size_limit(self) -> None:
        """Test session data size validation"""
        # Create data that's under the limit
        small_data = {"key": "a" * 1000}  # Small data
        assert validate_session_data(small_data) is True

        # Create data that exceeds 1MB limit
        large_data = {"key": "a" * (1024 * 1024 + 1)}  # >1MB
        assert validate_session_data(large_data) is False

    def test_non_serializable_data(self) -> None:
        """Test that non-JSON-serializable data is invalid"""
        # Functions are not JSON serializable
        invalid_data: dict[str, Any] = {"function": lambda x: x}
        assert validate_session_data(invalid_data) is False

        # Sets are not JSON serializable
        invalid_data = {"set": {1, 2, 3}}
        assert validate_session_data(invalid_data) is False

    def test_circular_reference_data(self) -> None:
        """Test handling of circular references"""
        # Create circular reference
        circular_data: dict[str, Any] = {"key": "value"}
        circular_data["self"] = circular_data

        assert validate_session_data(circular_data) is False

    def test_nested_data_size(self) -> None:
        """Test deeply nested data validation"""
        # Create deeply nested but small data
        nested_data = {"level1": {"level2": {"level3": {"value": "test"}}}}
        assert validate_session_data(nested_data) is True

        # Create nested data with large strings
        large_nested = {"level1": {"data": "x" * (1024 * 1024)}}
        assert validate_session_data(large_nested) is False

    def test_exactly_size_limit(self) -> None:
        """Test data exactly at the size limit"""
        # Calculate exact size for 1MB JSON
        # Accounting for JSON formatting: {"key":"..."} = 9 extra chars
        max_value_size = 1024 * 1024 - 12  # 1MB minus JSON overhead (more conservative)
        exact_data = {"key": "a" * max_value_size}

        # This should be right at or just under the limit
        serialized = json.dumps(exact_data)
        assert len(serialized) <= 1024 * 1024
        assert validate_session_data(exact_data) is True
