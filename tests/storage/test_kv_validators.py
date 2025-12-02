#!/usr/bin/env python3
"""
Tests for kv_validators module
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
    """Tests for validate_cache_entry function"""

    def test_valid_cache_entry(self):
        """Test with valid cache entry"""
        data = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(data) is True

    def test_missing_required_fields(self):
        """Test with missing required fields"""
        # Missing key
        data = {
            "value": "test_value",
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(data) is False

        # Missing value
        data = {
            "key": "test_key",
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(data) is False

        # Missing created_at
        data = {"key": "test_key", "value": "test_value", "ttl_seconds": 3600}
        assert validate_cache_entry(data) is False

        # Missing ttl_seconds
        data = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.utcnow().isoformat(),
        }
        assert validate_cache_entry(data) is False

    def test_invalid_key_type(self):
        """Test with invalid key type"""
        data = {
            "key": 123,  # Should be string
            "value": "test_value",
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(data) is False

    def test_invalid_ttl_seconds(self):
        """Test with invalid ttl_seconds"""
        # Not an integer
        data = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": "3600",
        }
        assert validate_cache_entry(data) is False

        # Negative value
        data = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.utcnow().isoformat(),
            "ttl_seconds": -1,  # type: ignore[dict-item]
        }
        assert validate_cache_entry(data) is False

    def test_invalid_timestamp(self):
        """Test with invalid timestamp"""
        # Invalid format
        data = {
            "key": "test_key",
            "value": "test_value",
            "created_at": "invalid_timestamp",
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(data) is False

        # Non-string timestamp
        data = {
            "key": "test_key",
            "value": "test_value",
            "created_at": 12345,
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(data) is False


class TestValidateMetricEvent:
    """Tests for validate_metric_event function"""

    def test_valid_metric_event(self):
        """Test with valid metric event"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": "cpu_usage",
            "value": 75.5,
            "tags": {"host": "server1", "region": "us-east"},
        }
        assert validate_metric_event(metric) is True

    def test_valid_metric_with_datetime_object(self):
        """Test with datetime object instead of string"""
        metric = {
            "timestamp": datetime.utcnow(),
            "metric_name": "cpu_usage",
            "value": 75.5,
            "tags": {"host": "server1"},
        }
        assert validate_metric_event(metric) is True

    def test_missing_fields(self):
        """Test with missing required fields"""
        base_metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": "cpu_usage",
            "value": 75.5,
            "tags": {},
        }

        # Test each missing field
        for field in ["timestamp", "metric_name", "value", "tags"]:
            metric = base_metric.copy()
            del metric[field]
            assert validate_metric_event(metric) is False

    def test_invalid_metric_name_type(self):
        """Test with invalid metric_name type"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": 123,  # Should be string
            "value": 75.5,
            "tags": {},
        }
        assert validate_metric_event(metric) is False

    def test_invalid_value_type(self):
        """Test with invalid value type"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": "cpu_usage",
            "value": "75.5",  # Should be numeric
            "tags": {},
        }
        assert validate_metric_event(metric) is False

    def test_invalid_tags_type(self):
        """Test with invalid tags type"""
        metric = {
            "timestamp": datetime.utcnow().isoformat(),
            "metric_name": "cpu_usage",
            "value": 75.5,
            "tags": ["tag1", "tag2"],  # Should be dict
        }
        assert validate_metric_event(metric) is False

    def test_invalid_timestamp_format(self):
        """Test with invalid timestamp format"""
        metric = {
            "timestamp": "invalid_timestamp",
            "metric_name": "cpu_usage",
            "value": 75.5,
            "tags": {},
        }
        assert validate_metric_event(metric) is False

    def test_invalid_timestamp_type(self):
        """Test with invalid timestamp type"""
        metric = {
            "timestamp": 12345,  # Neither string nor datetime
            "metric_name": "cpu_usage",
            "value": 75.5,
            "tags": {},
        }
        assert validate_metric_event(metric) is False


class TestSanitizeMetricName:
    """Tests for sanitize_metric_name function"""

    def test_valid_metric_name(self):
        """Test with already valid metric name"""
        assert sanitize_metric_name("cpu_usage.percent") == "cpu_usage.percent"
        assert sanitize_metric_name("disk-io_read") == "disk-io_read"

    def test_special_characters(self):
        """Test sanitization of special characters"""
        assert sanitize_metric_name("cpu@usage#percent") == "cpu_usage_percent"
        assert sanitize_metric_name("metric!@#$%^&*()") == "metric__________"

    def test_long_metric_name(self):
        """Test truncation of long metric names"""
        long_name = "a" * 150
        result = sanitize_metric_name(long_name)
        assert len(result) == 100
        assert result == "a" * 100

    def test_unicode_characters(self):
        """Test sanitization of unicode characters"""
        assert sanitize_metric_name("cpu_使用率") == "cpu____"
        assert sanitize_metric_name("métric_ñame") == "m_tric__ame"


class TestValidateTimeRange:
    """Tests for validate_time_range function"""

    def test_valid_time_range(self):
        """Test with valid time range"""
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(start, end) is True

    def test_start_after_end(self):
        """Test with start time after end time"""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow() - timedelta(days=7)
        assert validate_time_range(start, end) is False

    def test_start_equals_end(self):
        """Test with start time equal to end time"""
        time = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(time, time) is False

    def test_exceeds_max_days(self):
        """Test with range exceeding max days"""
        start = datetime.utcnow() - timedelta(days=100)
        end = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(start, end, max_days=90) is False

    def test_custom_max_days(self):
        """Test with custom max days"""
        start = datetime.utcnow() - timedelta(days=40)
        end = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(start, end, max_days=30) is False
        assert validate_time_range(start, end, max_days=50) is True

    def test_future_end_date(self):
        """Test with future end date"""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow() + timedelta(days=1)
        assert validate_time_range(start, end) is False


class TestValidateRedisKey:
    """Tests for validate_redis_key function"""

    def test_valid_redis_key(self):
        """Test with valid Redis key"""
        assert validate_redis_key("user:1234") is True
        assert validate_redis_key("session_abc123") is True
        assert validate_redis_key("cache:product:5678") is True

    def test_empty_key(self):
        """Test with empty key"""
        assert validate_redis_key("") is False
        assert validate_redis_key(None) is False  # type: ignore[arg-type]

    def test_invalid_type(self):
        """Test with invalid key type"""
        assert validate_redis_key(123) is False  # type: ignore[arg-type]
        assert validate_redis_key([]) is False  # type: ignore[arg-type]
        assert validate_redis_key({}) is False  # type: ignore[arg-type]

    def test_key_too_long(self):
        """Test with key exceeding length limit"""
        long_key = "a" * 1025
        assert validate_redis_key(long_key) is False

    def test_control_characters(self):
        """Test with control characters"""
        assert validate_redis_key("key\x00null") is False
        assert validate_redis_key("key\nline") is False
        assert validate_redis_key("key\ttest") is False


class TestValidateSessionData:
    """Tests for validate_session_data function"""

    def test_valid_session_data(self):
        """Test with valid session data"""
        data = {"user_id": 123, "username": "test_user", "roles": ["admin", "user"]}
        assert validate_session_data(data) is True

    def test_invalid_type(self):
        """Test with non-dict types"""
        assert validate_session_data("string") is False
        assert validate_session_data(123) is False
        assert validate_session_data([1, 2, 3]) is False
        assert validate_session_data(None) is False

    def test_data_too_large(self):
        """Test with data exceeding size limit"""
        # Create data larger than 1MB
        large_data = {"key": "x" * (1024 * 1024 + 1)}
        assert validate_session_data(large_data) is False

    def test_non_serializable_data(self):
        """Test with non-JSON-serializable data"""
        # Datetime objects are not JSON serializable by default
        data = {"timestamp": datetime.utcnow()}
        assert validate_session_data(data) is False

        # Set objects are not JSON serializable
        data = {"items": {1, 2, 3}}  # type: ignore[dict-item]
        assert validate_session_data(data) is False

    def test_circular_reference(self):
        """Test with circular reference"""
        data = {"key": "value"}
        data["self"] = data  # type: ignore[assignment] # Circular reference
        assert validate_session_data(data) is False

    def test_nested_valid_data(self):
        """Test with complex nested but valid data"""
        data = {
            "user": {"id": 123, "name": "test"},
            "settings": {"theme": "dark", "notifications": True},
            "history": [{"action": "login", "timestamp": "2023-01-01"}],
        }
        assert validate_session_data(data) is True
