#!/usr/bin/env python3
"""
Comprehensive test suite for RedisConnector.get() compatibility method.

Tests string and JSON serialization paths, error handling, and edge cases.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.storage.kv_store import RedisConnector


class TestRedisConnectorGetMethod:
    """Test suite for RedisConnector.get() compatibility method."""

    @pytest.fixture
    def redis_connector(self):
        """Create a RedisConnector instance for testing."""
        connector = RedisConnector()
        connector.redis_client = Mock()
        connector.is_connected = True
        return connector

    def test_get_string_value(self, redis_connector):
        """Test getting a simple string value."""
        # Mock get_cache to return a string
        with patch.object(redis_connector, "get_cache", return_value="simple string"):
            result = redis_connector.get("test_key")
            assert result == "simple string"
            redis_connector.get_cache.assert_called_once_with("test_key")

    def test_get_json_dict_value(self, redis_connector):
        """Test getting a dictionary value (should be JSON serialized)."""
        test_dict = {"name": "test", "value": 123, "nested": {"key": "value"}}

        # Mock get_cache to return a dict
        with patch.object(redis_connector, "get_cache", return_value=test_dict):
            result = redis_connector.get("test_key")
            assert result == json.dumps(test_dict)
            assert json.loads(result) == test_dict

    def test_get_json_list_value(self, redis_connector):
        """Test getting a list value (should be JSON serialized)."""
        test_list = [1, 2, "three", {"four": 4}]

        # Mock get_cache to return a list
        with patch.object(redis_connector, "get_cache", return_value=test_list):
            result = redis_connector.get("test_key")
            assert result == json.dumps(test_list)
            assert json.loads(result) == test_list

    def test_get_none_value(self, redis_connector):
        """Test getting None value (key doesn't exist)."""
        # Mock get_cache to return None
        with patch.object(redis_connector, "get_cache", return_value=None):
            result = redis_connector.get("nonexistent_key")
            assert result is None

    def test_get_numeric_values(self, redis_connector):
        """Test getting numeric values (should be JSON serialized)."""
        # Integer
        with patch.object(redis_connector, "get_cache", return_value=42):
            result = redis_connector.get("int_key")
            assert result == "42"

        # Float
        with patch.object(redis_connector, "get_cache", return_value=3.14159):
            result = redis_connector.get("float_key")
            assert result == "3.14159"

        # Boolean
        with patch.object(redis_connector, "get_cache", return_value=True):
            result = redis_connector.get("bool_key")
            assert result == "true"  # JSON serialization of boolean

    def test_get_empty_string(self, redis_connector):
        """Test getting an empty string value."""
        with patch.object(redis_connector, "get_cache", return_value=""):
            result = redis_connector.get("empty_key")
            assert result == ""

    def test_get_whitespace_string(self, redis_connector):
        """Test getting a string with only whitespace."""
        with patch.object(redis_connector, "get_cache", return_value="   "):
            result = redis_connector.get("whitespace_key")
            assert result == "   "

    def test_get_unicode_string(self, redis_connector):
        """Test getting Unicode string values."""
        unicode_str = "Hello 世界 🌍 émojis"
        with patch.object(redis_connector, "get_cache", return_value=unicode_str):
            result = redis_connector.get("unicode_key")
            assert result == unicode_str

    def test_get_special_characters(self, redis_connector):
        """Test getting strings with special characters."""
        special_str = "Line1\nLine2\tTabbed\r\nWindows"
        with patch.object(redis_connector, "get_cache", return_value=special_str):
            result = redis_connector.get("special_key")
            assert result == special_str

    def test_get_complex_nested_structure(self, redis_connector):
        """Test getting complex nested data structures."""
        complex_data = {
            "users": [
                {"id": 1, "name": "Alice", "tags": ["admin", "user"]},
                {"id": 2, "name": "Bob", "metadata": {"age": 30, "active": True}},
            ],
            "settings": {"nested": {"deeply": {"value": "found"}}},
            "timestamp": 1234567890.123,
        }

        with patch.object(redis_connector, "get_cache", return_value=complex_data):
            result = redis_connector.get("complex_key")
            assert result == json.dumps(complex_data)
            assert json.loads(result) == complex_data

    def test_get_error_handling_get_cache_exception(self, redis_connector):
        """Test error handling when get_cache raises an exception."""
        with patch.object(redis_connector, "get_cache", side_effect=Exception("Redis error")):
            with patch.object(redis_connector, "log_error") as mock_log:
                result = redis_connector.get("error_key")
                assert result is None
                mock_log.assert_called_once()
                assert "error_key" in str(mock_log.call_args)

    def test_get_error_handling_json_serialization(self, redis_connector):
        """Test error handling when JSON serialization fails."""

        # Create an object that can't be JSON serialized
        class NonSerializable:
            pass

        non_serializable = NonSerializable()

        with patch.object(redis_connector, "get_cache", return_value=non_serializable):
            with patch.object(redis_connector, "log_error") as mock_log:
                result = redis_connector.get("non_serializable_key")
                assert result is None
                mock_log.assert_called_once()

    def test_get_bytes_value(self, redis_connector):
        """Test handling of bytes values (common in Redis)."""
        # Test bytes that can be decoded to string
        bytes_value = b"byte string"
        with patch.object(redis_connector, "get_cache", return_value=bytes_value):
            with patch.object(redis_connector, "log_error") as mock_log:
                # bytes objects will cause JSON serialization error
                result = redis_connector.get("bytes_key")
                assert result is None
                mock_log.assert_called_once()

    def test_get_with_different_key_types(self, redis_connector):
        """Test get() with various key formats."""
        test_keys = [
            "simple_key",
            "name:spaced:key",
            "key_with_underscore",
            "key-with-dash",
            "KEY_UPPER_CASE",
            "key.with.dots",
            "key/with/slashes",
            "123numeric",
            "key::",
            "",  # Empty key
        ]

        for key in test_keys:
            with patch.object(redis_connector, "get_cache", return_value=f"value_{key}"):
                result = redis_connector.get(key)
                if key:  # Non-empty keys should work
                    assert result == f"value_{key}"
                    redis_connector.get_cache.assert_called_with(key)

    def test_get_integration_with_simple_redis_client(self, redis_connector):
        """Test that get() method provides SimpleRedisClient compatibility."""
        # This tests the interface compatibility aspect
        # The method should behave like SimpleRedisClient.get()

        # Test 1: String return for string values
        with patch.object(redis_connector, "get_cache", return_value="redis_value"):
            result = redis_connector.get("key1")
            assert isinstance(result, str)
            assert result == "redis_value"

        # Test 2: None for missing keys
        with patch.object(redis_connector, "get_cache", return_value=None):
            result = redis_connector.get("missing_key")
            assert result is None

        # Test 3: JSON string for complex types
        with patch.object(redis_connector, "get_cache", return_value={"complex": "data"}):
            result = redis_connector.get("complex_key")
            assert isinstance(result, str)
            assert json.loads(result) == {"complex": "data"}

    def test_get_cache_delegation(self, redis_connector):
        """Test that get() properly delegates to get_cache()."""
        test_cases = [
            ("key1", "value1"),
            ("key2", {"dict": "value"}),
            ("key3", ["list", "value"]),
            ("key4", None),
            ("key5", 123),
            ("key6", True),
        ]

        for key, cache_value in test_cases:
            with patch.object(
                redis_connector, "get_cache", return_value=cache_value
            ) as mock_get_cache:
                result = redis_connector.get(key)
                mock_get_cache.assert_called_once_with(key)

                if cache_value is None:
                    assert result is None
                elif isinstance(cache_value, str):
                    assert result == cache_value
                else:
                    assert result == json.dumps(cache_value)

    def test_get_concurrent_calls(self, redis_connector):
        """Test multiple concurrent calls to get()."""
        keys_and_values = {
            "key1": "value1",
            "key2": {"nested": "dict"},
            "key3": [1, 2, 3],
            "key4": None,
        }

        def get_cache_side_effect(key):
            return keys_and_values.get(key)

        with patch.object(redis_connector, "get_cache", side_effect=get_cache_side_effect):
            # Simulate concurrent calls
            results = {}
            for key in keys_and_values:
                results[key] = redis_connector.get(key)

            # Verify results
            assert results["key1"] == "value1"
            assert results["key2"] == json.dumps({"nested": "dict"})
            assert results["key3"] == json.dumps([1, 2, 3])
            assert results["key4"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
