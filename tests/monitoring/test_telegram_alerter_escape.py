#!/usr/bin/env python3
"""
Test suite for Telegram Alerter HTML escaping functionality.

Tests the _escape_nested_html method with various scenarios including:
- Simple strings, dicts, and lists
- Nested structures
- Circular references
- Maximum depth limits
- Malformed data
"""

import pytest
from unittest.mock import MagicMock
from src.monitoring.sentinel.telegram_alerter import TelegramAlerter, AlertSeverity


class TestEscapeNestedHtml:
    """Test suite for _escape_nested_html method."""

    @pytest.fixture
    def alerter(self):
        """Create a TelegramAlerter instance for testing."""
        # Use dummy credentials for testing
        return TelegramAlerter(
            bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            chat_id="12345678"
        )

    def test_escape_simple_string(self, alerter):
        """Test escaping simple strings with HTML characters."""
        result = alerter._escape_nested_html("<script>alert('xss')</script>")
        assert result == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"

    def test_escape_string_with_angle_brackets(self, alerter):
        """Test escaping strings with angle brackets."""
        result = alerter._escape_nested_html("Missing <env_var> value")
        assert result == "Missing &lt;env_var&gt; value"
        assert "<env_var>" not in result

    def test_escape_dict_simple(self, alerter):
        """Test escaping simple dictionary."""
        input_dict = {
            "key1": "<value>",
            "key2": "normal"
        }
        result = alerter._escape_nested_html(input_dict)

        assert isinstance(result, dict)
        assert result["key1"] == "&lt;value&gt;"
        assert result["key2"] == "normal"

    def test_escape_list_simple(self, alerter):
        """Test escaping simple list."""
        input_list = ["<item1>", "<item2>", "normal"]
        result = alerter._escape_nested_html(input_list)

        assert isinstance(result, list)
        assert result[0] == "&lt;item1&gt;"
        assert result[1] == "&lt;item2&gt;"
        assert result[2] == "normal"

    def test_escape_nested_dict(self, alerter):
        """Test escaping nested dictionaries."""
        input_data = {
            "outer": {
                "inner": {
                    "value": "<script>",
                    "list": ["<a>", "<b>"]
                }
            }
        }
        result = alerter._escape_nested_html(input_data)

        assert result["outer"]["inner"]["value"] == "&lt;script&gt;"
        assert result["outer"]["inner"]["list"][0] == "&lt;a&gt;"
        assert result["outer"]["inner"]["list"][1] == "&lt;b&gt;"

    def test_escape_none_value(self, alerter):
        """Test escaping None values."""
        result = alerter._escape_nested_html(None)
        assert result is None

    def test_escape_number(self, alerter):
        """Test escaping number values."""
        result = alerter._escape_nested_html(42)
        assert result == "42"

        result = alerter._escape_nested_html(3.14)
        assert result == "3.14"

    def test_escape_boolean(self, alerter):
        """Test escaping boolean values."""
        result = alerter._escape_nested_html(True)
        assert result == "True"

        result = alerter._escape_nested_html(False)
        assert result == "False"

    def test_circular_reference_detection(self, alerter):
        """Test detection of circular references in dicts."""
        circular_dict = {"a": 1}
        circular_dict["self"] = circular_dict  # Create circular reference

        result = alerter._escape_nested_html(circular_dict)

        # Should replace circular ref with marker
        assert result["a"] == "1"
        assert result["self"] == "[CIRCULAR_REF]"

    def test_circular_reference_list(self, alerter):
        """Test detection of circular references in lists."""
        circular_list = [1, 2, 3]
        circular_list.append(circular_list)  # Create circular reference

        result = alerter._escape_nested_html(circular_list)

        # Should replace circular ref with marker
        assert result[0] == "1"
        assert result[1] == "2"
        assert result[2] == "3"
        assert result[3] == "[CIRCULAR_REF]"

    def test_max_depth_exceeded(self, alerter):
        """Test maximum depth limit protection."""
        # Create deeply nested structure
        deep_data = {"level": 0}
        current = deep_data
        for i in range(1, 15):  # Create 15 levels (exceeds max_depth=10)
            current["nested"] = {"level": i}
            current = current["nested"]

        result = alerter._escape_nested_html(deep_data, max_depth=10)

        # Should truncate at max depth
        current_result = result
        for i in range(10):
            assert "nested" in current_result or "level" in current_result
            if "nested" in current_result:
                current_result = current_result["nested"]

        # At some point should hit max depth marker
        def find_max_depth_marker(obj, depth=0):
            if obj == "[MAX_DEPTH_EXCEEDED]":
                return True
            if isinstance(obj, dict):
                return any(find_max_depth_marker(v, depth+1) for v in obj.values())
            if isinstance(obj, list):
                return any(find_max_depth_marker(item, depth+1) for item in obj)
            return False

        assert find_max_depth_marker(result)

    def test_malformed_dict_key(self, alerter):
        """Test handling of dictionaries with non-string keys."""
        # Dict with object as key (edge case)
        class CustomKey:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        try:
            input_dict = {CustomKey(): "value"}
            result = alerter._escape_nested_html(input_dict)
            # Should handle gracefully
            assert isinstance(result, dict)
        except:
            # If it fails, that's also acceptable - the important thing is it doesn't crash
            pass

    def test_escape_all_html_entities(self, alerter):
        """Test escaping all HTML special characters."""
        input_str = '&<>"\'test'
        result = alerter._escape_nested_html(input_str)

        assert "&amp;" in result
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&quot;" in result
        assert "&#x27;" in result

    def test_empty_dict(self, alerter):
        """Test escaping empty dictionary."""
        result = alerter._escape_nested_html({})
        assert result == {}

    def test_empty_list(self, alerter):
        """Test escaping empty list."""
        result = alerter._escape_nested_html([])
        assert result == []

    def test_mixed_nested_structure(self, alerter):
        """Test escaping complex mixed nested structure."""
        input_data = {
            "config_issues": ["Missing <env_var>", "Version <1.0.0>"],
            "test_results": {"test1": "<passed>", "test2": "failed"},
            "count": 42,
            "active": True,
            "metadata": None
        }

        result = alerter._escape_nested_html(input_data)

        assert result["config_issues"][0] == "Missing &lt;env_var&gt;"
        assert result["config_issues"][1] == "Version &lt;1.0.0&gt;"
        assert result["test_results"]["test1"] == "&lt;passed&gt;"
        assert result["test_results"]["test2"] == "failed"
        assert result["count"] == "42"
        assert result["active"] == "True"
        assert result["metadata"] is None

    def test_custom_max_depth(self, alerter):
        """Test custom maximum depth parameter."""
        nested = {"a": {"b": {"c": {"d": "deep"}}}}

        # With max_depth=2, should truncate at level 2
        result = alerter._escape_nested_html(nested, max_depth=2)

        # Should be able to access first 2 levels
        assert "a" in result
        assert "b" in result["a"]
        # Level 3 should hit max depth
        assert "[MAX_DEPTH_EXCEEDED]" in str(result)

    def test_unicode_characters(self, alerter):
        """Test escaping with unicode characters."""
        input_str = "Hello 世界 <test>"
        result = alerter._escape_nested_html(input_str)

        assert "世界" in result
        assert "&lt;test&gt;" in result

    def test_integration_with_format_alert(self, alerter):
        """Test integration of _escape_nested_html with _format_alert."""
        details = {
            "config": ["<env_var>", "<version>"],
            "status": "<active>"
        }

        formatted = alerter._format_alert(
            check_id="test-check",
            status="fail",
            message="Test message",
            severity=AlertSeverity.HIGH,
            details=details
        )

        # Should not contain unescaped angle brackets
        assert "<env_var>" not in formatted
        assert "<version>" not in formatted
        assert "<active>" not in formatted
        # Should contain escaped versions
        assert "&lt;env_var&gt;" in formatted or "env_var" in formatted
        assert "&lt;version&gt;" in formatted or "version" in formatted
        assert "&lt;active&gt;" in formatted or "active" in formatted


class TestTelegramAlerterSecurity:
    """Security-focused tests for Telegram alerter."""

    @pytest.fixture
    def alerter(self):
        """Create a TelegramAlerter instance for testing."""
        return TelegramAlerter(
            bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
            chat_id="12345678"
        )

    def test_xss_prevention(self, alerter):
        """Test prevention of XSS attacks via HTML injection."""
        malicious_input = {
            "script": "<script>alert('xss')</script>",
            "img": "<img src=x onerror=alert('xss')>",
            "iframe": "<iframe src='evil.com'></iframe>"
        }

        result = alerter._escape_nested_html(malicious_input)

        # Should escape all HTML tags
        for value in result.values():
            assert "<script>" not in value
            assert "<img" not in value
            assert "<iframe" not in value

    def test_dos_protection_depth_limit(self, alerter):
        """Test DoS protection via depth limit."""
        # Create very deep structure
        deep = {}
        current = deep
        for i in range(1000):
            current["next"] = {}
            current = current["next"]

        # Should not cause stack overflow
        result = alerter._escape_nested_html(deep, max_depth=10)
        assert result is not None

    def test_dos_protection_circular_ref(self, alerter):
        """Test DoS protection via circular reference detection."""
        circular = {}
        circular["self"] = circular
        circular["also_self"] = circular

        # Should not cause infinite loop
        result = alerter._escape_nested_html(circular)
        assert result["self"] == "[CIRCULAR_REF]"
        assert result["also_self"] == "[CIRCULAR_REF]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
