#!/usr/bin/env python3
"""
Tests for error message sanitization functionality.

Tests that sensitive patterns are properly redacted from error messages
to prevent information leakage in logs and error responses.
"""

import pytest
from src.migration.data_migration import _sanitize_error_message


class TestErrorSanitization:
    """Test cases for error message sanitization."""

    def test_password_redaction(self):
        """Test that passwords are properly redacted."""
        test_cases = [
            ("Error: password=secret123", "Error: password=[REDACTED]"),
            ("Failed with password:secret123", "Failed with password:[REDACTED]"),
            ("PASSWORD=MySecretPass", "PASSWORD=[REDACTED]"),
            ("Error: Password = secret123", "Error: Password = [REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed to sanitize password in: {original}"

    def test_token_redaction(self):
        """Test that tokens are properly redacted."""
        test_cases = [
            ("Error: token=abc123xyz", "Error: token=[REDACTED]"),
            ("Failed with token:bearer_token_123", "Failed with token:[REDACTED]"),
            ("TOKEN=jwt.token.here", "TOKEN=[REDACTED]"),
            ("Auth token = secret_token", "Auth token = [REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed to sanitize token in: {original}"

    def test_api_key_redaction(self):
        """Test that API keys are properly redacted."""
        test_cases = [
            ("Error: api_key=sk-123456789", "Error: api_key=[REDACTED]"),
            ("Failed with api-key:secret_api_key", "Failed with api-key:[REDACTED]"),
            ("API_KEY=my_secret_key", "API_KEY=[REDACTED]"),
            ("Connection failed: apikey = secret123", "Connection failed: apikey = [REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed to sanitize API key in: {original}"

    def test_secret_redaction(self):
        """Test that secrets are properly redacted."""
        test_cases = [
            ("Error: secret=my_secret_value", "Error: secret=[REDACTED]"),
            ("Failed with secret:top_secret", "Failed with secret:[REDACTED]"),
            ("SECRET=confidential_data", "SECRET=[REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed to sanitize secret in: {original}"

    def test_ip_address_redaction(self):
        """Test that IP addresses are properly redacted."""
        test_cases = [
            ("Connection failed to 192.168.1.100", "Connection failed to [REDACTED]"),
            ("Error connecting to 10.0.0.1:5432", "Error connecting to [REDACTED]:5432"),
            ("Database at 172.16.254.1 unavailable", "Database at [REDACTED] unavailable"),
            ("Multiple IPs: 192.168.1.1, 10.0.0.1", "Multiple IPs: [REDACTED], [REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed to sanitize IP address in: {original}"

    def test_email_redaction(self):
        """Test that email addresses are properly redacted."""
        test_cases = [
            ("User admin@example.com not found", "User [REDACTED] not found"),
            ("Failed for user.name@domain.org", "Failed for [REDACTED]"),
            ("Contact support@company.com for help", "Contact [REDACTED] for help"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed to sanitize email in: {original}"

    def test_mixed_sensitive_data(self):
        """Test redaction of multiple sensitive patterns in one message."""
        original = "Auth failed: user admin@company.com with password=secret123 at 192.168.1.100"
        expected = "Auth failed: user [REDACTED] with password=[REDACTED] at [REDACTED]"
        
        result = _sanitize_error_message(original)
        assert result == expected

    def test_case_insensitive_redaction(self):
        """Test that redaction works regardless of case."""
        test_cases = [
            ("Error: PASSWORD=secret", "Error: PASSWORD=[REDACTED]"),
            ("Failed: Token=bearer123", "Failed: Token=[REDACTED]"),
            ("Issue: API_KEY=key123", "Issue: API_KEY=[REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed case-insensitive sanitization: {original}"

    def test_message_truncation(self):
        """Test that very long messages are truncated."""
        long_message = "Error: " + "x" * 300  # 307 characters total
        result = _sanitize_error_message(long_message)
        
        # Should be truncated to MAX_ERROR_MESSAGE_LENGTH (200) characters
        assert len(result) == 200
        assert result.endswith("...")

    def test_safe_content_preserved(self):
        """Test that safe content is not modified."""
        safe_messages = [
            "Database connection failed",
            "Invalid query parameter: limit must be positive",
            "Timeout after 30 seconds",
            "File not found: /path/to/file.txt",
        ]
        
        for message in safe_messages:
            result = _sanitize_error_message(message)
            assert result == message, f"Safe message was modified: {message}"

    def test_partial_matches_not_redacted(self):
        """Test that partial matches are not incorrectly redacted."""
        safe_messages = [
            "Error in keyboard input",  # Contains 'key' but not 'key='
            "Password validation failed",  # Contains 'password' but not 'password='
            "Token parsing error",  # Contains 'token' but not 'token='
        ]
        
        for message in safe_messages:
            result = _sanitize_error_message(message)
            assert result == message, f"Partial match incorrectly redacted: {message}"

    def test_empty_and_none_messages(self):
        """Test handling of empty and None messages."""
        assert _sanitize_error_message("") == ""
        assert _sanitize_error_message("   ") == "   "
        
        # Test with None converted to string
        assert _sanitize_error_message(str(None)) == "None"

    def test_redaction_with_special_characters(self):
        """Test redaction works with special characters in values."""
        test_cases = [
            ("password=P@$$w0rd!", "password=[REDACTED]"),
            ("token=jwt.eyJ0eXAi.OiJKV1Qi", "token=[REDACTED]"),
            ("api_key=sk-1234567890abcdef", "api_key=[REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = _sanitize_error_message(original)
            assert result == expected, f"Failed with special characters: {original}"