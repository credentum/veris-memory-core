#!/usr/bin/env python3
"""
Comprehensive tests for utility functions to achieve high coverage.

This test suite covers:
- core.utils utility functions (sanitize_error_message, get_environment, get_secure_connection_config)
- core.test_config configuration utilities (get_test_config, get_minimal_config, merge_configs)
- core.config_error exception classes (ConfigurationError, ConfigFileNotFoundError, etc.)
- Edge cases, security patterns, and error handling
"""

import base64
import os
import urllib.parse
import warnings
from unittest.mock import patch

import pytest

from src.core.config_error import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigurationError,
    ConfigValidationError,
)
from src.core.test_config import get_minimal_config, get_test_config, merge_configs
from src.core.utils import get_environment, get_secure_connection_config, sanitize_error_message


class TestSanitizeErrorMessage:
    """Test sanitize_error_message function."""

    def test_sanitize_error_message_empty_input(self):
        """Test with empty or None input."""
        assert sanitize_error_message("") == ""
        assert sanitize_error_message(None) is None

    def test_sanitize_error_message_no_sensitive_data(self):
        """Test with clean error message."""
        clean_message = "Connection failed to database"
        result = sanitize_error_message(clean_message)
        assert result == clean_message

    def test_sanitize_error_message_with_sensitive_values(self):
        """Test with provided sensitive values."""
        message = "Authentication failed with password 'secret123' and token 'abc123'"
        sensitive_values = ["secret123", "abc123"]

        result = sanitize_error_message(message, sensitive_values)

        assert "secret123" not in result
        assert "abc123" not in result
        assert "***" in result

    def test_sanitize_error_message_ignore_short_values(self):
        """Test that short sensitive values are ignored."""
        message = "Error with value 'ab' and token 'xy'"
        sensitive_values = ["ab", "xy"]  # Both too short (< 3 chars)

        result = sanitize_error_message(message, sensitive_values)

        # Short values should not be sanitized by the sensitive_values list
        # but 'x' might still be caught by other patterns
        assert "ab" in result

    def test_sanitize_error_message_url_encoded(self):
        """Test sanitization of URL-encoded values."""
        password = "pass@word#123"
        encoded_password = urllib.parse.quote(password)
        message = f"Failed with password {password} and encoded {encoded_password}"

        result = sanitize_error_message(message, [password])

        assert password not in result
        assert encoded_password not in result
        assert "***" in result

    def test_sanitize_error_message_base64_encoded(self):
        """Test sanitization of base64-encoded values."""
        password = "secret123"
        b64_password = base64.b64encode(password.encode()).decode()
        message = f"Auth failed with password {password} and b64 {b64_password}"

        result = sanitize_error_message(message, [password])

        assert password not in result
        assert b64_password not in result
        assert "***" in result

    def test_sanitize_error_message_base64_encoding_error(self):
        """Test handling of base64 encoding errors."""
        # Create a mock that will raise an exception during base64 encoding
        with patch("base64.b64encode", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "test")):
            message = "Error with password secret123"
            result = sanitize_error_message(message, ["secret123"])

            # Should still sanitize the original value even if base64 fails
            assert "secret123" not in result
            assert "***" in result

    def test_sanitize_connection_strings(self):
        """Test sanitization of database connection strings."""
        connection_strings = [
            "mongodb://user:password@localhost:27017/db",
            "postgresql://admin:secret@db.example.com:5432/mydb",
            "mysql://root:123456@mysql.local/database",
            "redis://user:pass@redis-server:6379/0",
            "neo4j://neo4j:password@graph.db:7687",
        ]

        for conn_str in connection_strings:
            result = sanitize_error_message(f"Connection failed: {conn_str}")

            # Should not contain original credentials
            assert (
                "password" not in result
                or result.count("password") <= conn_str.count("password") - 1
            )
            assert "secret" not in result
            assert "123456" not in result
            assert "pass" not in result or "***" in result

    def test_sanitize_bolt_protocol(self):
        """Test sanitization of Neo4j bolt protocol connections."""
        bolt_urls = [
            "bolt://neo4j:password@localhost:7687",
            "bolt+s://user:secret@secure.neo4j.com:7687",
            "bolt+ssc://admin:token123@enterprise.db:7687",
        ]

        for bolt_url in bolt_urls:
            result = sanitize_error_message(f"Neo4j error: {bolt_url}")

            assert "password" not in result
            assert "secret" not in result
            assert "token123" not in result
            assert "***:***@" in result  # Credentials should be sanitized

    def test_sanitize_auth_headers(self):
        """Test sanitization of authorization headers."""
        auth_headers = [
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature",
            "authorization: bearer simple_token_123",
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "AUTHORIZATION: BASIC YWRtaW46c2VjcmV0",
        ]

        for header in auth_headers:
            result = sanitize_error_message(f"HTTP error with header: {header}")

            assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
            assert "simple_token_123" not in result
            assert "dXNlcjpwYXNzd29yZA==" not in result
            assert "YWRtaW46c2VjcmV0" not in result
            assert "Authorization: ***" in result or "Bearer ***" in result or "Basic ***" in result

    def test_sanitize_credential_patterns(self):
        """Test sanitization of credential patterns."""
        credential_patterns = [
            'password: "secret123"',
            "api_key = 'token_abc123'",
            'token:"bearer_xyz789"',
            "secret: password123",
            "credential = myapikey456",
            "API-KEY: service_token_999",
        ]

        for pattern in credential_patterns:
            result = sanitize_error_message(f"Config error: {pattern}")

            assert "secret123" not in result
            assert "token_abc123" not in result
            assert "bearer_xyz789" not in result
            assert "password123" not in result
            assert "myapikey456" not in result
            assert "service_token_999" not in result
            assert "***" in result

    def test_sanitize_json_password_patterns(self):
        """Test sanitization of JSON password patterns."""
        json_patterns = [
            '{"password": "secret123"}',
            "{'password': 'hidden456'}",
            '"password":"confidential789"',
            "'password':'topsecret000'",
        ]

        for pattern in json_patterns:
            result = sanitize_error_message(f"JSON error: {pattern}")

            assert "secret123" not in result
            assert "hidden456" not in result
            assert "confidential789" not in result
            assert "topsecret000" not in result
            assert "***" in result

    def test_sanitize_case_insensitive(self):
        """Test case-insensitive sanitization."""
        message = "Error with PASSWORD secret123 and Token ABC123"
        sensitive_values = ["secret123", "abc123"]

        result = sanitize_error_message(message, sensitive_values)

        assert "secret123" not in result
        assert "ABC123" not in result
        assert "***" in result

    def test_sanitize_multiple_occurrences(self):
        """Test sanitization of multiple occurrences."""
        message = "First password secret123, second password secret123, token secret123"
        sensitive_values = ["secret123"]

        result = sanitize_error_message(message, sensitive_values)

        # All occurrences should be sanitized
        assert "secret123" not in result
        assert result.count("***") >= 3


class TestGetEnvironment:
    """Test get_environment function."""

    def test_get_environment_production(self):
        """Test production environment detection."""
        test_cases = [
            ("ENVIRONMENT", "production"),
            ("ENVIRONMENT", "prod"),
            ("ENV", "production"),
            ("NODE_ENV", "production"),
            ("ENVIRONMENT", "PRODUCTION"),  # Case insensitive
            ("ENV", "PROD"),
        ]

        for env_var, env_value in test_cases:
            with patch.dict(os.environ, {env_var: env_value}, clear=True):
                assert get_environment() == "production"

    def test_get_environment_staging(self):
        """Test staging environment detection."""
        test_cases = [
            ("ENVIRONMENT", "staging"),
            ("ENVIRONMENT", "stage"),
            ("ENV", "staging"),
            ("NODE_ENV", "stage"),
            ("ENVIRONMENT", "STAGING"),  # Case insensitive
        ]

        for env_var, env_value in test_cases:
            with patch.dict(os.environ, {env_var: env_value}, clear=True):
                assert get_environment() == "staging"

    def test_get_environment_development_default(self):
        """Test development as default environment."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_environment() == "development"

    def test_get_environment_development_explicit(self):
        """Test explicit development environment."""
        test_cases = [
            ("ENVIRONMENT", "development"),
            ("ENVIRONMENT", "dev"),
            ("ENV", "local"),
            ("NODE_ENV", "test"),
            ("ENVIRONMENT", "unknown"),  # Unknown defaults to development
        ]

        for env_var, env_value in test_cases:
            with patch.dict(os.environ, {env_var: env_value}, clear=True):
                result = get_environment()
                if env_value in ["development", "dev"]:
                    assert result == "development"
                else:
                    # Unknown values default to development
                    assert result == "development"

    def test_get_environment_precedence(self):
        """Test environment variable precedence."""
        # ENVIRONMENT should take precedence over ENV and NODE_ENV
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "production", "ENV": "staging", "NODE_ENV": "development"},
            clear=True,
        ):
            assert get_environment() == "production"

        # ENV should take precedence over NODE_ENV when ENVIRONMENT is not set
        with patch.dict(os.environ, {"ENV": "staging", "NODE_ENV": "development"}, clear=True):
            assert get_environment() == "staging"


class TestGetSecureConnectionConfig:
    """Test get_secure_connection_config function."""

    def test_get_secure_connection_config_basic(self):
        """Test basic secure connection configuration."""
        config = {"neo4j": {"host": "db.example.com", "port": 7687, "ssl": True}}

        with patch("core.utils.get_environment", return_value="development"):
            result = get_secure_connection_config(config, "neo4j")

        assert result["host"] == "db.example.com"
        assert result["port"] == 7687
        assert result["ssl"] is True
        assert result["verify_ssl"] is True
        assert result["timeout"] == 30
        assert result["environment"] == "development"

    def test_get_secure_connection_config_defaults(self):
        """Test configuration with default values."""
        config = {"qdrant": {}}

        with patch("core.utils.get_environment", return_value="development"):
            result = get_secure_connection_config(config, "qdrant")

        assert result["host"] == "localhost"
        assert result["port"] is None
        assert result["ssl"] is False  # Default for development
        assert result["verify_ssl"] is True
        assert result["timeout"] == 30

    def test_get_secure_connection_config_production_ssl_forced(self):
        """Test SSL forced in production environment."""
        config = {
            "neo4j": {
                "host": "prod.db.com",
                "port": 7687,
                # No ssl specified, should default to True in production
            }
        }

        with patch("core.utils.get_environment", return_value="production"):
            result = get_secure_connection_config(config, "neo4j")

        assert result["ssl"] is True
        assert result["environment"] == "production"

    def test_get_secure_connection_config_production_ssl_disabled_warning(self):
        """Test warning when SSL is disabled in production."""
        config = {
            "neo4j": {
                "host": "prod.db.com",
                "port": 7687,
                "ssl": False,  # Explicitly disabled in production
            }
        }

        with patch("core.utils.get_environment", return_value="production"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = get_secure_connection_config(config, "neo4j")

                assert len(w) == 1
                assert issubclass(w[0].category, RuntimeWarning)
                assert "SSL is disabled" in str(w[0].message)
                assert "security risk" in str(w[0].message)

        assert result["ssl"] is False

    def test_get_secure_connection_config_ssl_certificates(self):
        """Test SSL certificate configuration."""
        config = {
            "qdrant": {
                "host": "secure.db.com",
                "ssl": True,
                "ssl_cert_path": "/path/to/client.crt",
                "ssl_key_path": "/path/to/client.key",
                "ssl_ca_path": "/path/to/ca.crt",
            }
        }

        with patch("core.utils.get_environment", return_value="production"):
            result = get_secure_connection_config(config, "qdrant")

        assert result["ssl_cert_path"] == "/path/to/client.crt"
        assert result["ssl_key_path"] == "/path/to/client.key"
        assert result["ssl_ca_path"] == "/path/to/ca.crt"

    def test_get_secure_connection_config_partial_ssl_certs(self):
        """Test configuration with only some SSL certificate paths."""
        config = {
            "neo4j": {
                "host": "db.com",
                "ssl": True,
                "ssl_cert_path": "/path/to/client.crt",
                # Missing ssl_key_path and ssl_ca_path
            }
        }

        with patch("core.utils.get_environment", return_value="production"):
            result = get_secure_connection_config(config, "neo4j")

        assert result["ssl_cert_path"] == "/path/to/client.crt"
        assert "ssl_key_path" not in result
        assert "ssl_ca_path" not in result

    def test_get_secure_connection_config_custom_timeout(self):
        """Test custom timeout configuration."""
        config = {"redis": {"host": "cache.example.com", "port": 6379, "timeout": 60}}

        with patch("core.utils.get_environment", return_value="development"):
            result = get_secure_connection_config(config, "redis")

        assert result["timeout"] == 60

    def test_get_secure_connection_config_missing_service(self):
        """Test configuration for non-existent service."""
        config = {"other_service": {"host": "other.com"}}

        with patch("core.utils.get_environment", return_value="development"):
            result = get_secure_connection_config(config, "missing_service")

        # Should return defaults when service config is missing
        assert result["host"] == "localhost"
        assert result["port"] is None
        assert result["ssl"] is False


class TestTestConfig:
    """Test test_config utility functions."""

    def test_get_test_config_structure(self):
        """Test test configuration structure."""
        config = get_test_config()

        # Verify all expected sections exist
        expected_sections = ["neo4j", "qdrant", "redis", "storage", "embedding", "security"]
        for section in expected_sections:
            assert section in config
            assert isinstance(config[section], dict)

    def test_get_test_config_neo4j(self):
        """Test Neo4j test configuration."""
        config = get_test_config()
        neo4j_config = config["neo4j"]

        assert neo4j_config["host"] == "localhost"
        assert neo4j_config["port"] == 7687
        assert neo4j_config["database"] == "test_context"
        assert neo4j_config["username"] == "neo4j"
        assert neo4j_config["password"] == "test_password"
        assert neo4j_config["ssl"] is False

    def test_get_test_config_qdrant(self):
        """Test Qdrant test configuration."""
        config = get_test_config()
        qdrant_config = config["qdrant"]

        assert qdrant_config["host"] == "localhost"
        assert qdrant_config["port"] == 6333
        assert qdrant_config["collection_name"] == "test_contexts"
        assert qdrant_config["dimensions"] == 384
        assert qdrant_config["https"] is False

    def test_get_test_config_redis(self):
        """Test Redis test configuration."""
        config = get_test_config()
        redis_config = config["redis"]

        assert redis_config["host"] == "localhost"
        assert redis_config["port"] == 6379
        assert redis_config["database"] == 0
        assert redis_config["password"] is None
        assert redis_config["ssl"] is False

    def test_get_minimal_config_structure(self):
        """Test minimal configuration structure."""
        config = get_minimal_config()

        expected_sections = ["neo4j", "qdrant", "redis"]
        for section in expected_sections:
            assert section in config
            assert isinstance(config[section], dict)

    def test_get_minimal_config_content(self):
        """Test minimal configuration content."""
        config = get_minimal_config()

        assert config["neo4j"]["host"] == "localhost"
        assert config["neo4j"]["port"] == 7687
        assert config["qdrant"]["host"] == "localhost"
        assert config["qdrant"]["port"] == 6333
        assert config["redis"]["host"] == "localhost"
        assert config["redis"]["port"] == 6379

    def test_merge_configs_no_override(self):
        """Test config merging with no override."""
        base = {"key1": "value1", "key2": {"nested": "value"}}

        result = merge_configs(base, None)

        assert result == base
        assert result is not base  # Should be a copy

    def test_merge_configs_simple_override(self):
        """Test simple config override."""
        base = {"key1": "value1", "key2": "value2"}
        override = {"key2": "new_value", "key3": "value3"}

        result = merge_configs(base, override)

        assert result["key1"] == "value1"
        assert result["key2"] == "new_value"
        assert result["key3"] == "value3"

    def test_merge_configs_nested_merge(self):
        """Test nested dictionary merging."""
        base = {
            "database": {"host": "localhost", "port": 5432, "name": "mydb"},
            "cache": {"ttl": 3600},
        }

        override = {
            "database": {"host": "remote.db.com", "ssl": True},
            "new_section": {"setting": "value"},
        }

        result = merge_configs(base, override)

        # Database section should be merged
        assert result["database"]["host"] == "remote.db.com"  # Overridden
        assert result["database"]["port"] == 5432  # Preserved
        assert result["database"]["name"] == "mydb"  # Preserved
        assert result["database"]["ssl"] is True  # Added

        # Cache section should be preserved
        assert result["cache"]["ttl"] == 3600

        # New section should be added
        assert result["new_section"]["setting"] == "value"

    def test_merge_configs_deep_nesting(self):
        """Test deep nested dictionary merging."""
        base = {"level1": {"level2": {"level3": {"setting1": "value1", "setting2": "value2"}}}}

        override = {
            "level1": {"level2": {"level3": {"setting2": "new_value2", "setting3": "value3"}}}
        }

        result = merge_configs(base, override)

        level3 = result["level1"]["level2"]["level3"]
        assert level3["setting1"] == "value1"  # Preserved
        assert level3["setting2"] == "new_value2"  # Overridden
        assert level3["setting3"] == "value3"  # Added

    def test_merge_configs_non_dict_override(self):
        """Test merging when override value is not a dictionary."""
        base = {"section": {"nested_key": "nested_value"}}

        override = {"section": "string_value"}  # Not a dict

        result = merge_configs(base, override)

        # Should completely replace the section
        assert result["section"] == "string_value"


class TestConfigurationErrors:
    """Test configuration error classes."""

    def test_configuration_error_basic(self):
        """Test basic ConfigurationError."""
        error = ConfigurationError("Test error message")

        assert str(error) == "Test error message"
        assert error.config_path is None
        assert error.details == {}
        assert isinstance(error, Exception)

    def test_configuration_error_with_config_path(self):
        """Test ConfigurationError with config path."""
        error = ConfigurationError("Test error", config_path="/path/to/config.yaml")

        assert str(error) == "Test error"
        assert error.config_path == "/path/to/config.yaml"
        assert error.details == {}

    def test_configuration_error_with_details(self):
        """Test ConfigurationError with details."""
        details = {"key": "value", "error_code": 123}
        error = ConfigurationError("Test error", details=details)

        assert str(error) == "Test error"
        assert error.details == details

    def test_configuration_error_full(self):
        """Test ConfigurationError with all parameters."""
        details = {"validation_errors": ["error1", "error2"]}
        error = ConfigurationError(
            "Full error test", config_path="/config/app.yaml", details=details
        )

        assert str(error) == "Full error test"
        assert error.config_path == "/config/app.yaml"
        assert error.details == details

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError."""
        config_path = "/missing/config.yaml"
        error = ConfigFileNotFoundError(config_path)

        assert "Configuration file not found" in str(error)
        assert config_path in str(error)
        assert error.config_path == config_path
        assert isinstance(error, ConfigurationError)

    def test_config_parse_error(self):
        """Test ConfigParseError."""
        config_path = "/invalid/config.yaml"
        parse_error = "Invalid YAML syntax at line 5"
        error = ConfigParseError(config_path, parse_error)

        assert "Failed to parse configuration file" in str(error)
        assert config_path in str(error)
        assert parse_error in str(error)
        assert error.config_path == config_path
        assert error.details["parse_error"] == parse_error
        assert isinstance(error, ConfigurationError)

    def test_config_validation_error_basic(self):
        """Test basic ConfigValidationError."""
        error = ConfigValidationError("Validation failed")

        assert str(error) == "Validation failed"
        assert error.details == {}
        assert isinstance(error, ConfigurationError)

    def test_config_validation_error_with_fields(self):
        """Test ConfigValidationError with invalid fields."""
        invalid_fields = ["database.host", "redis.port", "ssl.enabled"]
        error = ConfigValidationError("Multiple validation errors", invalid_fields)

        assert str(error) == "Multiple validation errors"
        assert error.details["invalid_fields"] == invalid_fields

    def test_error_inheritance_chain(self):
        """Test error inheritance chain."""
        errors = [
            ConfigurationError("base error"),
            ConfigFileNotFoundError("/path"),
            ConfigParseError("/path", "parse error"),
            ConfigValidationError("validation error"),
        ]

        for error in errors:
            assert isinstance(error, Exception)
            assert isinstance(error, ConfigurationError)

    def test_error_string_representations(self):
        """Test string representations of all error types."""
        config_path = "/test/config.yaml"

        base_error = ConfigurationError("Base error", config_path)
        file_error = ConfigFileNotFoundError(config_path)
        parse_error = ConfigParseError(config_path, "YAML error")
        validation_error = ConfigValidationError("Invalid config", ["field1", "field2"])

        # All should have meaningful string representations
        assert len(str(base_error)) > 0
        assert len(str(file_error)) > 0
        assert len(str(parse_error)) > 0
        assert len(str(validation_error)) > 0

        # Specific content checks
        assert config_path in str(file_error)
        assert "YAML error" in str(parse_error)
        assert "Invalid config" in str(validation_error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
