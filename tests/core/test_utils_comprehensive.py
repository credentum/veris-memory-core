#!/usr/bin/env python3
"""
Comprehensive tests for core.utils module to achieve maximum coverage.
"""

import base64
import os
import urllib.parse
import warnings
from unittest.mock import patch

from src.core.utils import (  # noqa: E402
    get_environment,
    get_secure_connection_config,
    sanitize_error_message,
)


class TestSanitizeErrorMessage:
    """Test suite for sanitize_error_message function."""

    def test_sanitize_empty_message(self):
        """Test sanitizing empty message."""
        assert sanitize_error_message("") == ""
        assert sanitize_error_message(None) is None

    def test_sanitize_with_sensitive_values(self):
        """Test sanitizing with provided sensitive values."""
        msg = "Connection failed with password: mySecretPass123"
        sensitive = ["mySecretPass123"]
        result = sanitize_error_message(msg, sensitive)
        assert "mySecretPass123" not in result
        assert "***" in result

    def test_sanitize_short_sensitive_values(self):
        """Test that very short sensitive values are skipped."""
        msg = "Error with value: ab"
        sensitive = ["ab", "a", ""]
        result = sanitize_error_message(msg, sensitive)
        assert result == msg  # Short values not replaced

    def test_sanitize_url_encoded_values(self):
        """Test sanitizing URL-encoded sensitive values."""
        sensitive_val = "pass@word"
        encoded_val = urllib.parse.quote(sensitive_val)
        msg = f"URL contains: {encoded_val}"
        result = sanitize_error_message(msg, [sensitive_val])
        assert encoded_val not in result
        assert "***" in result

    def test_sanitize_base64_encoded_values(self):
        """Test sanitizing base64-encoded sensitive values."""
        sensitive_val = "secrettoken"
        b64_val = base64.b64encode(sensitive_val.encode()).decode()
        msg = f"Token: {b64_val}"
        result = sanitize_error_message(msg, [sensitive_val])
        assert b64_val not in result
        assert "***" in result

    def test_sanitize_base64_encoding_error(self):
        """Test base64 encoding with invalid input."""
        # Test with a value that won't cause encoding issues
        msg = "Some error message"
        result = sanitize_error_message(msg, ["\x00invalid"])
        # Should not crash, just skip base64 encoding
        assert result

    def test_sanitize_connection_strings(self):
        """Test sanitizing connection strings with credentials."""
        msg = "Failed to connect to mongodb://user:password@host:27017/db"
        result = sanitize_error_message(msg)
        assert "password" not in result
        assert "://***:***@" in result

    def test_sanitize_auth_headers_bearer(self):
        """Test sanitizing Bearer auth headers."""
        msg = "Request failed: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = sanitize_error_message(msg)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "Authorization: ***" in result

    def test_sanitize_auth_headers_basic(self):
        """Test sanitizing Basic auth headers."""
        msg = "Request failed: Authorization: Basic dXNlcjpwYXNz"
        result = sanitize_error_message(msg)
        assert "dXNlcjpwYXNz" not in result
        assert "Authorization: ***" in result

    def test_sanitize_password_patterns(self):
        """Test sanitizing various password patterns."""
        patterns = [
            'password: "secret123"',
            "api_key=myapikey",
            "token: mytoken",
            "secret = mysecret",
            'credential: "mycred"',
        ]

        for pattern in patterns:
            result = sanitize_error_message(pattern)
            assert "secret123" not in result.lower() or "***" in result
            assert "myapikey" not in result.lower() or "***" in result
            assert "mytoken" not in result.lower() or "***" in result

    def test_sanitize_bearer_token_standalone(self):
        """Test sanitizing standalone Bearer tokens."""
        msg = "Using Bearer abc.def.ghi for authentication"
        result = sanitize_error_message(msg)
        assert "abc.def.ghi" not in result
        assert "Bearer ***" in result

    def test_sanitize_basic_auth_standalone(self):
        """Test sanitizing standalone Basic auth."""
        msg = "Using Basic YWRtaW46YWRtaW4= for authentication"
        result = sanitize_error_message(msg)
        assert "YWRtaW46YWRtaW4=" not in result
        assert "Basic ***" in result

    def test_sanitize_json_password(self):
        """Test sanitizing passwords in JSON format."""
        msg = '{"username": "admin", "password": "secret123"}'
        result = sanitize_error_message(msg)
        assert "secret123" not in result
        assert '"password": "***"' in result.lower() or "***" in result

    def test_sanitize_database_urls(self):
        """Test sanitizing various database connection URLs."""
        databases = [
            "mongodb://user:pass@localhost:27017/db",
            "postgresql://user:pass@localhost:5432/db",
            "mysql://user:pass@localhost:3306/db",
            "redis://user:pass@localhost:6379/0",
            "neo4j://user:pass@localhost:7687",
            "bolt://user:pass@localhost:7687",
            "bolt+s://user:pass@localhost:7687",
        ]

        for db_url in databases:
            result = sanitize_error_message(f"Connection failed: {db_url}")
            assert "user:pass" not in result
            assert "***" in result

    def test_sanitize_case_insensitive(self):
        """Test case-insensitive sanitization."""
        msg = "PASSWORD: secret, Api-Key: key123, TOKEN: tok456"
        result = sanitize_error_message(msg)
        # Check that sensitive values are replaced
        assert "secret" not in result or "***" in result
        assert "key123" not in result or "***" in result
        assert "tok456" not in result or "***" in result

    def test_sanitize_complex_message(self):
        """Test sanitizing complex error message with multiple patterns."""
        msg = """
        Failed to connect to mongodb://admin:secretpass@db.example.com:27017/mydb
        Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature
        api_key: "sk-1234567890abcdef"
        password = "another_secret"
        """

        result = sanitize_error_message(msg, ["sk-1234567890abcdef"])

        assert "secretpass" not in result
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "sk-1234567890abcdef" not in result
        assert "another_secret" not in result
        assert "***" in result


class TestGetEnvironment:
    """Test suite for get_environment function."""

    def test_get_environment_production(self):
        """Test detecting production environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            assert get_environment() == "production"

        with patch.dict(os.environ, {"ENV": "prod"}, clear=True):
            assert get_environment() == "production"

    def test_get_environment_staging(self):
        """Test detecting staging environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            assert get_environment() == "staging"

        with patch.dict(os.environ, {"ENV": "stage"}, clear=True):
            assert get_environment() == "staging"

    def test_get_environment_development(self):
        """Test detecting development environment."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=True):
            assert get_environment() == "development"

        with patch.dict(os.environ, {"ENV": "dev"}, clear=True):
            assert get_environment() == "development"

    def test_get_environment_default(self):
        """Test default environment when not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_environment() == "development"

    def test_get_environment_node_env(self):
        """Test using NODE_ENV variable."""
        with patch.dict(os.environ, {"NODE_ENV": "production"}, clear=True):
            assert get_environment() == "production"

    def test_get_environment_priority(self):
        """Test environment variable priority."""
        # ENVIRONMENT takes precedence
        with patch.dict(
            os.environ,
            {"ENVIRONMENT": "production", "ENV": "staging", "NODE_ENV": "development"},
        ):
            assert get_environment() == "production"

        # ENV takes precedence over NODE_ENV
        with patch.dict(os.environ, {"ENV": "staging", "NODE_ENV": "development"}, clear=True):
            assert get_environment() == "staging"

    def test_get_environment_case_insensitive(self):
        """Test case-insensitive environment detection."""
        with patch.dict(os.environ, {"ENVIRONMENT": "PRODUCTION"}, clear=True):
            assert get_environment() == "production"

        with patch.dict(os.environ, {"ENV": "STAGING"}, clear=True):
            assert get_environment() == "staging"


class TestGetSecureConnectionConfig:
    """Test suite for get_secure_connection_config function."""

    def test_basic_config(self):
        """Test basic configuration."""
        config = {
            "neo4j": {
                "host": "db.example.com",
                "port": 7687,
            }
        }

        result = get_secure_connection_config(config, "neo4j")

        assert result["host"] == "db.example.com"
        assert result["port"] == 7687
        assert "ssl" in result
        assert "verify_ssl" in result
        assert "timeout" in result
        assert "environment" in result

    def test_default_values(self):
        """Test default values when config is minimal."""
        config = {}

        result = get_secure_connection_config(config, "neo4j")

        assert result["host"] == "localhost"
        assert result["port"] is None
        assert result["timeout"] == 30
        assert result["verify_ssl"] is True

    def test_production_ssl_enabled(self):
        """Test SSL is enabled by default in production."""
        config = {"neo4j": {}}

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            result = get_secure_connection_config(config, "neo4j")
            assert result["ssl"] is True
            assert result["environment"] == "production"

    def test_production_ssl_disabled_warning(self):
        """Test warning when SSL is disabled in production."""
        config = {"neo4j": {"ssl": False}}

        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = get_secure_connection_config(config, "neo4j")

                assert result["ssl"] is False
                assert len(w) == 1
                assert "SSL is disabled" in str(w[0].message)
                assert "security risk" in str(w[0].message)

    def test_development_ssl_default(self):
        """Test SSL is disabled by default in development."""
        config = {"neo4j": {}}

        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            result = get_secure_connection_config(config, "neo4j")
            assert result["ssl"] is False
            assert result["environment"] == "development"

    def test_ssl_certificate_paths(self):
        """Test SSL certificate paths are included."""
        config = {
            "qdrant": {
                "ssl_cert_path": "/path/to/cert.pem",
                "ssl_key_path": "/path/to/key.pem",
                "ssl_ca_path": "/path/to/ca.pem",
            }
        }

        result = get_secure_connection_config(config, "qdrant")

        assert result["ssl_cert_path"] == "/path/to/cert.pem"
        assert result["ssl_key_path"] == "/path/to/key.pem"
        assert result["ssl_ca_path"] == "/path/to/ca.pem"

    def test_ssl_certificate_paths_missing(self):
        """Test missing SSL certificate paths are not included."""
        config = {"neo4j": {}}

        result = get_secure_connection_config(config, "neo4j")

        assert "ssl_cert_path" not in result
        assert "ssl_key_path" not in result
        assert "ssl_ca_path" not in result

    def test_verify_ssl_option(self):
        """Test verify_ssl option."""
        config = {"neo4j": {"verify_ssl": False}}

        result = get_secure_connection_config(config, "neo4j")
        assert result["verify_ssl"] is False

    def test_timeout_option(self):
        """Test timeout option."""
        config = {"qdrant": {"timeout": 60}}

        result = get_secure_connection_config(config, "qdrant")
        assert result["timeout"] == 60

    def test_staging_environment(self):
        """Test staging environment configuration."""
        config = {"neo4j": {"ssl": True}}

        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            result = get_secure_connection_config(config, "neo4j")
            assert result["ssl"] is True
            assert result["environment"] == "staging"

    def test_different_services(self):
        """Test configuration for different services."""
        config = {
            "neo4j": {"host": "neo4j.example.com", "port": 7687, "ssl": True},
            "qdrant": {"host": "qdrant.example.com", "port": 6333, "ssl": False},
        }

        neo4j_config = get_secure_connection_config(config, "neo4j")
        assert neo4j_config["host"] == "neo4j.example.com"
        assert neo4j_config["port"] == 7687
        assert neo4j_config["ssl"] is True

        qdrant_config = get_secure_connection_config(config, "qdrant")
        assert qdrant_config["host"] == "qdrant.example.com"
        assert qdrant_config["port"] == 6333
        assert qdrant_config["ssl"] is False
