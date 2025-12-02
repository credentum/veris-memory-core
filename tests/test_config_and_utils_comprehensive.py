"""
Comprehensive config and utils tests for maximum coverage improvement.

Tests Config class, utils functions, and related functionality with comprehensive
coverage of all methods, error conditions, and edge cases.
"""

import tempfile
from unittest.mock import patch

import pytest
import yaml

from src.core.config import Config, ConfigurationError, get_config, reload_config
from src.core.utils import get_environment, get_secure_connection_config, sanitize_error_message


class TestConfig:
    """Comprehensive tests for Config class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, "test_config.yaml")

        # Create comprehensive test config
        self.test_config = {
            "embedding": {"model": "test-embedding-model", "dimensions": 512, "batch_size": 50},
            "neo4j": {
                "host": "localhost",
                "port": 7687,
                "database": "test_graph",
                "username": "neo4j",
                "password": "test_password",
            },
            "redis": {"host": "localhost", "port": 6379, "database": 0},
            "qdrant": {"host": "localhost", "port": 6333, "collection_name": "test_collection"},
            "security": {"max_query_length": 5000, "query_timeout": 15},
        }

        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_config_class_constants(self):
        """Test Config class constants and defaults."""
        assert Config.EMBEDDING_DIMENSIONS == 1536
        assert Config.EMBEDDING_BATCH_SIZE == 100
        assert Config.EMBEDDING_MAX_RETRIES == 3
        assert Config.NEO4J_DEFAULT_PORT == 7687
        assert Config.QDRANT_DEFAULT_PORT == 6333
        assert Config.REDIS_DEFAULT_PORT == 6379
        assert Config.CONNECTION_POOL_MIN_SIZE == 5
        assert Config.CONNECTION_POOL_MAX_SIZE == 20
        assert Config.RATE_LIMIT_REQUESTS_PER_MINUTE == 60
        assert Config.MAX_QUERY_LENGTH == 10000
        assert Config.QUERY_TIMEOUT_SECONDS == 30

    def test_config_allowed_operations(self):
        """Test allowed and forbidden Cypher operations."""
        allowed_ops = Config.ALLOWED_CYPHER_OPERATIONS
        forbidden_ops = Config.FORBIDDEN_CYPHER_OPERATIONS

        assert "MATCH" in allowed_ops
        assert "RETURN" in allowed_ops
        assert "WHERE" in allowed_ops
        assert "CREATE" in forbidden_ops
        assert "DELETE" in forbidden_ops
        assert "DROP" in forbidden_ops

    def test_load_from_file_success(self):
        """Test successful config loading from file."""
        result = Config.load_from_file(self.config_path)

        # Check that our custom values are present (config gets merged with defaults)
        assert result["embedding"]["model"] == "test-embedding-model"
        assert result["embedding"]["dimensions"] == 512  # Our custom value
        assert result["neo4j"]["port"] == 7687
        assert result["neo4j"]["database"] == "test_graph"

        # Check that defaults are also present
        assert "databases" in result  # From defaults
        assert "cache" in result  # From defaults

    def test_load_from_file_not_found(self):
        """Test config loading when file not found."""
        result = Config.load_from_file("nonexistent.yaml")

        # Should return defaults when file not found
        assert isinstance(result, dict)
        assert "embedding" in result

    def test_load_from_file_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        invalid_config_path = os.path.join(self.test_config_dir, "invalid.yaml")
        with open(invalid_config_path, "w") as f:
            f.write("invalid: yaml: content: [\n")

        with pytest.raises(ConfigurationError):
            Config.load_from_file(invalid_config_path)

    def test_load_from_file_none_path(self):
        """Test config loading with None path."""
        with patch.dict("os.environ", {}, clear=True):
            result = Config.load_from_file(None)

            # Should return defaults when config file doesn't exist
            assert isinstance(result, dict)
            assert "embedding" in result

    def test_get_defaults(self):
        """Test get_defaults method."""
        defaults = Config.get_defaults()

        assert isinstance(defaults, dict)
        assert "embedding" in defaults
        assert "databases" in defaults
        assert "rate_limiting" in defaults
        assert "security" in defaults

        # Check specific values
        assert defaults["embedding"]["dimensions"] == 1536
        assert defaults["databases"]["neo4j"]["port"] == 7687
        assert defaults["databases"]["neo4j"]["connection_pool"]["min_size"] == 5
        assert defaults["rate_limiting"]["requests_per_minute"] == 60
        assert defaults["security"]["max_query_length"] == 10000

    def test_deep_merge_simple(self):
        """Test _deep_merge with simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = Config._deep_merge(base, override)

        expected = {"a": 1, "b": 3, "c": 4}
        assert result == expected

    def test_deep_merge_nested(self):
        """Test _deep_merge with nested dictionaries."""
        base = {
            "database": {"host": "localhost", "port": 5432, "options": {"ssl": False}},
            "other": "value",
        }
        override = {"database": {"port": 5433, "options": {"ssl": True, "timeout": 30}}}

        result = Config._deep_merge(base, override)

        expected = {
            "database": {
                "host": "localhost",
                "port": 5433,
                "options": {"ssl": True, "timeout": 30},
            },
            "other": "value",
        }
        assert result == expected

    def test_deep_merge_empty_dicts(self):
        """Test _deep_merge with empty dictionaries."""
        assert Config._deep_merge({}, {}) == {}
        assert Config._deep_merge({"a": 1}, {}) == {"a": 1}
        assert Config._deep_merge({}, {"a": 1}) == {"a": 1}

    def test_deep_merge_non_dict_override(self):
        """Test _deep_merge when override value is not a dict."""
        base = {"nested": {"a": 1, "b": 2}}
        override = {"nested": "replaced"}

        result = Config._deep_merge(base, override)

        assert result == {"nested": "replaced"}

    def test_validate_configuration_valid(self):
        """Test configuration validation with valid config."""
        valid_config = {
            "embedding": {"model": "text-embedding-ada-002", "dimensions": 1536},
            "neo4j": {"host": "localhost", "port": 7687},
        }

        result = Config.validate_configuration(valid_config)
        assert result is True

    def test_validate_configuration_invalid_embedding_dimensions(self):
        """Test configuration validation with invalid embedding dimensions."""
        invalid_config = {"embedding": {"dimensions": 999}}  # Not in allowed list

        with pytest.raises(ConfigurationError):
            Config.validate_configuration(invalid_config)

    def test_validate_configuration_invalid_port(self):
        """Test configuration validation with invalid port."""
        invalid_config = {"databases": {"neo4j": {"port": "not_a_port"}}}

        with pytest.raises(ConfigurationError):
            Config.validate_configuration(invalid_config)

    def test_validate_configuration_out_of_range_port(self):
        """Test configuration validation with out-of-range port."""
        invalid_config = {"databases": {"redis": {"port": 70000}}}  # Too high

        with pytest.raises(ConfigurationError):
            Config.validate_configuration(invalid_config)

    def test_validate_configuration_empty(self):
        """Test configuration validation with empty config."""
        result = Config.validate_configuration({})
        assert result is True  # Empty config should be valid

    def test_get_config_function(self):
        """Test get_config function."""
        with patch.object(Config, "load_from_file") as mock_load:
            mock_load.return_value = {"test": "config"}

            result = get_config()

            assert result["test"] == "config"

    def test_reload_config_function(self):
        """Test reload_config function."""
        with patch.object(Config, "load_from_file") as mock_load:
            mock_load.return_value = {"reloaded": "config"}

            result = reload_config(self.config_path)

            assert result["reloaded"] == "config"
            mock_load.assert_called_with(self.config_path)

    def test_reload_config_no_path(self):
        """Test reload_config function without path."""
        with patch.object(Config, "load_from_file") as mock_load:
            mock_load.return_value = {"default": "config"}

            result = reload_config()

            assert result["default"] == "config"
            mock_load.assert_called_with(None)


class TestUtils:
    """Comprehensive tests for utils functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sanitize_error_message_basic(self):
        """Test basic error message sanitization."""
        error_msg = "Connection failed with password: secret123"
        result = sanitize_error_message(error_msg)

        assert "secret123" not in result
        assert "password: ***" in result

    def test_sanitize_error_message_with_sensitive_values(self):
        """Test error message sanitization with specific sensitive values."""
        error_msg = "Connection failed: mongodb://user:mypassword@localhost:27017/db"
        sensitive_values = ["mypassword"]

        result = sanitize_error_message(error_msg, sensitive_values)

        assert "mypassword" not in result
        assert "***" in result

    def test_sanitize_error_message_connection_strings(self):
        """Test sanitization of connection strings."""
        error_msg = "Failed to connect to mongodb://user:password@host:27017/db"
        result = sanitize_error_message(error_msg)

        assert "mongodb://***:***@***" in result

    def test_sanitize_error_message_auth_headers(self):
        """Test sanitization of auth headers."""
        error_msg = "Request failed: Authorization: Bearer eyJhbGciOiJIUzI1NiJ9"
        result = sanitize_error_message(error_msg)

        assert "Authorization: ***" in result

    def test_sanitize_error_message_empty(self):
        """Test sanitization of empty message."""
        result = sanitize_error_message("")
        assert result == ""

    def test_sanitize_error_message_none(self):
        """Test sanitization of None message."""
        result = sanitize_error_message(None)
        assert result is None

    def test_sanitize_error_message_url_encoded(self):
        """Test sanitization of URL encoded values."""
        error_msg = "Error with password%3Dsecret in URL"
        sensitive_values = ["password=secret"]

        result = sanitize_error_message(error_msg, sensitive_values)

        assert "***" in result

    def test_get_environment_production(self):
        """Test environment detection for production."""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "production"

            result = get_environment()

            assert result == "production"

    def test_get_environment_staging(self):
        """Test environment detection for staging."""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "staging"

            result = get_environment()

            assert result == "staging"

    def test_get_environment_development_default(self):
        """Test environment detection defaults to development."""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "development"

            result = get_environment()

            assert result == "development"

    def test_get_environment_variations(self):
        """Test environment detection with variations."""
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "prod"

            result = get_environment()

            assert result == "production"

    def test_get_secure_connection_config_basic(self):
        """Test basic secure connection config."""
        config = {"neo4j": {"host": "localhost", "port": 7687, "ssl": True}}

        result = get_secure_connection_config(config, "neo4j")

        assert result["host"] == "localhost"
        assert result["port"] == 7687
        assert result["ssl"] is True

    def test_get_secure_connection_config_missing_service(self):
        """Test secure connection config with missing service."""
        config = {"other_service": {"host": "localhost"}}

        result = get_secure_connection_config(config, "neo4j")

        assert result["host"] == "localhost"  # Default
        assert result["port"] is None

    def test_get_secure_connection_config_empty_config(self):
        """Test secure connection config with empty config."""
        result = get_secure_connection_config({}, "neo4j")

        assert result["host"] == "localhost"  # Default
        assert result["port"] is None

    def test_get_secure_connection_config_production_ssl(self):
        """Test secure connection config in production forces SSL."""
        config = {"neo4j": {"host": "prod.example.com", "port": 7687}}

        with patch("core.utils.get_environment") as mock_env:
            mock_env.return_value = "production"

            result = get_secure_connection_config(config, "neo4j")

            assert result["ssl"] is True  # Forced in production

    def test_get_secure_connection_config_ssl_certificates(self):
        """Test secure connection config with SSL certificates."""
        config = {
            "neo4j": {
                "host": "secure.example.com",
                "ssl_cert_path": "/path/to/cert.pem",
                "ssl_key_path": "/path/to/key.pem",
                "ssl_ca_path": "/path/to/ca.pem",
            }
        }

        result = get_secure_connection_config(config, "neo4j")

        assert result["ssl_cert_path"] == "/path/to/cert.pem"
        assert result["ssl_key_path"] == "/path/to/key.pem"
        assert result["ssl_ca_path"] == "/path/to/ca.pem"

    def test_get_secure_connection_config_timeout(self):
        """Test secure connection config with custom timeout."""
        config = {"neo4j": {"host": "slow.example.com", "timeout": 60}}

        result = get_secure_connection_config(config, "neo4j")

        assert result["timeout"] == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
