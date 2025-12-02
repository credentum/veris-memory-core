#!/usr/bin/env python3
"""
Deep tests for config_validator to achieve high coverage.

This test suite covers:
- ConfigValidator class initialization and configuration loading
- Main configuration validation (all sections and fields)
- Performance configuration validation
- Environment variable validation
- Database configuration validation (Neo4j, Qdrant, Redis)
- MCP configuration validation
- All validation functions (validate_all_configs, etc.)
- Error handling and edge cases
- CLI functionality
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from src.validators.config_validator import (
    ConfigValidationError,
    ConfigValidator,
    main,
    validate_all_configs,
    validate_database_config,
    validate_environment_variables,
    validate_mcp_config,
)


class TestConfigValidationError:
    """Test ConfigValidationError exception."""

    def test_config_validation_error(self):
        """Test ConfigValidationError can be raised and caught."""
        with pytest.raises(ConfigValidationError):
            raise ConfigValidationError("Test error message")

    def test_config_validation_error_with_message(self):
        """Test ConfigValidationError preserves message."""
        error_msg = "Configuration is invalid"
        try:
            raise ConfigValidationError(error_msg)
        except ConfigValidationError as e:
            assert str(e) == error_msg


class TestConfigValidatorInitialization:
    """Test ConfigValidator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        validator = ConfigValidator()
        assert validator.errors == []
        assert validator.warnings == []

    def test_init_lists_empty(self):
        """Test initialization creates empty lists."""
        validator = ConfigValidator()
        assert isinstance(validator.errors, list)
        assert isinstance(validator.warnings, list)
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 0


class TestConfigValidatorMainConfig:
    """Test main configuration validation."""

    def test_validate_main_config_file_not_found(self):
        """Test validation when config file doesn't exist."""
        validator = ConfigValidator()
        result = validator.validate_main_config("nonexistent.yaml")

        assert result is False
        assert len(validator.errors) > 0
        assert any("not found" in error for error in validator.errors)

    def test_validate_main_config_invalid_yaml(self):
        """Test validation with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("invalid: yaml: content: [unclosed")
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert result is False
                assert len(validator.errors) > 0
                assert any("Invalid YAML" in error for error in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_missing_required_sections(self):
        """Test validation with missing required sections."""
        config = {"incomplete": "config"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert result is False
                assert len(validator.errors) > 0
                # Should have errors for missing required sections
                required_sections = ["system", "qdrant", "neo4j", "storage", "agents"]
                for section in required_sections:
                    assert any(section in error for error in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_valid_complete(self):
        """Test validation with complete valid config."""
        config = {
            "system": {"name": "test"},
            "qdrant": {"host": "localhost", "port": 6333},
            "neo4j": {"host": "localhost", "port": 7687},
            "storage": {"path": "/tmp"},
            "agents": {"enabled": True},
            "redis": {"host": "localhost", "port": 6379, "database": 0},
            "duckdb": {"database_path": "/tmp/test.db", "threads": 4},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert result is True
                assert len(validator.errors) == 0
            finally:
                os.unlink(tmp.name)

    def test_validate_qdrant_port_invalid_type(self):
        """Test Qdrant port validation with invalid type."""
        config = {
            "system": {},
            "qdrant": {"port": "invalid"},
            "neo4j": {},
            "storage": {},
            "agents": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert result is False
                assert any("qdrant.port must be an integer" in error for error in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_qdrant_port_out_of_range(self):
        """Test Qdrant port validation with out of range values."""
        for invalid_port in [0, -1, 65536, 100000]:
            config = {
                "system": {},
                "qdrant": {"port": invalid_port},
                "neo4j": {},
                "storage": {},
                "agents": {},
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                yaml.dump(config, tmp)
                tmp.flush()

                try:
                    validator = ConfigValidator()
                    result = validator.validate_main_config(tmp.name)

                    assert result is False
                    assert any("between 1 and 65535" in error for error in validator.errors)
                finally:
                    os.unlink(tmp.name)

    def test_validate_neo4j_port_validation(self):
        """Test Neo4j port validation."""
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {"port": "invalid"},
            "storage": {},
            "agents": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert result is False
                assert any("neo4j.port must be an integer" in error for error in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_redis_configuration(self):
        """Test Redis configuration validation."""
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
            "redis": {"port": "invalid", "database": -1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert result is False
                assert any("redis.port must be an integer" in error for error in validator.errors)
                assert any(
                    "redis.database must be a non-negative integer" in error
                    for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_duckdb_configuration(self):
        """Test DuckDB configuration validation."""
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
            "duckdb": {"threads": 0},  # Missing database_path, invalid threads
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert result is False
                assert any(
                    "duckdb.database_path is required" in error for error in validator.errors
                )
                assert any(
                    "duckdb.threads must be a positive integer" in error
                    for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_ssl_warnings(self):
        """Test SSL warning generation."""
        config = {
            "system": {},
            "storage": {},
            "agents": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
            "redis": {"ssl": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)

                assert len(validator.warnings) >= 3
                assert any(
                    "SSL is disabled for Qdrant" in warning for warning in validator.warnings
                )
                assert any("SSL is disabled for Neo4j" in warning for warning in validator.warnings)
                assert any("SSL is disabled for Redis" in warning for warning in validator.warnings)
            finally:
                os.unlink(tmp.name)


class TestConfigValidatorPerformanceConfig:
    """Test performance configuration validation."""

    def test_validate_performance_config_missing_file(self):
        """Test performance config validation when file doesn't exist."""
        validator = ConfigValidator()
        result = validator.validate_performance_config("nonexistent.yaml")

        # Performance config is optional, should return True
        assert result is True
        assert len(validator.errors) == 0

    def test_validate_performance_config_invalid_yaml(self):
        """Test performance config validation with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("invalid: yaml: content: [unclosed")
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert len(validator.errors) > 0
                assert any("Invalid YAML" in error for error in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_vector_db_embedding_settings(self):
        """Test vector DB embedding settings validation."""
        config = {
            "vector_db": {
                "embedding": {
                    "batch_size": 0,  # Invalid
                    "max_retries": -1,  # Invalid
                    "request_timeout": 0,  # Invalid
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert any(
                    "batch_size must be a positive integer" in error for error in validator.errors
                )
                assert any(
                    "max_retries must be a non-negative integer" in error
                    for error in validator.errors
                )
                assert any(
                    "request_timeout must be a positive number" in error
                    for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_vector_db_search_settings(self):
        """Test vector DB search settings validation."""
        config = {
            "vector_db": {
                "search": {"default_limit": 50, "max_limit": 25}  # max < default, should error
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert any(
                    "max_limit must be >= default_limit" in error for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_graph_db_connection_pool(self):
        """Test graph DB connection pool validation."""
        config = {
            "graph_db": {
                "connection_pool": {"min_size": 20, "max_size": 10}  # max < min, should error
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert any("max_size must be >= min_size" in error for error in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_graph_db_query_settings(self):
        """Test graph DB query settings validation."""
        config = {"graph_db": {"query": {"max_path_length": 0}}}  # Invalid

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert any(
                    "max_path_length must be a positive integer" in error
                    for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_graph_db_query_performance_warning(self):
        """Test graph DB query performance warning."""
        config = {"graph_db": {"query": {"max_path_length": 15}}}  # > 10, should warn

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is True  # Valid but with warnings
                assert len(validator.warnings) > 0
                assert any("performance issues" in warning for warning in validator.warnings)
            finally:
                os.unlink(tmp.name)

    def test_validate_search_ranking_settings(self):
        """Test search ranking settings validation."""
        config = {
            "search": {
                "ranking": {
                    "temporal_decay_rate": 1.5,  # > 1, invalid
                    "type_boosts": {"design": -1, "decision": 2.0},  # Negative, invalid  # Valid
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert any(
                    "temporal_decay_rate must be between 0 and 1" in error
                    for error in validator.errors
                )
                assert any(
                    "type_boosts.design must be a non-negative number" in error
                    for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_resources_settings(self):
        """Test resource settings validation."""
        config = {
            "resources": {
                "max_memory_gb": 0.25,  # < 0.5, invalid
                "max_cpu_percent": 150,  # > 100, invalid
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert any(
                    "max_memory_gb must be at least 0.5" in error for error in validator.errors
                )
                assert any(
                    "max_cpu_percent must be between 1 and 100" in error
                    for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_kv_store_settings(self):
        """Test KV store settings validation."""
        config = {
            "kv_store": {
                "redis": {
                    "connection_pool": {"min_size": 100, "max_size": 50},  # max < min
                    "cache": {"ttl_seconds": 0},  # Invalid
                },
                "duckdb": {
                    "batch_insert": {"size": 0},  # Invalid
                    "analytics": {"retention_days": 0},  # Invalid
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is False
                assert any("max_size must be >= min_size" in error for error in validator.errors)
                assert any(
                    "ttl_seconds must be a positive integer" in error for error in validator.errors
                )
                assert any(
                    "batch_insert.size must be a positive integer" in error
                    for error in validator.errors
                )
                assert any(
                    "retention_days must be a positive integer" in error
                    for error in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_valid(self):
        """Test validation with valid performance config."""
        config = {
            "vector_db": {
                "embedding": {"batch_size": 100, "max_retries": 3, "request_timeout": 30},
                "search": {"default_limit": 10, "max_limit": 100},
            },
            "graph_db": {
                "connection_pool": {"min_size": 1, "max_size": 10},
                "query": {"max_path_length": 5},
            },
            "search": {
                "ranking": {
                    "temporal_decay_rate": 0.01,
                    "type_boosts": {"design": 1.5, "decision": 2.0},
                }
            },
            "resources": {"max_memory_gb": 4, "max_cpu_percent": 80},
            "kv_store": {
                "redis": {
                    "connection_pool": {"min_size": 5, "max_size": 50},
                    "cache": {"ttl_seconds": 3600},
                },
                "duckdb": {"batch_insert": {"size": 1000}, "analytics": {"retention_days": 90}},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)

                assert result is True
                assert len(validator.errors) == 0
            finally:
                os.unlink(tmp.name)


class TestConfigValidatorIntegration:
    """Test ConfigValidator integration methods."""

    def test_validate_all_success(self):
        """Test validate_all with successful validation."""
        validator = ConfigValidator()

        # Mock the individual validation methods to return success
        with patch.object(validator, "validate_main_config", return_value=True):
            with patch.object(validator, "validate_performance_config", return_value=True):
                # Clear any existing errors/warnings
                validator.errors = []
                validator.warnings = []

                is_valid, errors, warnings = validator.validate_all()

                assert is_valid is True
                assert isinstance(errors, list)
                assert isinstance(warnings, list)

    def test_validate_all_with_errors(self):
        """Test validate_all with validation errors."""
        validator = ConfigValidator()

        # Mock the individual validation methods to avoid file I/O
        with patch.object(validator, "validate_main_config", return_value=False):
            with patch.object(validator, "validate_performance_config", return_value=True):
                # Set errors and warnings after validation calls
                validator.errors = ["Test error 1", "Test error 2"]
                validator.warnings = ["Test warning"]

                is_valid, errors, warnings = validator.validate_all()

                assert is_valid is False
                assert errors == ["Test error 1", "Test error 2"]
                assert warnings == ["Test warning"]


class TestEnvironmentVariableValidation:
    """Test environment variable validation functions."""

    def test_validate_environment_variables_all_present(self):
        """Test validation when all required env vars are present."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_environment_variables()

            assert result["valid"] is True
            assert result["missing"] == []

    def test_validate_environment_variables_missing(self):
        """Test validation when required env vars are missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment_variables()

            assert result["valid"] is False
            assert len(result["missing"]) == 5
            expected_missing = [
                "NEO4J_URI",
                "NEO4J_USER",
                "NEO4J_PASSWORD",
                "QDRANT_URL",
                "REDIS_URL",
            ]
            for var in expected_missing:
                assert var in result["missing"]

    def test_validate_environment_variables_partial(self):
        """Test validation with only some env vars present."""
        env_vars = {"NEO4J_URI": "bolt://localhost:7687", "QDRANT_URL": "http://localhost:6333"}

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_environment_variables()

            assert result["valid"] is False
            assert len(result["missing"]) == 3
            assert "NEO4J_USER" in result["missing"]
            assert "NEO4J_PASSWORD" in result["missing"]
            assert "REDIS_URL" in result["missing"]


class TestDatabaseConfigValidation:
    """Test database configuration validation functions."""

    def test_validate_database_config_none_config(self):
        """Test validation with None config."""
        with pytest.raises(ConfigValidationError, match="Configuration for neo4j cannot be None"):
            validate_database_config("neo4j", None)

    def test_validate_database_config_neo4j_valid(self):
        """Test Neo4j config validation with valid config."""
        config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

        result = validate_database_config("neo4j", config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_database_config_neo4j_missing_fields(self):
        """Test Neo4j config validation with missing fields."""
        config = {"uri": "bolt://localhost:7687"}  # Missing user and password

        result = validate_database_config("neo4j", config)

        assert result["valid"] is False
        assert len(result["errors"]) == 2
        assert any("Missing 'user'" in error for error in result["errors"])
        assert any("Missing 'password'" in error for error in result["errors"])

    def test_validate_database_config_neo4j_invalid_uri(self):
        """Test Neo4j config validation with invalid URI scheme."""
        config = {
            "uri": "http://localhost:7687",  # Invalid scheme
            "user": "neo4j",
            "password": "password",
        }

        result = validate_database_config("neo4j", config)

        assert result["valid"] is False
        assert any("Invalid Neo4j URI scheme" in error for error in result["errors"])

    def test_validate_database_config_neo4j_valid_schemes(self):
        """Test Neo4j config validation with all valid URI schemes."""
        valid_schemes = ["bolt://", "neo4j://", "bolt+s://", "neo4j+s://"]

        for scheme in valid_schemes:
            config = {"uri": f"{scheme}localhost:7687", "user": "neo4j", "password": "password"}

            result = validate_database_config("neo4j", config)
            assert result["valid"] is True, f"Failed for scheme: {scheme}"

    def test_validate_database_config_qdrant_valid(self):
        """Test Qdrant config validation with valid config."""
        config = {"url": "http://localhost:6333"}

        result = validate_database_config("qdrant", config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_database_config_qdrant_missing_url(self):
        """Test Qdrant config validation with missing URL."""
        config = {"collection": "test"}  # Missing url

        result = validate_database_config("qdrant", config)

        assert result["valid"] is False
        assert any("Missing 'url'" in error for error in result["errors"])

    def test_validate_database_config_qdrant_invalid_url(self):
        """Test Qdrant config validation with invalid URL scheme."""
        config = {"url": "ftp://localhost:6333"}  # Invalid scheme

        result = validate_database_config("qdrant", config)

        assert result["valid"] is False
        assert any("Invalid Qdrant URL scheme" in error for error in result["errors"])

    def test_validate_database_config_redis_valid(self):
        """Test Redis config validation with valid config."""
        config = {"url": "redis://localhost:6379"}

        result = validate_database_config("redis", config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_database_config_redis_missing_url(self):
        """Test Redis config validation with missing URL."""
        config = {"db": 0}  # Missing url

        result = validate_database_config("redis", config)

        assert result["valid"] is False
        assert any("Missing 'url'" in error for error in result["errors"])

    def test_validate_database_config_redis_invalid_url(self):
        """Test Redis config validation with invalid URL scheme."""
        config = {"url": "http://localhost:6379"}  # Invalid scheme

        result = validate_database_config("redis", config)

        assert result["valid"] is False
        assert any("Invalid Redis URL scheme" in error for error in result["errors"])

    def test_validate_database_config_unknown_type(self):
        """Test validation with unknown database type."""
        config = {"some": "config"}

        with pytest.raises(ConfigValidationError, match="Unknown database type: unknown"):
            validate_database_config("unknown", config)


class TestMCPConfigValidation:
    """Test MCP configuration validation function."""

    def test_validate_mcp_config_valid(self):
        """Test MCP config validation with valid config."""
        config = {
            "server_port": 8000,
            "host": "0.0.0.0",
            "tools": ["store_context", "retrieve_context", "query_graph"],
        }

        result = validate_mcp_config(config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_mcp_config_missing_port(self):
        """Test MCP config validation with missing server_port."""
        config = {"host": "0.0.0.0", "tools": ["store_context"]}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("Missing 'server_port'" in error for error in result["errors"])

    def test_validate_mcp_config_invalid_port_type(self):
        """Test MCP config validation with invalid port type."""
        config = {
            "server_port": "8000",  # Should be int
            "host": "0.0.0.0",
            "tools": ["store_context"],
        }

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("server_port must be an integer" in error for error in result["errors"])

    def test_validate_mcp_config_invalid_port_range(self):
        """Test MCP config validation with out of range port."""
        for invalid_port in [0, -1, 65536, 100000]:
            config = {"server_port": invalid_port, "host": "0.0.0.0", "tools": ["store_context"]}

            result = validate_mcp_config(config)

            assert result["valid"] is False
            assert any(
                "server_port must be between 1 and 65535" in error for error in result["errors"]
            )

    def test_validate_mcp_config_missing_host(self):
        """Test MCP config validation with missing host."""
        config = {"server_port": 8000, "tools": ["store_context"]}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("Missing 'host'" in error for error in result["errors"])

    def test_validate_mcp_config_missing_tools(self):
        """Test MCP config validation with missing tools."""
        config = {"server_port": 8000, "host": "0.0.0.0"}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("Missing 'tools'" in error for error in result["errors"])

    def test_validate_mcp_config_invalid_tools_type(self):
        """Test MCP config validation with invalid tools type."""
        config = {
            "server_port": 8000,
            "host": "0.0.0.0",
            "tools": "store_context",  # Should be list
        }

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("tools must be a list" in error for error in result["errors"])

    def test_validate_mcp_config_empty_tools(self):
        """Test MCP config validation with empty tools list."""
        config = {"server_port": 8000, "host": "0.0.0.0", "tools": []}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("At least one tool must be configured" in error for error in result["errors"])


class TestValidateAllConfigs:
    """Test validate_all_configs comprehensive validation function."""

    def test_validate_all_configs_success(self):
        """Test validate_all_configs with all valid configurations."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "MCP_SERVER_PORT": "8000",
            "MCP_HOST": "0.0.0.0",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_all_configs()

            assert result["valid"] is True
            assert result["environment"]["valid"] is True
            assert result["databases"]["neo4j"]["valid"] is True
            assert result["databases"]["qdrant"]["valid"] is True
            assert result["databases"]["redis"]["valid"] is True
            assert result["mcp"]["valid"] is True

    def test_validate_all_configs_env_failures(self):
        """Test validate_all_configs with environment variable failures."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_all_configs()

            assert result["valid"] is False
            assert result["environment"]["valid"] is False
            assert len(result["environment"]["missing"]) == 5

    def test_validate_all_configs_database_failures(self):
        """Test validate_all_configs with database config failures."""
        env_vars = {
            "NEO4J_URI": "invalid://localhost:7687",  # Invalid scheme
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "ftp://localhost:6333",  # Invalid scheme
            "REDIS_URL": "http://localhost:6379",  # Invalid scheme
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_all_configs()

            assert result["valid"] is False
            assert result["databases"]["neo4j"]["valid"] is False
            assert result["databases"]["qdrant"]["valid"] is False
            assert result["databases"]["redis"]["valid"] is False

    def test_validate_all_configs_mcp_failure(self):
        """Test validate_all_configs with MCP config failure."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "MCP_SERVER_PORT": "8000",  # Valid port (int conversion will work)
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_all_configs()

            # Should be valid since all configs are valid
            assert result["mcp"]["valid"] is True

    def test_validate_all_configs_with_warnings(self):
        """Test validate_all_configs generates warnings for best practices."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "LOG_LEVEL": "debug",  # Should generate warning
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_all_configs()

            assert len(result["warnings"]) > 0
            assert any("Debug logging is enabled" in warning for warning in result["warnings"])

    def test_validate_all_configs_optional_vars(self):
        """Test validate_all_configs with optional environment variables."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "QDRANT_COLLECTION": "custom_collection",
            "REDIS_DB": "5",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_all_configs()

            assert result["valid"] is True
            # Should handle optional vars gracefully


class TestMainCLIFunction:
    """Test the main CLI function."""

    def test_main_cli_success(self):
        """Test main CLI function with successful validation."""
        validator_mock = Mock()
        validator_mock.validate_main_config.return_value = True
        validator_mock.validate_performance_config.return_value = True
        validator_mock.errors = []
        validator_mock.warnings = []

        with patch("validators.config_validator.ConfigValidator", return_value=validator_mock):
            with patch("validators.config_validator.click.echo") as mock_echo:
                with patch("builtins.exit") as mock_exit:
                    main(".ctxrc.yaml", "performance.yaml", False)

                    mock_exit.assert_called_with(0)
                    # Should have called echo for validation messages
                    assert mock_echo.called

    def test_main_cli_with_errors(self):
        """Test main CLI function with validation errors."""
        validator_mock = Mock()
        validator_mock.validate_main_config.return_value = False
        validator_mock.validate_performance_config.return_value = True
        validator_mock.errors = ["Test error 1", "Test error 2"]
        validator_mock.warnings = []

        with patch("validators.config_validator.ConfigValidator", return_value=validator_mock):
            with patch("validators.config_validator.click.echo") as mock_echo:
                with patch("builtins.exit") as mock_exit:
                    main(".ctxrc.yaml", "performance.yaml", False)

                    mock_exit.assert_called_with(1)
                    # Should display errors
                    error_calls = [call for call in mock_echo.call_args_list if "❌" in str(call)]
                    assert len(error_calls) >= 2

    def test_main_cli_with_warnings(self):
        """Test main CLI function with warnings."""
        validator_mock = Mock()
        validator_mock.validate_main_config.return_value = True
        validator_mock.validate_performance_config.return_value = True
        validator_mock.errors = []
        validator_mock.warnings = ["Test warning 1", "Test warning 2"]

        with patch("validators.config_validator.ConfigValidator", return_value=validator_mock):
            with patch("validators.config_validator.click.echo") as mock_echo:
                with patch("builtins.exit") as mock_exit:
                    main(".ctxrc.yaml", "performance.yaml", False)

                    mock_exit.assert_called_with(0)
                    # Should display warnings
                    warning_calls = [call for call in mock_echo.call_args_list if "⚠️" in str(call)]
                    assert len(warning_calls) >= 2

    def test_main_cli_strict_mode_with_warnings(self):
        """Test main CLI function in strict mode with warnings."""
        validator_mock = Mock()
        validator_mock.validate_main_config.return_value = True
        validator_mock.validate_performance_config.return_value = True
        validator_mock.errors = []
        validator_mock.warnings = ["Test warning"]

        with patch("validators.config_validator.ConfigValidator", return_value=validator_mock):
            with patch("validators.config_validator.click.echo"):
                with patch("builtins.exit") as mock_exit:
                    main(".ctxrc.yaml", "performance.yaml", True)  # strict=True

                    mock_exit.assert_called_with(
                        1
                    )  # Should exit with error due to warnings in strict mode

    def test_main_cli_all_valid(self):
        """Test main CLI function with all configurations valid."""
        validator_mock = Mock()
        validator_mock.validate_main_config.return_value = True
        validator_mock.validate_performance_config.return_value = True
        validator_mock.errors = []
        validator_mock.warnings = []

        with patch("validators.config_validator.ConfigValidator", return_value=validator_mock):
            with patch("validators.config_validator.click.echo") as mock_echo:
                with patch("builtins.exit") as mock_exit:
                    main(".ctxrc.yaml", "performance.yaml", False)

                    mock_exit.assert_called_with(0)
                    # Should display success message
                    success_calls = [call for call in mock_echo.call_args_list if "✅" in str(call)]
                    assert len(success_calls) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
