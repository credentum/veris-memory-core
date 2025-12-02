#!/usr/bin/env python3
"""
Comprehensive tests for config_validator module to achieve 100% coverage.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from src.validators.config_validator import (  # noqa: E402
    ConfigValidationError,
    ConfigValidator,
    main,
    validate_all_configs,
    validate_database_config,
    validate_environment_variables,
    validate_mcp_config,
)


class TestConfigValidator:
    """Test suite for ConfigValidator class."""

    def test_init(self):
        """Test ConfigValidator initialization."""
        validator = ConfigValidator()
        assert validator.errors == []
        assert validator.warnings == []

    def test_validate_main_config_success(self):
        """Test successful main config validation."""
        config = {
            "system": {},
            "qdrant": {"port": 6333},
            "neo4j": {"port": 7687},
            "storage": {},
            "agents": {},
            "redis": {"port": 6379, "database": 0},
            "duckdb": {"database_path": "/path/to/db", "threads": 4},
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

    def test_validate_main_config_file_not_found(self):
        """Test main config validation with missing file."""
        validator = ConfigValidator()
        result = validator.validate_main_config("/nonexistent/config.yaml")
        assert result is False
        assert len(validator.errors) == 1
        assert "not found" in validator.errors[0]

    def test_validate_main_config_invalid_yaml(self):
        """Test main config validation with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("invalid: yaml: content: [")
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)
                assert result is False
                assert len(validator.errors) == 1
                assert "Invalid YAML" in validator.errors[0]
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_missing_sections(self):
        """Test main config validation with missing required sections."""
        config = {"system": {}}  # Missing other required sections

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config(tmp.name)
                assert result is False
                assert any("Missing required section" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_invalid_qdrant_port(self):
        """Test validation with invalid Qdrant port."""
        config = {
            "system": {},
            "qdrant": {"port": "not_a_number"},
            "neo4j": {},
            "storage": {},
            "agents": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_main_config(tmp.name)
                assert any("qdrant.port must be an integer" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_qdrant_port_out_of_range(self):
        """Test validation with out of range Qdrant port."""
        config = {
            "system": {},
            "qdrant": {"port": 70000},
            "neo4j": {},
            "storage": {},
            "agents": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_main_config(tmp.name)
                assert any("qdrant.port must be between" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_invalid_neo4j_port(self):
        """Test validation with invalid Neo4j port."""
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
                validator.validate_main_config(tmp.name)
                assert any("neo4j.port must be an integer" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_neo4j_port_out_of_range(self):
        """Test validation with out of range Neo4j port."""
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {"port": 0},
            "storage": {},
            "agents": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_main_config(tmp.name)
                assert any("neo4j.port must be between" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_invalid_redis(self):
        """Test validation with invalid Redis configuration."""
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
                validator.validate_main_config(tmp.name)
                assert any("redis.port must be an integer" in e for e in validator.errors)
                assert any(
                    "redis.database must be a non-negative integer" in e for e in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_redis_port_out_of_range(self):
        """Test validation with out of range Redis port."""
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
            "redis": {"port": 100000},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_main_config(tmp.name)
                assert any("redis.port must be between" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_invalid_duckdb(self):
        """Test validation with invalid DuckDB configuration."""
        config = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
            "duckdb": {"threads": 0},  # Missing database_path and invalid threads
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_main_config(tmp.name)
                assert any("duckdb.database_path is required" in e for e in validator.errors)
                assert any(
                    "duckdb.threads must be a positive integer" in e for e in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config_ssl_warnings(self):
        """Test SSL warnings in configuration."""
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
                validator.validate_main_config(tmp.name)
                assert any("SSL is disabled for Qdrant" in w for w in validator.warnings)
                assert any("SSL is disabled for Neo4j" in w for w in validator.warnings)
                assert any("SSL is disabled for Redis" in w for w in validator.warnings)
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_success(self):
        """Test successful performance config validation."""
        config = {
            "vector_db": {
                "embedding": {
                    "batch_size": 100,
                    "max_retries": 3,
                    "request_timeout": 30,
                },
                "search": {"default_limit": 10, "max_limit": 100},
            },
            "graph_db": {
                "connection_pool": {"min_size": 1, "max_size": 10},
                "query": {"max_path_length": 5},
            },
            "search": {
                "ranking": {
                    "temporal_decay_rate": 0.5,
                    "type_boosts": {"documentation": 1.5, "code": 2.0},
                }
            },
            "resources": {"max_memory_gb": 4, "max_cpu_percent": 80},
            "kv_store": {
                "redis": {
                    "connection_pool": {"min_size": 5, "max_size": 50},
                    "cache": {"ttl_seconds": 3600},
                },
                "duckdb": {
                    "batch_insert": {"size": 1000},
                    "analytics": {"retention_days": 90},
                },
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

    def test_validate_performance_config_file_not_found(self):
        """Test performance config validation with missing file (optional)."""
        validator = ConfigValidator()
        result = validator.validate_performance_config("/nonexistent/perf.yaml")
        assert result is True  # Performance config is optional

    def test_validate_performance_config_invalid_yaml(self):
        """Test performance config validation with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("invalid: yaml: [")
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_performance_config(tmp.name)
                assert result is False
                assert any("Invalid YAML" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_embedding(self):
        """Test validation with invalid embedding configuration."""
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
                validator.validate_performance_config(tmp.name)
                assert any("batch_size must be a positive integer" in e for e in validator.errors)
                assert any(
                    "max_retries must be a non-negative integer" in e for e in validator.errors
                )
                assert any(
                    "request_timeout must be a positive number" in e for e in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_search(self):
        """Test validation with invalid search configuration."""
        config = {
            "vector_db": {
                "search": {
                    "default_limit": 100,
                    "max_limit": 10,
                }  # max_limit < default_limit
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any("max_limit must be >= default_limit" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_connection_pool(self):
        """Test validation with invalid connection pool configuration."""
        config = {
            "graph_db": {"connection_pool": {"min_size": 10, "max_size": 5}}  # max_size < min_size
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any("max_size must be >= min_size" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_query(self):
        """Test validation with invalid query configuration."""
        config = {"graph_db": {"query": {"max_path_length": 0}}}  # Invalid

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any(
                    "max_path_length must be a positive integer" in e for e in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_query_warning(self):
        """Test warning for large max_path_length."""
        config = {"graph_db": {"query": {"max_path_length": 15}}}  # > 10, should trigger warning

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any(
                    "max_path_length > 10 may cause performance issues" in w
                    for w in validator.warnings
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_ranking(self):
        """Test validation with invalid ranking configuration."""
        config = {
            "search": {
                "ranking": {
                    "temporal_decay_rate": 1.5,  # > 1
                    "type_boosts": {"documentation": -1},  # Negative boost
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any(
                    "temporal_decay_rate must be between 0 and 1" in e for e in validator.errors
                )
                assert any("must be a non-negative number" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_ranking_type(self):
        """Test validation with invalid ranking type."""
        config = {"search": {"ranking": {"temporal_decay_rate": "not_a_number"}}}  # Invalid type

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any("temporal_decay_rate must be a number" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_resources(self):
        """Test validation with invalid resources configuration."""
        config = {"resources": {"max_memory_gb": 0.2, "max_cpu_percent": 150}}  # < 0.5  # > 100

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any("max_memory_gb must be at least 0.5" in e for e in validator.errors)
                assert any(
                    "max_cpu_percent must be between 1 and 100" in e for e in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_redis_pool(self):
        """Test validation with invalid Redis pool configuration."""
        config = {
            "kv_store": {
                "redis": {
                    "connection_pool": {
                        "min_size": 50,
                        "max_size": 10,
                    },  # max_size < min_size
                    "cache": {"ttl_seconds": 0},  # Invalid
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any(
                    "redis.connection_pool.max_size must be >= min_size" in e
                    for e in validator.errors
                )
                assert any("ttl_seconds must be a positive integer" in e for e in validator.errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_performance_config_invalid_duckdb(self):
        """Test validation with invalid DuckDB configuration."""
        config = {
            "kv_store": {
                "duckdb": {
                    "batch_insert": {"size": 0},  # Invalid
                    "analytics": {"retention_days": 0},  # Invalid
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                validator.validate_performance_config(tmp.name)
                assert any(
                    "batch_insert.size must be a positive integer" in e for e in validator.errors
                )
                assert any(
                    "retention_days must be a positive integer" in e for e in validator.errors
                )
            finally:
                os.unlink(tmp.name)

    def test_validate_all(self):
        """Test validate_all method."""
        validator = ConfigValidator()

        # Create valid config files
        main_config = {
            "system": {},
            "qdrant": {},
            "neo4j": {},
            "storage": {},
            "agents": {},
        }
        perf_config = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as main_tmp:
            yaml.dump(main_config, main_tmp)
            main_tmp.flush()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as perf_tmp:
                yaml.dump(perf_config, perf_tmp)
                perf_tmp.flush()

                try:
                    # Mock the file paths
                    with patch.object(validator, "validate_main_config") as mock_main:
                        with patch.object(validator, "validate_performance_config") as mock_perf:
                            mock_main.return_value = True
                            mock_perf.return_value = True

                            valid, errors, warnings = validator.validate_all()

                            assert valid is True
                            assert errors == []
                            assert warnings == []

                            mock_main.assert_called_once_with()
                            mock_perf.assert_called_once_with()
                finally:
                    os.unlink(main_tmp.name)
                    os.unlink(perf_tmp.name)


class TestValidationFunctions:
    """Test suite for standalone validation functions."""

    def test_validate_environment_variables_all_present(self):
        """Test environment variable validation with all variables present."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "QDRANT_URL": "http://localhost:6333",
                "REDIS_URL": "redis://localhost:6379",
            },
        ):
            result = validate_environment_variables()
            assert result["valid"] is True
            assert result["missing"] == []

    def test_validate_environment_variables_missing(self):
        """Test environment variable validation with missing variables."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment_variables()
            assert result["valid"] is False
            assert len(result["missing"]) == 5

    def test_validate_database_config_neo4j_valid(self):
        """Test valid Neo4j configuration."""
        config = {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password",
        }
        result = validate_database_config("neo4j", config)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_database_config_neo4j_invalid_uri(self):
        """Test Neo4j configuration with invalid URI."""
        config = {
            "uri": "http://localhost:7687",  # Wrong scheme
            "user": "neo4j",
            "password": "password",
        }
        result = validate_database_config("neo4j", config)
        assert result["valid"] is False
        assert any("Invalid Neo4j URI scheme" in e for e in result["errors"])

    def test_validate_database_config_neo4j_missing_fields(self):
        """Test Neo4j configuration with missing fields."""
        config = {}
        result = validate_database_config("neo4j", config)
        assert result["valid"] is False
        assert any("Missing 'uri'" in e for e in result["errors"])
        assert any("Missing 'user'" in e for e in result["errors"])
        assert any("Missing 'password'" in e for e in result["errors"])

    def test_validate_database_config_neo4j_valid_schemes(self):
        """Test Neo4j configuration with valid URI schemes."""
        schemes = ["bolt://", "neo4j://", "bolt+s://", "neo4j+s://"]
        for scheme in schemes:
            config = {
                "uri": f"{scheme}localhost:7687",
                "user": "neo4j",
                "password": "password",
            }
            result = validate_database_config("neo4j", config)
            assert result["valid"] is True

    def test_validate_database_config_qdrant_valid(self):
        """Test valid Qdrant configuration."""
        config = {"url": "http://localhost:6333"}
        result = validate_database_config("qdrant", config)
        assert result["valid"] is True

    def test_validate_database_config_qdrant_https(self):
        """Test Qdrant configuration with HTTPS."""
        config = {"url": "https://localhost:6333"}
        result = validate_database_config("qdrant", config)
        assert result["valid"] is True

    def test_validate_database_config_qdrant_invalid_url(self):
        """Test Qdrant configuration with invalid URL."""
        config = {"url": "bolt://localhost:6333"}  # Wrong scheme
        result = validate_database_config("qdrant", config)
        assert result["valid"] is False
        assert any("Invalid Qdrant URL scheme" in e for e in result["errors"])

    def test_validate_database_config_qdrant_missing_url(self):
        """Test Qdrant configuration with missing URL."""
        config = {}
        result = validate_database_config("qdrant", config)
        assert result["valid"] is False
        assert any("Missing 'url'" in e for e in result["errors"])

    def test_validate_database_config_redis_valid(self):
        """Test valid Redis configuration."""
        config = {"url": "redis://localhost:6379"}
        result = validate_database_config("redis", config)
        assert result["valid"] is True

    def test_validate_database_config_redis_invalid_url(self):
        """Test Redis configuration with invalid URL."""
        config = {"url": "http://localhost:6379"}  # Wrong scheme
        result = validate_database_config("redis", config)
        assert result["valid"] is False
        assert any("Invalid Redis URL scheme" in e for e in result["errors"])

    def test_validate_database_config_redis_missing_url(self):
        """Test Redis configuration with missing URL."""
        config = {}
        result = validate_database_config("redis", config)
        assert result["valid"] is False
        assert any("Missing 'url'" in e for e in result["errors"])

    def test_validate_database_config_none(self):
        """Test database config validation with None config."""
        with pytest.raises(ConfigValidationError, match="cannot be None"):
            validate_database_config("neo4j", None)

    def test_validate_database_config_unknown_type(self):
        """Test database config validation with unknown database type."""
        with pytest.raises(ConfigValidationError, match="Unknown database type"):
            validate_database_config("unknown_db", {})

    def test_validate_mcp_config_valid(self):
        """Test valid MCP configuration."""
        config = {
            "server_port": 8000,
            "host": "0.0.0.0",
            "tools": ["store_context", "retrieve_context"],
        }
        result = validate_mcp_config(config)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_mcp_config_missing_fields(self):
        """Test MCP configuration with missing fields."""
        config = {}
        result = validate_mcp_config(config)
        assert result["valid"] is False
        assert any("Missing 'server_port'" in e for e in result["errors"])
        assert any("Missing 'host'" in e for e in result["errors"])
        assert any("Missing 'tools'" in e for e in result["errors"])

    def test_validate_mcp_config_invalid_port(self):
        """Test MCP configuration with invalid port."""
        config = {"server_port": "not_a_number", "host": "0.0.0.0", "tools": ["tool1"]}
        result = validate_mcp_config(config)
        assert result["valid"] is False
        assert any("server_port must be an integer" in e for e in result["errors"])

    def test_validate_mcp_config_port_out_of_range(self):
        """Test MCP configuration with out of range port."""
        config = {"server_port": 70000, "host": "0.0.0.0", "tools": ["tool1"]}
        result = validate_mcp_config(config)
        assert result["valid"] is False
        assert any("server_port must be between" in e for e in result["errors"])

    def test_validate_mcp_config_invalid_tools(self):
        """Test MCP configuration with invalid tools."""
        config = {
            "server_port": 8000,
            "host": "0.0.0.0",
            "tools": "not_a_list",
        }  # Should be a list
        result = validate_mcp_config(config)
        assert result["valid"] is False
        assert any("tools must be a list" in e for e in result["errors"])

    def test_validate_mcp_config_empty_tools(self):
        """Test MCP configuration with empty tools list."""
        config = {"server_port": 8000, "host": "0.0.0.0", "tools": []}
        result = validate_mcp_config(config)
        assert result["valid"] is False
        assert any("At least one tool must be configured" in e for e in result["errors"])

    def test_validate_all_configs_complete(self):
        """Test validate_all_configs with complete environment."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "QDRANT_URL": "http://localhost:6333",
                "QDRANT_COLLECTION": "test_collection",
                "REDIS_URL": "redis://localhost:6379",
                "REDIS_DB": "1",
                "MCP_SERVER_PORT": "8080",
                "MCP_HOST": "localhost",
                "LOG_LEVEL": "info",
            },
        ):
            result = validate_all_configs()
            assert result["valid"] is True
            assert "environment" in result
            assert "databases" in result
            assert "mcp" in result
            assert "warnings" in result

    def test_validate_all_configs_with_debug_warning(self):
        """Test validate_all_configs with debug logging warning."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "QDRANT_URL": "http://localhost:6333",
                "REDIS_URL": "redis://localhost:6379",
                "LOG_LEVEL": "debug",
            },
        ):
            result = validate_all_configs()
            assert any("Debug logging is enabled" in w for w in result["warnings"])

    def test_validate_all_configs_partial_environment(self):
        """Test validate_all_configs with partial environment."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                # Missing NEO4J_USER and NEO4J_PASSWORD
            },
            clear=True,
        ):
            result = validate_all_configs()
            assert result["valid"] is False
            assert not result["environment"]["valid"]

    def test_validate_all_configs_invalid_database(self):
        """Test validate_all_configs with invalid database config."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "invalid://localhost:7687",  # Invalid scheme
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "QDRANT_URL": "http://localhost:6333",
                "REDIS_URL": "redis://localhost:6379",
            },
        ):
            result = validate_all_configs()
            assert result["valid"] is False
            assert not result["databases"]["neo4j"]["valid"]


class TestMainFunction:
    """Test the main CLI function."""

    @patch("validators.config_validator.exit")
    @patch("validators.config_validator.click.echo")
    def test_main_all_valid(self, mock_echo, mock_exit):
        """Test main function with all valid configurations."""
        config = {"system": {}, "qdrant": {}, "neo4j": {}, "storage": {}, "agents": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                # Call main directly - it's a regular function decorated with @click.command
                main(config=tmp.name, perf_config="nonexistent.yaml", strict=False)

                # Should print success message
                assert any("✅" in str(call) for call in mock_echo.call_args_list)
                mock_exit.assert_called_with(0)
            finally:
                os.unlink(tmp.name)

    @patch("validators.config_validator.exit")
    @patch("validators.config_validator.click.echo")
    def test_main_with_errors(self, mock_echo, mock_exit):
        """Test main function with validation errors."""
        config = {}  # Missing required sections

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                main(config=tmp.name, perf_config="nonexistent.yaml", strict=False)

                # Should print errors
                assert any("❌" in str(call) for call in mock_echo.call_args_list)
                mock_exit.assert_called_with(1)
            finally:
                os.unlink(tmp.name)

    @patch("validators.config_validator.exit")
    @patch("validators.config_validator.click.echo")
    def test_main_with_warnings_non_strict(self, mock_echo, mock_exit):
        """Test main function with warnings in non-strict mode."""
        config = {
            "system": {},
            "storage": {},
            "agents": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                main(config=tmp.name, perf_config="nonexistent.yaml", strict=False)

                # Should print warnings
                assert any("⚠️" in str(call) for call in mock_echo.call_args_list)
                mock_exit.assert_called_with(0)  # Exit 0 in non-strict mode
            finally:
                os.unlink(tmp.name)

    @patch("validators.config_validator.exit")
    @patch("validators.config_validator.click.echo")
    def test_main_with_warnings_strict(self, mock_echo, mock_exit):
        """Test main function with warnings in strict mode."""
        config = {
            "system": {},
            "storage": {},
            "agents": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                main(config=tmp.name, perf_config="nonexistent.yaml", strict=True)

                # Should print warnings
                assert any("⚠️" in str(call) for call in mock_echo.call_args_list)
                mock_exit.assert_called_with(1)  # Exit 1 in strict mode
            finally:
                os.unlink(tmp.name)

    @patch("validators.config_validator.exit")
    @patch("validators.config_validator.click.echo")
    def test_main_entry_point(self, mock_echo, mock_exit):
        """Test main entry point."""
        config = {"system": {}, "qdrant": {}, "neo4j": {}, "storage": {}, "agents": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                # Test __main__ execution
                with patch("sys.argv", ["config_validator.py", "--config", tmp.name]):
                    with patch("validators.config_validator.__name__", "__main__"):
                        # This would normally trigger main() if module is run directly
                        pass
            finally:
                os.unlink(tmp.name)
