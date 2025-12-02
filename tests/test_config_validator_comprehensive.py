#!/usr/bin/env python3
"""
Comprehensive tests for config_validator.py to achieve high coverage.

This test suite covers:
- Main config validation with all scenarios
- Performance config validation
- Environment variable validation
- Database config validation (Neo4j, Qdrant, Redis)
- MCP config validation
- Error handling and edge cases
- All validation functions and paths
"""

import os
import tempfile
from typing import Any, Dict
from unittest.mock import patch

import pytest
import yaml

from src.validators.config_validator import (
    ConfigValidationError,
    ConfigValidator,
    validate_all_configs,
    validate_database_config,
    validate_environment_variables,
    validate_mcp_config,
)


class TestConfigValidator:
    """Test cases for ConfigValidator class."""

    @pytest.fixture
    def validator(self) -> "ConfigValidator":
        """Create a fresh validator instance."""
        return ConfigValidator()

    @pytest.fixture
    def valid_main_config(self) -> Dict[str, Any]:
        """Valid main configuration."""
        return {
            "system": {"name": "context-store"},
            "qdrant": {"host": "localhost", "port": 6333, "ssl": True},
            "neo4j": {"host": "localhost", "port": 7687, "ssl": True},
            "redis": {"host": "localhost", "port": 6379, "database": 0, "ssl": True},
            "duckdb": {"database_path": "/tmp/test.db", "threads": 4},
            "storage": {"type": "hybrid"},
            "agents": {"max_contexts": 1000},
        }

    @pytest.fixture
    def valid_performance_config(self) -> Dict[str, Any]:
        """Valid performance configuration."""
        return {
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
                    "type_boosts": {"context": 1.0, "decision": 1.5},
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

    def test_validator_initialization(self, validator: "ConfigValidator") -> None:
        """Test validator initializes correctly."""
        assert validator.errors == []
        assert validator.warnings == []

    def test_validate_main_config_success(
        self, validator: "ConfigValidator", valid_main_config: Dict[str, Any]
    ) -> None:
        """Test successful main config validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_main_config, f)
            f.flush()

            result = validator.validate_main_config(f.name)

            assert result is True
            assert len(validator.errors) == 0

        os.unlink(f.name)

    def test_validate_main_config_file_not_found(self, validator: "ConfigValidator") -> None:
        """Test main config validation with missing file."""
        result = validator.validate_main_config("nonexistent.yaml")

        assert result is False
        assert len(validator.errors) == 1
        assert "not found" in validator.errors[0]

    def test_validate_main_config_invalid_yaml(self, validator: "ConfigValidator") -> None:
        """Test main config validation with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [\n")
            f.flush()

            result = validator.validate_main_config(f.name)

            assert result is False
            assert len(validator.errors) == 1
            assert "Invalid YAML" in validator.errors[0]

        os.unlink(f.name)

    def test_validate_main_config_missing_required_sections(
        self, validator: "ConfigValidator"
    ) -> None:
        """Test main config validation with missing required sections."""
        incomplete_config = {"system": {"name": "test"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(incomplete_config, f)
            f.flush()

            result = validator.validate_main_config(f.name)

            assert result is False
            assert len(validator.errors) >= 4  # Missing qdrant, neo4j, storage, agents
            missing_sections = [
                err for err in validator.errors if "Missing required section" in err
            ]
            assert len(missing_sections) >= 4

        os.unlink(f.name)

    def test_validate_qdrant_port_validation(
        self, validator: "ConfigValidator", valid_main_config: Dict[str, Any]
    ) -> None:
        """Test Qdrant port validation."""
        # Test invalid port type
        config = valid_main_config.copy()
        config["qdrant"]["port"] = "not_a_number"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            result = validator.validate_main_config(f.name)

            assert result is False
            assert any("qdrant.port must be an integer" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_qdrant_port_range(
        self, validator: "ConfigValidator", valid_main_config: Dict[str, Any]
    ) -> None:
        """Test Qdrant port range validation."""
        # Test port out of range
        config = valid_main_config.copy()
        config["qdrant"]["port"] = 70000

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_main_config(f.name)

            assert any("qdrant.port must be between 1 and 65535" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_neo4j_port_validation(
        self, validator: "ConfigValidator", valid_main_config: Dict[str, Any]
    ) -> None:
        """Test Neo4j port validation."""
        config = valid_main_config.copy()
        config["neo4j"]["port"] = -1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_main_config(f.name)

            assert any("neo4j.port must be between 1 and 65535" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_redis_configuration(
        self, validator: "ConfigValidator", valid_main_config: Dict[str, Any]
    ) -> None:
        """Test Redis configuration validation."""
        config = valid_main_config.copy()
        config["redis"]["port"] = "invalid"
        config["redis"]["database"] = -1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_main_config(f.name)

            assert any("redis.port must be an integer" in err for err in validator.errors)
            assert any(
                "redis.database must be a non-negative integer" in err for err in validator.errors
            )

        os.unlink(f.name)

    def test_validate_duckdb_configuration(
        self, validator: "ConfigValidator", valid_main_config: Dict[str, Any]
    ) -> None:
        """Test DuckDB configuration validation."""
        config = valid_main_config.copy()
        del config["duckdb"]["database_path"]
        config["duckdb"]["threads"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_main_config(f.name)

            assert any("duckdb.database_path is required" in err for err in validator.errors)
            assert any(
                "duckdb.threads must be a positive integer" in err for err in validator.errors
            )

        os.unlink(f.name)

    def test_ssl_warnings(
        self, validator: "ConfigValidator", valid_main_config: Dict[str, Any]
    ) -> None:
        """Test SSL warning generation."""
        config = valid_main_config.copy()
        config["qdrant"]["ssl"] = False
        config["neo4j"]["ssl"] = False
        config["redis"]["ssl"] = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_main_config(f.name)

            assert len(validator.warnings) == 3
            assert any("SSL is disabled for Qdrant" in warn for warn in validator.warnings)
            assert any("SSL is disabled for Neo4j" in warn for warn in validator.warnings)
            assert any("SSL is disabled for Redis" in warn for warn in validator.warnings)

        os.unlink(f.name)

    def test_validate_performance_config_success(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test successful performance config validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_performance_config, f)
            f.flush()

            result = validator.validate_performance_config(f.name)

            assert result is True
            assert len(validator.errors) == 0

        os.unlink(f.name)

    def test_validate_performance_config_optional(self, validator: "ConfigValidator") -> None:
        """Test performance config is optional."""
        result = validator.validate_performance_config("nonexistent.yaml")

        assert result is True
        assert len(validator.errors) == 0

    def test_validate_performance_config_invalid_yaml(self, validator: "ConfigValidator") -> None:
        """Test performance config with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: [\n")
            f.flush()

            result = validator.validate_performance_config(f.name)

            assert result is False
            assert any("Invalid YAML" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_embedding_settings(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test embedding settings validation."""
        config = valid_performance_config.copy()
        config["vector_db"]["embedding"]["batch_size"] = 0
        config["vector_db"]["embedding"]["max_retries"] = -1
        config["vector_db"]["embedding"]["request_timeout"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any("batch_size must be a positive integer" in err for err in validator.errors)
            assert any(
                "max_retries must be a non-negative integer" in err for err in validator.errors
            )
            assert any(
                "request_timeout must be a positive number" in err for err in validator.errors
            )

        os.unlink(f.name)

    def test_validate_search_settings(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test search settings validation."""
        config = valid_performance_config.copy()
        config["vector_db"]["search"]["max_limit"] = 5
        config["vector_db"]["search"]["default_limit"] = 10

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any("max_limit must be >= default_limit" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_connection_pool_settings(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test connection pool settings validation."""
        config = valid_performance_config.copy()
        config["graph_db"]["connection_pool"]["max_size"] = 5
        config["graph_db"]["connection_pool"]["min_size"] = 10

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any("max_size must be >= min_size" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_query_settings(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test graph query settings validation."""
        config = valid_performance_config.copy()
        config["graph_db"]["query"]["max_path_length"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any(
                "max_path_length must be a positive integer" in err for err in validator.errors
            )

        os.unlink(f.name)

    def test_validate_query_settings_warning(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test graph query settings warning for high values."""
        config = valid_performance_config.copy()
        config["graph_db"]["query"]["max_path_length"] = 15

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any("may cause performance issues" in warn for warn in validator.warnings)

        os.unlink(f.name)

    def test_validate_temporal_decay_rate(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test temporal decay rate validation."""
        config = valid_performance_config.copy()
        config["search"]["ranking"]["temporal_decay_rate"] = "not_a_number"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any("temporal_decay_rate must be a number" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_temporal_decay_rate_range(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test temporal decay rate range validation."""
        config = valid_performance_config.copy()
        config["search"]["ranking"]["temporal_decay_rate"] = 1.5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any(
                "temporal_decay_rate must be between 0 and 1" in err for err in validator.errors
            )

        os.unlink(f.name)

    def test_validate_type_boosts(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test type boosts validation."""
        config = valid_performance_config.copy()
        config["search"]["ranking"]["type_boosts"]["invalid_type"] = -1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any(
                "type_boosts.invalid_type must be a non-negative number" in err
                for err in validator.errors
            )

        os.unlink(f.name)

    def test_validate_resources(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test resource validation."""
        config = valid_performance_config.copy()
        config["resources"]["max_memory_gb"] = 0.1
        config["resources"]["max_cpu_percent"] = 150

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any("max_memory_gb must be at least 0.5" in err for err in validator.errors)
            assert any(
                "max_cpu_percent must be between 1 and 100" in err for err in validator.errors
            )

        os.unlink(f.name)

    def test_validate_kv_store_redis_settings(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test KV store Redis settings validation."""
        config = valid_performance_config.copy()
        config["kv_store"]["redis"]["connection_pool"]["max_size"] = 1
        config["kv_store"]["redis"]["connection_pool"]["min_size"] = 10
        config["kv_store"]["redis"]["cache"]["ttl_seconds"] = 0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any(
                "redis.connection_pool.max_size must be >= min_size" in err
                for err in validator.errors
            )
            assert any("ttl_seconds must be a positive integer" in err for err in validator.errors)

        os.unlink(f.name)

    def test_validate_kv_store_duckdb_settings(
        self, validator: "ConfigValidator", valid_performance_config: Dict[str, Any]
    ) -> None:
        """Test KV store DuckDB settings validation."""
        config = valid_performance_config.copy()
        config["kv_store"]["duckdb"]["batch_insert"]["size"] = 0
        config["kv_store"]["duckdb"]["analytics"]["retention_days"] = -1

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            validator.validate_performance_config(f.name)

            assert any(
                "batch_insert.size must be a positive integer" in err for err in validator.errors
            )
            assert any(
                "retention_days must be a positive integer" in err for err in validator.errors
            )

        os.unlink(f.name)

    def test_validate_all_method(
        self,
        validator: "ConfigValidator",
        valid_main_config: Dict[str, Any],
        valid_performance_config: Dict[str, Any],
    ) -> None:
        """Test validate_all method."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as main_f:
            yaml.dump(valid_main_config, main_f)
            main_f.flush()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as perf_f:
                yaml.dump(valid_performance_config, perf_f)
                perf_f.flush()

                # Mock the default file paths
                with (
                    patch.object(validator, "validate_main_config") as mock_main,
                    patch.object(validator, "validate_performance_config") as mock_perf,
                ):
                    mock_main.return_value = True
                    mock_perf.return_value = True

                    is_valid, errors, warnings = validator.validate_all()

                    assert is_valid is True
                    assert errors == []
                    assert warnings == []
                    mock_main.assert_called_once()
                    mock_perf.assert_called_once()

        os.unlink(main_f.name)
        os.unlink(perf_f.name)


class TestStandaloneFunctions:
    """Test standalone validation functions."""

    def test_validate_environment_variables_success(self) -> None:
        """Test successful environment variable validation."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
        }

        with patch.dict(os.environ, env_vars):
            result = validate_environment_variables()

            assert result["valid"] is True
            assert result["missing"] == []

    def test_validate_environment_variables_missing(self) -> None:
        """Test environment variable validation with missing vars."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment_variables()

            assert result["valid"] is False
            assert len(result["missing"]) == 5
            assert "NEO4J_URI" in result["missing"]
            assert "QDRANT_URL" in result["missing"]

    def test_validate_database_config_none_config(self) -> None:
        """Test database config validation with None config."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_database_config("neo4j", None)

        assert "cannot be None" in str(exc_info.value)

    def test_validate_neo4j_config_success(self) -> None:
        """Test successful Neo4j config validation."""
        config = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "password"}

        result = validate_database_config("neo4j", config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_neo4j_config_missing_fields(self) -> None:
        """Test Neo4j config validation with missing fields."""
        config = {"uri": "bolt://localhost:7687"}

        result = validate_database_config("neo4j", config)

        assert result["valid"] is False
        assert len(result["errors"]) == 2
        assert any("Missing 'user'" in err for err in result["errors"])
        assert any("Missing 'password'" in err for err in result["errors"])

    def test_validate_neo4j_config_invalid_uri(self) -> None:
        """Test Neo4j config validation with invalid URI."""
        config = {"uri": "http://localhost:7687", "user": "neo4j", "password": "password"}

        result = validate_database_config("neo4j", config)

        assert result["valid"] is False
        assert any("Invalid Neo4j URI scheme" in err for err in result["errors"])

    def test_validate_qdrant_config_success(self) -> None:
        """Test successful Qdrant config validation."""
        config = {"url": "http://localhost:6333"}

        result = validate_database_config("qdrant", config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_qdrant_config_missing_url(self) -> None:
        """Test Qdrant config validation with missing URL."""
        config = {}

        result = validate_database_config("qdrant", config)

        assert result["valid"] is False
        assert any("Missing 'url'" in err for err in result["errors"])

    def test_validate_qdrant_config_invalid_url(self) -> None:
        """Test Qdrant config validation with invalid URL."""
        config = {"url": "ftp://localhost:6333"}

        result = validate_database_config("qdrant", config)

        assert result["valid"] is False
        assert any("Invalid Qdrant URL scheme" in err for err in result["errors"])

    def test_validate_redis_config_success(self) -> None:
        """Test successful Redis config validation."""
        config = {"url": "redis://localhost:6379"}

        result = validate_database_config("redis", config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_redis_config_invalid_url(self) -> None:
        """Test Redis config validation with invalid URL."""
        config = {"url": "http://localhost:6379"}

        result = validate_database_config("redis", config)

        assert result["valid"] is False
        assert any("Invalid Redis URL scheme" in err for err in result["errors"])

    def test_validate_database_config_unknown_type(self) -> None:
        """Test database config validation with unknown type."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_database_config("unknown", {})

        assert "Unknown database type" in str(exc_info.value)

    def test_validate_mcp_config_success(self) -> None:
        """Test successful MCP config validation."""
        config = {
            "server_port": 8000,
            "host": "0.0.0.0",
            "tools": ["store_context", "retrieve_context"],
        }

        result = validate_mcp_config(config)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_mcp_config_missing_fields(self) -> None:
        """Test MCP config validation with missing fields."""
        config = {}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert len(result["errors"]) == 3
        assert any("Missing 'server_port'" in err for err in result["errors"])
        assert any("Missing 'host'" in err for err in result["errors"])
        assert any("Missing 'tools'" in err for err in result["errors"])

    def test_validate_mcp_config_invalid_port(self) -> None:
        """Test MCP config validation with invalid port."""
        config = {"server_port": "not_a_number", "host": "0.0.0.0", "tools": ["store_context"]}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("server_port must be an integer" in err for err in result["errors"])

    def test_validate_mcp_config_port_out_of_range(self) -> None:
        """Test MCP config validation with port out of range."""
        config = {"server_port": 70000, "host": "0.0.0.0", "tools": ["store_context"]}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("server_port must be between 1 and 65535" in err for err in result["errors"])

    def test_validate_mcp_config_invalid_tools(self) -> None:
        """Test MCP config validation with invalid tools."""
        config = {"server_port": 8000, "host": "0.0.0.0", "tools": "not_a_list"}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("tools must be a list" in err for err in result["errors"])

    def test_validate_mcp_config_empty_tools(self) -> None:
        """Test MCP config validation with empty tools."""
        config = {"server_port": 8000, "host": "0.0.0.0", "tools": []}

        result = validate_mcp_config(config)

        assert result["valid"] is False
        assert any("At least one tool must be configured" in err for err in result["errors"])

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "MCP_SERVER_PORT": "8000",
        },
    )
    def test_validate_all_configs_success(self) -> None:
        """Test successful validation of all configs."""
        result = validate_all_configs()

        assert result["valid"] is True
        assert result["environment"]["valid"] is True
        assert result["databases"]["neo4j"]["valid"] is True
        assert result["databases"]["qdrant"]["valid"] is True
        assert result["databases"]["redis"]["valid"] is True
        assert result["mcp"]["valid"] is True

    @patch.dict(os.environ, {"LOG_LEVEL": "debug"}, clear=True)
    def test_validate_all_configs_missing_env(self) -> None:
        """Test validation of all configs with missing environment."""
        result = validate_all_configs()

        assert result["valid"] is False
        assert result["environment"]["valid"] is False
        assert len(result["environment"]["missing"]) == 5

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "LOG_LEVEL": "debug",
        },
    )
    def test_validate_all_configs_with_warnings(self) -> None:
        """Test validation of all configs with warnings."""
        result = validate_all_configs()

        assert result["valid"] is True
        assert len(result["warnings"]) == 1
        assert "Debug logging is enabled" in result["warnings"][0]

    @patch.dict(
        os.environ,
        {
            "NEO4J_URI": "invalid://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
        },
    )
    def test_validate_all_configs_invalid_neo4j(self) -> None:
        """Test validation of all configs with invalid Neo4j URI."""
        result = validate_all_configs()

        assert result["valid"] is False
        assert result["databases"]["neo4j"]["valid"] is False


class TestConfigValidationError:
    """Test ConfigValidationError exception."""

    def test_config_validation_error_creation(self) -> None:
        """Test creating ConfigValidationError."""
        error = ConfigValidationError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
