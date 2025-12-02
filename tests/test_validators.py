#!/usr/bin/env python3
"""
Comprehensive unit tests for validators module.
Consolidated from multiple test files to maintain single source of truth.
"""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.validators.config_validator import (
    ConfigValidator,
    main,
    validate_all_configs,
    validate_database_config,
    validate_environment_variables,
    validate_mcp_config,
)
from src.validators.kv_validators import (
    sanitize_metric_name,
    validate_cache_entry,
    validate_metric_event,
    validate_redis_key,
    validate_session_data,
    validate_time_range,
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


class TestConfigValidatorCLI:
    """Test suite for ConfigValidator CLI functionality."""

    def test_main_all_valid(self):
        """Test main function with all valid configurations."""
        config = {"system": {}, "qdrant": {}, "neo4j": {}, "storage": {}, "agents": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                runner = CliRunner()
                result = runner.invoke(
                    main, ["--config", tmp.name, "--perf-config", "nonexistent.yaml"]
                )
                assert result.exit_code == 0
            finally:
                os.unlink(tmp.name)

    def test_main_with_errors(self):
        """Test main function with validation errors."""
        config = {}  # Missing required sections

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                runner = CliRunner()
                result = runner.invoke(
                    main, ["--config", tmp.name, "--perf-config", "nonexistent.yaml"]
                )
                assert result.exit_code == 1
                assert "❌" in result.output
            finally:
                os.unlink(tmp.name)

    def test_main_help(self):
        """Test main function help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Validate configuration files" in result.output


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

    def test_validate_database_config_qdrant_valid(self):
        """Test valid Qdrant configuration."""
        config = {"url": "http://localhost:6333"}
        result = validate_database_config("qdrant", config)
        assert result["valid"] is True

    def test_validate_database_config_redis_valid(self):
        """Test valid Redis configuration."""
        config = {"url": "redis://localhost:6379"}
        result = validate_database_config("redis", config)
        assert result["valid"] is True

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

    def test_validate_all_configs_complete(self):
        """Test validate_all_configs with complete environment."""
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
            result = validate_all_configs()
            assert result["valid"] is True
            assert "environment" in result
            assert "databases" in result


class TestKVValidators:
    """Test suite for KV validators."""

    def test_valid_cache_entry(self):
        """Test validation of a valid cache entry."""
        entry = {
            "key": "test_key",
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is True

    def test_invalid_cache_entry_missing_field(self):
        """Test validation with missing required field."""
        entry = {
            "value": "test_value",
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": 3600,
        }
        assert validate_cache_entry(entry) is False

    def test_valid_metric_event(self):
        """Test validation of a valid metric event."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_name": "test.metric",
            "value": 42.5,
            "tags": {"env": "test", "service": "api"},
        }
        assert validate_metric_event(metric) is True

    def test_invalid_metric_event_missing_field(self):
        """Test validation with missing timestamp."""
        metric = {"metric_name": "test.metric", "value": 42, "tags": {}}
        assert validate_metric_event(metric) is False

    def test_sanitize_metric_name(self):
        """Test metric name sanitization."""
        name = "metric@#$%^&*()name"
        sanitized = sanitize_metric_name(name)
        assert "@" not in sanitized
        assert "#" not in sanitized
        assert all(c.isalnum() or c in "._-" for c in sanitized)

    def test_valid_time_range(self):
        """Test validation of a valid time range."""
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow() - timedelta(days=1)
        assert validate_time_range(start, end) is True

    def test_invalid_time_range_start_after_end(self):
        """Test validation with start time after end time."""
        start = datetime.utcnow() - timedelta(days=1)
        end = datetime.utcnow() - timedelta(days=7)
        assert validate_time_range(start, end) is False

    def test_valid_redis_key(self):
        """Test validation of a valid Redis key."""
        key = "user:session:12345"
        assert validate_redis_key(key) is True

    def test_invalid_redis_key_empty(self):
        """Test validation of empty key."""
        assert validate_redis_key("") is False

    def test_valid_session_data(self):
        """Test validation of valid session data."""
        data = {
            "user_id": 12345,
            "username": "testuser",
            "created_at": datetime.now().isoformat(),
            "preferences": {"theme": "dark", "language": "en"},
        }
        assert validate_session_data(data) is True

    def test_invalid_session_data_not_dict(self):
        """Test validation of non-dict session data."""
        assert validate_session_data("not a dict") is False
        assert validate_session_data(123) is False
        assert validate_session_data([1, 2, 3]) is False
