#!/usr/bin/env python3
"""
Comprehensive tests for config_validator module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.validators.config_validator import (  # noqa: E402
    ConfigValidationError,
    ConfigValidator,
    validate_all_configs,
    validate_database_config,
    validate_environment_variables,
    validate_mcp_config,
)


class TestConfigValidator:
    """Test suite for ConfigValidator class."""

    def test_init_with_config(self):
        """Test initialization with config dictionary."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {"test": "value"}
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                assert validator.config_path == tmp.name
                assert validator.config is not None
            finally:
                os.unlink(tmp.name)

    def test_init_missing_file(self):
        """Test initialization with missing file."""
        with pytest.raises(FileNotFoundError):
            ConfigValidator()

    @patch("validators.config_validator.yaml.safe_load")
    def test_load_config_valid(self, mock_yaml):
        """Test loading valid config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {"test": "value"}
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                mock_yaml.return_value = config
                validator = ConfigValidator()
                validator.validate_main_config()
                assert validator.config == config
            finally:
                os.unlink(tmp.name)

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write("invalid: yaml: content: [")
            tmp.flush()

            try:
                with pytest.raises(ConfigValidationError):
                    ConfigValidator()
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config():
        """Test validating required fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {"storage": {"neo4j": {"uri": "bolt://localhost:7687"}}}
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_main_config()
                assert result is True

                result = validator.validate_main_config()
                assert result is False
            finally:
                os.unlink(tmp.name)

    def test_validate_main_config():
        """Test validating config structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {
                "storage": {
                    "neo4j": {"uri": "bolt://localhost:7687"},
                    "qdrant": {"url": "http://localhost:6333"},
                }
            }
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                errors = validator.validate_main_config()
                # Structure validation should pass for basic config
                assert isinstance(errors, list)
            finally:
                os.unlink(tmp.name)

    def test_validate_values(self):
        """Test validating config values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {
                "storage": {
                    "neo4j": {"uri": "bolt://localhost:7687"},
                    "qdrant": {"url": "invalid_url"},  # Invalid URL
                }
            }
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                errors = validator.validate_values()
                assert isinstance(errors, list)
                # Should have error for invalid URL
                assert any("url" in str(e).lower() for e in errors)
            finally:
                os.unlink(tmp.name)

    def test_validate_complete(self):
        """Test complete validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {
                "storage": {
                    "neo4j": {"uri": "bolt://localhost:7687"},
                    "qdrant": {"url": "http://localhost:6333"},
                }
            }
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                result = validator.validate_all()
                assert isinstance(result, dict)
                assert "valid" in result
            finally:
                os.unlink(tmp.name)

    def test_get_errors(self):
        """Test getting validation errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {"test": "value"}
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                errors = validator.errors
                assert isinstance(errors, list)
            finally:
                os.unlink(tmp.name)

    def test_get_warnings(self):
        """Test getting validation warnings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            config = {"test": "value"}
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                validator = ConfigValidator()
                warnings = validator.warnings
                assert isinstance(warnings, list)
            finally:
                os.unlink(tmp.name)


class TestValidationFunctions:
    """Test suite for validation functions."""

    def test_validate_environment_variables(self):
        """Test validating environment variables."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "QDRANT_URL": "http://localhost:6333",
            },
        ):
            result = validate_environment_variables()
            assert isinstance(result, dict)
            assert "environment_variables" in result

    def test_validate_environment_variables_missing(self):
        """Test validating with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment_variables()
            assert isinstance(result, dict)
            assert "warnings" in result or "environment_variables" in result

    def test_validate_database_config_neo4j(self):
        """Test validating Neo4j database config."""
        config = {"uri": "bolt://localhost:7687", "database": "neo4j"}
        result = validate_database_config("neo4j", config)
        assert isinstance(result, dict)
        assert result.get("valid", False) is True

    def test_validate_database_config_neo4j_invalid(self):
        """Test validating invalid Neo4j config."""
        config = {"uri": "invalid_uri"}
        result = validate_database_config("neo4j", config)
        assert isinstance(result, dict)
        # May have errors for invalid URI format

    def test_validate_database_config_qdrant(self):
        """Test validating Qdrant database config."""
        config = {"url": "http://localhost:6333", "collection_name": "contexts"}
        result = validate_database_config("qdrant", config)
        assert isinstance(result, dict)
        assert result.get("valid", False) is True

    def test_validate_database_config_redis(self):
        """Test validating Redis database config."""
        config = {"url": "redis://localhost:6379", "db": 0}
        result = validate_database_config("redis", config)
        assert isinstance(result, dict)
        assert result.get("valid", False) is True

    def test_validate_database_config_unknown(self):
        """Test validating unknown database type."""
        config = {"some": "config"}
        result = validate_database_config("unknown_db", config)
        assert isinstance(result, dict)
        assert "error" in result or result.get("valid") is False

    def test_validate_mcp_config(self):
        """Test validating MCP config."""
        config = {
            "storage": {
                "neo4j": {"uri": "bolt://localhost:7687"},
                "qdrant": {"url": "http://localhost:6333"},
            },
            "rate_limits": {"enabled": True, "max_rpm": 60},
        }
        result = validate_mcp_config(config)
        assert isinstance(result, dict)

    def test_validate_mcp_config_empty(self):
        """Test validating empty MCP config."""
        result = validate_mcp_config({})
        assert isinstance(result, dict)

    @patch("validators.config_validator.Path")
    @patch("validators.config_validator.yaml.safe_load")
    def test_validate_all_configs(self, mock_yaml, mock_path):
        """Test validating all configurations."""
        # Mock the config file
        mock_path_instance = AsyncMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.parent = Path(".")
        mock_path.return_value = mock_path_instance

        mock_yaml.return_value = {
            "storage": {
                "neo4j": {"uri": "bolt://localhost:7687"},
                "qdrant": {"url": "http://localhost:6333"},
            }
        }

        with patch.dict(os.environ, {"NEO4J_URI": "bolt://localhost:7687"}):
            with patch("builtins.open", create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = "config: data"
                result = validate_all_configs()

                assert isinstance(result, dict)
                assert "config_files" in result or "validation" in result

    @patch("validators.config_validator.Path")
    def test_validate_all_configs_no_files(self, mock_path):
        """Test validate_all_configs with no config files."""
        mock_path_instance = AsyncMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        result = validate_all_configs()
        assert isinstance(result, dict)


class TestMainFunction:
    """Test the main CLI function."""

    @patch("validators.config_validator.validate_all_configs")
    @patch("validators.config_validator.click.echo")
    def test_main_default(self, mock_echo, mock_validate):
        """Test main function with default arguments."""
        from src.validators.config_validator import main

        mock_validate.return_value = {
            "valid": True,
            "config_files": [".ctxrc.yaml"],
            "warnings": [],
        }

        main(config=".ctxrc.yaml", perf_config="configs/perf.yaml", strict=False)

        mock_validate.assert_called_once_with()
        assert mock_echo.called

    @patch("validators.config_validator.validate_all_configs")
    @patch("validators.config_validator.click.echo")
    @patch("validators.config_validator.sys.exit")
    def test_main_strict_mode_failure(self, mock_exit, mock_echo, mock_validate):
        """Test main function in strict mode with validation failure."""
        from src.validators.config_validator import main

        mock_validate.return_value = {
            "valid": False,
            "errors": ["Test error"],
            "config_files": [],
        }

        main(config=".ctxrc.yaml", perf_config="configs/perf.yaml", strict=True)

        mock_exit.assert_called_with(1)

    @patch("validators.config_validator.validate_all_configs")
    @patch("validators.config_validator.click.echo")
    def test_main_with_warnings(self, mock_echo, mock_validate):
        """Test main function with warnings."""
        from src.validators.config_validator import main

        mock_validate.return_value = {
            "valid": True,
            "warnings": ["Test warning 1", "Test warning 2"],
            "config_files": [],
        }

        main(config=".ctxrc.yaml", perf_config="configs/perf.yaml", strict=False)

        # Should display warnings
        assert any("warning" in str(call).lower() for call in mock_echo.call_args_list)
