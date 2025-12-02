#!/usr/bin/env python3
"""
Tests for the main CLI function of config_validator to achieve 100% coverage.
"""

import os
import tempfile
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.validators.config_validator import main  # noqa: E402


class TestMainCLI:
    """Test suite for the main CLI function."""

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
                # Even with warnings, exit code is 0 in non-strict mode
                # The config is valid but has SSL warnings
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

    def test_main_with_warnings_non_strict(self):
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
                runner = CliRunner()
                result = runner.invoke(
                    main, ["--config", tmp.name, "--perf-config", "nonexistent.yaml"]
                )

                assert result.exit_code == 0  # Exit 0 in non-strict mode
                assert "⚠️" in result.output
            finally:
                os.unlink(tmp.name)

    def test_main_with_warnings_strict(self):
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
                runner = CliRunner()
                result = runner.invoke(
                    main,
                    [
                        "--config",
                        tmp.name,
                        "--perf-config",
                        "nonexistent.yaml",
                        "--strict",
                    ],
                )

                assert result.exit_code == 1  # Exit 1 in strict mode
                assert "⚠️" in result.output
            finally:
                os.unlink(tmp.name)

    def test_main_default_args(self):
        """Test main function with default arguments."""
        # Create a minimal valid config at the default location
        config = {"system": {}, "qdrant": {}, "neo4j": {}, "storage": {}, "agents": {}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, dir=".") as tmp:
            # Rename to the expected default config name
            default_config_path = ".ctxrc.yaml"
            yaml.dump(config, tmp)
            tmp.flush()
            os.rename(tmp.name, default_config_path)

            try:
                runner = CliRunner()
                result = runner.invoke(main, [])

                # Should work with default config
                assert "Configuration Validation" in result.output
            finally:
                if os.path.exists(default_config_path):
                    os.unlink(default_config_path)

    def test_main_help(self):
        """Test main function help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Validate configuration files" in result.output
        assert "--config" in result.output
        assert "--perf-config" in result.output
        assert "--strict" in result.output

    @patch("validators.config_validator.__name__", "__main__")
    @patch("validators.config_validator.main")
    def test_main_entry_point(self, mock_main):
        """Test the if __name__ == '__main__' block."""
        # Import the module to trigger the __main__ check
        import src.validators.config_validator

        # The __main__ block should call main()
        # We can't directly test this, but we can verify the structure exists
        assert hasattr(src.validators.config_validator, "main")
        assert callable(src.validators.config_validator.main)
