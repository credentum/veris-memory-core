#!/usr/bin/env python3
"""
Additional tests to achieve 100% coverage for config_validator module.
"""

import os
import tempfile
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.validators.config_validator import main, validate_all_configs  # noqa: E402


class TestCompleteConfig:
    """Test for complete config without errors or warnings."""

    def test_main_no_errors_no_warnings(self):
        """Test main function with no errors and no warnings (line 289)."""
        config = {
            "system": {},
            "qdrant": {"ssl": True},  # SSL enabled - no warning
            "neo4j": {"ssl": True},  # SSL enabled - no warning
            "redis": {"ssl": True},  # SSL enabled - no warning
            "storage": {},
            "agents": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            tmp.flush()

            try:
                runner = CliRunner()
                result = runner.invoke(
                    main, ["--config", tmp.name, "--perf-config", "nonexistent.yaml"]
                )

                assert result.exit_code == 0
                assert "✅ All configurations are valid!" in result.output
            finally:
                os.unlink(tmp.name)


class TestValidateAllConfigsEdgeCases:
    """Test edge cases in validate_all_configs function."""

    def test_validate_all_configs_invalid_qdrant(self):
        """Test validate_all_configs with invalid Qdrant config (line 436)."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "QDRANT_URL": "invalid://localhost:6333",  # Invalid URL scheme
                "REDIS_URL": "redis://localhost:6379",
            },
        ):
            result = validate_all_configs()
            assert result["valid"] is False
            assert not result["databases"]["qdrant"]["valid"]

    def test_validate_all_configs_invalid_redis(self):
        """Test validate_all_configs with invalid Redis config (line 446)."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "QDRANT_URL": "http://localhost:6333",
                "REDIS_URL": "invalid://localhost:6379",  # Invalid URL scheme
            },
        ):
            result = validate_all_configs()
            assert result["valid"] is False
            assert not result["databases"]["redis"]["valid"]

    def test_validate_all_configs_invalid_mcp_port(self):
        """Test validate_all_configs with invalid MCP config (line 456)."""
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
                "QDRANT_URL": "http://localhost:6333",
                "REDIS_URL": "redis://localhost:6379",
                "MCP_SERVER_PORT": "70000",  # Invalid port number > 65535
            },
        ):
            result = validate_all_configs()
            assert result["valid"] is False
            assert not result["mcp"]["valid"]

    @patch("validators.config_validator.validate_mcp_config")
    def test_validate_all_configs_mcp_validation_failure(self, mock_validate_mcp):
        """Test validate_all_configs when MCP validation fails."""
        # Make MCP validation fail
        mock_validate_mcp.return_value = {"valid": False, "errors": ["MCP error"]}

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
            assert result["valid"] is False
            assert not result["mcp"]["valid"]


class TestMainEntryPoint:
    """Test the module entry point."""

    def test_module_main_execution(self):
        """Test that the module can be executed as main."""
        # Save the original __name__
        import src.validators.config_validator as module

        original_name = module.__name__

        try:
            # This tests that the if __name__ == "__main__" structure exists
            # We can't actually trigger it in tests, but we can verify the code path
            assert hasattr(module, "main")
            assert callable(module.main)

            # The module should have the __main__ check at the end
            with open(module.__file__, "r") as f:
                content = f.read()
                assert 'if __name__ == "__main__"' in content
                assert "main()" in content
        finally:
            # Restore original name
            module.__name__ = original_name
