#!/usr/bin/env python3
"""
Tests for Sentinel Configuration Validator

Validates that S10 uses correct MCP field names and prevents regressions.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.validators.sentinel_config_validator import SentinelConfigValidator


class TestSentinelConfigValidator:
    """Test suite for Sentinel configuration validation."""

    def test_s10_content_type_field_validation_passes(self):
        """Test that S10 correctly uses 'content_type' field (not 'context_type')."""
        # Use actual S10 file
        validator = SentinelConfigValidator()
        is_valid, errors = validator.validate_s10_mcp_field_names()

        assert is_valid, f"S10 validation failed: {errors}"
        assert len(errors) == 0, "S10 should use correct 'content_type' field name"

    def test_s10_uses_content_type_not_context_type(self):
        """Test that S10 does NOT use the incorrect 'context_type' field (PR #273 bug prevention)."""
        validator = SentinelConfigValidator()

        # Read actual S10 file
        s10_file = validator.checks_dir / "s10_content_pipeline.py"
        with open(s10_file, 'r') as f:
            content = f.read()

        # Verify no usage of incorrect field name in payload construction
        # (Allow in comments for documentation purposes)
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '"context_type":' in line or "'context_type':" in line:
                # Split on '#' to separate code from comment
                code_part = line.split('#')[0] if '#' in line else line
                # Should NOT appear in actual code (OK in comments)
                assert '"context_type"' not in code_part and "'context_type'" not in code_part, \
                    f"Line {i} uses incorrect 'context_type' field in code - should be 'content_type': {line}"

    def test_s10_has_content_type_field_in_payloads(self):
        """Test that S10 actually uses 'content_type' field in MCP payloads."""
        validator = SentinelConfigValidator()

        s10_file = validator.checks_dir / "s10_content_pipeline.py"
        with open(s10_file, 'r') as f:
            content = f.read()

        # Verify correct field name is used
        assert '"content_type"' in content or "'content_type'" in content, \
            "S10 must use 'content_type' field for MCP payloads"

    def test_s9_s10_use_valid_mcp_types(self):
        """Test that S9 and S10 use valid MCP types (design, decision, trace, sprint, log)."""
        validator = SentinelConfigValidator()
        is_valid, errors = validator.validate_s9_s10_mcp_types()

        assert is_valid, f"S9/S10 MCP type validation failed: {errors}"
        assert len(errors) == 0, "S9/S10 should only use valid MCP types"

    def test_s9_does_not_use_invalid_graph_intent_test_type(self):
        """Test that S9 does NOT use 'graph_intent_test' as MCP type (PR #270 bug prevention)."""
        validator = SentinelConfigValidator()

        s9_file = validator.checks_dir / "s9_graph_intent.py"
        with open(s9_file, 'r') as f:
            content = f.read()

        # Check that invalid test type is not used as MCP type
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '"graph_intent_test"' in line and '"content_type":' in line:
                # Split on '#' to separate code from comment
                code_part = line.split('#')[0] if '#' in line else line
                # Check if both appear in actual code (not just comment)
                if '"graph_intent_test"' in code_part and '"content_type":' in code_part:
                    pytest.fail(
                        f"S9 line {i} uses invalid MCP type 'graph_intent_test' in code - "
                        f"should use valid MCP type (design, decision, trace, sprint, log)"
                    )

    def test_s10_does_not_use_invalid_pipeline_test_type(self):
        """Test that S10 does NOT use 'pipeline_test' as MCP type (PR #270 bug prevention)."""
        validator = SentinelConfigValidator()

        s10_file = validator.checks_dir / "s10_content_pipeline.py"
        with open(s10_file, 'r') as f:
            content = f.read()

        # Check that invalid test type is not used as MCP type
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '"pipeline_test"' in line and '"content_type":' in line:
                # Split on '#' to separate code from comment
                code_part = line.split('#')[0] if '#' in line else line
                # Check if both appear in actual code (not just comment)
                if '"pipeline_test"' in code_part and '"content_type":' in code_part:
                    pytest.fail(
                        f"S10 line {i} uses invalid MCP type 'pipeline_test' in code - "
                        f"should use valid MCP type (design, decision, trace, sprint, log)"
                    )

    def test_validate_all_checks_passes_on_current_codebase(self):
        """Test that all Sentinel validation checks pass on current codebase."""
        validator = SentinelConfigValidator()
        all_valid, results = validator.validate_all_checks()

        assert all_valid, f"Sentinel validation failed: {results}"
        assert len(results) == 0, "All Sentinel checks should pass validation"

    def test_validator_rejects_context_type_in_mock_file(self):
        """Test that validator correctly rejects 'context_type' usage."""
        # Create temporary directory with mock S10 file
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"

            # Write mock S10 with incorrect field name
            mock_s10.write_text('''
def create_payload():
    return {
        "context_type": "log",  # WRONG - should be content_type
        "content": "test"
    }
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert not is_valid, "Validator should reject 'context_type' usage"
            assert len(errors) > 0
            assert any("context_type" in err for err in errors)

    def test_validator_accepts_content_type_in_mock_file(self):
        """Test that validator correctly accepts 'content_type' usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"

            # Write mock S10 with correct field name
            mock_s10.write_text('''
def create_payload():
    return {
        "content_type": "log",  # CORRECT
        "content": "test"
    }
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert is_valid, f"Validator should accept 'content_type' usage, got errors: {errors}"
            assert len(errors) == 0

    def test_validator_allows_context_type_in_comments(self):
        """Test that validator allows 'context_type' in comments (for documentation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"

            # Write mock S10 with field name in comment
            mock_s10.write_text('''
def create_payload():
    # Fixed: Was using "context_type" (wrong), now using "content_type" (correct)
    return {
        "content_type": "log",
        "content": "test"
    }
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert is_valid, f"Validator should allow 'context_type' in comments, got errors: {errors}"
            assert len(errors) == 0


class TestSentinelValidatorCLI:
    """Test command-line interface for sentinel validator."""

    def test_cli_returns_0_on_valid_config(self):
        """Test that CLI returns exit code 0 when validation passes."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "src.validators.sentinel_config_validator"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Validator CLI should return 0 on success, got stderr: {result.stderr}"
        assert "All Sentinel configuration validations passed" in result.stdout


class TestSentinelValidatorEdgeCases:
    """Test edge cases and error handling for sentinel validator."""

    def test_validator_handles_missing_checks_directory(self):
        """Test validator handles missing sentinel checks directory gracefully."""
        with pytest.raises(ValueError, match="Sentinel checks directory not found"):
            SentinelConfigValidator(sentinel_checks_dir="/nonexistent/path")

    def test_validator_handles_missing_s10_file(self):
        """Test validator handles missing S10 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert not is_valid
            assert len(errors) == 1
            assert "S10 check file not found" in errors[0]

    def test_validator_handles_empty_s10_file(self):
        """Test validator handles empty S10 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"
            mock_s10.write_text("")  # Empty file

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert not is_valid
            assert any("must use MCP field 'content_type'" in err for err in errors)

    def test_validator_handles_malformed_python_syntax(self):
        """Test validator doesn't crash on malformed Python (treats as text)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"
            # Write malformed Python - missing closing quote
            mock_s10.write_text('def foo():\n    x = "unclosed string\n')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            # Should not crash, just analyze as text
            is_valid, errors = validator.validate_s10_mcp_field_names()

            # Will fail because no content_type found
            assert not is_valid

    def test_validator_detects_multiple_violations_per_file(self):
        """Test validator reports all violations, not just first one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"

            # Multiple violations
            mock_s10.write_text('''
payload1 = {"context_type": "log"}  # Line 2 - WRONG
payload2 = {"context_type": "design"}  # Line 3 - WRONG
payload3 = {"context_type": "trace"}  # Line 4 - WRONG
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert not is_valid
            assert len(errors) >= 3, "Should detect all 3 violations"

    def test_validator_handles_mixed_quotes(self):
        """Test validator detects violations with both single and double quotes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"

            mock_s10.write_text('''
payload1 = {"context_type": "log"}  # Double quotes - WRONG
payload2 = {'context_type': 'log'}  # Single quotes - WRONG
payload3 = {"content_type": "log"}  # CORRECT
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert not is_valid
            assert len(errors) >= 2, "Should detect both double and single quote violations"

    def test_validator_handles_multiline_payloads(self):
        """Test validator works with payloads spanning multiple lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"

            mock_s10.write_text('''
payload = {
    "content_type": "log",
    "content": {
        "data": "test"
    }
}
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert is_valid, f"Should accept multiline payloads, got errors: {errors}"

    def test_validator_handles_unicode_content(self):
        """Test validator handles files with unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_s10 = Path(tmpdir) / "s10_content_pipeline.py"

            mock_s10.write_text('''
# Test with unicode: émojis 🎉, Chinese: 测试, Arabic: اختبار
payload = {"content_type": "log", "content": "test"}
''', encoding='utf-8')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s10_mcp_field_names()

            assert is_valid, "Should handle unicode content"

    def test_validator_s9_s10_missing_files(self):
        """Test MCP type validation when S9/S10 files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s9_s10_mcp_types()

            assert not is_valid
            assert len(errors) >= 2  # Both S9 and S10 files missing
            assert any("s9_graph_intent.py" in err for err in errors)
            assert any("s10_content_pipeline.py" in err for err in errors)

    def test_validator_detects_invalid_mcp_type_in_metadata(self):
        """Test validator detects invalid MCP types even when used in nested structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "s9_graph_intent.py").write_text('''
payload = {
    "content_type": "graph_intent_test",  # WRONG - invalid MCP type
    "metadata": {"test": "data"}
}
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            is_valid, errors = validator.validate_s9_s10_mcp_types()

            assert not is_valid
            assert any("graph_intent_test" in err for err in errors)

    def test_validate_all_checks_returns_all_errors(self):
        """Test that validate_all_checks aggregates errors from all validation methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create S10 with wrong field name
            (Path(tmpdir) / "s10_content_pipeline.py").write_text('''
payload = {"context_type": "pipeline_test"}  # Two violations
''')

            validator = SentinelConfigValidator(sentinel_checks_dir=tmpdir)
            all_valid, results = validator.validate_all_checks()

            assert not all_valid
            # Should have errors from both field name and MCP type checks
            assert "s10_mcp_field_names" in results
            assert "s9_s10_mcp_types" in results  # S9 missing file


    def test_validator_cli_exit_code_on_failure(self):
        """Test that CLI returns non-zero exit code when validation fails."""
        import subprocess
        import sys

        # Create temporary invalid file
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "s10_content_pipeline.py").write_text('''
payload = {"context_type": "log"}  # WRONG
''')

            # Create temporary validator script
            validator_script = Path(tmpdir) / "test_validator.py"
            validator_script.write_text(f'''
import sys
sys.path.insert(0, "{Path(__file__).parent.parent.parent}")
from src.validators.sentinel_config_validator import SentinelConfigValidator

validator = SentinelConfigValidator("{tmpdir}")
all_valid, results = validator.validate_all_checks()
sys.exit(0 if all_valid else 1)
''')

            result = subprocess.run(
                [sys.executable, str(validator_script)],
                capture_output=True
            )

            assert result.returncode != 0, "CLI should return non-zero on validation failure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
