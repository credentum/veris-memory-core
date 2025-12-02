#!/usr/bin/env python3
"""
Sentinel Configuration Validator

Validates Sentinel monitoring check configurations to prevent common errors.
Specifically prevents regressions like the S10 content_type vs context_type bug (PR #273).
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple


class SentinelConfigValidator:
    """Validates Sentinel check configurations and prevents common bugs."""

    def __init__(self, sentinel_checks_dir: str = None):
        """Initialize validator with path to sentinel checks directory.

        Args:
            sentinel_checks_dir: Path to directory containing sentinel check modules.
                               Defaults to src/monitoring/sentinel/checks/
        """
        if sentinel_checks_dir is None:
            # Default to project structure
            project_root = Path(__file__).parent.parent.parent
            sentinel_checks_dir = project_root / "src" / "monitoring" / "sentinel" / "checks"

        self.checks_dir = Path(sentinel_checks_dir)
        if not self.checks_dir.exists():
            raise ValueError(f"Sentinel checks directory not found: {self.checks_dir}")

    def validate_s10_mcp_field_names(self) -> Tuple[bool, List[str]]:
        """Validate that S10 uses 'content_type' not 'context_type' (PR #273 regression prevention).

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        s10_file = self.checks_dir / "s10_content_pipeline.py"

        if not s10_file.exists():
            errors.append(f"S10 check file not found: {s10_file}")
            return False, errors

        # Read file content
        with open(s10_file, 'r') as f:
            content = f.read()

        # Check for incorrect field name "context_type"
        # This should NOT appear in MCP payload construction (but OK in comments)
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '"context_type":' in line or "'context_type':" in line:
                # Ignore if it's in a comment explaining the bug
                # Split on '#' to separate code from comment
                code_part = line.split('#')[0] if '#' in line else line
                if '"context_type"' in code_part or "'context_type'" in code_part:
                    errors.append(
                        f"S10 uses incorrect MCP field 'context_type' at line {i} - should be 'content_type'"
                    )

        # Check that correct field name "content_type" is used
        if '"content_type":' not in content and "'content_type':" not in content:
            errors.append("S10 must use MCP field 'content_type' for ingestion payloads")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_s9_s10_mcp_types(self) -> Tuple[bool, List[str]]:
        """Validate that S9 and S10 use valid MCP types (PR #270 regression prevention).

        Valid MCP types: design, decision, trace, sprint, log

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        valid_mcp_types = {"design", "decision", "trace", "sprint", "log"}

        for check_file in ["s9_graph_intent.py", "s10_content_pipeline.py"]:
            file_path = self.checks_dir / check_file
            if not file_path.exists():
                errors.append(f"Check file not found: {file_path}")
                continue

            with open(file_path, 'r') as f:
                content = f.read()

            # Look for invalid test types being used as MCP types
            invalid_patterns = [
                '"graph_intent_test"',  # S9 old invalid type
                '"pipeline_test"',      # S10 old invalid type
                '"content_pipeline_test"'  # S10 potential invalid type
            ]

            lines = content.split('\n')
            for pattern in invalid_patterns:
                for i, line in enumerate(lines, 1):
                    if pattern in line and '"content_type":' in line:
                        # Split on '#' to separate code from comment
                        code_part = line.split('#')[0] if '#' in line else line
                        # Check if invalid pattern appears in actual code (not just comment)
                        if pattern in code_part and '"content_type":' in code_part:
                            # This is using an invalid MCP type
                            errors.append(
                                f"{check_file} uses invalid MCP type {pattern} at line {i} - "
                                f"must use one of: {', '.join(valid_mcp_types)}"
                            )

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_all_checks(self) -> Tuple[bool, Dict[str, List[str]]]:
        """Run all validation checks on Sentinel configuration.

        Returns:
            Tuple of (all_valid, dict of check_name -> list of errors)
        """
        results = {}

        # S10 content_type field validation
        is_valid, errors = self.validate_s10_mcp_field_names()
        if not is_valid:
            results["s10_mcp_field_names"] = errors

        # S9/S10 MCP type validation
        is_valid, errors = self.validate_s9_s10_mcp_types()
        if not is_valid:
            results["s9_s10_mcp_types"] = errors

        all_valid = len(results) == 0
        return all_valid, results


def main():
    """Run validator as standalone script."""
    validator = SentinelConfigValidator()

    print("🔍 Validating Sentinel Check Configurations...\n")

    all_valid, results = validator.validate_all_checks()

    if all_valid:
        print("✅ All Sentinel configuration validations passed!")
        return 0
    else:
        print("❌ Sentinel configuration validation failures:\n")
        for check_name, errors in results.items():
            print(f"  {check_name}:")
            for error in errors:
                print(f"    - {error}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
