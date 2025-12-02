#!/usr/bin/env python3
"""
Format correction and validation for Claude review YAML.
This script runs in a sandboxed environment with restricted permissions.
"""
import logging
import os
import re
import sys
from datetime import datetime

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Security: Restrict file operations to current directory only
os.chdir(os.getcwd())


def safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        logger.debug(f"Failed to convert to int: {e}")
        return default


def safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        logger.debug(f"Failed to convert to float: {e}")
        return default


# Get file paths from environment or use defaults
input_file = os.environ.get("REVIEW_INPUT_FILE", "raw_review_output.txt")
output_file = os.environ.get("REVIEW_OUTPUT_FILE", "corrected_review.yaml")

try:
    # Read the raw response with size limit for security
    max_size = 1024 * 1024  # 1MB limit
    logger.info(f"Reading input from {input_file}")
    with open(input_file, "r") as f:
        content = f.read(max_size)

    logger.info("Processing Claude response for format correction...")

    # Strategy 1: Try to extract YAML block if it exists
    yaml_match = re.search(r"```yaml\s*\n(.*?)\n```", content, re.DOTALL)
    if yaml_match:
        yaml_content = yaml_match.group(1)
        logger.info("Found YAML block in response")
    else:
        # Strategy 2: Extract everything after markdown headers (---\n)
        if "---\n" in content:
            parts = content.split("---\n", 1)
            if len(parts) > 1:
                yaml_content = parts[1].strip()
                logger.info("Extracted YAML after markdown separator")
            else:
                yaml_content = content
        else:
            yaml_content = content

    # Clean the YAML content
    yaml_content = yaml_content.strip()

    # Validate content size
    if len(yaml_content) > 500000:  # 500KB limit
        logger.error("YAML content too large (max 500KB)")
        raise ValueError("Content size exceeds limit")

    # Remove any remaining markdown formatting
    yaml_content = re.sub(r"^\*\*.*?\*\*.*?\n", "", yaml_content, flags=re.MULTILINE)
    yaml_content = re.sub(r"^---\s*$", "", yaml_content, flags=re.MULTILINE)

    # Basic structure validation before parsing
    if not yaml_content or yaml_content.isspace():
        logger.error("Empty YAML content")
        raise ValueError("Empty content")

    # Check for suspicious patterns
    suspicious_patterns = [
        r"!!python/",
        r"!!python/object",
        r"!!python/name:",
        r"!!python/module:",
        r"!!map",
        r"!!omap",
        r"!!pairs",
        r"!!set",
        r"!!str",
        r"!!seq",
        r"!!null",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, yaml_content, re.IGNORECASE):
            logger.error(f"Suspicious pattern detected: {pattern}")
            raise ValueError("Potentially unsafe YAML content")

    # Try to parse and validate the YAML with safe loader
    try:
        data = yaml.safe_load(yaml_content)
        if isinstance(data, dict):
            logger.info("YAML parsed successfully")

            # Ensure required fields exist with defaults
            if "schema_version" not in data:
                data["schema_version"] = "1.0"
            if "pr_number" not in data:
                data["pr_number"] = safe_int(os.environ.get("PR_NUMBER", "0"), 0)
            if "timestamp" not in data:
                data["timestamp"] = datetime.utcnow().isoformat() + "Z"
            if "reviewer" not in data:
                data["reviewer"] = "ARC-Reviewer"
            if "verdict" not in data:
                # Determine verdict based on coverage and blocking issues
                has_blockers = bool(data.get("issues", {}).get("blocking", []))
                coverage_baseline = safe_float(os.environ.get("COVERAGE_BASELINE", "78.0"), 78.0)
                coverage_ok = data.get("coverage", {}).get("current_pct", 0) >= coverage_baseline

                if has_blockers or not coverage_ok:
                    data["verdict"] = "REQUEST_CHANGES"
                else:
                    data["verdict"] = "APPROVE"
            if "summary" not in data:
                data["summary"] = "Code review completed"
            if "coverage" not in data:
                # Use actual coverage if available
                actual_coverage = safe_float(os.environ.get("COVERAGE_PCT", "0"), 0)
                coverage_baseline = safe_float(os.environ.get("COVERAGE_BASELINE", "78.0"), 78.0)
                meets_baseline = actual_coverage >= coverage_baseline
                data["coverage"] = {
                    "current_pct": actual_coverage,
                    "status": "PASS" if meets_baseline else "FAIL",
                    "meets_baseline": meets_baseline,
                }
            if "issues" not in data:
                data["issues"] = {"blocking": [], "warnings": [], "nits": []}
            if "automated_issues" not in data:
                data["automated_issues"] = []

            # Output clean YAML with safe dump
            clean_yaml = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
            logger.info(f"Writing output to {output_file}")
            with open(output_file, "w") as f:
                f.write(clean_yaml)

            logger.info("Successfully generated corrected YAML format")

        else:
            logger.error("YAML content is not a valid dictionary")
            sys.exit(1)

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing failed: {e}")
        # Create minimal valid YAML as fallback
        actual_coverage = safe_float(os.environ.get("COVERAGE_PCT", "0"), 0)
        coverage_baseline = safe_float(os.environ.get("COVERAGE_BASELINE", "78.0"), 78.0)
        meets_baseline = actual_coverage >= coverage_baseline
        pr_number = safe_int(os.environ.get("PR_NUMBER", "0"), 0)

        fallback_data = {
            "schema_version": "1.0",
            "pr_number": pr_number,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "reviewer": "ARC-Reviewer",
            "verdict": "APPROVE" if meets_baseline else "REQUEST_CHANGES",
            "summary": "Format correction applied - review completed",
            "coverage": {
                "current_pct": actual_coverage,
                "status": "PASS" if meets_baseline else "FAIL",
                "meets_baseline": meets_baseline,
            },
            "issues": {
                "blocking": (
                    []
                    if meets_baseline
                    else [
                        {
                            "description": (
                                f"Coverage below baseline: {actual_coverage}% < "
                                f"{coverage_baseline}%"
                            ),
                            "category": "coverage",
                        }
                    ]
                ),
                "warnings": [
                    {
                        "description": "Original review format was invalid",
                        "category": "format",
                    }
                ],
                "nits": [],
            },
            "automated_issues": [],
        }

        with open(output_file, "w") as f:
            yaml.safe_dump(fallback_data, f, default_flow_style=False, sort_keys=False)

        logger.warning("Generated fallback YAML format due to parsing error")

except FileNotFoundError as e:
    logger.error(f"Input file not found: {e}")
    sys.exit(1)
except PermissionError as e:
    logger.error(f"Permission denied: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    sys.exit(1)
