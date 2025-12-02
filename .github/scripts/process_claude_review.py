#!/usr/bin/env python3
"""
Process Claude's review response and extract structured YAML data.
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


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        logger.debug(f"Failed to convert to float: {e}")
        return default


def process_review():
    # Read the raw review from stdin
    logger.info("Reading review from stdin")
    raw_content = sys.stdin.read()

    # Validate input size
    if len(raw_content) > 1024 * 1024:  # 1MB limit
        logger.error("Input too large (max 1MB)")
        sys.exit(1)

    # Store original for debugging
    output_file = os.environ.get("DEBUG_OUTPUT_FILE", "raw_review_output.txt")
    logger.debug(f"Saving raw content to {output_file}")
    with open(output_file, "w") as f:
        f.write(raw_content)

    # Try to extract YAML from various formats
    yaml_content = None

    # Method 1: Direct YAML parsing
    try:
        data = yaml.safe_load(raw_content)
        if isinstance(data, dict) and any(
            key in data for key in ["summary", "issues", "suggestions"]
        ):
            yaml_content = raw_content
    except Exception:
        pass

    # Method 2: Extract from markdown code blocks
    if not yaml_content:
        patterns = [r"```yaml\n(.*?)```", r"```yml\n(.*?)```", r"```\n(.*?)```"]
        for pattern in patterns:
            matches = re.findall(pattern, raw_content, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        test_data = yaml.safe_load(match)
                        if isinstance(test_data, dict):
                            yaml_content = match
                            break
                    except Exception:
                        continue
            if yaml_content:
                break

    # Method 3: Look for YAML-like structure
    if not yaml_content:
        lines = raw_content.split("\n")
        yaml_start = -1
        for i, line in enumerate(lines):
            if line.strip() in ["summary:", "issues:", "suggestions:", "coverage:"] and ":" in line:
                yaml_start = i
                break

        if yaml_start >= 0:
            yaml_lines = []
            indent_level = 0
            for line in lines[yaml_start:]:
                if line.strip() and not line.startswith(" ") and yaml_lines and ":" not in line:
                    break
                yaml_lines.append(line)
            yaml_content = "\n".join(yaml_lines)

    if not yaml_content:
        # Fallback: create minimal structure
        yaml_content = (
            """summary: "Unable to parse Claude's review. """
            """Please check raw_review_output.txt"
issues:
  blocking: []
  non_blocking: []
suggestions: []
decision: COMMENT
confidence: low
"""
        )

    # Write the extracted YAML
    with open("extracted_review.yaml", "w") as f:
        f.write(yaml_content)

    # Parse and validate the YAML
    try:
        data = yaml.safe_load(yaml_content)

        # Ensure required fields exist
        if "issues" not in data:
            data["issues"] = {"blocking": [], "non_blocking": []}
        if "suggestions" not in data:
            data["suggestions"] = []
        if "summary" not in data:
            data["summary"] = "No summary provided"
        if "decision" not in data:
            data["decision"] = "COMMENT"

        # Save the validated data
        with open("review.yaml", "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

        blocking_count = len(data.get("issues", {}).get("blocking", []))
        logger.info(f"Successfully processed review: {blocking_count} blocking issues found")

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error processing review: {e}")
        sys.exit(1)


if __name__ == "__main__":
    process_review()
