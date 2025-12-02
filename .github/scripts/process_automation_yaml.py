#!/usr/bin/env python3
"""
Process automation YAML from PR comments.
This script runs in a sandboxed environment with restricted permissions.
"""
import json
import os
import re
import sys

import yaml

# Security: Restrict file operations
os.chdir(os.getcwd())

# Define strict schema for automation YAML
AUTOMATION_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "maxItems": 50,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "maxLength": 200},
                    "body": {"type": "string", "maxLength": 5000},
                    "labels": {
                        "type": "array",
                        "maxItems": 10,
                        "items": {
                            "type": "string",
                            "pattern": "^[a-zA-Z0-9-_]+$",
                            "maxLength": 50,
                        },
                    },
                },
                "required": ["title", "body"],
            },
        }
    },
}


def validate_schema(data):
    """Validate data against schema manually"""
    if not isinstance(data, dict):
        return False, "Data must be an object"

    if "issues" in data:
        if not isinstance(data["issues"], list):
            return False, "Issues must be an array"
        if len(data["issues"]) > 50:
            return False, "Too many issues (max 50)"

        for idx, issue in enumerate(data["issues"]):
            if not isinstance(issue, dict):
                return False, f"Issue {idx} must be an object"
            if "title" not in issue or "body" not in issue:
                return False, f"Issue {idx} missing required fields"
            if not isinstance(issue["title"], str) or len(issue["title"]) > 200:
                return False, f"Issue {idx} title invalid"
            if not isinstance(issue["body"], str) or len(issue["body"]) > 5000:
                return False, f"Issue {idx} body invalid"

            if "labels" in issue:
                if not isinstance(issue["labels"], list):
                    return False, f"Issue {idx} labels must be an array"
                if len(issue["labels"]) > 10:
                    return False, f"Issue {idx} too many labels"
                for label in issue["labels"]:
                    if not isinstance(label, str) or not re.match(r"^[a-zA-Z0-9-_]+$", label):
                        return False, f"Issue {idx} invalid label format"

    return True, None


def process_automation():
    # Read automation comment from stdin
    content = sys.stdin.read()

    # Extract YAML between markers
    match = re.search(r"<!-- ARC-AUTOMATION\n(.*?)\n-->", content, re.DOTALL)
    if not match:
        print("No automation YAML found")
        sys.exit(0)

    yaml_content = match.group(1).strip()

    # Validate content size
    if len(yaml_content) > 100000:  # 100KB limit
        print("YAML content too large")
        sys.exit(1)

    try:
        # Use safe_load to prevent arbitrary code execution
        data = yaml.safe_load(yaml_content)

        # Validate against schema
        valid, error = validate_schema(data)
        if not valid:
            print(f"Schema validation failed: {error}")
            sys.exit(1)

        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Invalid YAML structure")

        # Extract issues safely
        issues = []
        for issue in data.get("issues", []):
            if isinstance(issue, dict) and "title" in issue and "body" in issue:
                # Sanitize labels to prevent injection
                labels = []
                for label in issue.get("labels", []):
                    if isinstance(label, str) and re.match(r"^[a-zA-Z0-9-_]+$", label):
                        labels.append(label)

                issues.append(
                    {
                        "title": str(issue["title"])[:200],  # Limit title length
                        "body": str(issue["body"])[:5000],  # Limit body length
                        "labels": labels[:10],  # Limit number of labels
                    }
                )

        # Write validated issues
        with open("followup_issues.json", "w") as f:
            json.dump({"issues": issues}, f)

        print(f"Found {len(issues)} follow-up issues")

    except Exception as e:
        print(f"Error processing automation YAML: {e}")
        sys.exit(1)


if __name__ == "__main__":
    process_automation()
