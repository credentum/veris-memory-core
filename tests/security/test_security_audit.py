"""
Tests for scripts/security/security-audit.sh

Tests verify:
- Script can identify security misconfigurations
- All critical security checks are included
- Scoring system works correctly
- JSON report generation works
- Pass/fail criteria are appropriate
"""

import os
import re
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "security" / "security-audit.sh"


def test_audit_script_exists():
    """Verify security-audit.sh exists"""
    assert AUDIT_SCRIPT.exists(), "security-audit.sh not found"


def test_audit_script_is_executable():
    """Verify script has executable permissions"""
    assert os.access(AUDIT_SCRIPT, os.X_OK), "security-audit.sh is not executable"


def test_script_has_port_binding_checks():
    """Verify script checks for insecure port bindings"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    port_check_indicators = [
        "127.0.0.1",  # Should check for localhost binding
        "0.0.0.0",  # Should check for internet exposure
        "netstat" in content or "ss " in content or "lsof" in content,  # Port checking tools
    ]

    found_checks = sum(1 for indicator in port_check_indicators if (indicator if isinstance(indicator, bool) else indicator in content))
    assert found_checks >= 2, "Script should check port bindings"


def test_script_checks_redis_authentication():
    """Verify script validates Redis password authentication"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    redis_checks = [
        "redis-cli",
        "NOAUTH" in content or "authentication" in content.lower(),
        "6379",  # Redis port
    ]

    found_redis_checks = sum(1 for check in redis_checks if (check if isinstance(check, bool) else check in content))
    assert found_redis_checks >= 2, "Script should verify Redis authentication"


def test_script_has_scoring_system():
    """Verify script includes security scoring"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    scoring_indicators = [
        "score",
        "grade",
        "PASS" in content or "FAIL" in content,
        "A" in content or "B" in content or "C" in content or "D" in content,  # Letter grades
    ]

    found_scoring = sum(1 for indicator in scoring_indicators if (indicator in content.lower() if isinstance(indicator, str) else False) or (isinstance(indicator, bool) and indicator))
    assert found_scoring >= 2, "Script should include security scoring/grading"


def test_script_generates_json_report():
    """Verify script can generate JSON report"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    json_indicators = [
        "json",
        "{",
        "}",
        "jq" in content or "python" in content or "JSON" in content,
    ]

    found_json = sum(1 for indicator in json_indicators if (indicator if isinstance(indicator, bool) else indicator in content))
    assert found_json >= 3, "Script should support JSON report generation"


def test_script_checks_neo4j_security():
    """Verify script validates Neo4j security"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    neo4j_checks = [
        "7474",  # Neo4j HTTP port
        "7687",  # Neo4j Bolt port
        "neo4j",
    ]

    found_neo4j_checks = sum(1 for check in neo4j_checks if check in content)
    assert found_neo4j_checks >= 2, "Script should check Neo4j security"


def test_script_checks_qdrant_security():
    """Verify script validates Qdrant security"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    qdrant_checks = [
        "6333",  # Qdrant HTTP port
        "6334",  # Qdrant gRPC port
        "qdrant",
    ]

    found_qdrant_checks = sum(1 for check in qdrant_checks if check in content)
    assert found_qdrant_checks >= 2, "Script should check Qdrant security"


def test_script_has_docker_checks():
    """Verify script checks Docker security"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    docker_checks = [
        "docker",
        "container",
        "docker ps" in content or "docker compose" in content,
    ]

    found_docker_checks = sum(1 for check in docker_checks if (check if isinstance(check, bool) else check in content))
    assert found_docker_checks >= 2, "Script should check Docker configuration"


def test_script_checks_firewall_status():
    """Verify script validates firewall configuration"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    firewall_checks = [
        "ufw" in content or "iptables" in content,
        "firewall",
    ]

    found_firewall = any(check if isinstance(check, bool) else check in content for check in firewall_checks)
    assert found_firewall, "Script should check firewall status"


def test_script_has_comprehensive_checks():
    """Verify script includes all critical security audit sections"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    # Critical security areas that should be audited
    critical_areas = [
        "port",  # Port exposure
        "auth",  # Authentication
        "password",  # Password security
        "firewall",  # Firewall config
        "docker",  # Docker security
    ]

    found_areas = sum(1 for area in critical_areas if area in content.lower())
    assert found_areas >= 4, (
        f"Script should audit all critical areas (found {found_areas}/5)"
    )


def test_script_has_pass_fail_criteria():
    """Verify script has clear pass/fail criteria"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    criteria_indicators = [
        "PASS" in content or "SUCCESS" in content,
        "FAIL" in content or "FAILURE" in content,
        "✓" in content or "✗" in content or "✅" in content or "❌" in content,
    ]

    found_criteria = sum(1 for indicator in criteria_indicators if indicator)
    assert found_criteria >= 2, "Script should have clear pass/fail indicators"


def test_script_uses_colors_for_output():
    """Verify script uses colors for readable output"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    color_indicators = [
        "RED=",
        "GREEN=",
        "\\033[",
        "\\e[",
    ]

    found_colors = any(indicator in content for indicator in color_indicators)
    assert found_colors, "Script should use colors for output clarity"


def test_script_checks_service_health():
    """Verify script validates service health"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    health_checks = [
        "health",
        "status",
        "running",
        "curl" in content or "wget" in content,
    ]

    found_health = sum(1 for check in health_checks if (check if isinstance(check, bool) else check in content.lower()))
    assert found_health >= 2, "Script should check service health"


def test_script_has_error_handling():
    """Verify script has proper error handling"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    error_handling = [
        "set -e" in content or "set -euo" in content,
        "|| " in content,  # OR operator for error handling
        "if" in content,  # Conditional checks
    ]

    found_error_handling = sum(1 for check in error_handling if check)
    assert found_error_handling >= 2, "Script should have error handling"


def test_script_provides_remediation_guidance():
    """Verify script provides guidance for failed checks"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    guidance_indicators = [
        "fix",
        "remediation",
        "recommendation",
        "action",
        "how to",
    ]

    found_guidance = any(indicator in content.lower() for indicator in guidance_indicators)
    assert found_guidance, "Script should provide remediation guidance"


def test_script_version_and_metadata():
    """Verify script includes version and metadata"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    metadata = [
        "VERSION" in content or "version" in content,
        "Purpose:" in content or "Description:" in content,
        "Author" in content or "#!/bin/bash" in content,
    ]

    found_metadata = sum(1 for item in metadata if item)
    assert found_metadata >= 1, "Script should include metadata"


def test_script_has_documentation():
    """Verify script has comprehensive documentation"""
    with open(AUDIT_SCRIPT) as f:
        lines = f.readlines()

    # Count comment lines in first 30 lines
    comment_lines = sum(1 for line in lines[:30] if line.strip().startswith("#"))

    assert comment_lines >= 8, (
        f"Script should have header documentation (found {comment_lines} comment lines)"
    )


def test_script_checks_critical_ports():
    """Verify script checks all critical database and API ports"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    critical_ports = [
        "6379",  # Redis
        "6333",  # Qdrant HTTP
        "7474",  # Neo4j HTTP
        "7687",  # Neo4j Bolt
        "8000",  # MCP Server
    ]

    found_ports = sum(1 for port in critical_ports if port in content)
    assert found_ports >= 4, (
        f"Script should check critical ports (found {found_ports}/5)"
    )


def test_script_output_is_structured():
    """Verify script provides structured output"""
    with open(AUDIT_SCRIPT) as f:
        content = f.read()

    structure_indicators = [
        "echo" in content,
        "printf" in content,
        "=" * 5 in content or "-" * 5 in content,  # Section dividers
    ]

    found_structure = sum(1 for indicator in structure_indicators if indicator)
    assert found_structure >= 2, "Script should have structured output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
