"""
Tests for scripts/security/emergency-lockdown.sh

Tests verify:
- Script exists and is executable
- Pre-flight checks are comprehensive
- Backup procedures are implemented
- Rollback capability exists
- Security fixes are applied correctly
- Validation steps are included
"""

import os
import re
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
LOCKDOWN_SCRIPT = REPO_ROOT / "scripts" / "security" / "emergency-lockdown.sh"


def test_lockdown_script_exists():
    """Verify emergency-lockdown.sh exists"""
    assert LOCKDOWN_SCRIPT.exists(), "emergency-lockdown.sh not found"


def test_lockdown_script_is_executable():
    """Verify script has executable permissions"""
    assert os.access(LOCKDOWN_SCRIPT, os.X_OK), (
        "emergency-lockdown.sh is not executable"
    )


def test_script_has_shebang():
    """Verify script has proper shebang"""
    with open(LOCKDOWN_SCRIPT) as f:
        first_line = f.readline()

    assert first_line.startswith("#!"), "Script missing shebang"
    assert "bash" in first_line, "Script should use bash"


def test_script_has_error_handling():
    """Verify script has proper error handling"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    # Check for bash strict mode
    assert "set -e" in content or "set -euo pipefail" in content, (
        "Script should exit on error (set -e)"
    )
    assert "set -u" in content or "set -euo pipefail" in content, (
        "Script should exit on undefined variables (set -u)"
    )


def test_preflight_checks_included():
    """Verify comprehensive pre-flight checks"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    required_checks = [
        "root",  # Check for root/sudo
        "docker",  # Check Docker is installed
        "docker-compose",  # Check docker-compose is available
        "NEO4J_PASSWORD",  # Check required env vars
    ]

    for check in required_checks:
        assert check in content, f"Missing pre-flight check for: {check}"


def test_backup_creation():
    """Verify backup is created before lockdown"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    backup_indicators = [
        "backup",
        "BACKUP_DIR",
        ".backup",
        "cp ",  # File copy for backup
    ]

    found_backup = any(indicator in content for indicator in backup_indicators)
    assert found_backup, "Script should create backup before making changes"


def test_rollback_capability():
    """Verify rollback procedures exist"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    rollback_indicators = [
        "rollback",
        "restore",
        "failure",
        "revert"
    ]

    found_rollback = any(indicator in content.lower() for indicator in rollback_indicators)
    assert found_rollback, "Script should have rollback capability"


def test_logging_configured():
    """Verify comprehensive logging is configured"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    assert "LOG_DIR" in content or "LOG_FILE" in content, (
        "Script should configure logging directory"
    )
    assert "tee" in content or "exec" in content, (
        "Script should log output to file"
    )


def test_redis_password_generation():
    """Verify script generates secure Redis password"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    assert "REDIS_PASSWORD" in content, "Script should handle REDIS_PASSWORD"
    assert "openssl rand" in content, "Should use openssl for password generation"
    assert "base64" in content, "Should use base64 encoding"


def test_docker_compose_deployment():
    """Verify script deploys secure docker-compose configuration"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    assert "docker-compose" in content or "docker compose" in content, (
        "Script should use docker compose"
    )
    assert "down" in content, "Should stop containers"
    assert "up" in content, "Should start containers"


def test_security_validation():
    """Verify script validates security after deployment"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    validation_checks = [
        "netstat",  # Check port bindings
        "127.0.0.1",  # Verify localhost binding
        "redis-cli",  # Test Redis auth
        "health",  # Health checks
    ]

    found_validations = sum(1 for check in validation_checks if check in content)
    assert found_validations >= 2, (
        f"Script should include security validation (found {found_validations}/4 checks)"
    )


def test_external_port_verification():
    """Verify script checks external port exposure"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    port_check_indicators = [
        "nmap",
        "netstat",
        "ss ",  # Socket statistics
        "lsof",
    ]

    found_port_check = any(indicator in content for indicator in port_check_indicators)
    assert found_port_check, "Script should verify port exposure"


def test_service_health_checks():
    """Verify script checks service health after deployment"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    assert "docker ps" in content or "docker compose ps" in content, (
        "Script should check container status"
    )
    assert "health" in content.lower(), "Script should verify service health"


def test_color_output_for_clarity():
    """Verify script uses colored output for important messages"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    color_indicators = [
        "RED=",
        "GREEN=",
        "YELLOW=",
        "\\033[",  # ANSI color codes
    ]

    found_colors = any(indicator in content for indicator in color_indicators)
    assert found_colors, "Script should use colors for visual clarity"


def test_critical_warning_displayed():
    """Verify script displays critical warning before execution"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    warning_indicators = [
        "CRITICAL",
        "WARNING",
        "EMERGENCY",
        "downtime",
    ]

    found_warnings = sum(1 for indicator in warning_indicators if indicator.upper() in content.upper())
    assert found_warnings >= 2, "Script should display critical warnings"


def test_script_version_metadata():
    """Verify script includes version metadata"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    metadata_items = [
        "VERSION",
        "DATE",
        "AUTHOR" in content or "Purpose:" in content,
    ]

    found_metadata = sum(1 for item in metadata_items if (item if isinstance(item, bool) else item in content))
    assert found_metadata >= 1, "Script should include version/metadata"


def test_required_commands_check():
    """Verify script checks for required commands"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    assert "command -v" in content or "which" in content, (
        "Script should check for required commands"
    )


def test_neo4j_password_verification():
    """Verify script verifies NEO4J_PASSWORD is set"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    assert 'NEO4J_PASSWORD' in content, "Script should check NEO4J_PASSWORD"
    assert '-z' in content, "Script should check for empty variables"


def test_completion_message():
    """Verify script displays completion message"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    completion_indicators = [
        "complete",
        "success",
        "finished",
        "done",
    ]

    found_completion = any(indicator in content.lower() for indicator in completion_indicators)
    assert found_completion, "Script should display completion message"


def test_time_tracking():
    """Verify script tracks execution time"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    time_tracking = [
        "START_TIME" in content or "start_time" in content,
        "date" in content,
        "time" in content.lower(),
    ]

    assert any(time_tracking), "Script should track execution time"


def test_script_documentation():
    """Verify script has comprehensive documentation"""
    with open(LOCKDOWN_SCRIPT) as f:
        lines = f.readlines()

    # Count comment lines in first 30 lines (header documentation)
    comment_lines = sum(1 for line in lines[:30] if line.strip().startswith("#"))

    assert comment_lines >= 10, (
        f"Script should have comprehensive header documentation (found {comment_lines} comment lines)"
    )


def test_no_hardcoded_credentials():
    """Verify no credentials are hardcoded in script"""
    with open(LOCKDOWN_SCRIPT) as f:
        content = f.read()

    # Patterns that might indicate hardcoded credentials
    insecure_patterns = [
        r'PASSWORD=["\'](?!\$)[a-zA-Z0-9]{8,}["\']',
        r'password=["\'](?!\$)[a-zA-Z0-9]{8,}["\']',
    ]

    issues = []
    for pattern in insecure_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.extend(matches)

    assert not issues, f"Found potentially hardcoded credentials: {issues}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
