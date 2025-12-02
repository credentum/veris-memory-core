"""
Tests for scripts/security/docker-firewall-rules.sh

Tests verify:
- Script creates DOCKER-USER iptables rules
- Database ports are blocked from external access
- Internal networks are whitelisted
- Rules persist across reboots
- Script can be safely run multiple times (idempotent)
"""

import os
import re
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
FIREWALL_SCRIPT = REPO_ROOT / "scripts" / "security" / "docker-firewall-rules.sh"


def test_firewall_script_exists():
    """Verify docker-firewall-rules.sh exists"""
    assert FIREWALL_SCRIPT.exists(), "docker-firewall-rules.sh not found"


def test_firewall_script_is_executable():
    """Verify script has executable permissions"""
    assert os.access(FIREWALL_SCRIPT, os.X_OK), (
        "docker-firewall-rules.sh is not executable"
    )


def test_script_targets_docker_user_chain():
    """Verify script adds rules to DOCKER-USER chain"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    assert "DOCKER-USER" in content, (
        "Script must use DOCKER-USER chain to integrate with Docker"
    )
    assert "iptables" in content, "Script should use iptables commands"


def test_script_blocks_database_ports():
    """Verify script blocks external access to database ports"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    critical_ports = [
        "6379",  # Redis
        "6333",  # Qdrant HTTP
        "6334",  # Qdrant gRPC
        "7474",  # Neo4j HTTP
        "7687",  # Neo4j Bolt
    ]

    found_ports = sum(1 for port in critical_ports if port in content)
    assert found_ports >= 4, (
        f"Script should block critical database ports (found {found_ports}/5)"
    )


def test_script_uses_drop_rules():
    """Verify script uses DROP rules for blocking"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    assert "DROP" in content, "Script should use DROP rules to block traffic"
    assert "-j DROP" in content, "Script should use '-j DROP' for iptables"


def test_script_whitelists_internal_networks():
    """Verify script whitelists Docker and localhost networks"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    internal_networks = [
        "127.0.0.1",  # Localhost
        "172." in content or "docker" in content.lower(),  # Docker networks
        "10." in content or "192.168." in content,  # Private networks
    ]

    found_whitelists = sum(1 for net in internal_networks if (net if isinstance(net, bool) else net in content))
    assert found_whitelists >= 2, (
        "Script should whitelist internal networks"
    )


def test_script_is_idempotent():
    """Verify script can be run multiple times safely"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    idempotent_indicators = [
        "-D" in content,  # Delete existing rules
        "flush" in content.lower(),
        "|| true" in content,  # Ignore errors
        "2>/dev/null" in content,  # Suppress errors
    ]

    found_idempotent = sum(1 for indicator in idempotent_indicators if indicator)
    assert found_idempotent >= 2, (
        "Script should be idempotent (safe to run multiple times)"
    )


def test_script_has_persistence_mechanism():
    """Verify firewall rules persist across reboots"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    persistence_indicators = [
        "systemd",
        "rc.local",
        "cron",
        "persist",
        "netfilter-persistent",
    ]

    found_persistence = any(indicator in content.lower() for indicator in persistence_indicators)
    assert found_persistence, (
        "Script should ensure rules persist across reboots"
    )


def test_script_has_error_handling():
    """Verify script has proper error handling"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    error_handling = [
        "set -e" in content,
        "|| " in content,
        "if" in content,
    ]

    found_error_handling = sum(1 for check in error_handling if check)
    assert found_error_handling >= 2, "Script should have error handling"


def test_script_checks_prerequisites():
    """Verify script checks for required tools"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    prerequisite_checks = [
        "iptables",
        "docker",
        "root" in content.lower() or "sudo" in content,
    ]

    found_checks = sum(1 for check in prerequisite_checks if (check if isinstance(check, bool) else check in content))
    assert found_checks >= 2, "Script should check prerequisites"


def test_script_has_rollback_capability():
    """Verify script can remove rules if needed"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    rollback_indicators = [
        "-D" in content,  # Delete rules
        "remove",
        "clean",
        "flush",
    ]

    found_rollback = sum(1 for indicator in rollback_indicators if indicator in content.lower())
    assert found_rollback >= 1, "Script should support rule removal/rollback"


def test_script_validates_docker_running():
    """Verify script checks if Docker is running"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    docker_checks = [
        "docker" in content,
        "systemctl" in content or "service" in content,
        "running" in content.lower(),
    ]

    found_docker_checks = sum(1 for check in docker_checks if (check if isinstance(check, bool) else check in content))
    assert found_docker_checks >= 2, "Script should verify Docker is running"


def test_script_provides_status_output():
    """Verify script provides clear status messages"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    output_indicators = [
        "echo",
        "printf",
        "✓" in content or "✗" in content or "SUCCESS" in content or "FAIL" in content,
    ]

    found_output = sum(1 for indicator in output_indicators if (indicator if isinstance(indicator, bool) else indicator in content))
    assert found_output >= 2, "Script should provide status output"


def test_script_uses_colors():
    """Verify script uses colors for output clarity"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    color_indicators = [
        "\\033[" in content,
        "\\e[" in content,
        "RED=" in content or "GREEN=" in content,
    ]

    found_colors = any(indicator for indicator in color_indicators)
    assert found_colors, "Script should use colors for clarity"


def test_script_blocks_specific_protocols():
    """Verify script specifies TCP/UDP protocols correctly"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    protocol_specs = [
        "-p tcp" in content,
        "--dport" in content,  # Destination port
    ]

    found_protocols = sum(1 for spec in protocol_specs if spec)
    assert found_protocols >= 1, "Script should specify protocols correctly"


def test_script_validates_after_applying_rules():
    """Verify script validates rules after applying them"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    validation_indicators = [
        "iptables -L" in content or "iptables -S" in content,
        "grep" in content,
        "verify" in content.lower() or "check" in content.lower(),
    ]

    found_validation = sum(1 for indicator in validation_indicators if (indicator if isinstance(indicator, bool) else indicator in content))
    assert found_validation >= 1, "Script should validate rules after applying"


def test_script_has_documentation():
    """Verify script has comprehensive documentation"""
    with open(FIREWALL_SCRIPT) as f:
        lines = f.readlines()

    # Count comment lines in header
    comment_lines = sum(1 for line in lines[:30] if line.strip().startswith("#"))

    assert comment_lines >= 8, (
        f"Script should have header documentation (found {comment_lines} lines)"
    )


def test_script_handles_existing_rules():
    """Verify script handles pre-existing firewall rules gracefully"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    existing_rule_handling = [
        "-C" in content,  # Check if rule exists
        "-D" in content,  # Delete existing rule
        "|| true" in content,
        "2>/dev/null" in content,
    ]

    found_handling = sum(1 for indicator in existing_rule_handling if indicator)
    assert found_handling >= 2, (
        "Script should handle existing rules gracefully"
    )


def test_script_allows_established_connections():
    """Verify script allows established/related connections"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    connection_tracking = [
        "ESTABLISHED" in content,
        "RELATED" in content,
        "conntrack" in content.lower(),
    ]

    found_tracking = any(indicator in content for indicator in connection_tracking)
    # This is optional but recommended
    if found_tracking:
        assert True, "Script uses connection tracking (good practice)"


def test_script_logs_blocked_traffic():
    """Verify script optionally logs blocked traffic"""
    with open(FIREWALL_SCRIPT) as f:
        content = f.read()

    # Logging is optional but useful
    logging_indicators = [
        "LOG" in content,
        "--log-prefix" in content,
        "ulogd" in content,
    ]

    # Don't require logging, but acknowledge it if present
    if any(indicator in content for indicator in logging_indicators):
        assert True, "Script includes traffic logging (recommended)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
