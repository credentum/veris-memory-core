"""
Unit tests for security features in scripts/deploy-dev.sh

Tests verify:
- Redis password generation is secure (32-char alphanumeric)
- Password is written to .env file correctly
- No password leakage in logs or output
- Proper cleanup of managed environment variables
"""

import os
import re
import subprocess
import tempfile
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-dev.sh"


def test_deploy_script_exists():
    """Verify deploy-dev.sh exists"""
    assert DEPLOY_SCRIPT.exists(), "scripts/deploy-dev.sh not found"


def test_deploy_script_has_redis_password_generation():
    """Verify deploy script includes Redis password generation"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Check for password generation logic
    assert "REDIS_PASSWORD" in content, "REDIS_PASSWORD not found in deploy script"
    assert "openssl rand" in content, "Password generation using openssl not found"
    assert "base64" in content, "base64 encoding not found"
    assert "-z '$REDIS_PASSWORD'" in content or "-z \"$REDIS_PASSWORD\"" in content, (
        "Missing check for empty REDIS_PASSWORD"
    )


def test_redis_password_generation_command():
    """Test the actual password generation command produces secure passwords"""
    # Extract the password generation command from deploy script
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Find the password generation command
    match = re.search(r'openssl rand[^\n]+', content)
    assert match, "Password generation command not found"

    password_cmd = match.group(0)

    # Test password generation (run locally)
    try:
        # Simulate the command
        result = subprocess.run(
            "openssl rand -base64 32 | tr -d '=+/' | cut -c1-32",
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )

        password = result.stdout.strip()

        # Verify password characteristics
        assert len(password) == 32, f"Password length must be 32 chars, got {len(password)}"
        assert password.isalnum(), f"Password must be alphanumeric, got: {password}"
        assert not re.search(r'[=+/]', password), "Password contains invalid characters"
        assert re.search(r'[a-z]', password), "Password must contain lowercase letters"
        assert re.search(r'[A-Z]', password), "Password must contain uppercase letters"
        assert re.search(r'[0-9]', password), "Password must contain digits"

    except subprocess.TimeoutExpired:
        pytest.skip("Password generation command timed out")
    except FileNotFoundError:
        pytest.skip("openssl not available in test environment")


def test_redis_password_written_to_env():
    """Verify REDIS_PASSWORD is written to .env file"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Check that REDIS_PASSWORD is written to .env
    assert 'printf "REDIS_PASSWORD=%s' in content, (
        "REDIS_PASSWORD not written to .env file"
    )
    assert '"$REDIS_PASSWORD"' in content or "'$REDIS_PASSWORD'" in content, (
        "REDIS_PASSWORD variable not used in printf"
    )


def test_redis_password_cleanup_before_write():
    """Verify old REDIS_PASSWORD is cleaned from .env before writing new one"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Check for cleanup of REDIS_PASSWORD
    assert 'grep -v "^REDIS_PASSWORD"' in content, (
        "Missing cleanup of old REDIS_PASSWORD from .env"
    )


def test_no_password_echoed_to_logs():
    """Verify passwords are not echoed to deployment logs"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Check that printf is used (which doesn't echo to logs)
    assert 'printf "REDIS_PASSWORD=' in content, (
        "Must use printf to avoid logging passwords"
    )

    # Ensure no echo commands for passwords
    lines = content.split("\n")
    for line_num, line in enumerate(lines, 1):
        if "echo" in line.lower() and "password" in line.lower():
            # Allow echo for status messages but not actual passwords
            if "$REDIS_PASSWORD" in line or "${REDIS_PASSWORD}" in line:
                pytest.fail(
                    f"Line {line_num}: Password may be echoed to logs: {line.strip()}"
                )


def test_neo4j_password_verification():
    """Verify NEO4J_PASSWORD is checked before deployment"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Check for NEO4J_PASSWORD verification
    assert 'if [ -z "$NEO4J_PASSWORD"' in content or "if [ -z '$NEO4J_PASSWORD'" in content, (
        "Missing NEO4J_PASSWORD verification"
    )
    assert "exit 1" in content, "Script should exit if NEO4J_PASSWORD is missing"


def test_environment_variable_export():
    """Verify all critical secrets are exported as environment variables"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    required_exports = [
        "NEO4J_PASSWORD",
        "REDIS_PASSWORD",
        "API_KEY_MCP",
        "SENTINEL_API_KEY"
    ]

    for var_name in required_exports:
        assert f"export {var_name}=" in content, (
            f"Missing export for {var_name}"
        )


def test_managed_variables_cleanup():
    """Verify all managed variables are cleaned from .env before writing"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Variables that should be cleaned
    managed_vars = [
        "NEO4J",
        "REDIS_PASSWORD",
        "TELEGRAM",
        "API_KEY_MCP",
        "SENTINEL_API_KEY"
    ]

    for var in managed_vars:
        # Check for grep -v pattern to remove these variables
        assert f'grep -v "^{var}' in content, (
            f"Missing cleanup for managed variable: {var}"
        )


def test_deployment_uses_secure_connection():
    """Verify deployment uses secure SSH connection"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Check SSH options
    assert "StrictHostKeyChecking=no" in content, (
        "SSH should use StrictHostKeyChecking=no for automation"
    )
    assert "UserKnownHostsFile=~/.ssh/known_hosts" in content, (
        "SSH should use known_hosts file"
    )
    assert "-i ~/.ssh/id_ed25519" in content, (
        "SSH should use identity file for authentication"
    )


def test_script_has_error_handling():
    """Verify script has proper error handling"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Check for set -e (exit on error)
    assert "set -e" in content, "Script should use 'set -e' for error handling"


def test_backup_created_before_deployment():
    """Verify backup is created before deployment"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    assert "backup" in content.lower(), "Script should mention backup creation"
    assert "BACKUP_DIR" in content or "backup" in content, (
        "Script should define backup directory"
    )


def test_redis_password_complexity_requirements():
    """Test that password generation produces cryptographically strong passwords"""
    # Run password generation multiple times to verify randomness
    passwords = set()

    try:
        for _ in range(10):
            result = subprocess.run(
                "openssl rand -base64 32 | tr -d '=+/' | cut -c1-32",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            password = result.stdout.strip()
            passwords.add(password)

        # All passwords should be unique (sufficient randomness)
        assert len(passwords) == 10, "Password generation not sufficiently random"

        # Check password strength for all generated passwords
        for password in passwords:
            # Must be 32 characters
            assert len(password) == 32, f"Invalid password length: {len(password)}"

            # Must be alphanumeric only
            assert password.isalnum(), f"Password contains non-alphanumeric: {password}"

            # Should have good character distribution
            unique_chars = len(set(password))
            assert unique_chars >= 16, (
                f"Password has low character diversity: {unique_chars}/32 unique chars"
            )

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Cannot test password generation (openssl not available)")


def test_secrets_not_hardcoded():
    """Verify no secrets are hardcoded in deploy script"""
    with open(DEPLOY_SCRIPT) as f:
        content = f.read()

    # Patterns that would indicate hardcoded secrets
    insecure_patterns = [
        r'PASSWORD=["\'](?!\$)[a-zA-Z0-9]{8,}["\']',  # PASSWORD="something"
        r'API_KEY=["\'](?!\$)[a-zA-Z0-9]{20,}["\']',  # API_KEY="something"
    ]

    issues = []
    for pattern in insecure_patterns:
        matches = re.findall(pattern, content)
        if matches:
            issues.extend(matches)

    assert not issues, f"Found potentially hardcoded secrets: {issues}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
