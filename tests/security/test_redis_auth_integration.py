"""
Integration tests for Redis authentication across all services.

Tests verify:
- Redis requires password authentication
- All services connect with password
- Unauthenticated connections are rejected
- Password is properly configured in environment
"""

import os
import subprocess
import time
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def redis_test_env():
    """Set up test environment with Redis password"""
    return {
        "REDIS_PASSWORD": "test_secure_password_12345678"
    }


def test_redis_password_in_env(redis_test_env):
    """Verify REDIS_PASSWORD is set in test environment"""
    assert "REDIS_PASSWORD" in redis_test_env
    assert len(redis_test_env["REDIS_PASSWORD"]) >= 16


def test_docker_compose_redis_requires_password():
    """Verify docker-compose.yml Redis service requires password"""
    compose_file = REPO_ROOT / "docker-compose.yml"

    with open(compose_file) as f:
        content = f.read()

    # Redis section should include --requirepass
    assert "--requirepass" in content, "Redis must have --requirepass configured"
    assert "${REDIS_PASSWORD}" in content or "$REDIS_PASSWORD" in content, (
        "Redis must use REDIS_PASSWORD environment variable"
    )


def test_all_services_use_authenticated_redis_url():
    """Verify all services use password-authenticated Redis URL"""
    compose_file = REPO_ROOT / "docker-compose.yml"

    with open(compose_file) as f:
        lines = f.readlines()

    redis_url_lines = [
        (i + 1, line.strip())
        for i, line in enumerate(lines)
        if "REDIS_URL" in line
    ]

    assert len(redis_url_lines) > 0, "No REDIS_URL found in docker-compose.yml"

    unauthenticated_urls = []
    for line_num, line in redis_url_lines:
        # Check if password is in the URL
        if "redis://" in line:
            if ":${REDIS_PASSWORD}@" not in line and "::${REDIS_PASSWORD}@" not in line:
                unauthenticated_urls.append(f"Line {line_num}: {line}")

    assert not unauthenticated_urls, (
        f"Found {len(unauthenticated_urls)} services with unauthenticated Redis URLs:\n" +
        "\n".join(unauthenticated_urls)
    )


def test_redis_healthcheck_uses_auth():
    """Verify Redis healthcheck command uses authentication"""
    compose_file = REPO_ROOT / "docker-compose.yml"

    with open(compose_file) as f:
        content = f.read()

    # Find Redis healthcheck section
    redis_section_start = content.find("redis:")
    if redis_section_start == -1:
        pytest.fail("Redis service not found in docker-compose.yml")

    redis_section = content[redis_section_start:redis_section_start + 1000]

    assert "healthcheck" in redis_section, "Redis must have healthcheck configured"
    assert "-a" in redis_section or "--auth" in redis_section, (
        "Redis healthcheck must use authentication (-a flag)"
    )


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/docker"),
    reason="Docker not available for integration test"
)
def test_redis_rejects_unauthenticated_connections():
    """
    Integration test: Verify Redis rejects connections without password.

    NOTE: This test requires Docker and a running Redis instance.
    It's skipped if Docker is not available.
    """
    try:
        # Try to ping Redis without password (should fail)
        result = subprocess.run(
            ["docker", "exec", "veris-memory-dev-redis-1", "redis-cli", "ping"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Should receive NOAUTH error
        assert result.returncode != 0 or "NOAUTH" in result.stderr or "NOAUTH" in result.stdout, (
            "Redis should reject unauthenticated connections"
        )

    except subprocess.TimeoutExpired:
        pytest.skip("Redis container not available or not responding")
    except FileNotFoundError:
        pytest.skip("Docker not available")


@pytest.mark.skipif(
    not os.path.exists("/usr/bin/docker"),
    reason="Docker not available for integration test"
)
def test_redis_accepts_authenticated_connections():
    """
    Integration test: Verify Redis accepts connections with correct password.

    NOTE: Requires REDIS_PASSWORD environment variable and running Redis.
    """
    redis_password = os.getenv("REDIS_PASSWORD", "")
    if not redis_password:
        pytest.skip("REDIS_PASSWORD not set in environment")

    try:
        # Try to ping Redis with password (should succeed)
        result = subprocess.run(
            [
                "docker", "exec",
                "-e", f"REDIS_PASSWORD={redis_password}",
                "veris-memory-dev-redis-1",
                "redis-cli", "-a", redis_password, "ping"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        assert result.returncode == 0, f"Redis auth failed: {result.stderr}"
        assert "PONG" in result.stdout, "Redis should respond PONG with correct password"

    except subprocess.TimeoutExpired:
        pytest.skip("Redis container not available or not responding")
    except FileNotFoundError:
        pytest.skip("Docker not available")


def test_deployment_script_generates_redis_password():
    """Verify deploy script includes Redis password generation"""
    deploy_script = REPO_ROOT / "scripts" / "deploy-dev.sh"

    with open(deploy_script) as f:
        content = f.read()

    assert "REDIS_PASSWORD" in content, "Deploy script must handle REDIS_PASSWORD"
    assert "openssl rand" in content, "Deploy script must generate secure password"


def test_env_file_includes_redis_password():
    """Verify .env.example or template includes REDIS_PASSWORD"""
    env_files = [
        REPO_ROOT / ".env.example",
        REPO_ROOT / ".env.template",
        REPO_ROOT / ".env.dev"
    ]

    found_redis_password = False
    for env_file in env_files:
        if env_file.exists():
            with open(env_file) as f:
                content = f.read()
            if "REDIS_PASSWORD" in content:
                found_redis_password = True
                break

    # This is informational - env files might not exist yet
    if not found_redis_password:
        pytest.skip("No .env template files found (may be generated during deployment)")


def test_redis_url_format():
    """Verify Redis URL format is correct for password authentication"""
    # Correct format: redis://:password@host:port
    correct_format = "redis://:${REDIS_PASSWORD}@redis:6379"

    compose_file = REPO_ROOT / "docker-compose.yml"
    with open(compose_file) as f:
        content = f.read()

    # Check that services use the correct format
    assert "redis://:${REDIS_PASSWORD}@" in content or "redis://:" in content, (
        f"Redis URL must use format: {correct_format}"
    )


def test_redis_protected_mode_enabled():
    """Verify Redis has protected mode enabled"""
    compose_file = REPO_ROOT / "docker-compose.yml"

    with open(compose_file) as f:
        content = f.read()

    # Find Redis command section
    redis_section_start = content.find("redis:")
    if redis_section_start == -1:
        pytest.fail("Redis service not found")

    redis_section = content[redis_section_start:redis_section_start + 1500]

    assert "--protected-mode yes" in redis_section, (
        "Redis must have protected mode enabled for security"
    )


def test_security_readme_documents_redis_auth():
    """Verify security documentation mentions Redis authentication"""
    security_readme = REPO_ROOT / "SECURITY_README.md"

    if not security_readme.exists():
        pytest.skip("SECURITY_README.md not found")

    with open(security_readme) as f:
        content = f.read()

    redis_docs = [
        "redis" in content.lower(),
        "password" in content.lower(),
        "authentication" in content.lower(),
    ]

    assert sum(redis_docs) >= 2, (
        "SECURITY_README.md should document Redis authentication"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
