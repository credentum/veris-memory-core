"""
Integration tests for docker-compose.yml security configurations.

Tests verify:
- All ports bound to localhost (127.0.0.1) only
- Redis requires password authentication
- No services exposed to 0.0.0.0 (internet)
- All environment variables correctly configured
"""

import os
import re
import yaml
import pytest
from pathlib import Path


# Get docker-compose.yml path
REPO_ROOT = Path(__file__).parent.parent.parent
DOCKER_COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"


@pytest.fixture
def docker_compose_config():
    """Load and parse docker-compose.yml"""
    with open(DOCKER_COMPOSE_FILE) as f:
        return yaml.safe_load(f)


def test_docker_compose_file_exists():
    """Verify docker-compose.yml exists"""
    assert DOCKER_COMPOSE_FILE.exists(), "docker-compose.yml not found"


def test_all_ports_localhost_bound(docker_compose_config):
    """CRITICAL: Verify all ports are bound to localhost (127.0.0.1) only"""
    services = docker_compose_config.get("services", {})
    issues = []

    for service_name, service_config in services.items():
        ports = service_config.get("ports", [])

        for port_mapping in ports:
            port_str = str(port_mapping)

            # Check if port is bound to 0.0.0.0 or has no host binding
            if not port_str.startswith("127.0.0.1:"):
                # Allow comments in port mappings
                if "#" in port_str and "127.0.0.1:" in port_str.split("#")[0]:
                    continue

                issues.append(
                    f"Service '{service_name}' has insecure port binding: {port_str}. "
                    f"Must start with '127.0.0.1:' to bind to localhost only."
                )

    assert not issues, f"Found {len(issues)} insecure port bindings:\n" + "\n".join(issues)


def test_redis_password_required(docker_compose_config):
    """CRITICAL: Verify Redis requires password authentication"""
    redis_service = docker_compose_config["services"].get("redis")
    assert redis_service, "Redis service not found in docker-compose.yml"

    # Check command includes --requirepass
    command = redis_service.get("command", "")
    command_str = " ".join(command) if isinstance(command, list) else str(command)

    assert "--requirepass" in command_str, (
        "Redis command must include '--requirepass ${REDIS_PASSWORD}' for authentication"
    )
    assert "${REDIS_PASSWORD}" in command_str or "$REDIS_PASSWORD" in command_str, (
        "Redis must use REDIS_PASSWORD environment variable"
    )
    assert "--protected-mode yes" in command_str, (
        "Redis must have protected mode enabled"
    )


def test_redis_healthcheck_uses_auth(docker_compose_config):
    """CRITICAL: Verify Redis healthcheck uses password authentication"""
    redis_service = docker_compose_config["services"].get("redis")
    assert redis_service, "Redis service not found"

    healthcheck = redis_service.get("healthcheck", {})
    test_command = healthcheck.get("test", [])
    test_str = " ".join(test_command) if isinstance(test_command, list) else str(test_command)

    assert "-a" in test_str or "--auth" in test_str, (
        "Redis healthcheck must use '-a' flag for password authentication"
    )
    assert "${REDIS_PASSWORD}" in test_str or "$REDIS_PASSWORD" in test_str, (
        "Redis healthcheck must use REDIS_PASSWORD variable"
    )


def test_all_services_use_redis_password(docker_compose_config):
    """CRITICAL: Verify all services connecting to Redis use password authentication"""
    services = docker_compose_config.get("services", {})
    issues = []

    for service_name, service_config in services.items():
        env_vars = service_config.get("environment", [])

        # Convert to dict if list
        if isinstance(env_vars, list):
            env_dict = {}
            for item in env_vars:
                if "=" in str(item):
                    key, value = str(item).split("=", 1)
                    env_dict[key.strip()] = value.strip()
        else:
            env_dict = env_vars

        # Check if service uses Redis
        redis_url = env_dict.get("REDIS_URL", "")

        if redis_url and "redis://" in redis_url:
            # Redis URL must include password
            if ":${REDIS_PASSWORD}@" not in redis_url and "::${REDIS_PASSWORD}@" not in redis_url:
                issues.append(
                    f"Service '{service_name}' uses Redis without password: {redis_url}. "
                    f"Must use 'redis://:${{REDIS_PASSWORD}}@redis:6379'"
                )

    assert not issues, f"Found {len(issues)} services with unauthenticated Redis:\n" + "\n".join(issues)


def test_no_default_passwords(docker_compose_config):
    """Verify no hardcoded default passwords in docker-compose.yml"""
    services = docker_compose_config.get("services", {})

    with open(DOCKER_COMPOSE_FILE) as f:
        content = f.read()

    # Check for common insecure patterns
    insecure_patterns = [
        r"password.*=.*['\"](?!.*\$\{)[\w]+['\"]",  # password="something"
        r"requirepass\s+['\"]?(?!\$\{)[\w]+['\"]?",  # requirepass something
    ]

    issues = []
    for pattern in insecure_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.extend(matches)

    assert not issues, f"Found potential hardcoded passwords: {issues}"


def test_critical_ports_covered():
    """Verify all critical database and API ports are configured"""
    expected_ports = {
        "8000": "MCP Server",
        "8001": "REST API",
        "6333": "Qdrant HTTP",
        "6334": "Qdrant gRPC",
        "6379": "Redis",
        "7474": "Neo4j HTTP",
        "7687": "Neo4j Bolt",
        "8080": "Monitoring Dashboard",
        "9090": "Sentinel"
    }

    with open(DOCKER_COMPOSE_FILE) as f:
        content = f.read()

    missing_ports = []
    for port, service_name in expected_ports.items():
        # Check if port appears with localhost binding
        if f"127.0.0.1:{port}:" not in content:
            missing_ports.append(f"{port} ({service_name})")

    assert not missing_ports, (
        f"Missing localhost bindings for critical ports: {', '.join(missing_ports)}"
    )


def test_neo4j_requires_password(docker_compose_config):
    """Verify Neo4j requires password authentication"""
    neo4j_service = docker_compose_config["services"].get("neo4j")
    assert neo4j_service, "Neo4j service not found"

    env_vars = neo4j_service.get("environment", [])

    # Convert to dict if list
    if isinstance(env_vars, list):
        env_dict = {}
        for item in env_vars:
            if "=" in str(item):
                key, value = str(item).split("=", 1)
                env_dict[key.strip()] = value.strip()
    else:
        env_dict = env_vars

    neo4j_auth = env_dict.get("NEO4J_AUTH", "")

    assert neo4j_auth, "NEO4J_AUTH must be configured"
    assert "${NEO4J_PASSWORD}" in neo4j_auth or "$NEO4J_PASSWORD" in neo4j_auth, (
        "Neo4j must use NEO4J_PASSWORD environment variable"
    )


def test_no_ports_exposed_to_internet():
    """CRITICAL: Verify no ports are exposed to 0.0.0.0 (internet)"""
    with open(DOCKER_COMPOSE_FILE) as f:
        content = f.read()

    # Find all port mappings
    port_lines = []
    for line_num, line in enumerate(content.split("\n"), 1):
        if re.search(r'^\s*-\s*["\']?\d+:\d+', line):
            port_lines.append((line_num, line.strip()))

    exposed_ports = []
    for line_num, line in port_lines:
        # Check if line doesn't start with 127.0.0.1
        if not re.search(r'127\.0\.0\.1:\d+', line):
            exposed_ports.append(f"Line {line_num}: {line}")

    assert not exposed_ports, (
        f"Found {len(exposed_ports)} ports exposed to internet (must use 127.0.0.1):\n" +
        "\n".join(exposed_ports)
    )


def test_services_have_healthchecks():
    """Verify critical services have health checks configured"""
    critical_services = [
        "context-store", "api", "qdrant", "neo4j", "redis",
        "monitoring-dashboard", "sentinel"
    ]

    with open(DOCKER_COMPOSE_FILE) as f:
        config = yaml.safe_load(f)

    services = config.get("services", {})
    missing_healthchecks = []

    for service_name in critical_services:
        if service_name in services:
            if "healthcheck" not in services[service_name]:
                missing_healthchecks.append(service_name)

    assert not missing_healthchecks, (
        f"Critical services missing healthchecks: {', '.join(missing_healthchecks)}"
    )


def test_security_comments_present():
    """Verify security-related comments are present for documentation"""
    with open(DOCKER_COMPOSE_FILE) as f:
        content = f.read()

    required_comments = [
        "SECURITY:",
        "localhost only",
        "password auth",
    ]

    missing_comments = []
    content_lower = content.lower()

    for comment in required_comments:
        if comment.lower() not in content_lower:
            missing_comments.append(comment)

    assert not missing_comments, (
        f"Missing security documentation comments: {', '.join(missing_comments)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
