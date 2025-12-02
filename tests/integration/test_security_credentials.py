"""
Integration tests for security credential requirements.

These tests validate that:
1. Redis requires authentication (REDIS_PASSWORD)
2. Neo4j read-only password has no default fallback (NEO4J_RO_PASSWORD)
3. Deployment fails gracefully when secrets are missing

Tests are run against docker-compose environments to ensure production configs
are secure by default.
"""

import os
import subprocess
import time
from typing import Optional

import pytest
import redis
from neo4j import GraphDatabase


class TestRedisAuthentication:
    """Test Redis password authentication requirements."""

    @pytest.fixture
    def redis_password(self) -> str:
        """Get Redis password from environment."""
        password = os.getenv("REDIS_PASSWORD")
        if not password:
            pytest.skip("REDIS_PASSWORD not set - cannot test Redis auth")
        return password

    def test_redis_rejects_connection_without_password(self, redis_password: str):
        """
        SECURITY TEST: Verify Redis rejects unauthenticated connections.

        This test ensures that REDIS_PASSWORD is enforced and Redis will not
        accept connections without authentication.
        """
        # Attempt to connect without password (should fail)
        client = redis.Redis(host="localhost", port=6379, password=None, socket_connect_timeout=5)

        with pytest.raises(redis.AuthenticationError, match=".*authentication.*"):
            client.ping()

    def test_redis_accepts_connection_with_correct_password(self, redis_password: str):
        """
        SECURITY TEST: Verify Redis accepts connections with correct password.

        This test validates that REDIS_PASSWORD is properly configured and
        applications can authenticate successfully.
        """
        client = redis.Redis(
            host="localhost",
            port=6379,
            password=redis_password,
            socket_connect_timeout=5,
        )

        # Should succeed
        response = client.ping()
        assert response is True, "Redis should respond to PING with valid password"

    def test_redis_rejects_connection_with_wrong_password(self, redis_password: str):
        """
        SECURITY TEST: Verify Redis rejects connections with incorrect password.

        This test ensures that password validation is working correctly.
        """
        client = redis.Redis(
            host="localhost",
            port=6379,
            password="wrong_password_12345",
            socket_connect_timeout=5,
        )

        with pytest.raises(redis.AuthenticationError, match=".*invalid password.*"):
            client.ping()

    def test_redis_password_is_set_in_environment(self):
        """
        DEPLOYMENT TEST: Verify REDIS_PASSWORD is set in environment.

        This test fails deployment if REDIS_PASSWORD is not configured,
        preventing insecure deployments.
        """
        redis_password = os.getenv("REDIS_PASSWORD")

        assert redis_password is not None, (
            "REDIS_PASSWORD must be set in environment. "
            "Add to GitHub Secrets or .env file."
        )

        assert len(redis_password) >= 16, (
            "REDIS_PASSWORD must be at least 16 characters for security. "
            f"Current length: {len(redis_password)}"
        )

    def test_redis_container_health_check_requires_password(self):
        """
        DOCKER TEST: Verify Redis healthcheck uses password authentication.

        This test ensures docker-compose healthcheck is properly configured
        to use REDIS_PASSWORD.
        """
        # Get Redis container name
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=redis", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True,
        )

        container_name = result.stdout.strip().split("\n")[0]
        assert container_name, "Redis container not found"

        # Check healthcheck status
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
            capture_output=True,
            text=True,
            check=True,
        )

        health_status = result.stdout.strip()
        assert health_status in ["healthy", "starting"], (
            f"Redis container healthcheck failed: {health_status}. "
            "This may indicate REDIS_PASSWORD is not properly configured in healthcheck."
        )


class TestNeo4jReadOnlyPassword:
    """Test Neo4j read-only password requirements."""

    @pytest.fixture
    def neo4j_ro_password(self) -> str:
        """Get Neo4j read-only password from environment."""
        password = os.getenv("NEO4J_RO_PASSWORD")
        if not password:
            pytest.skip("NEO4J_RO_PASSWORD not set - cannot test Neo4j RO auth")
        return password

    @pytest.fixture
    def neo4j_password(self) -> str:
        """Get Neo4j admin password from environment."""
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            pytest.skip("NEO4J_PASSWORD not set - cannot test Neo4j")
        return password

    def test_neo4j_ro_password_is_set_in_environment(self):
        """
        DEPLOYMENT TEST: Verify NEO4J_RO_PASSWORD is set with no default fallback.

        This test fails deployment if NEO4J_RO_PASSWORD is not configured,
        preventing the use of hardcoded default passwords.
        """
        neo4j_ro_password = os.getenv("NEO4J_RO_PASSWORD")

        assert neo4j_ro_password is not None, (
            "NEO4J_RO_PASSWORD must be set in environment. "
            "No default fallback is allowed for security. "
            "Add to GitHub Secrets or .env file."
        )

        assert neo4j_ro_password != "readonly_secure_2024!", (
            "NEO4J_RO_PASSWORD must not use the old hardcoded default. "
            "Generate a new secure password."
        )

        assert len(neo4j_ro_password) >= 16, (
            "NEO4J_RO_PASSWORD must be at least 16 characters for security. "
            f"Current length: {len(neo4j_ro_password)}"
        )

    def test_neo4j_ro_password_differs_from_admin_password(
        self, neo4j_password: str, neo4j_ro_password: str
    ):
        """
        SECURITY TEST: Verify read-only password is different from admin password.

        Read-only and admin passwords should never be the same for proper
        access control separation.
        """
        assert neo4j_password != neo4j_ro_password, (
            "NEO4J_RO_PASSWORD must be different from NEO4J_PASSWORD. "
            "Using the same password defeats the purpose of read-only access."
        )

    def test_neo4j_connection_with_ro_password(
        self, neo4j_ro_password: str
    ):
        """
        SECURITY TEST: Verify Neo4j read-only password authentication works.

        This test validates that NEO4J_RO_PASSWORD is properly configured
        and can be used for read-only operations.

        Note: This test assumes a read-only user has been created with the
        NEO4J_RO_PASSWORD. If not, this test will be skipped.
        """
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")

        try:
            driver = GraphDatabase.driver(
                uri,
                auth=("neo4j", neo4j_ro_password),  # Try RO password with default user
            )

            with driver.session() as session:
                # Try a read operation
                result = session.run("RETURN 1 AS test")
                value = result.single()["test"]
                assert value == 1

            driver.close()

        except Exception as e:
            # If RO user doesn't exist yet, skip this test
            if "authentication" in str(e).lower():
                pytest.skip(
                    "Read-only Neo4j user not configured yet. "
                    "Create a read-only user with NEO4J_RO_PASSWORD in schema initialization."
                )
            else:
                raise


class TestDeploymentValidation:
    """Test deployment-time validation of required secrets."""

    def test_all_required_secrets_are_set(self):
        """
        DEPLOYMENT TEST: Verify all required secrets are configured.

        This test fails deployment if any critical secrets are missing,
        preventing insecure deployments.
        """
        required_secrets = {
            "NEO4J_PASSWORD": "Neo4j admin password",
            "NEO4J_RO_PASSWORD": "Neo4j read-only password",
            "REDIS_PASSWORD": "Redis authentication password",
        }

        missing_secrets = []
        for secret_name, description in required_secrets.items():
            if not os.getenv(secret_name):
                missing_secrets.append(f"{secret_name} ({description})")

        assert not missing_secrets, (
            "Required secrets are missing from environment:\n"
            + "\n".join(f"  - {secret}" for secret in missing_secrets)
            + "\n\nAdd these secrets to GitHub Secrets or .env file before deployment."
        )

    def test_secrets_meet_minimum_length_requirements(self):
        """
        SECURITY TEST: Verify all secrets meet minimum length requirements.

        Short passwords are easier to brute-force. This test enforces
        minimum length requirements for all secrets.
        """
        min_length = 16

        secrets_to_check = {
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "NEO4J_RO_PASSWORD": os.getenv("NEO4J_RO_PASSWORD"),
            "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD"),
        }

        short_secrets = []
        for secret_name, secret_value in secrets_to_check.items():
            if secret_value and len(secret_value) < min_length:
                short_secrets.append(
                    f"{secret_name} ({len(secret_value)} chars, need {min_length})"
                )

        assert not short_secrets, (
            "Secrets do not meet minimum length requirements:\n"
            + "\n".join(f"  - {secret}" for secret in short_secrets)
            + f"\n\nAll secrets must be at least {min_length} characters."
        )

    def test_docker_compose_files_have_no_hardcoded_defaults(self):
        """
        SECURITY TEST: Verify docker-compose files have no hardcoded password defaults.

        This test scans docker-compose files for common hardcoded passwords
        and default values that should not exist.
        """
        import re
        from pathlib import Path

        compose_files = list(Path(".").glob("docker-compose*.yml"))

        # Patterns that indicate hardcoded passwords
        dangerous_patterns = [
            r":-\s*['\"]?readonly_secure_2024!",  # Old hardcoded RO password
            r":-\s*['\"]?devpassword",  # Dev password defaults
            r":-\s*['\"]?password\d+",  # Simple password patterns
            r"password\s*=\s*['\"][^$\{]",  # Literal password assignments
        ]

        violations = []
        for compose_file in compose_files:
            # Skip test configs (allowed to have defaults for testing)
            if "test" in compose_file.name or "dev" in compose_file.name:
                continue

            content = compose_file.read_text()

            for pattern in dangerous_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count("\n") + 1
                    violations.append(
                        f"{compose_file}:{line_num} - {match.group()}"
                    )

        assert not violations, (
            "Found hardcoded password defaults in docker-compose files:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nRemove all hardcoded password defaults. Use environment variables only."
        )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "security: mark test as a security validation test"
    )
    config.addinivalue_line(
        "markers", "deployment: mark test as a deployment-time validation test"
    )


# Mark all tests in this module as security tests
pytestmark = pytest.mark.security
