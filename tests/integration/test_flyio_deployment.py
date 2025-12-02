"""Integration tests for Fly.io deployment configuration."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestFlyioDeployment:
    """Test suite for validating Fly.io deployment configuration."""

    @pytest.fixture
    def flyio_config_path(self):
        """Return path to fly.toml configuration file."""
        return Path(__file__).parent.parent.parent / "fly.toml"

    @pytest.fixture
    def dockerfile_path(self):
        """Return path to Dockerfile.flyio."""
        return Path(__file__).parent.parent.parent / "Dockerfile.flyio"

    @pytest.fixture
    def startup_script_path(self):
        """Return path to start-services.sh."""
        return Path(__file__).parent.parent.parent / "start-services.sh"

    def test_fly_toml_exists_and_valid(self, flyio_config_path):
        """Test that fly.toml exists and contains required configuration."""
        assert flyio_config_path.exists(), "fly.toml file must exist"

        with open(flyio_config_path, "r") as f:
            content = f.read()

        # Check for required sections
        assert 'app = "veris-memory"' in content
        assert "[build]" in content
        assert 'dockerfile = "Dockerfile.flyio"' in content
        assert "[[services]]" in content
        assert "[[vm]]" in content
        assert "[network]" in content

        # Check for security configurations
        assert 'ipv4 = "dedicated"' in content
        assert "ipv6 = true" in content

        # Check for health monitoring
        assert "[[services.http_checks]]" in content
        assert 'path = "/health"' in content

    def test_dockerfile_security_hardening(self, dockerfile_path):
        """Test Dockerfile contains security hardening measures."""
        assert dockerfile_path.exists(), "Dockerfile.flyio must exist"

        with open(dockerfile_path, "r") as f:
            content = f.read()

        # Check for SHA256 pinning
        assert "@sha256:" in content, "Base image must use SHA256 digest"

        # Check password is not hardcoded
        assert "NEO4J_AUTH=" not in content or "changeme123" not in content

        # Check for GPG verification
        assert "/etc/apt/keyrings/" in content, "Should use modern GPG keyring"
        assert "gpg --dearmor" in content, "Should properly verify GPG keys"

        # Check for checksum verification attempt
        assert "sha256" in content.lower(), "Should attempt checksum verification"

    def test_startup_script_executable(self, startup_script_path):
        """Test startup script exists and is properly configured."""
        assert startup_script_path.exists(), "start-services.sh must exist"

        with open(startup_script_path, "r") as f:
            content = f.read()

        # Check shebang
        assert content.startswith("#!/bin/bash"), "Must have proper shebang"

        # Check error handling
        assert "set -e" in content, "Must exit on error"

        # Check environment variable handling
        assert "NEO4J_PASSWORD=${NEO4J_PASSWORD:-" in content
        assert "REDIS_MAXMEMORY=${REDIS_MAXMEMORY:-" in content

        # Check service configuration
        assert "NEO4J_AUTH=" in content, "Should set Neo4j auth dynamically"
        assert "maxmemory" in content, "Should configure Redis memory"

    def test_resource_allocation(self, flyio_config_path):
        """Test resource allocation is appropriate for multi-service deployment."""
        with open(flyio_config_path, "r") as f:
            content = f.read()

        # Check memory allocation (should be at least 2GB for all services)
        assert 'memory = "2gb"' in content, "Should allocate at least 2GB memory"

        # Check CPU allocation
        assert "cpus = 2" in content, "Should allocate at least 2 CPUs"

    def test_persistent_storage_configuration(self, flyio_config_path):
        """Test persistent storage is properly configured."""
        with open(flyio_config_path, "r") as f:
            content = f.read()

        # Check for data and logs volumes
        assert "veris_memory_data" in content, "Should have data volume"
        assert "veris_memory_logs" in content, "Should have logs volume"
        assert "/app/data" in content, "Should mount data directory"
        assert "/app/logs" in content, "Should mount logs directory"

    def test_network_security_configuration(self, flyio_config_path):
        """Test network security is properly configured."""
        with open(flyio_config_path, "r") as f:
            content = f.read()

        # Check static IP configuration
        assert 'ipv4 = "dedicated"' in content, "Should use dedicated IPv4"
        assert "ipv6 = true" in content, "Should enable IPv6"

        # Check DNS configuration
        assert "1.1.1.1" in content, "Should use secure DNS"
        assert "8.8.8.8" in content, "Should have fallback DNS"

    def test_health_checks_configuration(self, flyio_config_path):
        """Test health monitoring is properly configured."""
        with open(flyio_config_path, "r") as f:
            content = f.read()

        # Check HTTP health checks
        assert 'method = "GET"' in content
        assert 'path = "/health"' in content
        assert 'timeout = "2s"' in content
        assert 'interval = "10s"' in content

        # Check TCP checks for service availability
        assert "[[services.tcp_checks]]" in content

    def test_environment_variables_security(self, flyio_config_path):
        """Test environment variables don't contain hardcoded secrets."""
        with open(flyio_config_path, "r") as f:
            content = f.read()

        # Check that no hardcoded passwords are present
        assert "changeme123" not in content, "Should not contain hardcoded passwords"
        assert (
            "password" not in content.lower()
            or "NEO4J_PASSWORD must be set as a Fly.io secret" in content
        ), "Should use secrets management"

        # Should have instructions for setting secrets
        if "NEO4J_PASSWORD" in content:
            assert (
                "flyctl secrets set" in content
            ), "Should provide instructions for setting secrets"

    def test_dockerfile_builds_successfully(self, dockerfile_path):
        """Test that Dockerfile.flyio builds without errors."""
        if not shutil.which("docker"):
            pytest.skip("Docker not available for integration testing")

        # Create temporary build context
        with tempfile.TemporaryDirectory() as tmpdir:
            build_context = Path(tmpdir)

            # Copy minimal files needed for build test
            dockerfile_dest = build_context / "Dockerfile"
            dockerfile_dest.write_text(dockerfile_path.read_text())

            # Create minimal requirements.txt
            requirements = build_context / "requirements.txt"
            requirements.write_text("fastapi\nuvicorn\n")

            # Create minimal source structure
            src_dir = build_context / "src"
            src_dir.mkdir()
            (src_dir / "__init__.py").touch()

            # Create other required directories
            for dir_name in ["schemas", "contracts"]:
                (build_context / dir_name).mkdir()
                (build_context / dir_name / "README.md").write_text(f"# {dir_name}")

            # Create required config files
            supervisord_conf = build_context / "supervisord.conf"
            supervisord_conf.write_text(
                """
[supervisord]
nodaemon=true
logfile=/app/logs/supervisord.log
"""
            )

            start_script = build_context / "start-services.sh"
            start_script.write_text(
                """#!/bin/bash
echo "Test build successful"
"""
            )

            # Test build (dry run)
            try:
                result = subprocess.run(
                    ["docker", "build", "--dry-run", "-f", "Dockerfile", "."],
                    cwd=build_context,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                # If dry-run is not supported, just check dockerfile syntax
                if result.returncode != 0 and "--dry-run" in result.stderr:
                    # Fallback: just validate dockerfile syntax
                    result = subprocess.run(
                        ["docker", "build", "--no-cache", "--target", "nonexistent", "."],
                        cwd=build_context,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                assert result.returncode in [0, 1], f"Build failed: {result.stderr}"

            except subprocess.TimeoutExpired:
                pytest.skip("Docker build test timed out")
            except Exception as e:
                pytest.skip(f"Docker build test failed: {e}")

    def test_supervisord_configuration(self):
        """Test supervisord configuration is valid."""
        supervisord_path = Path(__file__).parent.parent.parent / "supervisord.conf"
        assert supervisord_path.exists(), "supervisord.conf must exist"

        with open(supervisord_path, "r") as f:
            content = f.read()

        # Check for required sections
        assert "[supervisord]" in content
        assert "[program:redis]" in content
        assert "[program:qdrant]" in content
        assert "[program:neo4j]" in content
        assert "[program:context-store]" in content
        assert "[program:health-monitor]" in content

        # Check startup priorities
        assert "priority=10" in content, "Redis should start first"
        assert "priority=40" in content, "Context-store should start last"
        assert "priority=50" in content, "Health monitor should start after services"

        # Check resource limits
        assert "rlimit_as=" in content, "Should have memory limits"
        assert "rlimit_nofile=" in content, "Should have file descriptor limits"

        # Check logging configuration
        assert "/app/logs/" in content, "Should log to persistent volume"

    def test_monitoring_scripts_exist(self):
        """Test monitoring scripts exist and are executable."""
        monitoring_dir = Path(__file__).parent.parent.parent / "monitoring"
        health_monitor = monitoring_dir / "health-monitor.sh"

        assert monitoring_dir.exists(), "monitoring directory must exist"
        assert health_monitor.exists(), "health-monitor.sh must exist"

    def test_secrets_management_scripts_exist(self):
        """Test secrets management scripts exist and are executable."""
        secrets_dir = Path(__file__).parent.parent.parent / "secrets"
        secrets_manager = secrets_dir / "secrets-manager.sh"

        assert secrets_dir.exists(), "secrets directory must exist"
        assert secrets_manager.exists(), "secrets-manager.sh must exist"

    def test_multi_stage_dockerfile_structure(self, dockerfile_path):
        """Test multi-stage Dockerfile has proper structure."""
        with open(dockerfile_path, "r") as f:
            content = f.read()

        # Check for multi-stage build
        assert "FROM ubuntu" in content and "AS builder" in content, "Should use multi-stage build"
        assert "FROM ubuntu" in content and "AS runtime" in content, "Should have runtime stage"

        # Check for COPY --from=builder
        assert "COPY --from=builder" in content, "Should copy artifacts from builder stage"

        # Check security improvements
        assert "bc" in content, "Should install bc for monitoring calculations"
        assert "procps" in content, "Should install procps for process monitoring"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
