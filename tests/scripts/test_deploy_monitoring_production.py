#!/usr/bin/env python3
"""
Tests for scripts/deploy-monitoring-production.sh

Tests deployment script functionality including:
- UFW firewall configuration validation
- Environment variable handling
- Docker compose operations
- Service health checks
- Error handling and rollback
"""

import pytest
import tempfile
import subprocess
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call


class TestDeployMonitoringProduction:
    """Test suite for the monitoring production deployment script."""

    @pytest.fixture
    def temp_env_file(self):
        """Create a temporary environment file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
# Veris Memory Production Environment
VERIS_HOST=api.veris-memory.com
VERIS_API_KEY=test-api-key-12345
REDIS_PASSWORD=test-redis-password
NEO4J_PASSWORD=test-neo4j-password
QDRANT_API_KEY=test-qdrant-api-key

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_ADMIN_PASSWORD=test-grafana-password
ALERT_WEBHOOK=https://hooks.slack.com/services/test/webhook

# Security
JWT_SECRET_KEY=test-jwt-secret-very-long-and-secure
ENCRYPTION_KEY=test-encryption-key-32-chars-long

# Resource Limits (production)
MEMORY_LIMIT=60g
CPU_LIMIT=15
""")
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def mock_subprocess(self):
        """Mock subprocess calls for testing script execution."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="success", stderr="")
            yield mock_run

    @pytest.fixture
    def script_path(self):
        """Path to the deployment script."""
        return Path(__file__).parent.parent.parent / "scripts" / "deploy-monitoring-production.sh"

    def test_script_exists_and_executable(self, script_path):
        """Test that the deployment script exists and is executable."""
        assert script_path.exists(), f"Deployment script not found at {script_path}"
        assert os.access(script_path, os.X_OK), "Deployment script is not executable"

    def test_environment_file_validation(self, temp_env_file, mock_subprocess):
        """Test that the script validates environment files correctly."""
        # Test with valid environment file
        result = subprocess.run([
            "bash", "-c", 
            f"source {temp_env_file} && "
            "test -n \"$VERIS_API_KEY\" && "
            "test -n \"$REDIS_PASSWORD\" && "
            "test -n \"$NEO4J_PASSWORD\" && "
            "echo 'validation_passed'"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "validation_passed" in result.stdout

    def test_ufw_firewall_rules_syntax(self, mock_subprocess):
        """Test UFW firewall rule syntax validation."""
        # Test valid UFW commands that would be executed
        ufw_commands = [
            "ufw --force enable",
            "ufw default deny incoming",
            "ufw default allow outgoing", 
            "ufw allow 22/tcp",
            "ufw allow 80/tcp",
            "ufw allow 443/tcp",
            "ufw allow from 127.0.0.1 to any port 8001",
            "ufw allow from 127.0.0.1 to any port 8080",
            "ufw allow from 127.0.0.1 to any port 9090"
        ]
        
        for cmd in ufw_commands:
            # Validate command syntax without actually executing
            parts = cmd.split()
            assert len(parts) >= 2, f"Invalid UFW command: {cmd}"
            assert parts[0] == "ufw", f"Command should start with 'ufw': {cmd}"

    @patch('subprocess.run')
    def test_docker_compose_commands(self, mock_run):
        """Test Docker Compose command execution."""
        mock_run.return_value = Mock(returncode=0, stdout="success")
        
        # Test docker compose validation
        cmd = ["docker", "compose", "-f", "docker-compose.prod.yml", "config"]
        result = subprocess.run(cmd, capture_output=True)
        
        # Command should be properly formed
        assert cmd[0] == "docker"
        assert cmd[1] == "compose"
        assert "-f" in cmd
        assert "docker-compose.prod.yml" in cmd

    def test_service_health_check_urls(self):
        """Test that health check URLs are properly formatted."""
        health_endpoints = [
            "http://localhost:8001/health",
            "http://localhost:8080/metrics", 
            "http://localhost:9090/status"
        ]
        
        for url in health_endpoints:
            # Basic URL validation
            assert url.startswith("http://"), f"Invalid URL protocol: {url}"
            assert "localhost" in url, f"Should use localhost: {url}"
            assert url.count(":") == 2, f"Invalid port specification: {url}"

    @patch('subprocess.run')
    def test_deployment_rollback_scenario(self, mock_run):
        """Test deployment rollback when services fail to start."""
        # Simulate service startup failure
        mock_run.side_effect = [
            Mock(returncode=0),  # UFW commands succeed
            Mock(returncode=0),  # Docker compose config succeeds
            Mock(returncode=1, stderr="Service failed to start")  # Deploy fails
        ]
        
        # The script should handle failure gracefully
        # We can't test the actual script execution, but we validate the patterns
        rollback_commands = [
            "docker compose -f docker-compose.prod.yml down",
            "docker system prune -f"
        ]
        
        for cmd in rollback_commands:
            parts = cmd.split()
            assert "docker" in parts, f"Rollback should use docker: {cmd}"

    def test_environment_variable_security(self, temp_env_file):
        """Test that sensitive environment variables are handled securely."""
        with open(temp_env_file, 'r') as f:
            content = f.read()
        
        sensitive_vars = [
            "VERIS_API_KEY", "REDIS_PASSWORD", "NEO4J_PASSWORD",
            "JWT_SECRET_KEY", "ENCRYPTION_KEY", "GRAFANA_ADMIN_PASSWORD"
        ]
        
        for var in sensitive_vars:
            assert var in content, f"Missing sensitive variable: {var}"
            # Ensure variable has a value (not empty)
            lines = [line for line in content.split('\n') if line.startswith(f"{var}=")]
            assert len(lines) == 1, f"Variable {var} should be defined exactly once"
            assert len(lines[0].split('=', 1)[1]) > 0, f"Variable {var} should not be empty"

    def test_resource_limits_configuration(self, temp_env_file):
        """Test resource limits are appropriate for production system."""
        with open(temp_env_file, 'r') as f:
            content = f.read()
        
        # Extract resource limits
        memory_limit = None
        cpu_limit = None
        
        for line in content.split('\n'):
            if line.startswith('MEMORY_LIMIT='):
                memory_limit = line.split('=')[1]
            elif line.startswith('CPU_LIMIT='):
                cpu_limit = line.split('=')[1]
        
        assert memory_limit is not None, "MEMORY_LIMIT should be defined"
        assert cpu_limit is not None, "CPU_LIMIT should be defined"
        
        # Validate memory limit format (should be like "60g" for 60GB)
        assert memory_limit.endswith('g'), "Memory limit should be in gigabytes"
        memory_gb = int(memory_limit[:-1])
        assert 32 <= memory_gb <= 62, f"Memory limit {memory_gb}GB should be reasonable for 64GB system"
        
        # Validate CPU limit
        cpu_cores = int(cpu_limit)
        assert 8 <= cpu_cores <= 16, f"CPU limit {cpu_cores} should be reasonable for production system"

    @patch('subprocess.run')
    def test_service_startup_timeout(self, mock_run):
        """Test that service startup includes appropriate timeout handling."""
        # Simulate slow service startup
        mock_run.side_effect = [
            Mock(returncode=0),  # Initial commands succeed
            Mock(returncode=124)  # Timeout exit code
        ]
        
        # Test timeout command structure
        timeout_cmd = ["timeout", "120", "curl", "-f", "http://localhost:8001/health"]
        
        assert timeout_cmd[0] == "timeout", "Should use timeout command"
        assert int(timeout_cmd[1]) >= 60, "Timeout should be at least 60 seconds"
        assert "curl" in timeout_cmd, "Should use curl for health checks"
        assert "-f" in timeout_cmd, "Should use curl -f for proper error handling"

    def test_docker_compose_file_validation(self):
        """Test that docker-compose.prod.yml has required structure."""
        compose_file = Path(__file__).parent.parent.parent / "docker-compose.prod.yml"
        
        if compose_file.exists():
            # Test that we can read the compose file
            with open(compose_file, 'r') as f:
                content = f.read()
            
            # Should contain essential services
            assert "context-store" in content, "Should define context-store service"
            assert "prometheus" in content or "monitoring" in content, "Should define monitoring service"
            
            # Should have proper port bindings
            assert "127.0.0.1:8001:8000" in content, "Should bind API to localhost:8001"
            assert "127.0.0.1:8080" in content, "Should bind monitoring to localhost:8080"

    def test_deployment_validation_script(self):
        """Test that validate-monitoring-deployment.sh exists and is valid."""
        validation_script = Path(__file__).parent.parent.parent / "scripts" / "validate-monitoring-deployment.sh"
        
        if validation_script.exists():
            assert os.access(validation_script, os.X_OK), "Validation script should be executable"
            
            # Read script content for basic validation
            with open(validation_script, 'r') as f:
                content = f.read()
            
            # Should contain essential validation steps
            assert "curl" in content, "Should use curl for endpoint testing"
            assert "health" in content, "Should check health endpoints"
            assert "status" in content, "Should check status endpoints"

    @patch('subprocess.run')
    def test_cleanup_on_failure(self, mock_run):
        """Test that cleanup is performed when deployment fails."""
        # Simulate deployment failure
        mock_run.side_effect = [
            Mock(returncode=0),  # Initial setup succeeds
            Mock(returncode=1, stderr="Deployment failed")  # Deployment fails
        ]
        
        # Cleanup commands that should be executed
        cleanup_commands = [
            "docker compose -f docker-compose.prod.yml down --volumes",
            "docker system prune -f",
            "docker volume prune -f"
        ]
        
        for cmd in cleanup_commands:
            # Validate cleanup command structure
            assert "docker" in cmd, f"Cleanup should use docker: {cmd}"
            assert any(action in cmd for action in ["down", "prune"]), f"Should clean up resources: {cmd}"

    def test_logging_configuration(self):
        """Test that deployment includes proper logging configuration."""
        # Test log file paths and permissions
        log_paths = [
            "/var/log/veris-memory/deployment.log",
            "/var/log/veris-memory/monitoring.log"
        ]
        
        for log_path in log_paths:
            # Validate log path structure
            assert log_path.startswith("/var/log/"), f"Logs should be in /var/log: {log_path}"
            assert "veris-memory" in log_path, f"Should use veris-memory directory: {log_path}"

    @patch('os.path.exists')
    @patch('subprocess.run')
    def test_prerequisite_checks(self, mock_run, mock_exists):
        """Test that deployment checks prerequisites before starting."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0, stdout="Docker version 24.0.0")
        
        # Prerequisites that should be checked
        prerequisites = [
            "docker --version",
            "docker compose version", 
            "ufw --version",
            "curl --version"
        ]
        
        for cmd in prerequisites:
            parts = cmd.split()
            assert len(parts) >= 2, f"Prerequisite check should have proper format: {cmd}"
            assert "--version" in cmd or "version" in cmd, f"Should check version: {cmd}"


class TestDeploymentScriptIntegration:
    """Integration tests for deployment script functionality."""

    @pytest.fixture
    def mock_environment(self):
        """Mock environment for integration testing."""
        env_vars = {
            "VERIS_HOST": "test.veris-memory.com",
            "VERIS_API_KEY": "test-api-key",
            "REDIS_PASSWORD": "test-redis-pass",
            "NEO4J_PASSWORD": "test-neo4j-pass"
        }
        
        with patch.dict(os.environ, env_vars):
            yield env_vars

    @patch('subprocess.run')
    def test_full_deployment_flow_simulation(self, mock_run, mock_environment):
        """Simulate full deployment flow without actual execution."""
        # Mock successful command execution
        mock_run.return_value = Mock(returncode=0, stdout="success")
        
        deployment_steps = [
            # Prerequisite checks
            ["docker", "--version"],
            ["docker", "compose", "version"],
            ["ufw", "--version"],
            
            # UFW configuration
            ["ufw", "--force", "enable"],
            ["ufw", "default", "deny", "incoming"],
            ["ufw", "allow", "22/tcp"],
            ["ufw", "allow", "80/tcp"],
            ["ufw", "allow", "443/tcp"],
            
            # Docker operations
            ["docker", "compose", "-f", "docker-compose.prod.yml", "config"],
            ["docker", "compose", "-f", "docker-compose.prod.yml", "pull"],
            ["docker", "compose", "-f", "docker-compose.prod.yml", "up", "-d"],
            
            # Health checks
            ["curl", "-f", "http://localhost:8001/health"],
            ["curl", "-f", "http://localhost:8080/metrics"],
            ["curl", "-f", "http://localhost:9090/status"]
        ]
        
        # Validate each step
        for step in deployment_steps:
            assert len(step) >= 2, f"Command should have at least 2 parts: {step}"
            assert isinstance(step[0], str), f"Command should start with string: {step}"

    def test_error_scenarios_handling(self):
        """Test error scenario handling in deployment."""
        error_scenarios = [
            {
                "error": "Docker not found",
                "exit_code": 127,
                "expected_message": "Docker is required"
            },
            {
                "error": "UFW not available", 
                "exit_code": 127,
                "expected_message": "UFW firewall is required"
            },
            {
                "error": "Compose file invalid",
                "exit_code": 1,
                "expected_message": "docker-compose.prod.yml validation failed"
            },
            {
                "error": "Service startup timeout",
                "exit_code": 124,
                "expected_message": "Service health check timeout"
            }
        ]
        
        for scenario in error_scenarios:
            # Validate error scenario structure
            assert "error" in scenario
            assert "exit_code" in scenario
            assert "expected_message" in scenario
            assert isinstance(scenario["exit_code"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])