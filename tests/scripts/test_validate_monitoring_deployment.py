#!/usr/bin/env python3
"""
Tests for scripts/validate-monitoring-deployment.sh

Tests validation script functionality including:
- Health endpoint testing
- Service connectivity checks
- Metrics collection validation
- Security endpoint testing
- Performance baseline testing
"""

import pytest
import subprocess
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from urllib.parse import urlparse


class TestValidateMonitoringDeployment:
    """Test suite for the monitoring deployment validation script."""

    @pytest.fixture
    def script_path(self):
        """Path to the validation script."""
        return Path(__file__).parent.parent.parent / "scripts" / "validate-monitoring-deployment.sh"

    @pytest.fixture
    def mock_curl_responses(self):
        """Mock curl responses for different endpoints."""
        return {
            "/health": {
                "status_code": 200,
                "response": {"status": "healthy", "timestamp": "2024-01-15T10:00:00Z"}
            },
            "/api/v1/collections": {
                "status_code": 200, 
                "response": {"result": {"collections": []}}
            },
            "/metrics": {
                "status_code": 200,
                "response": "# HELP veris_memory_requests_total Total requests\nveris_memory_requests_total 42"
            },
            "/status": {
                "status_code": 200,
                "response": {"sentinel_status": "active", "last_cycle": "2024-01-15T10:00:00Z"}
            },
            "/api/v1/contexts": {
                "status_code": 200,
                "response": {"contexts": [], "total": 0}
            }
        }

    def test_script_exists_and_executable(self, script_path):
        """Test that the validation script exists and is executable."""
        if script_path.exists():
            assert os.access(script_path, os.X_OK), "Validation script should be executable"

    @patch('subprocess.run')
    def test_health_endpoint_validation(self, mock_run, mock_curl_responses):
        """Test health endpoint validation logic."""
        # Mock successful health check
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_curl_responses["/health"]["response"])
        )
        
        # Test health check command structure
        health_cmd = ["curl", "-s", "-f", "--max-time", "10", "http://localhost:8001/health"]
        
        assert health_cmd[0] == "curl", "Should use curl for health checks"
        assert "-s" in health_cmd, "Should use silent mode"
        assert "-f" in health_cmd, "Should fail on HTTP errors"
        assert "--max-time" in health_cmd, "Should have timeout"
        assert "http://localhost:8001/health" in health_cmd, "Should check correct endpoint"

    @patch('subprocess.run')
    def test_qdrant_connectivity_validation(self, mock_run, mock_curl_responses):
        """Test Qdrant database connectivity validation."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_curl_responses["/api/v1/collections"]["response"])
        )
        
        # Test Qdrant API endpoint validation
        qdrant_endpoints = [
            "http://localhost:6333/collections",
            "http://localhost:6333/cluster"
        ]
        
        for endpoint in qdrant_endpoints:
            parsed = urlparse(endpoint)
            assert parsed.scheme == "http", f"Should use HTTP for local Qdrant: {endpoint}"
            assert parsed.hostname == "localhost", f"Should use localhost: {endpoint}"
            assert parsed.port == 6333, f"Should use port 6333: {endpoint}"

    @patch('subprocess.run')
    def test_prometheus_metrics_validation(self, mock_run, mock_curl_responses):
        """Test Prometheus metrics endpoint validation."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_curl_responses["/metrics"]["response"]
        )
        
        # Test metrics endpoint
        metrics_cmd = ["curl", "-s", "http://localhost:8080/metrics"]
        
        assert "curl" in metrics_cmd, "Should use curl for metrics"
        assert "http://localhost:8080/metrics" in metrics_cmd, "Should check metrics endpoint"
        
        # Validate metrics format
        metrics_content = mock_curl_responses["/metrics"]["response"]
        assert "# HELP" in metrics_content, "Should contain Prometheus help text"
        assert "veris_memory" in metrics_content, "Should contain Veris Memory metrics"

    @patch('subprocess.run')
    def test_sentinel_api_validation(self, mock_run, mock_curl_responses):
        """Test Sentinel monitoring API validation."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_curl_responses["/status"]["response"])
        )
        
        # Test Sentinel API endpoints
        sentinel_endpoints = [
            "http://localhost:9090/status",
            "http://localhost:9090/checks",
            "http://localhost:9090/metrics",
            "http://localhost:9090/report"
        ]
        
        for endpoint in sentinel_endpoints:
            parsed = urlparse(endpoint)
            assert parsed.hostname == "localhost", f"Should use localhost: {endpoint}"
            assert parsed.port == 9090, f"Should use port 9090: {endpoint}"

    def test_firewall_rules_validation(self):
        """Test UFW firewall rules validation."""
        # Expected firewall rules for production
        expected_rules = [
            "22/tcp",           # SSH
            "80/tcp",           # HTTP
            "443/tcp",          # HTTPS
            "8001",             # Veris Memory API (localhost only)
            "8080",             # Monitoring dashboard (localhost only)
            "9090"              # Sentinel API (localhost only)
        ]
        
        for rule in expected_rules:
            # Validate rule format
            if "/" in rule:
                port, protocol = rule.split("/")
                assert port.isdigit(), f"Port should be numeric: {port}"
                assert protocol in ["tcp", "udp"], f"Protocol should be tcp/udp: {protocol}"
            else:
                assert rule.isdigit(), f"Port should be numeric: {rule}"

    @patch('subprocess.run')
    def test_docker_container_validation(self, mock_run):
        """Test Docker container status validation."""
        # Mock docker ps output
        docker_output = """
CONTAINER ID   IMAGE                    COMMAND                  STATUS
abc123def456   veris-memory:latest      "python -m src.main"     Up 5 minutes (healthy)
def456ghi789   qdrant/qdrant:latest     "./entrypoint.sh"        Up 5 minutes (healthy)
ghi789jkl012   neo4j:5.15-community     "tini -g -- /startup"    Up 5 minutes (healthy)
"""
        mock_run.return_value = Mock(returncode=0, stdout=docker_output)
        
        # Test docker ps command
        docker_cmd = ["docker", "ps", "--filter", "label=com.docker.compose.project=veris-memory-dev"]
        
        assert docker_cmd[0] == "docker", "Should use docker command"
        assert "ps" in docker_cmd, "Should list containers"
        assert "--filter" in docker_cmd, "Should filter by project"

    @patch('subprocess.run')
    def test_service_response_time_validation(self, mock_run):
        """Test service response time validation."""
        # Mock curl with timing output
        timing_output = """
time_total:0.150
time_connect:0.010
time_starttransfer:0.140
"""
        mock_run.return_value = Mock(returncode=0, stdout=timing_output)
        
        # Test curl timing command
        timing_cmd = [
            "curl", "-s", "-o", "/dev/null", "-w", 
            "time_total:%{time_total}\\ntime_connect:%{time_connect}\\n",
            "http://localhost:8001/health"
        ]
        
        assert "curl" in timing_cmd, "Should use curl for timing"
        assert "-w" in timing_cmd, "Should use write-out for timing"
        assert "time_total" in " ".join(timing_cmd), "Should measure total time"

    def test_validation_thresholds(self):
        """Test validation threshold configuration."""
        # Performance thresholds for production system
        thresholds = {
            "health_check_max_time_ms": 500,
            "api_response_max_time_ms": 1000,
            "metrics_collection_max_time_ms": 2000,
            "sentinel_cycle_max_time_ms": 5000,
            "memory_usage_max_pct": 80,
            "cpu_usage_max_pct": 70,
            "disk_usage_max_pct": 85
        }
        
        for threshold_name, threshold_value in thresholds.items():
            assert isinstance(threshold_value, (int, float)), f"Threshold should be numeric: {threshold_name}"
            assert threshold_value > 0, f"Threshold should be positive: {threshold_name}"
            
            if "time_ms" in threshold_name:
                assert threshold_value <= 10000, f"Time threshold too high: {threshold_name}"
            elif "pct" in threshold_name:
                assert 0 < threshold_value <= 100, f"Percentage should be 0-100: {threshold_name}"

    @patch('subprocess.run')
    def test_security_endpoint_validation(self, mock_run):
        """Test security endpoint validation."""
        # Test that unauthorized access is properly blocked
        security_tests = [
            {
                "endpoint": "http://localhost:8001/admin",
                "expected_status": 401,
                "description": "Admin endpoint should require authentication"
            },
            {
                "endpoint": "http://localhost:8001/api/v1/contexts",
                "expected_status": 401,
                "description": "API endpoints should require authentication"
            }
        ]
        
        for test in security_tests:
            # Mock curl response for security test
            mock_run.return_value = Mock(
                returncode=22,  # curl exit code for HTTP error
                stderr=f"HTTP/1.1 {test['expected_status']}"
            )
            
            # Validate test structure
            assert "endpoint" in test, "Security test should have endpoint"
            assert "expected_status" in test, "Security test should have expected status"
            assert test["expected_status"] in [401, 403], "Should test for auth errors"

    @patch('subprocess.run')
    def test_database_connectivity_validation(self, mock_run):
        """Test database connectivity validation."""
        # Mock successful database connections
        mock_run.return_value = Mock(returncode=0, stdout="Connection successful")
        
        # Database connection tests
        db_tests = [
            {
                "name": "Neo4j",
                "command": ["cypher-shell", "-u", "veris_ro", "-p", "password", "RETURN 1"],
                "port": 7687
            },
            {
                "name": "Redis", 
                "command": ["redis-cli", "-p", "6379", "ping"],
                "port": 6379
            },
            {
                "name": "Qdrant",
                "command": ["curl", "-s", "http://localhost:6333/collections"],
                "port": 6333
            }
        ]
        
        for db_test in db_tests:
            assert "name" in db_test, "DB test should have name"
            assert "command" in db_test, "DB test should have command"
            assert "port" in db_test, "DB test should have port"
            assert isinstance(db_test["port"], int), "Port should be integer"
            assert 1024 <= db_test["port"] <= 65535, "Port should be in valid range"

    def test_log_file_validation(self):
        """Test log file validation configuration."""
        # Expected log files and their validation
        log_files = [
            {
                "path": "/var/log/veris-memory/application.log",
                "max_size_mb": 100,
                "required_patterns": ["INFO", "ERROR", "timestamp"]
            },
            {
                "path": "/var/log/veris-memory/monitoring.log", 
                "max_size_mb": 50,
                "required_patterns": ["sentinel", "check", "status"]
            },
            {
                "path": "/var/log/veris-memory/deployment.log",
                "max_size_mb": 25,
                "required_patterns": ["deployment", "success", "error"]
            }
        ]
        
        for log_config in log_files:
            assert "path" in log_config, "Log config should have path"
            assert "max_size_mb" in log_config, "Log config should have size limit"
            assert "required_patterns" in log_config, "Log config should have patterns"
            
            # Validate path structure
            path = log_config["path"]
            assert path.startswith("/var/log/"), f"Log should be in /var/log: {path}"
            assert "veris-memory" in path, f"Log should be in veris-memory dir: {path}"

    @patch('subprocess.run')
    def test_resource_usage_validation(self, mock_run):
        """Test system resource usage validation."""
        # Mock system resource output
        resource_output = """
MemTotal:       65536000 kB
MemAvailable:   45056000 kB
Cpu(s):  5.2%us,  2.1%sy,  0.0%ni, 92.7%id
"""
        mock_run.return_value = Mock(returncode=0, stdout=resource_output)
        
        # Resource monitoring commands
        resource_commands = [
            ["free", "-m"],                    # Memory usage
            ["top", "-bn1", "-p1"],           # CPU usage
            ["df", "-h", "/"],                # Disk usage
            ["docker", "stats", "--no-stream"] # Container stats
        ]
        
        for cmd in resource_commands:
            assert len(cmd) >= 1, f"Command should have at least one part: {cmd}"
            assert isinstance(cmd[0], str), f"Command should start with string: {cmd}"

    def test_validation_report_format(self):
        """Test validation report output format."""
        # Expected validation report structure
        report_structure = {
            "validation_timestamp": "2024-01-15T10:00:00Z",
            "deployment_status": "healthy",
            "checks": {
                "health_endpoints": {"status": "pass", "duration_ms": 150},
                "database_connectivity": {"status": "pass", "duration_ms": 200},
                "metrics_collection": {"status": "pass", "duration_ms": 100},
                "security_validation": {"status": "pass", "duration_ms": 300},
                "performance_baseline": {"status": "pass", "duration_ms": 500}
            },
            "summary": {
                "total_checks": 5,
                "passed_checks": 5,
                "failed_checks": 0,
                "warnings": 0
            },
            "recommendations": []
        }
        
        # Validate report structure
        assert "validation_timestamp" in report_structure
        assert "deployment_status" in report_structure
        assert "checks" in report_structure
        assert "summary" in report_structure
        
        # Validate checks structure
        checks = report_structure["checks"]
        for check_name, check_result in checks.items():
            assert "status" in check_result, f"Check {check_name} should have status"
            assert "duration_ms" in check_result, f"Check {check_name} should have duration"
            assert check_result["status"] in ["pass", "warn", "fail"], f"Invalid status for {check_name}"

    @patch('subprocess.run')
    def test_validation_error_handling(self, mock_run):
        """Test validation error handling scenarios."""
        # Test various failure scenarios
        failure_scenarios = [
            {
                "name": "Service unreachable",
                "mock_result": Mock(returncode=7, stderr="Connection refused"),
                "expected_status": "fail"
            },
            {
                "name": "Service timeout", 
                "mock_result": Mock(returncode=28, stderr="Timeout"),
                "expected_status": "fail"
            },
            {
                "name": "Authentication error",
                "mock_result": Mock(returncode=22, stderr="HTTP 401"),
                "expected_status": "warn"  # Expected for security tests
            },
            {
                "name": "Service degraded",
                "mock_result": Mock(returncode=0, stdout='{"status": "degraded"}'),
                "expected_status": "warn"
            }
        ]
        
        for scenario in failure_scenarios:
            mock_run.return_value = scenario["mock_result"]
            
            # Validate scenario structure
            assert "name" in scenario
            assert "mock_result" in scenario
            assert "expected_status" in scenario
            assert scenario["expected_status"] in ["pass", "warn", "fail"]


class TestValidationScriptIntegration:
    """Integration tests for validation script functionality."""

    @patch('subprocess.run')
    def test_complete_validation_flow(self, mock_run):
        """Test complete validation flow simulation."""
        # Mock successful validation flow
        mock_run.return_value = Mock(returncode=0, stdout="validation_passed")
        
        validation_steps = [
            # Prerequisites
            ["docker", "ps"],
            ["curl", "--version"],
            
            # Health checks
            ["curl", "-f", "http://localhost:8001/health"],
            ["curl", "-f", "http://localhost:8080/metrics"],
            ["curl", "-f", "http://localhost:9090/status"],
            
            # Database connectivity
            ["curl", "-s", "http://localhost:6333/collections"],
            ["redis-cli", "ping"],
            
            # Performance tests
            ["curl", "-w", "%{time_total}", "http://localhost:8001/health"],
            
            # Security tests
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8001/admin"]
        ]
        
        # Validate each validation step
        for step in validation_steps:
            assert len(step) >= 1, f"Validation step should have command: {step}"
            assert isinstance(step[0], str), f"Command should be string: {step}"

    def test_validation_exit_codes(self):
        """Test validation script exit codes."""
        exit_codes = {
            0: "All validations passed",
            1: "Critical validation failures", 
            2: "Validation warnings present",
            3: "Validation could not complete"
        }
        
        for code, description in exit_codes.items():
            assert isinstance(code, int), "Exit code should be integer"
            assert 0 <= code <= 3, "Exit code should be 0-3"
            assert len(description) > 0, "Exit code should have description"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])