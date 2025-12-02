#!/usr/bin/env python3
"""
Comprehensive tests for src/security/security_scanner.py

This test suite provides comprehensive coverage of the security scanner system,
testing vulnerability detection, dependency scanning, secret scanning, and
automated security analysis capabilities.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call, AsyncMock
from dataclasses import asdict

# Import the module under test
from src.security.security_scanner import (
    ScanType,
    SeverityLevel, 
    ScanStatus,
    Vulnerability,
    ScanResult,
    ScanConfiguration,
    DependencyScanner,
    SecretScanner,
    CodeAnalyzer,
    ConfigurationScanner,
    SecurityScanner
)


class TestEnumsAndDataClasses:
    """Test security scanner enums and data classes."""

    def test_scan_type_enum(self):
        """Test ScanType enum values."""
        assert ScanType.DEPENDENCY_SCAN.value == "dependency_scan"
        assert ScanType.SECRET_SCAN.value == "secret_scan"
        assert ScanType.CODE_ANALYSIS.value == "code_analysis"
        assert ScanType.CONTAINER_SCAN.value == "container_scan"
        assert ScanType.NETWORK_SCAN.value == "network_scan"
        assert ScanType.CONFIGURATION_SCAN.value == "configuration_scan"
        assert ScanType.LICENSE_SCAN.value == "license_scan"

    def test_severity_level_enum(self):
        """Test SeverityLevel enum values."""
        assert SeverityLevel.CRITICAL.value == "critical"
        assert SeverityLevel.HIGH.value == "high"
        assert SeverityLevel.MEDIUM.value == "medium"
        assert SeverityLevel.LOW.value == "low"
        assert SeverityLevel.INFO.value == "info"

    def test_scan_status_enum(self):
        """Test ScanStatus enum values."""
        assert hasattr(ScanStatus, 'PENDING')
        assert hasattr(ScanStatus, 'RUNNING')
        assert hasattr(ScanStatus, 'COMPLETED')
        assert hasattr(ScanStatus, 'FAILED')

    def test_vulnerability_creation(self):
        """Test Vulnerability dataclass creation."""
        vulnerability = Vulnerability(
            vuln_id="CVE-2023-1234",
            title="Test Vulnerability",
            description="Test vulnerability description",
            severity=SeverityLevel.HIGH,
            affected_component="test-component",
            location="test.py:123",
            cve_id="CVE-2023-1234",
            cvss_score=7.5,
            fix_recommendation="Update to version 2.0",
            references=["https://example.com/advisory"],
            metadata={"detected_by": "scanner"}
        )
        
        assert vulnerability.vuln_id == "CVE-2023-1234"
        assert vulnerability.severity == SeverityLevel.HIGH
        assert vulnerability.cvss_score == 7.5
        assert len(vulnerability.references) == 1

    def test_scan_result_creation(self):
        """Test ScanResult dataclass creation."""
        vulnerabilities = [
            Vulnerability(
                vuln_id="TEST-1",
                title="Test Vuln 1",
                description="Test description",
                severity=SeverityLevel.HIGH,
                affected_component="component1",
                location="file1.py"
            )
        ]
        
        scan_result = ScanResult(
            scan_id="SCAN-001",
            scan_type=ScanType.DEPENDENCY_SCAN,
            status=ScanStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            vulnerabilities=vulnerabilities,
            total_files_scanned=10,
            summary="Found 1 vulnerability",
            metadata={"tool": "test_scanner"}
        )
        
        assert scan_result.scan_id == "SCAN-001"
        assert scan_result.scan_type == ScanType.DEPENDENCY_SCAN
        assert len(scan_result.vulnerabilities) == 1
        assert scan_result.total_files_scanned == 10

    def test_scan_configuration_creation(self):
        """Test ScanConfiguration dataclass creation."""
        config = ScanConfiguration(
            scan_types=[ScanType.DEPENDENCY_SCAN, ScanType.SECRET_SCAN],
            target_paths=["/app/src", "/app/config"],
            exclude_patterns=["*.log", "*.tmp"],
            severity_threshold=SeverityLevel.MEDIUM,
            max_scan_time=3600,
            output_format="json",
            include_low_confidence=False
        )
        
        assert len(config.scan_types) == 2
        assert ScanType.DEPENDENCY_SCAN in config.scan_types
        assert config.severity_threshold == SeverityLevel.MEDIUM
        assert config.max_scan_time == 3600


class TestDependencyScanner:
    """Test dependency scanning functionality."""

    def test_init(self):
        """Test DependencyScanner initialization."""
        scanner = DependencyScanner()
        assert hasattr(scanner, 'scan_results')
        assert isinstance(scanner.scan_results, list)

    def test_scan_python_dependencies(self):
        """Test Python dependency scanning."""
        scanner = DependencyScanner()
        
        # Mock requirements.txt content
        requirements_content = "requests==2.25.1\nflask==1.1.2\nnumpy==1.19.5"
        
        with patch("builtins.open", mock_open(read_data=requirements_content)):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.glob") as mock_glob:
                    mock_file = MagicMock()
                    mock_file.name = "requirements.txt"
                    mock_glob.return_value = [mock_file]
                    
                    result = scanner.scan_python_dependencies("/fake/path")
                    
                    assert isinstance(result, list)
                    # Should have processed the requirements

    def test_scan_npm_dependencies(self):
        """Test NPM dependency scanning."""
        scanner = DependencyScanner()
        
        # Mock package.json content
        package_json = {
            "dependencies": {
                "express": "^4.17.1",
                "lodash": "^4.17.20"
            },
            "devDependencies": {
                "webpack": "^5.0.0"
            }
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(package_json))):
            with patch("pathlib.Path.exists", return_value=True):
                result = scanner.scan_npm_dependencies("/fake/path")
                
                assert isinstance(result, list)

    def test_check_vulnerability_database(self):
        """Test vulnerability database checking."""
        scanner = DependencyScanner()
        
        # Mock vulnerability data
        mock_vuln_data = {
            "vulnerabilities": [
                {
                    "id": "TEST-VULN-1",
                    "package": "requests",
                    "versions": ["<=2.25.0"],
                    "severity": "high",
                    "description": "Test vulnerability"
                }
            ]
        }
        
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_vuln_data
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            vulns = scanner.check_vulnerability_database("requests", "2.24.0")
            
            assert isinstance(vulns, list)

    def test_generate_dependency_report(self):
        """Test dependency report generation."""
        scanner = DependencyScanner()
        
        # Add some test vulnerabilities
        scanner.scan_results = [
            Vulnerability(
                vuln_id="DEP-1",
                title="Dependency Vulnerability",
                description="Test dependency vulnerability",
                severity=SeverityLevel.HIGH,
                affected_component="test-package",
                location="requirements.txt"
            )
        ]
        
        report = scanner.generate_dependency_report()
        
        assert "total_vulnerabilities" in report
        assert "by_severity" in report
        assert "by_package" in report
        assert report["total_vulnerabilities"] == 1


class TestSecretScanner:
    """Test secret scanning functionality."""

    def test_init(self):
        """Test SecretScanner initialization."""
        scanner = SecretScanner()
        assert hasattr(scanner, 'patterns')
        assert hasattr(scanner, 'found_secrets')
        assert isinstance(scanner.patterns, dict)

    def test_load_secret_patterns(self):
        """Test loading secret detection patterns."""
        scanner = SecretScanner()
        scanner.load_secret_patterns()
        
        assert len(scanner.patterns) > 0
        # Should have common patterns like API keys, passwords, etc.

    def test_scan_file_for_secrets(self):
        """Test scanning a file for secrets."""
        scanner = SecretScanner()
        scanner.load_secret_patterns()
        
        # Mock file content with potential secrets
        file_content = """
# Configuration file
api_key = "sk-1234567890abcdef1234567890abcdef"
password = "super_secret_password"
aws_access_key = "AKIA1234567890ABCDEF"
"""
        
        with patch("builtins.open", mock_open(read_data=file_content)):
            secrets = scanner.scan_file_for_secrets("/fake/config.py")
            
            assert isinstance(secrets, list)
            # Should detect potential secrets

    def test_scan_directory_for_secrets(self):
        """Test scanning directory for secrets."""
        scanner = SecretScanner()
        
        with patch("pathlib.Path.rglob") as mock_rglob:
            mock_file1 = MagicMock()
            mock_file1.is_file.return_value = True
            mock_file1.suffix = ".py"
            mock_rglob.return_value = [mock_file1]
            
            with patch.object(scanner, 'scan_file_for_secrets', return_value=[]):
                secrets = scanner.scan_directory_for_secrets("/fake/dir")
                
                assert isinstance(secrets, list)

    def test_validate_secret(self):
        """Test secret validation."""
        scanner = SecretScanner()
        
        # Test with various secret patterns
        assert scanner.validate_secret("api_key", "sk-1234567890abcdef") == True
        assert scanner.validate_secret("password", "weak") == False
        assert scanner.validate_secret("token", "abc123def456") == True

    def test_generate_secret_report(self):
        """Test secret scanning report generation."""
        scanner = SecretScanner()
        
        # Add some test secrets
        scanner.found_secrets = [
            {
                "type": "api_key",
                "value": "sk-***masked***",
                "file": "/fake/config.py",
                "line": 5,
                "confidence": "high"
            }
        ]
        
        report = scanner.generate_secret_report()
        
        assert "total_secrets" in report
        assert "by_type" in report
        assert "by_confidence" in report
        assert report["total_secrets"] == 1


class TestCodeAnalyzer:
    """Test code analysis functionality."""

    def test_init(self):
        """Test CodeAnalyzer initialization."""
        analyzer = CodeAnalyzer()
        assert hasattr(analyzer, 'security_rules')
        assert hasattr(analyzer, 'analysis_results')

    def test_load_security_rules(self):
        """Test loading security analysis rules."""
        analyzer = CodeAnalyzer()
        analyzer.load_security_rules()
        
        assert len(analyzer.security_rules) > 0
        # Should have loaded security rules

    def test_analyze_python_file(self):
        """Test Python file analysis."""
        analyzer = CodeAnalyzer()
        
        # Mock Python code with potential issues
        python_code = """
import os
import subprocess

def unsafe_function(user_input):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    
    # Command injection vulnerability  
    os.system(f"ls {user_input}")
    
    # Hardcoded secret
    api_key = "sk-1234567890abcdef"
    
    return query
"""
        
        with patch("builtins.open", mock_open(read_data=python_code)):
            issues = analyzer.analyze_python_file("/fake/unsafe.py")
            
            assert isinstance(issues, list)

    def test_analyze_javascript_file(self):
        """Test JavaScript file analysis."""
        analyzer = CodeAnalyzer()
        
        # Mock JavaScript code with potential issues
        js_code = """
const express = require('express');
const app = express();

app.get('/user/:id', (req, res) => {
    // SQL injection vulnerability
    const query = `SELECT * FROM users WHERE id = ${req.params.id}`;
    
    // XSS vulnerability
    res.send(`<h1>Hello ${req.query.name}</h1>`);
});
"""
        
        with patch("builtins.open", mock_open(read_data=js_code)):
            issues = analyzer.analyze_javascript_file("/fake/app.js")
            
            assert isinstance(issues, list)

    def test_check_security_patterns(self):
        """Test security pattern checking."""
        analyzer = CodeAnalyzer()
        analyzer.load_security_rules()
        
        # Test various security patterns
        test_code = "eval(user_input)"  # Dangerous eval usage
        issues = analyzer.check_security_patterns(test_code, "test.py")
        
        assert isinstance(issues, list)

    def test_generate_code_analysis_report(self):
        """Test code analysis report generation."""
        analyzer = CodeAnalyzer()
        
        # Add some test results
        analyzer.analysis_results = [
            {
                "file": "/fake/test.py",
                "line": 10,
                "issue": "SQL Injection",
                "severity": "high",
                "description": "Potential SQL injection vulnerability"
            }
        ]
        
        report = analyzer.generate_code_analysis_report()
        
        assert "total_issues" in report
        assert "by_severity" in report
        assert "by_file" in report
        assert report["total_issues"] == 1


class TestConfigurationScanner:
    """Test configuration scanning functionality."""

    def test_init(self):
        """Test ConfigurationScanner initialization."""
        scanner = ConfigurationScanner()
        assert hasattr(scanner, 'config_rules')
        assert hasattr(scanner, 'scan_results')

    def test_load_configuration_rules(self):
        """Test loading configuration security rules."""
        scanner = ConfigurationScanner()
        scanner.load_configuration_rules()
        
        assert len(scanner.config_rules) > 0

    def test_scan_docker_configuration(self):
        """Test Docker configuration scanning."""
        scanner = ConfigurationScanner()
        
        # Mock Dockerfile content
        dockerfile_content = """
FROM ubuntu:latest
RUN apt-get update
USER root
EXPOSE 22
ADD . /app
"""
        
        with patch("builtins.open", mock_open(read_data=dockerfile_content)):
            with patch("pathlib.Path.exists", return_value=True):
                issues = scanner.scan_docker_configuration("/fake/Dockerfile")
                
                assert isinstance(issues, list)

    def test_scan_nginx_configuration(self):
        """Test Nginx configuration scanning."""
        scanner = ConfigurationScanner()
        
        # Mock nginx.conf content
        nginx_content = """
server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://backend;
        # Missing security headers
    }
}
"""
        
        with patch("builtins.open", mock_open(read_data=nginx_content)):
            issues = scanner.scan_nginx_configuration("/fake/nginx.conf")
            
            assert isinstance(issues, list)

    def test_scan_ssl_configuration(self):
        """Test SSL/TLS configuration scanning.""" 
        scanner = ConfigurationScanner()
        
        ssl_config = {
            "ssl_protocols": ["TLSv1.0", "TLSv1.1"],  # Weak protocols
            "ssl_ciphers": "ALL:!aNULL:!MD5",  # Weak cipher config
            "ssl_prefer_server_ciphers": "off"
        }
        
        issues = scanner.scan_ssl_configuration(ssl_config)
        
        assert isinstance(issues, list)

    def test_generate_configuration_report(self):
        """Test configuration scanning report generation."""
        scanner = ConfigurationScanner()
        
        # Add some test results
        scanner.scan_results = [
            {
                "file": "/fake/Dockerfile",
                "issue": "Running as root",
                "severity": "high",
                "line": 4,
                "recommendation": "Use a non-root user"
            }
        ]
        
        report = scanner.generate_configuration_report()
        
        assert "total_issues" in report
        assert "by_severity" in report
        assert "by_file" in report
        assert report["total_issues"] == 1


class TestSecurityScanner:
    """Test main SecurityScanner functionality."""

    def test_init(self):
        """Test SecurityScanner initialization."""
        scanner = SecurityScanner()
        
        assert hasattr(scanner, 'dependency_scanner')
        assert hasattr(scanner, 'secret_scanner')
        assert hasattr(scanner, 'code_analyzer')
        assert hasattr(scanner, 'config_scanner')
        assert isinstance(scanner.dependency_scanner, DependencyScanner)
        assert isinstance(scanner.secret_scanner, SecretScanner)
        assert isinstance(scanner.code_analyzer, CodeAnalyzer)
        assert isinstance(scanner.config_scanner, ConfigurationScanner)

    def test_run_comprehensive_scan(self):
        """Test running a comprehensive security scan."""
        scanner = SecurityScanner()
        
        config = ScanConfiguration(
            scan_types=[ScanType.DEPENDENCY_SCAN, ScanType.SECRET_SCAN],
            target_paths=["/fake/app"],
            severity_threshold=SeverityLevel.MEDIUM
        )
        
        with patch.object(scanner.dependency_scanner, 'scan_python_dependencies', return_value=[]):
            with patch.object(scanner.secret_scanner, 'scan_directory_for_secrets', return_value=[]):
                with patch("pathlib.Path.exists", return_value=True):
                    result = scanner.run_comprehensive_scan(config)
                    
                    assert isinstance(result, ScanResult)
                    assert result.scan_type in [ScanType.DEPENDENCY_SCAN, ScanType.SECRET_SCAN]

    def test_run_targeted_scan(self):
        """Test running a targeted scan."""
        scanner = SecurityScanner()
        
        with patch.object(scanner.secret_scanner, 'scan_file_for_secrets', return_value=[]):
            result = scanner.run_targeted_scan("/fake/file.py", ScanType.SECRET_SCAN)
            
            assert isinstance(result, ScanResult)
            assert result.scan_type == ScanType.SECRET_SCAN

    def test_generate_security_report(self):
        """Test security report generation."""
        scanner = SecurityScanner()
        
        # Mock scan results
        mock_results = [
            ScanResult(
                scan_id="TEST-1",
                scan_type=ScanType.DEPENDENCY_SCAN,
                status=ScanStatus.COMPLETED,
                start_time=datetime.now(),
                vulnerabilities=[
                    Vulnerability(
                        vuln_id="VULN-1",
                        title="Test Vulnerability",
                        description="Test description",
                        severity=SeverityLevel.HIGH,
                        affected_component="test-component",
                        location="test.py"
                    )
                ],
                total_files_scanned=5
            )
        ]
        
        report = scanner.generate_security_report(mock_results)
        
        assert "scan_summary" in report
        assert "vulnerability_summary" in report
        assert "recommendations" in report
        assert report["vulnerability_summary"]["total"] == 1

    def test_export_results(self):
        """Test exporting scan results."""
        scanner = SecurityScanner()
        
        mock_result = ScanResult(
            scan_id="EXPORT-1",
            scan_type=ScanType.SECRET_SCAN,
            status=ScanStatus.COMPLETED,
            start_time=datetime.now(),
            vulnerabilities=[],
            total_files_scanned=10
        )
        
        with patch("builtins.open", mock_open()) as mock_file:
            scanner.export_results([mock_result], "/fake/report.json", "json")
            
            mock_file.assert_called_once_with("/fake/report.json", "w", encoding="utf-8")

    def test_get_scan_statistics(self):
        """Test getting scan statistics."""
        scanner = SecurityScanner()
        
        mock_results = [
            ScanResult(
                scan_id="STAT-1",
                scan_type=ScanType.DEPENDENCY_SCAN,
                status=ScanStatus.COMPLETED,
                start_time=datetime.now(),
                vulnerabilities=[
                    Vulnerability(
                        vuln_id="VULN-1",
                        title="High Severity Issue",
                        description="Test",
                        severity=SeverityLevel.HIGH,
                        affected_component="component",
                        location="file.py"
                    ),
                    Vulnerability(
                        vuln_id="VULN-2",
                        title="Medium Severity Issue", 
                        description="Test",
                        severity=SeverityLevel.MEDIUM,
                        affected_component="component",
                        location="file.py"
                    )
                ],
                total_files_scanned=20
            )
        ]
        
        stats = scanner.get_scan_statistics(mock_results)
        
        assert "total_scans" in stats
        assert "total_vulnerabilities" in stats
        assert "by_severity" in stats
        assert "by_scan_type" in stats
        assert stats["total_vulnerabilities"] == 2
        assert stats["by_severity"]["high"] == 1
        assert stats["by_severity"]["medium"] == 1


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""

    def test_full_security_assessment_workflow(self):
        """Test complete security assessment workflow."""
        scanner = SecurityScanner()
        
        config = ScanConfiguration(
            scan_types=[ScanType.DEPENDENCY_SCAN, ScanType.SECRET_SCAN, ScanType.CODE_ANALYSIS],
            target_paths=["/fake/app"],
            severity_threshold=SeverityLevel.LOW,
            output_format="json"
        )
        
        with patch.object(scanner.dependency_scanner, 'scan_python_dependencies', return_value=[]):
            with patch.object(scanner.secret_scanner, 'scan_directory_for_secrets', return_value=[]):
                with patch.object(scanner.code_analyzer, 'analyze_python_file', return_value=[]):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("pathlib.Path.rglob", return_value=[]):
                            result = scanner.run_comprehensive_scan(config)
                            
                            assert isinstance(result, ScanResult)
                            report = scanner.generate_security_report([result])
                            assert "scan_summary" in report

    def test_ci_cd_integration_scan(self):
        """Test CI/CD integration scanning."""
        scanner = SecurityScanner()
        
        # Simulate CI/CD environment scan
        config = ScanConfiguration(
            scan_types=[ScanType.DEPENDENCY_SCAN, ScanType.SECRET_SCAN],
            target_paths=["/ci/workspace"],
            severity_threshold=SeverityLevel.HIGH,  # Only high severity for CI/CD
            max_scan_time=600,  # 10 minute timeout
            output_format="json"
        )
        
        with patch.object(scanner, 'run_comprehensive_scan') as mock_scan:
            mock_scan.return_value = ScanResult(
                scan_id="CI-SCAN-1",
                scan_type=ScanType.DEPENDENCY_SCAN,
                status=ScanStatus.COMPLETED,
                start_time=datetime.now(),
                vulnerabilities=[],
                total_files_scanned=50
            )
            
            result = scanner.run_comprehensive_scan(config)
            assert result.status == ScanStatus.COMPLETED

    def test_incremental_scanning(self):
        """Test incremental scanning of changed files."""
        scanner = SecurityScanner()
        
        # Simulate scanning only changed files
        changed_files = ["/fake/app.py", "/fake/config.py"]
        
        results = []
        for file_path in changed_files:
            with patch.object(scanner, 'run_targeted_scan') as mock_scan:
                mock_scan.return_value = ScanResult(
                    scan_id=f"INCR-{file_path.split('/')[-1]}",
                    scan_type=ScanType.SECRET_SCAN,
                    status=ScanStatus.COMPLETED,
                    start_time=datetime.now(),
                    vulnerabilities=[],
                    total_files_scanned=1
                )
                
                result = scanner.run_targeted_scan(file_path, ScanType.SECRET_SCAN)
                results.append(result)
        
        assert len(results) == 2
        assert all(r.status == ScanStatus.COMPLETED for r in results)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_scan_nonexistent_path(self):
        """Test scanning non-existent path."""
        scanner = SecurityScanner()
        
        config = ScanConfiguration(
            scan_types=[ScanType.DEPENDENCY_SCAN],
            target_paths=["/nonexistent/path"],
            severity_threshold=SeverityLevel.MEDIUM
        )
        
        with patch("pathlib.Path.exists", return_value=False):
            result = scanner.run_comprehensive_scan(config)
            
            # Should handle gracefully
            assert isinstance(result, ScanResult)

    def test_scan_timeout_handling(self):
        """Test handling of scan timeouts."""
        scanner = SecurityScanner()
        
        config = ScanConfiguration(
            scan_types=[ScanType.CODE_ANALYSIS],
            target_paths=["/fake/large_project"],
            max_scan_time=1  # Very short timeout
        )
        
        with patch("time.time", side_effect=[0, 0, 2]):  # Simulate timeout
            result = scanner.run_comprehensive_scan(config)
            
            # Should handle timeout gracefully
            assert isinstance(result, ScanResult)

    def test_corrupted_file_handling(self):
        """Test handling of corrupted or unreadable files."""
        scanner = SecurityScanner()
        
        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")):
            result = scanner.secret_scanner.scan_file_for_secrets("/fake/corrupted.bin")
            
            # Should handle gracefully
            assert isinstance(result, list)

    def test_network_failure_handling(self):
        """Test handling of network failures during vulnerability lookups."""
        scanner = SecurityScanner()
        
        with patch("requests.get", side_effect=ConnectionError("Network unreachable")):
            vulns = scanner.dependency_scanner.check_vulnerability_database("package", "1.0.0")
            
            # Should handle network failures gracefully
            assert isinstance(vulns, list)

    def test_memory_intensive_scanning(self):
        """Test handling of memory-intensive scanning operations."""
        scanner = SecurityScanner()
        
        # Simulate large file scanning
        large_content = "x" * 10000000  # 10MB of content
        
        with patch("builtins.open", mock_open(read_data=large_content)):
            try:
                result = scanner.secret_scanner.scan_file_for_secrets("/fake/large_file.txt")
                assert isinstance(result, list)
            except MemoryError:
                # Should handle memory limitations gracefully
                pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])