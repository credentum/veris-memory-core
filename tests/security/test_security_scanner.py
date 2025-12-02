"""
Security Scanner Tests
Sprint 10 Phase 3 - Issue 008: SEC-108
Tests the automated security scanning system
"""

import pytest
import os
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.security.security_scanner import (
    SecurityScanner,
    ScanType,
    SeverityLevel,
    ScanStatus,
    Vulnerability,
    ScanResult,
    ScanConfiguration,
    DependencyScanner,
    SecretScanner,
    CodeAnalyzer,
    ConfigurationScanner
)


class TestDependencyScanner:
    """Test ID: SEC-108-A - Dependency Vulnerability Scanning"""
    
    def test_requirements_file_parsing(self):
        """Test parsing Python requirements.txt file"""
        scanner = DependencyScanner()
        
        # Create temporary requirements file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
# Core dependencies
requests==2.25.1
urllib3==1.25.8
django>=3.0.0
flask~=2.0.0
numpy>1.20.0
# Development dependencies
pytest
""")
            req_file = f.name
        
        try:
            # Test parsing
            test_lines = [
                "requests==2.25.1",
                "urllib3>=1.25.8", 
                "django~=3.0.0",
                "flask>2.0.0",
                "pytest"
            ]
            
            for line in test_lines:
                result = scanner._parse_requirement_line(line)
                assert result is not None
                package_name, version = result
                assert len(package_name) > 0
                
            # Test invalid lines
            invalid_lines = ["", "# comment", "git+https://github.com/user/repo.git"]
            for line in invalid_lines[:2]:  # Skip git URL for now
                result = scanner._parse_requirement_line(line)
                if line.startswith('#') or not line.strip():
                    assert result is None
        
        finally:
            os.unlink(req_file)
    
    def test_vulnerability_detection(self):
        """Test detection of vulnerable packages"""
        scanner = DependencyScanner()
        
        # Test known vulnerable packages
        urllib3_vulns = scanner._check_package_vulnerabilities("urllib3", "1.25.8")
        assert len(urllib3_vulns) > 0
        assert any("CVE-2020-26137" in vuln.cve_ids for vuln in urllib3_vulns)
        
        requests_vulns = scanner._check_package_vulnerabilities("requests", "2.25.1")
        assert len(requests_vulns) > 0
        assert any("CVE-2023-32681" in vuln.cve_ids for vuln in requests_vulns)
        
        # Test non-vulnerable package
        safe_vulns = scanner._check_package_vulnerabilities("numpy", "1.21.0")
        assert len(safe_vulns) == 0
    
    def test_requirements_file_scanning(self):
        """Test full requirements file vulnerability scanning"""
        scanner = DependencyScanner()
        
        # Create requirements file with vulnerable packages
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
urllib3==1.25.8
requests==2.25.1
safe_package==1.0.0
""")
            req_file = f.name
        
        try:
            vulnerabilities = scanner.scan_requirements_file(req_file)
            
            # Should find vulnerabilities in urllib3 and requests
            assert len(vulnerabilities) >= 2
            
            vuln_packages = [v.affected_component for v in vulnerabilities]
            assert "urllib3" in vuln_packages
            assert "requests" in vuln_packages
            
            # Check vulnerability details
            for vuln in vulnerabilities:
                assert vuln.severity in [SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
                assert vuln.file_path == req_file
                assert vuln.line_number > 0
                assert vuln.remediation is not None
        
        finally:
            os.unlink(req_file)
    
    def test_package_json_scanning(self):
        """Test Node.js package.json vulnerability scanning"""
        scanner = DependencyScanner()
        
        # Create package.json with dependencies
        package_data = {
            "name": "test-project",
            "dependencies": {
                "express": "^4.17.1",
                "lodash": "^4.17.20"
            },
            "devDependencies": {
                "mocha": "^8.0.0"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(package_data, f, indent=2)
            package_file = f.name
        
        try:
            vulnerabilities = scanner.scan_package_json(package_file)
            # Currently returns empty list (would integrate with npm audit API)
            assert isinstance(vulnerabilities, list)
        
        finally:
            os.unlink(package_file)


class TestSecretScanner:
    """Test ID: SEC-108-B - Secret Detection Scanning"""
    
    def test_file_secret_detection(self):
        """Test secret detection in individual files"""
        scanner = SecretScanner()
        
        # Create file with various secrets
        secret_content = '''
# Configuration file
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"
AWS_ACCESS_KEY_ID = "AKIA1234567890ABCDEF"
JWT_SECRET = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
password = "MySecretPassword123!"

# GitHub token
GITHUB_TOKEN = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"

# Private key
private_key = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890abcdef...
-----END RSA PRIVATE KEY-----"""
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(secret_content)
            secret_file = f.name
        
        try:
            vulnerabilities = scanner.scan_file(secret_file)
            
            assert len(vulnerabilities) >= 3  # Should detect multiple secrets
            
            # Check detected secrets
            secret_types = [v.title for v in vulnerabilities]
            assert any("api" in s.lower() for s in secret_types)
            assert any("token" in s.lower() or "key" in s.lower() for s in secret_types)
            
            # Check vulnerability details
            for vuln in vulnerabilities:
                assert vuln.severity in [SeverityLevel.HIGH, SeverityLevel.MEDIUM]
                assert vuln.file_path == secret_file
                assert vuln.line_number > 0
                assert vuln.evidence is not None
                assert "secret" in vuln.tags
        
        finally:
            os.unlink(secret_file)
    
    def test_directory_secret_scanning(self):
        """Test recursive directory secret scanning"""
        scanner = SecretScanner()
        
        # Create temporary directory structure
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create files with secrets
            config_dir = os.path.join(temp_dir, "config")
            os.makedirs(config_dir)
            
            # Python file with secret
            with open(os.path.join(temp_dir, "app.py"), 'w') as f:
                f.write('API_KEY = "sk-abc123def456ghi789jkl012mno345pqr"')
            
            # Config file with secret
            with open(os.path.join(config_dir, "settings.json"), 'w') as f:
                json.dump({"database_password": "SuperSecret123!"}, f)
            
            # Environment file
            with open(os.path.join(temp_dir, ".env"), 'w') as f:
                f.write("JWT_SECRET=very_secret_jwt_signing_key_here")
            
            vulnerabilities = scanner.scan_directory(temp_dir)
            
            assert len(vulnerabilities) >= 2  # Should find secrets in multiple files
            
            file_paths = [v.file_path for v in vulnerabilities]
            assert any("app.py" in path for path in file_paths)
            assert any(".env" in path for path in file_paths)
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_secret_pattern_matching(self):
        """Test specific secret pattern detection"""
        scanner = SecretScanner()
        
        test_secrets = [
            ("AKIA1234567890ABCDEF", "aws_access_key"),
            ("ghp_" + "x" * 36, "github_token"),
            ("sk_test_" + "x" * 24, "stripe_key"), 
            ("xoxb-" + "x" * 51, "slack_token"),
        ]
        
        for secret_value, expected_type in test_secrets:
            content = f'token = "{secret_value}"'
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(content)
                test_file = f.name
            
            try:
                vulnerabilities = scanner.scan_file(test_file)
                
                # Should detect at least one secret
                assert len(vulnerabilities) >= 1
                
                # Check if expected type pattern was detected
                titles = [v.title.lower() for v in vulnerabilities]
                assert any(expected_type.replace('_', ' ') in title or 
                          "secret" in title for title in titles)
            
            finally:
                os.unlink(test_file)
    
    def test_false_positive_filtering(self):
        """Test filtering of common false positives"""
        scanner = SecretScanner()
        
        # Content that looks like secrets but isn't
        false_positive_content = '''
# Example configuration 
example_key = "your_api_key_here"
placeholder = "xxxxxxxxxxxxxxxx"
test_secret = "test_value_123"
demo_token = "demo123demo123demo123"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(false_positive_content)
            test_file = f.name
        
        try:
            vulnerabilities = scanner.scan_file(test_file)
            
            # Should have minimal false positives
            # Real secrets would have higher entropy
            low_confidence = [v for v in vulnerabilities if v.confidence < 0.9]
            assert len(low_confidence) <= 2  # Some pattern matches might occur
        
        finally:
            os.unlink(test_file)


class TestCodeAnalyzer:
    """Test ID: SEC-108-C - Static Code Analysis"""
    
    def test_python_dangerous_functions(self):
        """Test detection of dangerous Python function usage"""
        analyzer = CodeAnalyzer()
        
        dangerous_code = '''
import os
import pickle
import subprocess

def vulnerable_function(user_input):
    # Dangerous: code injection
    result = eval(user_input)
    
    # Dangerous: command injection  
    os.system(f"ls {user_input}")
    
    # Dangerous: deserialization
    data = pickle.loads(user_input)
    
    # Dangerous: subprocess without validation
    subprocess.call(user_input, shell=True)
    
    return result
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(dangerous_code)
            py_file = f.name
        
        try:
            vulnerabilities = analyzer.scan_python_file(py_file)
            
            assert len(vulnerabilities) >= 4  # Should detect all dangerous functions
            
            detected_functions = [v.title for v in vulnerabilities]
            assert any("eval" in title.lower() for title in detected_functions)
            assert any("system" in title.lower() for title in detected_functions)
            assert any("pickle" in title.lower() for title in detected_functions)
            assert any("subprocess" in title.lower() for title in detected_functions)
            
            # Check vulnerability details
            for vuln in vulnerabilities:
                assert vuln.severity == SeverityLevel.HIGH
                assert "CWE-94" in vuln.cwe_ids  # Code Injection
                assert vuln.file_path == py_file
                assert vuln.line_number > 0
        
        finally:
            os.unlink(py_file)
    
    def test_hardcoded_credentials_detection(self):
        """Test detection of hardcoded credentials"""
        analyzer = CodeAnalyzer()
        
        credential_code = '''
# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'password': 'hardcoded_db_password_123',
    'user': 'admin'
}

# API configuration
api_key = "hardcoded_api_key_abc123def456"
secret = "this_is_a_hardcoded_secret"

# OAuth settings
CLIENT_SECRET = "oauth_client_secret_xyz789"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(credential_code)
            py_file = f.name
        
        try:
            vulnerabilities = analyzer.scan_python_file(py_file)
            
            assert len(vulnerabilities) >= 2  # Should detect hardcoded credentials
            
            cred_vulns = [v for v in vulnerabilities if "hardcoded" in v.title.lower()]
            assert len(cred_vulns) >= 1
            
            for vuln in cred_vulns:
                assert vuln.severity == SeverityLevel.MEDIUM
                assert "CWE-798" in vuln.cwe_ids  # Hard-coded Credentials
        
        finally:
            os.unlink(py_file)
    
    def test_javascript_security_issues(self):
        """Test detection of JavaScript security issues"""
        analyzer = CodeAnalyzer()
        
        js_code = '''
function vulnerableFunction(userInput) {
    // Dangerous: code injection
    var result = eval(userInput);
    
    // Dangerous: XSS via innerHTML
    document.getElementById('content').innerHTML = userInput;
    
    // Dangerous: XSS via document.write
    document.write('<p>' + userInput + '</p>');
    
    // Dangerous: setTimeout with string
    setTimeout('alert("' + userInput + '")', 1000);
    
    return result;
}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(js_code)
            js_file = f.name
        
        try:
            vulnerabilities = analyzer.scan_javascript_file(js_file)
            
            assert len(vulnerabilities) >= 3  # Should detect multiple JS issues
            
            issue_types = [v.description.lower() for v in vulnerabilities]
            assert any("eval" in desc for desc in issue_types)
            assert any("innerhtml" in desc for desc in issue_types)
            assert any("settimeout" in desc for desc in issue_types)
        
        finally:
            os.unlink(js_file)
    
    def test_safe_code_no_false_positives(self):
        """Test that safe code doesn't trigger false positives"""
        analyzer = CodeAnalyzer()
        
        safe_code = '''
import json
import logging

def safe_function(data):
    # Safe operations
    logger = logging.getLogger(__name__)
    logger.info("Processing data")
    
    # Safe JSON operations
    result = json.loads(data)
    
    # Safe string operations
    clean_data = data.strip().lower()
    
    return {"result": result, "clean": clean_data}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(safe_code)
            py_file = f.name
        
        try:
            vulnerabilities = analyzer.scan_python_file(py_file)
            
            # Should not detect any vulnerabilities in safe code
            assert len(vulnerabilities) == 0
        
        finally:
            os.unlink(py_file)


class TestConfigurationScanner:
    """Test ID: SEC-108-D - Configuration Security Scanning"""
    
    def test_dockerfile_security_issues(self):
        """Test Dockerfile security issue detection"""
        scanner = ConfigurationScanner()
        
        dockerfile_content = '''
FROM ubuntu:20.04

# Security issue: running as root
USER root

# Security issue: using ADD instead of COPY  
ADD package.tar.gz /app/

# Better: using COPY
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y python3

WORKDIR /app
CMD ["python3", "app.py"]
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='', delete=False) as f:
            f.write(dockerfile_content)
            dockerfile = f.name
        
        try:
            vulnerabilities = scanner.scan_docker_file(dockerfile)
            
            assert len(vulnerabilities) >= 2  # Should detect root user and ADD usage
            
            issue_types = [v.title.lower() for v in vulnerabilities]
            assert any("root" in title for title in issue_types)
            assert any("add" in title for title in issue_types)
            
            # Check severity levels
            root_vulns = [v for v in vulnerabilities if "root" in v.title.lower()]
            if root_vulns:
                assert root_vulns[0].severity == SeverityLevel.HIGH
            
            add_vulns = [v for v in vulnerabilities if "add" in v.title.lower()]
            if add_vulns:
                assert add_vulns[0].severity == SeverityLevel.LOW
        
        finally:
            os.unlink(dockerfile)
    
    def test_nginx_security_headers(self):
        """Test Nginx security header detection"""
        scanner = ConfigurationScanner()
        
        # Nginx config missing security headers
        nginx_config = '''
server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Missing security headers:
    # - X-Frame-Options
    # - X-Content-Type-Options  
    # - X-XSS-Protection
    # - Strict-Transport-Security
}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write(nginx_config)
            f.flush()  # Ensure content is written
            nginx_file = f.name
        
        try:
            vulnerabilities = scanner.scan_nginx_config(nginx_file)
            
            # Should detect missing security headers
            assert len(vulnerabilities) >= 3
            
            header_types = [v.title for v in vulnerabilities]
            assert any("X-Frame-Options" in title for title in header_types)
            assert any("X-Content-Type-Options" in title for title in header_types)
            assert any("X-XSS-Protection" in title for title in header_types)
            
            for vuln in vulnerabilities:
                assert vuln.severity == SeverityLevel.MEDIUM
                assert "security-headers" in vuln.tags
        
        finally:
            os.unlink(nginx_file)
    
    def test_secure_nginx_config(self):
        """Test that secure Nginx config doesn't trigger issues"""
        scanner = ConfigurationScanner()
        
        secure_nginx_config = '''
server {
    listen 443 ssl;
    server_name example.com;
    
    # Security headers present
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-Content-Type-Options "nosniff";
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    location / {
        proxy_pass http://backend;
    }
}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
            f.write(secure_nginx_config)
            nginx_file = f.name
        
        try:
            vulnerabilities = scanner.scan_nginx_config(nginx_file)
            
            # Should not detect missing headers
            missing_headers = [v for v in vulnerabilities if "Missing" in v.title]
            assert len(missing_headers) == 0
        
        finally:
            os.unlink(nginx_file)


class TestSecurityScanner:
    """Test ID: SEC-108-E - Complete Security Scanner"""
    
    def test_project_scanning_integration(self):
        """Test complete project scanning workflow"""
        scanner = SecurityScanner()
        
        # Create test project structure
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create vulnerable files
            
            # Python file with issues
            with open(os.path.join(temp_dir, "app.py"), 'w') as f:
                f.write('''
import os

def vulnerable_func(user_input):
    # Secret
    api_key = "sk-1234567890abcdef1234567890abcdef12345678"
    
    # Dangerous function
    result = eval(user_input)
    
    return result
''')
            
            # Requirements with vulnerable packages
            with open(os.path.join(temp_dir, "requirements.txt"), 'w') as f:
                f.write("urllib3==1.25.8\nrequests==2.25.1\n")
            
            # Dockerfile with issues
            with open(os.path.join(temp_dir, "Dockerfile"), 'w') as f:
                f.write('''
FROM ubuntu:20.04
USER root
ADD app.tar.gz /app/
''')
            
            # Run complete scan
            result = scanner.scan_project(temp_dir)
            
            assert result.status == ScanStatus.COMPLETED
            assert len(result.vulnerabilities) >= 5  # Should find multiple issues
            
            # Check different types of vulnerabilities found
            vuln_types = set()
            for vuln in result.vulnerabilities:
                if "secret" in vuln.tags:
                    vuln_types.add("secret")
                elif "dangerous-function" in vuln.tags:
                    vuln_types.add("code")
                elif vuln.affected_component:
                    vuln_types.add("dependency")
                elif "docker" in vuln.tags:
                    vuln_types.add("docker")
            
            # Should detect multiple vulnerability types
            assert len(vuln_types) >= 3
            
            # Check summary
            summary = result.get_severity_counts()
            assert summary["high"] >= 1
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_scan_type_filtering(self):
        """Test scanning with specific scan type filters"""
        scanner = SecurityScanner()
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create file with secret
            with open(os.path.join(temp_dir, "config.py"), 'w') as f:
                f.write('SECRET_KEY = "sk-abc123def456ghi789jkl012mno345pqr"')
            
            # Create requirements with vulnerable package
            with open(os.path.join(temp_dir, "requirements.txt"), 'w') as f:
                f.write("urllib3==1.25.8")
            
            # Scan only for secrets
            secret_result = scanner.scan_project(temp_dir, {ScanType.SECRET_SCAN})
            
            secret_vulns = [v for v in secret_result.vulnerabilities if "secret" in v.tags]
            assert len(secret_vulns) >= 1
            
            # Scan only for dependencies
            dep_result = scanner.scan_project(temp_dir, {ScanType.DEPENDENCY_SCAN})
            
            dep_vulns = [v for v in dep_result.vulnerabilities if v.affected_component]
            assert len(dep_vulns) >= 1
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_vulnerability_deduplication(self):
        """Test that duplicate vulnerabilities are removed"""
        scanner = SecurityScanner()
        
        # Create vulnerabilities with duplicates
        vulnerabilities = [
            Vulnerability(
                vuln_id="TEST1",
                title="Test Vulnerability",
                description="Test description", 
                severity=SeverityLevel.HIGH,
                file_path="test.py",
                line_number=10
            ),
            Vulnerability(
                vuln_id="TEST2", 
                title="Test Vulnerability",  # Same title
                description="Test description",
                severity=SeverityLevel.HIGH,
                file_path="test.py",  # Same file
                line_number=10       # Same line
            ),
            Vulnerability(
                vuln_id="TEST3",
                title="Different Vulnerability",
                description="Different description",
                severity=SeverityLevel.MEDIUM,
                file_path="test.py",
                line_number=20  # Different line
            )
        ]
        
        deduplicated = scanner._deduplicate_vulnerabilities(vulnerabilities)
        
        # Should remove the duplicate but keep the different one
        assert len(deduplicated) == 2
        
        titles = [v.title for v in deduplicated]
        assert "Test Vulnerability" in titles
        assert "Different Vulnerability" in titles
    
    def test_scan_history_tracking(self):
        """Test that scan history is properly tracked"""
        scanner = SecurityScanner()
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create simple test file
            with open(os.path.join(temp_dir, "test.py"), 'w') as f:
                f.write("print('hello world')")
            
            # Run multiple scans
            result1 = scanner.scan_project(temp_dir)
            result2 = scanner.scan_project(temp_dir)
            
            # Check history
            history = scanner.get_scan_results()
            assert len(history) >= 2
            
            # Check that results are in chronological order (newest first)
            assert history[0].start_time >= history[1].start_time
            
            # Test filtering by scan ID
            specific_result = scanner.get_scan_results(result1.scan_id)
            assert len(specific_result) == 1
            assert specific_result[0].scan_id == result1.scan_id
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_vulnerability_summary(self):
        """Test vulnerability summary generation"""
        scanner = SecurityScanner()
        
        # Add some mock scan results
        mock_vulnerabilities = [
            Vulnerability("V1", "Critical Issue", "Description", SeverityLevel.CRITICAL),
            Vulnerability("V2", "High Issue", "Description", SeverityLevel.HIGH),
            Vulnerability("V3", "Medium Issue", "Description", SeverityLevel.MEDIUM),
            Vulnerability("V4", "Low Issue", "Description", SeverityLevel.LOW),
        ]
        
        mock_result = ScanResult(
            scan_id="test_scan",
            scan_type=ScanType.CODE_ANALYSIS,
            status=ScanStatus.COMPLETED,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            vulnerabilities=mock_vulnerabilities
        )
        
        scanner.scan_history.append(mock_result)
        
        summary = scanner.get_vulnerability_summary(days=1)
        
        assert summary["total_scans"] >= 1
        assert summary["total_vulnerabilities"] >= 4
        assert summary["high_severity_count"] >= 2  # Critical + High
        assert summary["severity_breakdown"]["critical"] >= 1
        assert summary["severity_breakdown"]["high"] >= 1
    
    def test_export_json_format(self):
        """Test JSON export functionality"""
        scanner = SecurityScanner()
        
        # Create mock scan result
        mock_vuln = Vulnerability(
            vuln_id="EXPORT_TEST",
            title="Test Vulnerability",
            description="Test export functionality",
            severity=SeverityLevel.HIGH,
            cve_ids=["CVE-2023-12345"],
            file_path="test.py",
            line_number=42,
            remediation="Fix the issue"
        )
        
        mock_result = ScanResult(
            scan_id="export_test_scan",
            scan_type=ScanType.CODE_ANALYSIS,
            status=ScanStatus.COMPLETED,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            vulnerabilities=[mock_vuln]
        )
        
        scanner.scan_history.append(mock_result)
        
        # Export to JSON
        json_export = scanner.export_results("export_test_scan", "json")
        
        # Validate JSON structure
        data = json.loads(json_export)
        
        assert data["scan_id"] == "export_test_scan"
        assert data["status"] == "completed"
        assert len(data["vulnerabilities"]) == 1
        
        vuln_data = data["vulnerabilities"][0]
        assert vuln_data["id"] == "EXPORT_TEST"
        assert vuln_data["severity"] == "high"
        assert vuln_data["cve_ids"] == ["CVE-2023-12345"]
        assert vuln_data["line_number"] == 42
    
    def test_export_sarif_format(self):
        """Test SARIF export functionality"""
        scanner = SecurityScanner()
        
        # Create mock scan result
        mock_vuln = Vulnerability(
            vuln_id="SARIF_TEST",
            title="SARIF Test Vulnerability",
            description="Test SARIF export",
            severity=SeverityLevel.CRITICAL,
            file_path="test.py",
            line_number=15
        )
        
        mock_result = ScanResult(
            scan_id="sarif_test_scan",
            scan_type=ScanType.CODE_ANALYSIS,
            status=ScanStatus.COMPLETED,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
            vulnerabilities=[mock_vuln]
        )
        
        scanner.scan_history.append(mock_result)
        
        # Export to SARIF
        sarif_export = scanner.export_results("sarif_test_scan", "sarif")
        
        # Validate SARIF structure
        sarif_data = json.loads(sarif_export)
        
        assert sarif_data["version"] == "2.1.0"
        assert len(sarif_data["runs"]) == 1
        
        run = sarif_data["runs"][0]
        assert run["tool"]["driver"]["name"] == "SecurityScanner"
        assert len(run["results"]) == 1
        
        result = run["results"][0]
        assert result["ruleId"] == "SARIF_TEST"
        assert result["level"] == "error"  # Critical maps to error
        assert len(result["locations"]) == 1
        assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "test.py"
        assert result["locations"][0]["physicalLocation"]["region"]["startLine"] == 15


class TestIntegration:
    """Test ID: SEC-108-F - Integration Tests"""
    
    def test_end_to_end_scanning_workflow(self):
        """Test complete scanning workflow"""
        # Create comprehensive test project
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create complex project structure
            os.makedirs(os.path.join(temp_dir, "src"))
            os.makedirs(os.path.join(temp_dir, "config"))
            
            # Main application with various issues
            with open(os.path.join(temp_dir, "src", "main.py"), 'w') as f:
                f.write('''
import os
import pickle

# Hardcoded secret
DATABASE_PASSWORD = "super_secret_db_password_123"
API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"

def process_user_data(user_input, serialized_data):
    # Dangerous: code injection
    result = eval(f"process({user_input})")
    
    # Dangerous: deserialization  
    data = pickle.loads(serialized_data)
    
    # Dangerous: command injection
    os.system(f"echo {user_input}")
    
    return {"result": result, "data": data}

def admin_function():
    # Another hardcoded credential
    admin_key = "admin_secret_key_xyz789"
    return admin_key
''')
            
            # JavaScript file with XSS issues
            with open(os.path.join(temp_dir, "src", "frontend.js"), 'w') as f:
                f.write('''
function displayUserContent(userInput) {
    // XSS vulnerability
    document.getElementById('content').innerHTML = userInput;
    
    // Another XSS issue
    document.write('<div>' + userInput + '</div>');
    
    // Code injection
    setTimeout('executeCommand("' + userInput + '")', 1000);
}

// Hardcoded API endpoint
const API_ENDPOINT = "https://api.example.com/secret";
const API_TOKEN = "abc123def456ghi789jkl012";
''')
            
            # Configuration files
            with open(os.path.join(temp_dir, "config", "database.json"), 'w') as f:
                json.dump({
                    "host": "localhost",
                    "password": "database_secret_password",
                    "api_key": "config_api_key_secret"
                }, f, indent=2)
            
            # Requirements with vulnerable packages
            with open(os.path.join(temp_dir, "requirements.txt"), 'w') as f:
                f.write('''
# Vulnerable packages
urllib3==1.25.8
requests==2.25.1

# Safe packages  
numpy==1.21.0
pandas==1.3.0
''')
            
            # Dockerfile with security issues
            with open(os.path.join(temp_dir, "Dockerfile"), 'w') as f:
                f.write('''
FROM python:3.9
USER root
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "src/main.py"]
''')
            
            # Environment file
            with open(os.path.join(temp_dir, ".env"), 'w') as f:
                f.write('''
SECRET_KEY=env_file_secret_key_123
JWT_SECRET=jwt_signing_secret_key_456
POSTGRES_PASSWORD=postgres_secret_pass_789
''')
            
            # Run comprehensive scan
            scanner = SecurityScanner()
            result = scanner.scan_project(temp_dir)
            
            # Verify comprehensive results
            assert result.status == ScanStatus.COMPLETED
            assert len(result.vulnerabilities) >= 10  # Should find many issues
            
            # Categorize vulnerabilities
            secrets = [v for v in result.vulnerabilities if "secret" in v.tags]
            code_issues = [v for v in result.vulnerabilities if "dangerous-function" in v.tags or "code-analysis" in v.tags]
            dependencies = [v for v in result.vulnerabilities if v.affected_component]
            docker_issues = [v for v in result.vulnerabilities if "docker" in v.tags]
            
            # Should find issues in each category
            assert len(secrets) >= 4  # Multiple secrets in different files
            assert len(code_issues) >= 5  # Multiple code issues
            assert len(dependencies) >= 2  # Vulnerable urllib3 and requests
            assert len(docker_issues) >= 2  # Root user and ADD instruction
            
            # Check severity distribution
            severity_counts = result.get_severity_counts()
            assert severity_counts["high"] >= 3
            assert severity_counts["medium"] >= 2
            
            # Verify high-severity count
            high_severity_count = result.get_high_severity_count()
            assert high_severity_count >= 5
            
            # Test export functionality
            json_report = scanner.export_results(result.scan_id, "json")
            assert len(json_report) > 1000  # Should be substantial report
            
            sarif_report = scanner.export_results(result.scan_id, "sarif")
            sarif_data = json.loads(sarif_report)
            assert len(sarif_data["runs"][0]["results"]) >= 10
            
            # Test vulnerability summary
            summary = scanner.get_vulnerability_summary(days=1)
            assert summary["total_vulnerabilities"] >= 10
            assert summary["high_severity_count"] >= 5
            
            # Verify file paths are correct
            file_paths = [v.file_path for v in result.vulnerabilities if v.file_path]
            assert any("main.py" in path for path in file_paths)
            assert any("frontend.js" in path for path in file_paths)
            assert any("requirements.txt" in path for path in file_paths)
            assert any("Dockerfile" in path for path in file_paths)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_scanner_configuration(self):
        """Test scanner configuration options"""
        config = ScanConfiguration(
            enabled_scans={ScanType.SECRET_SCAN, ScanType.CODE_ANALYSIS},
            severity_threshold=SeverityLevel.MEDIUM,
            timeout_minutes=60
        )
        
        scanner = SecurityScanner(config)
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            with open(os.path.join(temp_dir, "app.py"), 'w') as f:
                f.write('SECRET = "test_secret_123456789_long_enough"\nresult = eval(user_input)')
            
            with open(os.path.join(temp_dir, "requirements.txt"), 'w') as f:
                f.write("urllib3==1.25.8")
            
            # Should only run enabled scan types
            result = scanner.scan_project(temp_dir)
            
            # Should find secret and code issues, but not dependency issues (not enabled)
            found_secrets = any("secret" in v.tags for v in result.vulnerabilities)
            found_code = any("dangerous-function" in v.tags for v in result.vulnerabilities)
            found_deps = any(v.affected_component for v in result.vulnerabilities)
            
            assert found_secrets
            assert found_code
            assert not found_deps  # Dependency scan not enabled
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run security scanner tests
    pytest.main([__file__, "-v", "-s"])