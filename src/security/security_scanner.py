"""
Automated Security Scanner
Sprint 10 Phase 3 - Issue 008: SEC-108
Provides automated security vulnerability scanning and analysis
"""

import os
import re
import json
import asyncio
import subprocess
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import tempfile
import threading
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of security scans"""
    DEPENDENCY_SCAN = "dependency_scan"
    SECRET_SCAN = "secret_scan"
    CODE_ANALYSIS = "code_analysis"
    CONTAINER_SCAN = "container_scan"
    NETWORK_SCAN = "network_scan"
    CONFIGURATION_SCAN = "configuration_scan"
    LICENSE_SCAN = "license_scan"


class SeverityLevel(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ScanStatus(Enum):
    """Scan execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Vulnerability:
    """Security vulnerability finding"""
    vuln_id: str
    title: str
    description: str
    severity: SeverityLevel
    cve_ids: List[str] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)
    affected_component: Optional[str] = None
    affected_version: Optional[str] = None
    fixed_version: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    evidence: Optional[str] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    first_detected: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ScanResult:
    """Results from a security scan"""
    scan_id: str
    scan_type: ScanType
    status: ScanStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    target: Optional[str] = None
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of vulnerabilities by severity"""
        counts = {severity.value: 0 for severity in SeverityLevel}
        for vuln in self.vulnerabilities:
            counts[vuln.severity.value] += 1
        return counts
    
    def get_high_severity_count(self) -> int:
        """Get count of high and critical vulnerabilities"""
        return sum(1 for v in self.vulnerabilities 
                  if v.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH])


@dataclass
class ScanConfiguration:
    """Configuration for security scans"""
    enabled_scans: Set[ScanType] = field(default_factory=lambda: set(ScanType))
    scan_schedule: Dict[ScanType, int] = field(default_factory=dict)  # hours
    severity_threshold: SeverityLevel = SeverityLevel.MEDIUM
    exclusions: Dict[str, List[str]] = field(default_factory=dict)
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)
    timeout_minutes: int = 30
    parallel_scans: bool = True
    notification_webhooks: List[str] = field(default_factory=list)


class DependencyScanner:
    """Scans dependencies for known vulnerabilities"""
    
    def __init__(self):
        """Initialize dependency scanner"""
        self.vulnerability_db = self._load_vulnerability_database()
    
    def scan_requirements_file(self, file_path: str) -> List[Vulnerability]:
        """Scan Python requirements file"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse requirement line
                package_info = self._parse_requirement_line(line)
                if package_info:
                    package_name, version = package_info
                    
                    # Check for known vulnerabilities
                    package_vulns = self._check_package_vulnerabilities(package_name, version)
                    for vuln in package_vulns:
                        vuln.file_path = file_path
                        vuln.line_number = line_num
                        vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.error(f"Failed to scan requirements file {file_path}: {e}")
        
        return vulnerabilities
    
    def scan_package_json(self, file_path: str) -> List[Vulnerability]:
        """Scan Node.js package.json file"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                package_data = json.load(f)
            
            # Check dependencies
            for dep_type in ['dependencies', 'devDependencies']:
                if dep_type in package_data:
                    for package_name, version in package_data[dep_type].items():
                        vulns = self._check_npm_vulnerabilities(package_name, version)
                        for vuln in vulns:
                            vuln.file_path = file_path
                            vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.error(f"Failed to scan package.json {file_path}: {e}")
        
        return vulnerabilities
    
    def _parse_requirement_line(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse Python requirement line"""
        # Handle various formats: package==1.0.0, package>=1.0.0, etc.
        patterns = [
            r'^([a-zA-Z0-9_-]+)==(.+)$',
            r'^([a-zA-Z0-9_-]+)>=(.+)$',
            r'^([a-zA-Z0-9_-]+)~=(.+)$',
            r'^([a-zA-Z0-9_-]+)>(.+)$',
            r'^([a-zA-Z0-9_-]+)$',  # No version specified
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                if len(match.groups()) == 2:
                    return match.group(1), match.group(2)
                else:
                    return match.group(1), "latest"
        
        return None
    
    def _check_package_vulnerabilities(self, package_name: str, version: str) -> List[Vulnerability]:
        """Check package for known vulnerabilities"""
        vulnerabilities = []
        
        # Check against known vulnerable packages (simplified implementation)
        vulnerable_packages = {
            "urllib3": {
                "1.25.8": [
                    Vulnerability(
                        vuln_id="CVE-2020-26137",
                        title="CRLF injection in urllib3",
                        description="urllib3 before 1.25.9 allows CRLF injection",
                        severity=SeverityLevel.MEDIUM,
                        cve_ids=["CVE-2020-26137"],
                        affected_component=package_name,
                        affected_version=version,
                        fixed_version="1.25.9",
                        remediation="Upgrade urllib3 to version 1.25.9 or later"
                    )
                ]
            },
            "requests": {
                "2.25.1": [
                    Vulnerability(
                        vuln_id="CVE-2023-32681", 
                        title="Requests Proxy-Authorization header leak",
                        description="Proxy-Authorization header sent to destination server",
                        severity=SeverityLevel.MEDIUM,
                        cve_ids=["CVE-2023-32681"],
                        affected_component=package_name,
                        affected_version=version,
                        fixed_version="2.31.0",
                        remediation="Upgrade requests to version 2.31.0 or later"
                    )
                ]
            }
        }
        
        if package_name in vulnerable_packages:
            if version in vulnerable_packages[package_name]:
                vulnerabilities.extend(vulnerable_packages[package_name][version])
        
        return vulnerabilities
    
    def _check_npm_vulnerabilities(self, package_name: str, version: str) -> List[Vulnerability]:
        """Check npm package for vulnerabilities"""
        # Simplified implementation - would integrate with npm audit API
        return []
    
    def _load_vulnerability_database(self) -> Dict[str, Any]:
        """Load vulnerability database"""
        # In production, this would load from CVE databases, GitHub Security Advisories, etc.
        return {}


class SecretScanner:
    """Scans code for exposed secrets"""
    
    def __init__(self):
        """Initialize secret scanner with detection patterns"""
        from .secrets_manager import SecretDetector
        self.detector = SecretDetector()
        
        # Additional patterns for secret scanning
        self.secret_patterns = {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'[0-9a-zA-Z/+]{40}',
            'github_token': r'ghp_[0-9a-zA-Z]{36}',
            'slack_token': r'xox[baprs]-[0-9a-zA-Z\-]{10,48}',
            'stripe_key': r'sk_live_[0-9a-zA-Z]{24}',
            'private_key': r'-----BEGIN [A-Z ]+ PRIVATE KEY-----',
            'generic_secret': r'["\']?[a-zA-Z0-9_]*(?:secret|key|token|pass)[a-zA-Z0-9_]*["\']?\s*[=:]\s*["\']?[a-zA-Z0-9_\-+/=]{16,}["\']?',
        }
    
    def scan_file(self, file_path: str) -> List[Vulnerability]:
        """Scan single file for secrets"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Use existing detector
            detected_secrets = self.detector.detect_secrets(content)
            
            for secret_type, secret_value, position in detected_secrets:
                # Calculate line number
                line_num = content[:position].count('\n') + 1
                
                vuln = Vulnerability(
                    vuln_id=f"SECRET_{hashlib.md5(f'{file_path}:{position}'.encode()).hexdigest()[:8].upper()}",
                    title=f"Exposed {secret_type.value.replace('_', ' ').title()}",
                    description=f"Potential {secret_type.value} found in source code",
                    severity=SeverityLevel.HIGH,
                    file_path=file_path,
                    line_number=line_num,
                    evidence=secret_value[:20] + "..." if len(secret_value) > 20 else secret_value,
                    remediation="Remove secret from code and use environment variables or secret management system",
                    tags=["secret", secret_type.value]
                )
                vulnerabilities.append(vuln)
            
            # Additional pattern-based scanning
            for secret_name, pattern in self.secret_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    vuln = Vulnerability(
                        vuln_id=f"SECRET_{hashlib.md5(f'{file_path}:{match.start()}'.encode()).hexdigest()[:8].upper()}",
                        title=f"Potential {secret_name.replace('_', ' ').title()}",
                        description=f"Pattern matching {secret_name} detected",
                        severity=SeverityLevel.MEDIUM,
                        file_path=file_path,
                        line_number=line_num,
                        evidence=match.group()[:30] + "..." if len(match.group()) > 30 else match.group(),
                        remediation="Verify if this is a real secret and remove from code if confirmed",
                        tags=["secret", secret_name],
                        confidence=0.8  # Pattern-based detection has lower confidence
                    )
                    vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.error(f"Failed to scan file {file_path} for secrets: {e}")
        
        return vulnerabilities
    
    def scan_directory(self, directory_path: str, 
                      file_extensions: Optional[Set[str]] = None) -> List[Vulnerability]:
        """Scan directory recursively for secrets"""
        if file_extensions is None:
            file_extensions = {'.py', '.js', '.json', '.yaml', '.yml', '.env', '.config', '.cfg'}
        
        vulnerabilities = []
        
        for root, dirs, files in os.walk(directory_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in file_extensions or file in {'.env', '.gitignore', 'Dockerfile'}:
                    file_vulns = self.scan_file(file_path)
                    vulnerabilities.extend(file_vulns)
        
        return vulnerabilities


class CodeAnalyzer:
    """Performs static code analysis for security issues"""
    
    def __init__(self):
        """Initialize code analyzer"""
        self.security_patterns = self._load_security_patterns()
    
    def scan_python_file(self, file_path: str) -> List[Vulnerability]:
        """Scan Python file for security issues"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for dangerous function usage
            dangerous_functions = [
                ('eval(', 'Code injection risk: eval() executes arbitrary code'),
                ('exec(', 'Code injection risk: exec() executes arbitrary code'),
                ('subprocess.call(', 'Command injection risk: validate input to subprocess.call'),
                ('os.system(', 'Command injection risk: os.system() executes shell commands'),
                ('pickle.loads(', 'Deserialization risk: pickle.loads() can execute arbitrary code'),
                ('yaml.load(', 'Deserialization risk: use yaml.safe_load() instead'),
            ]
            
            for pattern, description in dangerous_functions:
                for match in re.finditer(re.escape(pattern), content):
                    line_num = content[:match.start()].count('\n') + 1
                    
                    vuln = Vulnerability(
                        vuln_id=f"CODE_{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8].upper()}",
                        title=f"Dangerous Function Usage: {pattern.rstrip('(')}",
                        description=description,
                        severity=SeverityLevel.HIGH,
                        cwe_ids=["CWE-94"],  # Code Injection
                        file_path=file_path,
                        line_number=line_num,
                        remediation="Use safer alternatives and validate all input",
                        tags=["code-analysis", "dangerous-function"]
                    )
                    vulnerabilities.append(vuln)
            
            # Check for hardcoded credentials patterns
            cred_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password'),
                (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key'),
                (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret'),
            ]
            
            for pattern, description in cred_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    
                    vuln = Vulnerability(
                        vuln_id=f"CRED_{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8].upper()}",
                        title="Hardcoded Credentials",
                        description=description,
                        severity=SeverityLevel.MEDIUM,
                        cwe_ids=["CWE-798"],  # Use of Hard-coded Credentials
                        file_path=file_path,
                        line_number=line_num,
                        remediation="Use environment variables or secure configuration",
                        tags=["code-analysis", "hardcoded-credentials"]
                    )
                    vulnerabilities.append(vuln)
            
        except Exception as e:
            logger.error(f"Failed to analyze Python file {file_path}: {e}")
        
        return vulnerabilities
    
    def scan_javascript_file(self, file_path: str) -> List[Vulnerability]:
        """Scan JavaScript file for security issues"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for dangerous JavaScript patterns
            js_patterns = [
                (r'eval\s*\(', 'Code injection risk: eval() executes arbitrary JavaScript'),
                (r'innerHTML\s*=', 'XSS risk: innerHTML can execute scripts'),
                (r'document\.write\s*\(', 'XSS risk: document.write can introduce XSS'),
                (r'setTimeout\s*\(\s*["\']', 'Code injection risk: setTimeout with string parameter'),
            ]
            
            for pattern, description in js_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    
                    vuln = Vulnerability(
                        vuln_id=f"JS_{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8].upper()}",
                        title="JavaScript Security Issue",
                        description=description,
                        severity=SeverityLevel.MEDIUM,
                        file_path=file_path,
                        line_number=line_num,
                        remediation="Use safer alternatives and sanitize input",
                        tags=["code-analysis", "javascript"]
                    )
                    vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.error(f"Failed to analyze JavaScript file {file_path}: {e}")
        
        return vulnerabilities
    
    def _load_security_patterns(self) -> Dict[str, Any]:
        """Load security analysis patterns"""
        # In production, load from SAST rule databases
        return {}


class ConfigurationScanner:
    """Scans configuration files for security issues"""
    
    def scan_docker_file(self, file_path: str) -> List[Vulnerability]:
        """Scan Dockerfile for security issues"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Check for running as root
                if line.startswith('USER root'):
                    vuln = Vulnerability(
                        vuln_id=f"DOCKER_ROOT_{line_num}",
                        title="Running as Root User",
                        description="Container runs as root user",
                        severity=SeverityLevel.HIGH,
                        file_path=file_path,
                        line_number=line_num,
                        remediation="Create and use a non-root user",
                        tags=["docker", "privilege-escalation"]
                    )
                    vulnerabilities.append(vuln)
                
                # Check for ADD instruction (prefer COPY)
                if line.startswith('ADD '):
                    vuln = Vulnerability(
                        vuln_id=f"DOCKER_ADD_{line_num}",
                        title="Use of ADD Instruction",
                        description="ADD instruction has implicit behavior, prefer COPY",
                        severity=SeverityLevel.LOW,
                        file_path=file_path,
                        line_number=line_num,
                        remediation="Use COPY instead of ADD unless you need ADD's extra features",
                        tags=["docker", "best-practices"]
                    )
                    vulnerabilities.append(vuln)
        
        except Exception as e:
            logger.error(f"Failed to scan Dockerfile {file_path}: {e}")
        
        return vulnerabilities
    
    def scan_nginx_config(self, file_path: str) -> List[Vulnerability]:
        """Scan Nginx configuration for security issues"""
        vulnerabilities = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            logger.debug(f"Scanning nginx config file {file_path}, content length: {len(content)}")
            
            # Check for missing security headers
            security_headers = [
                'X-Frame-Options',
                'X-Content-Type-Options',
                'X-XSS-Protection',
                'Strict-Transport-Security'
            ]
            
            for header in security_headers:
                # Look for actual header directive, not just header name in comments
                header_directive = f"add_header {header}"
                if header_directive.lower() not in content.lower():
                    vuln = Vulnerability(
                        vuln_id=f"NGINX_{header.replace('-', '_').upper()}",
                        title=f"Missing Security Header: {header}",
                        description=f"Nginx configuration missing {header} security header",
                        severity=SeverityLevel.MEDIUM,
                        file_path=file_path,
                        remediation=f"Add 'add_header {header}' directive",
                        tags=["nginx", "security-headers"]
                    )
                    vulnerabilities.append(vuln)
                    
            logger.debug(f"Found {len(vulnerabilities)} nginx vulnerabilities")
        
        except Exception as e:
            logger.error(f"Failed to scan Nginx config {file_path}: {e}")
        
        return vulnerabilities


class SecurityScanner:
    """Main security scanner orchestrator"""
    
    def __init__(self, config: Optional[ScanConfiguration] = None):
        """Initialize security scanner"""
        self.config = config or ScanConfiguration()
        self.dependency_scanner = DependencyScanner()
        self.secret_scanner = SecretScanner()
        self.code_analyzer = CodeAnalyzer()
        self.config_scanner = ConfigurationScanner()
        
        self.scan_history: List[ScanResult] = []
        self._lock = threading.RLock()
        self._running_scans: Dict[str, threading.Thread] = {}
    
    def scan_project(self, project_path: str, scan_types: Optional[Set[ScanType]] = None) -> ScanResult:
        """Scan entire project for security issues"""
        if scan_types is None:
            scan_types = self.config.enabled_scans
        
        import uuid
        scan_id = f"scan_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        result = ScanResult(
            scan_id=scan_id,
            scan_type=ScanType.CODE_ANALYSIS,  # Mixed scan type
            status=ScanStatus.RUNNING,
            start_time=datetime.utcnow(),
            target=project_path
        )
        
        try:
            all_vulnerabilities = []
            
            # Run different scan types
            for scan_type in scan_types:
                try:
                    scan_vulns = self._run_scan_type(scan_type, project_path)
                    all_vulnerabilities.extend(scan_vulns)
                except Exception as e:
                    logger.error(f"Failed to run {scan_type} scan: {e}")
            
            # Deduplicate vulnerabilities
            result.vulnerabilities = self._deduplicate_vulnerabilities(all_vulnerabilities)
            result.summary = result.get_severity_counts()
            result.status = ScanStatus.COMPLETED
            result.end_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Project scan failed: {e}")
            result.status = ScanStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.utcnow()
        
        with self._lock:
            self.scan_history.append(result)
        
        return result
    
    def scan_dependencies(self, project_path: str) -> List[Vulnerability]:
        """Scan project dependencies"""
        vulnerabilities = []
        
        # Scan Python requirements
        requirements_files = [
            'requirements.txt',
            'requirements-dev.txt',
            'pyproject.toml',
            'Pipfile'
        ]
        
        for req_file in requirements_files:
            req_path = os.path.join(project_path, req_file)
            if os.path.exists(req_path) and req_file == 'requirements.txt':
                vulns = self.dependency_scanner.scan_requirements_file(req_path)
                vulnerabilities.extend(vulns)
        
        # Scan Node.js dependencies
        package_json = os.path.join(project_path, 'package.json')
        if os.path.exists(package_json):
            vulns = self.dependency_scanner.scan_package_json(package_json)
            vulnerabilities.extend(vulns)
        
        return vulnerabilities
    
    def scan_secrets(self, project_path: str) -> List[Vulnerability]:
        """Scan for exposed secrets"""
        return self.secret_scanner.scan_directory(project_path)
    
    def scan_code_quality(self, project_path: str) -> List[Vulnerability]:
        """Perform static code analysis"""
        vulnerabilities = []
        
        for root, dirs, files in os.walk(project_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext == '.py':
                    vulns = self.code_analyzer.scan_python_file(file_path)
                    vulnerabilities.extend(vulns)
                elif file_ext in {'.js', '.jsx', '.ts', '.tsx'}:
                    vulns = self.code_analyzer.scan_javascript_file(file_path)
                    vulnerabilities.extend(vulns)
        
        return vulnerabilities
    
    def scan_configuration(self, project_path: str) -> List[Vulnerability]:
        """Scan configuration files"""
        vulnerabilities = []
        
        # Scan Dockerfile
        dockerfile_paths = [
            os.path.join(project_path, 'Dockerfile'),
            os.path.join(project_path, 'docker', 'Dockerfile'),
        ]
        
        for dockerfile in dockerfile_paths:
            if os.path.exists(dockerfile):
                vulns = self.config_scanner.scan_docker_file(dockerfile)
                vulnerabilities.extend(vulns)
        
        # Scan nginx configs
        nginx_paths = [
            os.path.join(project_path, 'nginx.conf'),
            os.path.join(project_path, 'nginx', '*.conf'),
        ]
        
        for nginx_path in nginx_paths:
            if os.path.exists(nginx_path):
                vulns = self.config_scanner.scan_nginx_config(nginx_path)
                vulnerabilities.extend(vulns)
        
        return vulnerabilities
    
    def get_scan_results(self, scan_id: Optional[str] = None, limit: int = 10) -> List[ScanResult]:
        """Get scan results"""
        with self._lock:
            if scan_id:
                return [r for r in self.scan_history if r.scan_id == scan_id]
            else:
                return sorted(self.scan_history, key=lambda x: x.start_time, reverse=True)[:limit]
    
    def get_vulnerability_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get vulnerability summary for recent scans"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_scans = [s for s in self.scan_history if s.start_time > cutoff_date]
        
        total_vulns = 0
        severity_counts = {severity.value: 0 for severity in SeverityLevel}
        scan_type_counts = {scan_type.value: 0 for scan_type in ScanType}
        
        for scan in recent_scans:
            total_vulns += len(scan.vulnerabilities)
            scan_type_counts[scan.scan_type.value] += 1
            
            for vuln in scan.vulnerabilities:
                severity_counts[vuln.severity.value] += 1
        
        return {
            "total_scans": len(recent_scans),
            "total_vulnerabilities": total_vulns,
            "severity_breakdown": severity_counts,
            "scan_type_breakdown": scan_type_counts,
            "high_severity_count": severity_counts["critical"] + severity_counts["high"],
            "period_days": days
        }
    
    def _run_scan_type(self, scan_type: ScanType, project_path: str) -> List[Vulnerability]:
        """Run specific type of scan"""
        if scan_type == ScanType.DEPENDENCY_SCAN:
            return self.scan_dependencies(project_path)
        elif scan_type == ScanType.SECRET_SCAN:
            return self.scan_secrets(project_path)
        elif scan_type == ScanType.CODE_ANALYSIS:
            return self.scan_code_quality(project_path)
        elif scan_type == ScanType.CONFIGURATION_SCAN:
            return self.scan_configuration(project_path)
        else:
            logger.warning(f"Unsupported scan type: {scan_type}")
            return []
    
    def _deduplicate_vulnerabilities(self, vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        """Remove duplicate vulnerabilities"""
        seen = set()
        deduplicated = []
        
        for vuln in vulnerabilities:
            # Create unique key based on file path, line number, and vulnerability type
            key = (vuln.file_path, vuln.line_number, vuln.title)
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(vuln)
        
        return deduplicated
    
    def export_results(self, scan_id: str, format: str = "json") -> str:
        """Export scan results in specified format"""
        scan_results = self.get_scan_results(scan_id)
        
        if not scan_results:
            raise ValueError(f"Scan {scan_id} not found")
        
        result = scan_results[0]
        
        if format == "json":
            return self._export_json(result)
        elif format == "sarif":
            return self._export_sarif(result)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, result: ScanResult) -> str:
        """Export results as JSON"""
        data = {
            "scan_id": result.scan_id,
            "scan_type": result.scan_type.value,
            "status": result.status.value,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "target": result.target,
            "summary": result.summary,
            "vulnerabilities": []
        }
        
        for vuln in result.vulnerabilities:
            vuln_data = {
                "id": vuln.vuln_id,
                "title": vuln.title,
                "description": vuln.description,
                "severity": vuln.severity.value,
                "cve_ids": vuln.cve_ids,
                "cwe_ids": vuln.cwe_ids,
                "affected_component": vuln.affected_component,
                "file_path": vuln.file_path,
                "line_number": vuln.line_number,
                "evidence": vuln.evidence,
                "remediation": vuln.remediation,
                "confidence": vuln.confidence,
                "tags": vuln.tags
            }
            data["vulnerabilities"].append(vuln_data)
        
        return json.dumps(data, indent=2, default=str)
    
    def _export_sarif(self, result: ScanResult) -> str:
        """Export results in SARIF format"""
        # SARIF (Static Analysis Results Interchange Format)
        sarif = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "SecurityScanner",
                        "version": "1.0.0"
                    }
                },
                "results": []
            }]
        }
        
        for vuln in result.vulnerabilities:
            sarif_result = {
                "ruleId": vuln.vuln_id,
                "message": {
                    "text": vuln.description
                },
                "level": self._severity_to_sarif_level(vuln.severity),
                "locations": []
            }
            
            if vuln.file_path:
                location = {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": vuln.file_path
                        }
                    }
                }
                
                if vuln.line_number:
                    location["physicalLocation"]["region"] = {
                        "startLine": vuln.line_number
                    }
                
                sarif_result["locations"].append(location)
            
            sarif["runs"][0]["results"].append(sarif_result)
        
        return json.dumps(sarif, indent=2)
    
    def _severity_to_sarif_level(self, severity: SeverityLevel) -> str:
        """Convert severity to SARIF level"""
        mapping = {
            SeverityLevel.CRITICAL: "error",
            SeverityLevel.HIGH: "error", 
            SeverityLevel.MEDIUM: "warning",
            SeverityLevel.LOW: "note",
            SeverityLevel.INFO: "note"
        }
        return mapping.get(severity, "note")


# Export main components
__all__ = [
    "SecurityScanner",
    "ScanType",
    "SeverityLevel", 
    "ScanStatus",
    "Vulnerability",
    "ScanResult",
    "ScanConfiguration",
    "DependencyScanner",
    "SecretScanner",
    "CodeAnalyzer",
    "ConfigurationScanner",
]