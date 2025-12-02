"""
Compliance Reporting System
Sprint 10 Phase 3 - Issue 009: SEC-109
Provides comprehensive security compliance reporting and audit capabilities
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import tempfile
import threading
import uuid
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Compliance frameworks"""
    SOC2 = "soc2"
    ISO_27001 = "iso27001" 
    ISO27001 = "iso27001"  # Backward compatibility alias
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST = "nist"
    OWASP = "owasp"
    CIS = "cis"
    CCPA = "ccpa"
    SOX = "sox"


class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ControlCategory(Enum):
    """Security control categories"""
    ACCESS_CONTROL = "access_control"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_PROTECTION = "data_protection"
    ENCRYPTION = "encryption"
    LOGGING_MONITORING = "logging_monitoring"
    NETWORK_SECURITY = "network_security"
    VULNERABILITY_MANAGEMENT = "vulnerability_management"
    INCIDENT_RESPONSE = "incident_response"
    BUSINESS_CONTINUITY = "business_continuity"
    RISK_MANAGEMENT = "risk_management"
    GOVERNANCE = "governance"


class EvidenceType(Enum):
    """Types of compliance evidence"""
    SYSTEM_CONFIG = "system_config"
    SECURITY_CONFIG = "security_config"
    ACCESS_CONTROL = "access_control"
    ENCRYPTION = "encryption"
    AUDIT_LOG = "audit_log"
    NETWORK_SECURITY = "network_security"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    BACKUP_RECOVERY = "backup_recovery"
    DOCUMENTATION = "documentation"
    POLICY = "policy"


# Alias for backward compatibility
ControlStatus = ComplianceStatus


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    framework: ComplianceFramework
    category: ControlCategory
    title: str
    description: str
    requirements: List[str] = field(default_factory=list)
    implementation_guidance: str = ""
    testing_procedures: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    automation_possible: bool = False
    risk_level: str = "medium"  # low, medium, high, critical


@dataclass
class ComplianceEvidence:
    """Evidence supporting compliance"""
    evidence_id: str
    control_id: str
    evidence_type: str  # document, configuration, log, screenshot, test_result
    title: str
    description: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    collected_at: datetime = field(default_factory=datetime.utcnow)
    collected_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAssessment:
    """Assessment of a compliance control"""
    assessment_id: str
    control_id: str
    status: ComplianceStatus
    score: Optional[float] = None  # 0-100
    findings: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessed_by: str = "automated"
    next_assessment_due: Optional[datetime] = None
    remediation_notes: str = ""
    risk_rating: str = "medium"
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    report_id: str
    framework: ComplianceFramework
    report_type: str  # full, summary, delta, executive
    generated_at: datetime
    reporting_period_start: datetime
    reporting_period_end: datetime
    assessments: List[ComplianceAssessment] = field(default_factory=list)
    overall_score: Optional[float] = None
    compliance_percentage: float = 0.0
    generated_by: str = "automated"
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    executive_summary: str = ""


class ComplianceControlLibrary:
    """Library of compliance controls for different frameworks"""
    
    def __init__(self):
        """Initialize compliance control library"""
        self.controls = self._load_control_definitions()
    
    def get_controls(self, framework: ComplianceFramework) -> List[ComplianceControl]:
        """Get all controls for a framework"""
        return [control for control in self.controls if control.framework == framework]
    
    def get_control(self, control_id: str) -> Optional[ComplianceControl]:
        """Get specific control by ID"""
        for control in self.controls:
            if control.control_id == control_id:
                return control
        return None
    
    def _load_control_definitions(self) -> List[ComplianceControl]:
        """Load compliance control definitions"""
        controls = []
        
        # SOC 2 Controls
        controls.extend(self._load_soc2_controls())
        
        # GDPR Controls
        controls.extend(self._load_gdpr_controls())
        
        # OWASP Controls
        controls.extend(self._load_owasp_controls())
        
        # PCI DSS Controls
        controls.extend(self._load_pci_dss_controls())
        
        # ISO 27001 Controls
        controls.extend(self._load_iso27001_controls())
        
        return controls
    
    def _load_soc2_controls(self) -> List[ComplianceControl]:
        """Load SOC 2 compliance controls"""
        return [
            ComplianceControl(
                control_id="CC6.1",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.ACCESS_CONTROL,
                title="Logical Access Controls",
                description="System access is restricted to authorized users",
                requirements=[
                    "User access provisioning and deprovisioning processes",
                    "Role-based access control implementation",
                    "Regular access reviews and certifications"
                ],
                implementation_guidance="Implement RBAC with regular access reviews",
                testing_procedures=[
                    "Review user access lists",
                    "Test access control enforcement",
                    "Verify segregation of duties"
                ],
                automation_possible=True,
                risk_level="high"
            ),
            ComplianceControl(
                control_id="CC6.7",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.DATA_PROTECTION,
                title="Data Transmission",
                description="Data is protected during transmission",
                requirements=[
                    "Encryption of data in transit",
                    "Secure communication protocols",
                    "Certificate management"
                ],
                implementation_guidance="Use TLS 1.3 for all data transmission",
                testing_procedures=[
                    "Verify TLS configuration",
                    "Test certificate validity",
                    "Review encryption standards"
                ],
                automation_possible=True,
                risk_level="high"
            ),
            ComplianceControl(
                control_id="CC7.2",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.LOGGING_MONITORING,
                title="System Monitoring",
                description="System activities are monitored and logged",
                requirements=[
                    "Comprehensive logging of system activities",
                    "Log monitoring and alerting",
                    "Log retention and protection"
                ],
                implementation_guidance="Implement centralized logging with SIEM",
                testing_procedures=[
                    "Review log completeness",
                    "Test monitoring alerts",
                    "Verify log retention"
                ],
                automation_possible=True,
                risk_level="medium"
            )
        ]
    
    def _load_gdpr_controls(self) -> List[ComplianceControl]:
        """Load GDPR compliance controls"""
        return [
            ComplianceControl(
                control_id="GDPR.25",
                framework=ComplianceFramework.GDPR,
                category=ControlCategory.DATA_PROTECTION,
                title="Data Protection by Design and Default",
                description="Privacy protection measures built into systems",
                requirements=[
                    "Privacy by design implementation",
                    "Data minimization principles",
                    "Purpose limitation compliance"
                ],
                implementation_guidance="Implement privacy controls at system design level",
                testing_procedures=[
                    "Review data collection practices",
                    "Verify data minimization",
                    "Test privacy controls"
                ],
                automation_possible=True,
                risk_level="high"
            ),
            ComplianceControl(
                control_id="GDPR.32",
                framework=ComplianceFramework.GDPR,
                category=ControlCategory.ENCRYPTION,
                title="Security of Processing",
                description="Appropriate technical and organizational measures",
                requirements=[
                    "Encryption of personal data",
                    "System security measures",
                    "Regular security testing"
                ],
                implementation_guidance="Implement encryption and security controls",
                testing_procedures=[
                    "Verify encryption implementation",
                    "Test security controls",
                    "Review access controls"
                ],
                automation_possible=True,
                risk_level="critical"
            )
        ]
    
    def _load_owasp_controls(self) -> List[ComplianceControl]:
        """Load OWASP Top 10 controls"""
        return [
            ComplianceControl(
                control_id="OWASP.A01",
                framework=ComplianceFramework.OWASP,
                category=ControlCategory.ACCESS_CONTROL,
                title="Broken Access Control",
                description="Prevent broken access control vulnerabilities",
                requirements=[
                    "Implement proper access controls",
                    "Enforce least privilege principle",
                    "Regular access control testing"
                ],
                implementation_guidance="Implement RBAC and regular testing",
                testing_procedures=[
                    "Test access control bypass",
                    "Review privilege escalation",
                    "Verify authorization checks"
                ],
                automation_possible=True,
                risk_level="critical"
            ),
            ComplianceControl(
                control_id="OWASP.A03",
                framework=ComplianceFramework.OWASP,
                category=ControlCategory.DATA_PROTECTION,
                title="Injection",
                description="Prevent injection vulnerabilities",
                requirements=[
                    "Input validation and sanitization",
                    "Parameterized queries",
                    "Output encoding"
                ],
                implementation_guidance="Use prepared statements and input validation",
                testing_procedures=[
                    "Test SQL injection",
                    "Test command injection",
                    "Review input validation"
                ],
                automation_possible=True,
                risk_level="critical"
            )
        ]
    
    def _load_pci_dss_controls(self) -> List[ComplianceControl]:
        """Load PCI DSS controls"""
        return [
            ComplianceControl(
                control_id="PCI.3.4",
                framework=ComplianceFramework.PCI_DSS,
                category=ControlCategory.ENCRYPTION,
                title="Cardholder Data Encryption",
                description="Render cardholder data unreadable",
                requirements=[
                    "Strong cryptography implementation",
                    "Key management procedures",
                    "Encryption at rest and in transit"
                ],
                implementation_guidance="Use AES-256 encryption for cardholder data",
                testing_procedures=[
                    "Verify encryption implementation",
                    "Test key management",
                    "Review cryptographic standards"
                ],
                automation_possible=True,
                risk_level="critical"
            ),
            ComplianceControl(
                control_id="PCI.8.2",
                framework=ComplianceFramework.PCI_DSS,
                category=ControlCategory.AUTHENTICATION,
                title="User Authentication",
                description="Strong authentication for all users",
                requirements=[
                    "Multi-factor authentication",
                    "Strong password policies",
                    "Account lockout mechanisms"
                ],
                implementation_guidance="Implement MFA and strong password policies",
                testing_procedures=[
                    "Test MFA implementation",
                    "Review password policies",
                    "Verify account lockout"
                ],
                automation_possible=True,
                risk_level="high"
            )
        ]
    
    def _load_iso27001_controls(self) -> List[ComplianceControl]:
        """Load ISO 27001 controls"""
        return [
            ComplianceControl(
                control_id="A.9.1.2",
                framework=ComplianceFramework.ISO_27001,
                category=ControlCategory.ACCESS_CONTROL,
                title="Access to Networks and Network Services",
                description="Users shall only be provided access to networks and network services",
                requirements=[
                    "Network access control policy",
                    "User access provisioning",
                    "Network segregation"
                ],
                implementation_guidance="Implement network access controls and segregation",
                testing_procedures=[
                    "Review network access policies",
                    "Test network segregation",
                    "Verify access controls"
                ],
                automation_possible=True,
                risk_level="high"
            ),
            ComplianceControl(
                control_id="A.10.1.1",
                framework=ComplianceFramework.ISO_27001,
                category=ControlCategory.ENCRYPTION,
                title="Cryptographic Policy",
                description="Policy on the use of cryptographic controls",
                requirements=[
                    "Cryptographic policy development",
                    "Key management procedures",
                    "Algorithm selection criteria"
                ],
                implementation_guidance="Develop comprehensive cryptographic policy",
                testing_procedures=[
                    "Review cryptographic policy",
                    "Test key management",
                    "Verify algorithm usage"
                ],
                automation_possible=False,
                risk_level="medium"
            )
        ]
    
    def search_controls(self, query: str) -> List[ComplianceControl]:
        """Search controls by keyword
        
        Args:
            query: Search query string
            
        Returns:
            List of controls matching the query
        """
        if not query:
            return []
        
        query_lower = query.lower()
        matching_controls = []
        
        for control in self.controls:
            # Search in title, description, and requirements
            searchable_text = (
                f"{control.title} {control.description} "
                f"{' '.join(control.requirements)} "
                f"{control.implementation_guidance}"
            ).lower()
            
            if query_lower in searchable_text:
                matching_controls.append(control)
        
        return matching_controls


class AutomatedComplianceAssessor:
    """Performs automated compliance assessments"""
    
    def __init__(self):
        """Initialize automated assessor"""
        self.control_library = ComplianceControlLibrary()
        self.assessors = {
            "access_control": self._assess_access_control,
            "encryption": self._assess_encryption,
            "logging": self._assess_logging,
            "authentication": self._assess_authentication,
            "network_security": self._assess_network_security
        }
    
    def assess_control(self, control: ComplianceControl, 
                      evidence: List[ComplianceEvidence] = None) -> ComplianceAssessment:
        """Assess a single compliance control"""
        
        assessment_id = f"assess_{control.control_id}_{int(datetime.utcnow().timestamp())}"
        
        # Get appropriate assessor
        assessor = self.assessors.get(control.category.value)
        
        if assessor and control.automation_possible:
            status, score, findings = assessor(control, evidence or [])
        else:
            # Manual assessment required
            status = ComplianceStatus.UNKNOWN
            score = None
            findings = ["Manual assessment required"]
        
        return ComplianceAssessment(
            assessment_id=assessment_id,
            control_id=control.control_id,
            status=status,
            score=score,
            findings=findings,
            evidence_ids=[e.evidence_id for e in (evidence or [])],
            assessed_at=datetime.utcnow(),
            assessed_by="automated",
            next_assessment_due=datetime.utcnow() + timedelta(days=90),
            risk_rating=control.risk_level
        )
    
    def _assess_access_control(self, control: ComplianceControl, 
                              evidence: List[ComplianceEvidence]) -> Tuple[ComplianceStatus, float, List[str]]:
        """Assess access control compliance"""
        findings = []
        score = 100.0
        
        # Check for RBAC implementation
        rbac_implemented = any("rbac" in e.title.lower() or "role" in e.title.lower() 
                              for e in evidence)
        if not rbac_implemented:
            findings.append("RBAC implementation not found in evidence")
            score -= 30
        
        # Check for access reviews
        access_reviews = any("review" in e.title.lower() or "audit" in e.title.lower()
                           for e in evidence)
        if not access_reviews:
            findings.append("Access review processes not documented")
            score -= 20
        
        # Check for user provisioning processes
        provisioning = any("provision" in e.title.lower() or "deprovisioning" in e.title.lower()
                          for e in evidence)
        if not provisioning:
            findings.append("User provisioning/deprovisioning processes not documented")
            score -= 25
        
        # Determine status based on score
        if score >= 90:
            status = ComplianceStatus.COMPLIANT
        elif score >= 70:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
            findings.append("Partial compliance - some controls missing")
        else:
            status = ComplianceStatus.NON_COMPLIANT
            findings.append("Non-compliant - major controls missing")
        
        return status, score, findings
    
    def _assess_encryption(self, control: ComplianceControl,
                          evidence: List[ComplianceEvidence]) -> Tuple[ComplianceStatus, float, List[str]]:
        """Assess encryption compliance"""
        findings = []
        score = 100.0
        
        # Check for encryption at rest
        encryption_rest = any("encryption" in e.title.lower() and "rest" in e.title.lower()
                             for e in evidence)
        if not encryption_rest:
            findings.append("Encryption at rest not documented")
            score -= 35
        
        # Check for encryption in transit
        encryption_transit = any("tls" in e.title.lower() or "ssl" in e.title.lower()
                                for e in evidence)
        if not encryption_transit:
            findings.append("Encryption in transit (TLS/SSL) not documented")
            score -= 35
        
        # Check for key management
        key_management = any("key" in e.title.lower() and "management" in e.title.lower()
                           for e in evidence)
        if not key_management:
            findings.append("Key management procedures not documented")
            score -= 30
        
        if score >= 90:
            status = ComplianceStatus.COMPLIANT
        elif score >= 70:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return status, score, findings
    
    def _assess_logging(self, control: ComplianceControl,
                       evidence: List[ComplianceEvidence]) -> Tuple[ComplianceStatus, float, List[str]]:
        """Assess logging and monitoring compliance"""
        findings = []
        score = 100.0
        
        # Check for comprehensive logging
        logging_config = any("log" in e.title.lower() for e in evidence)
        if not logging_config:
            findings.append("Logging configuration not documented")
            score -= 40
        
        # Check for monitoring and alerting
        monitoring = any("monitor" in e.title.lower() or "alert" in e.title.lower()
                        for e in evidence)
        if not monitoring:
            findings.append("Monitoring and alerting not documented")
            score -= 35
        
        # Check for log retention
        retention = any("retention" in e.title.lower() for e in evidence)
        if not retention:
            findings.append("Log retention policy not documented")
            score -= 25
        
        if score >= 85:
            status = ComplianceStatus.COMPLIANT
        elif score >= 65:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return status, score, findings
    
    def _assess_authentication(self, control: ComplianceControl,
                             evidence: List[ComplianceEvidence]) -> Tuple[ComplianceStatus, float, List[str]]:
        """Assess authentication compliance"""
        findings = []
        score = 100.0
        
        # Check for MFA
        mfa = any("mfa" in e.title.lower() or "multi-factor" in e.title.lower()
                 for e in evidence)
        if not mfa:
            findings.append("Multi-factor authentication not documented")
            score -= 40
        
        # Check for password policy
        password_policy = any("password" in e.title.lower() and "policy" in e.title.lower()
                            for e in evidence)
        if not password_policy:
            findings.append("Password policy not documented")
            score -= 30
        
        # Check for account lockout
        lockout = any("lockout" in e.title.lower() or "lock" in e.title.lower()
                     for e in evidence)
        if not lockout:
            findings.append("Account lockout mechanism not documented")
            score -= 30
        
        if score >= 85:
            status = ComplianceStatus.COMPLIANT
        elif score >= 65:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return status, score, findings
    
    def _assess_network_security(self, control: ComplianceControl,
                               evidence: List[ComplianceEvidence]) -> Tuple[ComplianceStatus, float, List[str]]:
        """Assess network security compliance"""
        findings = []
        score = 100.0
        
        # Check for firewall configuration
        firewall = any("firewall" in e.title.lower() or "waf" in e.title.lower()
                      for e in evidence)
        if not firewall:
            findings.append("Firewall/WAF configuration not documented")
            score -= 40
        
        # Check for network segmentation
        segmentation = any("segment" in e.title.lower() or "vlan" in e.title.lower()
                          for e in evidence)
        if not segmentation:
            findings.append("Network segmentation not documented")
            score -= 35
        
        # Check for intrusion detection
        ids = any("intrusion" in e.title.lower() or "ids" in e.title.lower()
                 for e in evidence)
        if not ids:
            findings.append("Intrusion detection system not documented")
            score -= 25
        
        if score >= 80:
            status = ComplianceStatus.COMPLIANT
        elif score >= 60:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        return status, score, findings


class EvidenceCollector:
    """Collects compliance evidence from various sources"""
    
    def __init__(self, evidence_dir: Optional[str] = None):
        """Initialize evidence collector
        
        Args:
            evidence_dir: Optional directory path for storing evidence files
        """
        self.evidence_store = []
        self.evidence_dir = evidence_dir or tempfile.gettempdir()
        self._lock = threading.RLock()
    
    def collect_configuration_evidence(self, config_path: str, 
                                     control_id: str, title: str) -> ComplianceEvidence:
        """Collect configuration file as evidence"""
        evidence_id = f"config_{hashlib.md5(f'{control_id}_{config_path}'.encode()).hexdigest()[:8]}"
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            evidence = ComplianceEvidence(
                evidence_id=evidence_id,
                control_id=control_id,
                evidence_type="configuration",
                title=title,
                description=f"Configuration file: {config_path}",
                file_path=config_path,
                content=content,
                metadata={"file_size": len(content), "file_type": "config"}
            )
            
            with self._lock:
                self.evidence_store.append(evidence)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Failed to collect configuration evidence from {config_path}: {e}")
            raise
    
    def collect_log_evidence(self, log_data: str, control_id: str, 
                           title: str, description: str) -> ComplianceEvidence:
        """Collect log data as evidence"""
        evidence_id = f"log_{hashlib.md5(f'{control_id}_{title}'.encode()).hexdigest()[:8]}"
        
        evidence = ComplianceEvidence(
            evidence_id=evidence_id,
            control_id=control_id,
            evidence_type="log",
            title=title,
            description=description,
            content=log_data,
            metadata={"log_lines": len(log_data.split('\n'))}
        )
        
        with self._lock:
            self.evidence_store.append(evidence)
        
        return evidence
    
    def collect_test_result_evidence(self, test_results: Dict[str, Any],
                                   control_id: str, title: str) -> ComplianceEvidence:
        """Collect test results as evidence"""
        evidence_id = f"test_{hashlib.md5(f'{control_id}_{title}'.encode()).hexdigest()[:8]}"
        
        evidence = ComplianceEvidence(
            evidence_id=evidence_id,
            control_id=control_id,
            evidence_type="test_result",
            title=title,
            description="Automated test results",
            content=json.dumps(test_results, indent=2),
            metadata={"test_count": len(test_results), "test_type": "automated"}
        )
        
        with self._lock:
            self.evidence_store.append(evidence)
        
        return evidence
    
    def collect_document_evidence(self, document_path: str, control_id: str,
                                title: str, description: str) -> ComplianceEvidence:
        """Collect document as evidence"""
        evidence_id = f"doc_{hashlib.md5(f'{control_id}_{document_path}'.encode()).hexdigest()[:8]}"
        
        try:
            with open(document_path, 'r') as f:
                content = f.read()
            
            evidence = ComplianceEvidence(
                evidence_id=evidence_id,
                control_id=control_id,
                evidence_type="document",
                title=title,
                description=description,
                file_path=document_path,
                content=content,
                metadata={"document_type": "policy"}
            )
            
            with self._lock:
                self.evidence_store.append(evidence)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Failed to collect document evidence from {document_path}: {e}")
            raise
    
    def get_evidence_for_control(self, control_id: str) -> List[ComplianceEvidence]:
        """Get all evidence for a specific control"""
        with self._lock:
            return [e for e in self.evidence_store if e.control_id == control_id]
    
    def get_all_evidence(self) -> List[ComplianceEvidence]:
        """Get all collected evidence"""
        with self._lock:
            return self.evidence_store.copy()


class ComplianceReporter:
    """Main compliance reporting system"""
    
    def __init__(self, output_dir: str = "./compliance_reports"):
        """Initialize compliance reporter"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.control_library = ComplianceControlLibrary()
        self.assessor = AutomatedComplianceAssessor()
        self.evidence_collector = EvidenceCollector()
        
        self.reports = []
        self._lock = threading.RLock()
    
    def generate_compliance_report(self, framework: ComplianceFramework,
                                 report_type: str = "full",
                                 period_days: int = 30) -> ComplianceReport:
        """Generate comprehensive compliance report"""
        
        report_id = f"{framework.value}_{report_type}_{int(datetime.utcnow().timestamp())}"
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        # Get controls for framework
        controls = self.control_library.get_controls(framework)
        
        # Perform assessments
        assessments = []
        for control in controls:
            evidence = self.evidence_collector.get_evidence_for_control(control.control_id)
            assessment = self.assessor.assess_control(control, evidence)
            assessments.append(assessment)
        
        # Calculate overall compliance
        compliant_count = sum(1 for a in assessments if a.status == ComplianceStatus.COMPLIANT)
        total_count = len(assessments)
        compliance_percentage = (compliant_count / total_count * 100) if total_count > 0 else 0
        
        # Calculate overall score
        scores = [a.score for a in assessments if a.score is not None]
        overall_score = sum(scores) / len(scores) if scores else None
        
        # Generate summary
        status_counts = {}
        for status in ComplianceStatus:
            status_counts[status.value] = sum(1 for a in assessments if a.status == status)
        
        summary = {
            "total_controls": total_count,
            "compliant_controls": compliant_count,
            "compliance_percentage": compliance_percentage,
            "status_breakdown": status_counts,
            "high_risk_controls": sum(1 for a in assessments if a.risk_rating == "high"),
            "critical_risk_controls": sum(1 for a in assessments if a.risk_rating == "critical")
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(assessments, controls)
        
        report = ComplianceReport(
            report_id=report_id,
            framework=framework,
            report_type=report_type,
            generated_at=datetime.utcnow(),
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            assessments=assessments,
            overall_score=overall_score,
            compliance_percentage=compliance_percentage,
            summary=summary,
            recommendations=recommendations
        )
        
        with self._lock:
            self.reports.append(report)
        
        return report
    
    def export_report(self, report: ComplianceReport, format: str = "json") -> str:
        """Export compliance report in specified format"""
        if format == "json":
            return self._export_json_report(report)
        elif format == "csv":
            return self._export_csv_report(report)
        elif format == "html":
            return self._export_html_report(report)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def save_report(self, report: ComplianceReport, format: str = "json") -> str:
        """Save compliance report to file"""
        filename = f"{report.report_id}_{report.framework.value}_{format}"
        
        if format == "json":
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                f.write(self._export_json_report(report))
        
        elif format == "csv":
            filepath = self.output_dir / f"{filename}.csv"
            with open(filepath, 'w') as f:
                f.write(self._export_csv_report(report))
        
        elif format == "html":
            filepath = self.output_dir / f"{filename}.html"
            with open(filepath, 'w') as f:
                f.write(self._export_html_report(report))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(filepath)
    
    def _generate_recommendations(self, assessments: List[ComplianceAssessment],
                                controls: List[ComplianceControl]) -> List[str]:
        """Generate remediation recommendations"""
        recommendations = []
        
        # High priority recommendations for non-compliant critical controls
        critical_non_compliant = [
            a for a in assessments 
            if a.status == ComplianceStatus.NON_COMPLIANT and a.risk_rating == "critical"
        ]
        
        if critical_non_compliant:
            recommendations.append(
                f"URGENT: {len(critical_non_compliant)} critical controls are non-compliant. "
                "Immediate remediation required."
            )
        
        # Recommendations for partially compliant controls
        partial_compliant = [a for a in assessments if a.status == ComplianceStatus.PARTIALLY_COMPLIANT]
        if partial_compliant:
            recommendations.append(
                f"ATTENTION: {len(partial_compliant)} controls are partially compliant. "
                "Review findings and implement missing controls."
            )
        
        # Category-specific recommendations
        category_issues = {}
        for assessment in assessments:
            if assessment.status != ComplianceStatus.COMPLIANT:
                control = next((c for c in controls if c.control_id == assessment.control_id), None)
                if control:
                    category = control.category.value
                    if category not in category_issues:
                        category_issues[category] = 0
                    category_issues[category] += 1
        
        for category, count in category_issues.items():
            if count >= 2:
                recommendations.append(
                    f"Focus on {category.replace('_', ' ')} controls - {count} issues identified"
                )
        
        # General recommendations
        overall_compliance = sum(1 for a in assessments if a.status == ComplianceStatus.COMPLIANT)
        total_controls = len(assessments)
        compliance_rate = (overall_compliance / total_controls * 100) if total_controls > 0 else 0
        
        if compliance_rate < 70:
            recommendations.append(
                "Overall compliance is below 70%. Consider comprehensive security program review."
            )
        elif compliance_rate < 90:
            recommendations.append(
                "Good compliance progress. Focus on remaining non-compliant controls."
            )
        else:
            recommendations.append(
                "Excellent compliance posture. Maintain current controls and monitor for changes."
            )
        
        return recommendations
    
    def _export_json_report(self, report: ComplianceReport) -> str:
        """Export report as JSON"""
        data = {
            "report_metadata": {
                "report_id": report.report_id,
                "framework": report.framework.value,
                "report_type": report.report_type,
                "generated_at": report.generated_at.isoformat(),
                "reporting_period": {
                    "start": report.reporting_period_start.isoformat(),
                    "end": report.reporting_period_end.isoformat()
                }
            },
            "compliance_summary": {
                "overall_score": report.overall_score,
                "compliance_percentage": report.compliance_percentage,
                "summary": report.summary
            },
            "assessments": [],
            "recommendations": report.recommendations
        }
        
        for assessment in report.assessments:
            assessment_data = {
                "control_id": assessment.control_id,
                "status": assessment.status.value,
                "score": assessment.score,
                "risk_rating": assessment.risk_rating,
                "findings": assessment.findings,
                "assessed_at": assessment.assessed_at.isoformat(),
                "next_assessment_due": assessment.next_assessment_due.isoformat() if assessment.next_assessment_due else None
            }
            data["assessments"].append(assessment_data)
        
        return json.dumps(data, indent=2, default=str)
    
    def _export_csv_report(self, report: ComplianceReport) -> str:
        """Export report as CSV"""
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Control ID", "Status", "Score", "Risk Rating", 
            "Findings Count", "Assessment Date", "Next Due"
        ])
        
        # Assessment rows
        for assessment in report.assessments:
            writer.writerow([
                assessment.control_id,
                assessment.status.value,
                assessment.score,
                assessment.risk_rating,
                len(assessment.findings),
                assessment.assessed_at.isoformat(),
                assessment.next_assessment_due.isoformat() if assessment.next_assessment_due else ""
            ])
        
        return output.getvalue()
    
    def _export_html_report(self, report: ComplianceReport) -> str:
        """Export report as HTML"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Compliance Report - {report.framework.value.upper()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .compliance-score {{ font-size: 2em; color: #28a745; font-weight: bold; }}
        .controls-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .controls-table th, .controls-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .controls-table th {{ background-color: #f2f2f2; }}
        .status-compliant {{ color: #28a745; font-weight: bold; }}
        .status-non-compliant {{ color: #dc3545; font-weight: bold; }}
        .status-partial {{ color: #ffc107; font-weight: bold; }}
        .recommendations {{ background: #e9ecef; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Compliance Report</h1>
        <p><strong>Framework:</strong> {report.framework.value.upper()}</p>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Period:</strong> {report.reporting_period_start.strftime('%Y-%m-%d')} to {report.reporting_period_end.strftime('%Y-%m-%d')}</p>
    </div>
    
    <div class="summary">
        <h2>Compliance Summary</h2>
        <div class="compliance-score">{report.compliance_percentage:.1f}% Compliant</div>
        <p><strong>Overall Score:</strong> {report.overall_score:.1f if report.overall_score else 'N/A'}</p>
        <p><strong>Total Controls:</strong> {report.summary['total_controls']}</p>
        <p><strong>Compliant Controls:</strong> {report.summary['compliant_controls']}</p>
    </div>
    
    <h2>Control Assessments</h2>
    <table class="controls-table">
        <thead>
            <tr>
                <th>Control ID</th>
                <th>Status</th>
                <th>Score</th>
                <th>Risk Rating</th>
                <th>Findings</th>
                <th>Next Assessment</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for assessment in report.assessments:
            status_class = {
                ComplianceStatus.COMPLIANT: "status-compliant",
                ComplianceStatus.NON_COMPLIANT: "status-non-compliant", 
                ComplianceStatus.PARTIALLY_COMPLIANT: "status-partial"
            }.get(assessment.status, "")
            
            html += f"""
            <tr>
                <td>{assessment.control_id}</td>
                <td class="{status_class}">{assessment.status.value.replace('_', ' ').title()}</td>
                <td>{assessment.score:.1f if assessment.score else 'N/A'}</td>
                <td>{assessment.risk_rating.title()}</td>
                <td>{len(assessment.findings)}</td>
                <td>{assessment.next_assessment_due.strftime('%Y-%m-%d') if assessment.next_assessment_due else 'N/A'}</td>
            </tr>
            """
        
        html += f"""
        </tbody>
    </table>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
"""
        
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
        </ul>
    </div>
    
</body>
</html>
"""
        
        return html
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard"""
        if not self.reports:
            return {"message": "No compliance reports available"}
        
        latest_reports = {}
        for report in self.reports:
            framework = report.framework.value
            if framework not in latest_reports or report.generated_at > latest_reports[framework].generated_at:
                latest_reports[framework] = report
        
        dashboard_data = {
            "last_updated": datetime.utcnow().isoformat(),
            "frameworks": {}
        }
        
        for framework, report in latest_reports.items():
            dashboard_data["frameworks"][framework] = {
                "compliance_percentage": report.compliance_percentage,
                "overall_score": report.overall_score,
                "total_controls": report.summary["total_controls"],
                "compliant_controls": report.summary["compliant_controls"],
                "last_assessment": report.generated_at.isoformat(),
                "status_breakdown": report.summary["status_breakdown"],
                "high_priority_issues": len([a for a in report.assessments 
                                           if a.status == ComplianceStatus.NON_COMPLIANT and a.risk_rating in ["high", "critical"]])
            }
        
        return dashboard_data
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate dashboard data for compliance reporting
        
        Returns:
            Dictionary containing dashboard data with overall_stats and frameworks
        """
        base_data = self.get_compliance_dashboard_data()
        
        if "message" in base_data:
            # No reports available
            return {
                "overall_stats": {
                    "total_frameworks": 0,
                    "avg_compliance": 0.0,
                    "total_controls": 0,
                    "last_updated": datetime.utcnow().isoformat()
                },
                "frameworks": {}
            }
        
        # Calculate overall statistics
        frameworks = base_data.get("frameworks", {})
        total_frameworks = len(frameworks)
        
        if total_frameworks > 0:
            avg_compliance = sum(f["compliance_percentage"] for f in frameworks.values()) / total_frameworks
            total_controls = sum(f["total_controls"] for f in frameworks.values())
        else:
            avg_compliance = 0.0
            total_controls = 0
        
        return {
            "overall_stats": {
                "total_frameworks": total_frameworks,
                "avg_compliance": avg_compliance,
                "total_controls": total_controls,
                "last_updated": base_data.get("last_updated", datetime.utcnow().isoformat())
            },
            "frameworks": frameworks
        }


# Export main components
__all__ = [
    "ComplianceReporter",
    "ComplianceFramework",
    "ComplianceStatus",
    "ControlStatus",  # Alias for ComplianceStatus
    "ControlCategory",
    "EvidenceType",
    "ComplianceControl",
    "ComplianceEvidence",
    "ComplianceAssessment",
    "ComplianceReport",
    "ComplianceControlLibrary",
    "AutomatedComplianceAssessor",
    "EvidenceCollector",
]
