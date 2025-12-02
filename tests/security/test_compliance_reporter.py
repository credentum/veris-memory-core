"""
Compliance Reporting Tests  
Sprint 10 Phase 3 - Issue 009: SEC-109
Tests the enterprise compliance reporting system
"""

import pytest
import os
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.security.compliance_reporter import (
    ComplianceReporter,
    ComplianceFramework,
    ComplianceControlLibrary,
    AutomatedComplianceAssessor,
    EvidenceCollector,
    ComplianceControl,
    ControlStatus,
    ComplianceAssessment,
    ComplianceReport,
    ComplianceEvidence,
    EvidenceType
)


class TestComplianceControlLibrary:
    """Test ID: SEC-109-A - Compliance Control Library"""
    
    def test_soc2_controls_loaded(self):
        """Test that SOC2 controls are properly loaded"""
        library = ComplianceControlLibrary()
        
        soc2_controls = library.get_controls(ComplianceFramework.SOC2)
        
        assert len(soc2_controls) > 0
        
        # Check for key SOC2 controls
        control_ids = [c.control_id for c in soc2_controls]
        assert "CC6.1" in control_ids  # Logical access controls
        assert "CC6.7" in control_ids  # Data transmission
        assert "CC7.2" in control_ids  # System monitoring
        
        # Verify control structure
        cc61_control = next(c for c in soc2_controls if c.control_id == "CC6.1")
        assert cc61_control.title
        assert cc61_control.description
        assert cc61_control.requirements
        assert len(cc61_control.requirements) > 0
    
    def test_gdpr_controls_loaded(self):
        """Test that GDPR controls are properly loaded"""
        library = ComplianceControlLibrary()
        
        gdpr_controls = library.get_controls(ComplianceFramework.GDPR)
        
        assert len(gdpr_controls) > 0
        
        # Check for key GDPR articles
        control_ids = [c.control_id for c in gdpr_controls]
        assert "Art25" in control_ids  # Data protection by design
        assert "Art32" in control_ids  # Security of processing
        assert "Art35" in control_ids  # Data protection impact assessment
        
        # Verify control structure
        art32_control = next(c for c in gdpr_controls if c.control_id == "Art32")
        assert art32_control.title
        assert "security" in art32_control.description.lower()
        assert len(art32_control.requirements) > 0
    
    def test_owasp_controls_loaded(self):
        """Test that OWASP controls are properly loaded"""
        library = ComplianceControlLibrary()
        
        owasp_controls = library.get_controls(ComplianceFramework.OWASP)
        
        assert len(owasp_controls) > 0
        
        # Check for OWASP Top 10 categories
        control_ids = [c.control_id for c in owasp_controls]
        assert "A01" in control_ids  # Broken access control
        assert "A02" in control_ids  # Cryptographic failures
        assert "A03" in control_ids  # Injection
        
        # Verify control structure
        a01_control = next(c for c in owasp_controls if c.control_id == "A01")
        assert "access control" in a01_control.title.lower()
        assert len(a01_control.requirements) > 0
    
    def test_pci_dss_controls_loaded(self):
        """Test that PCI DSS controls are properly loaded"""
        library = ComplianceControlLibrary()
        
        pci_controls = library.get_controls(ComplianceFramework.PCI_DSS)
        
        assert len(pci_controls) > 0
        
        # Check for key PCI DSS requirements
        control_ids = [c.control_id for c in pci_controls]
        assert "2.1" in control_ids  # Change default passwords
        assert "3.4" in control_ids  # Encrypt cardholder data
        assert "6.5" in control_ids  # Secure coding practices
        
        # Verify control structure
        req34_control = next(c for c in pci_controls if c.control_id == "3.4")
        assert "encrypt" in req34_control.title.lower()
        assert len(req34_control.requirements) > 0
    
    def test_iso27001_controls_loaded(self):
        """Test that ISO 27001 controls are properly loaded"""
        library = ComplianceControlLibrary()
        
        iso_controls = library.get_controls(ComplianceFramework.ISO27001)
        
        assert len(iso_controls) > 0
        
        # Check for key ISO 27001 controls
        control_ids = [c.control_id for c in iso_controls]
        assert "A.9.1.1" in control_ids  # Access control policy
        assert "A.10.1.1" in control_ids  # Cryptographic controls
        assert "A.12.2.1" in control_ids  # Malware protection
        
        # Verify control structure
        a911_control = next(c for c in iso_controls if c.control_id == "A.9.1.1")
        assert "access control" in a911_control.title.lower()
        assert len(a911_control.requirements) > 0
    
    def test_control_search_functionality(self):
        """Test control search and filtering"""
        library = ComplianceControlLibrary()
        
        # Search by keyword
        access_controls = library.search_controls("access control")
        assert len(access_controls) > 0
        
        # All results should contain "access" or "control"
        for control in access_controls:
            text = f"{control.title} {control.description}".lower()
            assert "access" in text or "control" in text
        
        # Search by encryption
        encryption_controls = library.search_controls("encryption")
        assert len(encryption_controls) > 0
        
        for control in encryption_controls:
            text = f"{control.title} {control.description}".lower()
            assert "encrypt" in text or "cryptograph" in text
    
    def test_control_categorization(self):
        """Test that controls are properly categorized"""
        library = ComplianceControlLibrary()
        
        # Get all controls
        all_controls = []
        for framework in ComplianceFramework:
            all_controls.extend(library.get_controls(framework))
        
        assert len(all_controls) > 20  # Should have substantial control library
        
        # Check that controls have required fields
        for control in all_controls:
            assert control.control_id
            assert control.title
            assert control.description
            assert control.framework
            assert isinstance(control.requirements, list)
            assert len(control.requirements) > 0


class TestEvidenceCollector:
    """Test ID: SEC-109-B - Evidence Collection"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = EvidenceCollector(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_collect_system_evidence(self):
        """Test collection of system configuration evidence"""
        self.setUp()
        
        try:
            evidence_list = self.collector.collect_system_evidence()
            
            assert len(evidence_list) > 0
            
            # Should have various types of system evidence
            evidence_types = [e.evidence_type for e in evidence_list]
            assert EvidenceType.SYSTEM_CONFIG in evidence_types
            assert EvidenceType.SECURITY_CONFIG in evidence_types
            
            # Check evidence structure
            for evidence in evidence_list:
                assert evidence.evidence_id
                assert evidence.evidence_type
                assert evidence.title
                assert evidence.description
                assert evidence.collected_at
                assert evidence.evidence_data
            
        finally:
            self.tearDown()
    
    def test_collect_access_control_evidence(self):
        """Test collection of access control evidence"""
        self.setUp()
        
        try:
            evidence_list = self.collector.collect_access_control_evidence()
            
            assert len(evidence_list) > 0
            
            # Should find access control evidence
            titles = [e.title for e in evidence_list]
            assert any("User Access" in title for title in titles)
            assert any("Permission" in title for title in titles)
            
            # Check for specific evidence types
            for evidence in evidence_list:
                assert evidence.evidence_type == EvidenceType.ACCESS_CONTROL
                assert "access" in evidence.description.lower() or "permission" in evidence.description.lower()
            
        finally:
            self.tearDown()
    
    def test_collect_encryption_evidence(self):
        """Test collection of encryption evidence"""
        self.setUp()
        
        try:
            evidence_list = self.collector.collect_encryption_evidence()
            
            assert len(evidence_list) > 0
            
            # Should find encryption evidence
            for evidence in evidence_list:
                assert evidence.evidence_type == EvidenceType.ENCRYPTION
                assert "encrypt" in evidence.description.lower() or "crypto" in evidence.description.lower()
                
            # Check for common encryption evidence
            descriptions = [e.description for e in evidence_list]
            has_tls = any("tls" in desc.lower() or "ssl" in desc.lower() for desc in descriptions)
            has_disk = any("disk" in desc.lower() or "storage" in desc.lower() for desc in descriptions)
            
            # At least one type should be found
            assert has_tls or has_disk
            
        finally:
            self.tearDown()
    
    def test_collect_audit_log_evidence(self):
        """Test collection of audit log evidence"""
        self.setUp()
        
        try:
            evidence_list = self.collector.collect_audit_log_evidence()
            
            assert len(evidence_list) > 0
            
            # Should find audit log evidence
            for evidence in evidence_list:
                assert evidence.evidence_type == EvidenceType.AUDIT_LOG
                assert "log" in evidence.description.lower() or "audit" in evidence.description.lower()
            
            # Check for log-related evidence
            titles = [e.title for e in evidence_list]
            assert any("Log" in title for title in titles)
            
        finally:
            self.tearDown()
    
    def test_collect_network_security_evidence(self):
        """Test collection of network security evidence"""
        self.setUp()
        
        try:
            evidence_list = self.collector.collect_network_security_evidence()
            
            assert len(evidence_list) > 0
            
            # Should find network security evidence
            for evidence in evidence_list:
                assert evidence.evidence_type == EvidenceType.NETWORK_SECURITY
                text = f"{evidence.title} {evidence.description}".lower()
                assert any(keyword in text for keyword in ["firewall", "network", "port", "security"])
            
        finally:
            self.tearDown()
    
    def test_collect_vulnerability_evidence(self):
        """Test collection of vulnerability assessment evidence"""
        self.setUp()
        
        try:
            evidence_list = self.collector.collect_vulnerability_evidence()
            
            assert len(evidence_list) > 0
            
            # Should find vulnerability evidence
            for evidence in evidence_list:
                assert evidence.evidence_type == EvidenceType.VULNERABILITY_ASSESSMENT
                text = f"{evidence.title} {evidence.description}".lower()
                assert any(keyword in text for keyword in ["vulnerability", "scan", "assessment", "security"])
            
        finally:
            self.tearDown()
    
    def test_collect_backup_evidence(self):
        """Test collection of backup and recovery evidence"""
        self.setUp()
        
        try:
            evidence_list = self.collector.collect_backup_evidence()
            
            assert len(evidence_list) > 0
            
            # Should find backup evidence
            for evidence in evidence_list:
                assert evidence.evidence_type == EvidenceType.BACKUP_RECOVERY
                text = f"{evidence.title} {evidence.description}".lower()
                assert any(keyword in text for keyword in ["backup", "recovery", "restore"])
            
        finally:
            self.tearDown()
    
    def test_evidence_persistence(self):
        """Test that evidence can be stored and retrieved"""
        self.setUp()
        
        try:
            # Create test evidence
            evidence = ComplianceEvidence(
                evidence_id="test_evidence_1",
                evidence_type=EvidenceType.SYSTEM_CONFIG,
                title="Test System Configuration",
                description="Test evidence for system configuration",
                collected_at=datetime.utcnow(),
                evidence_data={"config_file": "/etc/test.conf", "value": "secure"},
                source="test_collector"
            )
            
            # Store evidence
            success = self.collector.store_evidence(evidence)
            assert success
            
            # Retrieve evidence
            retrieved = self.collector.get_evidence("test_evidence_1")
            assert retrieved is not None
            assert retrieved.evidence_id == "test_evidence_1"
            assert retrieved.title == "Test System Configuration"
            assert retrieved.evidence_data["value"] == "secure"
            
        finally:
            self.tearDown()
    
    def test_evidence_search(self):
        """Test evidence search functionality"""
        self.setUp()
        
        try:
            # Create multiple test evidence items
            evidence_items = [
                ComplianceEvidence(
                    evidence_id=f"search_test_{i}",
                    evidence_type=EvidenceType.ACCESS_CONTROL,
                    title=f"Access Control Test {i}",
                    description=f"Test evidence for access control {i}",
                    collected_at=datetime.utcnow(),
                    evidence_data={"test": f"data_{i}"},
                    source="test"
                )
                for i in range(3)
            ]
            
            # Store all evidence
            for evidence in evidence_items:
                self.collector.store_evidence(evidence)
            
            # Search for evidence
            results = self.collector.search_evidence("access control")
            assert len(results) >= 3
            
            # Search by type
            access_evidence = self.collector.get_evidence_by_type(EvidenceType.ACCESS_CONTROL)
            assert len(access_evidence) >= 3
            
        finally:
            self.tearDown()


class TestAutomatedComplianceAssessor:
    """Test ID: SEC-109-C - Automated Assessment"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = EvidenceCollector(self.temp_dir)
        self.assessor = AutomatedComplianceAssessor(self.collector)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_assess_access_control_compliance(self):
        """Test assessment of access control compliance"""
        self.setUp()
        
        try:
            # Create test control
            control = ComplianceControl(
                control_id="TEST_AC_1",
                title="Test Access Control",
                description="Test access control implementation",
                framework=ComplianceFramework.SOC2,
                requirements=["Implement user authentication", "Enforce authorization"]
            )
            
            assessment = self.assessor.assess_control(control)
            
            assert assessment is not None
            assert assessment.control_id == "TEST_AC_1"
            assert assessment.assessment_date
            assert assessment.status in [ControlStatus.COMPLIANT, ControlStatus.NON_COMPLIANT, ControlStatus.PARTIAL]
            assert assessment.evidence_count >= 0
            assert assessment.findings
            
            # Should have some analysis
            assert len(assessment.findings) > 0
            
        finally:
            self.tearDown()
    
    def test_assess_encryption_compliance(self):
        """Test assessment of encryption compliance"""
        self.setUp()
        
        try:
            control = ComplianceControl(
                control_id="TEST_ENC_1",
                title="Test Encryption Control",
                description="Test encryption implementation",
                framework=ComplianceFramework.PCI_DSS,
                requirements=["Encrypt data at rest", "Encrypt data in transit"]
            )
            
            assessment = self.assessor.assess_control(control)
            
            assert assessment is not None
            assert assessment.control_id == "TEST_ENC_1"
            assert assessment.status in [ControlStatus.COMPLIANT, ControlStatus.NON_COMPLIANT, ControlStatus.PARTIAL]
            
            # Should find encryption evidence
            assert assessment.evidence_count > 0
            
        finally:
            self.tearDown()
    
    def test_assess_audit_logging_compliance(self):
        """Test assessment of audit logging compliance"""
        self.setUp()
        
        try:
            control = ComplianceControl(
                control_id="TEST_LOG_1",
                title="Test Audit Logging",
                description="Test audit logging implementation",
                framework=ComplianceFramework.SOC2,
                requirements=["Enable comprehensive logging", "Protect log integrity"]
            )
            
            assessment = self.assessor.assess_control(control)
            
            assert assessment is not None
            assert assessment.evidence_count >= 0
            
            # Should have findings about logging
            findings_text = " ".join(assessment.findings).lower()
            assert "log" in findings_text or "audit" in findings_text
            
        finally:
            self.tearDown()
    
    def test_assess_network_security_compliance(self):
        """Test assessment of network security compliance"""
        self.setUp()
        
        try:
            control = ComplianceControl(
                control_id="TEST_NET_1",
                title="Test Network Security",
                description="Test network security controls",
                framework=ComplianceFramework.OWASP,
                requirements=["Implement firewall", "Secure network configurations"]
            )
            
            assessment = self.assessor.assess_control(control)
            
            assert assessment is not None
            assert assessment.evidence_count >= 0
            
            # Should analyze network security
            findings_text = " ".join(assessment.findings).lower()
            network_keywords = ["network", "firewall", "port", "security"]
            assert any(keyword in findings_text for keyword in network_keywords)
            
        finally:
            self.tearDown()
    
    def test_assess_vulnerability_management(self):
        """Test assessment of vulnerability management compliance"""
        self.setUp()
        
        try:
            control = ComplianceControl(
                control_id="TEST_VULN_1",
                title="Test Vulnerability Management",
                description="Test vulnerability management processes",
                framework=ComplianceFramework.ISO_27001,
                requirements=["Regular vulnerability scanning", "Timely patch management"]
            )
            
            assessment = self.assessor.assess_control(control)
            
            assert assessment is not None
            assert assessment.evidence_count >= 0
            
            # Should find vulnerability-related evidence
            findings_text = " ".join(assessment.findings).lower()
            vuln_keywords = ["vulnerabilit", "scan", "patch", "assessment"]
            assert any(keyword in findings_text for keyword in vuln_keywords)
            
        finally:
            self.tearDown()
    
    def test_assessment_scoring(self):
        """Test assessment scoring logic"""
        self.setUp()
        
        try:
            control = ComplianceControl(
                control_id="TEST_SCORE_1",
                title="Test Scoring",
                description="Test control for scoring",
                framework=ComplianceFramework.SOC2,
                requirements=["Test requirement"]
            )
            
            assessment = self.assessor.assess_control(control)
            
            # Score should be between 0 and 100
            assert 0 <= assessment.compliance_score <= 100
            
            # Status should align with score
            if assessment.compliance_score >= 90:
                assert assessment.status == ControlStatus.COMPLIANT
            elif assessment.compliance_score >= 60:
                assert assessment.status == ControlStatus.PARTIAL
            else:
                assert assessment.status == ControlStatus.NON_COMPLIANT
            
        finally:
            self.tearDown()
    
    def test_batch_assessment(self):
        """Test batch assessment of multiple controls"""
        self.setUp()
        
        try:
            # Create multiple controls
            controls = [
                ComplianceControl(
                    control_id=f"BATCH_TEST_{i}",
                    title=f"Batch Test Control {i}",
                    description=f"Test control {i} for batch assessment",
                    framework=ComplianceFramework.SOC2,
                    requirements=[f"Test requirement {i}"]
                )
                for i in range(3)
            ]
            
            assessments = self.assessor.assess_controls(controls)
            
            assert len(assessments) == 3
            
            for i, assessment in enumerate(assessments):
                assert assessment.control_id == f"BATCH_TEST_{i}"
                assert assessment.status in [ControlStatus.COMPLIANT, ControlStatus.NON_COMPLIANT, ControlStatus.PARTIAL]
            
        finally:
            self.tearDown()


class TestComplianceReporter:
    """Test ID: SEC-109-D - Compliance Reporting"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = ComplianceReporter(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_generate_soc2_report(self):
        """Test SOC2 compliance report generation"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.SOC2)
            
            assert report is not None
            assert report.framework == ComplianceFramework.SOC2
            assert report.report_date
            assert report.assessments
            assert len(report.assessments) > 0
            
            # Should have SOC2-specific controls
            control_ids = [a.control_id for a in report.assessments]
            assert any("CC" in control_id for control_id in control_ids)  # SOC2 uses CC prefix
            
            # Check report summary
            assert report.summary
            assert "total_controls" in report.summary
            assert "compliant_controls" in report.summary
            assert "overall_compliance_percentage" in report.summary
            
        finally:
            self.tearDown()
    
    def test_generate_gdpr_report(self):
        """Test GDPR compliance report generation"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.GDPR)
            
            assert report.framework == ComplianceFramework.GDPR
            assert len(report.assessments) > 0
            
            # Should have GDPR-specific articles
            control_ids = [a.control_id for a in report.assessments]
            assert any("Art" in control_id for control_id in control_ids)  # GDPR uses Art prefix
            
            # Check for data protection focus
            descriptions = [a.control_id for a in report.assessments]
            gdpr_keywords = ["Art25", "Art32", "Art35"]  # Key GDPR articles
            assert any(keyword in descriptions for keyword in gdpr_keywords)
            
        finally:
            self.tearDown()
    
    def test_generate_owasp_report(self):
        """Test OWASP compliance report generation"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.OWASP)
            
            assert report.framework == ComplianceFramework.OWASP
            assert len(report.assessments) > 0
            
            # Should have OWASP Top 10 categories
            control_ids = [a.control_id for a in report.assessments]
            owasp_categories = ["A01", "A02", "A03"]  # Top OWASP categories
            assert any(cat in control_ids for cat in owasp_categories)
            
        finally:
            self.tearDown()
    
    def test_generate_pci_dss_report(self):
        """Test PCI DSS compliance report generation"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.PCI_DSS)
            
            assert report.framework == ComplianceFramework.PCI_DSS
            assert len(report.assessments) > 0
            
            # Should have PCI DSS requirements
            control_ids = [a.control_id for a in report.assessments]
            pci_requirements = ["2.1", "3.4", "6.5"]  # Key PCI DSS requirements
            assert any(req in control_ids for req in pci_requirements)
            
        finally:
            self.tearDown()
    
    def test_generate_iso27001_report(self):
        """Test ISO 27001 compliance report generation"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.ISO_27001)
            
            assert report.framework == ComplianceFramework.ISO_27001
            assert len(report.assessments) > 0
            
            # Should have ISO 27001 controls
            control_ids = [a.control_id for a in report.assessments]
            iso_controls = ["A.9.1.1", "A.10.1.1", "A.12.2.1"]  # Key ISO controls
            assert any(ctrl in control_ids for ctrl in iso_controls)
            
        finally:
            self.tearDown()
    
    def test_export_json_report(self):
        """Test JSON report export"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.SOC2)
            
            # Export to JSON
            json_output = self.reporter.export_report(report, "json")
            
            assert json_output
            
            # Parse JSON to verify structure
            data = json.loads(json_output)
            
            assert "framework" in data
            assert "report_date" in data
            assert "assessments" in data
            assert "summary" in data
            
            assert data["framework"] == "SOC2"
            assert len(data["assessments"]) > 0
            
            # Check assessment structure
            assessment = data["assessments"][0]
            assert "control_id" in assessment
            assert "status" in assessment
            assert "compliance_score" in assessment
            
        finally:
            self.tearDown()
    
    def test_export_csv_report(self):
        """Test CSV report export"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.OWASP)
            
            # Export to CSV
            csv_output = self.reporter.export_report(report, "csv")
            
            assert csv_output
            
            # Check CSV structure
            lines = csv_output.strip().split('\n')
            assert len(lines) > 1  # Header + data rows
            
            # Check header
            header = lines[0]
            expected_columns = ["Control ID", "Title", "Status", "Compliance Score", "Evidence Count"]
            for col in expected_columns:
                assert col in header
            
            # Check data rows
            for line in lines[1:]:
                assert len(line.split(',')) >= 5  # At least 5 columns
            
        finally:
            self.tearDown()
    
    def test_export_html_report(self):
        """Test HTML report export"""
        self.setUp()
        
        try:
            report = self.reporter.generate_compliance_report(ComplianceFramework.GDPR)
            
            # Export to HTML
            html_output = self.reporter.export_report(report, "html")
            
            assert html_output
            
            # Check HTML structure
            assert "<html>" in html_output
            assert "<head>" in html_output
            assert "<body>" in html_output
            assert "</html>" in html_output
            
            # Check content elements
            assert "GDPR Compliance Report" in html_output
            assert "Overall Compliance" in html_output
            assert "table" in html_output.lower()
            
            # Should include CSS styling
            assert "style" in html_output.lower() or "css" in html_output.lower()
            
        finally:
            self.tearDown()
    
    def test_report_dashboard_data(self):
        """Test dashboard data generation"""
        self.setUp()
        
        try:
            dashboard_data = self.reporter.generate_dashboard_data()
            
            assert dashboard_data
            
            # Check structure
            assert "frameworks" in dashboard_data
            assert "overall_stats" in dashboard_data
            assert "recent_assessments" in dashboard_data
            assert "compliance_trends" in dashboard_data
            
            # Check frameworks data
            frameworks = dashboard_data["frameworks"]
            assert len(frameworks) > 0
            
            for framework_data in frameworks:
                assert "framework" in framework_data
                assert "compliance_percentage" in framework_data
                assert "total_controls" in framework_data
                assert "compliant_controls" in framework_data
            
            # Check overall stats
            overall = dashboard_data["overall_stats"]
            assert "total_frameworks" in overall
            assert "average_compliance" in overall
            assert "total_controls_assessed" in overall
            
        finally:
            self.tearDown()
    
    def test_report_filtering(self):
        """Test report filtering by status and score"""
        self.setUp()
        
        try:
            # Generate base report
            report = self.reporter.generate_compliance_report(ComplianceFramework.SOC2)
            
            # Filter by compliant status
            compliant_assessments = [a for a in report.assessments if a.status == ControlStatus.COMPLIANT]
            
            # Filter by high compliance score
            high_score_assessments = [a for a in report.assessments if a.compliance_score >= 80]
            
            # Should be able to filter assessments
            assert isinstance(compliant_assessments, list)
            assert isinstance(high_score_assessments, list)
            
            # Verify filtering logic
            for assessment in compliant_assessments:
                assert assessment.status == ControlStatus.COMPLIANT
            
            for assessment in high_score_assessments:
                assert assessment.compliance_score >= 80
            
        finally:
            self.tearDown()


class TestIntegration:
    """Test ID: SEC-109-E - Integration Tests"""
    
    def test_end_to_end_compliance_workflow(self):
        """Test complete compliance reporting workflow"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Initialize components
            reporter = ComplianceReporter(temp_dir)
            
            # 2. Generate comprehensive report
            report = reporter.generate_compliance_report(ComplianceFramework.SOC2)
            
            assert report is not None
            assert report.framework == ComplianceFramework.SOC2
            assert len(report.assessments) > 0
            
            # 3. Verify evidence collection worked
            total_evidence = sum(a.evidence_count for a in report.assessments)
            assert total_evidence > 0
            
            # 4. Check assessment quality
            assessed_controls = len(report.assessments)
            assert assessed_controls > 10  # Should assess multiple controls
            
            # 5. Verify report summary
            summary = report.summary
            assert summary["total_controls"] == assessed_controls
            assert 0 <= summary["overall_compliance_percentage"] <= 100
            
            # 6. Export in multiple formats
            json_export = reporter.export_report(report, "json")
            csv_export = reporter.export_report(report, "csv")
            html_export = reporter.export_report(report, "html")
            
            assert json_export
            assert csv_export
            assert html_export
            
            # 7. Verify exports contain data
            json_data = json.loads(json_export)
            assert len(json_data["assessments"]) == assessed_controls
            
            csv_lines = csv_export.strip().split('\n')
            assert len(csv_lines) == assessed_controls + 1  # Header + data
            
            assert "SOC2" in html_export
            
            # 8. Generate dashboard data
            dashboard = reporter.generate_dashboard_data()
            assert dashboard["overall_stats"]["total_frameworks"] >= 1
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_multi_framework_assessment(self):
        """Test assessment across multiple compliance frameworks"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            reporter = ComplianceReporter(temp_dir)
            
            # Generate reports for multiple frameworks
            frameworks = [
                ComplianceFramework.SOC2,
                ComplianceFramework.GDPR,
                ComplianceFramework.OWASP
            ]
            
            reports = {}
            for framework in frameworks:
                report = reporter.generate_compliance_report(framework)
                reports[framework] = report
                
                assert report.framework == framework
                assert len(report.assessments) > 0
            
            # Verify each framework has unique controls
            all_control_ids = set()
            for report in reports.values():
                control_ids = {a.control_id for a in report.assessments}
                # Some overlap is expected, but each should have unique elements
                all_control_ids.update(control_ids)
            
            # Should have substantial coverage across frameworks
            assert len(all_control_ids) > 20
            
            # Generate combined dashboard
            dashboard = reporter.generate_dashboard_data()
            
            # Should show data for multiple frameworks
            framework_data = dashboard["frameworks"]
            framework_names = [f["framework"] for f in framework_data]
            
            assert len(framework_names) >= 3
            assert "SOC2" in framework_names
            assert "GDPR" in framework_names
            assert "OWASP" in framework_names
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_compliance_evidence_integration(self):
        """Test integration between evidence collection and assessment"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize components
            collector = EvidenceCollector(temp_dir)
            assessor = AutomatedComplianceAssessor(collector)
            
            # Collect evidence first
            system_evidence = collector.collect_system_evidence()
            access_evidence = collector.collect_access_control_evidence()
            
            total_evidence = len(system_evidence) + len(access_evidence)
            assert total_evidence > 0
            
            # Create a control that should use this evidence
            control = ComplianceControl(
                control_id="INTEGRATION_TEST",
                title="Integration Test Control",
                description="Test control for evidence integration",
                framework=ComplianceFramework.SOC2,
                requirements=["System security configuration", "Access control implementation"]
            )
            
            # Assess the control
            assessment = assessor.assess_control(control)
            
            # Assessment should have found and used evidence
            assert assessment.evidence_count > 0
            assert len(assessment.findings) > 0
            
            # Findings should reference evidence
            findings_text = " ".join(assessment.findings).lower()
            evidence_keywords = ["evidence", "found", "configuration", "control"]
            assert any(keyword in findings_text for keyword in evidence_keywords)
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_report_persistence_and_retrieval(self):
        """Test that reports can be stored and retrieved"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            reporter = ComplianceReporter(temp_dir)
            
            # Generate and store report
            original_report = reporter.generate_compliance_report(ComplianceFramework.ISO_27001)
            
            # Save report
            report_id = f"test_report_{int(datetime.utcnow().timestamp())}"
            success = reporter.save_report(original_report, report_id)
            assert success
            
            # Retrieve report
            retrieved_report = reporter.load_report(report_id)
            
            assert retrieved_report is not None
            assert retrieved_report.framework == ComplianceFramework.ISO_27001
            assert len(retrieved_report.assessments) == len(original_report.assessments)
            
            # Compare assessment details
            original_ids = {a.control_id for a in original_report.assessments}
            retrieved_ids = {a.control_id for a in retrieved_report.assessments}
            assert original_ids == retrieved_ids
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run compliance reporting tests
    pytest.main([__file__, "-v", "-s"])