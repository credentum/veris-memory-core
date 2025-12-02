#!/usr/bin/env python3
"""
Comprehensive tests for Security Compliance Reporter - Phase 7 Coverage

This test module provides comprehensive coverage for the compliance reporting system
including control libraries, evidence collection, automated assessment, and reporting.
"""
import pytest
import tempfile
import os
import json
import csv
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock, mock_open
from typing import Dict, Any, List

# Import compliance reporter components
try:
    from src.security.compliance_reporter import (
        ComplianceFramework, ComplianceStatus, ControlCategory, EvidenceType,
        ComplianceControl, ComplianceEvidence, ComplianceAssessment, ComplianceReport,
        ComplianceControlLibrary, AutomatedComplianceAssessor, EvidenceCollector, 
        ComplianceReporter, ControlStatus
    )
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestComplianceEnums:
    """Test compliance framework enums and constants"""
    
    def test_compliance_framework_enum(self):
        """Test ComplianceFramework enum values"""
        assert ComplianceFramework.SOC2.value == "soc2"
        assert ComplianceFramework.ISO_27001.value == "iso27001"
        assert ComplianceFramework.ISO27001.value == "iso27001"  # Backward compatibility
        assert ComplianceFramework.GDPR.value == "gdpr"
        assert ComplianceFramework.HIPAA.value == "hipaa"
        assert ComplianceFramework.PCI_DSS.value == "pci_dss"
        assert ComplianceFramework.NIST.value == "nist"
        assert ComplianceFramework.OWASP.value == "owasp"
        assert ComplianceFramework.CIS.value == "cis"
        assert ComplianceFramework.CCPA.value == "ccpa"
        assert ComplianceFramework.SOX.value == "sox"
    
    def test_compliance_status_enum(self):
        """Test ComplianceStatus enum values"""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.NOT_APPLICABLE.value == "not_applicable"
        assert ComplianceStatus.UNKNOWN.value == "unknown"
    
    def test_control_category_enum(self):
        """Test ControlCategory enum values"""
        assert ControlCategory.ACCESS_CONTROL.value == "access_control"
        assert ControlCategory.AUTHENTICATION.value == "authentication"
        assert ControlCategory.AUTHORIZATION.value == "authorization"
        assert ControlCategory.DATA_PROTECTION.value == "data_protection"
        assert ControlCategory.ENCRYPTION.value == "encryption"
        assert ControlCategory.LOGGING_MONITORING.value == "logging_monitoring"
        assert ControlCategory.NETWORK_SECURITY.value == "network_security"
        assert ControlCategory.VULNERABILITY_MANAGEMENT.value == "vulnerability_management"
        assert ControlCategory.INCIDENT_RESPONSE.value == "incident_response"
        assert ControlCategory.BUSINESS_CONTINUITY.value == "business_continuity"
        assert ControlCategory.RISK_MANAGEMENT.value == "risk_management"
        assert ControlCategory.GOVERNANCE.value == "governance"
    
    def test_evidence_type_enum(self):
        """Test EvidenceType enum values"""
        assert EvidenceType.SYSTEM_CONFIG.value == "system_config"
        assert EvidenceType.SECURITY_CONFIG.value == "security_config"
        assert EvidenceType.ACCESS_CONTROL.value == "access_control"
        assert EvidenceType.ENCRYPTION.value == "encryption"
        assert EvidenceType.AUDIT_LOG.value == "audit_log"
        assert EvidenceType.NETWORK_SECURITY.value == "network_security"
        assert EvidenceType.VULNERABILITY_ASSESSMENT.value == "vulnerability_assessment"
        assert EvidenceType.BACKUP_RECOVERY.value == "backup_recovery"
        assert EvidenceType.DOCUMENTATION.value == "documentation"
        assert EvidenceType.POLICY.value == "policy"
    
    def test_control_status_alias(self):
        """Test ControlStatus backward compatibility alias"""
        assert ControlStatus == ComplianceStatus
        assert ControlStatus.COMPLIANT == ComplianceStatus.COMPLIANT


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestComplianceDataModels:
    """Test compliance data model classes"""
    
    def test_compliance_control_creation(self):
        """Test ComplianceControl dataclass creation"""
        control = ComplianceControl(
            control_id="CC6.1",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.ACCESS_CONTROL,
            title="Test Control",
            description="Test description",
            requirements=["req1", "req2"],
            implementation_guidance="Test guidance",
            testing_procedures=["test1", "test2"],
            references=["ref1", "ref2"],
            automation_possible=True,
            risk_level="high"
        )
        
        assert control.control_id == "CC6.1"
        assert control.framework == ComplianceFramework.SOC2
        assert control.category == ControlCategory.ACCESS_CONTROL
        assert control.title == "Test Control"
        assert control.description == "Test description"
        assert control.requirements == ["req1", "req2"]
        assert control.implementation_guidance == "Test guidance"
        assert control.testing_procedures == ["test1", "test2"]
        assert control.references == ["ref1", "ref2"]
        assert control.automation_possible is True
        assert control.risk_level == "high"
    
    def test_compliance_control_defaults(self):
        """Test ComplianceControl default values"""
        control = ComplianceControl(
            control_id="TEST-001",
            framework=ComplianceFramework.GDPR,
            category=ControlCategory.DATA_PROTECTION,
            title="Test Control",
            description="Test description"
        )
        
        assert control.requirements == []
        assert control.implementation_guidance == ""
        assert control.testing_procedures == []
        assert control.references == []
        assert control.automation_possible is False
        assert control.risk_level == "medium"
    
    def test_compliance_evidence_creation(self):
        """Test ComplianceEvidence dataclass creation"""
        now = datetime.utcnow()
        evidence = ComplianceEvidence(
            evidence_id="EVD-001",
            control_id="CC6.1",
            evidence_type="configuration",
            title="Test Evidence",
            description="Test evidence description",
            file_path="/path/to/evidence.json",
            content="evidence content",
            collected_at=now,
            collected_by="test_user",
            metadata={"key": "value"}
        )
        
        assert evidence.evidence_id == "EVD-001"
        assert evidence.control_id == "CC6.1"
        assert evidence.evidence_type == "configuration"
        assert evidence.title == "Test Evidence"
        assert evidence.description == "Test evidence description"
        assert evidence.file_path == "/path/to/evidence.json"
        assert evidence.content == "evidence content"
        assert evidence.collected_at == now
        assert evidence.collected_by == "test_user"
        assert evidence.metadata == {"key": "value"}
    
    def test_compliance_assessment_creation(self):
        """Test ComplianceAssessment dataclass creation"""
        now = datetime.utcnow()
        next_due = now + timedelta(days=90)
        
        assessment = ComplianceAssessment(
            assessment_id="ASS-001",
            control_id="CC6.1",
            status=ComplianceStatus.COMPLIANT,
            score=95.0,
            findings=["Finding 1", "Finding 2"],
            evidence_ids=["EVD-001", "EVD-002"],
            assessed_at=now,
            assessed_by="assessor",
            next_assessment_due=next_due,
            remediation_notes="Test remediation",
            risk_rating="low",
            recommendations=["Rec 1", "Rec 2"],
            metadata={"assessment_type": "automated"},
            summary="Test summary"
        )
        
        assert assessment.assessment_id == "ASS-001"
        assert assessment.control_id == "CC6.1"
        assert assessment.status == ComplianceStatus.COMPLIANT
        assert assessment.score == 95.0
        assert assessment.findings == ["Finding 1", "Finding 2"]
        assert assessment.evidence_ids == ["EVD-001", "EVD-002"]
        assert assessment.assessed_at == now
        assert assessment.assessed_by == "assessor"
        assert assessment.next_assessment_due == next_due
        assert assessment.remediation_notes == "Test remediation"
        assert assessment.risk_rating == "low"
        assert assessment.recommendations == ["Rec 1", "Rec 2"]
        assert assessment.metadata == {"assessment_type": "automated"}
        assert assessment.summary == "Test summary"
    
    def test_compliance_report_creation(self):
        """Test ComplianceReport dataclass creation"""
        now = datetime.utcnow()
        start_date = now - timedelta(days=30)
        
        assessment1 = ComplianceAssessment(
            assessment_id="ASS-001",
            control_id="CC6.1",
            status=ComplianceStatus.COMPLIANT
        )
        
        assessment2 = ComplianceAssessment(
            assessment_id="ASS-002",
            control_id="CC6.2",
            status=ComplianceStatus.NON_COMPLIANT
        )
        
        report = ComplianceReport(
            report_id="RPT-001",
            framework=ComplianceFramework.SOC2,
            report_type="full",
            generated_at=now,
            reporting_period_start=start_date,
            reporting_period_end=now,
            assessments=[assessment1, assessment2],
            overall_score=85.5,
            compliance_percentage=75.0,
            generated_by="automated_system",
            recommendations=["Improve access controls"],
            metadata={"version": "1.0"},
            summary={"total_controls": 2, "compliant": 1},
            executive_summary="Executive summary text"
        )
        
        assert report.report_id == "RPT-001"
        assert report.framework == ComplianceFramework.SOC2
        assert report.report_type == "full"
        assert report.generated_at == now
        assert report.reporting_period_start == start_date
        assert report.reporting_period_end == now
        assert len(report.assessments) == 2
        assert report.overall_score == 85.5
        assert report.compliance_percentage == 75.0
        assert report.generated_by == "automated_system"
        assert report.recommendations == ["Improve access controls"]
        assert report.metadata == {"version": "1.0"}
        assert report.summary == {"total_controls": 2, "compliant": 1}
        assert report.executive_summary == "Executive summary text"


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestComplianceControlLibrary:
    """Test compliance control library"""
    
    def setup_method(self):
        """Setup test environment"""
        self.library = ComplianceControlLibrary()
    
    def test_library_initialization(self):
        """Test control library initialization"""
        assert self.library is not None
        assert hasattr(self.library, 'controls')
        assert isinstance(self.library.controls, list)
        assert len(self.library.controls) > 0
    
    def test_get_controls_by_framework(self):
        """Test getting controls by framework"""
        soc2_controls = self.library.get_controls(ComplianceFramework.SOC2)
        assert isinstance(soc2_controls, list)
        assert len(soc2_controls) > 0
        
        # Verify all returned controls are SOC2
        for control in soc2_controls:
            assert control.framework == ComplianceFramework.SOC2
        
        gdpr_controls = self.library.get_controls(ComplianceFramework.GDPR)
        assert isinstance(gdpr_controls, list)
        assert len(gdpr_controls) > 0
        
        # Verify all returned controls are GDPR
        for control in gdpr_controls:
            assert control.framework == ComplianceFramework.GDPR
    
    def test_get_specific_control(self):
        """Test getting specific control by ID"""
        # Test existing control
        control = self.library.get_control("CC6.1")
        assert control is not None
        assert control.control_id == "CC6.1"
        assert control.framework == ComplianceFramework.SOC2
        
        # Test non-existent control
        non_existent = self.library.get_control("NON-EXISTENT")
        assert non_existent is None
    
    def test_control_framework_coverage(self):
        """Test that all major frameworks have controls"""
        frameworks_to_test = [
            ComplianceFramework.SOC2,
            ComplianceFramework.GDPR,
            ComplianceFramework.OWASP,
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.ISO_27001
        ]
        
        for framework in frameworks_to_test:
            controls = self.library.get_controls(framework)
            assert len(controls) > 0, f"No controls found for {framework.value}"
    
    def test_control_categories_coverage(self):
        """Test that controls cover all major categories"""
        all_controls = self.library.controls
        categories_found = set()
        
        for control in all_controls:
            categories_found.add(control.category)
        
        # Should have multiple categories
        assert len(categories_found) >= 5
        
        # Check for key categories
        expected_categories = [
            ControlCategory.ACCESS_CONTROL,
            ControlCategory.AUTHENTICATION,
            ControlCategory.ENCRYPTION,
            ControlCategory.LOGGING_MONITORING
        ]
        
        for category in expected_categories:
            assert category in categories_found
    
    def test_control_risk_levels(self):
        """Test control risk level distribution"""
        all_controls = self.library.controls
        risk_levels = set()
        
        for control in all_controls:
            risk_levels.add(control.risk_level)
        
        # Should have various risk levels
        expected_levels = ["low", "medium", "high", "critical"]
        for level in expected_levels:
            controls_with_level = [c for c in all_controls if c.risk_level == level]
            # At least some controls should exist for each level
            if level in ["medium", "high"]:  # Common levels
                assert len(controls_with_level) > 0


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestAutomatedComplianceAssessor:
    """Test automated compliance assessor"""
    
    def setup_method(self):
        """Setup test environment"""
        self.assessor = AutomatedComplianceAssessor()
    
    @patch('src.security.compliance_reporter.os.path.exists')
    @patch('src.security.compliance_reporter.open', new_callable=mock_open)
    def test_assess_control_with_evidence(self, mock_file, mock_exists):
        """Test control assessment with evidence"""
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = "test configuration"
        
        control = ComplianceControl(
            control_id="TEST-001",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.ACCESS_CONTROL,
            title="Test Control",
            description="Test description",
            automation_possible=True
        )
        
        evidence = [
            ComplianceEvidence(
                evidence_id="EVD-001",
                control_id="TEST-001",
                evidence_type="configuration",
                title="Config Evidence",
                description="Configuration evidence",
                file_path="/test/config.json"
            )
        ]
        
        assessment = self.assessor.assess_control(control, evidence)
        
        assert assessment is not None
        assert assessment.control_id == "TEST-001"
        assert assessment.status in [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.PARTIALLY_COMPLIANT
        ]
        assert isinstance(assessment.score, (int, float, type(None)))
        assert isinstance(assessment.findings, list)
        assert isinstance(assessment.recommendations, list)
    
    def test_assess_control_without_evidence(self):
        """Test control assessment without evidence"""
        control = ComplianceControl(
            control_id="TEST-002",
            framework=ComplianceFramework.GDPR,
            category=ControlCategory.DATA_PROTECTION,
            title="Test Control",
            description="Test description"
        )
        
        assessment = self.assessor.assess_control(control, [])
        
        assert assessment is not None
        assert assessment.control_id == "TEST-002"
        assert assessment.status == ComplianceStatus.UNKNOWN
        assert "No evidence available" in " ".join(assessment.findings)
    
    def test_assess_multiple_controls(self):
        """Test assessing multiple controls"""
        controls = [
            ComplianceControl(
                control_id="TEST-001",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.ACCESS_CONTROL,
                title="Test Control 1",
                description="Test description 1"
            ),
            ComplianceControl(
                control_id="TEST-002",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.ENCRYPTION,
                title="Test Control 2",
                description="Test description 2"
            )
        ]
        
        assessments = self.assessor.assess_controls(controls, [])
        
        assert len(assessments) == 2
        assert assessments[0].control_id == "TEST-001"
        assert assessments[1].control_id == "TEST-002"
    
    def test_automation_scoring(self):
        """Test automation scoring logic"""
        # Test high automation score for automated controls
        control_automated = ComplianceControl(
            control_id="AUTO-001",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.LOGGING_MONITORING,
            title="Automated Control",
            description="Automated monitoring control",
            automation_possible=True
        )
        
        evidence_automated = [
            ComplianceEvidence(
                evidence_id="EVD-AUTO",
                control_id="AUTO-001",
                evidence_type="audit_log",
                title="Automated Log Evidence",
                description="Automated log collection evidence"
            )
        ]
        
        assessment_automated = self.assessor.assess_control(control_automated, evidence_automated)
        
        # Automated controls with evidence should score higher
        if assessment_automated.score is not None:
            assert assessment_automated.score >= 60
    
    def test_risk_level_impact_on_scoring(self):
        """Test risk level impact on scoring"""
        high_risk_control = ComplianceControl(
            control_id="HIGH-RISK",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.ACCESS_CONTROL,
            title="High Risk Control",
            description="High risk control",
            risk_level="high"
        )
        
        low_risk_control = ComplianceControl(
            control_id="LOW-RISK",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.ACCESS_CONTROL,
            title="Low Risk Control",
            description="Low risk control",
            risk_level="low"
        )
        
        # Both with minimal evidence
        minimal_evidence = [
            ComplianceEvidence(
                evidence_id="EVD-MIN",
                control_id="MIN-001",
                evidence_type="documentation",
                title="Minimal Evidence",
                description="Minimal documentation"
            )
        ]
        
        high_risk_assessment = self.assessor.assess_control(high_risk_control, minimal_evidence)
        low_risk_assessment = self.assessor.assess_control(low_risk_control, minimal_evidence)
        
        # High risk controls should be more stringent in scoring
        assert high_risk_assessment.risk_rating in ["high", "critical"]
        assert low_risk_assessment.risk_rating in ["low", "medium"]


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestEvidenceCollector:
    """Test evidence collector"""
    
    def setup_method(self):
        """Setup test environment"""
        self.collector = EvidenceCollector()
    
    @patch('src.security.compliance_reporter.os.path.exists')
    @patch('src.security.compliance_reporter.os.path.isfile')
    @patch('src.security.compliance_reporter.open', new_callable=mock_open)
    def test_collect_file_evidence(self, mock_file, mock_isfile, mock_exists):
        """Test collecting file-based evidence"""
        mock_exists.return_value = True
        mock_isfile.return_value = True
        mock_file.return_value.read.return_value = "test file content"
        
        evidence = self.collector.collect_file_evidence(
            control_id="TEST-001",
            file_path="/test/config.json",
            evidence_type="configuration",
            title="Test Config",
            description="Test configuration file"
        )
        
        assert evidence is not None
        assert evidence.control_id == "TEST-001"
        assert evidence.file_path == "/test/config.json"
        assert evidence.evidence_type == "configuration"
        assert evidence.title == "Test Config"
        assert evidence.description == "Test configuration file"
        assert evidence.content == "test file content"
        assert evidence.collected_by == "system"
    
    @patch('src.security.compliance_reporter.os.path.exists')
    def test_collect_file_evidence_missing_file(self, mock_exists):
        """Test collecting evidence for missing file"""
        mock_exists.return_value = False
        
        evidence = self.collector.collect_file_evidence(
            control_id="TEST-002",
            file_path="/missing/file.json",
            evidence_type="configuration",
            title="Missing Config",
            description="Missing configuration file"
        )
        
        assert evidence is None
    
    def test_collect_system_evidence(self):
        """Test collecting system-based evidence"""
        evidence = self.collector.collect_system_evidence(
            control_id="SYS-001",
            evidence_type="system_config",
            title="System Evidence",
            description="System configuration evidence",
            content="system information"
        )
        
        assert evidence is not None
        assert evidence.control_id == "SYS-001"
        assert evidence.evidence_type == "system_config"
        assert evidence.title == "System Evidence"
        assert evidence.description == "System configuration evidence"
        assert evidence.content == "system information"
        assert evidence.file_path is None
    
    @patch('src.security.compliance_reporter.subprocess.run')
    def test_collect_command_evidence(self, mock_subprocess):
        """Test collecting command-based evidence"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "command output"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        evidence = self.collector.collect_command_evidence(
            control_id="CMD-001",
            command=["ls", "-la"],
            evidence_type="system_config",
            title="Directory Listing",
            description="System directory listing"
        )
        
        assert evidence is not None
        assert evidence.control_id == "CMD-001"
        assert evidence.evidence_type == "system_config"
        assert evidence.title == "Directory Listing"
        assert evidence.content == "command output"
        assert "Command: ls -la" in evidence.description
    
    @patch('src.security.compliance_reporter.subprocess.run')
    def test_collect_command_evidence_failure(self, mock_subprocess):
        """Test collecting evidence for failed command"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "command failed"
        mock_subprocess.return_value = mock_result
        
        evidence = self.collector.collect_command_evidence(
            control_id="CMD-002",
            command=["failing-command"],
            evidence_type="system_config",
            title="Failed Command",
            description="Failed command execution"
        )
        
        assert evidence is not None
        assert evidence.control_id == "CMD-002"
        assert "Command failed" in evidence.content
        assert "command failed" in evidence.content
    
    def test_bulk_evidence_collection(self):
        """Test collecting multiple pieces of evidence"""
        evidence_configs = [
            {
                "type": "system",
                "control_id": "BULK-001",
                "evidence_type": "system_config",
                "title": "System Evidence 1",
                "description": "First evidence",
                "content": "content 1"
            },
            {
                "type": "system",
                "control_id": "BULK-002",
                "evidence_type": "system_config",
                "title": "System Evidence 2",
                "description": "Second evidence",
                "content": "content 2"
            }
        ]
        
        evidence_list = []
        for config in evidence_configs:
            evidence = self.collector.collect_system_evidence(
                control_id=config["control_id"],
                evidence_type=config["evidence_type"],
                title=config["title"],
                description=config["description"],
                content=config["content"]
            )
            if evidence:
                evidence_list.append(evidence)
        
        assert len(evidence_list) == 2
        assert evidence_list[0].control_id == "BULK-001"
        assert evidence_list[1].control_id == "BULK-002"


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestComplianceReporter:
    """Test main compliance reporter"""
    
    def setup_method(self):
        """Setup test environment"""
        self.reporter = ComplianceReporter()
    
    def test_reporter_initialization(self):
        """Test compliance reporter initialization"""
        assert self.reporter is not None
        assert hasattr(self.reporter, 'control_library')
        assert hasattr(self.reporter, 'assessor')
        assert hasattr(self.reporter, 'evidence_collector')
        assert isinstance(self.reporter.control_library, ComplianceControlLibrary)
        assert isinstance(self.reporter.assessor, AutomatedComplianceAssessor)
        assert isinstance(self.reporter.evidence_collector, EvidenceCollector)
    
    def test_generate_framework_report(self):
        """Test generating a compliance report for a framework"""
        report = self.reporter.generate_compliance_report(
            framework=ComplianceFramework.SOC2,
            report_type="summary"
        )
        
        assert report is not None
        assert report.framework == ComplianceFramework.SOC2
        assert report.report_type == "summary"
        assert isinstance(report.assessments, list)
        assert len(report.assessments) > 0
        assert report.compliance_percentage >= 0
        assert report.compliance_percentage <= 100
        assert isinstance(report.summary, dict)
    
    def test_generate_report_with_date_range(self):
        """Test generating report with specific date range"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        report = self.reporter.generate_compliance_report(
            framework=ComplianceFramework.GDPR,
            report_type="full",
            start_date=start_date,
            end_date=end_date
        )
        
        assert report.reporting_period_start == start_date
        assert report.reporting_period_end == end_date
        assert report.framework == ComplianceFramework.GDPR
        assert report.report_type == "full"
    
    @patch('src.security.compliance_reporter.os.makedirs')
    @patch('src.security.compliance_reporter.open', new_callable=mock_open)
    def test_export_report_json(self, mock_file, mock_makedirs):
        """Test exporting report to JSON"""
        report = self.reporter.generate_compliance_report(
            framework=ComplianceFramework.SOC2,
            report_type="summary"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "report.json")
            result = self.reporter.export_report(report, output_path, format="json")
            
            assert result is True
            mock_file.assert_called()
    
    @patch('src.security.compliance_reporter.os.makedirs')
    @patch('src.security.compliance_reporter.open', new_callable=mock_open)
    def test_export_report_csv(self, mock_file, mock_makedirs):
        """Test exporting report to CSV"""
        report = self.reporter.generate_compliance_report(
            framework=ComplianceFramework.GDPR,
            report_type="summary"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "report.csv")
            result = self.reporter.export_report(report, output_path, format="csv")
            
            assert result is True
            mock_file.assert_called()
    
    def test_get_compliance_score(self):
        """Test compliance score calculation"""
        report = self.reporter.generate_compliance_report(
            framework=ComplianceFramework.OWASP,
            report_type="summary"
        )
        
        score = self.reporter.get_compliance_score(ComplianceFramework.OWASP)
        
        assert isinstance(score, (int, float))
        assert score >= 0
        assert score <= 100
    
    def test_get_control_status_summary(self):
        """Test control status summary"""
        summary = self.reporter.get_control_status_summary(ComplianceFramework.SOC2)
        
        assert isinstance(summary, dict)
        assert "total_controls" in summary
        assert "compliant" in summary
        assert "non_compliant" in summary
        assert "partially_compliant" in summary
        assert "not_assessed" in summary
        
        # Verify counts make sense
        total = summary["total_controls"]
        compliant = summary["compliant"]
        non_compliant = summary["non_compliant"]
        partially_compliant = summary["partially_compliant"]
        not_assessed = summary["not_assessed"]
        
        assert total == compliant + non_compliant + partially_compliant + not_assessed
    
    def test_get_framework_recommendations(self):
        """Test getting framework-specific recommendations"""
        recommendations = self.reporter.get_framework_recommendations(ComplianceFramework.PCI_DSS)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 0
        
        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
    
    def test_compare_frameworks(self):
        """Test comparing compliance across frameworks"""
        frameworks = [ComplianceFramework.SOC2, ComplianceFramework.GDPR]
        comparison = self.reporter.compare_frameworks(frameworks)
        
        assert isinstance(comparison, dict)
        assert "frameworks" in comparison
        assert "comparison_matrix" in comparison
        assert "recommendations" in comparison
        
        assert len(comparison["frameworks"]) == 2
        assert ComplianceFramework.SOC2.value in comparison["frameworks"]
        assert ComplianceFramework.GDPR.value in comparison["frameworks"]
    
    def test_schedule_assessment(self):
        """Test scheduling compliance assessment"""
        schedule_info = self.reporter.schedule_assessment(
            framework=ComplianceFramework.ISO_27001,
            frequency_days=90,
            next_assessment=datetime.utcnow() + timedelta(days=30)
        )
        
        assert isinstance(schedule_info, dict)
        assert "framework" in schedule_info
        assert "frequency_days" in schedule_info
        assert "next_assessment" in schedule_info
        assert "scheduled" in schedule_info
        
        assert schedule_info["framework"] == ComplianceFramework.ISO_27001.value
        assert schedule_info["frequency_days"] == 90
        assert schedule_info["scheduled"] is True


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestComplianceIntegrationScenarios:
    """Test integrated compliance scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.reporter = ComplianceReporter()
    
    def test_end_to_end_compliance_workflow(self):
        """Test complete compliance workflow"""
        # 1. Generate evidence
        evidence = self.reporter.evidence_collector.collect_system_evidence(
            control_id="CC6.1",
            evidence_type="security_config",
            title="Access Control Configuration",
            description="System access control settings",
            content="access_control_enabled=true"
        )
        
        # 2. Get specific control
        control = self.reporter.control_library.get_control("CC6.1")
        assert control is not None
        
        # 3. Assess control with evidence
        assessment = self.reporter.assessor.assess_control(control, [evidence])
        assert assessment is not None
        
        # 4. Generate comprehensive report
        report = self.reporter.generate_compliance_report(
            framework=ComplianceFramework.SOC2,
            report_type="full"
        )
        
        assert report is not None
        assert len(report.assessments) > 0
    
    def test_multi_framework_compliance(self):
        """Test compliance across multiple frameworks"""
        frameworks = [
            ComplianceFramework.SOC2,
            ComplianceFramework.GDPR,
            ComplianceFramework.OWASP
        ]
        
        reports = []
        for framework in frameworks:
            report = self.reporter.generate_compliance_report(
                framework=framework,
                report_type="summary"
            )
            reports.append(report)
        
        assert len(reports) == 3
        
        # Each report should be for different framework
        framework_values = [report.framework for report in reports]
        assert len(set(framework_values)) == 3
    
    def test_compliance_trend_analysis(self):
        """Test compliance trend analysis over time"""
        # Generate reports for different time periods
        now = datetime.utcnow()
        periods = [
            (now - timedelta(days=90), now - timedelta(days=60)),
            (now - timedelta(days=60), now - timedelta(days=30)),
            (now - timedelta(days=30), now)
        ]
        
        trend_data = []
        for start_date, end_date in periods:
            report = self.reporter.generate_compliance_report(
                framework=ComplianceFramework.SOC2,
                report_type="summary",
                start_date=start_date,
                end_date=end_date
            )
            
            trend_data.append({
                "period_end": end_date,
                "compliance_percentage": report.compliance_percentage,
                "total_assessments": len(report.assessments)
            })
        
        assert len(trend_data) == 3
        
        # Verify trend data structure
        for data_point in trend_data:
            assert "period_end" in data_point
            assert "compliance_percentage" in data_point
            assert "total_assessments" in data_point
            assert isinstance(data_point["compliance_percentage"], (int, float))
    
    def test_control_automation_coverage(self):
        """Test automation coverage across controls"""
        all_controls = self.reporter.control_library.controls
        
        automation_stats = {
            "total_controls": len(all_controls),
            "automatable": 0,
            "manual_only": 0,
            "by_category": {}
        }
        
        for control in all_controls:
            if control.automation_possible:
                automation_stats["automatable"] += 1
            else:
                automation_stats["manual_only"] += 1
            
            category = control.category.value
            if category not in automation_stats["by_category"]:
                automation_stats["by_category"][category] = {
                    "total": 0,
                    "automatable": 0
                }
            
            automation_stats["by_category"][category]["total"] += 1
            if control.automation_possible:
                automation_stats["by_category"][category]["automatable"] += 1
        
        # Verify automation coverage
        assert automation_stats["total_controls"] > 0
        assert automation_stats["automatable"] + automation_stats["manual_only"] == automation_stats["total_controls"]
        
        # Check that some controls are automatable
        automation_percentage = (automation_stats["automatable"] / automation_stats["total_controls"]) * 100
        assert automation_percentage > 0  # Should have some automation
    
    def test_risk_based_prioritization(self):
        """Test risk-based control prioritization"""
        all_controls = self.reporter.control_library.controls
        
        # Group controls by risk level
        risk_groups = {}
        for control in all_controls:
            risk_level = control.risk_level
            if risk_level not in risk_groups:
                risk_groups[risk_level] = []
            risk_groups[risk_level].append(control)
        
        # Priority order: critical > high > medium > low
        priority_order = ["critical", "high", "medium", "low"]
        
        prioritized_controls = []
        for risk_level in priority_order:
            if risk_level in risk_groups:
                prioritized_controls.extend(risk_groups[risk_level])
        
        # Verify prioritization
        assert len(prioritized_controls) > 0
        
        # First controls should be higher risk
        if len(prioritized_controls) >= 2:
            first_control_risk = prioritized_controls[0].risk_level
            last_control_risk = prioritized_controls[-1].risk_level
            
            # Map risk levels to numeric values for comparison
            risk_values = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            
            first_value = risk_values.get(first_control_risk, 0)
            last_value = risk_values.get(last_control_risk, 0)
            
            assert first_value >= last_value


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestComplianceControlLibraryAdvanced:
    """Advanced tests for ComplianceControlLibrary with comprehensive coverage"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.library = ComplianceControlLibrary()
    
    def test_search_controls_functionality(self):
        """Test comprehensive search controls functionality"""
        # Test basic search
        access_controls = self.library.search_controls("access")
        assert len(access_controls) > 0
        
        # Test case insensitive search
        case_insensitive = self.library.search_controls("ACCESS")
        assert len(case_insensitive) > 0
        
        # Test search in requirements
        requirement_search = self.library.search_controls("role-based")
        assert isinstance(requirement_search, list)
        
        # Test search in implementation guidance
        guidance_search = self.library.search_controls("implement")
        assert isinstance(guidance_search, list)
        
        # Test empty query
        empty_results = self.library.search_controls("")
        assert empty_results == []
        
        # Test no matches
        no_matches = self.library.search_controls("xyznonexistent")
        assert no_matches == []
    
    def test_load_specific_framework_controls(self):
        """Test loading specific framework controls"""
        # Test SOC 2 controls with specific validation
        soc2_controls = self.library._load_soc2_controls()
        assert len(soc2_controls) >= 3
        
        cc61_control = next((c for c in soc2_controls if c.control_id == "CC6.1"), None)
        assert cc61_control is not None
        assert cc61_control.framework == ComplianceFramework.SOC2
        assert cc61_control.category == ControlCategory.ACCESS_CONTROL
        assert cc61_control.automation_possible is True
        assert cc61_control.risk_level == "high"
        
        # Test GDPR controls
        gdpr_controls = self.library._load_gdpr_controls()
        assert len(gdpr_controls) >= 2
        
        gdpr25_control = next((c for c in gdpr_controls if c.control_id == "GDPR.25"), None)
        assert gdpr25_control is not None
        assert gdpr25_control.framework == ComplianceFramework.GDPR
        assert gdpr25_control.category == ControlCategory.DATA_PROTECTION
        
        # Test OWASP controls
        owasp_controls = self.library._load_owasp_controls()
        assert len(owasp_controls) >= 2
        
        a01_control = next((c for c in owasp_controls if c.control_id == "OWASP.A01"), None)
        assert a01_control is not None
        assert a01_control.risk_level == "critical"
        
        # Test PCI DSS controls
        pci_controls = self.library._load_pci_dss_controls()
        assert len(pci_controls) >= 2
        
        pci34_control = next((c for c in pci_controls if c.control_id == "PCI.3.4"), None)
        assert pci34_control is not None
        assert pci34_control.category == ControlCategory.ENCRYPTION
        
        # Test ISO 27001 controls
        iso_controls = self.library._load_iso27001_controls()
        assert len(iso_controls) >= 2
        
        a912_control = next((c for c in iso_controls if c.control_id == "A.9.1.2"), None)
        assert a912_control is not None
        assert a912_control.automation_possible is True


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestAutomatedComplianceAssessorAdvanced:
    """Advanced tests for AutomatedComplianceAssessor with comprehensive coverage"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.assessor = AutomatedComplianceAssessor()
        self.library = ComplianceControlLibrary()
    
    def test_assess_control_automated_vs_manual(self):
        """Test assessment differences between automated and manual controls"""
        # Create automated control
        automated_control = ComplianceControl(
            control_id="AUTO-001",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.ACCESS_CONTROL,
            title="Automated Control",
            description="Automated assessment control",
            automation_possible=True,
            risk_level="medium"
        )
        
        # Create manual control
        manual_control = ComplianceControl(
            control_id="MANUAL-001",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.GOVERNANCE,
            title="Manual Control", 
            description="Manual assessment control",
            automation_possible=False,
            risk_level="medium"
        )
        
        evidence = [
            ComplianceEvidence(
                evidence_id="test_evidence",
                control_id="AUTO-001",
                evidence_type="configuration",
                title="Test Evidence",
                description="Test evidence"
            )
        ]
        
        # Test automated assessment
        auto_assessment = self.assessor.assess_control(automated_control, evidence)
        assert auto_assessment.status != ComplianceStatus.UNKNOWN
        assert auto_assessment.score is not None
        
        # Test manual assessment
        manual_assessment = self.assessor.assess_control(manual_control, evidence)
        assert manual_assessment.status == ComplianceStatus.UNKNOWN
        assert manual_assessment.score is None
        assert "Manual assessment required" in manual_assessment.findings
    
    def test_assessment_category_specific_logic(self):
        """Test category-specific assessment logic"""
        # Test access control assessment with comprehensive evidence
        access_control = self.library.get_control("CC6.1")
        if access_control:
            rbac_evidence = ComplianceEvidence(
                "rbac_001", "CC6.1", "config", "RBAC Configuration", "Role-based access control"
            )
            review_evidence = ComplianceEvidence(
                "review_001", "CC6.1", "audit", "Access Review Process", "Regular access reviews"
            )
            provision_evidence = ComplianceEvidence(
                "provision_001", "CC6.1", "doc", "User Provisioning Process", "User provisioning workflow"
            )
            
            complete_evidence = [rbac_evidence, review_evidence, provision_evidence]
            status, score, findings = self.assessor._assess_access_control(access_control, complete_evidence)
            
            assert status == ComplianceStatus.COMPLIANT
            assert score >= 90
            assert len(findings) == 0
            
            # Test with incomplete evidence
            incomplete_evidence = [rbac_evidence]
            status, score, findings = self.assessor._assess_access_control(access_control, incomplete_evidence)
            
            assert status in [ComplianceStatus.PARTIALLY_COMPLIANT, ComplianceStatus.NON_COMPLIANT]
            assert score < 90
            assert len(findings) > 0
    
    def test_encryption_assessment_detailed(self):
        """Test detailed encryption assessment logic"""
        encryption_control = self.library.get_control("GDPR.32")
        if encryption_control:
            # Test with all encryption evidence
            rest_evidence = ComplianceEvidence(
                "rest_001", "GDPR.32", "config", "Encryption at Rest Configuration", "Database encryption at rest"
            )
            transit_evidence = ComplianceEvidence(
                "tls_001", "GDPR.32", "config", "TLS Configuration", "SSL/TLS transport encryption"
            )
            key_evidence = ComplianceEvidence(
                "key_001", "GDPR.32", "doc", "Key Management Policy", "Cryptographic key management procedures"
            )
            
            complete_evidence = [rest_evidence, transit_evidence, key_evidence]
            status, score, findings = self.assessor._assess_encryption(encryption_control, complete_evidence)
            
            assert status == ComplianceStatus.COMPLIANT
            assert score >= 90
            
            # Test missing key management
            no_key_mgmt = [rest_evidence, transit_evidence]
            status, score, findings = self.assessor._assess_encryption(encryption_control, no_key_mgmt)
            assert "Key management procedures not documented" in findings
            assert score < 90
            
            # Test missing transit encryption
            no_transit = [rest_evidence, key_evidence]
            status, score, findings = self.assessor._assess_encryption(encryption_control, no_transit)
            assert "Encryption in transit (TLS/SSL) not documented" in findings
    
    def test_logging_assessment_detailed(self):
        """Test detailed logging assessment logic"""
        logging_control = self.library.get_control("CC7.2")
        if logging_control:
            # Test complete logging evidence
            log_config_evidence = ComplianceEvidence(
                "log_001", "CC7.2", "config", "Logging Configuration", "System logging setup"
            )
            monitoring_evidence = ComplianceEvidence(
                "monitor_001", "CC7.2", "config", "Monitoring and Alerting", "Alert configuration"
            )
            retention_evidence = ComplianceEvidence(
                "retention_001", "CC7.2", "doc", "Log Retention Policy", "Log retention procedures"
            )
            
            complete_evidence = [log_config_evidence, monitoring_evidence, retention_evidence]
            status, score, findings = self.assessor._assess_logging(logging_control, complete_evidence)
            
            assert status == ComplianceStatus.COMPLIANT
            assert score >= 85
            
            # Test without retention policy
            no_retention = [log_config_evidence, monitoring_evidence]
            status, score, findings = self.assessor._assess_logging(logging_control, no_retention)
            assert "Log retention policy not documented" in findings
    
    def test_authentication_assessment_detailed(self):
        """Test detailed authentication assessment logic"""
        auth_control = self.library.get_control("PCI.8.2")
        if auth_control:
            # Test complete authentication evidence
            mfa_evidence = ComplianceEvidence(
                "mfa_001", "PCI.8.2", "config", "Multi-Factor Authentication", "MFA implementation"
            )
            password_evidence = ComplianceEvidence(
                "pwd_001", "PCI.8.2", "doc", "Password Policy Document", "Strong password policy"
            )
            lockout_evidence = ComplianceEvidence(
                "lockout_001", "PCI.8.2", "config", "Account Lockout Configuration", "Account lockout mechanism"
            )
            
            complete_evidence = [mfa_evidence, password_evidence, lockout_evidence]
            status, score, findings = self.assessor._assess_authentication(auth_control, complete_evidence)
            
            assert status == ComplianceStatus.COMPLIANT
            assert score >= 85
            
            # Test without MFA (critical component)
            no_mfa = [password_evidence, lockout_evidence]
            status, score, findings = self.assessor._assess_authentication(auth_control, no_mfa)
            assert "Multi-factor authentication not documented" in findings
            assert score <= 60  # Should drop significantly without MFA
    
    def test_network_security_assessment_detailed(self):
        """Test detailed network security assessment logic"""
        network_control = self.library.get_control("A.9.1.2")
        if network_control:
            # Test complete network security evidence
            firewall_evidence = ComplianceEvidence(
                "fw_001", "A.9.1.2", "config", "Firewall Configuration", "Network firewall rules"
            )
            segment_evidence = ComplianceEvidence(
                "seg_001", "A.9.1.2", "config", "Network Segmentation", "VLAN segmentation"
            )
            ids_evidence = ComplianceEvidence(
                "ids_001", "A.9.1.2", "config", "Intrusion Detection System", "IDS monitoring"
            )
            
            complete_evidence = [firewall_evidence, segment_evidence, ids_evidence]
            status, score, findings = self.assessor._assess_network_security(network_control, complete_evidence)
            
            assert status == ComplianceStatus.COMPLIANT
            assert score >= 80
            
            # Test with only firewall
            minimal_evidence = [firewall_evidence]
            status, score, findings = self.assessor._assess_network_security(network_control, minimal_evidence)
            assert "Network segmentation not documented" in findings
            assert "Intrusion detection system not documented" in findings
            assert score < 80
    
    def test_assessor_registry_functionality(self):
        """Test assessor registry and method mapping"""
        # Verify all assessor methods are registered
        expected_assessors = {
            "access_control": self.assessor._assess_access_control,
            "encryption": self.assessor._assess_encryption,
            "logging": self.assessor._assess_logging,
            "authentication": self.assessor._assess_authentication,
            "network_security": self.assessor._assess_network_security
        }
        
        for assessor_type, method in expected_assessors.items():
            assert assessor_type in self.assessor.assessors
            assert self.assessor.assessors[assessor_type] == method
            assert callable(method)
    
    def test_assessment_id_generation(self):
        """Test assessment ID generation consistency"""
        control = ComplianceControl(
            control_id="TEST-001",
            framework=ComplianceFramework.SOC2,
            category=ControlCategory.ACCESS_CONTROL,
            title="Test Control",
            description="Test control"
        )
        
        # Generate multiple assessments
        assessment1 = self.assessor.assess_control(control)
        assessment2 = self.assessor.assess_control(control)
        
        # Assessment IDs should be unique
        assert assessment1.assessment_id != assessment2.assessment_id
        
        # Both should start with "assess_"
        assert assessment1.assessment_id.startswith("assess_TEST-001_")
        assert assessment2.assessment_id.startswith("assess_TEST-001_")


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestEvidenceCollectorAdvanced:
    """Advanced tests for EvidenceCollector with comprehensive coverage"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = EvidenceCollector(evidence_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_collect_configuration_evidence_detailed(self):
        """Test detailed configuration evidence collection"""
        # Create test configuration file
        config_file = os.path.join(self.temp_dir, "security.conf")
        config_content = """
# Security Configuration
security.enabled=true
encryption.algorithm=AES-256
logging.level=INFO
authentication.mfa=required
        """.strip()
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        evidence = self.collector.collect_configuration_evidence(
            config_path=config_file,
            control_id="SEC-001",
            title="Security Configuration File"
        )
        
        assert evidence is not None
        assert evidence.evidence_id.startswith("config_")
        assert evidence.control_id == "SEC-001"
        assert evidence.evidence_type == "configuration"
        assert evidence.title == "Security Configuration File"
        assert evidence.description == f"Configuration file: {config_file}"
        assert evidence.file_path == config_file
        assert evidence.content == config_content
        assert evidence.metadata["file_size"] == len(config_content)
        assert evidence.metadata["file_type"] == "config"
        assert isinstance(evidence.collected_at, datetime)
        assert evidence.collected_by == "system"
        
        # Verify evidence is in store
        stored = self.collector.get_evidence_for_control("SEC-001")
        assert len(stored) == 1
        assert stored[0].evidence_id == evidence.evidence_id
    
    def test_collect_log_evidence_detailed(self):
        """Test detailed log evidence collection"""
        log_data = """
2023-12-01 10:00:00 INFO User john.doe logged in successfully
2023-12-01 10:01:15 WARN Failed login attempt for user admin from 192.168.1.100
2023-12-01 10:02:30 INFO User jane.smith accessed restricted resource /admin/users
2023-12-01 10:03:45 ERROR Database connection failed - retrying
2023-12-01 10:04:00 INFO System backup completed successfully
        """.strip()
        
        evidence = self.collector.collect_log_evidence(
            log_data=log_data,
            control_id="LOG-001",
            title="Authentication and Access Logs",
            description="System authentication and access control logs"
        )
        
        assert evidence is not None
        assert evidence.evidence_id.startswith("log_")
        assert evidence.control_id == "LOG-001"
        assert evidence.evidence_type == "log"
        assert evidence.title == "Authentication and Access Logs"
        assert evidence.description == "System authentication and access control logs"
        assert evidence.content == log_data
        assert evidence.file_path is None
        assert evidence.metadata["log_lines"] == 5
        assert isinstance(evidence.collected_at, datetime)
        assert evidence.collected_by == "system"
    
    def test_collect_test_result_evidence_detailed(self):
        """Test detailed test result evidence collection"""
        test_results = {
            "encryption_tests": {
                "algorithm_strength": {"status": "passed", "score": 95, "details": "AES-256 encryption verified"},
                "key_rotation": {"status": "passed", "score": 88, "details": "Key rotation policy compliant"},
                "certificate_validation": {"status": "failed", "score": 65, "details": "Certificate expiring soon"}
            },
            "access_control_tests": {
                "rbac_implementation": {"status": "passed", "score": 92, "details": "RBAC properly configured"},
                "privilege_escalation": {"status": "passed", "score": 90, "details": "No privilege escalation vulnerabilities"},
                "session_management": {"status": "passed", "score": 85, "details": "Session timeout configured"}
            },
            "summary": {
                "total_tests": 6,
                "passed": 5,
                "failed": 1,
                "overall_score": 85.8
            }
        }
        
        evidence = self.collector.collect_test_result_evidence(
            test_results=test_results,
            control_id="TEST-001",
            title="Comprehensive Security Test Suite"
        )
        
        assert evidence is not None
        assert evidence.evidence_id.startswith("test_")
        assert evidence.control_id == "TEST-001"
        assert evidence.evidence_type == "test_result"
        assert evidence.title == "Comprehensive Security Test Suite"
        assert evidence.description == "Automated test results"
        assert evidence.metadata["test_count"] == len(test_results)
        assert evidence.metadata["test_type"] == "automated"
        
        # Verify content is valid JSON and matches input
        parsed_content = json.loads(evidence.content)
        assert parsed_content == test_results
        assert parsed_content["summary"]["total_tests"] == 6
        assert parsed_content["summary"]["overall_score"] == 85.8
    
    def test_collect_document_evidence_detailed(self):
        """Test detailed document evidence collection"""
        # Create test document
        doc_file = os.path.join(self.temp_dir, "incident_response_plan.md")
        doc_content = """
# Incident Response Plan

## Overview
This document outlines the organization's incident response procedures.

## Incident Classification
- Critical: System compromise, data breach
- High: Service disruption, security vulnerability
- Medium: Policy violation, suspicious activity
- Low: General security inquiry

## Response Team Contacts
- Security Lead: security@company.com
- IT Manager: it@company.com
- Legal Counsel: legal@company.com

## Response Procedures
1. Detection and Analysis
2. Containment, Eradication, and Recovery
3. Post-Incident Activity
        """.strip()
        
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        
        evidence = self.collector.collect_document_evidence(
            document_path=doc_file,
            control_id="DOC-001",
            title="Incident Response Plan",
            description="Organization incident response procedures and contacts"
        )
        
        assert evidence is not None
        assert evidence.evidence_id.startswith("doc_")
        assert evidence.control_id == "DOC-001"
        assert evidence.evidence_type == "document"
        assert evidence.title == "Incident Response Plan"
        assert evidence.description == "Organization incident response procedures and contacts"
        assert evidence.file_path == doc_file
        assert evidence.content == doc_content
        assert evidence.metadata["document_type"] == "policy"
        assert isinstance(evidence.collected_at, datetime)
        assert evidence.collected_by == "system"
    
    def test_evidence_store_thread_safety(self):
        """Test evidence store thread safety with concurrent operations"""
        import threading
        import time
        
        def collect_evidence_worker(worker_id):
            for i in range(5):
                # Collect different types of evidence
                log_evidence = self.collector.collect_log_evidence(
                    f"Worker {worker_id} log entry {i}",
                    f"WORKER-{worker_id:03d}",
                    f"Worker {worker_id} Log {i}",
                    f"Log from worker {worker_id}"
                )
                
                test_evidence = self.collector.collect_test_result_evidence(
                    {"test": f"result_{worker_id}_{i}", "score": 80 + i},
                    f"WORKER-{worker_id:03d}",
                    f"Worker {worker_id} Test {i}"
                )
                
                time.sleep(0.001)  # Small delay to increase race condition chances
        
        # Start multiple worker threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=collect_evidence_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all evidence was collected properly
        all_evidence = self.collector.get_all_evidence()
        assert len(all_evidence) >= 30  # 3 workers * 5 iterations * 2 evidence types
        
        # Verify no duplicate evidence IDs
        evidence_ids = [e.evidence_id for e in all_evidence]
        assert len(evidence_ids) == len(set(evidence_ids))
        
        # Verify evidence integrity
        for evidence in all_evidence:
            assert evidence.evidence_id
            assert evidence.control_id
            assert evidence.evidence_type
            assert evidence.title
            assert isinstance(evidence.collected_at, datetime)
    
    def test_evidence_collection_error_handling(self):
        """Test error handling in evidence collection"""
        # Test configuration evidence with non-existent file
        with pytest.raises(Exception):
            self.collector.collect_configuration_evidence(
                config_path="/non/existent/path.conf",
                control_id="ERROR-001",
                title="Missing Config"
            )
        
        # Test document evidence with non-existent file
        with pytest.raises(Exception):
            self.collector.collect_document_evidence(
                document_path="/non/existent/document.md",
                control_id="ERROR-002",
                title="Missing Document",
                description="Non-existent document"
            )
        
        # Verify that failed collections don't corrupt the evidence store
        initial_count = len(self.collector.get_all_evidence())
        
        # Successful collection should still work
        success_evidence = self.collector.collect_log_evidence(
            "Test log after error",
            "SUCCESS-001",
            "Success Evidence",
            "Evidence collected successfully after error"
        )
        
        assert success_evidence is not None
        assert len(self.collector.get_all_evidence()) == initial_count + 1


@pytest.mark.skipif(not COMPLIANCE_AVAILABLE, reason="Compliance reporter not available")
class TestComplianceReporterAdvanced:
    """Advanced tests for ComplianceReporter with comprehensive coverage"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.reporter = ComplianceReporter(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_generate_compliance_report_comprehensive(self):
        """Test comprehensive compliance report generation with all features"""
        # Add comprehensive evidence for better assessment
        self._add_comprehensive_evidence()
        
        # Generate report with custom parameters
        report = self.reporter.generate_compliance_report(
            framework=ComplianceFramework.SOC2,
            report_type="full",
            period_days=90
        )
        
        # Verify basic report structure
        assert report is not None
        assert report.framework == ComplianceFramework.SOC2
        assert report.report_type == "full"
        assert len(report.assessments) >= 3  # At least SOC2 controls
        assert 0 <= report.compliance_percentage <= 100
        assert isinstance(report.overall_score, (float, type(None)))
        
        # Verify time period
        time_diff = report.reporting_period_end - report.reporting_period_start
        assert time_diff.days == 90
        
        # Verify summary structure
        assert "total_controls" in report.summary
        assert "compliant_controls" in report.summary
        assert "status_breakdown" in report.summary
        assert "high_risk_controls" in report.summary
        assert "critical_risk_controls" in report.summary
        
        # Verify recommendations
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0
        
        # Verify report is stored
        assert report in self.reporter.reports
    
    def test_export_formats_comprehensive(self):
        """Test all export formats comprehensively"""
        report = self.reporter.generate_compliance_report(ComplianceFramework.GDPR)
        
        # Test JSON export
        json_output = self.reporter.export_report(report, format="json")
        assert isinstance(json_output, str)
        
        parsed_json = json.loads(json_output)
        assert "report_metadata" in parsed_json
        assert "compliance_summary" in parsed_json
        assert "assessments" in parsed_json
        assert "recommendations" in parsed_json
        
        # Verify JSON structure
        metadata = parsed_json["report_metadata"]
        assert metadata["framework"] == "gdpr"
        assert metadata["report_type"] == "full"
        assert "reporting_period" in metadata
        
        summary = parsed_json["compliance_summary"]
        assert "overall_score" in summary
        assert "compliance_percentage" in summary
        assert "summary" in summary
        
        # Test CSV export
        csv_output = self.reporter.export_report(report, format="csv")
        assert isinstance(csv_output, str)
        
        csv_lines = csv_output.strip().split('\n')
        assert len(csv_lines) >= 2  # Header + at least one data row
        
        header = csv_lines[0].split(',')
        expected_csv_headers = [
            "Control ID", "Status", "Score", "Risk Rating",
            "Findings Count", "Assessment Date", "Next Due"
        ]
        assert header == expected_csv_headers
        
        # Test HTML export
        html_output = self.reporter.export_report(report, format="html")
        assert isinstance(html_output, str)
        assert "<!DOCTYPE html>" in html_output
        assert "<html>" in html_output
        assert "</html>" in html_output
        
        # Verify HTML contains report data
        assert report.framework.value.upper() in html_output
        assert report.report_id in html_output
        assert f"{report.compliance_percentage:.1f}%" in html_output
        
        # Verify CSS styling
        assert "font-family: Arial" in html_output
        assert ".compliance-score" in html_output
        assert ".controls-table" in html_output
        assert ".recommendations" in html_output
        
        # Verify table structure
        assert "<table class=\"controls-table\">" in html_output
        assert "<th>Control ID</th>" in html_output
        assert "<th>Status</th>" in html_output
        assert "<th>Score</th>" in html_output
    
    def test_save_report_formats(self):
        """Test saving reports in all formats"""
        report = self.reporter.generate_compliance_report(ComplianceFramework.OWASP)
        
        # Test JSON save
        json_path = self.reporter.save_report(report, format="json")
        assert os.path.exists(json_path)
        assert json_path.endswith(".json")
        assert report.report_id in json_path
        assert "owasp" in json_path
        
        with open(json_path, 'r') as f:
            json_content = json.load(f)
            assert json_content["report_metadata"]["framework"] == "owasp"
        
        # Test CSV save
        csv_path = self.reporter.save_report(report, format="csv")
        assert os.path.exists(csv_path)
        assert csv_path.endswith(".csv")
        
        with open(csv_path, 'r') as f:
            csv_content = f.read()
            assert "Control ID,Status" in csv_content
        
        # Test HTML save
        html_path = self.reporter.save_report(report, format="html")
        assert os.path.exists(html_path)
        assert html_path.endswith(".html")
        
        with open(html_path, 'r') as f:
            html_content = f.read()
            assert "<!DOCTYPE html>" in html_content
            assert "OWASP" in html_content
    
    def test_recommendations_generation_detailed(self):
        """Test detailed recommendation generation logic"""
        # Create mock assessments with various statuses and risk levels
        assessments = [
            ComplianceAssessment(
                assessment_id="critical_fail_1",
                control_id="CRIT-001",
                status=ComplianceStatus.NON_COMPLIANT,
                risk_rating="critical"
            ),
            ComplianceAssessment(
                assessment_id="critical_fail_2",
                control_id="CRIT-002", 
                status=ComplianceStatus.NON_COMPLIANT,
                risk_rating="critical"
            ),
            ComplianceAssessment(
                assessment_id="partial_1",
                control_id="PART-001",
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                risk_rating="high"
            ),
            ComplianceAssessment(
                assessment_id="partial_2",
                control_id="PART-002",
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                risk_rating="medium"
            ),
            ComplianceAssessment(
                assessment_id="compliant_1",
                control_id="COMP-001",
                status=ComplianceStatus.COMPLIANT,
                risk_rating="low"
            )
        ]
        
        controls = [
            ComplianceControl(
                control_id="CRIT-001",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.ACCESS_CONTROL,
                title="Critical Access Control",
                description="Critical access control"
            ),
            ComplianceControl(
                control_id="CRIT-002",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.ACCESS_CONTROL, 
                title="Critical Access Control 2",
                description="Another critical access control"
            ),
            ComplianceControl(
                control_id="PART-001",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.ENCRYPTION,
                title="Partial Encryption Control",
                description="Partially compliant encryption"
            ),
            ComplianceControl(
                control_id="PART-002",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.LOGGING_MONITORING,
                title="Partial Logging Control",
                description="Partially compliant logging"
            ),
            ComplianceControl(
                control_id="COMP-001",
                framework=ComplianceFramework.SOC2,
                category=ControlCategory.AUTHENTICATION,
                title="Compliant Auth Control",
                description="Compliant authentication"
            )
        ]
        
        recommendations = self.reporter._generate_recommendations(assessments, controls)
        
        # Verify recommendation structure
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include urgent recommendation for critical controls
        urgent_recs = [r for r in recommendations if "URGENT" in r and "critical" in r]
        assert len(urgent_recs) > 0
        
        # Should mention "2 critical controls"
        critical_mention = any("2 critical controls" in r for r in recommendations)
        assert critical_mention
        
        # Should include partial compliance recommendation
        partial_recs = [r for r in recommendations if "partially compliant" in r]
        assert len(partial_recs) > 0
        
        # Should mention "2 controls are partially compliant"
        partial_mention = any("2 controls are partially compliant" in r for r in recommendations)
        assert partial_mention
        
        # Should include category-specific recommendations for access control (2 issues)
        access_recs = [r for r in recommendations if "access control" in r and "2 issues" in r]
        assert len(access_recs) > 0
        
        # Should include overall compliance recommendation
        # With 1 compliant out of 5 total = 20% compliance
        overall_recs = [r for r in recommendations if "below 70%" in r or "comprehensive" in r]
        assert len(overall_recs) > 0
    
    def test_dashboard_data_generation_detailed(self):
        """Test detailed dashboard data generation"""
        # Generate reports for multiple frameworks
        frameworks = [
            ComplianceFramework.SOC2,
            ComplianceFramework.GDPR,
            ComplianceFramework.OWASP,
            ComplianceFramework.PCI_DSS
        ]
        
        for framework in frameworks:
            self.reporter.generate_compliance_report(framework)
        
        # Test get_compliance_dashboard_data
        dashboard_data = self.reporter.get_compliance_dashboard_data()
        
        assert "last_updated" in dashboard_data
        assert "frameworks" in dashboard_data
        
        frameworks_data = dashboard_data["frameworks"]
        assert len(frameworks_data) == len(frameworks)
        
        for framework in frameworks:
            framework_key = framework.value
            assert framework_key in frameworks_data
            
            framework_data = frameworks_data[framework_key]
            assert "compliance_percentage" in framework_data
            assert "overall_score" in framework_data
            assert "total_controls" in framework_data
            assert "compliant_controls" in framework_data
            assert "last_assessment" in framework_data
            assert "status_breakdown" in framework_data
            assert "high_priority_issues" in framework_data
            
            # Verify data types and ranges
            assert isinstance(framework_data["compliance_percentage"], (int, float))
            assert 0 <= framework_data["compliance_percentage"] <= 100
            assert framework_data["total_controls"] > 0
            assert framework_data["compliant_controls"] >= 0
            assert isinstance(framework_data["status_breakdown"], dict)
            assert framework_data["high_priority_issues"] >= 0
        
        # Test generate_dashboard_data
        dashboard_summary = self.reporter.generate_dashboard_data()
        
        assert "overall_stats" in dashboard_summary
        assert "frameworks" in dashboard_summary
        
        overall_stats = dashboard_summary["overall_stats"]
        assert overall_stats["total_frameworks"] == len(frameworks)
        assert 0 <= overall_stats["avg_compliance"] <= 100
        assert overall_stats["total_controls"] > 0
        assert "last_updated" in overall_stats
        
        # Verify framework data consistency
        assert dashboard_summary["frameworks"] == frameworks_data
    
    def test_report_thread_safety(self):
        """Test thread safety of report generation and storage"""
        import threading
        
        results = []
        errors = []
        
        def generate_report_worker(framework):
            try:
                report = self.reporter.generate_compliance_report(framework)
                results.append((framework.value, report.report_id))
            except Exception as e:
                errors.append(str(e))
        
        # Generate reports concurrently
        frameworks = [
            ComplianceFramework.SOC2,
            ComplianceFramework.GDPR,
            ComplianceFramework.OWASP,
            ComplianceFramework.PCI_DSS,
            ComplianceFramework.ISO_27001
        ]
        
        threads = []
        for framework in frameworks:
            thread = threading.Thread(target=generate_report_worker, args=(framework,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Verify all reports generated
        assert len(results) == len(frameworks)
        
        # Verify all report IDs are unique
        report_ids = [result[1] for result in results]
        assert len(report_ids) == len(set(report_ids))
        
        # Verify all frameworks represented
        framework_values = [result[0] for result in results]
        expected_values = [f.value for f in frameworks]
        assert set(framework_values) == set(expected_values)
        
        # Verify reports are stored correctly
        assert len(self.reporter.reports) == len(frameworks)
    
    def _add_comprehensive_evidence(self):
        """Helper method to add comprehensive evidence for testing"""
        evidence_collector = self.reporter.evidence_collector
        
        # Add evidence for SOC2 controls
        evidence_collector.collect_log_evidence(
            "2023-01-01 10:00:00 INFO: RBAC system active, role assignments verified",
            "CC6.1",
            "RBAC Implementation Logs",
            "Role-based access control implementation evidence"
        )
        
        evidence_collector.collect_test_result_evidence(
            {
                "access_control_test": {"status": "passed", "score": 92},
                "role_validation": {"status": "passed", "score": 88},
                "privilege_verification": {"status": "passed", "score": 90}
            },
            "CC6.1",
            "Access Control Test Results"
        )
        
        evidence_collector.collect_log_evidence(
            "2023-01-01 10:00:00 INFO: TLS 1.3 encryption active on all endpoints",
            "CC6.7",
            "Encryption Configuration Logs",
            "Data transmission encryption evidence"
        )
        
        evidence_collector.collect_log_evidence(
            "2023-01-01 10:00:00 INFO: System monitoring active, 0 alerts in past 24h",
            "CC7.2", 
            "System Monitoring Logs",
            "System monitoring and alerting evidence"
        )