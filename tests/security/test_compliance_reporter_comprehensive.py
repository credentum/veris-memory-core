#!/usr/bin/env python3
"""
Comprehensive tests for src/security/compliance_reporter.py

This test suite provides comprehensive coverage of the compliance reporting system,
testing all major functionality including control libraries, assessments, evidence
collection, and report generation.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call
from dataclasses import asdict

# Import the module under test
from src.security.compliance_reporter import (
    ComplianceFramework,
    ComplianceStatus,
    ControlCategory,
    EvidenceType,
    ComplianceControl,
    ComplianceEvidence,
    ComplianceAssessment,
    ComplianceReport,
    ComplianceControlLibrary,
    AutomatedComplianceAssessor,
    EvidenceCollector,
    ComplianceReporter
)


class TestEnumsAndDataClasses:
    """Test compliance enums and data classes."""

    def test_compliance_framework_enum(self):
        """Test ComplianceFramework enum values."""
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
        """Test ComplianceStatus enum values."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.NOT_APPLICABLE.value == "not_applicable"
        assert ComplianceStatus.UNKNOWN.value == "unknown"

    def test_control_category_enum(self):
        """Test ControlCategory enum values."""
        assert hasattr(ControlCategory, 'ACCESS_CONTROL')
        assert hasattr(ControlCategory, 'DATA_PROTECTION')
        assert hasattr(ControlCategory, 'NETWORK_SECURITY')

    def test_evidence_type_enum(self):
        """Test EvidenceType enum values."""
        assert hasattr(EvidenceType, 'DOCUMENTATION')
        assert hasattr(EvidenceType, 'LOG_FILE')
        assert hasattr(EvidenceType, 'SCREENSHOT')

    def test_compliance_control_creation(self):
        """Test ComplianceControl dataclass creation."""
        control = ComplianceControl(
            control_id="AC-1",
            title="Access Control Policy and Procedures",
            description="Test control",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2, ComplianceFramework.ISO_27001],
            severity="high",
            implementation_guidance="Test guidance",
            testing_procedures="Test procedures",
            references=["SOC2-CC6.1"]
        )
        
        assert control.control_id == "AC-1"
        assert control.title == "Access Control Policy and Procedures"
        assert control.category == ControlCategory.ACCESS_CONTROL
        assert len(control.frameworks) == 2
        assert ComplianceFramework.SOC2 in control.frameworks

    def test_compliance_evidence_creation(self):
        """Test ComplianceEvidence dataclass creation."""
        evidence = ComplianceEvidence(
            evidence_id="EV-001",
            control_id="AC-1",
            evidence_type=EvidenceType.DOCUMENTATION,
            title="Access Control Policy",
            description="Policy document",
            file_path="/path/to/policy.pdf",
            collected_date=datetime.now(),
            metadata={"version": "1.0"}
        )
        
        assert evidence.evidence_id == "EV-001"
        assert evidence.control_id == "AC-1"
        assert evidence.evidence_type == EvidenceType.DOCUMENTATION
        assert evidence.metadata["version"] == "1.0"

    def test_compliance_assessment_creation(self):
        """Test ComplianceAssessment dataclass creation."""
        assessment = ComplianceAssessment(
            assessment_id="ASSESS-001",
            control_id="AC-1",
            status=ComplianceStatus.COMPLIANT,
            assessed_date=datetime.now(),
            assessor="John Doe",
            findings="Control implemented correctly",
            recommendations="No recommendations",
            evidence_ids=["EV-001", "EV-002"],
            score=100.0,
            metadata={"risk_level": "low"}
        )
        
        assert assessment.assessment_id == "ASSESS-001"
        assert assessment.status == ComplianceStatus.COMPLIANT
        assert assessment.score == 100.0
        assert len(assessment.evidence_ids) == 2

    def test_compliance_report_creation(self):
        """Test ComplianceReport dataclass creation."""
        report = ComplianceReport(
            report_id="RPT-001",
            framework=ComplianceFramework.SOC2,
            organization="Test Org",
            assessment_period_start=datetime.now() - timedelta(days=365),
            assessment_period_end=datetime.now(),
            generated_date=datetime.now(),
            assessments=[],
            overall_score=95.5,
            compliant_controls=45,
            total_controls=50,
            non_compliant_controls=3,
            partially_compliant_controls=2,
            metadata={"version": "1.0"}
        )
        
        assert report.report_id == "RPT-001"
        assert report.framework == ComplianceFramework.SOC2
        assert report.overall_score == 95.5
        assert report.total_controls == 50


class TestComplianceControlLibrary:
    """Test compliance control library functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.library = ComplianceControlLibrary()

    def test_init(self):
        """Test ComplianceControlLibrary initialization."""
        library = ComplianceControlLibrary()
        assert isinstance(library.controls, dict)
        assert isinstance(library.framework_mappings, dict)

    def test_add_control(self):
        """Test adding a control to the library."""
        library = ComplianceControlLibrary()
        
        control = ComplianceControl(
            control_id="TEST-1",
            title="Test Control",
            description="Test description",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="medium",
            implementation_guidance="Test guidance",
            testing_procedures="Test procedures"
        )
        
        library.add_control(control)
        
        assert "TEST-1" in library.controls
        assert library.controls["TEST-1"] == control
        assert ComplianceFramework.SOC2 in library.framework_mappings
        assert "TEST-1" in library.framework_mappings[ComplianceFramework.SOC2]

    def test_get_controls_by_framework(self):
        """Test getting controls by framework."""
        library = ComplianceControlLibrary()
        
        soc2_control = ComplianceControl(
            control_id="SOC2-1",
            title="SOC2 Control",
            description="SOC2 description",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="high"
        )
        
        gdpr_control = ComplianceControl(
            control_id="GDPR-1",
            title="GDPR Control",
            description="GDPR description",
            category=ControlCategory.DATA_PROTECTION,
            frameworks=[ComplianceFramework.GDPR],
            severity="high"
        )
        
        library.add_control(soc2_control)
        library.add_control(gdpr_control)
        
        soc2_controls = library.get_controls_by_framework(ComplianceFramework.SOC2)
        assert len(soc2_controls) == 1
        assert soc2_controls[0].control_id == "SOC2-1"
        
        gdpr_controls = library.get_controls_by_framework(ComplianceFramework.GDPR)
        assert len(gdpr_controls) == 1
        assert gdpr_controls[0].control_id == "GDPR-1"

    def test_get_controls_by_category(self):
        """Test getting controls by category."""
        library = ComplianceControlLibrary()
        
        access_control = ComplianceControl(
            control_id="AC-1",
            title="Access Control",
            description="Access control description",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="high"
        )
        
        data_protection = ComplianceControl(
            control_id="DP-1",
            title="Data Protection",
            description="Data protection description",
            category=ControlCategory.DATA_PROTECTION,
            frameworks=[ComplianceFramework.GDPR],
            severity="high"
        )
        
        library.add_control(access_control)
        library.add_control(data_protection)
        
        ac_controls = library.get_controls_by_category(ControlCategory.ACCESS_CONTROL)
        assert len(ac_controls) == 1
        assert ac_controls[0].control_id == "AC-1"

    def test_load_from_file(self):
        """Test loading controls from file."""
        library = ComplianceControlLibrary()
        
        mock_controls_data = {
            "controls": [
                {
                    "control_id": "FILE-1",
                    "title": "File Control",
                    "description": "File control description",
                    "category": "ACCESS_CONTROL",
                    "frameworks": ["soc2"],
                    "severity": "high",
                    "implementation_guidance": "Implementation guidance",
                    "testing_procedures": "Testing procedures",
                    "references": ["REF-1"]
                }
            ]
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_controls_data))):
            library.load_from_file("/fake/path.json")
        
        assert "FILE-1" in library.controls
        assert library.controls["FILE-1"].title == "File Control"

    def test_save_to_file(self):
        """Test saving controls to file."""
        library = ComplianceControlLibrary()
        
        control = ComplianceControl(
            control_id="SAVE-1",
            title="Save Control",
            description="Save control description",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="medium"
        )
        
        library.add_control(control)
        
        with patch("builtins.open", mock_open()) as mock_file:
            library.save_to_file("/fake/path.json")
            
            mock_file.assert_called_once_with("/fake/path.json", "w", encoding="utf-8")
            # Verify that json.dump was called with control data
            written_data = mock_file().write.call_args_list
            assert len(written_data) > 0


class TestEvidenceCollector:
    """Test evidence collection functionality."""

    def test_init(self):
        """Test EvidenceCollector initialization."""
        collector = EvidenceCollector()
        assert isinstance(collector.evidence_store, dict)
        assert isinstance(collector.collection_strategies, dict)

    def test_register_collection_strategy(self):
        """Test registering collection strategies."""
        collector = EvidenceCollector()
        
        def mock_strategy(control_id, params):
            return []
        
        collector.register_collection_strategy(EvidenceType.LOG_FILE, mock_strategy)
        
        assert EvidenceType.LOG_FILE in collector.collection_strategies
        assert collector.collection_strategies[EvidenceType.LOG_FILE] == mock_strategy

    def test_collect_evidence(self):
        """Test evidence collection."""
        collector = EvidenceCollector()
        
        evidence = ComplianceEvidence(
            evidence_id="TEST-EV-1",
            control_id="TEST-1",
            evidence_type=EvidenceType.DOCUMENTATION,
            title="Test Evidence",
            description="Test evidence description",
            collected_date=datetime.now()
        )
        
        collector.collect_evidence("TEST-1", evidence)
        
        assert "TEST-1" in collector.evidence_store
        assert len(collector.evidence_store["TEST-1"]) == 1
        assert collector.evidence_store["TEST-1"][0] == evidence

    def test_get_evidence_for_control(self):
        """Test getting evidence for a control."""
        collector = EvidenceCollector()
        
        evidence1 = ComplianceEvidence(
            evidence_id="TEST-EV-1",
            control_id="TEST-1",
            evidence_type=EvidenceType.DOCUMENTATION,
            title="Test Evidence 1",
            description="Test evidence 1",
            collected_date=datetime.now()
        )
        
        evidence2 = ComplianceEvidence(
            evidence_id="TEST-EV-2",
            control_id="TEST-1",
            evidence_type=EvidenceType.LOG_FILE,
            title="Test Evidence 2",
            description="Test evidence 2",
            collected_date=datetime.now()
        )
        
        collector.collect_evidence("TEST-1", evidence1)
        collector.collect_evidence("TEST-1", evidence2)
        
        evidence_list = collector.get_evidence_for_control("TEST-1")
        assert len(evidence_list) == 2
        assert evidence1 in evidence_list
        assert evidence2 in evidence_list

    def test_generate_evidence_report(self):
        """Test generating evidence report."""
        collector = EvidenceCollector()
        
        evidence = ComplianceEvidence(
            evidence_id="REPORT-EV-1",
            control_id="REPORT-1",
            evidence_type=EvidenceType.DOCUMENTATION,
            title="Report Evidence",
            description="Report evidence description",
            collected_date=datetime.now(),
            metadata={"source": "automated"}
        )
        
        collector.collect_evidence("REPORT-1", evidence)
        
        report = collector.generate_evidence_report()
        
        assert "total_evidence" in report
        assert "evidence_by_type" in report
        assert "evidence_by_control" in report
        assert report["total_evidence"] == 1
        assert EvidenceType.DOCUMENTATION in report["evidence_by_type"]


class TestAutomatedComplianceAssessor:
    """Test automated compliance assessment functionality."""

    def test_init(self):
        """Test AutomatedComplianceAssessor initialization."""
        assessor = AutomatedComplianceAssessor()
        assert isinstance(assessor.assessment_rules, dict)
        assert isinstance(assessor.scoring_weights, dict)

    def test_register_assessment_rule(self):
        """Test registering assessment rules."""
        assessor = AutomatedComplianceAssessor()
        
        def mock_rule(control, evidence):
            return ComplianceStatus.COMPLIANT, 100.0, "Mock assessment"
        
        assessor.register_assessment_rule("TEST-RULE", mock_rule)
        
        assert "TEST-RULE" in assessor.assessment_rules
        assert assessor.assessment_rules["TEST-RULE"] == mock_rule

    def test_assess_control(self):
        """Test assessing a control."""
        assessor = AutomatedComplianceAssessor()
        
        control = ComplianceControl(
            control_id="ASSESS-1",
            title="Assessment Control",
            description="Control for assessment testing",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="high"
        )
        
        evidence = [
            ComplianceEvidence(
                evidence_id="ASSESS-EV-1",
                control_id="ASSESS-1",
                evidence_type=EvidenceType.DOCUMENTATION,
                title="Assessment Evidence",
                description="Evidence for assessment",
                collected_date=datetime.now()
            )
        ]
        
        assessment = assessor.assess_control(control, evidence, "Test Assessor")
        
        assert isinstance(assessment, ComplianceAssessment)
        assert assessment.control_id == "ASSESS-1"
        assert assessment.assessor == "Test Assessor"
        assert isinstance(assessment.status, ComplianceStatus)
        assert isinstance(assessment.score, (int, float))

    def test_calculate_compliance_score(self):
        """Test compliance score calculation."""
        assessor = AutomatedComplianceAssessor()
        
        assessments = [
            ComplianceAssessment(
                assessment_id="SCORE-1",
                control_id="CTRL-1",
                status=ComplianceStatus.COMPLIANT,
                assessed_date=datetime.now(),
                assessor="Test",
                score=100.0
            ),
            ComplianceAssessment(
                assessment_id="SCORE-2",
                control_id="CTRL-2",
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                assessed_date=datetime.now(),
                assessor="Test",
                score=75.0
            ),
            ComplianceAssessment(
                assessment_id="SCORE-3",
                control_id="CTRL-3",
                status=ComplianceStatus.NON_COMPLIANT,
                assessed_date=datetime.now(),
                assessor="Test",
                score=0.0
            )
        ]
        
        score = assessor.calculate_compliance_score(assessments)
        expected_score = (100.0 + 75.0 + 0.0) / 3
        assert score == expected_score

    def test_generate_recommendations(self):
        """Test generating recommendations."""
        assessor = AutomatedComplianceAssessor()
        
        non_compliant_assessment = ComplianceAssessment(
            assessment_id="REC-1",
            control_id="REC-CTRL-1",
            status=ComplianceStatus.NON_COMPLIANT,
            assessed_date=datetime.now(),
            assessor="Test",
            score=25.0,
            findings="Control implementation insufficient"
        )
        
        recommendations = assessor.generate_recommendations([non_compliant_assessment])
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should contain recommendations for non-compliant controls


class TestComplianceReporter:
    """Test compliance reporter main functionality."""

    def test_init(self):
        """Test ComplianceReporter initialization."""
        reporter = ComplianceReporter()
        
        assert isinstance(reporter.control_library, ComplianceControlLibrary)
        assert isinstance(reporter.evidence_collector, EvidenceCollector)
        assert isinstance(reporter.assessor, AutomatedComplianceAssessor)
        assert isinstance(reporter.reports, dict)

    def test_generate_report(self):
        """Test generating a compliance report."""
        reporter = ComplianceReporter()
        
        # Add a control
        control = ComplianceControl(
            control_id="RPT-1",
            title="Report Control",
            description="Control for report testing",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="high"
        )
        reporter.control_library.add_control(control)
        
        # Add evidence
        evidence = ComplianceEvidence(
            evidence_id="RPT-EV-1",
            control_id="RPT-1",
            evidence_type=EvidenceType.DOCUMENTATION,
            title="Report Evidence",
            description="Evidence for report",
            collected_date=datetime.now()
        )
        reporter.evidence_collector.collect_evidence("RPT-1", evidence)
        
        report = reporter.generate_report(
            framework=ComplianceFramework.SOC2,
            organization="Test Organization",
            assessor="Test Assessor"
        )
        
        assert isinstance(report, ComplianceReport)
        assert report.framework == ComplianceFramework.SOC2
        assert report.organization == "Test Organization"
        assert len(report.assessments) > 0

    def test_save_report(self):
        """Test saving a report to file."""
        reporter = ComplianceReporter()
        
        report = ComplianceReport(
            report_id="SAVE-RPT-1",
            framework=ComplianceFramework.SOC2,
            organization="Save Test Org",
            assessment_period_start=datetime.now() - timedelta(days=30),
            assessment_period_end=datetime.now(),
            generated_date=datetime.now(),
            assessments=[],
            overall_score=85.0,
            compliant_controls=17,
            total_controls=20,
            non_compliant_controls=2,
            partially_compliant_controls=1
        )
        
        with patch("builtins.open", mock_open()) as mock_file:
            reporter.save_report(report, "/fake/report.json")
            
            mock_file.assert_called_once_with("/fake/report.json", "w", encoding="utf-8")

    def test_load_report(self):
        """Test loading a report from file."""
        reporter = ComplianceReporter()
        
        mock_report_data = {
            "report_id": "LOAD-RPT-1",
            "framework": "soc2",
            "organization": "Load Test Org",
            "assessment_period_start": datetime.now().isoformat(),
            "assessment_period_end": datetime.now().isoformat(),
            "generated_date": datetime.now().isoformat(),
            "assessments": [],
            "overall_score": 90.0,
            "compliant_controls": 18,
            "total_controls": 20,
            "non_compliant_controls": 1,
            "partially_compliant_controls": 1,
            "metadata": {}
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_report_data, default=str))):
            report = reporter.load_report("/fake/report.json")
            
            assert isinstance(report, ComplianceReport)
            assert report.report_id == "LOAD-RPT-1"
            assert report.organization == "Load Test Org"


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""

    def test_full_compliance_assessment_workflow(self):
        """Test complete compliance assessment workflow."""
        reporter = ComplianceReporter()
        
        # Step 1: Add controls to library
        access_control = ComplianceControl(
            control_id="WORKFLOW-AC-1",
            title="Access Control Policy",
            description="Access control policy implementation",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="high",
            implementation_guidance="Implement access control policies",
            testing_procedures="Review access control documentation"
        )
        reporter.control_library.add_control(access_control)
        
        # Step 2: Collect evidence
        policy_evidence = ComplianceEvidence(
            evidence_id="WORKFLOW-EV-1",
            control_id="WORKFLOW-AC-1",
            evidence_type=EvidenceType.DOCUMENTATION,
            title="Access Control Policy Document",
            description="Policy document for access control",
            file_path="/policies/access_control.pdf",
            collected_date=datetime.now(),
            metadata={"version": "2.1", "approved_by": "CISO"}
        )
        reporter.evidence_collector.collect_evidence("WORKFLOW-AC-1", policy_evidence)
        
        # Step 3: Generate report
        report = reporter.generate_report(
            framework=ComplianceFramework.SOC2,
            organization="Workflow Test Organization",
            assessor="Integration Test Assessor"
        )
        
        # Verify the workflow
        assert isinstance(report, ComplianceReport)
        assert report.framework == ComplianceFramework.SOC2
        assert len(report.assessments) == 1
        assert report.assessments[0].control_id == "WORKFLOW-AC-1"
        assert report.total_controls == 1

    def test_multi_framework_compliance(self):
        """Test compliance assessment across multiple frameworks."""
        reporter = ComplianceReporter()
        
        # Control applicable to multiple frameworks
        data_protection_control = ComplianceControl(
            control_id="MULTI-DP-1",
            title="Data Protection Control",
            description="Data protection across frameworks",
            category=ControlCategory.DATA_PROTECTION,
            frameworks=[ComplianceFramework.SOC2, ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
            severity="critical",
            implementation_guidance="Implement comprehensive data protection",
            testing_procedures="Verify data protection controls"
        )
        reporter.control_library.add_control(data_protection_control)
        
        # Generate reports for different frameworks
        soc2_report = reporter.generate_report(
            framework=ComplianceFramework.SOC2,
            organization="Multi-Framework Org",
            assessor="Test Assessor"
        )
        
        gdpr_report = reporter.generate_report(
            framework=ComplianceFramework.GDPR,
            organization="Multi-Framework Org",
            assessor="Test Assessor"
        )
        
        # Both reports should include the control
        assert len(soc2_report.assessments) == 1
        assert len(gdpr_report.assessments) == 1
        assert soc2_report.assessments[0].control_id == "MULTI-DP-1"
        assert gdpr_report.assessments[0].control_id == "MULTI-DP-1"

    def test_evidence_aggregation_and_scoring(self):
        """Test evidence aggregation and compliance scoring."""
        reporter = ComplianceReporter()
        
        # Control with multiple evidence types
        security_control = ComplianceControl(
            control_id="AGG-SEC-1",
            title="Security Control",
            description="Security control with multiple evidence",
            category=ControlCategory.NETWORK_SECURITY,
            frameworks=[ComplianceFramework.NIST],
            severity="high"
        )
        reporter.control_library.add_control(security_control)
        
        # Multiple pieces of evidence
        doc_evidence = ComplianceEvidence(
            evidence_id="AGG-DOC-1",
            control_id="AGG-SEC-1",
            evidence_type=EvidenceType.DOCUMENTATION,
            title="Security Documentation",
            description="Security policy documentation",
            collected_date=datetime.now()
        )
        
        log_evidence = ComplianceEvidence(
            evidence_id="AGG-LOG-1",
            control_id="AGG-SEC-1",
            evidence_type=EvidenceType.LOG_FILE,
            title="Security Logs",
            description="Security monitoring logs",
            collected_date=datetime.now()
        )
        
        config_evidence = ComplianceEvidence(
            evidence_id="AGG-CONFIG-1",
            control_id="AGG-SEC-1",
            evidence_type=EvidenceType.CONFIGURATION,
            title="Security Configuration",
            description="Security system configuration",
            collected_date=datetime.now()
        )
        
        reporter.evidence_collector.collect_evidence("AGG-SEC-1", doc_evidence)
        reporter.evidence_collector.collect_evidence("AGG-SEC-1", log_evidence)
        reporter.evidence_collector.collect_evidence("AGG-SEC-1", config_evidence)
        
        # Generate report and verify evidence aggregation
        report = reporter.generate_report(
            framework=ComplianceFramework.NIST,
            organization="Evidence Aggregation Org",
            assessor="Test Assessor"
        )
        
        assert len(report.assessments) == 1
        assessment = report.assessments[0]
        assert len(assessment.evidence_ids) == 3


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_framework_handling(self):
        """Test handling of invalid framework."""
        reporter = ComplianceReporter()
        
        # This should handle gracefully
        try:
            report = reporter.generate_report(
                framework="invalid_framework",
                organization="Test Org",
                assessor="Test Assessor"
            )
            # If it doesn't raise an exception, verify it handles gracefully
            assert report is not None
        except (ValueError, TypeError, AttributeError):
            # Expected behavior for invalid framework
            pass

    def test_missing_evidence_handling(self):
        """Test handling when no evidence is available."""
        reporter = ComplianceReporter()
        
        # Add control without evidence
        control = ComplianceControl(
            control_id="NO-EVIDENCE-1",
            title="Control Without Evidence",
            description="Control with no evidence",
            category=ControlCategory.ACCESS_CONTROL,
            frameworks=[ComplianceFramework.SOC2],
            severity="medium"
        )
        reporter.control_library.add_control(control)
        
        # Generate report
        report = reporter.generate_report(
            framework=ComplianceFramework.SOC2,
            organization="No Evidence Org",
            assessor="Test Assessor"
        )
        
        # Should handle gracefully
        assert isinstance(report, ComplianceReport)
        assert len(report.assessments) == 1
        # Assessment should exist but may have unknown status

    def test_file_operation_errors(self):
        """Test handling of file operation errors."""
        reporter = ComplianceReporter()
        
        # Test file read error handling
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            try:
                reporter.control_library.load_from_file("/nonexistent/file.json")
            except FileNotFoundError:
                pass  # Expected behavior
            except Exception as e:
                # Should handle gracefully or raise appropriate exception
                assert isinstance(e, (IOError, OSError, ValueError))

    def test_concurrent_access_safety(self):
        """Test thread safety for concurrent operations."""
        import threading
        import time
        
        reporter = ComplianceReporter()
        results = []
        
        def add_controls_concurrently(thread_id):
            for i in range(5):
                control = ComplianceControl(
                    control_id=f"THREAD-{thread_id}-{i}",
                    title=f"Thread {thread_id} Control {i}",
                    description=f"Control from thread {thread_id}",
                    category=ControlCategory.ACCESS_CONTROL,
                    frameworks=[ComplianceFramework.SOC2],
                    severity="medium"
                )
                reporter.control_library.add_control(control)
                time.sleep(0.001)  # Small delay
        
        # Create multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=add_controls_concurrently, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all controls were added
        total_controls = len(reporter.control_library.controls)
        assert total_controls == 15  # 3 threads * 5 controls each


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])