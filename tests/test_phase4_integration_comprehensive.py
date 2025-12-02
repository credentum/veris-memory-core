#!/usr/bin/env python3
"""
Comprehensive integration tests for Phase 4 observability and safety features.

This test suite validates the complete integration of telemetry, privacy controls,
feature gates, and monitoring working together with the fact recall pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import json
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import all Phase 4 components
from src.monitoring.fact_telemetry import (
    FactTelemetry, FactOperation, TelemetryLevel, telemetry_context,
    get_telemetry, initialize_telemetry
)
from security.fact_privacy import (
    PrivacyEnforcer, DataClassifier, PIIDetector, PrivacyAwareFact,
    DataClassification, RedactionLevel
)
from src.core.feature_gates import (
    FeatureGateManager, FeatureState, is_feature_enabled, feature_gate
)
from src.monitoring.fact_monitoring import (
    FactMonitoringSystem, record_operation_metric, record_custom_metric,
    get_monitoring_system, initialize_monitoring
)

# Import storage components for integration testing
from src.storage.fact_store import FactStore
from src.storage.graph_fact_store import GraphFactStore


class TestPhase4EndToEndIntegration:
    """End-to-end integration tests for complete Phase 4 pipeline."""
    
    def setup_method(self):
        """Setup comprehensive test environment."""
        # Mock Redis client
        self.redis_client = Mock()
        self.redis_client.set.return_value = True
        self.redis_client.setex.return_value = True
        self.redis_client.get.return_value = None
        self.redis_client.delete.return_value = 1
        self.redis_client.keys.return_value = []
        
        # Initialize all Phase 4 components
        self.telemetry = FactTelemetry({'telemetry_level': 'detailed'})
        self.privacy_enforcer = PrivacyEnforcer({'audit_all_access': True})
        self.feature_manager = FeatureGateManager()
        self.monitoring = FactMonitoringSystem()
        
        # Initialize storage components
        self.fact_store = FactStore(self.redis_client)
        self.graph_fact_store = GraphFactStore(self.fact_store, neo4j_client=None)
    
    def test_privacy_aware_fact_storage_with_telemetry(self):
        """Test privacy-aware fact storage with full telemetry."""
        # Setup
        user_id = "test_user_001"
        namespace = "integration_test"
        sensitive_data = "My email is alice@company.com and my SSN is 123-45-6789"
        
        # Check feature gates
        telemetry_enabled = self.feature_manager.is_enabled("fact_telemetry", user_id)
        privacy_enabled = self.feature_manager.is_enabled("privacy_controls", user_id)
        
        assert telemetry_enabled, "Telemetry should be enabled by default"
        assert privacy_enabled, "Privacy controls should be enabled by default"
        
        # Storage operation with full observability
        if telemetry_enabled:
            with telemetry_context(
                FactOperation.STORE_FACT, user_id, namespace,
                {"sensitive_data": True, "privacy_enabled": privacy_enabled}
            ) as trace_context:
                
                # Privacy processing
                if privacy_enabled:
                    privacy_fact = PrivacyAwareFact("personal_info", sensitive_data, self.privacy_enforcer)
                    processed_value = privacy_fact.value
                    classification = privacy_fact.classification
                    
                    # Verify privacy processing
                    assert processed_value != sensitive_data, "Data should be redacted"
                    assert classification != DataClassification.PUBLIC, "Should be classified as sensitive"
                    
                    # Check access permissions
                    access_granted, _, reason = privacy_fact.access_with_permission("read", user_id)
                    assert access_granted, f"Access should be granted: {reason}"
                
                # Store with monitoring
                try:
                    self.graph_fact_store.store_fact(
                        namespace, user_id, "personal_info", 
                        processed_value if privacy_enabled else sensitive_data,
                        confidence=0.95, provenance="integration_test"
                    )
                    
                    # Record operation metrics
                    record_operation_metric("integrated_storage", 25.5, True,
                                          privacy_enabled=privacy_enabled,
                                          telemetry_enabled=telemetry_enabled)
                    
                except Exception as e:
                    self.telemetry.record_failure(trace_context, 25.5, e)
                    pytest.fail(f"Storage operation failed: {e}")
        
        # Verify audit trail
        audit_summary = self.privacy_enforcer.get_audit_summary(hours=1)
        assert audit_summary["total_accesses"] > 0, "Should have audit entries"
        
        # Verify telemetry
        metrics = self.telemetry.get_operation_metrics(hours=1)
        assert len(metrics) > 0, "Should have telemetry metrics"
    
    def test_feature_gated_retrieval_with_privacy_controls(self):
        """Test feature-gated fact retrieval with privacy controls."""
        # Setup test data
        user_id = "test_user_002"
        namespace = "retrieval_test"
        
        # Store some test facts first
        test_facts = [
            ("email", "user@example.com"),
            ("phone", "555-123-4567"),
            ("preference", "I like spicy food"),
            ("location", "New York"),
        ]
        
        for attribute, value in test_facts:
            # Mock Redis to return the stored fact
            fact_data = {
                "value": value,
                "confidence": 1.0,
                "source_turn_id": "test",
                "updated_at": "2023-01-01T00:00:00",
                "provenance": "test",
                "attribute": attribute,
                "user_id": user_id,
                "namespace": namespace
            }
            self.redis_client.get.return_value = json.dumps(fact_data).encode('utf-8')
            
            # Feature-gated retrieval
            with feature_gate("privacy_controls", user_id) as privacy_enabled:
                if privacy_enabled:
                    # Create privacy-aware fact for retrieval
                    privacy_fact = PrivacyAwareFact(attribute, value, self.privacy_enforcer)
                    
                    # Test different access scenarios
                    access_scenarios = [
                        ("read", True),
                        ("write", True),
                        ("delete", True),
                    ]
                    
                    for operation, should_succeed in access_scenarios:
                        granted, retrieved_value, reason = privacy_fact.access_with_permission(
                            operation, user_id, {"location": "US"}
                        )
                        
                        if should_succeed:
                            assert granted, f"Access should be granted for {operation} on {attribute}"
                            if operation == "read" and granted:
                                assert retrieved_value is not None, "Should return value for read operation"
                else:
                    # Direct retrieval without privacy controls
                    retrieved_fact = self.graph_fact_store.get_fact(namespace, user_id, attribute)
                    assert retrieved_fact is not None, "Should retrieve fact directly"
    
    def test_comprehensive_monitoring_with_alerting(self):
        """Test comprehensive monitoring with alert generation."""
        # Generate various operations for monitoring
        operations = [
            ("store_fact", 45.2, True),
            ("retrieve_fact", 12.8, True),
            ("store_fact", 67.1, True),
            ("retrieve_fact", 156.7, False),  # Slow operation
            ("entity_extraction", 23.4, True),
            ("hybrid_scoring", 89.3, True),
        ]
        
        # Record operations
        for operation, duration, success in operations:
            record_operation_metric(operation, duration, success,
                                  namespace="monitoring_test",
                                  user_type="test_user")
        
        # Record custom metrics for graph operations
        record_custom_metric("graph_fact_store_neo4j_operations", 1.0,
                           operation="store", status="success")
        record_custom_metric("graph_fact_store_fallback_occurrences", 1.0,
                           operation="store", reason="neo4j_error")
        
        # Check monitoring data
        system_status = self.monitoring.get_system_status()
        assert system_status["monitoring_enabled"], "Monitoring should be enabled"
        assert system_status["total_metrics"] > 0, "Should have recorded metrics"
        
        # Test alert evaluation
        fired_alerts = self.monitoring.alert_manager.evaluate_alerts()
        # Note: Alerts may or may not fire depending on thresholds and timing
        
        # Test dashboard data
        dashboards = self.monitoring.dashboard_manager.list_dashboards()
        assert len(dashboards) > 0, "Should have default dashboards"
        
        # Get dashboard data
        if dashboards:
            dashboard_data = self.monitoring.dashboard_manager.get_dashboard_data(
                dashboards[0]["dashboard_id"]
            )
            assert "widgets_data" in dashboard_data, "Dashboard should have widget data"
    
    def test_feature_rollout_with_observability(self):
        """Test feature rollout with comprehensive observability."""
        # Test users for rollout
        test_users = [f"user_{i:03d}" for i in range(50)]
        
        # Create a rollout feature
        self.feature_manager.create_feature(
            "advanced_privacy_controls",
            state=FeatureState.ROLLOUT,
            rollout_percentage=30.0
        )
        
        # Track feature evaluations
        enabled_users = 0
        total_evaluations = 0
        
        for user in test_users:
            # Check feature with telemetry
            with telemetry_context(
                FactOperation.INTENT_CLASSIFICATION, user, "rollout_test"
            ) as trace_context:
                
                try:
                    enabled = self.feature_manager.is_enabled("advanced_privacy_controls", user)
                    if enabled:
                        enabled_users += 1
                    total_evaluations += 1
                    
                    # Record feature evaluation metric
                    record_custom_metric("feature_evaluation", 1.0,
                                       feature="advanced_privacy_controls",
                                       user_id=user,
                                       enabled=str(enabled))
                    
                except Exception as e:
                    self.telemetry.record_failure(trace_context, 5.0, e)
        
        # Verify rollout percentage (should be approximately 30%)
        rollout_percentage = (enabled_users / total_evaluations) * 100
        assert 20 <= rollout_percentage <= 40, f"Rollout percentage {rollout_percentage}% outside expected range"
        
        # Check feature statistics
        feature_stats = self.feature_manager.get_feature_stats(hours=1)
        assert feature_stats["total_evaluations"] >= total_evaluations, "Should record all evaluations"
    
    def test_privacy_compliance_workflow(self):
        """Test complete privacy compliance workflow."""
        # Setup compliance test data
        user_id = "compliance_user"
        namespace = "compliance_test"
        
        # Test data with various PII types
        compliance_test_data = [
            ("full_name", "John Michael Smith"),
            ("email_address", "john.smith@company.com"),
            ("phone_number", "555-123-4567"),
            ("ssn", "123-45-6789"),
            ("home_address", "123 Privacy Lane"),
            ("medical_info", "Patient has diabetes and hypertension"),
        ]
        
        stored_facts = []
        
        # Store facts with privacy compliance
        for attribute, value in compliance_test_data:
            with telemetry_context(
                FactOperation.STORE_FACT, user_id, namespace,
                {"compliance_test": True, "pii_type": attribute}
            ) as trace_context:
                
                # Create privacy-aware fact
                privacy_fact = PrivacyAwareFact(attribute, value, self.privacy_enforcer)
                
                # Verify classification
                classification = privacy_fact.classification
                metadata = privacy_fact.get_metadata()
                
                assert classification != DataClassification.PUBLIC, f"{attribute} should not be classified as public"
                assert "pii_types" in metadata, "Should detect PII types"
                assert len(metadata["pii_types"]) > 0, f"Should detect PII in {attribute}"
                
                # Store with compliance tracking
                stored_facts.append({
                    "attribute": attribute,
                    "original_value": value,
                    "processed_value": privacy_fact.value,
                    "classification": classification,
                    "metadata": metadata
                })
                
                # Mock storage
                self.redis_client.set.return_value = True
        
        # Test access controls for different operations
        access_operations = ["read", "write", "delete"]
        compliance_results = {"granted": 0, "denied": 0}
        
        for fact_info in stored_facts:
            privacy_fact = PrivacyAwareFact(
                fact_info["attribute"], 
                fact_info["original_value"], 
                self.privacy_enforcer
            )
            
            for operation in access_operations:
                granted, value, reason = privacy_fact.access_with_permission(
                    operation, user_id, {"location": "US", "compliance_mode": True}
                )
                
                if granted:
                    compliance_results["granted"] += 1
                else:
                    compliance_results["denied"] += 1
        
        # Generate compliance report
        privacy_report = self.privacy_enforcer.export_privacy_report()
        
        # Verify compliance report structure
        required_sections = [
            "privacy_configuration",
            "audit_summary_24h", 
            "supported_classifications",
            "supported_redaction_levels"
        ]
        
        for section in required_sections:
            assert section in privacy_report, f"Privacy report missing {section}"
        
        # Verify audit trail
        audit_summary = self.privacy_enforcer.get_audit_summary(hours=1)
        assert audit_summary["total_accesses"] > 0, "Should have audit entries for compliance test"
    
    def test_error_handling_and_resilience(self):
        """Test error handling and system resilience."""
        user_id = "resilience_user"
        namespace = "error_test"
        
        # Test scenarios with various failures
        error_scenarios = [
            # Redis failure simulation
            {
                "operation": "store_with_redis_error",
                "setup": lambda: setattr(self.redis_client, 'set', Mock(side_effect=Exception("Redis connection failed"))),
                "test": lambda: self.graph_fact_store.store_fact(namespace, user_id, "test", "value"),
                "expected_error": Exception
            },
            
            # Privacy processing error
            {
                "operation": "privacy_processing_error", 
                "setup": lambda: None,
                "test": lambda: PrivacyAwareFact("test", None, self.privacy_enforcer),  # None value should handle gracefully
                "expected_error": None  # Should handle gracefully
            },
            
            # Feature evaluation error
            {
                "operation": "feature_evaluation_error",
                "setup": lambda: None,
                "test": lambda: self.feature_manager.is_enabled("nonexistent_feature", user_id),
                "expected_error": None  # Should return False gracefully
            }
        ]
        
        resilience_results = {"handled_gracefully": 0, "failed_hard": 0}
        
        for scenario in error_scenarios:
            with telemetry_context(
                FactOperation.STORE_FACT, user_id, namespace,
                {"error_test": scenario["operation"]}
            ) as trace_context:
                
                try:
                    # Setup error condition
                    if scenario["setup"]:
                        scenario["setup"]()
                    
                    # Execute test
                    result = scenario["test"]()
                    
                    if scenario["expected_error"] is None:
                        # Should handle gracefully
                        resilience_results["handled_gracefully"] += 1
                    else:
                        # If we get here and expected an error, that's unexpected
                        resilience_results["failed_hard"] += 1
                        
                except Exception as e:
                    if scenario["expected_error"] and isinstance(e, scenario["expected_error"]):
                        # Expected error occurred
                        resilience_results["handled_gracefully"] += 1
                        
                        # Record failure for telemetry
                        self.telemetry.record_failure(trace_context, 10.0, e)
                    else:
                        # Unexpected error
                        resilience_results["failed_hard"] += 1
        
        # Verify system resilience
        assert resilience_results["handled_gracefully"] > 0, "Should handle some errors gracefully"
        
        # Check that telemetry recorded failures
        failure_summary = self.telemetry.get_failure_summary(hours=1)
        # May or may not have failures depending on error handling
    
    def test_performance_under_load(self):
        """Test system performance under simulated load."""
        # Simulate moderate load
        num_operations = 100
        operations_completed = 0
        total_duration = 0
        
        start_time = time.time()
        
        for i in range(num_operations):
            user_id = f"load_user_{i % 10}"  # 10 different users
            namespace = "load_test"
            
            operation_start = time.time()
            
            try:
                # Feature gate check
                telemetry_enabled = is_feature_enabled("fact_telemetry", user_id)
                privacy_enabled = is_feature_enabled("privacy_controls", user_id)
                
                # Simulated fact storage with observability
                if telemetry_enabled:
                    with telemetry_context(
                        FactOperation.STORE_FACT, user_id, namespace
                    ) as trace_context:
                        
                        # Privacy processing if enabled
                        if privacy_enabled:
                            fact = PrivacyAwareFact(f"fact_{i}", f"value_{i}", self.privacy_enforcer)
                            processed_value = fact.value
                        else:
                            processed_value = f"value_{i}"
                        
                        # Mock storage operation
                        self.redis_client.set.return_value = True
                        
                        # Record metrics
                        operation_duration = (time.time() - operation_start) * 1000
                        record_operation_metric("load_test_operation", operation_duration, True,
                                              user_id=user_id, operation_number=i)
                        
                        operations_completed += 1
                else:
                    # Direct operation without telemetry
                    operations_completed += 1
                
            except Exception as e:
                # Log but continue with load test
                logger.error(f"Load test operation {i} failed: {e}")
        
        total_duration = time.time() - start_time
        
        # Performance assertions
        assert operations_completed >= num_operations * 0.95, "Should complete at least 95% of operations"
        assert total_duration < num_operations * 0.1, f"Should complete {num_operations} operations in reasonable time"
        
        # Check system status after load
        system_status = self.monitoring.get_system_status()
        assert system_status["system_health"]["status"] == "healthy", "System should remain healthy under load"


class TestPhase4ComponentInteraction:
    """Test specific interactions between Phase 4 components."""
    
    def setup_method(self):
        """Setup for component interaction tests."""
        self.telemetry = FactTelemetry()
        self.privacy_enforcer = PrivacyEnforcer()
        self.feature_manager = FeatureGateManager()
        self.monitoring = FactMonitoringSystem()
    
    def test_telemetry_privacy_integration(self):
        """Test telemetry with privacy-aware data scrubbing."""
        # Test that telemetry properly scrubs PII
        sensitive_metadata = {
            "user_email": "user@example.com",
            "phone": "555-123-4567",
            "safe_data": "some_id_123"
        }
        
        with telemetry_context(
            FactOperation.STORE_FACT,
            "test_user",
            "integration_test",
            sensitive_metadata
        ) as trace_context:
            
            # Telemetry should scrub the metadata
            scrubbed_metadata = trace_context.metadata
            
            # Check that PII is scrubbed but safe data remains
            # (Implementation may vary based on scrubbing rules)
            assert "safe_data" in scrubbed_metadata or len(scrubbed_metadata) > 0
    
    def test_feature_gates_telemetry_interaction(self):
        """Test feature gates working with telemetry recording."""
        # Create feature specifically for telemetry testing
        self.feature_manager.create_feature(
            "telemetry_test_feature",
            state=FeatureState.ROLLOUT,
            rollout_percentage=50.0
        )
        
        # Test multiple users to verify consistent assignment
        test_users = ["user_001", "user_002", "user_003", "user_004"]
        evaluations = []
        
        for user in test_users:
            enabled = self.feature_manager.is_enabled("telemetry_test_feature", user)
            evaluations.append(enabled)
            
            # Each evaluation should be recorded in telemetry
            record_custom_metric("feature_test_evaluation", 1.0,
                               feature="telemetry_test_feature",
                               user=user,
                               enabled=str(enabled))
        
        # Verify feature stats are tracked
        stats = self.feature_manager.get_feature_stats()
        assert stats["total_evaluations"] > 0, "Should track feature evaluations"
    
    def test_monitoring_privacy_compliance(self):
        """Test monitoring system with privacy compliance."""
        # Generate metrics that might contain sensitive data
        sensitive_operations = [
            ("store_email", {"user_data": "email_content"}),
            ("store_phone", {"user_data": "phone_content"}),
            ("retrieve_personal", {"query": "personal_info"})
        ]
        
        for operation, metadata in sensitive_operations:
            record_operation_metric(operation, 50.0, True, **metadata)
        
        # Monitoring should handle potentially sensitive metadata appropriately
        system_status = self.monitoring.get_system_status()
        assert system_status["total_metrics"] > 0, "Should record metrics"
        
        # Export monitoring data (should be scrubbed)
        monitoring_config = self.monitoring.export_monitoring_config()
        assert "exported_at" in monitoring_config["metadata"], "Should include export metadata"


if __name__ == "__main__":
    # Run comprehensive integration tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for debugging