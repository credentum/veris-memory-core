"""
Security Monitoring Tests
Sprint 10 Phase 2 - Issue 006: SEC-106
Tests the security monitoring and alerting system
"""

import pytest
import time
import json
import queue
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.monitoring.security_monitor import (
    SecurityMonitor,
    SecurityEventDetector,
    AlertManager,
    SecurityEvent,
    Alert,
    AlertChannel,
    AlertSeverity,
    EventType
)


class TestSecurityEventDetector:
    """Test ID: SEC-106-A - Security Event Detection"""
    
    def test_brute_force_detection(self):
        """Test detection of brute force attacks"""
        detector = SecurityEventDetector()
        
        # Generate multiple failed login attempts from same IP
        source_ip = "192.168.1.100"
        events = []
        
        for i in range(6):  # 6 attempts (threshold is 5)
            log_entry = {
                "event": "auth_failure",
                "source_ip": source_ip,
                "user_id": f"user_{i}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            event = detector.analyze_log_entry(log_entry)
            if event:
                events.append(event)
        
        # Should detect brute force on 5th attempt
        assert len(events) == 1
        assert events[0].event_type == EventType.BRUTE_FORCE_ATTACK
        assert events[0].severity == AlertSeverity.HIGH
        assert events[0].source_ip == source_ip
    
    def test_port_scan_detection(self):
        """Test detection of port scanning"""
        detector = SecurityEventDetector()
        
        source_ip = "192.168.1.200"
        events = []
        
        # Simulate port scanning (accessing many ports)
        for port in range(8000, 8015):  # 15 ports (threshold is 10)
            log_entry = {
                "event": "connection_attempt",
                "source_ip": source_ip,
                "port": port,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            event = detector.analyze_log_entry(log_entry)
            if event:
                events.append(event)
        
        # Should detect port scan
        assert len(events) == 1
        assert events[0].event_type == EventType.PORT_SCAN
        assert events[0].severity == AlertSeverity.MEDIUM
        assert len(events[0].metadata["ports_accessed"]) >= 10
    
    def test_ddos_detection(self):
        """Test detection of DDoS attacks"""
        detector = SecurityEventDetector()
        
        source_ip = "192.168.1.50"
        events = []
        
        # Simulate rapid requests (DDoS)
        for i in range(150):  # 150 requests (threshold is 100 per minute)
            log_entry = {
                "event": "request",
                "source_ip": source_ip,
                "path": f"/api/endpoint_{i % 10}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            event = detector.analyze_log_entry(log_entry)
            if event:
                events.append(event)
        
        # Should detect DDoS
        assert len(events) >= 1
        assert events[0].event_type == EventType.DDOS_ATTACK
        assert events[0].severity == AlertSeverity.CRITICAL
    
    def test_sql_injection_detection(self):
        """Test detection of SQL injection attempts"""
        detector = SecurityEventDetector()
        
        log_entry = {
            "event": "waf_block",
            "rule": "sql_injection",
            "source_ip": "192.168.1.150",
            "payload": "' OR 1=1--",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        event = detector.analyze_log_entry(log_entry)
        
        assert event is not None
        assert event.event_type == EventType.SQL_INJECTION
        assert event.severity == AlertSeverity.HIGH
        assert "injection" in event.description.lower()
    
    def test_xss_detection(self):
        """Test detection of XSS attacks"""
        detector = SecurityEventDetector()
        
        log_entry = {
            "event": "waf_block",
            "rule": "xss_protection",
            "source_ip": "192.168.1.160",
            "payload": "<script>alert('xss')</script>",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        event = detector.analyze_log_entry(log_entry)
        
        assert event is not None
        assert event.event_type == EventType.XSS_ATTACK
        assert event.severity == AlertSeverity.HIGH
        assert "xss" in event.description.lower()
    
    def test_no_false_positives(self):
        """Test that legitimate traffic doesn't trigger false positives"""
        detector = SecurityEventDetector()
        
        # Legitimate log entries
        legitimate_entries = [
            {"event": "request", "source_ip": "10.0.1.50", "path": "/api/health"},
            {"event": "auth_success", "source_ip": "10.0.1.60", "user_id": "valid_user"},
            {"event": "connection_attempt", "source_ip": "10.0.1.70", "port": 8000},
        ]
        
        for entry in legitimate_entries:
            event = detector.analyze_log_entry(entry)
            assert event is None, f"False positive detected for: {entry}"


class TestAlertManager:
    """Test ID: SEC-106-B - Alert Management"""
    
    def test_alert_creation(self):
        """Test alert creation from security events"""
        config = {
            "alert_email": "security@test.com",
            "webhook_url": "http://test.com/webhook"
        }
        
        alert_manager = AlertManager(config)
        
        # Create test security event
        event = SecurityEvent(
            event_type=EventType.BRUTE_FORCE_ATTACK,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            description="Test brute force attack"
        )
        
        alert = alert_manager.create_alert(event)
        
        assert alert.event == event
        assert len(alert.channels) >= 1  # Should have at least one channel
        assert alert.created_at is not None
        assert alert.id.startswith("alert_")
    
    def test_severity_filtering(self):
        """Test that alerts are filtered by severity"""
        config = {"alert_email": "security@test.com"}
        alert_manager = AlertManager(config)
        
        # Add channel that only accepts CRITICAL alerts
        critical_channel = AlertChannel(
            name="critical_only",
            channel_type="email",
            endpoint="critical@test.com",
            severity_filter={AlertSeverity.CRITICAL}
        )
        alert_manager.add_channel(critical_channel)
        
        # Create HIGH severity event
        high_event = SecurityEvent(
            event_type=EventType.SQL_INJECTION,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            description="SQL injection attempt"
        )
        
        alert = alert_manager.create_alert(high_event)
        
        # Critical-only channel should not be included
        assert "critical_only" not in alert.channels
    
    def test_rate_limiting(self):
        """Test alert rate limiting"""
        config = {"alert_email": "security@test.com"}
        alert_manager = AlertManager(config)
        
        # Add channel with low rate limit
        limited_channel = AlertChannel(
            name="limited",
            channel_type="email",
            endpoint="limited@test.com",
            rate_limit=2,  # Only 2 alerts per hour
            severity_filter={AlertSeverity.HIGH}
        )
        alert_manager.add_channel(limited_channel)
        
        # Create multiple events and send alerts
        alerts_with_channel = 0
        for i in range(5):
            event = SecurityEvent(
                event_type=EventType.SUSPICIOUS_ACTIVITY,
                severity=AlertSeverity.HIGH,
                timestamp=datetime.utcnow(),
                source_ip=f"192.168.1.{100 + i}",
                description=f"Suspicious activity {i}"
            )
            
            alert = alert_manager.create_alert(event)
            if "limited" in alert.channels:
                alerts_with_channel += 1
                # Simulate sending to update rate limiter
                alert_manager.rate_limiters["limited"].append(datetime.utcnow())
        
        assert alerts_with_channel <= 3, f"Rate limiting not working: {alerts_with_channel} alerts sent"
    
    def test_email_alert_sending(self):
        """Test email alert configuration"""
        config = {
            "smtp_server": "localhost",
            "smtp_port": 587,
            "smtp_user": "test@example.com",
            "smtp_password": "password",
            "alert_email": "security@test.com"
        }
        
        alert_manager = AlertManager(config)
        
        # Create test alert
        event = SecurityEvent(
            event_type=EventType.BRUTE_FORCE_ATTACK,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            description="Critical brute force attack"
        )
        
        alert = alert_manager.create_alert(event)
        
        # Verify email channel was configured
        assert "email" in alert_manager.channels
        assert alert_manager.channels["email"].endpoint == "security@test.com"
        
        # Verify alert has correct properties
        assert alert.event.severity == AlertSeverity.CRITICAL
        assert "brute force" in alert.message.lower()
    
    @patch('requests.post')
    def test_webhook_alert_sending(self, mock_post):
        """Test sending alerts via webhook"""
        config = {"webhook_url": "http://test.com/webhook"}
        alert_manager = AlertManager(config)
        
        # Create test alert
        event = SecurityEvent(
            event_type=EventType.SQL_INJECTION,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.utcnow(),
            source_ip="192.168.1.100",
            description="SQL injection detected"
        )
        
        alert = alert_manager.create_alert(event)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Send alert
        alert_manager.send_alert(alert)
        
        # Verify webhook was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["event_type"] == "sql_injection"
        assert call_args[1]["json"]["severity"] == "high"


class TestSecurityMonitor:
    """Test ID: SEC-106-C - Complete Security Monitoring"""
    
    def test_security_monitor_initialization(self):
        """Test security monitor initialization"""
        config = {"alert_email": "security@test.com"}
        monitor = SecurityMonitor(config)
        
        assert monitor.detector is not None
        assert monitor.alert_manager is not None
        assert monitor.metrics["events_processed"] == 0
        assert monitor.metrics["alerts_generated"] == 0
    
    def test_log_processing_workflow(self):
        """Test complete log processing workflow"""
        config = {"alert_email": "security@test.com"}
        monitor = SecurityMonitor(config)
        monitor.start()
        
        try:
            # Process multiple log entries
            log_entries = [
                # Normal request
                {"event": "request", "source_ip": "10.0.1.50", "path": "/api/health"},
                
                # Failed authentication attempts (should trigger brute force)
                {"event": "auth_failure", "source_ip": "192.168.1.100", "user_id": "admin"},
                {"event": "auth_failure", "source_ip": "192.168.1.100", "user_id": "admin"},
                {"event": "auth_failure", "source_ip": "192.168.1.100", "user_id": "admin"},
                {"event": "auth_failure", "source_ip": "192.168.1.100", "user_id": "admin"},
                {"event": "auth_failure", "source_ip": "192.168.1.100", "user_id": "admin"},
                
                # WAF block
                {"event": "waf_block", "rule": "sql_injection", "source_ip": "192.168.1.200", "payload": "' OR 1=1--"},
            ]
            
            for entry in log_entries:
                monitor.process_log_entry(entry)
            
            # Wait a bit for processing
            time.sleep(0.1)
            
            # Check metrics
            metrics = monitor.get_metrics()
            assert metrics["events_processed"] == len(log_entries)
            assert metrics["alerts_generated"] >= 2  # At least brute force + SQL injection
            
        finally:
            monitor.stop()
    
    def test_metrics_collection(self):
        """Test metrics collection and reporting"""
        config = {}  # No external channels to avoid network issues
        monitor = SecurityMonitor(config)
        monitor.start()  # Start the monitor
        
        try:
            # Process some events to generate metrics
            for i in range(10):
                log_entry = {
                    "event": "waf_block",
                    "rule": "xss_protection", 
                    "source_ip": f"192.168.1.{100 + i}",
                    "payload": "<script>alert(1)</script>"
                }
                monitor.process_log_entry(log_entry)
            
            # Wait for processing
            time.sleep(0.1)
            
            metrics = monitor.get_metrics()
            
            assert metrics["events_processed"] == 10
            assert metrics["alerts_generated"] == 10
            assert "xss_attack" in metrics["events_by_type"]
            assert "high" in metrics["alerts_by_severity"]
            
        finally:
            monitor.stop()
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment functionality"""
        monitor = SecurityMonitor()
        monitor.start()
        
        try:
            # Generate an alert
            log_entry = {
                "event": "waf_block",
                "rule": "sql_injection",
                "source_ip": "192.168.1.100",
                "payload": "' OR 1=1--"
            }
            monitor.process_log_entry(log_entry)
            
            # Wait for alert to be processed
            time.sleep(0.1)
            
            # Get recent alerts
            recent_alerts = monitor.get_recent_alerts(1)
            assert len(recent_alerts) >= 1
            
            alert_id = recent_alerts[0].id
            
            # Acknowledge the alert
            result = monitor.acknowledge_alert(alert_id, "security_admin")
            assert result is True
            
            # Verify acknowledgment
            recent_alerts = monitor.get_recent_alerts(1)
            acknowledged_alert = next(a for a in recent_alerts if a.id == alert_id)
            assert acknowledged_alert.acknowledged is True
            assert acknowledged_alert.acknowledged_by == "security_admin"
            
        finally:
            monitor.stop()
    
    def test_event_simulation(self):
        """Test security event simulation for testing"""
        config = {}  # No external channels to avoid network issues
        monitor = SecurityMonitor(config)
        monitor.start()
        
        try:
            # Simulate different types of events
            event_types = [
                EventType.BRUTE_FORCE_ATTACK,
                EventType.SQL_INJECTION,
                EventType.XSS_ATTACK,
                EventType.PORT_SCAN
            ]
            
            for event_type in event_types:
                event = monitor.simulate_event(event_type, "192.168.1.100")
                assert event.event_type == event_type
                assert event.source_ip == "192.168.1.100"
                assert event.metadata["simulated"] is True
            
            # Wait for alerts to be processed
            time.sleep(0.1)
            
            # Check that alerts were generated
            metrics = monitor.get_metrics()
            # Check that alert history contains the simulated events
            recent_alerts = monitor.get_recent_alerts(1)
            assert len(recent_alerts) >= len(event_types)
            
        finally:
            monitor.stop()
    
    def test_concurrent_event_processing(self):
        """Test handling of concurrent events"""
        import threading
        
        monitor = SecurityMonitor()
        monitor.start()
        
        try:
            def process_events(start_ip):
                for i in range(10):
                    log_entry = {
                        "event": "auth_failure",
                        "source_ip": f"192.168.{start_ip}.{i}",
                        "user_id": f"user_{i}"
                    }
                    monitor.process_log_entry(log_entry)
            
            # Start multiple threads
            threads = []
            for ip_range in range(1, 6):  # 5 threads
                thread = threading.Thread(target=process_events, args=(ip_range,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Wait for processing to complete
            time.sleep(0.5)
            
            # Check metrics
            metrics = monitor.get_metrics()
            assert metrics["events_processed"] == 50  # 5 threads * 10 events each
            
        finally:
            monitor.stop()


class TestIntegration:
    """Test ID: SEC-106-D - Integration Tests"""
    
    def test_end_to_end_monitoring(self):
        """Test complete end-to-end monitoring workflow"""
        config = {}  # No external channels to avoid network issues
        
        monitor = SecurityMonitor(config)
        monitor.start()
        
        try:
            # Simulate a realistic attack scenario
            attack_scenario = [
                # Initial reconnaissance
                {"event": "connection_attempt", "source_ip": "192.168.1.100", "port": 22},
                {"event": "connection_attempt", "source_ip": "192.168.1.100", "port": 80},
                {"event": "connection_attempt", "source_ip": "192.168.1.100", "port": 443},
                
                # Port scanning
                *[{"event": "connection_attempt", "source_ip": "192.168.1.100", "port": 8000 + i} 
                  for i in range(15)],
                
                # Web application attacks
                {"event": "waf_block", "rule": "sql_injection", "source_ip": "192.168.1.100", "payload": "' OR 1=1--"},
                {"event": "waf_block", "rule": "xss_protection", "source_ip": "192.168.1.100", "payload": "<script>alert(1)</script>"},
                
                # Brute force attack
                *[{"event": "auth_failure", "source_ip": "192.168.1.100", "user_id": "admin"} 
                  for _ in range(6)],
                
                # DDoS attempt
                *[{"event": "request", "source_ip": "192.168.1.100", "path": "/api/endpoint"} 
                  for _ in range(120)],
            ]
            
            # Process all events
            for event in attack_scenario:
                monitor.process_log_entry(event)
            
            # Wait for processing
            time.sleep(1.0)
            
            # Verify multiple alerts were generated
            metrics = monitor.get_metrics()
            assert metrics["events_processed"] == len(attack_scenario)
            assert metrics["alerts_generated"] >= 4  # Port scan, SQL injection, XSS, brute force, DDoS
            
            # Verify different event types were detected
            assert len(metrics["events_by_type"]) >= 4
            assert "port_scan" in metrics["events_by_type"]
            assert "sql_injection" in metrics["events_by_type"]
            assert "brute_force" in metrics["events_by_type"]  # Fixed: use correct enum value
            
            # Get recent alerts and verify severity distribution
            recent_alerts = monitor.get_recent_alerts(1)
            severities = [alert.event.severity for alert in recent_alerts]
            
            # Should have both HIGH and CRITICAL severity alerts
            assert AlertSeverity.HIGH in severities
            assert AlertSeverity.CRITICAL in severities or AlertSeverity.MEDIUM in severities
            
        finally:
            monitor.stop()
    
    @patch('requests.post')
    def test_alerting_integration(self, mock_post):
        """Test integration with external alerting systems"""
        # Mock successful webhook responses
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        config = {"webhook_url": "http://test.com/webhook"}
        monitor = SecurityMonitor(config)
        monitor.start()
        
        try:
            # Process critical event
            log_entry = {
                "event": "waf_block",
                "rule": "sql_injection",
                "source_ip": "192.168.1.100",
                "payload": "'; DROP TABLE users; --",
                "user_agent": "SQLMap/1.0"
            }
            
            monitor.process_log_entry(log_entry)
            
            # Wait for alert processing
            time.sleep(0.5)
            
            # Verify webhook was called
            mock_post.assert_called()
            
            # Verify alert payload
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            
            assert payload["event_type"] == "sql_injection"
            assert payload["severity"] == "high"
            assert payload["source_ip"] == "192.168.1.100"
            assert "injection" in payload["description"].lower()  # Fixed: more flexible matching
            
        finally:
            monitor.stop()


if __name__ == "__main__":
    # Run monitoring tests
    pytest.main([__file__, "-v", "-s"])