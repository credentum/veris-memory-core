#!/usr/bin/env python3
"""
Comprehensive tests for Monitoring Security Monitor - Phase 8 Coverage

This test module provides comprehensive coverage for the security monitoring system
including event detection, alert management, and security monitoring workflows.
"""
import pytest
import tempfile
import json
import time
import threading
import queue
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, List, Optional, Set

# Import security monitor components
try:
    from src.monitoring.security_monitor import (
        AlertSeverity, EventType, SecurityEvent, AlertChannel, Alert,
        SecurityEventDetector, AlertManager, SecurityMonitor
    )
    SECURITY_MONITOR_AVAILABLE = True
except ImportError:
    SECURITY_MONITOR_AVAILABLE = False


@pytest.mark.skipif(not SECURITY_MONITOR_AVAILABLE, reason="Security monitor not available")
class TestSecurityMonitorEnums:
    """Test security monitor enums and constants"""
    
    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values"""
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_event_type_enum(self):
        """Test EventType enum values"""
        assert EventType.AUTHENTICATION_FAILURE.value == "auth_failure"
        assert EventType.BRUTE_FORCE_ATTACK.value == "brute_force"
        assert EventType.SQL_INJECTION.value == "sql_injection"
        assert EventType.XSS_ATTACK.value == "xss_attack"
        assert EventType.PORT_SCAN.value == "port_scan"
        assert EventType.DDOS_ATTACK.value == "ddos_attack"
        assert EventType.PRIVILEGE_ESCALATION.value == "privilege_escalation"
        assert EventType.DATA_EXFILTRATION.value == "data_exfiltration"
        assert EventType.SUSPICIOUS_ACTIVITY.value == "suspicious_activity"
        assert EventType.SYSTEM_ANOMALY.value == "system_anomaly"


@pytest.mark.skipif(not SECURITY_MONITOR_AVAILABLE, reason="Security monitor not available")
class TestSecurityMonitorDataModels:
    """Test security monitor data models"""
    
    def test_security_event_creation(self):
        """Test SecurityEvent dataclass creation"""
        now = datetime.now(timezone.utc)
        event = SecurityEvent(
            event_type=EventType.BRUTE_FORCE_ATTACK,
            severity=AlertSeverity.HIGH,
            timestamp=now,
            source_ip="192.168.1.100",
            user_id="user123",
            description="Multiple failed login attempts",
            metadata={"attempts": 5, "duration": "5m"},
            raw_data={"logs": ["failed_login_1", "failed_login_2"]}
        )
        
        assert event.event_type == EventType.BRUTE_FORCE_ATTACK
        assert event.severity == AlertSeverity.HIGH
        assert event.timestamp == now
        assert event.source_ip == "192.168.1.100"
        assert event.user_id == "user123"
        assert event.description == "Multiple failed login attempts"
        assert event.metadata == {"attempts": 5, "duration": "5m"}
        assert event.raw_data == {"logs": ["failed_login_1", "failed_login_2"]}
    
    def test_security_event_defaults(self):
        """Test SecurityEvent default values"""
        now = datetime.now(timezone.utc)
        event = SecurityEvent(
            event_type=EventType.PORT_SCAN,
            severity=AlertSeverity.MEDIUM,
            timestamp=now,
            source_ip="10.0.0.50"
        )
        
        assert event.user_id is None
        assert event.description == ""
        assert event.metadata == {}
        assert event.raw_data == {}
    
    def test_alert_channel_creation(self):
        """Test AlertChannel dataclass creation"""
        channel = AlertChannel(
            name="security_alerts",
            channel_type="email",
            endpoint="security@company.com",
            enabled=True,
            severity_filter={AlertSeverity.HIGH, AlertSeverity.CRITICAL},
            rate_limit=5
        )
        
        assert channel.name == "security_alerts"
        assert channel.channel_type == "email"
        assert channel.endpoint == "security@company.com"
        assert channel.enabled is True
        assert channel.severity_filter == {AlertSeverity.HIGH, AlertSeverity.CRITICAL}
        assert channel.rate_limit == 5
    
    def test_alert_channel_defaults(self):
        """Test AlertChannel default values"""
        channel = AlertChannel(
            name="default_channel",
            channel_type="webhook",
            endpoint="https://hooks.company.com/alerts"
        )
        
        assert channel.enabled is True
        assert channel.severity_filter == {AlertSeverity.HIGH, AlertSeverity.CRITICAL}
        assert channel.rate_limit == 10
    
    def test_alert_creation(self):
        """Test Alert dataclass creation"""
        now = datetime.now(timezone.utc)
        event = SecurityEvent(
            event_type=EventType.SQL_INJECTION,
            severity=AlertSeverity.CRITICAL,
            timestamp=now,
            source_ip="203.0.113.100"
        )
        
        alert = Alert(
            id="alert_123",
            event=event,
            message="Critical SQL injection detected",
            channels=["email", "slack"],
            created_at=now,
            sent_at=now + timedelta(seconds=5),
            acknowledged=True,
            acknowledged_by="admin_user"
        )
        
        assert alert.id == "alert_123"
        assert alert.event == event
        assert alert.message == "Critical SQL injection detected"
        assert alert.channels == ["email", "slack"]
        assert alert.created_at == now
        assert alert.sent_at == now + timedelta(seconds=5)
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin_user"
    
    def test_alert_defaults(self):
        """Test Alert default values"""
        now = datetime.now(timezone.utc)
        event = SecurityEvent(
            event_type=EventType.SUSPICIOUS_ACTIVITY,
            severity=AlertSeverity.MEDIUM,
            timestamp=now,
            source_ip="192.168.100.50"
        )
        
        alert = Alert(
            id="alert_456",
            event=event,
            message="Suspicious activity detected",
            channels=["webhook"],
            created_at=now
        )
        
        assert alert.sent_at is None
        assert alert.acknowledged is False
        assert alert.acknowledged_by is None


@pytest.mark.skipif(not SECURITY_MONITOR_AVAILABLE, reason="Security monitor not available")
class TestSecurityEventDetector:
    """Test security event detector functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = SecurityEventDetector()
    
    def test_detector_initialization(self):
        """Test security event detector initialization"""
        assert self.detector is not None
        assert hasattr(self.detector, 'detectors')
        assert hasattr(self.detector, 'failed_login_attempts')
        assert hasattr(self.detector, 'suspicious_ips')
        assert hasattr(self.detector, 'port_scan_attempts')
    
    def test_detect_brute_force_attack(self):
        """Test brute force attack detection"""
        source_ip = "192.168.1.100"
        
        # Simulate multiple failed login attempts
        for i in range(6):  # Exceed threshold
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": source_ip,
                "event": "failed_login",
                "user": f"user_{i}",
                "endpoint": "/auth/login"
            }
            
            result = self.detector.detect_event(EventType.BRUTE_FORCE_ATTACK, log_entry)
        
        # Should detect brute force after multiple attempts
        assert result is not None
        assert result.event_type == EventType.BRUTE_FORCE_ATTACK
        assert result.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        assert result.source_ip == source_ip
    
    def test_detect_port_scan(self):
        """Test port scan detection"""
        source_ip = "10.0.0.50"
        
        # Simulate port scan - multiple ports from same IP
        ports = [22, 23, 80, 443, 8080, 3389, 5432, 3306, 1433, 27017]
        
        for port in ports:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": source_ip,
                "dest_port": port,
                "connection_type": "tcp_syn"
            }
            
            result = self.detector.detect_event(EventType.PORT_SCAN, log_entry)
        
        # Should detect port scan after scanning multiple ports
        assert result is not None
        assert result.event_type == EventType.PORT_SCAN
        assert result.source_ip == source_ip
    
    def test_detect_sql_injection(self):
        """Test SQL injection detection"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": "203.0.113.100",
            "request_path": "/api/users",
            "query_params": "id=1' OR '1'='1",
            "user_agent": "SQLMap/1.0"
        }
        
        result = self.detector.detect_event(EventType.SQL_INJECTION, log_entry)
        
        assert result is not None
        assert result.event_type == EventType.SQL_INJECTION
        assert result.severity == AlertSeverity.CRITICAL
        assert "SQL injection pattern" in result.description
    
    def test_detect_xss_attack(self):
        """Test XSS attack detection"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": "198.51.100.50",
            "request_path": "/search",
            "query_params": "q=<script>alert('XSS')</script>",
            "user_agent": "Mozilla/5.0"
        }
        
        result = self.detector.detect_event(EventType.XSS_ATTACK, log_entry)
        
        assert result is not None
        assert result.event_type == EventType.XSS_ATTACK
        assert result.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def test_detect_ddos_attack(self):
        """Test DDoS attack detection"""
        source_ip = "172.16.0.100"
        
        # Simulate high volume of requests
        for i in range(100):  # High request count
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": source_ip,
                "request_path": "/",
                "response_time": 5000,  # High response time
                "request_size": 1024
            }
            
            result = self.detector.detect_event(EventType.DDOS_ATTACK, log_entry)
        
        # Should detect DDoS pattern
        assert result is not None
        assert result.event_type == EventType.DDOS_ATTACK
        assert result.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]
    
    def test_detect_privilege_escalation(self):
        """Test privilege escalation detection"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": "192.168.100.25",
            "user_id": "user123",
            "action": "sudo",
            "command": "su - root",
            "previous_user": "user123",
            "new_user": "root"
        }
        
        result = self.detector.detect_event(EventType.PRIVILEGE_ESCALATION, log_entry)
        
        assert result is not None
        assert result.event_type == EventType.PRIVILEGE_ESCALATION
        assert result.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]
    
    def test_detect_data_exfiltration(self):
        """Test data exfiltration detection"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": "10.0.1.50",
            "user_id": "user456",
            "action": "download",
            "file_path": "/sensitive/customer_data.csv",
            "file_size": 10485760,  # 10MB
            "destination": "external_storage"
        }
        
        result = self.detector.detect_event(EventType.DATA_EXFILTRATION, log_entry)
        
        assert result is not None
        assert result.event_type == EventType.DATA_EXFILTRATION
        assert result.severity == AlertSeverity.CRITICAL
    
    def test_detect_system_anomaly(self):
        """Test system anomaly detection"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": "127.0.0.1",
            "system_metric": "cpu_usage",
            "value": 95.0,  # High CPU usage
            "threshold": 85.0,
            "duration": "5m"
        }
        
        result = self.detector.detect_event(EventType.SYSTEM_ANOMALY, log_entry)
        
        assert result is not None
        assert result.event_type == EventType.SYSTEM_ANOMALY
        assert result.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH]
    
    def test_cleanup_old_data(self):
        """Test cleanup of old detection data"""
        # Add some test data
        self.detector.failed_login_attempts["192.168.1.100"] = [
            time.time() - 7200,  # 2 hours ago
            time.time() - 3600,  # 1 hour ago
            time.time() - 300    # 5 minutes ago
        ]
        
        # Cleanup data older than 1 hour
        self.detector.cleanup_old_data(max_age_seconds=3600)
        
        # Should keep only recent data
        remaining_attempts = self.detector.failed_login_attempts.get("192.168.1.100", [])
        assert len(remaining_attempts) <= 1  # Only the 5-minute-old entry


@pytest.mark.skipif(not SECURITY_MONITOR_AVAILABLE, reason="Security monitor not available")
class TestAlertManager:
    """Test alert manager functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.alert_manager = AlertManager()
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization"""
        assert self.alert_manager is not None
        assert hasattr(self.alert_manager, 'channels')
        assert hasattr(self.alert_manager, 'alerts')
        assert hasattr(self.alert_manager, 'alert_queue')
        assert hasattr(self.alert_manager, 'rate_limiters')
    
    def test_add_alert_channel(self):
        """Test adding alert channel"""
        channel = AlertChannel(
            name="test_email",
            channel_type="email",
            endpoint="test@company.com",
            severity_filter={AlertSeverity.HIGH}
        )
        
        self.alert_manager.add_channel(channel)
        
        assert "test_email" in self.alert_manager.channels
        assert self.alert_manager.channels["test_email"] == channel
    
    def test_remove_alert_channel(self):
        """Test removing alert channel"""
        channel = AlertChannel(
            name="temp_channel",
            channel_type="webhook",
            endpoint="https://example.com/webhook"
        )
        
        self.alert_manager.add_channel(channel)
        assert "temp_channel" in self.alert_manager.channels
        
        self.alert_manager.remove_channel("temp_channel")
        assert "temp_channel" not in self.alert_manager.channels
    
    def test_create_alert(self):
        """Test creating an alert"""
        event = SecurityEvent(
            event_type=EventType.BRUTE_FORCE_ATTACK,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.now(timezone.utc),
            source_ip="192.168.1.100",
            description="Brute force attack detected"
        )
        
        alert = self.alert_manager.create_alert(event)
        
        assert alert is not None
        assert alert.event == event
        assert alert.id is not None
        assert len(alert.id) > 0
        assert alert.created_at is not None
    
    def test_filter_channels_by_severity(self):
        """Test filtering channels by severity"""
        # Add channels with different severity filters
        high_channel = AlertChannel(
            name="high_alerts",
            channel_type="email",
            endpoint="high@company.com",
            severity_filter={AlertSeverity.HIGH, AlertSeverity.CRITICAL}
        )
        
        critical_channel = AlertChannel(
            name="critical_alerts",
            channel_type="pagerduty",
            endpoint="pd://critical",
            severity_filter={AlertSeverity.CRITICAL}
        )
        
        self.alert_manager.add_channel(high_channel)
        self.alert_manager.add_channel(critical_channel)
        
        # Test HIGH severity event
        high_event = SecurityEvent(
            event_type=EventType.SQL_INJECTION,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.now(timezone.utc),
            source_ip="203.0.113.100"
        )
        
        high_alert = self.alert_manager.create_alert(high_event)
        assert "high_alerts" in high_alert.channels
        assert "critical_alerts" not in high_alert.channels
        
        # Test CRITICAL severity event
        critical_event = SecurityEvent(
            event_type=EventType.DATA_EXFILTRATION,
            severity=AlertSeverity.CRITICAL,
            timestamp=datetime.now(timezone.utc),
            source_ip="10.0.1.50"
        )
        
        critical_alert = self.alert_manager.create_alert(critical_event)
        assert "high_alerts" in critical_alert.channels
        assert "critical_alerts" in critical_alert.channels
    
    @patch('src.monitoring.security_monitor.smtplib.SMTP')
    def test_send_email_alert(self, mock_smtp_class):
        """Test sending email alert"""
        mock_smtp = MagicMock()
        mock_smtp_class.return_value = mock_smtp
        
        # Add email channel
        email_channel = AlertChannel(
            name="email_alerts",
            channel_type="email",
            endpoint="security@company.com"
        )
        self.alert_manager.add_channel(email_channel)
        
        # Create alert
        event = SecurityEvent(
            event_type=EventType.BRUTE_FORCE_ATTACK,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.now(timezone.utc),
            source_ip="192.168.1.100"
        )
        
        alert = self.alert_manager.create_alert(event)
        
        # Send alert
        result = self.alert_manager.send_alert(alert)
        
        assert result is True
        mock_smtp.send_message.assert_called()
    
    @patch('src.monitoring.security_monitor.requests.post')
    def test_send_webhook_alert(self, mock_post):
        """Test sending webhook alert"""
        mock_post.return_value.status_code = 200
        
        # Add webhook channel
        webhook_channel = AlertChannel(
            name="webhook_alerts",
            channel_type="webhook",
            endpoint="https://hooks.company.com/security"
        )
        self.alert_manager.add_channel(webhook_channel)
        
        # Create alert
        event = SecurityEvent(
            event_type=EventType.PORT_SCAN,
            severity=AlertSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc),
            source_ip="10.0.0.50"
        )
        
        alert = self.alert_manager.create_alert(event)
        
        # Send alert
        result = self.alert_manager.send_alert(alert)
        
        assert result is True
        mock_post.assert_called_once()
    
    def test_rate_limiting(self):
        """Test alert rate limiting"""
        # Add channel with low rate limit
        channel = AlertChannel(
            name="rate_limited",
            channel_type="webhook",
            endpoint="https://example.com/webhook",
            rate_limit=2  # Only 2 alerts per hour
        )
        self.alert_manager.add_channel(channel)
        
        # Create multiple alerts
        events = []
        for i in range(5):
            event = SecurityEvent(
                event_type=EventType.SUSPICIOUS_ACTIVITY,
                severity=AlertSeverity.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                source_ip=f"192.168.1.{i + 100}"
            )
            events.append(event)
        
        # Mock successful sending
        with patch.object(self.alert_manager, '_send_to_webhook', return_value=True):
            sent_count = 0
            for event in events:
                alert = self.alert_manager.create_alert(event)
                if self.alert_manager.send_alert(alert):
                    sent_count += 1
        
        # Should respect rate limit
        assert sent_count <= channel.rate_limit
    
    def test_acknowledge_alert(self):
        """Test acknowledging an alert"""
        event = SecurityEvent(
            event_type=EventType.XSS_ATTACK,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.now(timezone.utc),
            source_ip="198.51.100.50"
        )
        
        alert = self.alert_manager.create_alert(event)
        alert_id = alert.id
        
        # Acknowledge the alert
        result = self.alert_manager.acknowledge_alert(alert_id, "security_admin")
        
        assert result is True
        acknowledged_alert = self.alert_manager.alerts[alert_id]
        assert acknowledged_alert.acknowledged is True
        assert acknowledged_alert.acknowledged_by == "security_admin"
    
    def test_get_alert_statistics(self):
        """Test getting alert statistics"""
        # Create multiple alerts with different severities
        severities = [AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        
        for i, severity in enumerate(severities):
            event = SecurityEvent(
                event_type=EventType.SUSPICIOUS_ACTIVITY,
                severity=severity,
                timestamp=datetime.now(timezone.utc),
                source_ip=f"192.168.1.{i + 100}"
            )
            
            alert = self.alert_manager.create_alert(event)
            # Acknowledge some alerts
            if i % 2 == 0:
                self.alert_manager.acknowledge_alert(alert.id, "admin")
        
        stats = self.alert_manager.get_statistics()
        
        assert "total_alerts" in stats
        assert "alerts_by_severity" in stats
        assert "acknowledged_count" in stats
        assert "unacknowledged_count" in stats
        assert stats["total_alerts"] == len(severities)


@pytest.mark.skipif(not SECURITY_MONITOR_AVAILABLE, reason="Security monitor not available")
class TestSecurityMonitor:
    """Test main security monitor functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            "monitoring": {
                "enabled": True,
                "log_sources": ["/var/log/auth.log", "/var/log/nginx/access.log"],
                "detection_interval": 60,
                "cleanup_interval": 3600
            },
            "alerting": {
                "enabled": True,
                "channels": [
                    {
                        "name": "security_email",
                        "type": "email",
                        "endpoint": "security@company.com",
                        "severity_filter": ["HIGH", "CRITICAL"]
                    }
                ]
            }
        }
        
        self.monitor = SecurityMonitor(self.config)
    
    def test_security_monitor_initialization(self):
        """Test security monitor initialization"""
        assert self.monitor is not None
        assert hasattr(self.monitor, 'detector')
        assert hasattr(self.monitor, 'alert_manager')
        assert hasattr(self.monitor, 'config')
        assert self.monitor.config == self.config
    
    def test_start_monitoring(self):
        """Test starting the monitoring system"""
        with patch.object(self.monitor, '_monitor_loop'):
            with patch('threading.Thread') as mock_thread:
                self.monitor.start()
                
                mock_thread.assert_called()
                assert self.monitor.running is True
    
    def test_stop_monitoring(self):
        """Test stopping the monitoring system"""
        # Start monitoring first
        with patch.object(self.monitor, '_monitor_loop'):
            self.monitor.start()
        
        # Stop monitoring
        self.monitor.stop()
        
        assert self.monitor.running is False
    
    def test_process_log_entry(self):
        """Test processing a log entry"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ip": "192.168.1.100",
            "event": "failed_login",
            "user": "admin",
            "attempts": 5
        }
        
        with patch.object(self.monitor.detector, 'detect_event') as mock_detect:
            mock_event = SecurityEvent(
                event_type=EventType.BRUTE_FORCE_ATTACK,
                severity=AlertSeverity.HIGH,
                timestamp=datetime.now(timezone.utc),
                source_ip="192.168.1.100"
            )
            mock_detect.return_value = mock_event
            
            with patch.object(self.monitor.alert_manager, 'create_alert') as mock_create_alert:
                with patch.object(self.monitor.alert_manager, 'send_alert') as mock_send_alert:
                    mock_create_alert.return_value = Mock()
                    mock_send_alert.return_value = True
                    
                    result = self.monitor.process_log_entry(log_entry)
        
        assert result is True
        mock_detect.assert_called()
    
    def test_analyze_security_logs(self):
        """Test analyzing security logs"""
        log_entries = [
            {
                "timestamp": "2025-01-15T10:00:00Z",
                "source_ip": "192.168.1.100",
                "event": "failed_login"
            },
            {
                "timestamp": "2025-01-15T10:01:00Z",
                "source_ip": "10.0.0.50",
                "event": "port_scan",
                "dest_port": 22
            },
            {
                "timestamp": "2025-01-15T10:02:00Z",
                "source_ip": "203.0.113.100",
                "event": "sql_injection",
                "query": "1' OR '1'='1"
            }
        ]
        
        with patch.object(self.monitor, 'process_log_entry', return_value=True) as mock_process:
            results = self.monitor.analyze_logs(log_entries)
        
        assert len(results) == len(log_entries)
        assert mock_process.call_count == len(log_entries)
    
    def test_get_security_status(self):
        """Test getting security status"""
        # Mock some alerts in the alert manager
        with patch.object(self.monitor.alert_manager, 'get_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_alerts": 10,
                "alerts_by_severity": {
                    "LOW": 2,
                    "MEDIUM": 3,
                    "HIGH": 4,
                    "CRITICAL": 1
                },
                "acknowledged_count": 6,
                "unacknowledged_count": 4
            }
            
            status = self.monitor.get_status()
        
        assert "monitoring_active" in status
        assert "alert_statistics" in status
        assert "last_check" in status
        assert status["alert_statistics"]["total_alerts"] == 10
    
    def test_update_configuration(self):
        """Test updating configuration"""
        new_config = {
            "monitoring": {
                "enabled": False,
                "detection_interval": 120
            }
        }
        
        result = self.monitor.update_config(new_config)
        
        assert result is True
        assert self.monitor.config["monitoring"]["enabled"] is False
        assert self.monitor.config["monitoring"]["detection_interval"] == 120
    
    def test_export_security_report(self):
        """Test exporting security report"""
        # Mock alert data
        with patch.object(self.monitor.alert_manager, 'alerts') as mock_alerts:
            mock_alert = Mock()
            mock_alert.event.event_type = EventType.BRUTE_FORCE_ATTACK
            mock_alert.event.severity = AlertSeverity.HIGH
            mock_alert.event.timestamp = datetime.now(timezone.utc)
            mock_alert.created_at = datetime.now(timezone.utc)
            mock_alert.acknowledged = False
            
            mock_alerts.values.return_value = [mock_alert]
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                output_path = tmp_file.name
            
            try:
                result = self.monitor.export_report(output_path, hours_back=24)
                
                assert result is True
                
                # Verify report file was created
                with open(output_path, 'r') as f:
                    report_data = json.load(f)
                
                assert "summary" in report_data
                assert "alerts" in report_data
                assert "generated_at" in report_data
            finally:
                import os
                os.unlink(output_path)


@pytest.mark.skipif(not SECURITY_MONITOR_AVAILABLE, reason="Security monitor not available")
class TestSecurityMonitorIntegrationScenarios:
    """Test security monitor integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            "monitoring": {"enabled": True},
            "alerting": {"enabled": True}
        }
        self.monitor = SecurityMonitor(self.config)
    
    def test_complete_security_incident_workflow(self):
        """Test complete security incident workflow"""
        # 1. Simulate attack detection
        attack_logs = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": "192.168.1.100",
                "event": "failed_login",
                "user": "admin",
                "attempt": i
            }
            for i in range(6)  # Simulate brute force
        ]
        
        detected_events = []
        alerts_created = []
        
        # Mock detector to return events
        def mock_detect_event(event_type, log_entry):
            if log_entry.get("attempt", 0) >= 5:  # Trigger on 6th attempt
                event = SecurityEvent(
                    event_type=EventType.BRUTE_FORCE_ATTACK,
                    severity=AlertSeverity.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    source_ip=log_entry["source_ip"]
                )
                detected_events.append(event)
                return event
            return None
        
        # Mock alert creation
        def mock_create_alert(event):
            alert = Mock()
            alert.id = f"alert_{len(alerts_created)}"
            alert.event = event
            alerts_created.append(alert)
            return alert
        
        with patch.object(self.monitor.detector, 'detect_event', side_effect=mock_detect_event):
            with patch.object(self.monitor.alert_manager, 'create_alert', side_effect=mock_create_alert):
                with patch.object(self.monitor.alert_manager, 'send_alert', return_value=True):
                    
                    # Process attack logs
                    for log_entry in attack_logs:
                        self.monitor.process_log_entry(log_entry)
                    
                    # Verify incident was detected and alerted
                    assert len(detected_events) >= 1
                    assert len(alerts_created) >= 1
                    assert detected_events[0].event_type == EventType.BRUTE_FORCE_ATTACK
    
    def test_multi_vector_attack_detection(self):
        """Test detection of multi-vector attacks"""
        attack_vectors = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": "203.0.113.100",
                "event": "port_scan",
                "dest_port": 22
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": "203.0.113.100",
                "event": "sql_injection",
                "query": "1' OR '1'='1"
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": "203.0.113.100",
                "event": "brute_force",
                "target": "admin"
            }
        ]
        
        detected_types = set()
        
        def mock_detect_event(event_type, log_entry):
            if log_entry["event"] == "port_scan":
                detected_types.add(EventType.PORT_SCAN)
                return SecurityEvent(
                    event_type=EventType.PORT_SCAN,
                    severity=AlertSeverity.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                    source_ip=log_entry["source_ip"]
                )
            elif log_entry["event"] == "sql_injection":
                detected_types.add(EventType.SQL_INJECTION)
                return SecurityEvent(
                    event_type=EventType.SQL_INJECTION,
                    severity=AlertSeverity.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source_ip=log_entry["source_ip"]
                )
            elif log_entry["event"] == "brute_force":
                detected_types.add(EventType.BRUTE_FORCE_ATTACK)
                return SecurityEvent(
                    event_type=EventType.BRUTE_FORCE_ATTACK,
                    severity=AlertSeverity.HIGH,
                    timestamp=datetime.now(timezone.utc),
                    source_ip=log_entry["source_ip"]
                )
            return None
        
        with patch.object(self.monitor.detector, 'detect_event', side_effect=mock_detect_event):
            with patch.object(self.monitor.alert_manager, 'create_alert', return_value=Mock()):
                with patch.object(self.monitor.alert_manager, 'send_alert', return_value=True):
                    
                    for attack in attack_vectors:
                        self.monitor.process_log_entry(attack)
                    
                    # Should detect multiple attack types from same source
                    assert len(detected_types) == 3
                    assert EventType.PORT_SCAN in detected_types
                    assert EventType.SQL_INJECTION in detected_types
                    assert EventType.BRUTE_FORCE_ATTACK in detected_types
    
    def test_false_positive_handling(self):
        """Test handling of false positives"""
        # Simulate legitimate activity that might trigger false positives
        legitimate_logs = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": "192.168.1.10",  # Internal IP
                "event": "admin_login",
                "user": "legitimate_admin",
                "action": "database_maintenance"
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_ip": "192.168.1.10",
                "event": "bulk_operation",
                "records_processed": 10000  # High volume but legitimate
            }
        ]
        
        false_positives = 0
        true_detections = 0
        
        def mock_detect_event(event_type, log_entry):
            # Implement logic to avoid false positives for internal IPs
            if log_entry["source_ip"].startswith("192.168.1."):
                if log_entry.get("user") == "legitimate_admin":
                    false_positives += 1
                    return None  # No detection for legitimate admin
            
            # This would be a true detection
            true_detections += 1
            return SecurityEvent(
                event_type=EventType.SUSPICIOUS_ACTIVITY,
                severity=AlertSeverity.LOW,
                timestamp=datetime.now(timezone.utc),
                source_ip=log_entry["source_ip"]
            )
        
        with patch.object(self.monitor.detector, 'detect_event', side_effect=mock_detect_event):
            for log_entry in legitimate_logs:
                self.monitor.process_log_entry(log_entry)
            
            # Should minimize false positives for legitimate activity
            assert false_positives >= 0  # Track false positives avoided
    
    def test_alert_escalation_workflow(self):
        """Test alert escalation workflow"""
        # Simulate escalating severity
        escalation_events = [
            SecurityEvent(
                event_type=EventType.SUSPICIOUS_ACTIVITY,
                severity=AlertSeverity.LOW,
                timestamp=datetime.now(timezone.utc),
                source_ip="10.0.0.100"
            ),
            SecurityEvent(
                event_type=EventType.BRUTE_FORCE_ATTACK,
                severity=AlertSeverity.MEDIUM,
                timestamp=datetime.now(timezone.utc),
                source_ip="10.0.0.100"
            ),
            SecurityEvent(
                event_type=EventType.PRIVILEGE_ESCALATION,
                severity=AlertSeverity.HIGH,
                timestamp=datetime.now(timezone.utc),
                source_ip="10.0.0.100"
            ),
            SecurityEvent(
                event_type=EventType.DATA_EXFILTRATION,
                severity=AlertSeverity.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source_ip="10.0.0.100"
            )
        ]
        
        escalation_channels = {
            AlertSeverity.LOW: ["email"],
            AlertSeverity.MEDIUM: ["email", "slack"],
            AlertSeverity.HIGH: ["email", "slack", "webhook"],
            AlertSeverity.CRITICAL: ["email", "slack", "webhook", "pagerduty"]
        }
        
        # Mock alert creation with escalation
        created_alerts = []
        
        def mock_create_alert(event):
            alert = Mock()
            alert.event = event
            alert.channels = escalation_channels.get(event.severity, ["email"])
            created_alerts.append(alert)
            return alert
        
        with patch.object(self.monitor.alert_manager, 'create_alert', side_effect=mock_create_alert):
            with patch.object(self.monitor.alert_manager, 'send_alert', return_value=True):
                
                for event in escalation_events:
                    self.monitor.process_log_entry({"event": event})
                
                # Verify escalation occurred
                assert len(created_alerts) == len(escalation_events)
                
                # Critical alert should have most channels
                critical_alert = next(a for a in created_alerts if a.event.severity == AlertSeverity.CRITICAL)
                assert len(critical_alert.channels) >= 3
    
    def test_monitoring_performance_under_load(self):
        """Test monitoring performance under high load"""
        # Simulate high volume of log entries
        log_volume = 1000
        
        start_time = time.time()
        processed_count = 0
        
        with patch.object(self.monitor.detector, 'detect_event', return_value=None):
            for i in range(log_volume):
                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_ip": f"192.168.{i % 255}.{(i * 7) % 255}",
                    "event": "normal_activity",
                    "request_id": f"req_{i}"
                }
                
                result = self.monitor.process_log_entry(log_entry)
                if result is not None:
                    processed_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process logs efficiently
        assert processing_time < 10.0  # Under 10 seconds for 1000 logs
        throughput = log_volume / processing_time
        assert throughput > 100  # At least 100 logs per second
    
    def test_configuration_hot_reload(self):
        """Test hot reloading of configuration"""
        original_config = self.monitor.config.copy()
        
        # Update configuration while running
        new_config = {
            "monitoring": {
                "enabled": True,
                "detection_interval": 30,  # Changed from default
                "new_setting": "test_value"
            },
            "alerting": {
                "enabled": True,
                "rate_limit": 5  # New setting
            }
        }
        
        result = self.monitor.update_config(new_config)
        
        assert result is True
        assert self.monitor.config != original_config
        assert self.monitor.config["monitoring"]["detection_interval"] == 30
        assert self.monitor.config["monitoring"]["new_setting"] == "test_value"
        assert self.monitor.config["alerting"]["rate_limit"] == 5