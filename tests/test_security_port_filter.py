#!/usr/bin/env python3
"""
Comprehensive tests for Security Port Filter - Phase 7 Coverage

This test module provides comprehensive coverage for the port filtering system
including allowlisting, scan detection, service management, and network firewall.
"""
import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, List

# Import port filter components
try:
    from src.security.port_filter import (
        ServiceType, PortRule, AccessResult, ScanDetectionResult,
        PortFilter, PortScanDetector, ServicePortManager, NetworkFirewall
    )
    PORT_FILTER_AVAILABLE = True
except ImportError:
    PORT_FILTER_AVAILABLE = False


@pytest.mark.skipif(not PORT_FILTER_AVAILABLE, reason="Port filter not available")
class TestPortFilterEnums:
    """Test port filter enums and constants"""
    
    def test_service_type_enum(self):
        """Test ServiceType enum values"""
        assert ServiceType.MCP_SERVER.value == "mcp_server"
        assert ServiceType.NEO4J.value == "neo4j"
        assert ServiceType.QDRANT.value == "qdrant"
        assert ServiceType.REDIS.value == "redis"
        assert ServiceType.HTTPS.value == "https"
        assert ServiceType.HTTP.value == "http"


@pytest.mark.skipif(not PORT_FILTER_AVAILABLE, reason="Port filter not available")
class TestPortFilterDataModels:
    """Test port filter data models"""
    
    def test_port_rule_creation(self):
        """Test PortRule dataclass creation"""
        rule = PortRule(
            port=8000,
            service=ServiceType.MCP_SERVER,
            allowed_sources=["192.168.1.0/24", "10.0.0.0/8"],
            protocol="tcp",
            description="MCP Server access",
            enabled=True
        )
        
        assert rule.port == 8000
        assert rule.service == ServiceType.MCP_SERVER
        assert rule.allowed_sources == ["192.168.1.0/24", "10.0.0.0/8"]
        assert rule.protocol == "tcp"
        assert rule.description == "MCP Server access"
        assert rule.enabled is True
    
    def test_port_rule_defaults(self):
        """Test PortRule default values"""
        rule = PortRule(
            port=443,
            service=ServiceType.HTTPS,
            allowed_sources=["0.0.0.0/0"]
        )
        
        assert rule.protocol == "tcp"
        assert rule.description == ""
        assert rule.enabled is True
    
    def test_access_result_creation(self):
        """Test AccessResult dataclass creation"""
        result = AccessResult(
            allowed=True,
            reason="Port allowed for service",
            service="mcp_server",
            logged=True
        )
        
        assert result.allowed is True
        assert result.reason == "Port allowed for service"
        assert result.service == "mcp_server"
        assert result.logged is True
    
    def test_access_result_defaults(self):
        """Test AccessResult default values"""
        result = AccessResult(allowed=False)
        
        assert result.reason is None
        assert result.service is None
        assert result.logged is False
    
    def test_scan_detection_result_creation(self):
        """Test ScanDetectionResult dataclass creation"""
        result = ScanDetectionResult(
            is_scan=True,
            ports_scanned=25,
            time_window=60,
            action_taken="IP blocked temporarily"
        )
        
        assert result.is_scan is True
        assert result.ports_scanned == 25
        assert result.time_window == 60
        assert result.action_taken == "IP blocked temporarily"
    
    def test_scan_detection_result_defaults(self):
        """Test ScanDetectionResult default values"""
        result = ScanDetectionResult(is_scan=False)
        
        assert result.ports_scanned == 0
        assert result.time_window == 0
        assert result.action_taken is None


@pytest.mark.skipif(not PORT_FILTER_AVAILABLE, reason="Port filter not available")
class TestPortFilter:
    """Test main port filter functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.filter = PortFilter()
    
    def test_port_filter_initialization(self):
        """Test port filter initialization"""
        assert self.filter is not None
        assert hasattr(self.filter, 'allowed_ports')
        assert hasattr(self.filter, 'blocked_ports')
        assert hasattr(self.filter, 'port_descriptions')
        
        # Should have default allowed ports
        assert 8000 in self.filter.allowed_ports  # MCP Server
        assert 7687 in self.filter.allowed_ports  # Neo4j
        assert 6333 in self.filter.allowed_ports  # Qdrant
        assert 6379 in self.filter.allowed_ports  # Redis
        assert 443 in self.filter.allowed_ports   # HTTPS
        assert 80 in self.filter.allowed_ports    # HTTP
    
    def test_port_filter_with_custom_config(self):
        """Test port filter with custom configuration"""
        config = {
            "allowed_ports": [
                {"port": 9000, "description": "Custom service"},
                {"port": 9001, "description": "Another service"},
                3000  # Port only
            ]
        }
        
        filter_custom = PortFilter(config)
        
        # Should have default ports plus custom ones
        assert 9000 in filter_custom.allowed_ports
        assert 9001 in filter_custom.allowed_ports
        assert 3000 in filter_custom.allowed_ports
        
        # Should have descriptions
        assert filter_custom.port_descriptions[9000] == "Custom service"
        assert filter_custom.port_descriptions[9001] == "Another service"
    
    def test_is_port_allowed(self):
        """Test port allowlist checking"""
        # Test allowed ports
        assert self.filter.is_port_allowed(8000) is True  # MCP Server
        assert self.filter.is_port_allowed(443) is True   # HTTPS
        
        # Test blocked ports
        assert self.filter.is_port_allowed(22) is False   # SSH
        assert self.filter.is_port_allowed(23) is False   # Telnet
        
        # Test random high port (should be blocked by default)
        assert self.filter.is_port_allowed(12345) is False
    
    def test_add_allowed_port(self):
        """Test adding allowed ports dynamically"""
        # Port should not be allowed initially
        assert self.filter.is_port_allowed(9999) is False
        
        # Add port
        self.filter.add_allowed_port(9999, "Test service")
        
        # Port should now be allowed
        assert self.filter.is_port_allowed(9999) is True
        assert self.filter.port_descriptions[9999] == "Test service"
    
    def test_remove_allowed_port(self):
        """Test removing allowed ports"""
        # Add a port first
        self.filter.add_allowed_port(8888, "Temporary service")
        assert self.filter.is_port_allowed(8888) is True
        
        # Remove the port
        self.filter.remove_allowed_port(8888)
        assert self.filter.is_port_allowed(8888) is False
        assert 8888 not in self.filter.port_descriptions
    
    def test_get_port_description(self):
        """Test getting port descriptions"""
        # Test default ports
        assert "MCP Server" in self.filter.get_port_description(8000)
        assert "Neo4j" in self.filter.get_port_description(7687)
        assert "HTTPS" in self.filter.get_port_description(443)
        
        # Test unknown port
        description = self.filter.get_port_description(99999)
        assert "Unknown" in description or description == ""
    
    def test_list_allowed_ports(self):
        """Test listing all allowed ports"""
        allowed_ports = self.filter.list_allowed_ports()
        
        assert isinstance(allowed_ports, dict)
        assert 8000 in allowed_ports
        assert 443 in allowed_ports
        assert len(allowed_ports) >= 6  # At least the default ports
    
    def test_is_port_blocked(self):
        """Test blocked port checking"""
        # Test known blocked ports
        assert self.filter.is_port_blocked(22) is True    # SSH
        assert self.filter.is_port_blocked(23) is True    # Telnet
        assert self.filter.is_port_blocked(21) is True    # FTP
        
        # Test allowed ports should not be blocked
        assert self.filter.is_port_blocked(443) is False  # HTTPS
        assert self.filter.is_port_blocked(8000) is False # MCP Server
    
    def test_validate_port_range(self):
        """Test port range validation"""
        # Valid ports
        assert self.filter.validate_port(80) is True
        assert self.filter.validate_port(443) is True
        assert self.filter.validate_port(8000) is True
        assert self.filter.validate_port(65535) is True
        
        # Invalid ports
        assert self.filter.validate_port(0) is False
        assert self.filter.validate_port(-1) is False
        assert self.filter.validate_port(65536) is False
        assert self.filter.validate_port(100000) is False
    
    def test_bulk_port_operations(self):
        """Test bulk port operations"""
        ports_to_add = [
            {"port": 5000, "description": "Service A"},
            {"port": 5001, "description": "Service B"},
            {"port": 5002, "description": "Service C"}
        ]
        
        # Add multiple ports
        self.filter.add_allowed_ports(ports_to_add)
        
        # Verify all were added
        for port_info in ports_to_add:
            port = port_info["port"]
            assert self.filter.is_port_allowed(port) is True
            assert self.filter.port_descriptions[port] == port_info["description"]
        
        # Remove multiple ports
        ports_to_remove = [5000, 5001, 5002]
        self.filter.remove_allowed_ports(ports_to_remove)
        
        # Verify all were removed
        for port in ports_to_remove:
            assert self.filter.is_port_allowed(port) is False


@pytest.mark.skipif(not PORT_FILTER_AVAILABLE, reason="Port filter not available")
class TestPortScanDetector:
    """Test port scan detection functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.detector = PortScanDetector()
    
    def test_detector_initialization(self):
        """Test scan detector initialization"""
        assert self.detector is not None
        assert hasattr(self.detector, 'connection_history')
        assert hasattr(self.detector, 'scan_threshold')
        assert hasattr(self.detector, 'time_window')
    
    def test_single_connection_tracking(self):
        """Test tracking single connections"""
        client_ip = "192.168.1.100"
        port = 8000
        
        # Record a connection
        self.detector.record_connection(client_ip, port)
        
        # Should not be detected as scan (single connection)
        result = self.detector.detect_scan(client_ip)
        assert result.is_scan is False
    
    def test_multiple_port_scan_detection(self):
        """Test detection of port scans"""
        client_ip = "10.0.0.50"
        
        # Simulate port scan - try many ports quickly
        scan_ports = [80, 443, 22, 23, 21, 25, 53, 110, 143, 993, 995, 8000, 8080, 9000]
        
        for port in scan_ports:
            self.detector.record_connection(client_ip, port)
        
        # Should detect as port scan
        result = self.detector.detect_scan(client_ip)
        assert result.is_scan is True
        assert result.ports_scanned >= len(scan_ports)
    
    def test_time_window_scan_detection(self):
        """Test scan detection within time window"""
        client_ip = "172.16.0.10"
        
        # Record connections to many ports within short time
        start_time = time.time()
        for port in range(1000, 1020):  # 20 ports
            self.detector.record_connection(client_ip, port, timestamp=start_time)
        
        result = self.detector.detect_scan(client_ip)
        assert result.is_scan is True
        assert result.time_window <= self.detector.time_window
    
    def test_legitimate_traffic_not_flagged(self):
        """Test that legitimate traffic is not flagged as scan"""
        client_ip = "192.168.100.5"
        
        # Simulate legitimate connections to allowed services
        legitimate_ports = [443, 80, 8000]  # HTTPS, HTTP, MCP Server
        
        # Multiple connections to same services over time
        for _ in range(10):
            for port in legitimate_ports:
                self.detector.record_connection(client_ip, port)
            time.sleep(0.1)  # Small delay
        
        result = self.detector.detect_scan(client_ip)
        # Should not be flagged as scan (legitimate repeated connections)
        assert result.is_scan is False
    
    def test_scan_detection_with_whitelist(self):
        """Test scan detection with IP whitelist"""
        # Configure detector with whitelist
        config = {"whitelist_ips": ["192.168.1.0/24", "10.0.0.100"]}
        detector_with_whitelist = PortScanDetector(config)
        
        whitelisted_ip = "192.168.1.50"
        
        # Simulate port scan from whitelisted IP
        for port in range(2000, 2030):
            detector_with_whitelist.record_connection(whitelisted_ip, port)
        
        result = detector_with_whitelist.detect_scan(whitelisted_ip)
        # Whitelisted IPs should not trigger scan detection
        assert result.is_scan is False
    
    def test_scan_history_cleanup(self):
        """Test cleanup of old scan history"""
        client_ip = "203.0.113.10"
        
        # Record old connections
        old_timestamp = time.time() - 3600  # 1 hour ago
        for port in range(3000, 3010):
            self.detector.record_connection(client_ip, port, timestamp=old_timestamp)
        
        # Clean up old history
        self.detector.cleanup_old_connections()
        
        # Old connections should be cleaned up
        if client_ip in self.detector.connection_history:
            recent_connections = [
                conn for conn in self.detector.connection_history[client_ip]
                if time.time() - conn["timestamp"] < self.detector.time_window
            ]
            assert len(recent_connections) == 0
    
    def test_concurrent_scan_detection(self):
        """Test detection of scans from multiple IPs"""
        scan_ips = ["198.51.100.10", "198.51.100.20", "198.51.100.30"]
        
        # Simulate coordinated port scan
        for ip in scan_ips:
            for port in range(4000, 4020):
                self.detector.record_connection(ip, port)
        
        # All IPs should be detected as scanning
        for ip in scan_ips:
            result = self.detector.detect_scan(ip)
            assert result.is_scan is True
    
    def test_scan_severity_assessment(self):
        """Test scan severity assessment"""
        high_severity_ip = "1.2.3.4"
        low_severity_ip = "5.6.7.8"
        
        # High severity: many ports, quick succession
        for port in range(5000, 5100):  # 100 ports
            self.detector.record_connection(high_severity_ip, port)
        
        # Low severity: fewer ports
        for port in range(6000, 6010):  # 10 ports
            self.detector.record_connection(low_severity_ip, port)
        
        high_result = self.detector.detect_scan(high_severity_ip)
        low_result = self.detector.detect_scan(low_severity_ip)
        
        assert high_result.is_scan is True
        assert low_result.is_scan is True  # Still a scan, but less severe
        assert high_result.ports_scanned > low_result.ports_scanned


@pytest.mark.skipif(not PORT_FILTER_AVAILABLE, reason="Port filter not available")
class TestServicePortManager:
    """Test service port management functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.manager = ServicePortManager()
    
    def test_manager_initialization(self):
        """Test service port manager initialization"""
        assert self.manager is not None
        assert hasattr(self.manager, 'service_ports')
        assert hasattr(self.manager, 'port_rules')
    
    def test_register_service(self):
        """Test registering a service"""
        rule = PortRule(
            port=9000,
            service=ServiceType.MCP_SERVER,
            allowed_sources=["192.168.1.0/24"],
            description="Custom MCP Server"
        )
        
        result = self.manager.register_service("custom_mcp", rule)
        assert result is True
        
        # Service should be registered
        assert "custom_mcp" in self.manager.service_ports
        assert self.manager.service_ports["custom_mcp"] == 9000
    
    def test_unregister_service(self):
        """Test unregistering a service"""
        # Register first
        rule = PortRule(
            port=9001,
            service=ServiceType.REDIS,
            allowed_sources=["10.0.0.0/8"]
        )
        self.manager.register_service("temp_redis", rule)
        assert "temp_redis" in self.manager.service_ports
        
        # Unregister
        result = self.manager.unregister_service("temp_redis")
        assert result is True
        assert "temp_redis" not in self.manager.service_ports
    
    def test_check_service_access(self):
        """Test checking service access"""
        # Register a service with restricted access
        rule = PortRule(
            port=7000,
            service=ServiceType.NEO4J,
            allowed_sources=["192.168.1.0/24", "10.0.0.100"],
            description="Restricted Neo4j"
        )
        self.manager.register_service("restricted_neo4j", rule)
        
        # Test allowed source
        result_allowed = self.manager.check_access("192.168.1.50", 7000)
        assert result_allowed.allowed is True
        
        # Test denied source
        result_denied = self.manager.check_access("203.0.113.10", 7000)
        assert result_denied.allowed is False
    
    def test_service_health_monitoring(self):
        """Test service health monitoring"""
        # Register services
        services = [
            ("web_server", PortRule(80, ServiceType.HTTP, ["0.0.0.0/0"])),
            ("api_server", PortRule(8080, ServiceType.MCP_SERVER, ["192.168.1.0/24"])),
            ("db_server", PortRule(5432, ServiceType.NEO4J, ["10.0.0.0/8"]))
        ]
        
        for service_name, rule in services:
            self.manager.register_service(service_name, rule)
        
        # Check service health
        health_status = self.manager.get_service_health()
        
        assert isinstance(health_status, dict)
        assert len(health_status) >= len(services)
        
        for service_name, _ in services:
            assert service_name in health_status
    
    def test_service_metrics_collection(self):
        """Test service metrics collection"""
        # Register a service
        rule = PortRule(
            port=8888,
            service=ServiceType.MCP_SERVER,
            allowed_sources=["0.0.0.0/0"]
        )
        self.manager.register_service("metrics_test", rule)
        
        # Simulate some connections
        for i in range(10):
            self.manager.check_access(f"192.168.1.{i}", 8888)
        
        # Get metrics
        metrics = self.manager.get_service_metrics("metrics_test")
        
        assert isinstance(metrics, dict)
        assert "connections_total" in metrics
        assert metrics["connections_total"] >= 10
    
    def test_dynamic_rule_updates(self):
        """Test dynamic updating of service rules"""
        # Register initial service
        initial_rule = PortRule(
            port=7777,
            service=ServiceType.QDRANT,
            allowed_sources=["192.168.1.0/24"]
        )
        self.manager.register_service("dynamic_service", initial_rule)
        
        # Initial access check
        result1 = self.manager.check_access("192.168.1.100", 7777)
        assert result1.allowed is True
        
        result2 = self.manager.check_access("10.0.0.100", 7777)
        assert result2.allowed is False
        
        # Update rule to allow more sources
        updated_rule = PortRule(
            port=7777,
            service=ServiceType.QDRANT,
            allowed_sources=["192.168.1.0/24", "10.0.0.0/8"]
        )
        self.manager.update_service_rule("dynamic_service", updated_rule)
        
        # Access should now be allowed
        result3 = self.manager.check_access("10.0.0.100", 7777)
        assert result3.allowed is True
    
    def test_service_port_conflicts(self):
        """Test detection of service port conflicts"""
        # Register first service
        rule1 = PortRule(
            port=6666,
            service=ServiceType.REDIS,
            allowed_sources=["192.168.1.0/24"]
        )
        result1 = self.manager.register_service("service1", rule1)
        assert result1 is True
        
        # Try to register conflicting service
        rule2 = PortRule(
            port=6666,  # Same port
            service=ServiceType.QDRANT,
            allowed_sources=["10.0.0.0/8"]
        )
        result2 = self.manager.register_service("service2", rule2)
        assert result2 is False  # Should fail due to port conflict
    
    def test_bulk_service_operations(self):
        """Test bulk service operations"""
        services = [
            ("bulk1", PortRule(8001, ServiceType.HTTP, ["0.0.0.0/0"])),
            ("bulk2", PortRule(8002, ServiceType.HTTPS, ["0.0.0.0/0"])),
            ("bulk3", PortRule(8003, ServiceType.MCP_SERVER, ["192.168.1.0/24"]))
        ]
        
        # Bulk register
        results = self.manager.register_services_bulk(services)
        assert all(results.values())
        
        # Verify all registered
        for service_name, _ in services:
            assert service_name in self.manager.service_ports
        
        # Bulk unregister
        service_names = [name for name, _ in services]
        unregister_results = self.manager.unregister_services_bulk(service_names)
        assert all(unregister_results.values())
        
        # Verify all unregistered
        for service_name in service_names:
            assert service_name not in self.manager.service_ports


@pytest.mark.skipif(not PORT_FILTER_AVAILABLE, reason="Port filter not available")
class TestNetworkFirewall:
    """Test network firewall functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.firewall = NetworkFirewall()
    
    def test_firewall_initialization(self):
        """Test firewall initialization"""
        assert self.firewall is not None
        assert hasattr(self.firewall, 'port_filter')
        assert hasattr(self.firewall, 'scan_detector')
        assert hasattr(self.firewall, 'service_manager')
    
    @patch('src.security.port_filter.subprocess.run')
    def test_apply_firewall_rules(self, mock_subprocess):
        """Test applying firewall rules"""
        # Mock successful iptables execution
        mock_subprocess.return_value.returncode = 0
        
        result = self.firewall.apply_rules()
        assert result is True
        
        # Should have called iptables commands
        assert mock_subprocess.called
    
    @patch('src.security.port_filter.subprocess.run')
    def test_block_ip_address(self, mock_subprocess):
        """Test blocking IP address"""
        mock_subprocess.return_value.returncode = 0
        
        suspicious_ip = "203.0.113.100"
        result = self.firewall.block_ip(suspicious_ip, duration=300)
        
        assert result is True
        mock_subprocess.assert_called()
    
    @patch('src.security.port_filter.subprocess.run')
    def test_unblock_ip_address(self, mock_subprocess):
        """Test unblocking IP address"""
        mock_subprocess.return_value.returncode = 0
        
        blocked_ip = "203.0.113.200"
        
        # Block first
        self.firewall.block_ip(blocked_ip)
        
        # Then unblock
        result = self.firewall.unblock_ip(blocked_ip)
        assert result is True
    
    def test_comprehensive_access_check(self):
        """Test comprehensive access checking"""
        client_ip = "192.168.1.100"
        port = 8000
        
        # Should integrate port filter, scan detection, and service management
        result = self.firewall.check_access(client_ip, port)
        
        assert isinstance(result, AccessResult)
        assert isinstance(result.allowed, bool)
        assert result.logged is not None
    
    def test_firewall_rule_generation(self):
        """Test firewall rule generation"""
        rules = self.firewall.generate_iptables_rules()
        
        assert isinstance(rules, list)
        assert len(rules) > 0
        
        # Should contain basic rules
        rule_text = "\n".join(rules)
        assert "ACCEPT" in rule_text
        assert "DROP" in rule_text or "REJECT" in rule_text
    
    def test_firewall_status_monitoring(self):
        """Test firewall status monitoring"""
        status = self.firewall.get_status()
        
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "rules_count" in status
        assert "blocked_ips" in status
    
    def test_emergency_lockdown(self):
        """Test emergency lockdown functionality"""
        # Simulate emergency situation
        result = self.firewall.emergency_lockdown()
        
        assert isinstance(result, bool)
        
        # In lockdown, only essential services should be accessible
        lockdown_status = self.firewall.get_lockdown_status()
        assert isinstance(lockdown_status, dict)
        assert "active" in lockdown_status
    
    def test_firewall_logging(self):
        """Test firewall logging functionality"""
        # Enable logging
        self.firewall.enable_logging()
        
        # Simulate some access attempts
        test_accesses = [
            ("192.168.1.50", 8000),   # Allowed
            ("10.0.0.100", 443),      # Allowed
            ("203.0.113.50", 22),     # Blocked
        ]
        
        for ip, port in test_accesses:
            self.firewall.check_access(ip, port)
        
        # Get access logs
        logs = self.firewall.get_access_logs(limit=10)
        
        assert isinstance(logs, list)
        assert len(logs) >= len(test_accesses)
    
    def test_automated_threat_response(self):
        """Test automated threat response"""
        malicious_ip = "198.51.100.100"
        
        # Simulate port scan
        for port in range(1000, 1050):  # 50 ports
            self.firewall.scan_detector.record_connection(malicious_ip, port)
        
        # Check if threat is detected and response triggered
        scan_result = self.firewall.scan_detector.detect_scan(malicious_ip)
        assert scan_result.is_scan is True
        
        # Automated response should trigger
        response_result = self.firewall.handle_detected_threat(malicious_ip, scan_result)
        assert isinstance(response_result, dict)
        assert "action_taken" in response_result


@pytest.mark.skipif(not PORT_FILTER_AVAILABLE, reason="Port filter not available")
class TestPortFilterIntegrationScenarios:
    """Test integrated port filter scenarios"""
    
    def test_complete_network_protection_workflow(self):
        """Test complete network protection workflow"""
        firewall = NetworkFirewall()
        
        # 1. Configure allowed services
        mcp_rule = PortRule(
            port=8000,
            service=ServiceType.MCP_SERVER,
            allowed_sources=["192.168.1.0/24"],
            description="Production MCP Server"
        )
        firewall.service_manager.register_service("prod_mcp", mcp_rule)
        
        # 2. Test legitimate access
        legitimate_ip = "192.168.1.50"
        access_result = firewall.check_access(legitimate_ip, 8000)
        assert access_result.allowed is True
        
        # 3. Test unauthorized access
        unauthorized_ip = "203.0.113.100"
        access_result = firewall.check_access(unauthorized_ip, 8000)
        assert access_result.allowed is False
        
        # 4. Simulate port scan
        for port in range(8000, 8050):
            firewall.scan_detector.record_connection(unauthorized_ip, port)
        
        # 5. Detect and respond to threat
        scan_result = firewall.scan_detector.detect_scan(unauthorized_ip)
        assert scan_result.is_scan is True
        
        response = firewall.handle_detected_threat(unauthorized_ip, scan_result)
        assert response["action_taken"] is not None
    
    def test_multi_service_environment_protection(self):
        """Test protection for multi-service environment"""
        manager = ServicePortManager()
        
        # Configure multiple services
        services = [
            ("web", PortRule(80, ServiceType.HTTP, ["0.0.0.0/0"])),
            ("api", PortRule(8000, ServiceType.MCP_SERVER, ["192.168.1.0/24"])),
            ("database", PortRule(7687, ServiceType.NEO4J, ["10.0.0.0/8"])),
            ("cache", PortRule(6379, ServiceType.REDIS, ["192.168.1.0/24"])),
            ("search", PortRule(6333, ServiceType.QDRANT, ["192.168.1.0/24"]))
        ]
        
        for service_name, rule in services:
            result = manager.register_service(service_name, rule)
            assert result is True
        
        # Test access from different network segments
        test_cases = [
            ("192.168.1.100", 80, True),    # Public web access
            ("192.168.1.100", 8000, True),  # Internal API access
            ("10.0.0.50", 7687, True),      # Database access from app network
            ("203.0.113.50", 8000, False),  # External API access (denied)
            ("203.0.113.50", 7687, False),  # External database access (denied)
        ]
        
        for client_ip, port, expected_allowed in test_cases:
            result = manager.check_access(client_ip, port)
            assert result.allowed == expected_allowed
    
    def test_dynamic_threat_adaptation(self):
        """Test dynamic adaptation to threats"""
        firewall = NetworkFirewall()
        detector = firewall.scan_detector
        
        # Simulate escalating threat
        threat_ips = [f"198.51.100.{i}" for i in range(10, 20)]
        
        # Phase 1: Single IP scanning
        for port in range(2000, 2050):
            detector.record_connection(threat_ips[0], port)
        
        scan1 = detector.detect_scan(threat_ips[0])
        assert scan1.is_scan is True
        
        # Phase 2: Coordinated scanning from multiple IPs
        for ip in threat_ips:
            for port in range(3000, 3020):
                detector.record_connection(ip, port)
        
        # All IPs should be detected
        detected_count = 0
        for ip in threat_ips:
            if detector.detect_scan(ip).is_scan:
                detected_count += 1
        
        assert detected_count >= len(threat_ips) * 0.8  # At least 80% detected
    
    def test_performance_under_load(self):
        """Test performance under high load"""
        firewall = NetworkFirewall()
        
        # Simulate high volume of connection checks
        import time
        start_time = time.time()
        
        for i in range(1000):
            client_ip = f"192.168.{i % 255}.{(i * 7) % 255}"
            port = 8000 + (i % 100)
            firewall.check_access(client_ip, port)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle 1000 checks reasonably quickly
        assert duration < 10.0  # Less than 10 seconds
        
        # Performance metrics should be reasonable
        avg_time_per_check = duration / 1000
        assert avg_time_per_check < 0.01  # Less than 10ms per check
    
    def test_configuration_persistence(self):
        """Test configuration persistence and recovery"""
        manager = ServicePortManager()
        
        # Configure services
        original_services = [
            ("service1", PortRule(9001, ServiceType.HTTP, ["192.168.1.0/24"])),
            ("service2", PortRule(9002, ServiceType.HTTPS, ["10.0.0.0/8"]))
        ]
        
        for service_name, rule in original_services:
            manager.register_service(service_name, rule)
        
        # Export configuration
        config = manager.export_configuration()
        assert isinstance(config, dict)
        assert len(config) >= len(original_services)
        
        # Create new manager and import configuration
        new_manager = ServicePortManager()
        import_result = new_manager.import_configuration(config)
        assert import_result is True
        
        # Verify services were restored
        for service_name, _ in original_services:
            assert service_name in new_manager.service_ports
    
    def test_security_audit_compliance(self):
        """Test security audit and compliance features"""
        firewall = NetworkFirewall()
        
        # Enable comprehensive logging
        firewall.enable_audit_logging()
        
        # Simulate various activities
        activities = [
            ("192.168.1.100", 8000, "legitimate_access"),
            ("203.0.113.50", 22, "blocked_ssh_attempt"),
            ("10.0.0.200", 443, "https_access"),
            ("198.51.100.100", 1234, "suspicious_port")
        ]
        
        for ip, port, activity_type in activities:
            result = firewall.check_access(ip, port)
            firewall.log_security_event(ip, port, result, activity_type)
        
        # Generate audit report
        audit_report = firewall.generate_audit_report()
        
        assert isinstance(audit_report, dict)
        assert "summary" in audit_report
        assert "security_events" in audit_report
        assert "compliance_status" in audit_report
        
        # Verify audit trail completeness
        assert len(audit_report["security_events"]) >= len(activities)