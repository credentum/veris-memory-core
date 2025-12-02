"""
Port Filtering and Allowlisting Implementation
Sprint 10 - Issue 003: WAF & Port Allowlisting
"""

import logging
import time
import ipaddress
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service types for port management"""
    MCP_SERVER = "mcp_server"
    NEO4J = "neo4j"
    QDRANT = "qdrant"
    REDIS = "redis"
    HTTPS = "https"
    HTTP = "http"


@dataclass
class PortRule:
    """Port access rule definition"""
    port: int
    service: ServiceType
    allowed_sources: List[str]  # CIDR blocks or IP addresses
    protocol: str = "tcp"
    description: str = ""
    enabled: bool = True


@dataclass
class AccessResult:
    """Result of port access check"""
    allowed: bool
    reason: Optional[str] = None
    service: Optional[str] = None
    logged: bool = False


@dataclass
class ScanDetectionResult:
    """Result of port scan detection"""
    is_scan: bool
    ports_scanned: int = 0
    time_window: int = 0
    action_taken: Optional[str] = None


class PortFilter:
    """Port filtering and allowlisting manager"""
    
    # Default allowed ports
    DEFAULT_ALLOWED_PORTS = {
        8000: "MCP Server",
        7687: "Neo4j Bolt",
        6333: "Qdrant",
        6379: "Redis",
        443: "HTTPS",
        80: "HTTP Redirect"
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize port filter"""
        self.config = config or {}
        self.allowed_ports: Set[int] = set()
        self.port_descriptions: Dict[int, str] = {}
        self.blocked_ports: Set[int] = set()
        self._initialize_ports()
    
    def _initialize_ports(self):
        """Initialize allowed and blocked ports"""
        # Add default allowed ports
        for port, description in self.DEFAULT_ALLOWED_PORTS.items():
            self.allowed_ports.add(port)
            self.port_descriptions[port] = description
        
        # Add custom allowed ports from config
        custom_allowed = self.config.get("allowed_ports", [])
        for port in custom_allowed:
            if isinstance(port, dict):
                self.allowed_ports.add(port["port"])
                self.port_descriptions[port["port"]] = port.get("description", "")
            else:
                self.allowed_ports.add(port)
        
        # Define commonly blocked ports
        self.blocked_ports = {
            22,     # SSH (use bastion instead)
            23,     # Telnet (insecure)
            21,     # FTP (insecure)
            20,     # FTP Data
            25,     # SMTP (mail relay)
            110,    # POP3
            143,    # IMAP
            3306,   # MySQL
            5432,   # PostgreSQL
            27017,  # MongoDB
            1433,   # MS SQL Server
            3389,   # RDP
            5900,   # VNC
            135,    # RPC
            139,    # NetBIOS
            445,    # SMB
            1521,   # Oracle
            5984,   # CouchDB
            9200,   # Elasticsearch
            11211,  # Memcached
            27018,  # MongoDB (sharding)
            27019,  # MongoDB (config)
        }
    
    def is_allowed(self, port: int) -> bool:
        """Check if a port is allowed"""
        # Explicitly blocked ports
        if port in self.blocked_ports:
            return False
        
        # Check allowed list
        return port in self.allowed_ports
    
    def add_allowed_port(self, port: int, description: str = ""):
        """Add a port to the allowed list"""
        self.allowed_ports.add(port)
        if description:
            self.port_descriptions[port] = description
        logger.info(f"Added allowed port: {port} ({description})")
    
    def remove_allowed_port(self, port: int):
        """Remove a port from the allowed list"""
        if port in self.allowed_ports:
            self.allowed_ports.remove(port)
            if port in self.port_descriptions:
                del self.port_descriptions[port]
            logger.info(f"Removed allowed port: {port}")
    
    def get_allowed_ports(self) -> List[Tuple[int, str]]:
        """Get list of allowed ports with descriptions"""
        return [
            (port, self.port_descriptions.get(port, ""))
            for port in sorted(self.allowed_ports)
        ]
    
    def get_blocked_ports(self) -> List[int]:
        """Get list of explicitly blocked ports"""
        return sorted(self.blocked_ports)


class PortScanDetector:
    """Detect and prevent port scanning attempts"""
    
    def __init__(
        self,
        threshold_ports: int = 10,
        time_window: int = 60,
        block_duration: int = 3600
    ):
        """
        Initialize port scan detector.
        
        Args:
            threshold_ports: Number of different ports to trigger detection
            time_window: Time window in seconds for detection
            block_duration: How long to block detected scanners (seconds)
        """
        self.threshold_ports = threshold_ports
        self.time_window = time_window
        self.block_duration = block_duration
        
        # Track access attempts per IP
        self.access_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}
        self.scan_attempts: Dict[str, int] = defaultdict(int)
    
    def check_access(self, source_ip: str, port: int) -> bool:
        """
        Check if access indicates port scanning.
        
        Args:
            source_ip: Source IP address
            port: Port being accessed
            
        Returns:
            True if scanning detected, False otherwise
        """
        current_time = time.time()
        
        # Check if IP is already blocked
        if self.is_blocked(source_ip):
            return True
        
        # Clean old access history
        self._clean_history(source_ip, current_time)
        
        # Record this access
        self.access_history[source_ip].append((port, current_time))
        
        # Get unique ports accessed in time window
        recent_ports = set()
        for accessed_port, timestamp in self.access_history[source_ip]:
            if current_time - timestamp <= self.time_window:
                recent_ports.add(accessed_port)
        
        # Check if threshold exceeded
        if len(recent_ports) >= self.threshold_ports:
            # Port scanning detected
            self._block_ip(source_ip, current_time)
            logger.warning(
                f"Port scan detected from {source_ip}: "
                f"{len(recent_ports)} ports in {self.time_window}s"
            )
            return True
        
        return False
    
    def is_blocked(self, source_ip: str) -> bool:
        """Check if an IP is currently blocked"""
        if source_ip in self.blocked_ips:
            block_until = self.blocked_ips[source_ip]
            if time.time() < block_until:
                return True
            else:
                # Unblock expired
                del self.blocked_ips[source_ip]
                logger.info(f"Unblocked IP {source_ip} after timeout")
        return False
    
    def _block_ip(self, source_ip: str, current_time: float):
        """Block an IP address"""
        self.blocked_ips[source_ip] = current_time + self.block_duration
        self.scan_attempts[source_ip] += 1
        
        # Increase block duration for repeat offenders
        if self.scan_attempts[source_ip] > 3:
            self.blocked_ips[source_ip] = current_time + (self.block_duration * 4)
            logger.warning(f"Extended block for repeat scanner: {source_ip}")
    
    def _clean_history(self, source_ip: str, current_time: float):
        """Clean old access history"""
        if source_ip in self.access_history:
            self.access_history[source_ip] = [
                (port, ts) for port, ts in self.access_history[source_ip]
                if current_time - ts <= self.time_window
            ]
    
    def get_blocked_ips(self) -> List[str]:
        """Get list of currently blocked IPs"""
        current_time = time.time()
        blocked = []
        for ip, block_until in self.blocked_ips.items():
            if current_time < block_until:
                blocked.append(ip)
        return blocked
    
    def unblock_ip(self, source_ip: str):
        """Manually unblock an IP"""
        if source_ip in self.blocked_ips:
            del self.blocked_ips[source_ip]
            logger.info(f"Manually unblocked IP: {source_ip}")
    
    def cleanup_expired(self):
        """Clean up expired IP blocks"""
        current_time = time.time()
        expired_ips = []
        
        for ip, block_until in self.blocked_ips.items():
            if current_time >= block_until:
                expired_ips.append(ip)
        
        for ip in expired_ips:
            del self.blocked_ips[ip]
            logger.info(f"Cleanup: Unblocked expired IP {ip}")
        
        return len(expired_ips)


class ServicePortManager:
    """Manage service-specific port access rules"""
    
    def __init__(self):
        """Initialize service port manager"""
        self.service_rules: Dict[str, PortRule] = {}
        self._initialize_service_rules()
    
    def _initialize_service_rules(self):
        """Initialize default service rules"""
        default_rules = [
            # MCP Server - accessible from app servers
            PortRule(
                port=8000,
                service=ServiceType.MCP_SERVER,
                allowed_sources=["10.0.1.0/24", "127.0.0.1/32"],
                description="MCP Server API"
            ),
            
            # Neo4j - internal only
            PortRule(
                port=7687,
                service=ServiceType.NEO4J,
                allowed_sources=["10.0.1.0/24", "10.0.2.0/24", "127.0.0.1/32"],
                description="Neo4j Bolt Protocol"
            ),
            
            # Qdrant - internal only
            PortRule(
                port=6333,
                service=ServiceType.QDRANT,
                allowed_sources=["10.0.1.0/24", "10.0.2.0/24", "127.0.0.1/32"],
                description="Qdrant Vector DB"
            ),
            
            # Redis - internal only
            PortRule(
                port=6379,
                service=ServiceType.REDIS,
                allowed_sources=["10.0.2.0/24", "127.0.0.1/32"],
                description="Redis Cache"
            ),
            
            # HTTPS - public
            PortRule(
                port=443,
                service=ServiceType.HTTPS,
                allowed_sources=["0.0.0.0/0"],
                description="HTTPS Traffic"
            ),
            
            # HTTP - redirect only
            PortRule(
                port=80,
                service=ServiceType.HTTP,
                allowed_sources=["0.0.0.0/0"],
                description="HTTP Redirect to HTTPS"
            ),
        ]
        
        for rule in default_rules:
            self.service_rules[rule.service.value] = rule
    
    def check_service_access(
        self,
        service: str,
        source_ip: str,
        port: int
    ) -> AccessResult:
        """
        Check if source IP can access service on port.
        
        Args:
            service: Service name
            source_ip: Source IP address
            port: Port number
            
        Returns:
            AccessResult with access decision
        """
        # Get service rule
        if service not in self.service_rules:
            return AccessResult(
                allowed=False,
                reason="Unknown service",
                service=service
            )
        
        rule = self.service_rules[service]
        
        # Check if rule is enabled
        if not rule.enabled:
            return AccessResult(
                allowed=False,
                reason="Service disabled",
                service=service
            )
        
        # Check port matches
        if rule.port != port:
            return AccessResult(
                allowed=False,
                reason=f"Invalid port for service {service}",
                service=service
            )
        
        # Check source IP against allowed sources
        try:
            source_addr = ipaddress.ip_address(source_ip)
            
            for allowed_source in rule.allowed_sources:
                # Handle special case for any source
                if allowed_source == "0.0.0.0/0":
                    return AccessResult(
                        allowed=True,
                        reason="Public access allowed",
                        service=service,
                        logged=True
                    )
                
                # Check if source IP is in allowed network
                network = ipaddress.ip_network(allowed_source, strict=False)
                if source_addr in network:
                    return AccessResult(
                        allowed=True,
                        reason=f"Source in allowed network: {allowed_source}",
                        service=service,
                        logged=True
                    )
            
            # Source not in any allowed network
            return AccessResult(
                allowed=False,
                reason=f"Source IP {source_ip} not in allowed networks",
                service=service,
                logged=True
            )
            
        except ValueError as e:
            logger.error(f"Invalid IP address: {source_ip} - {e}")
            return AccessResult(
                allowed=False,
                reason="Invalid source IP",
                service=service
            )
    
    def add_service_rule(self, rule: PortRule):
        """Add or update a service rule"""
        self.service_rules[rule.service.value] = rule
        logger.info(f"Added/updated service rule for {rule.service.value}")
    
    def get_service_rule(self, service: str) -> Optional[PortRule]:
        """Get rule for a service"""
        return self.service_rules.get(service)
    
    def list_service_rules(self) -> List[PortRule]:
        """List all service rules"""
        return list(self.service_rules.values())


class NetworkFirewall:
    """Main network firewall orchestrator"""
    
    def __init__(self):
        """Initialize network firewall"""
        self.port_filter = PortFilter()
        self.scan_detector = PortScanDetector()
        self.service_manager = ServicePortManager()
        self.connection_log: List[Dict] = []
    
    def check_connection(
        self,
        source_ip: str,
        dest_port: int,
        service: Optional[str] = None
    ) -> AccessResult:
        """
        Check if a connection should be allowed.
        
        Args:
            source_ip: Source IP address
            dest_port: Destination port
            service: Optional service name
            
        Returns:
            AccessResult with decision
        """
        # Check for port scanning
        if self.scan_detector.check_access(source_ip, dest_port):
            self._log_connection(source_ip, dest_port, False, "Port scan detected")
            return AccessResult(
                allowed=False,
                reason="Port scanning detected",
                logged=True
            )
        
        # Check if port is allowed
        if not self.port_filter.is_allowed(dest_port):
            self._log_connection(source_ip, dest_port, False, "Port not allowed")
            return AccessResult(
                allowed=False,
                reason=f"Port {dest_port} not allowed",
                logged=True
            )
        
        # Check service-specific rules if service specified
        if service:
            result = self.service_manager.check_service_access(
                service, source_ip, dest_port
            )
            self._log_connection(
                source_ip, dest_port, result.allowed,
                result.reason, service
            )
            return result
        
        # Port is allowed, no service-specific check
        self._log_connection(source_ip, dest_port, True, "Port allowed")
        return AccessResult(
            allowed=True,
            reason="Port allowed",
            logged=True
        )
    
    def _log_connection(
        self,
        source_ip: str,
        dest_port: int,
        allowed: bool,
        reason: str,
        service: Optional[str] = None
    ):
        """Log connection attempt"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc),
            "source_ip": source_ip,
            "dest_port": dest_port,
            "service": service,
            "allowed": allowed,
            "reason": reason
        }
        
        self.connection_log.append(log_entry)
        
        # Keep only recent logs (last 1000)
        if len(self.connection_log) > 1000:
            self.connection_log = self.connection_log[-1000:]
        
        # Log to system
        if allowed:
            logger.info(
                f"Connection allowed: {source_ip} -> :{dest_port} "
                f"({service or 'unknown'}) - {reason}"
            )
        else:
            logger.warning(
                f"Connection blocked: {source_ip} -> :{dest_port} "
                f"({service or 'unknown'}) - {reason}"
            )
    
    def get_recent_logs(
        self,
        limit: int = 100,
        allowed_only: bool = False,
        blocked_only: bool = False
    ) -> List[Dict]:
        """Get recent connection logs"""
        logs = self.connection_log
        
        if allowed_only:
            logs = [l for l in logs if l["allowed"]]
        elif blocked_only:
            logs = [l for l in logs if not l["allowed"]]
        
        return logs[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get firewall statistics"""
        total_connections = len(self.connection_log)
        allowed = sum(1 for l in self.connection_log if l["allowed"])
        blocked = total_connections - allowed
        
        blocked_ips = self.scan_detector.get_blocked_ips()
        
        return {
            "total_connections": total_connections,
            "allowed_connections": allowed,
            "blocked_connections": blocked,
            "blocked_ips": len(blocked_ips),
            "active_services": len(self.service_manager.service_rules),
            "allowed_ports": len(self.port_filter.allowed_ports),
            "blocked_ports": len(self.port_filter.blocked_ports)
        }


# Export main components
__all__ = [
    "PortFilter",
    "PortScanDetector",
    "ServicePortManager",
    "NetworkFirewall",
    "PortRule",
    "AccessResult",
    "ServiceType",
]