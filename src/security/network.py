"""
Network Security Module
Sprint 10 - Missing Security Module Implementation
"""

import ipaddress
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class NetworkZone(Enum):
    """Network security zones"""
    DMZ = "dmz"
    INTERNAL = "internal"
    SECURE = "secure"
    EXTERNAL = "external"


@dataclass
class NetworkRule:
    """Network security rule definition"""
    source_zone: NetworkZone
    target_zone: NetworkZone
    allowed_ports: List[int]
    protocol: str = "tcp"
    description: str = ""


class NetworkZoneManager:
    """Manages network security zones and access control"""
    
    ZONES = {
        NetworkZone.DMZ: '10.0.1.0/24',
        NetworkZone.INTERNAL: '10.0.2.0/24', 
        NetworkZone.SECURE: '10.0.3.0/24',
        NetworkZone.EXTERNAL: '0.0.0.0/0'
    }
    
    def __init__(self):
        self.zone_networks = {}
        self._initialize_zones()
        
    def _initialize_zones(self):
        """Initialize network zone mappings"""
        for zone, cidr in self.ZONES.items():
            try:
                self.zone_networks[zone] = ipaddress.ip_network(cidr, strict=False)
            except ValueError as e:
                logger.warning(f"Invalid CIDR for zone {zone}: {cidr} - {e}")
                
    def get_zone_for_ip(self, ip_address: str) -> Optional[NetworkZone]:
        """Determine which zone an IP address belongs to"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check each zone
            for zone, network in self.zone_networks.items():
                if zone == NetworkZone.EXTERNAL:
                    continue  # Skip external zone for internal checks
                if ip in network:
                    return zone
                    
            # Default to external if not in any internal zone
            return NetworkZone.EXTERNAL
            
        except ValueError as e:
            logger.error(f"Invalid IP address: {ip_address} - {e}")
            return None
            
    def validate_zone_access(self, source_ip: str, target_zone: NetworkZone, 
                           target_port: int = 80) -> bool:
        """Validate if source IP can access target zone"""
        source_zone = self.get_zone_for_ip(source_ip)
        
        if source_zone is None:
            return False
            
        # Basic zone access rules
        if source_zone == NetworkZone.EXTERNAL:
            # External can only access DMZ on specific ports
            return target_zone == NetworkZone.DMZ and target_port in [80, 443, 8000]
            
        elif source_zone == NetworkZone.DMZ:
            # DMZ can access internal on specific ports
            return target_zone in [NetworkZone.INTERNAL, NetworkZone.DMZ]
            
        elif source_zone == NetworkZone.INTERNAL:
            # Internal can access internal and secure zones
            return target_zone in [NetworkZone.INTERNAL, NetworkZone.SECURE, NetworkZone.DMZ]
            
        elif source_zone == NetworkZone.SECURE:
            # Secure zone has full access
            return True
            
        return False
        
    def get_allowed_ports_for_zones(self, source_zone: NetworkZone, 
                                  target_zone: NetworkZone) -> List[int]:
        """Get allowed ports between zones"""
        # Default port mappings
        port_rules = {
            (NetworkZone.EXTERNAL, NetworkZone.DMZ): [80, 443, 8000],
            (NetworkZone.DMZ, NetworkZone.INTERNAL): [7687, 6333],  # Neo4j, Qdrant
            (NetworkZone.INTERNAL, NetworkZone.SECURE): [6379],     # Redis
            (NetworkZone.INTERNAL, NetworkZone.INTERNAL): [7687, 6333, 6379, 8000],
            (NetworkZone.SECURE, NetworkZone.SECURE): list(range(1024, 65536))
        }
        
        return port_rules.get((source_zone, target_zone), [])


class NetworkPolicy:
    """Enforces network segmentation and access policies"""
    
    def __init__(self, zone_manager: NetworkZoneManager):
        self.zone_manager = zone_manager
        self.rules = self._create_default_rules()
        self.violations = []
        
    def _create_default_rules(self) -> List[NetworkRule]:
        """Create default network security rules"""
        return [
            NetworkRule(
                source_zone=NetworkZone.EXTERNAL,
                target_zone=NetworkZone.DMZ,
                allowed_ports=[80, 443, 8000],
                description="External access to DMZ services"
            ),
            NetworkRule(
                source_zone=NetworkZone.DMZ,
                target_zone=NetworkZone.INTERNAL,
                allowed_ports=[7687, 6333],
                description="DMZ to internal database access"
            ),
            NetworkRule(
                source_zone=NetworkZone.INTERNAL,
                target_zone=NetworkZone.SECURE,
                allowed_ports=[6379],
                description="Internal to secure Redis access"
            )
        ]
        
    def enforce_segmentation(self, source_ip: str, target_ip: str, 
                           target_port: int) -> bool:
        """Enforce network segmentation rules"""
        source_zone = self.zone_manager.get_zone_for_ip(source_ip)
        target_zone = self.zone_manager.get_zone_for_ip(target_ip)
        
        if source_zone is None or target_zone is None:
            logger.warning(f"Unable to determine zones for {source_ip} -> {target_ip}")
            return False
            
        # Check if access is allowed
        access_allowed = self.zone_manager.validate_zone_access(
            source_ip, target_zone, target_port
        )
        
        if not access_allowed:
            violation = {
                "timestamp": time.time(),
                "source_ip": source_ip,
                "source_zone": source_zone.value,
                "target_ip": target_ip,
                "target_zone": target_zone.value,
                "target_port": target_port,
                "action": "blocked"
            }
            self.violations.append(violation)
            logger.warning(f"Network policy violation: {violation}")
            
        return access_allowed
        
    def get_violations(self, since_timestamp: Optional[float] = None) -> List[Dict]:
        """Get network policy violations"""
        if since_timestamp is None:
            return self.violations
            
        return [v for v in self.violations if v["timestamp"] >= since_timestamp]
        
    def clear_violations(self):
        """Clear violation history"""
        self.violations.clear()


class PortFilter:
    """Port-based access control and monitoring"""
    
    ALLOWED_PORTS = {
        8000: "MCP Server",
        7687: "Neo4j", 
        6333: "Qdrant",
        6379: "Redis",
        443: "HTTPS",
        80: "HTTP"
    }
    
    def __init__(self):
        self.scan_attempts = {}
        self.blocked_ips = set()
        
    def is_port_allowed(self, port: int, source_zone: NetworkZone = NetworkZone.EXTERNAL) -> bool:
        """Check if port access is allowed from source zone"""
        if port in self.ALLOWED_PORTS:
            # External zone has limited access
            if source_zone == NetworkZone.EXTERNAL:
                return port in [80, 443, 8000]
            return True
            
        return False
        
    def detect_port_scan(self, source_ip: str, target_port: int) -> bool:
        """Detect potential port scanning"""
        current_time = time.time()
        
        # Initialize tracking for this IP
        if source_ip not in self.scan_attempts:
            self.scan_attempts[source_ip] = []
            
        # Add this attempt
        self.scan_attempts[source_ip].append({
            "port": target_port,
            "timestamp": current_time
        })
        
        # Clean old attempts (older than 60 seconds)
        self.scan_attempts[source_ip] = [
            attempt for attempt in self.scan_attempts[source_ip]
            if current_time - attempt["timestamp"] < 60
        ]
        
        # Check if this looks like scanning
        recent_attempts = len(self.scan_attempts[source_ip])
        unique_ports = len(set(attempt["port"] for attempt in self.scan_attempts[source_ip]))
        
        # Port scan detection: 5+ unique ports in 60 seconds
        if unique_ports >= 5 and recent_attempts >= 5:
            self.blocked_ips.add(source_ip)
            logger.warning(f"Port scan detected from {source_ip}: {unique_ports} ports in 60s")
            return True
            
        return False
        
    def is_ip_blocked(self, source_ip: str) -> bool:
        """Check if IP is blocked due to port scanning"""
        return source_ip in self.blocked_ips