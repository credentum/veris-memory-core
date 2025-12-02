"""
Security Audit Logging Module
Sprint 10 - Missing Security Module Implementation
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import os

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events to audit"""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHORIZATION_DENIED = "authz_denied"
    WAF_BLOCK = "waf_block"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    DATA_ACCESS = "data_access"
    ADMIN_ACTION = "admin_action"


class SeverityLevel(Enum):
    """Security event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: SecurityEventType
    severity: SeverityLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    message: str
    details: Dict[str, Any]
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class SecurityAuditLogger:
    """Main security audit logging system"""
    
    def __init__(self, log_directory: str = "./logs/security", 
                 retention_days: int = 90):
        self.log_directory = log_directory
        self.retention_days = retention_days
        self.events = []  # In-memory event cache
        self.max_memory_events = 1000
        
        # Create log directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)
        
        # Setup file logger
        self.file_logger = self._setup_file_logger()
        
    def _setup_file_logger(self) -> logging.Logger:
        """Setup file-based security event logger"""
        security_logger = logging.getLogger("security_audit")
        security_logger.setLevel(logging.INFO)
        
        # Create file handler with rotation
        log_file = os.path.join(self.log_directory, "security_events.log")
        handler = logging.FileHandler(log_file)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
        )
        handler.setFormatter(formatter)
        
        security_logger.addHandler(handler)
        return security_logger
        
    def _generate_event_id(self, event_data: str) -> str:
        """Generate unique event ID"""
        current_time = str(time.time())
        return hashlib.sha256(f"{event_data}{current_time}".encode()).hexdigest()[:16]
        
    def log_security_event(self, event_type: SecurityEventType, 
                          severity: SeverityLevel,
                          message: str,
                          source_ip: str = "unknown",
                          user_id: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None) -> str:
        """Log a security event"""
        
        if details is None:
            details = {}
        if tags is None:
            tags = []
            
        # Create event
        event = SecurityEvent(
            event_id=self._generate_event_id(f"{event_type.value}{message}"),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            message=message,
            details=details,
            tags=tags
        )
        
        # Add to memory cache
        self.events.append(event)
        
        # Maintain cache size
        if len(self.events) > self.max_memory_events:
            self.events = self.events[-self.max_memory_events:]
            
        # Log to file
        try:
            self.file_logger.info(json.dumps(event.to_dict()))
        except Exception as e:
            logger.error(f"Failed to write security event to log: {e}")
            
        # Log to console for high/critical events
        if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            logger.warning(f"SECURITY EVENT [{severity.value.upper()}]: {message}")
            
        return event.event_id
        
    def log_authentication_failure(self, source_ip: str, username: str, 
                                 reason: str = "Invalid credentials"):
        """Log authentication failure"""
        return self.log_security_event(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            severity=SeverityLevel.MEDIUM,
            message=f"Authentication failed for user: {username}",
            source_ip=source_ip,
            user_id=username,
            details={"reason": reason, "username": username},
            tags=["authentication", "failure"]
        )
        
    def log_waf_block(self, source_ip: str, rule_name: str, 
                     request_data: Dict[str, Any]):
        """Log WAF block event"""
        return self.log_security_event(
            event_type=SecurityEventType.WAF_BLOCK,
            severity=SeverityLevel.HIGH,
            message=f"WAF blocked request from {source_ip} - Rule: {rule_name}",
            source_ip=source_ip,
            details={
                "rule_name": rule_name,
                "request_data": str(request_data)[:500]  # Truncate large requests
            },
            tags=["waf", "block", rule_name]
        )
        
    def log_injection_attempt(self, source_ip: str, attack_type: str,
                            payload: str, user_id: Optional[str] = None):
        """Log injection attack attempt"""
        return self.log_security_event(
            event_type=SecurityEventType.INJECTION_ATTEMPT,
            severity=SeverityLevel.CRITICAL,
            message=f"{attack_type} injection attempt from {source_ip}",
            source_ip=source_ip,
            user_id=user_id,
            details={
                "attack_type": attack_type,
                "payload": payload[:200]  # Truncate payload
            },
            tags=["injection", attack_type.lower(), "attack"]
        )
        
    def log_rate_limit_exceeded(self, source_ip: str, endpoint: str,
                              limit: int, user_id: Optional[str] = None):
        """Log rate limit exceeded"""
        return self.log_security_event(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SeverityLevel.MEDIUM,
            message=f"Rate limit exceeded for {endpoint} from {source_ip}",
            source_ip=source_ip,
            user_id=user_id,
            details={"endpoint": endpoint, "limit": limit},
            tags=["rate_limit", "exceeded"]
        )
        
    def log_admin_action(self, user_id: str, action: str, 
                        target: str, source_ip: str):
        """Log administrative action"""
        return self.log_security_event(
            event_type=SecurityEventType.ADMIN_ACTION,
            severity=SeverityLevel.HIGH,
            message=f"Admin action: {action} on {target}",
            source_ip=source_ip,
            user_id=user_id,
            details={"action": action, "target": target},
            tags=["admin", "action"]
        )
        
    def get_events(self, event_type: Optional[SecurityEventType] = None,
                  severity: Optional[SeverityLevel] = None,
                  since: Optional[datetime] = None,
                  limit: int = 100) -> List[SecurityEvent]:
        """Retrieve security events with filters"""
        filtered_events = self.events
        
        # Apply filters
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
            
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
            
        if since:
            filtered_events = [e for e in filtered_events if e.timestamp >= since]
            
        # Sort by timestamp (newest first) and limit
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_events[:limit]
        
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security event summary for the last N hours"""
        since = datetime.utcnow() - timedelta(hours=hours)
        recent_events = self.get_events(since=since, limit=10000)
        
        # Count by type
        type_counts = {}
        for event in recent_events:
            event_type = event.event_type.value
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
        # Count by severity
        severity_counts = {}
        for event in recent_events:
            severity = event.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        # Top source IPs
        ip_counts = {}
        for event in recent_events:
            ip = event.source_ip
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
            
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "events_by_type": type_counts,
            "events_by_severity": severity_counts,
            "top_source_ips": top_ips,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    def cleanup_old_events(self):
        """Remove events older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        # Clean memory cache
        self.events = [e for e in self.events if e.timestamp >= cutoff_date]
        
        # Note: File cleanup would require log rotation implementation
        logger.info(f"Cleaned up events older than {cutoff_date}")
        
    def export_events(self, start_date: datetime, end_date: datetime,
                     format_type: str = "json") -> str:
        """Export events for compliance reporting"""
        events = [e for e in self.events 
                 if start_date <= e.timestamp <= end_date]
                 
        if format_type == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        elif format_type == "csv":
            # Simple CSV format
            lines = ["event_id,timestamp,event_type,severity,source_ip,user_id,message"]
            for event in events:
                line = f"{event.event_id},{event.timestamp.isoformat()},{event.event_type.value},{event.severity.value},{event.source_ip},{event.user_id or ''},{event.message}"
                lines.append(line)
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


# Global audit logger instance
_audit_logger: Optional[SecurityAuditLogger] = None


def get_audit_logger() -> SecurityAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger


def log_security_event(event_type: SecurityEventType, 
                      severity: SeverityLevel,
                      message: str,
                      **kwargs) -> str:
    """Convenience function to log security events"""
    return get_audit_logger().log_security_event(
        event_type, severity, message, **kwargs
    )