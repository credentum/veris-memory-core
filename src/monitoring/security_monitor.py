"""
Security Monitoring and Alerting System
Sprint 10 Phase 2 - Issue 006: SEC-106
Monitors security events and triggers alerts
"""

import logging
import asyncio
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import queue
import smtplib
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False
import requests
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Security event types"""
    AUTHENTICATION_FAILURE = "auth_failure"
    BRUTE_FORCE_ATTACK = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    PORT_SCAN = "port_scan"
    DDOS_ATTACK = "ddos_attack"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SYSTEM_ANOMALY = "system_anomaly"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: EventType
    severity: AlertSeverity
    timestamp: datetime
    source_ip: str
    user_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertChannel:
    """Alert channel configuration"""
    name: str
    channel_type: str  # email, webhook, slack, pagerduty
    endpoint: str
    enabled: bool = True
    severity_filter: Set[AlertSeverity] = field(default_factory=lambda: {AlertSeverity.HIGH, AlertSeverity.CRITICAL})
    rate_limit: int = 10  # Max alerts per hour


@dataclass
class Alert:
    """Alert message"""
    id: str
    event: SecurityEvent
    message: str
    channels: List[str]
    created_at: datetime
    sent_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None


class SecurityEventDetector:
    """Detects security events from logs and metrics"""
    
    def __init__(self):
        """Initialize security event detector"""
        self.detectors = {
            EventType.BRUTE_FORCE_ATTACK: self._detect_brute_force,
            EventType.PORT_SCAN: self._detect_port_scan,
            EventType.DDOS_ATTACK: self._detect_ddos,
            EventType.SQL_INJECTION: self._detect_sql_injection,
            EventType.XSS_ATTACK: self._detect_xss,
        }
        
        # Tracking state for detections
        self.failed_logins = {}  # IP -> count
        self.port_accesses = {}  # IP -> ports accessed
        self.request_rates = {}  # IP -> request timestamps
        self.triggered_alerts = set()  # Track already triggered alerts
        
    def analyze_log_entry(self, log_entry: Dict[str, Any]) -> Optional[SecurityEvent]:
        """
        Analyze a log entry for security events.
        
        Args:
            log_entry: Log entry data
            
        Returns:
            SecurityEvent if detected, None otherwise
        """
        for event_type, detector in self.detectors.items():
            event = detector(log_entry)
            if event:
                return event
        
        return None
    
    def _detect_brute_force(self, log_entry: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect brute force attacks"""
        if log_entry.get("event") == "auth_failure":
            source_ip = log_entry.get("source_ip", "unknown")
            
            # Track failed login attempts
            if source_ip not in self.failed_logins:
                self.failed_logins[source_ip] = []
            
            now = datetime.utcnow()
            self.failed_logins[source_ip].append(now)
            
            # Remove old entries (older than 10 minutes)
            cutoff = now - timedelta(minutes=10)
            self.failed_logins[source_ip] = [
                t for t in self.failed_logins[source_ip] if t > cutoff
            ]
            
            # Check if threshold exceeded and not already alerted
            alert_key = f"brute_force_{source_ip}"
            if len(self.failed_logins[source_ip]) >= 5 and alert_key not in self.triggered_alerts:
                self.triggered_alerts.add(alert_key)
                return SecurityEvent(
                    event_type=EventType.BRUTE_FORCE_ATTACK,
                    severity=AlertSeverity.HIGH,
                    timestamp=now,
                    source_ip=source_ip,
                    user_id=log_entry.get("user_id"),
                    description=f"Brute force attack detected from {source_ip}",
                    metadata={"failed_attempts": len(self.failed_logins[source_ip])},
                    raw_data=log_entry
                )
        
        return None
    
    def _detect_port_scan(self, log_entry: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect port scanning"""
        if log_entry.get("event") == "connection_attempt":
            source_ip = log_entry.get("source_ip", "unknown")
            port = log_entry.get("port")
            
            if port:
                if source_ip not in self.port_accesses:
                    self.port_accesses[source_ip] = set()
                
                self.port_accesses[source_ip].add(port)
                
                # Check for port scanning (accessing many different ports)
                alert_key = f"port_scan_{source_ip}"
                if len(self.port_accesses[source_ip]) >= 10 and alert_key not in self.triggered_alerts:
                    self.triggered_alerts.add(alert_key)
                    return SecurityEvent(
                        event_type=EventType.PORT_SCAN,
                        severity=AlertSeverity.MEDIUM,
                        timestamp=datetime.utcnow(),
                        source_ip=source_ip,
                        description=f"Port scan detected from {source_ip}",
                        metadata={"ports_accessed": list(self.port_accesses[source_ip])},
                        raw_data=log_entry
                    )
        
        return None
    
    def _detect_ddos(self, log_entry: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect DDoS attacks"""
        if log_entry.get("event") == "request":
            source_ip = log_entry.get("source_ip", "unknown")
            
            if source_ip not in self.request_rates:
                self.request_rates[source_ip] = []
            
            now = datetime.utcnow()
            self.request_rates[source_ip].append(now)
            
            # Keep only requests from last minute
            cutoff = now - timedelta(minutes=1)
            self.request_rates[source_ip] = [
                t for t in self.request_rates[source_ip] if t > cutoff
            ]
            
            # Check for excessive request rate (>100 requests per minute)
            alert_key = f"ddos_{source_ip}_{now.strftime('%Y%m%d%H%M')}"  # Reset alert per minute
            if len(self.request_rates[source_ip]) >= 100 and alert_key not in self.triggered_alerts:
                self.triggered_alerts.add(alert_key)
                return SecurityEvent(
                    event_type=EventType.DDOS_ATTACK,
                    severity=AlertSeverity.CRITICAL,
                    timestamp=now,
                    source_ip=source_ip,
                    description=f"DDoS attack detected from {source_ip}",
                    metadata={"requests_per_minute": len(self.request_rates[source_ip])},
                    raw_data=log_entry
                )
        
        return None
    
    def _detect_sql_injection(self, log_entry: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect SQL injection attempts"""
        if log_entry.get("event") == "waf_block" and log_entry.get("rule") == "sql_injection":
            return SecurityEvent(
                event_type=EventType.SQL_INJECTION,
                severity=AlertSeverity.HIGH,
                timestamp=datetime.utcnow(),
                source_ip=log_entry.get("source_ip", "unknown"),
                description="SQL injection attempt blocked",
                metadata={"blocked_payload": log_entry.get("payload", "")[:100]},
                raw_data=log_entry
            )
        
        return None
    
    def _detect_xss(self, log_entry: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Detect XSS attacks"""
        if log_entry.get("event") == "waf_block" and log_entry.get("rule") == "xss_protection":
            return SecurityEvent(
                event_type=EventType.XSS_ATTACK,
                severity=AlertSeverity.HIGH,
                timestamp=datetime.utcnow(),
                source_ip=log_entry.get("source_ip", "unknown"),
                description="XSS attack attempt blocked",
                metadata={"blocked_payload": log_entry.get("payload", "")[:100]},
                raw_data=log_entry
            )
        
        return None


class AlertManager:
    """Manages alert generation and delivery"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize alert manager"""
        self.config = config or {}
        self.channels = {}
        self.alert_history = []
        self.rate_limiters = {}  # Channel -> timestamps
        self.alert_queue = queue.Queue()
        self.running = False
        self.worker_thread = None
        
        # Setup default channels
        self._setup_default_channels()
    
    def _setup_default_channels(self):
        """Setup default alert channels"""
        # Email channel
        if self.config.get("smtp_server"):
            self.add_channel(AlertChannel(
                name="email",
                channel_type="email",
                endpoint=self.config.get("alert_email", "security@example.com"),
                severity_filter={AlertSeverity.HIGH, AlertSeverity.CRITICAL}
            ))
        
        # Webhook channel
        if self.config.get("webhook_url"):
            self.add_channel(AlertChannel(
                name="webhook",
                channel_type="webhook",
                endpoint=self.config["webhook_url"],
                severity_filter={AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL}
            ))
        
        # Slack channel
        if self.config.get("slack_webhook"):
            self.add_channel(AlertChannel(
                name="slack",
                channel_type="slack",
                endpoint=self.config["slack_webhook"],
                severity_filter={AlertSeverity.HIGH, AlertSeverity.CRITICAL}
            ))
    
    def add_channel(self, channel: AlertChannel):
        """Add an alert channel"""
        self.channels[channel.name] = channel
        self.rate_limiters[channel.name] = []
        logger.info(f"Added alert channel: {channel.name}")
    
    def remove_channel(self, channel_name: str):
        """Remove an alert channel"""
        if channel_name in self.channels:
            del self.channels[channel_name]
            del self.rate_limiters[channel_name]
            logger.info(f"Removed alert channel: {channel_name}")
    
    def create_alert(self, event: SecurityEvent) -> Alert:
        """
        Create an alert from a security event.
        
        Args:
            event: Security event
            
        Returns:
            Created alert
        """
        # Determine which channels to use
        alert_channels = []
        for channel_name, channel in self.channels.items():
            if channel.enabled and event.severity in channel.severity_filter:
                if not self._is_rate_limited(channel_name):
                    alert_channels.append(channel_name)
        
        # Generate alert message
        message = self._generate_alert_message(event)
        
        alert = Alert(
            id=f"alert_{int(time.time())}_{event.event_type.value}",
            event=event,
            message=message,
            channels=alert_channels,
            created_at=datetime.utcnow()
        )
        
        self.alert_history.append(alert)
        return alert
    
    def _generate_alert_message(self, event: SecurityEvent) -> str:
        """Generate alert message from event"""
        severity_emoji = {
            AlertSeverity.LOW: "🟡",
            AlertSeverity.MEDIUM: "🟠", 
            AlertSeverity.HIGH: "🔴",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        emoji = severity_emoji.get(event.severity, "⚠️")
        
        message = f"""
{emoji} SECURITY ALERT - {event.severity.value.upper()}

Event Type: {event.event_type.value.replace('_', ' ').title()}
Time: {event.timestamp.isoformat()}
Source IP: {event.source_ip}
User: {event.user_id or 'N/A'}

Description: {event.description}

Additional Details:
{json.dumps(event.metadata, indent=2)}

Event ID: {id(event)}
""".strip()
        
        return message
    
    def _is_rate_limited(self, channel_name: str) -> bool:
        """Check if channel is rate limited"""
        if channel_name not in self.rate_limiters:
            return False
        
        channel = self.channels[channel_name]
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=1)
        
        # Remove old timestamps
        self.rate_limiters[channel_name] = [
            t for t in self.rate_limiters[channel_name] if t > cutoff
        ]
        
        # Check rate limit
        return len(self.rate_limiters[channel_name]) >= channel.rate_limit
    
    def send_alert(self, alert: Alert):
        """Send alert to configured channels"""
        if not alert.channels:
            logger.warning(f"No channels configured for alert {alert.id}")
            # Still mark as sent even if no channels
            alert.sent_at = datetime.utcnow()
            return
        
        for channel_name in alert.channels:
            try:
                self._send_to_channel(alert, channel_name)
                
                # Update rate limiter
                self.rate_limiters[channel_name].append(datetime.utcnow())
                
            except Exception as e:
                logger.error(f"Failed to send alert to {channel_name}: {e}")
        
        alert.sent_at = datetime.utcnow()
    
    def _send_to_channel(self, alert: Alert, channel_name: str):
        """Send alert to specific channel"""
        channel = self.channels[channel_name]
        
        if channel.channel_type == "email":
            self._send_email(alert, channel)
        elif channel.channel_type == "webhook":
            self._send_webhook(alert, channel)
        elif channel.channel_type == "slack":
            self._send_slack(alert, channel)
        else:
            logger.warning(f"Unknown channel type: {channel.channel_type}")
    
    def _send_email(self, alert: Alert, channel: AlertChannel):
        """Send alert via email"""
        if not HAS_EMAIL:
            logger.error("Email functionality not available - missing email libraries")
            return
        
        try:
            # Email configuration from environment
            smtp_server = self.config.get("smtp_server", "localhost")
            smtp_port = self.config.get("smtp_port", 587)
            smtp_user = self.config.get("smtp_user", "")
            smtp_password = self.config.get("smtp_password", "")
            
            msg = MimeMultipart()
            msg["From"] = smtp_user or "security@example.com"
            msg["To"] = channel.endpoint
            msg["Subject"] = f"Security Alert: {alert.event.event_type.value.replace('_', ' ').title()}"
            
            msg.attach(MimeText(alert.message, "plain"))
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_user and smtp_password:
                    server.starttls()
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {channel.endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook(self, alert: Alert, channel: AlertChannel):
        """Send alert via webhook"""
        try:
            payload = {
                "alert_id": alert.id,
                "event_type": alert.event.event_type.value,
                "severity": alert.event.severity.value,
                "timestamp": alert.event.timestamp.isoformat(),
                "source_ip": alert.event.source_ip,
                "user_id": alert.event.user_id,
                "description": alert.event.description,
                "metadata": alert.event.metadata,
                "message": alert.message
            }
            
            response = requests.post(
                channel.endpoint,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent to {channel.endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_slack(self, alert: Alert, channel: AlertChannel):
        """Send alert to Slack"""
        try:
            severity_colors = {
                AlertSeverity.LOW: "#ffeb3b",
                AlertSeverity.MEDIUM: "#ff9800",
                AlertSeverity.HIGH: "#f44336",
                AlertSeverity.CRITICAL: "#9c27b0"
            }
            
            payload = {
                "attachments": [{
                    "color": severity_colors.get(alert.event.severity, "#9e9e9e"),
                    "title": f"Security Alert: {alert.event.event_type.value.replace('_', ' ').title()}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.event.severity.value.upper(), "short": True},
                        {"title": "Source IP", "value": alert.event.source_ip, "short": True},
                        {"title": "Time", "value": alert.event.timestamp.isoformat(), "short": False}
                    ],
                    "ts": int(alert.event.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                channel.endpoint,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Slack alert sent to {channel.endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def start(self):
        """Start alert processing"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_alerts)
        self.worker_thread.start()
        logger.info("Alert manager started")
    
    def stop(self):
        """Stop alert processing"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Alert manager stopped")
    
    def _process_alerts(self):
        """Process alerts from queue"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                self.send_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def queue_alert(self, alert: Alert):
        """Queue alert for processing"""
        self.alert_queue.put(alert)


class SecurityMonitor:
    """Main security monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security monitor"""
        self.config = config or {}
        self.detector = SecurityEventDetector()
        self.alert_manager = AlertManager(config)
        self.metrics = {
            "events_processed": 0,
            "alerts_generated": 0,
            "events_by_type": {},
            "alerts_by_severity": {}
        }
        self.running = False
        
    def start(self):
        """Start security monitoring"""
        self.alert_manager.start()
        self.running = True
        logger.info("Security monitor started")
    
    def stop(self):
        """Stop security monitoring"""
        self.running = False
        self.alert_manager.stop()
        logger.info("Security monitor stopped")
    
    def process_log_entry(self, log_entry: Dict[str, Any]):
        """
        Process a log entry for security events.
        
        Args:
            log_entry: Log entry to process
        """
        if not self.running:
            return
        
        self.metrics["events_processed"] += 1
        
        # Detect security event
        event = self.detector.analyze_log_entry(log_entry)
        
        if event:
            # Update metrics
            event_type = event.event_type.value
            severity = event.severity.value
            
            self.metrics["events_by_type"][event_type] = \
                self.metrics["events_by_type"].get(event_type, 0) + 1
            self.metrics["alerts_by_severity"][severity] = \
                self.metrics["alerts_by_severity"].get(severity, 0) + 1
            
            # Create and queue alert
            alert = self.alert_manager.create_alert(event)
            self.alert_manager.queue_alert(alert)
            
            self.metrics["alerts_generated"] += 1
            
            logger.warning(f"Security event detected: {event.event_type.value} from {event.source_ip}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics"""
        return {
            **self.metrics,
            "alert_channels": len(self.alert_manager.channels),
            "pending_alerts": self.alert_manager.alert_queue.qsize(),
            "total_alerts": len(self.alert_manager.alert_history)
        }
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User acknowledging the alert
            
        Returns:
            True if acknowledged, False if not found
        """
        for alert in self.alert_manager.alert_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
        
        return False
    
    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """
        Get recent alerts.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_manager.alert_history
            if alert.created_at > cutoff
        ]
    
    def simulate_event(self, event_type: EventType, source_ip: str = "127.0.0.1") -> SecurityEvent:
        """
        Simulate a security event for testing.
        
        Args:
            event_type: Type of event to simulate
            source_ip: Source IP for the event
            
        Returns:
            Generated security event
        """
        event = SecurityEvent(
            event_type=event_type,
            severity=AlertSeverity.HIGH,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            description=f"Simulated {event_type.value} event",
            metadata={"simulated": True}
        )
        
        # Create and queue alert
        alert = self.alert_manager.create_alert(event)
        self.alert_manager.queue_alert(alert)
        
        logger.info(f"Simulated security event: {event_type.value}")
        return event


# Export main components
__all__ = [
    "SecurityMonitor",
    "SecurityEvent",
    "AlertSeverity",
    "EventType",
    "AlertChannel",
    "Alert",
    "SecurityEventDetector",
    "AlertManager",
]