"""
Web Application Firewall (WAF) Implementation
Sprint 10 - Issue 003: WAF & Port Allowlisting
"""

import re
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class WAFRule:
    """WAF rule definition"""
    name: str
    pattern: str
    severity: str  # low, medium, high, critical
    action: str  # block, log, alert
    description: str
    enabled: bool = True
    compiled_pattern: Optional[re.Pattern] = None
    
    def __post_init__(self):
        """Compile regex pattern after initialization."""
        if self.pattern and not self.compiled_pattern:
            self.compiled_pattern = re.compile(self.pattern, re.IGNORECASE)


@dataclass
class WAFResult:
    """Result of WAF check"""
    blocked: bool
    rule: Optional[str] = None
    severity: Optional[str] = None
    message: Optional[str] = None
    matched_pattern: Optional[str] = None


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int = 0
    retry_after: int = 0
    limit: int = 0


class WAFConfig:
    """WAF configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize WAF configuration"""
        self.enabled = True
        self.rules: Dict[str, WAFRule] = {}
        self._load_default_rules()
        
        if config_path:
            self._load_config(config_path)
    
    def _load_default_rules(self):
        """Load default WAF rules"""
        default_rules = [
            # SQL Injection Protection
            WAFRule(
                name="sql_injection",
                pattern=r"((\b(DROP|ALTER|CREATE|TRUNCATE)\s+(TABLE|DATABASE)\b)|(--\s*$)|(;\s*(DROP|DELETE|TRUNCATE|ALTER|CREATE)\s+)|(\bOR\s+['\"]\w*['\"]?\s*=\s*['\"]\w*['\"])|(\bUNION\s+(ALL\s+)?SELECT\b)|(\bEXEC(UTE)?\s*\()|(\bWAITFOR\s+DELAY\b)|(\bBENCHMARK\s*\()|(pg_sleep)|(%20(SELECT|DROP|UNION|INSERT)%20)|(0x[0-9a-fA-F]+)|(ＳＥＬＥＣＴ|ＤＲＯＰ|ＵＮＩＯＮ)|(/\*.*?\*/)|(\s+SELECT\s+)|(\nSELECT\s)|(\rSELECT\s)|(\'?\|\|\'?)|(\/\*\!\d+.*?\*\/)|(OR\s+'1'\s*=\s*'1))",
                severity="critical",
                action="block",
                description="SQL injection attempt detected"
            ),
            
            # XSS Protection
            WAFRule(
                name="xss_protection",
                pattern=r"(<script[^>]*>.*?</script>)|(<iframe[^>]*>)|(<object[^>]*>)|(<embed[^>]*>)|(javascript:)|(on\w+\s*=)|(alert\s*\()|(document\.(cookie|write))|(window\.(location|open))",
                severity="high",
                action="block",
                description="Cross-site scripting attempt detected"
            ),
            
            # Path Traversal Protection
            WAFRule(
                name="path_traversal",
                pattern=r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c|\.\.%252f|%252e%252e|\.\.;|\.\.%00|\.\.%0d%0a|\.\.\\x|\.\.%c0%af|\.\.\/\/|\.\.\/|\.\.\\){2,}|(\.\./etc/)|(\.\./windows/)|(\.\./boot/)|(\.\./var/)|(\.\./proc/)|(\.\.\.\.//|%2e%2e%2f%2e%2e%2f|\.\.%252f|%c0%af|\.\.%5c|\.\.\.\.\\\.\.\.\.\\|\.\.;/)",
                severity="high",
                action="block",
                description="Path traversal attempt detected"
            ),
            
            # Command Injection Protection - Enhanced patterns
            WAFRule(
                name="command_injection",
                pattern=r"((\||;|&|`|\$\(|\)|\{|\}|<|>).*?(cat|ls|rm|del|wget|curl|bash|sh|cmd|powershell|eval|exec|whoami|id|netstat|nc|nohup))|(%3B|%7C|%26|%60|%24%28).*?(ls|cat|whoami|bash|sh|cmd|powershell)|(\.\.\/.*?(bin\/bash|sh))|(powershell\s+-enc)|(IEX\(New-Object)|(nohup\s+nc\s+-e)|(DownloadString)",
                severity="critical",
                action="block",
                description="Command injection attempt detected"
            ),
            
            # XXE Protection
            WAFRule(
                name="xxe_protection",
                pattern=r"<!DOCTYPE[^>]*\[(<!ENTITY)|<!ELEMENT]|<\?xml[^>]*<!DOCTYPE|SYSTEM\s+[\"']file://|SYSTEM\s+[\"']http://",
                severity="high",
                action="block",
                description="XML external entity attack detected"
            ),
            
            # LDAP Injection Protection
            WAFRule(
                name="ldap_injection",
                pattern=r"[*()\\|&=].*?(cn=|ou=|dc=|objectClass=)",
                severity="medium",
                action="block",
                description="LDAP injection attempt detected"
            ),
            
            # NoSQL Injection Protection - Enhanced with array injection detection
            WAFRule(
                name="nosql_injection",
                pattern=r"(\$ne|\$eq|\$gt|\$gte|\$lt|\$lte|\$in|\$nin|\$and|\$or|\$not|\$regex|\$where|\$exists)|(\[\$ne\]|\[\$eq\]|\[\$gt\]|\[\$in\]|\[\$or\])|({.*\$ne.*}|{.*\$eq.*}|{.*\$gt.*}|{.*\$in.*}|{.*\$or.*}|{.*\$regex.*}|{.*\$where.*})|(\"(username|password|email|user|admin)\"\s*:\s*\[.*\])",
                severity="high",
                action="block",
                description="NoSQL injection attempt detected"
            ),
            
            # Header Injection Protection
            WAFRule(
                name="header_injection",
                pattern=r"(\r\n|\r|\n).*?(Content-Type:|Content-Length:|Location:|Set-Cookie:)",
                severity="medium",
                action="block",
                description="HTTP header injection attempt detected"
            ),
            
            # File Upload Protection
            WAFRule(
                name="malicious_file_upload",
                pattern=r"\.(exe|sh|bat|cmd|com|pif|scr|vbs|js|jar|zip|tar|gz|rar)$",
                severity="medium",
                action="block",
                description="Potentially malicious file upload detected"
            ),
            
            # Protocol Injection
            WAFRule(
                name="protocol_injection",
                pattern=r"(file|gopher|dict|ftp|sftp|ldap|tftp|telnet)://",
                severity="medium",
                action="block",
                description="Protocol injection attempt detected"
            ),
            
            # OWASP A04: Insecure Design - Privilege Escalation
            WAFRule(
                name="privilege_escalation",
                pattern=r"role.*?admin|admin.*role",
                severity="high",
                action="block",
                description="Privilege escalation attempt detected"
            ),
            
            # OWASP A05: Security Misconfiguration - Debug Mode
            WAFRule(
                name="debug_mode_exposure",
                pattern=r"debug.*?true|true.*debug",
                severity="medium",
                action="block",
                description="Debug mode exposure attempt detected"
            ),
            
            # OWASP A06: Vulnerable Components - Version Disclosure
            WAFRule(
                name="version_disclosure",
                pattern=r"version.*?\d+\.\d+(\.\d+)?",
                severity="low",
                action="block",
                description="Version information disclosure detected"
            ),
            
            # OWASP A07: Authentication Failures - Weak Credentials
            WAFRule(
                name="weak_credentials",
                pattern=r"username.*?admin.*?password.*?admin|admin.*?admin",
                severity="high",
                action="block",
                description="Weak credential usage detected"
            ),
            
            # OWASP A09: Security Logging Failures - Log Tampering
            WAFRule(
                name="log_tampering",
                pattern=r"(delete|remove|clear|truncate|modify).*?(log|audit|event|history)",
                severity="high",
                action="block",
                description="Log tampering attempt detected"
            ),
            
            # OWASP A10: SSRF - Server-Side Request Forgery
            WAFRule(
                name="ssrf_protection",
                pattern=r"url.*?localhost|localhost.*url",
                severity="high",
                action="block",
                description="Server-Side Request Forgery (SSRF) attempt detected"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.name] = rule
    
    def _load_config(self, config_path: str):
        """Load WAF configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
                # Update enabled status
                self.enabled = config.get("enabled", True)
                
                # Load custom rules
                for rule_config in config.get("rules", []):
                    rule = WAFRule(**rule_config)
                    self.rules[rule.name] = rule
                    
        except Exception as e:
            logger.error(f"Failed to load WAF config: {e}")
    
    def is_enabled(self) -> bool:
        """Check if WAF is enabled"""
        return self.enabled
    
    def get_rules(self) -> List[WAFRule]:
        """Get all WAF rules"""
        return list(self.rules.values())
    
    def has_rule(self, rule_name: str) -> bool:
        """Check if a rule exists"""
        return rule_name in self.rules
    
    def get_rule(self, rule_name: str) -> Optional[WAFRule]:
        """Get a specific rule"""
        return self.rules.get(rule_name)


class WAFFilter:
    """WAF request filter"""
    
    def __init__(self, config: Optional[WAFConfig] = None):
        """Initialize WAF filter"""
        self.config = config or WAFConfig()
        self.blocked_ips: Set[str] = set()
        self.alert_callbacks = []
    
    def check_request(self, request_data: Dict[str, Any]) -> WAFResult:
        """
        Check request against WAF rules.
        
        Args:
            request_data: Dictionary containing request fields to check
            
        Returns:
            WAFResult indicating if request should be blocked
        """
        if not self.config.is_enabled():
            return WAFResult(blocked=False)
        
        # Convert all keys and values to strings for pattern matching
        text_to_check = []
        for key, value in request_data.items():
            # Include the key in the text to check
            text_to_check.append(str(key))
            
            if value is not None:
                if isinstance(value, (list, dict)):
                    text_to_check.append(json.dumps(value))
                else:
                    text_to_check.append(str(value))
        
        combined_text = " ".join(text_to_check)
        
        # Check against all enabled rules
        for rule in self.config.get_rules():
            if not rule.enabled:
                continue
            
            if rule.compiled_pattern and rule.compiled_pattern.search(combined_text):
                # Pattern matched - take action based on rule
                if rule.action == "block":
                    logger.warning(
                        f"WAF blocked request - Rule: {rule.name}, "
                        f"Severity: {rule.severity}"
                    )
                    
                    return WAFResult(
                        blocked=True,
                        rule=rule.name,
                        severity=rule.severity,
                        message=rule.description,
                        matched_pattern=rule.pattern
                    )
                
                elif rule.action == "log":
                    logger.info(f"WAF logged request - Rule: {rule.name}")
                
                elif rule.action == "alert":
                    self._send_alert(rule, request_data)
        
        return WAFResult(blocked=False)
    
    def check_ip(self, ip_address: str) -> bool:
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def block_ip(self, ip_address: str, duration: int = 3600):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address: {ip_address} for {duration} seconds")
        
        # TODO: Implement automatic unblock after duration
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP address: {ip_address}")
    
    def _send_alert(self, rule: WAFRule, request_data: Dict[str, Any]):
        """Send alert for rule match"""
        for callback in self.alert_callbacks:
            try:
                callback(rule, request_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback):
        """Register callback for alerts"""
        self.alert_callbacks.append(callback)


class WAFRateLimiter:
    """WAF rate limiting implementation"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        global_requests_per_minute: int = 1000
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per client
            burst_size: Maximum burst size per client
            global_requests_per_minute: Maximum total requests per minute (DDoS protection)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.global_requests_per_minute = global_requests_per_minute
        self.request_counts: Dict[str, List[float]] = defaultdict(list)
        self.blocked_clients: Dict[str, float] = {}
        self.global_requests: List[float] = []
    
    def check_rate_limit(self, client_id: str) -> RateLimitResult:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier (IP address, user ID, etc.)
            
        Returns:
            RateLimitResult with rate limit status
        """
        current_time = time.time()
        
        # Check if client is temporarily blocked
        if client_id in self.blocked_clients:
            block_until = self.blocked_clients[client_id]
            if current_time < block_until:
                retry_after = int(block_until - current_time)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    retry_after=retry_after,
                    limit=self.requests_per_minute
                )
            else:
                # Unblock client
                del self.blocked_clients[client_id]
        
        # Clean old requests (older than 1 minute)
        window_start = current_time - 60
        self.request_counts[client_id] = [
            ts for ts in self.request_counts[client_id]
            if ts > window_start
        ]
        
        # Clean old global requests
        self.global_requests = [
            ts for ts in self.global_requests
            if ts > window_start
        ]
        
        # Check global rate limit (DDoS protection)
        global_request_count = len(self.global_requests)
        if global_request_count >= self.global_requests_per_minute:
            logger.warning(
                f"Global rate limit exceeded: {global_request_count} requests in last minute"
            )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=60,
                limit=self.global_requests_per_minute
            )
        
        # Check per-client rate limit
        request_count = len(self.request_counts[client_id])
        
        if request_count >= self.requests_per_minute:
            # Block client for 1 minute
            self.blocked_clients[client_id] = current_time + 60
            
            logger.warning(
                f"Rate limit exceeded for client {client_id}: "
                f"{request_count} requests in last minute"
            )
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=60,
                limit=self.requests_per_minute
            )
        
        # Check burst limit (requests in last 10 seconds)
        burst_window = current_time - 10
        burst_count = sum(1 for ts in self.request_counts[client_id] if ts > burst_window)
        
        if burst_count >= self.burst_size:
            # Temporary slowdown - wait 1 second
            return RateLimitResult(
                allowed=False,
                remaining=self.requests_per_minute - request_count,
                retry_after=1,
                limit=self.requests_per_minute
            )
        
        # Allow request and record timestamp
        self.request_counts[client_id].append(current_time)
        self.global_requests.append(current_time)
        
        return RateLimitResult(
            allowed=True,
            remaining=self.requests_per_minute - request_count - 1,
            retry_after=0,
            limit=self.requests_per_minute
        )
    
    def reset_client(self, client_id: str):
        """Reset rate limit for a client"""
        if client_id in self.request_counts:
            del self.request_counts[client_id]
        if client_id in self.blocked_clients:
            del self.blocked_clients[client_id]


class WAFLogger:
    """WAF event logger"""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize WAF logger"""
        self.log_file = log_file
        self.events: List[Dict[str, Any]] = []
    
    def log_blocked_request(
        self,
        client_ip: str,
        rule_name: str,
        severity: str,
        request_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Log a blocked request"""
        event = {
            "timestamp": timestamp or datetime.utcnow(),
            "event_type": "request_blocked",
            "client_ip": client_ip,
            "rule_name": rule_name,
            "severity": severity,
            "request_data": request_data
        }
        
        self.events.append(event)
        
        # Log to file if configured
        if self.log_file:
            self._write_to_file(event)
        
        # Log to system logger
        logger.warning(
            f"WAF blocked request from {client_ip} - "
            f"Rule: {rule_name}, Severity: {severity}"
        )
    
    def log_rate_limit(
        self,
        client_id: str,
        requests_count: int,
        limit: int,
        timestamp: Optional[datetime] = None
    ):
        """Log rate limit event"""
        event = {
            "timestamp": timestamp or datetime.utcnow(),
            "event_type": "rate_limit_exceeded",
            "client_id": client_id,
            "requests_count": requests_count,
            "limit": limit
        }
        
        self.events.append(event)
        
        logger.warning(
            f"Rate limit exceeded for {client_id}: "
            f"{requests_count}/{limit} requests"
        )
    
    def get_recent_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent WAF events"""
        events = self.events
        
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]
        
        return events[-limit:]
    
    def _write_to_file(self, event: Dict[str, Any]):
        """Write event to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write WAF log: {e}")


class WAFResponseFilter:
    """Filter responses for sensitive data leakage"""
    
    def __init__(self):
        """Initialize response filter"""
        self.sensitive_patterns = [
            # API keys and tokens
            r"(api[_-]?key|apikey|api_secret)[\"']?\s*[:=]\s*[\"']?[\w-]+",
            # Database connection strings
            r"(mongodb|postgres|mysql|redis)://[^\"'\s]+",
            # Private keys
            r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
            # AWS credentials
            r"AKIA[0-9A-Z]{16}",
            # JWT tokens
            r"eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+",
        ]
        
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.sensitive_patterns
        ]
    
    def filter_response(self, response_data: str) -> Tuple[str, bool]:
        """
        Filter response for sensitive data.
        
        Args:
            response_data: Response content to filter
            
        Returns:
            Tuple of (filtered_response, has_sensitive_data)
        """
        has_sensitive = False
        filtered = response_data
        
        for pattern in self.compiled_patterns:
            if pattern.search(filtered):
                has_sensitive = True
                # Redact sensitive data
                filtered = pattern.sub("[REDACTED]", filtered)
        
        if has_sensitive:
            logger.warning("Sensitive data detected and redacted in response")
        
        return filtered, has_sensitive


# Export main components
__all__ = [
    "WAFConfig",
    "WAFFilter",
    "WAFRateLimiter",
    "WAFLogger",
    "WAFResponseFilter",
    "WAFRule",
    "WAFResult",
    "RateLimitResult",
]
