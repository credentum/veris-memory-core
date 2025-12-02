#!/usr/bin/env python3
"""
payload_validator.py: Enhanced Payload Security Validation for Sprint 11 Phase 4

Implements comprehensive payload validation with security checks including:
- Deep JSON structure validation
- Size limits and depth limits
- XSS and injection attack prevention
- Schema validation with sanitization
- Sensitive data detection
"""

import re
import json
import logging
import hashlib
import html
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import base64
import urllib.parse
from datetime import datetime

from src.core.error_codes import ErrorCode, create_error_response

logger = logging.getLogger(__name__)


class SecurityThreat(str, Enum):
    """Types of security threats detected"""
    XSS_SCRIPT = "xss_script"
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    XXE_ATTACK = "xxe_attack"
    SENSITIVE_DATA = "sensitive_data"
    MALFORMED_JSON = "malformed_json"
    SIZE_LIMIT_EXCEEDED = "size_limit_exceeded"
    DEPTH_LIMIT_EXCEEDED = "depth_limit_exceeded"
    INVALID_ENCODING = "invalid_encoding"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation security issue"""
    threat: SecurityThreat
    severity: ValidationSeverity
    message: str
    field_path: str
    detected_value: Optional[str] = None
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat": self.threat.value,
            "severity": self.severity.value,
            "message": self.message,
            "field_path": self.field_path,
            "detected_value": self.detected_value,
            "suggested_fix": self.suggested_fix
        }


@dataclass
class ValidationResult:
    """Result of payload validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    sanitized_payload: Optional[Dict[str, Any]] = None
    security_score: float = 1.0  # 0.0 = high risk, 1.0 = secure
    
    def has_critical_issues(self) -> bool:
        """Check if any critical security issues were found"""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    def has_high_severity_issues(self) -> bool:
        """Check if any high severity issues were found"""
        return any(issue.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL] 
                  for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "security_score": self.security_score,
            "issues": [issue.to_dict() for issue in self.issues],
            "has_critical_issues": self.has_critical_issues(),
            "has_high_severity_issues": self.has_high_severity_issues()
        }


class EnhancedPayloadValidator:
    """Enhanced security-focused payload validator"""
    
    def __init__(self):
        """Initialize payload validator with security patterns"""
        self.max_payload_size = 10 * 1024 * 1024  # 10MB
        self.max_json_depth = 10
        self.max_string_length = 100_000  # 100KB per string
        self.max_array_length = 10_000
        
        # Security patterns for threat detection
        self._setup_security_patterns()
        
        # Allowlisted HTML tags (very restrictive)
        self.allowed_html_tags = {"p", "br", "strong", "em", "li", "ul", "ol"}
    
    def _setup_security_patterns(self):
        """Setup regex patterns for security threat detection"""
        
        # XSS patterns (common script injection attempts)
        self.xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:[^"\']*', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),  # onclick, onload, etc.
            re.compile(r'<iframe[^>]*>', re.IGNORECASE),
            re.compile(r'<object[^>]*>', re.IGNORECASE),
            re.compile(r'<embed[^>]*>', re.IGNORECASE),
            re.compile(r'<form[^>]*>', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'expression\s*\(', re.IGNORECASE),  # CSS expression
        ]
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            re.compile(r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b.*\b(FROM|INTO|SET|WHERE|OR|AND)\b', re.IGNORECASE),
            re.compile(r'(\bOR\b|\bAND\b)\s+\w+\s*=\s*\w+', re.IGNORECASE),
            re.compile(r'\'.*(\bOR\b|\bAND\b).*\'', re.IGNORECASE),
            re.compile(r'--.*', re.IGNORECASE),  # SQL comments
            re.compile(r'/\*.*\*/', re.IGNORECASE | re.DOTALL),  # SQL block comments
            re.compile(r'(\w+\s*=\s*\w+|\d+\s*=\s*\d+).*(\bOR\b|\bAND\b)', re.IGNORECASE),
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            re.compile(r'[;&|`$(){}[\]\\]', re.IGNORECASE),  # Shell metacharacters
            re.compile(r'\b(cat|ls|pwd|whoami|id|ps|kill|rm|mv|cp|chmod|chown|sudo|su)\b', re.IGNORECASE),
            re.compile(r'(\||;|&|\$\(|\`)', re.IGNORECASE),  # Command chaining
            re.compile(r'\$\{[^}]*\}', re.IGNORECASE),  # Variable expansion
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            re.compile(r'\.\.[\\/]', re.IGNORECASE),
            re.compile(r'[\\/]\.\.[\\/]', re.IGNORECASE),
            re.compile(r'[\\/]etc[\\/]passwd', re.IGNORECASE),
            re.compile(r'[\\/]proc[\\/]', re.IGNORECASE),
            re.compile(r'[\\/]sys[\\/]', re.IGNORECASE),
            re.compile(r'[\\/]dev[\\/]', re.IGNORECASE),
        ]
        
        # XXE attack patterns
        self.xxe_patterns = [
            re.compile(r'<!DOCTYPE[^>]+SYSTEM', re.IGNORECASE),
            re.compile(r'<!ENTITY[^>]+SYSTEM', re.IGNORECASE),
            re.compile(r'&\w+;', re.IGNORECASE),  # Entity references
        ]
        
        # Sensitive data patterns
        self.sensitive_data_patterns = [
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Credit card
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b(?:password|pwd|pass|secret|key|token)["\']?\s*[:=]\s*["\']?[A-Za-z0-9!@#$%^&*()_+=-]{6,}', re.IGNORECASE),
            re.compile(r'\b(?:api[_-]?key|auth[_-]?token|access[_-]?token)["\']?\s*[:=]\s*["\']?[A-Za-z0-9!@#$%^&*()_+=-]{10,}', re.IGNORECASE),
        ]
    
    async def validate_payload(
        self, 
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Comprehensive payload validation with security checks
        
        Args:
            payload: The payload to validate
            context: Optional context for validation (e.g., endpoint, user)
            
        Returns:
            ValidationResult with security analysis
        """
        issues = []
        
        try:
            # Convert to JSON for size and structure analysis
            payload_json = json.dumps(payload, ensure_ascii=False)
            
            # Basic size and structure validation
            issues.extend(await self._validate_size_limits(payload, payload_json))
            issues.extend(await self._validate_structure_limits(payload))
            
            # Deep security validation
            issues.extend(await self._validate_security_threats(payload, ""))
            
            # Content validation
            issues.extend(await self._validate_content_safety(payload, ""))
            
            # Calculate security score
            security_score = self._calculate_security_score(issues)
            
            # Determine if payload is valid
            is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
            
            # Generate sanitized payload if needed
            sanitized_payload = None
            if issues and is_valid:
                sanitized_payload = await self._sanitize_payload(payload, issues)
            
            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                sanitized_payload=sanitized_payload,
                security_score=security_score
            )
            
        except Exception as e:
            logger.error(f"Payload validation error: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    threat=SecurityThreat.MALFORMED_JSON,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Payload validation failed: {str(e)}",
                    field_path="root"
                )],
                security_score=0.0
            )
    
    async def _validate_size_limits(self, payload: Dict[str, Any], payload_json: str) -> List[ValidationIssue]:
        """Validate payload size limits"""
        issues = []
        
        # Check overall payload size
        payload_size = len(payload_json.encode('utf-8'))
        if payload_size > self.max_payload_size:
            issues.append(ValidationIssue(
                threat=SecurityThreat.SIZE_LIMIT_EXCEEDED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Payload size {payload_size} bytes exceeds limit of {self.max_payload_size} bytes",
                field_path="root",
                suggested_fix=f"Reduce payload size by {payload_size - self.max_payload_size} bytes"
            ))
        
        return issues
    
    async def _validate_structure_limits(self, obj: Any, path: str = "", depth: int = 0) -> List[ValidationIssue]:
        """Validate JSON structure limits"""
        issues = []
        
        # Check depth limit
        if depth > self.max_json_depth:
            issues.append(ValidationIssue(
                threat=SecurityThreat.DEPTH_LIMIT_EXCEEDED,
                severity=ValidationSeverity.HIGH,
                message=f"JSON depth {depth} exceeds limit of {self.max_json_depth}",
                field_path=path,
                suggested_fix="Flatten the JSON structure"
            ))
            return issues
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                issues.extend(await self._validate_structure_limits(value, new_path, depth + 1))
                
        elif isinstance(obj, list):
            if len(obj) > self.max_array_length:
                issues.append(ValidationIssue(
                    threat=SecurityThreat.SIZE_LIMIT_EXCEEDED,
                    severity=ValidationSeverity.HIGH,
                    message=f"Array length {len(obj)} exceeds limit of {self.max_array_length}",
                    field_path=path,
                    suggested_fix=f"Reduce array size by {len(obj) - self.max_array_length} elements"
                ))
            
            for i, item in enumerate(obj[:self.max_array_length]):  # Only check first N items
                new_path = f"{path}[{i}]"
                issues.extend(await self._validate_structure_limits(item, new_path, depth + 1))
                
        elif isinstance(obj, str):
            if len(obj) > self.max_string_length:
                issues.append(ValidationIssue(
                    threat=SecurityThreat.SIZE_LIMIT_EXCEEDED,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"String length {len(obj)} exceeds limit of {self.max_string_length}",
                    field_path=path,
                    detected_value=obj[:100] + "..." if len(obj) > 100 else obj,
                    suggested_fix=f"Truncate string by {len(obj) - self.max_string_length} characters"
                ))
        
        return issues
    
    async def _validate_security_threats(self, obj: Any, path: str = "") -> List[ValidationIssue]:
        """Deep validation for security threats"""
        issues = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                
                # Check key for threats
                issues.extend(self._check_string_for_threats(str(key), f"{new_path}[key]"))
                
                # Recursively check value
                issues.extend(await self._validate_security_threats(value, new_path))
                
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                issues.extend(await self._validate_security_threats(item, new_path))
                
        elif isinstance(obj, str):
            issues.extend(self._check_string_for_threats(obj, path))
        
        return issues
    
    def _check_string_for_threats(self, text: str, field_path: str) -> List[ValidationIssue]:
        """Check a string value for security threats"""
        issues = []
        
        if not isinstance(text, str) or not text:
            return issues
        
        # XSS detection
        for pattern in self.xss_patterns:
            if pattern.search(text):
                issues.append(ValidationIssue(
                    threat=SecurityThreat.XSS_SCRIPT,
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential XSS attack detected in input",
                    field_path=field_path,
                    detected_value=text[:200] + "..." if len(text) > 200 else text,
                    suggested_fix="Remove or encode HTML/JavaScript content"
                ))
        
        # SQL injection detection
        for pattern in self.sql_injection_patterns:
            if pattern.search(text):
                issues.append(ValidationIssue(
                    threat=SecurityThreat.SQL_INJECTION,
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential SQL injection detected in input",
                    field_path=field_path,
                    detected_value=text[:200] + "..." if len(text) > 200 else text,
                    suggested_fix="Use parameterized queries or escape SQL characters"
                ))
        
        # Command injection detection
        for pattern in self.command_injection_patterns:
            if pattern.search(text):
                issues.append(ValidationIssue(
                    threat=SecurityThreat.COMMAND_INJECTION,
                    severity=ValidationSeverity.HIGH,
                    message="Potential command injection detected in input",
                    field_path=field_path,
                    detected_value=text[:200] + "..." if len(text) > 200 else text,
                    suggested_fix="Remove shell metacharacters or use safe command execution"
                ))
        
        # Path traversal detection
        for pattern in self.path_traversal_patterns:
            if pattern.search(text):
                issues.append(ValidationIssue(
                    threat=SecurityThreat.PATH_TRAVERSAL,
                    severity=ValidationSeverity.HIGH,
                    message="Potential path traversal attack detected",
                    field_path=field_path,
                    detected_value=text[:200] + "..." if len(text) > 200 else text,
                    suggested_fix="Use absolute paths or validate file access"
                ))
        
        # XXE attack detection
        for pattern in self.xxe_patterns:
            if pattern.search(text):
                issues.append(ValidationIssue(
                    threat=SecurityThreat.XXE_ATTACK,
                    severity=ValidationSeverity.HIGH,
                    message="Potential XXE attack detected",
                    field_path=field_path,
                    detected_value=text[:200] + "..." if len(text) > 200 else text,
                    suggested_fix="Disable external entity processing"
                ))
        
        # Sensitive data detection
        for pattern in self.sensitive_data_patterns:
            if pattern.search(text):
                issues.append(ValidationIssue(
                    threat=SecurityThreat.SENSITIVE_DATA,
                    severity=ValidationSeverity.MEDIUM,
                    message="Potential sensitive data detected",
                    field_path=field_path,
                    detected_value="[REDACTED]",
                    suggested_fix="Remove or encrypt sensitive data"
                ))
        
        return issues
    
    async def _validate_content_safety(self, obj: Any, path: str = "") -> List[ValidationIssue]:
        """Additional content safety validation"""
        issues = []
        
        # Check for encoding issues
        if isinstance(obj, str):
            try:
                # Try to encode/decode to detect encoding issues
                obj.encode('utf-8').decode('utf-8')
            except UnicodeError:
                issues.append(ValidationIssue(
                    threat=SecurityThreat.INVALID_ENCODING,
                    severity=ValidationSeverity.MEDIUM,
                    message="Invalid character encoding detected",
                    field_path=path,
                    suggested_fix="Use valid UTF-8 encoding"
                ))
        
        # Recursively check nested structures
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                issues.extend(await self._validate_content_safety(value, new_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                issues.extend(await self._validate_content_safety(item, new_path))
        
        return issues
    
    def _calculate_security_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate security score based on detected issues"""
        if not issues:
            return 1.0
        
        # Scoring weights by severity
        severity_weights = {
            ValidationSeverity.LOW: 0.05,
            ValidationSeverity.MEDIUM: 0.15,
            ValidationSeverity.HIGH: 0.35,
            ValidationSeverity.CRITICAL: 0.75
        }
        
        total_penalty = 0.0
        for issue in issues:
            total_penalty += severity_weights.get(issue.severity, 0.1)
        
        # Cap penalty at 1.0 to avoid negative scores
        total_penalty = min(total_penalty, 1.0)
        
        return max(0.0, 1.0 - total_penalty)
    
    async def _sanitize_payload(self, payload: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate sanitized version of payload"""
        # This is a simplified sanitization - in production, you'd want more sophisticated logic
        sanitized = json.loads(json.dumps(payload))  # Deep copy
        
        for issue in issues:
            if issue.threat == SecurityThreat.XSS_SCRIPT:
                # HTML encode the problematic content
                field_parts = issue.field_path.split('.')
                self._sanitize_field(sanitized, field_parts, html.escape)
            elif issue.threat == SecurityThreat.SENSITIVE_DATA:
                # Redact sensitive data
                field_parts = issue.field_path.split('.')
                self._sanitize_field(sanitized, field_parts, lambda x: "[REDACTED]")
        
        return sanitized
    
    def _sanitize_field(self, obj: Dict[str, Any], field_parts: List[str], sanitizer_func):
        """Apply sanitization function to a specific field"""
        try:
            current = obj
            for part in field_parts[:-1]:
                if '[' in part and ']' in part:
                    # Handle array access
                    key, index = part.split('[')
                    index = int(index.rstrip(']'))
                    current = current[key][index]
                else:
                    current = current[part]
            
            # Apply sanitizer to the final field
            final_part = field_parts[-1]
            if '[' in final_part and ']' in final_part:
                key, index = final_part.split('[')
                index = int(index.rstrip(']'))
                if isinstance(current[key][index], str):
                    current[key][index] = sanitizer_func(current[key][index])
            else:
                if isinstance(current[final_part], str):
                    current[final_part] = sanitizer_func(current[final_part])
        except (KeyError, IndexError, ValueError):
            # Field might not exist or be accessible for sanitization
            pass


# Global validator instance
payload_validator = EnhancedPayloadValidator()


async def validate_request_payload(
    payload: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Global function for payload validation
    
    Args:
        payload: Request payload to validate
        context: Optional validation context
        
    Returns:
        ValidationResult with security analysis
    """
    return await payload_validator.validate_payload(payload, context)


def create_security_violation_error(validation_result: ValidationResult) -> Dict[str, Any]:
    """Create error response for security violations"""
    critical_issues = [issue for issue in validation_result.issues 
                      if issue.severity == ValidationSeverity.CRITICAL]
    
    if critical_issues:
        primary_issue = critical_issues[0]
        message = f"Security violation detected: {primary_issue.message}"
    else:
        message = f"Payload validation failed. Security score: {validation_result.security_score:.2f}"
    
    return create_error_response(
        ErrorCode.VALIDATION,
        message,
        context={
            "security_score": validation_result.security_score,
            "issues": len(validation_result.issues),
            "critical_issues": len([i for i in validation_result.issues 
                                  if i.severity == ValidationSeverity.CRITICAL]),
            "validation_details": validation_result.to_dict()
        }
    )


def is_payload_safe(validation_result: ValidationResult, strict: bool = True) -> bool:
    """Check if payload is safe based on validation results
    
    Args:
        validation_result: Validation result to check
        strict: If True, reject any high/critical issues. If False, only reject critical.
        
    Returns:
        True if payload is considered safe
    """
    if strict:
        return not validation_result.has_high_severity_issues()
    else:
        return not validation_result.has_critical_issues()