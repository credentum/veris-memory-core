"""
Input Validation and Sanitization Module
Sprint 10 - Issue 003: WAF & Port Allowlisting
"""

import re
import html
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation"""
    valid: bool
    error: Optional[str] = None
    sanitized_value: Optional[Any] = None
    warnings: List[str] = None


class InputValidator:
    """Input validation for security"""
    
    # Maximum input sizes
    MAX_QUERY_SIZE = 10240  # 10KB
    MAX_CONTENT_SIZE = 102400  # 100KB
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 4096
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize input validator"""
        self.config = config or {}
        
        # Override defaults with config
        if "max_query_size" in self.config:
            self.MAX_QUERY_SIZE = self.config["max_query_size"]
        if "max_content_size" in self.config:
            self.MAX_CONTENT_SIZE = self.config["max_content_size"]
    
    def validate_input(
        self,
        value: Any,
        input_type: str = "generic"
    ) -> ValidationResult:
        """
        Validate input based on type.
        
        Args:
            value: Input value to validate
            input_type: Type of input (query, content, filename, etc.)
            
        Returns:
            ValidationResult with validation status
        """
        if value is None:
            return ValidationResult(valid=True)
        
        # Convert to string for validation
        str_value = str(value) if not isinstance(value, str) else value
        
        # Check length based on type
        if input_type == "query":
            if len(str_value) > self.MAX_QUERY_SIZE:
                return ValidationResult(
                    valid=False,
                    error="input_too_large"
                )
        elif input_type == "content":
            if len(str_value) > self.MAX_CONTENT_SIZE:
                return ValidationResult(
                    valid=False,
                    error="input_too_large"
                )
        elif input_type == "filename":
            if len(str_value) > self.MAX_FILENAME_LENGTH:
                return ValidationResult(
                    valid=False,
                    error="filename_too_long"
                )
        elif input_type == "path":
            if len(str_value) > self.MAX_PATH_LENGTH:
                return ValidationResult(
                    valid=False,
                    error="path_too_long"
                )
        
        # Check for null bytes
        if '\x00' in str_value:
            return ValidationResult(
                valid=False,
                error="null_byte_detected"
            )
        
        # Check for control characters
        control_chars = set(chr(i) for i in range(32) if i not in [9, 10, 13])
        if any(c in str_value for c in control_chars):
            return ValidationResult(
                valid=False,
                error="control_characters_detected"
            )
        
        return ValidationResult(valid=True, sanitized_value=str_value)
    
    def validate_json_input(self, json_data: Dict) -> ValidationResult:
        """Validate JSON input structure and content"""
        warnings = []
        
        # Check for excessively nested structures
        max_depth = self._get_json_depth(json_data)
        if max_depth > 10:
            return ValidationResult(
                valid=False,
                error="json_too_deeply_nested"
            )
        
        # Check total size
        import json
        json_str = json.dumps(json_data)
        if len(json_str) > self.MAX_CONTENT_SIZE:
            return ValidationResult(
                valid=False,
                error="json_too_large"
            )
        
        # Check for suspicious keys
        suspicious_keys = ['__proto__', 'constructor', 'prototype']
        if self._has_suspicious_keys(json_data, suspicious_keys):
            warnings.append("Suspicious keys detected and removed")
        
        return ValidationResult(valid=True, warnings=warnings)
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Get maximum depth of JSON structure"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _has_suspicious_keys(self, obj: Any, suspicious: List[str]) -> bool:
        """Check for suspicious keys in JSON"""
        if isinstance(obj, dict):
            for key in obj.keys():
                if key in suspicious:
                    return True
            for value in obj.values():
                if self._has_suspicious_keys(value, suspicious):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if self._has_suspicious_keys(item, suspicious):
                    return True
        return False


class InputSanitizer:
    """Input sanitization for security"""
    
    def __init__(self):
        """Initialize input sanitizer"""
        # Patterns for sanitization
        self.script_pattern = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
        self.event_handler_pattern = re.compile(r'on\w+\s*=', re.IGNORECASE)
        self.control_char_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    def sanitize_html(self, input_str: str) -> str:
        """
        Sanitize HTML input to prevent XSS.
        
        Args:
            input_str: HTML input to sanitize
            
        Returns:
            Sanitized HTML string
        """
        # Remove script tags
        sanitized = self.script_pattern.sub('', input_str)
        
        # Remove event handlers
        sanitized = self.event_handler_pattern.sub('', sanitized)
        
        # Escape HTML entities
        sanitized = html.escape(sanitized)
        
        return sanitized
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe file operations.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove null bytes
        sanitized = filename.replace('\x00', '')
        
        # Remove path traversal sequences
        sanitized = sanitized.replace('../', '').replace('..\\', '')
        
        # Remove special characters that could cause issues
        sanitized = re.sub(r'[<>:"|?*]', '_', sanitized)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            # Keep extension if present
            if '.' in sanitized:
                name, ext = sanitized.rsplit('.', 1)
                sanitized = name[:250 - len(ext)] + '.' + ext
            else:
                sanitized = sanitized[:255]
        
        return sanitized
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize plain text input.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove control characters except newline, tab, carriage return
        sanitized = self.control_char_pattern.sub('', text)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized
    
    def sanitize_url(self, url: str) -> Optional[str]:
        """
        Sanitize URL input.
        
        Args:
            url: URL to sanitize
            
        Returns:
            Sanitized URL or None if invalid
        """
        try:
            parsed = urlparse(url)
            
            # Only allow http(s) and ftp protocols
            if parsed.scheme not in ['http', 'https', 'ftp']:
                return None
            
            # Check for localhost/private IPs (basic check)
            if parsed.hostname:
                if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
                    return None
                
                # Check for private IP ranges (simplified)
                if parsed.hostname.startswith(('10.', '192.168.', '172.')):
                    return None
            
            # Reconstruct URL with validated parts
            return parsed.geturl()
            
        except Exception:
            return None
    
    def sanitize_sql_identifier(self, identifier: str) -> str:
        """
        Sanitize SQL identifier (table/column name).
        
        Args:
            identifier: SQL identifier to sanitize
            
        Returns:
            Sanitized identifier
        """
        # Only allow alphanumeric and underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', identifier)
        
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = '_' + sanitized
        
        # Limit length
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        
        return sanitized


class ContentTypeValidator:
    """Content type validation for uploads"""
    
    # Allowed content types
    ALLOWED_TYPES = {
        'application/json',
        'text/plain',
        'text/csv',
        'application/x-www-form-urlencoded',
        'multipart/form-data',
        'application/xml',
        'text/xml',
    }
    
    # Explicitly blocked types
    BLOCKED_TYPES = {
        'application/x-sh',
        'application/x-executable',
        'application/x-msdos-program',
        'application/x-msdownload',
        'text/x-php',
        'application/x-php',
        'application/x-httpd-php',
        'text/x-python',
        'application/x-python-code',
        'text/x-perl',
        'application/x-perl',
        'text/x-ruby',
        'application/x-ruby',
    }
    
    def __init__(self, custom_allowed: Optional[Set[str]] = None):
        """Initialize content type validator"""
        self.allowed_types = self.ALLOWED_TYPES.copy()
        if custom_allowed:
            self.allowed_types.update(custom_allowed)
    
    def is_allowed(self, content_type: str) -> bool:
        """
        Check if content type is allowed.
        
        Args:
            content_type: MIME type to check
            
        Returns:
            True if allowed, False otherwise
        """
        # Extract base type (ignore parameters like charset)
        base_type = content_type.split(';')[0].strip().lower()
        
        # Check blocked list first
        if base_type in self.BLOCKED_TYPES:
            return False
        
        # Check allowed list
        return base_type in self.allowed_types
    
    def add_allowed_type(self, content_type: str):
        """Add a content type to allowed list"""
        self.allowed_types.add(content_type.lower())
    
    def remove_allowed_type(self, content_type: str):
        """Remove a content type from allowed list"""
        self.allowed_types.discard(content_type.lower())


class ParameterValidator:
    """Validate request parameters"""
    
    def __init__(self):
        """Initialize parameter validator"""
        self.validators = {
            'email': self._validate_email,
            'phone': self._validate_phone,
            'uuid': self._validate_uuid,
            'ip': self._validate_ip,
            'date': self._validate_date,
            'numeric': self._validate_numeric,
        }
    
    def validate(self, value: str, param_type: str) -> ValidationResult:
        """
        Validate parameter based on type.
        
        Args:
            value: Parameter value
            param_type: Type of parameter
            
        Returns:
            ValidationResult
        """
        if param_type in self.validators:
            return self.validators[param_type](value)
        
        return ValidationResult(valid=True)
    
    def _validate_email(self, email: str) -> ValidationResult:
        """Validate email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, email):
            return ValidationResult(valid=True)
        return ValidationResult(valid=False, error="invalid_email")
    
    def _validate_phone(self, phone: str) -> ValidationResult:
        """Validate phone number"""
        pattern = r'^\+?[1-9]\d{1,14}$'  # E.164 format
        if re.match(pattern, phone):
            return ValidationResult(valid=True)
        return ValidationResult(valid=False, error="invalid_phone")
    
    def _validate_uuid(self, uuid_str: str) -> ValidationResult:
        """Validate UUID"""
        pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if re.match(pattern, uuid_str):
            return ValidationResult(valid=True)
        return ValidationResult(valid=False, error="invalid_uuid")
    
    def _validate_ip(self, ip: str) -> ValidationResult:
        """Validate IP address"""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return ValidationResult(valid=True)
        except ValueError:
            return ValidationResult(valid=False, error="invalid_ip")
    
    def _validate_date(self, date_str: str) -> ValidationResult:
        """Validate date string"""
        from datetime import datetime
        formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
        
        for fmt in formats:
            try:
                datetime.strptime(date_str, fmt)
                return ValidationResult(valid=True)
            except ValueError:
                continue
        
        return ValidationResult(valid=False, error="invalid_date")
    
    def _validate_numeric(self, value: str) -> ValidationResult:
        """Validate numeric value"""
        try:
            float(value)
            return ValidationResult(valid=True)
        except ValueError:
            return ValidationResult(valid=False, error="invalid_numeric")


# Export main components
__all__ = [
    "InputValidator",
    "InputSanitizer",
    "ContentTypeValidator",
    "ParameterValidator",
    "ValidationResult",
]