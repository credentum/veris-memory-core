#!/usr/bin/env python3
"""
error_codes.py: Standardized v1.0 Error Codes for Veris Memory

This module provides the standardized error code system for Sprint 11 v1.0 API contract,
ensuring all errors include proper error_code, message, and trace_id fields.
"""

import uuid
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class ErrorCode(str, Enum):
    """Standardized v1.0 error codes for Veris Memory API"""
    
    # Input Validation Errors (4xx)
    VALIDATION = "ERR_VALIDATION"
    AUTH = "ERR_AUTH"  
    RATE_LIMIT = "ERR_RATE_LIMIT"
    
    # Server/Dependency Errors (5xx)
    TIMEOUT = "ERR_TIMEOUT"
    DEPENDENCY_DOWN = "ERR_DEPENDENCY_DOWN"
    
    # Sprint 11 Specific Errors
    DIMENSION_MISMATCH = "ERR_DIMENSION_MISMATCH"  # Critical for Phase 2


class ErrorSeverity(str, Enum):
    """Error severity levels for logging and monitoring"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StandardizedError:
    """Standardized error response builder for v1.0 API contract"""
    
    def __init__(self):
        self.trace_id_prefix = "trace_"
    
    def generate_trace_id(self) -> str:
        """Generate unique trace ID for request tracking"""
        return f"{self.trace_id_prefix}{uuid.uuid4().hex[:12]}"
    
    def create_error_response(
        self,
        error_code: ErrorCode,
        message: str,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Dict[str, Any]:
        """Create standardized v1.0 error response
        
        Args:
            error_code: Standardized error code from ErrorCode enum
            message: Human-readable error description
            trace_id: Optional trace ID (auto-generated if not provided)
            details: Optional additional error context
            severity: Error severity for monitoring
            
        Returns:
            Standardized error response dict
        """
        if trace_id is None:
            trace_id = self.generate_trace_id()
        
        error_response = {
            "success": False,
            "error_code": error_code.value,
            "message": message,
            "trace_id": trace_id,
            "details": details or {}
        }
        
        # Add metadata for internal use
        error_response["_metadata"] = {
            "severity": severity.value,
            "timestamp": datetime.utcnow().isoformat(),
            "category": self._get_error_category(error_code)
        }
        
        return error_response
    
    def _get_error_category(self, error_code: ErrorCode) -> str:
        """Categorize error for monitoring purposes"""
        if error_code in [ErrorCode.VALIDATION, ErrorCode.AUTH]:
            return "client_error"
        elif error_code == ErrorCode.RATE_LIMIT:
            return "rate_limiting"
        elif error_code in [ErrorCode.TIMEOUT, ErrorCode.DEPENDENCY_DOWN]:
            return "infrastructure"
        elif error_code == ErrorCode.DIMENSION_MISMATCH:
            return "data_integrity"
        else:
            return "unknown"


# Pre-configured error builders for common scenarios
class CommonErrors:
    """Pre-configured error responses for common scenarios"""
    
    def __init__(self):
        self.error_builder = StandardizedError()
    
    def validation_error(
        self, 
        field: str, 
        reason: str,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create validation error response"""
        return self.error_builder.create_error_response(
            ErrorCode.VALIDATION,
            f"Input validation failed: {reason}",
            trace_id=trace_id,
            details={
                "field": field,
                "reason": reason,
                "fix_suggestion": f"Please check the {field} field and ensure it meets the required format"
            },
            severity=ErrorSeverity.LOW
        )
    
    def rate_limit_error(
        self,
        limit: int,
        window: str,
        retry_after: int,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create rate limit error response"""
        return self.error_builder.create_error_response(
            ErrorCode.RATE_LIMIT,
            f"Rate limit exceeded: {limit} requests per {window}",
            trace_id=trace_id,
            details={
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            },
            severity=ErrorSeverity.MEDIUM
        )
    
    def dependency_down_error(
        self,
        service: str,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create dependency unavailable error response"""
        return self.error_builder.create_error_response(
            ErrorCode.DEPENDENCY_DOWN,
            f"Service dependency unavailable: {service}",
            trace_id=trace_id,
            details={
                "service": service,
                "impact": "Reduced functionality while service is unavailable",
                "recommended_action": "Check service health and retry"
            },
            severity=ErrorSeverity.HIGH
        )
    
    def dimension_mismatch_error(
        self,
        expected: int,
        actual: int,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create dimension mismatch error response (Critical for Sprint 11 Phase 2)"""
        return self.error_builder.create_error_response(
            ErrorCode.DIMENSION_MISMATCH,
            f"Vector dimension mismatch: expected {expected}, got {actual}",
            trace_id=trace_id,
            details={
                "expected_dimensions": expected,
                "actual_dimensions": actual,
                "required_action": "Vector collections must be recreated with correct dimensions",
                "breaking_change": True
            },
            severity=ErrorSeverity.CRITICAL
        )
    
    def timeout_error(
        self,
        operation: str,
        timeout_seconds: int,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create timeout error response"""
        return self.error_builder.create_error_response(
            ErrorCode.TIMEOUT,
            f"Operation timeout: {operation} exceeded {timeout_seconds}s",
            trace_id=trace_id,
            details={
                "operation": operation,
                "timeout_seconds": timeout_seconds,
                "suggested_action": "Retry with exponential backoff"
            },
            severity=ErrorSeverity.MEDIUM
        )
    
    def auth_error(
        self,
        reason: str = "Invalid authentication",
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create authentication error response"""
        return self.error_builder.create_error_response(
            ErrorCode.AUTH,
            f"Authentication failed: {reason}",
            trace_id=trace_id,
            details={
                "reason": reason,
                "required_action": "Provide valid authentication credentials"
            },
            severity=ErrorSeverity.HIGH
        )


# HTTP Status Code Mapping for v1.0 API
ERROR_HTTP_STATUS_MAP = {
    ErrorCode.VALIDATION: 400,  # Bad Request
    ErrorCode.AUTH: 401,        # Unauthorized
    ErrorCode.RATE_LIMIT: 429,  # Too Many Requests
    ErrorCode.TIMEOUT: 408,     # Request Timeout
    ErrorCode.DEPENDENCY_DOWN: 503,  # Service Unavailable
    ErrorCode.DIMENSION_MISMATCH: 409,  # Conflict
}


def get_http_status_for_error(error_code: ErrorCode) -> int:
    """Get appropriate HTTP status code for error"""
    return ERROR_HTTP_STATUS_MAP.get(error_code, 500)  # Default to 500


# Global instances for easy import
error_builder = StandardizedError()
common_errors = CommonErrors()


def create_error_response(
    error_code: ErrorCode,
    message: str,
    trace_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Dict[str, Any]:
    """Global function for creating standardized error responses
    
    Args:
        error_code: Standardized error code from ErrorCode enum
        message: Human-readable error description
        trace_id: Optional trace ID (auto-generated if not provided)
        context: Optional additional error context
        severity: Error severity for monitoring
        
    Returns:
        Standardized error response dict
    """
    return error_builder.create_error_response(
        error_code=error_code,
        message=message,
        trace_id=trace_id,
        details=context,
        severity=severity
    )


# Export main components
__all__ = [
    "ErrorCode",
    "ErrorSeverity", 
    "StandardizedError",
    "CommonErrors",
    "create_error_response",
    "error_builder",
    "common_errors",
    "get_http_status_for_error"
]