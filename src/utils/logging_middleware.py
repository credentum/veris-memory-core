#!/usr/bin/env python3
"""
Structured logging middleware with trace ID support for Veris Memory.

This module provides enhanced logging capabilities with structured JSON output,
trace ID propagation, and performance timing for observability.
"""

import json
import time
import uuid
import logging
from contextvars import ContextVar
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, asdict
from enum import Enum


# Context variables for request tracing
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')
request_start_time_var: ContextVar[float] = ContextVar('request_start_time', default=0.0)


class LogLevel(str, Enum):
    """Logging levels for structured output."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry format."""
    timestamp: float
    level: str
    message: str
    logger_name: str
    trace_id: str
    module: Optional[str] = None
    function: Optional[str] = None
    duration_ms: Optional[float] = None
    backend: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Remove None values
        return {k: v for k, v in data.items() if v is not None}


class StructuredLogger:
    """
    Enhanced logger with structured JSON output and trace ID support.
    
    Provides consistent logging format across all Veris Memory components
    with built-in performance timing and metadata support.
    """
    
    def __init__(self, name: str, enable_console: bool = True, pii_redaction: bool = True):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically module name)
            enable_console: Whether to enable console output
            pii_redaction: Whether to redact sensitive information
        """
        self.name = name
        self.enable_console = enable_console
        self.pii_redaction = pii_redaction
        
        # Set up standard Python logger as fallback
        self.stdlib_logger = logging.getLogger(name)
        
        # PII patterns to redact
        self.pii_patterns = [
            'password', 'secret', 'key', 'token', 'auth',
            'user_id', 'email', 'phone', 'ssn', 'credit_card'
        ]
    
    def _redact_pii(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact potentially sensitive information."""
        if not self.pii_redaction:
            return data
        
        redacted = {}
        for key, value in data.items():
            if any(pattern in key.lower() for pattern in self.pii_patterns):
                if isinstance(value, str) and len(value) > 4:
                    redacted[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    redacted[key] = '***REDACTED***'
            elif isinstance(value, dict):
                redacted[key] = self._redact_pii(value)
            else:
                redacted[key] = value
        return redacted
    
    def _emit_log(self, level: LogLevel, message: str, **kwargs):
        """Emit a structured log entry."""
        # Extract trace_id from kwargs if present to avoid conflict
        provided_trace_id = kwargs.pop('trace_id', None)
        trace_id = provided_trace_id or trace_id_var.get() or str(uuid.uuid4())[:8]
        
        # Extract known LogEntry fields from kwargs
        log_entry_fields = {}
        for field in ['module', 'function', 'duration_ms', 'backend']:
            if field in kwargs:
                log_entry_fields[field] = kwargs.pop(field)
        
        # Put remaining kwargs in metadata
        metadata = kwargs if kwargs else None
        
        log_entry = LogEntry(
            timestamp=time.time(),
            level=level.value,
            message=message,
            logger_name=self.name,
            trace_id=trace_id,
            metadata=metadata,
            **log_entry_fields
        )
        
        # Convert to dictionary and redact PII
        log_dict = log_entry.to_dict()
        if log_dict.get('metadata'):
            log_dict['metadata'] = self._redact_pii(log_dict['metadata'])
        
        # Output structured JSON
        if self.enable_console:
            print(json.dumps(log_dict))
        
        # Also log to stdlib logger for compatibility
        self.stdlib_logger.log(getattr(logging, level.value), message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._emit_log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._emit_log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._emit_log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._emit_log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._emit_log(LogLevel.CRITICAL, message, **kwargs)
    
    def log_backend_timing(self, backend: str, operation: str, duration_ms: float, 
                          result_count: Optional[int] = None, top_score: Optional[float] = None,
                          **metadata):
        """Log backend operation timing."""
        self.info(
            f"Backend operation completed",
            backend=backend,
            module=operation,
            duration_ms=duration_ms,
            metadata={
                'result_count': result_count,
                'top_score': top_score,
                **metadata
            }
        )


@asynccontextmanager
async def log_backend_timing(backend_name: str, operation: str, logger: StructuredLogger):
    """
    Async context manager for timing backend operations.
    
    Args:
        backend_name: Name of the backend (vector, graph, kv)
        operation: Operation being performed (search, health_check, etc.)
        logger: Logger instance to use
        
    Yields:
        Dict to store operation metadata
    """
    start_time = time.time()
    metadata = {}
    
    try:
        yield metadata
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Backend operation failed",
            backend=backend_name,
            module=operation,
            duration_ms=duration_ms,
            metadata={'error': str(e), **metadata}
        )
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.log_backend_timing(
            backend_name, 
            operation, 
            duration_ms, 
            **metadata
        )


@contextmanager
def trace_context(trace_id: Optional[str] = None):
    """
    Context manager for setting trace ID for a request.
    
    Args:
        trace_id: Optional trace ID (will generate if not provided)
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    token = trace_id_var.set(trace_id)
    start_token = request_start_time_var.set(time.time())
    
    try:
        yield trace_id
    finally:
        trace_id_var.reset(token)
        request_start_time_var.reset(start_token)


def get_current_trace_id() -> str:
    """Get the current trace ID from context."""
    return trace_id_var.get() or str(uuid.uuid4())[:8]


def get_request_duration_ms() -> float:
    """Get the current request duration in milliseconds."""
    start_time = request_start_time_var.get()
    if start_time > 0:
        return (time.time() - start_time) * 1000
    return 0.0


class TimingCollector:
    """Collect timing information across multiple operations."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def record_timing(self, operation: str, duration_ms: float, **metadata):
        """Record a timing measurement."""
        if operation not in self.timings:
            self.timings[operation] = []
            self.metadata[operation] = {}
        
        self.timings[operation].append(duration_ms)
        self.metadata[operation].update(metadata)
    
    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get timing summary with statistics."""
        summary = {}
        
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'metadata': self.metadata.get(operation, {})
                }
        
        return summary


# Global logger instances for common components
search_logger = StructuredLogger("veris.search")
backend_logger = StructuredLogger("veris.backend")
api_logger = StructuredLogger("veris.api")
ranking_logger = StructuredLogger("veris.ranking")


def setup_logging(level: str = "INFO", enable_json: bool = True):
    """
    Set up global logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to enable JSON structured output
    """
    # Configure standard logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up structured loggers based on configuration
    for logger_name in ['veris.search', 'veris.backend', 'veris.api', 'veris.ranking']:
        structured_logger = StructuredLogger(logger_name, enable_console=enable_json)
        globals()[logger_name.split('.')[-1] + '_logger'] = structured_logger