#!/usr/bin/env python3
"""
API Middleware Components

Custom middleware for error handling, request validation, logging,
and observability features.
"""

import json
import time
import traceback
import uuid
from typing import Callable, Dict, Any, Optional

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from .models import ErrorResponse, ErrorDetail, ErrorCode
from ..utils.logging_middleware import api_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware with structured logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive logging."""
        # Generate trace ID
        trace_id = str(uuid.uuid4())[:8]
        request.state.trace_id = trace_id
        
        start_time = time.time()
        
        # Log request
        api_logger.info(
            "API request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            trace_id=trace_id
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate timing
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            api_logger.info(
                "API request completed",
                status_code=response.status_code,
                duration_ms=duration_ms,
                trace_id=trace_id
            )
            
            # Add trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            api_logger.error(
                "API request failed",
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                trace_id=trace_id,
                traceback=traceback.format_exc()
            )
            
            # Re-raise to be handled by error middleware
            raise


class ValidationMiddleware(BaseHTTPMiddleware):
    """Request validation middleware with detailed error responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with validation error handling."""
        try:
            return await call_next(request)
            
        except ValidationError as e:
            trace_id = getattr(request.state, 'trace_id', 'unknown')
            
            # Format validation errors
            error_details = []
            for error in e.errors():
                error_details.append({
                    "field": ".".join(str(x) for x in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input")
                })
            
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.VALIDATION_ERROR,
                    message="Request validation failed",
                    details={"validation_errors": error_details},
                    trace_id=trace_id
                )
            )
            
            api_logger.warning(
                "Request validation failed",
                validation_errors=error_details,
                trace_id=trace_id
            )
            
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=error_response.model_dump()
            )
        
        except Exception:
            # Let other exceptions pass through to error handler
            raise


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware with structured error responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive error handling."""
        try:
            return await call_next(request)
            
        except Exception as e:
            trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4())[:8])
            
            # Determine error type and response
            error_code, status_code, message, details = self._classify_error(e)
            
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code=error_code,
                    message=message,
                    details=details,
                    trace_id=trace_id
                )
            )
            
            api_logger.error(
                f"API error: {error_code}",
                error_message=message,
                error_type=type(e).__name__,
                status_code=status_code,
                trace_id=trace_id,
                details=details,
                exception=str(e)
            )
            
            return JSONResponse(
                status_code=status_code,
                content=error_response.model_dump(),
                headers={"X-Trace-ID": trace_id}
            )
    
    def _classify_error(self, error: Exception) -> tuple[ErrorCode, int, str, Optional[Dict[str, Any]]]:
        """Classify error and determine appropriate response."""
        error_type = type(error).__name__
        error_str = str(error)
        
        # Authentication/Authorization errors
        if "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
            return (
                ErrorCode.AUTHENTICATION_ERROR,
                status.HTTP_401_UNAUTHORIZED,
                "Authentication required",
                {"error_type": error_type}
            )
        
        if "authorization" in error_str.lower() or "forbidden" in error_str.lower():
            return (
                ErrorCode.AUTHORIZATION_ERROR,
                status.HTTP_403_FORBIDDEN,
                "Access denied",
                {"error_type": error_type}
            )
        
        # Rate limiting errors
        if "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
            return (
                ErrorCode.RATE_LIMIT_ERROR,
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Rate limit exceeded",
                {"error_type": error_type}
            )
        # Timeout errors (must come before backend errors to avoid being caught by "timeout" keyword)
        if "timeout" in error_str.lower() or error_type in ["TimeoutError", "asyncio.TimeoutError"]:
            return (
                ErrorCode.TIMEOUT_ERROR,
                status.HTTP_504_GATEWAY_TIMEOUT,
                "Request timeout",
                {"error_type": error_type}
            )
        
        # Backend errors
        if any(keyword in error_str.lower() for keyword in ["backend", "database", "connection"]):
            return (
                ErrorCode.BACKEND_ERROR,
                status.HTTP_502_BAD_GATEWAY,
                "Backend service error",
                {"error_type": error_type}
            )
        
        # Not found errors
        if "not found" in error_str.lower() or error_type in ["FileNotFoundError", "KeyError"]:
            return (
                ErrorCode.NOT_FOUND,
                status.HTTP_404_NOT_FOUND,
                "Resource not found",
                {"error_type": error_type}
            )
        
        # Default to internal error
        return (
            ErrorCode.INTERNAL_ERROR,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            {
                "error_type": error_type,
                "message": error_str[:200]  # Limit error message length
            }
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with configurable limits."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old requests
        self._cleanup_old_requests(client_ip, current_time)
        
        # Check rate limit
        if self._is_rate_limited(client_ip, current_time):
            trace_id = getattr(request.state, 'trace_id', str(uuid.uuid4())[:8])
            
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code=ErrorCode.RATE_LIMIT_ERROR,
                    message="Rate limit exceeded",
                    details={
                        "limit": self.requests_per_minute,
                        "window": "1 minute",
                        "client_ip": client_ip
                    },
                    trace_id=trace_id
                )
            )
            
            api_logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                requests_per_minute=self.requests_per_minute,
                trace_id=trace_id
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_response.model_dump(),
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 60)),
                    "Retry-After": "60"
                }
            )
        
        # Record request
        self._record_request(client_ip, current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, self.requests_per_minute - len(self.client_requests.get(client_ip, [])))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check for forwarded headers (from proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_requests(self, client_ip: str, current_time: float):
        """Remove requests older than 1 minute."""
        if client_ip not in self.client_requests:
            return
        
        cutoff_time = current_time - 60  # 1 minute ago
        self.client_requests[client_ip] = [
            req_time for req_time in self.client_requests[client_ip]
            if req_time > cutoff_time
        ]
    
    def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client has exceeded rate limit."""
        if client_ip not in self.client_requests:
            return False
        
        return len(self.client_requests[client_ip]) >= self.requests_per_minute
    
    def _record_request(self, client_ip: str, current_time: float):
        """Record a request for rate limiting."""
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []
        
        self.client_requests[client_ip].append(current_time)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Metrics collection middleware for observability."""
    
    # Class-level shared state so all instances share the same metrics
    request_count = 0
    response_times = []
    status_counts = {}
    endpoint_metrics = {}
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with metrics collection."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Collect metrics
            duration_ms = (time.time() - start_time) * 1000
            self._record_metrics(request, response, duration_ms)
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration_ms = (time.time() - start_time) * 1000
            self._record_error_metrics(request, e, duration_ms)
            raise
    
    def _record_metrics(self, request: Request, response: Response, duration_ms: float):
        """Record successful request metrics."""
        MetricsMiddleware.request_count += 1
        MetricsMiddleware.response_times.append(duration_ms)
        
        # Limit response times list to prevent memory growth
        if len(MetricsMiddleware.response_times) > 10000:
            MetricsMiddleware.response_times = MetricsMiddleware.response_times[-5000:]
        
        # Status code metrics
        status_code = response.status_code
        MetricsMiddleware.status_counts[status_code] = MetricsMiddleware.status_counts.get(status_code, 0) + 1
        
        # Endpoint metrics
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in MetricsMiddleware.endpoint_metrics:
            MetricsMiddleware.endpoint_metrics[endpoint] = {"count": 0, "total_time": 0, "errors": 0}
        
        MetricsMiddleware.endpoint_metrics[endpoint]["count"] += 1
        MetricsMiddleware.endpoint_metrics[endpoint]["total_time"] += duration_ms
    
    def _record_error_metrics(self, request: Request, error: Exception, duration_ms: float):
        """Record error metrics."""
        MetricsMiddleware.request_count += 1
        MetricsMiddleware.response_times.append(duration_ms)
        
        # Error metrics
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in MetricsMiddleware.endpoint_metrics:
            MetricsMiddleware.endpoint_metrics[endpoint] = {"count": 0, "total_time": 0, "errors": 0}
        
        MetricsMiddleware.endpoint_metrics[endpoint]["errors"] += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if not MetricsMiddleware.response_times:
            return {
                "request_count": 0,
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "p99_response_time_ms": 0,
                "status_counts": {},
                "endpoint_metrics": {}
            }
        
        sorted_times = sorted(MetricsMiddleware.response_times)
        count = len(sorted_times)
        
        return {
            "request_count": MetricsMiddleware.request_count,
            "avg_response_time_ms": sum(sorted_times) / count,
            "p95_response_time_ms": sorted_times[int(count * 0.95)] if count > 0 else 0,
            "p99_response_time_ms": sorted_times[int(count * 0.99)] if count > 0 else 0,
            "status_counts": MetricsMiddleware.status_counts.copy(),
            "endpoint_metrics": MetricsMiddleware.endpoint_metrics.copy()
        }


# Global metrics instance
metrics_middleware = MetricsMiddleware(None)