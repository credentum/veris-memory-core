#!/usr/bin/env python3
"""
Request Metrics Middleware for Real-Time Latency and Error Tracking

Provides FastAPI middleware for tracking:
- Request latency (avg, p95, p99)
- Error rates by endpoint
- Request counts and throughput
- Time-series data for trending
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
# FastAPI imports - these are required dependencies
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
FASTAPI_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Individual request metric data point."""
    timestamp: datetime
    method: str
    path: str
    status_code: int
    duration_ms: float
    error: Optional[str] = None


@dataclass
class EndpointStats:
    """Statistics for a specific endpoint."""
    request_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    durations: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_request: Optional[datetime] = None
    
    @property
    def error_rate_percent(self) -> float:
        """Calculate error rate percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.request_count == 0:
            return 0.0
        return self.total_duration_ms / self.request_count
    
    @property
    def p95_duration_ms(self) -> float:
        """Calculate 95th percentile duration."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]
    
    @property
    def p99_duration_ms(self) -> float:
        """Calculate 99th percentile duration."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.99)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]


class RequestMetricsCollector:
    """Collects and aggregates request metrics in real-time with security controls."""
    
    def __init__(self, max_history_minutes: int = 60, max_requests_stored: int = 10000):
        # Input validation
        if max_history_minutes <= 0 or max_history_minutes > 1440:  # Max 24 hours
            raise ValueError("max_history_minutes must be between 1 and 1440 (24 hours)")
        if max_requests_stored <= 0 or max_requests_stored > 50000:  # Max 50k requests
            raise ValueError("max_requests_stored must be between 1 and 50000")
            
        self.max_history_minutes = max_history_minutes
        self.max_requests_stored = max_requests_stored
        self.endpoint_stats: Dict[str, EndpointStats] = defaultdict(EndpointStats)
        self.recent_requests: deque = deque(maxlen=max_requests_stored)
        self.total_requests = 0
        self.total_errors = 0
        self._lock = asyncio.Lock()
        
        # Bounded task queue for metrics recording
        self._metrics_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._processing_enabled = True
        
        # Rate limiting for cleanup operations
        self._last_cleanup = datetime.utcnow()
        self._cleanup_interval_seconds = 30  # Cleanup every 30 seconds max
        
    async def start_queue_processor(self) -> None:
        """Start the bounded queue processor for metrics recording."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._processing_enabled = True
            self._queue_processor_task = asyncio.create_task(self._process_metrics_queue())
            logger.info("Started metrics queue processor")
    
    async def stop_queue_processor(self) -> None:
        """Stop the metrics queue processor gracefully."""
        self._processing_enabled = False
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped metrics queue processor")
    
    async def record_request(self, method: str, path: str, status_code: int, 
                           duration_ms: float, error: Optional[str] = None) -> None:
        """Record a request metric using bounded queue for backpressure handling."""
        # Input validation
        if not method or len(method) > 10:
            logger.warning(f"Invalid method: {method}")
            return
        if not path or len(path) > 200:
            logger.warning(f"Invalid path length: {len(path) if path else 0}")
            return
        if duration_ms < 0 or duration_ms > 300000:  # Max 5 minutes
            logger.warning(f"Invalid duration: {duration_ms}ms")
            return
        
        metric_data = {
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration_ms': duration_ms,
            'error': error,
            'timestamp': datetime.utcnow()
        }
        
        try:
            # Non-blocking put with immediate failure if queue is full
            self._metrics_queue.put_nowait(metric_data)
        except asyncio.QueueFull:
            logger.warning("Metrics queue full, dropping request metric")
            # Could implement sampling here - only record every Nth request when overloaded
    
    async def _process_metrics_queue(self) -> None:
        """Process metrics from the bounded queue."""
        while self._processing_enabled:
            try:
                # Wait for metrics with timeout to allow graceful shutdown
                metric_data = await asyncio.wait_for(
                    self._metrics_queue.get(), 
                    timeout=1.0
                )
                
                await self._record_metric_direct(metric_data)
                self._metrics_queue.task_done()
                
            except asyncio.TimeoutError:
                # Normal timeout, continue processing
                continue
            except asyncio.CancelledError:
                logger.info("Metrics queue processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing metric: {e}")
                continue
    
    async def _record_metric_direct(self, metric_data: Dict) -> None:
        """Directly record a metric (internal method with full validation already done)."""
        async with self._lock:
            endpoint_key = f"{metric_data['method']} {metric_data['path']}"
            metric = RequestMetric(
                timestamp=metric_data['timestamp'],
                method=metric_data['method'],
                path=metric_data['path'],
                status_code=metric_data['status_code'],
                duration_ms=metric_data['duration_ms'],
                error=metric_data['error']
            )
            
            # Update endpoint stats
            stats = self.endpoint_stats[endpoint_key]
            stats.request_count += 1
            stats.total_duration_ms += metric_data['duration_ms']
            stats.durations.append(metric_data['duration_ms'])
            stats.last_request = metric.timestamp
            
            if metric_data['status_code'] >= 400:
                stats.error_count += 1
                self.total_errors += 1
            
            # Add to recent requests
            self.recent_requests.append(metric)
            self.total_requests += 1
            
            # Rate-limited cleanup
            await self._cleanup_old_data_rate_limited()
    
    async def _cleanup_old_data_rate_limited(self) -> None:
        """Remove data older than max_history_minutes with rate limiting."""
        now = datetime.utcnow()
        if (now - self._last_cleanup).total_seconds() >= self._cleanup_interval_seconds:
            await self._cleanup_old_data()
            self._last_cleanup = now
    
    async def _cleanup_old_data(self) -> None:
        """Remove data older than max_history_minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=self.max_history_minutes)
        
        # Clean recent requests with batch processing to avoid blocking
        cleanup_count = 0
        max_cleanup_per_cycle = 1000  # Limit cleanup operations per cycle
        
        while (self.recent_requests and 
               self.recent_requests[0].timestamp < cutoff_time and
               cleanup_count < max_cleanup_per_cycle):
            self.recent_requests.popleft()
            cleanup_count += 1
            
        if cleanup_count > 0:
            logger.debug(f"Cleaned up {cleanup_count} old request metrics")
    
    async def get_global_stats(self) -> Dict[str, float]:
        """Get global statistics across all endpoints."""
        async with self._lock:
            if not self.recent_requests:
                return {
                    'total_requests': 0,
                    'total_errors': 0,
                    'error_rate_percent': 0.0,
                    'avg_duration_ms': 0.0,
                    'p95_duration_ms': 0.0,
                    'p99_duration_ms': 0.0,
                    'requests_per_minute': 0.0
                }
            
            # Calculate from recent requests
            durations = [r.duration_ms for r in self.recent_requests]
            errors = [r for r in self.recent_requests if r.status_code >= 400]
            
            # Calculate requests per minute
            now = datetime.utcnow()
            one_minute_ago = now - timedelta(minutes=1)
            recent_minute_requests = [
                r for r in self.recent_requests 
                if r.timestamp >= one_minute_ago
            ]
            
            # Sort durations for percentiles
            sorted_durations = sorted(durations)
            
            return {
                'total_requests': len(self.recent_requests),
                'total_errors': len(errors),
                'error_rate_percent': (len(errors) / len(self.recent_requests)) * 100,
                'avg_duration_ms': sum(durations) / len(durations) if durations else 0.0,
                'p95_duration_ms': sorted_durations[int(len(sorted_durations) * 0.95)] if sorted_durations else 0.0,
                'p99_duration_ms': sorted_durations[int(len(sorted_durations) * 0.99)] if sorted_durations else 0.0,
                'requests_per_minute': len(recent_minute_requests)
            }
    
    async def get_endpoint_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics by endpoint."""
        async with self._lock:
            return {
                endpoint: {
                    'request_count': stats.request_count,
                    'error_count': stats.error_count,
                    'error_rate_percent': stats.error_rate_percent,
                    'avg_duration_ms': stats.avg_duration_ms,
                    'p95_duration_ms': stats.p95_duration_ms,
                    'p99_duration_ms': stats.p99_duration_ms,
                    'last_request': stats.last_request.isoformat() if stats.last_request else None
                }
                for endpoint, stats in self.endpoint_stats.items()
            }
    
    async def get_trending_data(self, minutes: int = 5) -> List[Dict[str, float]]:
        """Get trending data for the last N minutes with input validation."""
        # Input validation for DoS protection
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        if minutes > 1440:  # Max 24 hours
            raise ValueError("minutes cannot exceed 1440 (24 hours)")
        
        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            recent = [r for r in self.recent_requests if r.timestamp >= cutoff_time]
            
            # Limit processing for very large datasets
            max_requests_to_process = 50000
            if len(recent) > max_requests_to_process:
                logger.warning(f"Limiting trending analysis to {max_requests_to_process} most recent requests")
                recent = recent[-max_requests_to_process:]
            
            # Group by minute
            minute_buckets = defaultdict(list)
            for request in recent:
                minute_key = request.timestamp.replace(second=0, microsecond=0)
                minute_buckets[minute_key].append(request)
            
            # Limit number of minute buckets to prevent memory exhaustion
            max_buckets = min(minutes, 1440)  # Never more than requested minutes or 24 hours
            if len(minute_buckets) > max_buckets:
                # Keep only the most recent buckets
                sorted_minutes = sorted(minute_buckets.keys())
                recent_minutes = sorted_minutes[-max_buckets:]
                minute_buckets = {k: minute_buckets[k] for k in recent_minutes}
            
            # Calculate stats per minute
            trending = []
            for minute, requests in sorted(minute_buckets.items()):
                durations = [r.duration_ms for r in requests]
                errors = [r for r in requests if r.status_code >= 400]
                
                trending.append({
                    'timestamp': minute.isoformat(),
                    'request_count': len(requests),
                    'error_count': len(errors),
                    'avg_duration_ms': sum(durations) / len(durations) if durations else 0.0,
                    'error_rate_percent': (len(errors) / len(requests)) * 100 if requests else 0.0
                })
            
            return trending


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for request metrics collection with improved error handling."""
    
    def __init__(self, app, metrics_collector: RequestMetricsCollector):
        super().__init__(app)
        self.metrics_collector = metrics_collector
        # Ensure queue processor is started
        asyncio.create_task(self.metrics_collector.start_queue_processor())
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics with improved error handling."""
        start_time = time.time()
        error_message = None
        response = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions properly
            error_message = f"HTTP {e.status_code}: {e.detail}"
            status_code = e.status_code
            # Re-raise HTTPException to let FastAPI handle it properly
            response = Response(
                content=str(e.detail),
                status_code=e.status_code,
                headers=getattr(e, 'headers', None)
            )
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            error_message = "Request cancelled"
            status_code = 499  # Client closed connection
            response = Response(
                content="Request cancelled",
                status_code=499
            )
        except Exception as e:
            # Handle unexpected exceptions
            error_message = f"Unexpected error: {type(e).__name__}: {str(e)}"
            status_code = 500
            logger.error(f"Unexpected error in request middleware: {e}", exc_info=True)
            response = Response(
                content="Internal Server Error",
                status_code=500
            )
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Record metrics using bounded queue (non-blocking)
        # The bounded queue handles backpressure automatically
        try:
            await self.metrics_collector.record_request(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
                error=error_message
            )
        except Exception as e:
            # Don't let metrics recording interfere with request processing
            logger.debug(f"Failed to record request metrics: {e}")
        
        return response


# Global metrics collector instance
global_metrics_collector = RequestMetricsCollector()


def get_metrics_collector() -> RequestMetricsCollector:
    """Get the global metrics collector instance."""
    return global_metrics_collector


def create_metrics_middleware() -> RequestMetricsMiddleware:
    """Create a metrics middleware instance."""
    return RequestMetricsMiddleware(None, global_metrics_collector)