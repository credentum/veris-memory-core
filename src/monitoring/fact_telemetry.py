"""
Comprehensive telemetry and observability for fact retrieval pipeline.

This module provides detailed tracing, metrics, and logging for all fact-related
operations to enable monitoring, debugging, and optimization.
"""

import time
import json
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import uuid

# Try OpenTelemetry imports with fallback
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

logger = logging.getLogger(__name__)


class FactOperation(Enum):
    """Types of fact operations for telemetry."""
    STORE_FACT = "store_fact"
    RETRIEVE_FACT = "retrieve_fact"
    INTENT_CLASSIFICATION = "intent_classification"
    ENTITY_EXTRACTION = "entity_extraction"
    QUERY_EXPANSION = "query_expansion"
    HYBRID_SCORING = "hybrid_scoring"
    GRAPH_ENHANCEMENT = "graph_enhancement"
    FACT_RANKING = "fact_ranking"
    DELETE_FACTS = "delete_facts"


class TelemetryLevel(Enum):
    """Telemetry detail levels."""
    BASIC = "basic"          # Only success/failure and latency
    DETAILED = "detailed"    # Include content metadata
    DEBUG = "debug"          # Full request/response data
    OFF = "off"             # No telemetry


@dataclass
class TraceContext:
    """Context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: FactOperation
    user_id: str
    namespace: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class FactMetric:
    """Individual fact operation metric."""
    operation: FactOperation
    timestamp: float
    duration_ms: float
    success: bool
    user_id: str
    namespace: str
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class RoutingDecision:
    """Routing decision for fact queries."""
    query: str
    intent_detected: str
    route_taken: str
    confidence: float
    alternatives_considered: List[str]
    reasoning: str
    timestamp: float


@dataclass
class RankingExplanation:
    """Explanation of ranking decisions."""
    query: str
    total_results: int
    scoring_mode: str
    component_weights: Dict[str, float]
    top_result_scores: List[Dict[str, Any]]
    ranking_factors: List[str]
    timestamp: float


class RingBuffer:
    """Thread-safe ring buffer for failure tracking."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
    
    def add(self, item: Any) -> None:
        """Add item to ring buffer."""
        with self.lock:
            self.buffer.append(item)
    
    def get_recent(self, count: int = 100) -> List[Any]:
        """Get recent items from buffer."""
        with self.lock:
            return list(self.buffer)[-count:]
    
    def get_all(self) -> List[Any]:
        """Get all items from buffer."""
        with self.lock:
            return list(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()


class FactTelemetry:
    """
    Comprehensive telemetry system for fact operations.
    
    Provides tracing, metrics collection, failure tracking, and performance
    monitoring with configurable detail levels and privacy controls.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.telemetry_level = TelemetryLevel(
            self.config.get('telemetry_level', 'detailed')
        )
        
        # OpenTelemetry setup
        self.tracer = None
        self.meter = None
        if OTEL_AVAILABLE and self.telemetry_level != TelemetryLevel.OFF:
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            self._setup_metrics()
        
        # Internal metrics storage
        self.metrics = []
        self.routing_decisions = []
        self.ranking_explanations = []
        self.failure_buffer = RingBuffer(capacity=1000)
        
        # Aggregated counters
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.latency_stats = defaultdict(list)
        
        # Privacy controls
        self.pii_scrubbing = self.config.get('scrub_pii', True)
        self.content_redaction = self.config.get('redact_content', False)
        self.retention_hours = self.config.get('retention_hours', 24)
        
        # Performance limits
        self.max_metrics_stored = self.config.get('max_metrics_stored', 10000)
        self.metrics_lock = threading.Lock()
        
        logger.info(f"FactTelemetry initialized with level: {self.telemetry_level.value}")
    
    def _setup_metrics(self) -> None:
        """Setup OpenTelemetry metrics."""
        if not self.meter:
            return
        
        # Operation counters
        self.operation_counter = self.meter.create_counter(
            name="fact_operations_total",
            description="Total number of fact operations",
            unit="1"
        )
        
        # Latency histogram
        self.latency_histogram = self.meter.create_histogram(
            name="fact_operation_duration_ms",
            description="Duration of fact operations in milliseconds",
            unit="ms"
        )
        
        # Error counter
        self.error_counter = self.meter.create_counter(
            name="fact_operation_errors_total",
            description="Total number of fact operation errors",
            unit="1"
        )
        
        # Active operations gauge
        self.active_operations = self.meter.create_up_down_counter(
            name="fact_operations_active",
            description="Number of currently active fact operations",
            unit="1"
        )
    
    def trace_request(self, operation: FactOperation, user_id: str, namespace: str,
                     metadata: Optional[Dict[str, Any]] = None) -> TraceContext:
        """Start tracing a fact operation request."""
        if self.telemetry_level == TelemetryLevel.OFF:
            return TraceContext(
                trace_id="", span_id="", parent_span_id=None,
                operation=operation, user_id=user_id, namespace=namespace,
                timestamp=time.time(), metadata={}
            )
        
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation=operation,
            user_id=self._scrub_user_id(user_id) if self.pii_scrubbing else user_id,
            namespace=namespace,
            timestamp=time.time(),
            metadata=self._scrub_metadata(metadata or {})
        )
        
        # OpenTelemetry span
        if self.tracer:
            span = self.tracer.start_span(operation.value)
            span.set_attribute("fact.operation", operation.value)
            span.set_attribute("fact.user_id", context.user_id)
            span.set_attribute("fact.namespace", namespace)
            
            # Add metadata as attributes
            for key, value in context.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"fact.{key}", value)
        
        # Update active operations counter
        if self.meter:
            self.active_operations.add(1, {
                "operation": operation.value,
                "namespace": namespace
            })
        
        logger.debug(f"Started trace for {operation.value}: {trace_id}")
        return context
    
    def record_success(self, context: TraceContext, duration_ms: float,
                      result_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record successful completion of operation."""
        if self.telemetry_level == TelemetryLevel.OFF:
            return
        
        # Create metric
        metric = FactMetric(
            operation=context.operation,
            timestamp=context.timestamp,
            duration_ms=duration_ms,
            success=True,
            user_id=context.user_id,
            namespace=context.namespace,
            metadata={
                **context.metadata,
                **(self._scrub_metadata(result_metadata or {}))
            }
        )
        
        self._store_metric(metric)
        
        # OpenTelemetry recording
        if self.tracer:
            # Update span
            span = trace.get_current_span()
            if span:
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("fact.duration_ms", duration_ms)
                span.set_attribute("fact.success", True)
                span.end()
        
        # Update counters
        if self.meter:
            labels = {
                "operation": context.operation.value,
                "namespace": context.namespace,
                "status": "success"
            }
            self.operation_counter.add(1, labels)
            self.latency_histogram.record(duration_ms, labels)
            self.active_operations.add(-1, {
                "operation": context.operation.value,
                "namespace": context.namespace
            })
        
        # Update internal stats
        self.operation_counts[context.operation.value] += 1
        self.latency_stats[context.operation.value].append(duration_ms)
        
        logger.debug(f"Recorded success for {context.operation.value}: {duration_ms:.2f}ms")
    
    def record_failure(self, context: TraceContext, duration_ms: float,
                      error: Exception, error_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record failed completion of operation."""
        if self.telemetry_level == TelemetryLevel.OFF:
            return
        
        error_message = str(error) if not self.pii_scrubbing else self._scrub_error_message(str(error))
        
        # Create metric
        metric = FactMetric(
            operation=context.operation,
            timestamp=context.timestamp,
            duration_ms=duration_ms,
            success=False,
            user_id=context.user_id,
            namespace=context.namespace,
            metadata={
                **context.metadata,
                **(self._scrub_metadata(error_metadata or {}))
            },
            error_message=error_message
        )
        
        self._store_metric(metric)
        self.failure_buffer.add(metric)
        
        # OpenTelemetry recording
        if self.tracer:
            span = trace.get_current_span()
            if span:
                span.set_status(Status(StatusCode.ERROR, error_message))
                span.set_attribute("fact.duration_ms", duration_ms)
                span.set_attribute("fact.success", False)
                span.set_attribute("fact.error", error_message)
                span.end()
        
        # Update counters
        if self.meter:
            labels = {
                "operation": context.operation.value,
                "namespace": context.namespace,
                "status": "error",
                "error_type": type(error).__name__
            }
            self.operation_counter.add(1, labels)
            self.error_counter.add(1, labels)
            self.latency_histogram.record(duration_ms, labels)
            self.active_operations.add(-1, {
                "operation": context.operation.value,
                "namespace": context.namespace
            })
        
        # Update internal stats
        self.error_counts[f"{context.operation.value}:{type(error).__name__}"] += 1
        
        logger.warning(f"Recorded failure for {context.operation.value}: {error_message}")
    
    def log_routing_decision(self, query: str, intent: str, route: str,
                           confidence: float, alternatives: List[str],
                           reasoning: str) -> None:
        """Log routing decision for fact queries."""
        if self.telemetry_level in [TelemetryLevel.OFF, TelemetryLevel.BASIC]:
            return
        
        decision = RoutingDecision(
            query=self._scrub_query(query) if self.content_redaction else query,
            intent_detected=intent,
            route_taken=route,
            confidence=confidence,
            alternatives_considered=alternatives,
            reasoning=reasoning,
            timestamp=time.time()
        )
        
        with self.metrics_lock:
            self.routing_decisions.append(decision)
            # Maintain size limit
            if len(self.routing_decisions) > self.max_metrics_stored:
                self.routing_decisions = self.routing_decisions[-self.max_metrics_stored//2:]
        
        logger.info(f"Routing decision: {intent} -> {route} (confidence: {confidence:.3f})")
    
    def record_ranking_explanation(self, query: str, total_results: int,
                                 scoring_mode: str, component_weights: Dict[str, float],
                                 top_scores: List[Dict[str, Any]],
                                 ranking_factors: List[str]) -> None:
        """Record explanation of ranking decisions."""
        if self.telemetry_level in [TelemetryLevel.OFF, TelemetryLevel.BASIC]:
            return
        
        explanation = RankingExplanation(
            query=self._scrub_query(query) if self.content_redaction else query,
            total_results=total_results,
            scoring_mode=scoring_mode,
            component_weights=component_weights,
            top_result_scores=top_scores,
            ranking_factors=ranking_factors,
            timestamp=time.time()
        )
        
        with self.metrics_lock:
            self.ranking_explanations.append(explanation)
            # Maintain size limit
            if len(self.ranking_explanations) > self.max_metrics_stored:
                self.ranking_explanations = self.ranking_explanations[-self.max_metrics_stored//2:]
        
        logger.debug(f"Ranking explanation recorded: {scoring_mode} mode, {total_results} results")
    
    def get_operation_metrics(self, operation: Optional[FactOperation] = None,
                            hours: int = 1) -> List[FactMetric]:
        """Get operation metrics for analysis."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.metrics_lock:
            filtered_metrics = [
                m for m in self.metrics
                if m.timestamp >= cutoff_time and
                (operation is None or m.operation == operation)
            ]
        
        return filtered_metrics
    
    def get_failure_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of recent failures."""
        cutoff_time = time.time() - (hours * 3600)
        recent_failures = [
            f for f in self.failure_buffer.get_all()
            if f.timestamp >= cutoff_time
        ]
        
        # Group by error type
        error_groups = defaultdict(list)
        for failure in recent_failures:
            error_type = failure.error_message.split(':')[0] if failure.error_message else "Unknown"
            error_groups[error_type].append(failure)
        
        return {
            "total_failures": len(recent_failures),
            "failure_rate": len(recent_failures) / max(len(self.metrics), 1),
            "error_groups": {
                error_type: {
                    "count": len(failures),
                    "operations": list(set(f.operation.value for f in failures)),
                    "latest_error": failures[-1].error_message if failures else None
                }
                for error_type, failures in error_groups.items()
            },
            "time_range_hours": hours
        }
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for operations."""
        metrics = self.get_operation_metrics(hours=hours)
        
        if not metrics:
            return {"message": "No metrics available"}
        
        # Calculate per-operation stats
        operation_stats = defaultdict(lambda: {"latencies": [], "successes": 0, "failures": 0})
        
        for metric in metrics:
            op_stats = operation_stats[metric.operation.value]
            op_stats["latencies"].append(metric.duration_ms)
            if metric.success:
                op_stats["successes"] += 1
            else:
                op_stats["failures"] += 1
        
        # Generate summary
        summary = {}
        for operation, stats in operation_stats.items():
            latencies = stats["latencies"]
            if latencies:
                summary[operation] = {
                    "total_requests": len(latencies),
                    "success_rate": stats["successes"] / len(latencies),
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 20 else max(latencies),
                    "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 100 else max(latencies)
                }
        
        return {
            "time_range_hours": hours,
            "total_operations": len(metrics),
            "overall_success_rate": sum(1 for m in metrics if m.success) / len(metrics),
            "operation_stats": summary
        }
    
    def export_metrics(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export metrics in specified format."""
        data = {
            "telemetry_config": {
                "level": self.telemetry_level.value,
                "pii_scrubbing": self.pii_scrubbing,
                "content_redaction": self.content_redaction,
                "retention_hours": self.retention_hours
            },
            "summary": self.get_performance_summary(hours=self.retention_hours),
            "failures": self.get_failure_summary(hours=self.retention_hours),
            "recent_routing_decisions": self.routing_decisions[-100:] if self.routing_decisions else [],
            "recent_ranking_explanations": self.ranking_explanations[-50:] if self.ranking_explanations else []
        }
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            return data
    
    def _store_metric(self, metric: FactMetric) -> None:
        """Store metric with size management."""
        with self.metrics_lock:
            self.metrics.append(metric)
            # Maintain size limit
            if len(self.metrics) > self.max_metrics_stored:
                self.metrics = self.metrics[-self.max_metrics_stored//2:]
    
    def _scrub_user_id(self, user_id: str) -> str:
        """Scrub PII from user ID."""
        if not self.pii_scrubbing:
            return user_id
        
        # Simple hash-based scrubbing
        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:8]
    
    def _scrub_query(self, query: str) -> str:
        """Scrub sensitive content from queries."""
        if not self.content_redaction:
            return query
        
        # Replace email patterns
        import re
        query = re.sub(r'\b[\w._%+-]+@[\w.-]+\.\w+\b', '[EMAIL]', query)
        # Replace phone patterns
        query = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', query)
        # Replace potential names (simple heuristic)
        query = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', query)
        
        return query
    
    def _scrub_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Scrub sensitive data from metadata."""
        if not self.pii_scrubbing and not self.content_redaction:
            return metadata
        
        scrubbed = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                if self.content_redaction:
                    value = self._scrub_query(value)
                scrubbed[key] = value
            elif isinstance(value, (int, float, bool)):
                scrubbed[key] = value
            else:
                scrubbed[key] = str(value)[:100]  # Truncate complex objects
        
        return scrubbed
    
    def _scrub_error_message(self, error_msg: str) -> str:
        """Scrub PII from error messages."""
        if not self.pii_scrubbing:
            return error_msg
        
        # Remove potential file paths
        import re
        error_msg = re.sub(r'/[^\s]+', '[PATH]', error_msg)
        # Remove potential user data
        error_msg = self._scrub_query(error_msg)
        
        return error_msg
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics (for testing/reset)."""
        with self.metrics_lock:
            self.metrics.clear()
            self.routing_decisions.clear()
            self.ranking_explanations.clear()
            self.failure_buffer.clear()
            self.operation_counts.clear()
            self.error_counts.clear()
            self.latency_stats.clear()
        
        logger.info("All telemetry metrics cleared")


# Global telemetry instance
_telemetry_instance: Optional[FactTelemetry] = None


def get_telemetry() -> FactTelemetry:
    """Get global telemetry instance."""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = FactTelemetry()
    return _telemetry_instance


def initialize_telemetry(config: Dict[str, Any]) -> FactTelemetry:
    """Initialize global telemetry with configuration."""
    global _telemetry_instance
    _telemetry_instance = FactTelemetry(config)
    return _telemetry_instance


# Context manager for automatic telemetry
class telemetry_context:
    """Context manager for automatic telemetry recording."""
    
    def __init__(self, operation: FactOperation, user_id: str, namespace: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 telemetry_instance: Optional[FactTelemetry] = None):
        self.telemetry = telemetry_instance or get_telemetry()
        self.operation = operation
        self.user_id = user_id
        self.namespace = namespace
        self.metadata = metadata
        self.context = None
        self.start_time = None
    
    def __enter__(self) -> TraceContext:
        self.start_time = time.time()
        self.context = self.telemetry.trace_request(
            self.operation, self.user_id, self.namespace, self.metadata
        )
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time and self.context:
            duration_ms = (time.time() - self.start_time) * 1000
            
            if exc_type is None:
                self.telemetry.record_success(self.context, duration_ms)
            else:
                self.telemetry.record_failure(self.context, duration_ms, exc_val)
        
        return False  # Don't suppress exceptions