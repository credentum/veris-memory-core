#!/usr/bin/env python3
"""
Monitoring and metrics for MCP tools.

Implements Prometheus metrics, distributed tracing, and health checks
for all MCP tool operations.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Try to import Prometheus client
try:
    from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Install with: pip install prometheus-client")

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.status import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning(
        "OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
    )


class MCPMetrics:
    """Prometheus metrics for MCP operations."""

    def __init__(self, registry=None):
        """Initialize MCP metrics.
        
        Args:
            registry: Optional Prometheus registry. If None, uses the global registry.
                     This allows for isolated registries in tests.
        """
        if not PROMETHEUS_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True
        self.registry = registry  # Store registry reference for metric creation

        # Request counters - use custom registry to avoid collisions in tests
        try:
            self.request_total = Counter(
                "mcp_requests_total", "Total number of MCP requests", ["endpoint", "status"],
                registry=self.registry
            )
        except ValueError:
            # Already registered, get existing instance
            from prometheus_client import REGISTRY
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name') and collector._name == 'mcp_requests_total':
                    self.request_total = collector
                    break

        try:
            self.request_duration = Histogram(
                "mcp_request_duration_seconds",
                "Time spent processing MCP requests",
                ["endpoint"],
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self.registry
            )
        except ValueError:
            # Already registered, get existing instance
            from prometheus_client import REGISTRY
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name') and collector._name == 'mcp_request_duration_seconds':
                    self.request_duration = collector
                    break

        # Storage operation metrics
        try:
            self.storage_operations = Counter(
                "mcp_storage_operations_total",
                "Total storage operations",
                ["backend", "operation", "status"],
                registry=self.registry
            )
        except ValueError:
            from prometheus_client import REGISTRY
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name') and collector._name == 'mcp_storage_operations_total':
                    self.storage_operations = collector
                    break

        try:
            self.storage_duration = Histogram(
                "mcp_storage_duration_seconds",
                "Time spent on storage operations",
                ["backend", "operation"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
                registry=self.registry
            )
        except ValueError:
            from prometheus_client import REGISTRY
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, '_name') and collector._name == 'mcp_storage_duration_seconds':
                    self.storage_duration = collector
                    break

        # Vector operations
        self.embedding_operations = Counter(
            "mcp_embedding_operations_total",
            "Total embedding operations",
            ["provider", "status"],
            registry=self.registry
        )

        self.embedding_duration = Histogram(
            "mcp_embedding_duration_seconds",
            "Time spent generating embeddings",
            ["provider"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )

        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            "mcp_rate_limit_hits_total",
            "Number of rate limit hits",
            ["endpoint", "client_type"],
            registry=self.registry
        )

        # Health metrics
        self.health_status = Gauge(
            "mcp_health_status", "Health status of MCP components", ["component"],
            registry=self.registry
        )

        # Context metrics
        self.contexts_stored = Counter(
            "mcp_contexts_stored_total", "Total contexts stored", ["type"],
            registry=self.registry
        )

        self.contexts_retrieved = Counter(
            "mcp_contexts_retrieved_total", "Total contexts retrieved", ["search_mode"],
            registry=self.registry
        )

        # Server info
        self.server_info = Info("mcp_server_info", "MCP server information", registry=self.registry)

        logger.info("✅ Prometheus metrics initialized")

    def record_request(self, endpoint: str, status: str, duration: float):
        """Record MCP request metrics."""
        if not self.enabled:
            return

        self.request_total.labels(endpoint=endpoint, status=status).inc()
        self.request_duration.labels(endpoint=endpoint).observe(duration)

    def record_storage_operation(self, backend: str, operation: str, status: str, duration: float):
        """Record storage operation metrics."""
        if not self.enabled:
            return

        self.storage_operations.labels(backend=backend, operation=operation, status=status).inc()
        self.storage_duration.labels(backend=backend, operation=operation).observe(duration)

    def record_embedding_operation(self, provider: str, status: str, duration: float):
        """Record embedding operation metrics."""
        if not self.enabled:
            return

        self.embedding_operations.labels(provider=provider, status=status).inc()
        self.embedding_duration.labels(provider=provider).observe(duration)

    def record_rate_limit_hit(self, endpoint: str, client_type: str = "unknown"):
        """Record rate limit hit."""
        if not self.enabled:
            return

        self.rate_limit_hits.labels(endpoint=endpoint, client_type=client_type).inc()

    def set_health_status(self, component: str, healthy: bool):
        """Set component health status."""
        if not self.enabled:
            return

        self.health_status.labels(component=component).set(1 if healthy else 0)

    def record_context_stored(self, context_type: str):
        """Record context storage."""
        if not self.enabled:
            return

        self.contexts_stored.labels(type=context_type).inc()

    def record_context_retrieved(self, search_mode: str, count: int):
        """Record context retrieval."""
        if not self.enabled:
            return

        self.contexts_retrieved.labels(search_mode=search_mode).inc(count)

    def set_server_info(self, version: str, features: str):
        """Set server information."""
        if not self.enabled:
            return

        self.server_info.info(
            {"version": version, "features": features, "prometheus_enabled": "true"}
        )

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.enabled:
            return "# Prometheus not available\n"

        return generate_latest()


class MCPTracing:
    """OpenTelemetry tracing for MCP operations."""

    def __init__(self, service_name: str = "mcp-context-store"):
        """Initialize tracing."""
        if not OPENTELEMETRY_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True
        self.service_name = service_name
        self._span_processor = None
        self._jaeger_exporter = None

        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)

        # Set up Jaeger exporter if configured
        jaeger_endpoint = os.environ.get("JAEGER_ENDPOINT")
        if jaeger_endpoint:
            self._jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split("//")[1].split(":")[0],
                agent_port=int(jaeger_endpoint.split(":")[-1]),
            )
            self._span_processor = BatchSpanProcessor(self._jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(self._span_processor)
            logger.info(f"✅ Jaeger tracing initialized: {jaeger_endpoint}")
        else:
            logger.info("✅ OpenTelemetry tracing initialized (no exporter)")

    def cleanup(self):
        """Clean up tracing resources."""
        if self._span_processor:
            try:
                self._span_processor.shutdown()
                logger.info("Span processor shutdown complete")
            except Exception as e:
                logger.warning(f"Error shutting down span processor: {e}")

        if self._jaeger_exporter:
            try:
                # Jaeger exporter cleanup if it has a shutdown method
                if hasattr(self._jaeger_exporter, "shutdown"):
                    self._jaeger_exporter.shutdown()
                logger.info("Jaeger exporter cleanup complete")
            except Exception as e:
                logger.warning(f"Error cleaning up Jaeger exporter: {e}")

    @asynccontextmanager
    async def trace_operation(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Trace an async operation."""
        if not self.enabled:
            yield None
            return

        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))

            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise


class MCPMonitor:
    """Comprehensive monitoring for MCP operations."""

    def __init__(self):
        """Initialize MCP monitor."""
        self.metrics = MCPMetrics()
        self.tracing = MCPTracing()
        self.start_time = time.time()

        # Initialize server info
        if self.metrics.enabled:
            self.metrics.set_server_info(
                version="1.0.0",
                features="store_context,retrieve_context,query_graph,rate_limiting,ssl",
            )

    def monitor_request(self, endpoint: str):
        """Decorator to monitor MCP requests."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"

                async with self.tracing.trace_operation(
                    f"mcp.{endpoint}", {"endpoint": endpoint, "function": func.__name__}
                ) as span:
                    try:
                        result = await func(*args, **kwargs)

                        # Check if result indicates failure
                        if isinstance(result, dict) and not result.get("success", True):
                            status = "error"
                            if span:
                                span.set_attribute(
                                    "error.type", result.get("error_type", "unknown")
                                )
                                span.set_attribute("error.message", result.get("message", ""))

                        # Record context-specific metrics
                        if (
                            endpoint == "store_context"
                            and isinstance(result, dict)
                            and result.get("success")
                        ):
                            context_type = args[0].get("type", "unknown") if args else "unknown"
                            self.metrics.record_context_stored(context_type)

                        elif (
                            endpoint == "retrieve_context"
                            and isinstance(result, dict)
                            and result.get("success")
                        ):
                            search_mode = (
                                args[0].get("search_mode", "unknown") if args else "unknown"
                            )
                            result_count = len(result.get("results", []))
                            self.metrics.record_context_retrieved(search_mode, result_count)

                        return result

                    except Exception as e:
                        status = "error"
                        logger.error(f"Error in {endpoint}: {e}")
                        raise

                    finally:
                        duration = time.time() - start_time
                        self.metrics.record_request(endpoint, status, duration)

            return wrapper

        return decorator

    def monitor_storage_operation(self, backend: str, operation: str):
        """Decorator to monitor storage operations."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"

                async with self.tracing.trace_operation(
                    f"storage.{backend}.{operation}",
                    {"backend": backend, "operation": operation},
                ):
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception:
                        status = "error"
                        raise
                    finally:
                        duration = time.time() - start_time
                        self.metrics.record_storage_operation(backend, operation, status, duration)

            return wrapper

        return decorator

    def monitor_embedding_operation(self, provider: str):
        """Decorator to monitor embedding operations."""

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                status = "success"

                async with self.tracing.trace_operation(
                    f"embedding.{provider}", {"provider": provider}
                ):
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception:
                        status = "error"
                        raise
                    finally:
                        duration = time.time() - start_time
                        self.metrics.record_embedding_operation(provider, status, duration)

            return wrapper

        return decorator

    def record_rate_limit_hit(self, endpoint: str, client_info: Dict = None):
        """Record rate limit hit with client classification."""
        client_type = "unknown"
        if client_info:
            if "user_agent" in client_info:
                if "curl" in client_info["user_agent"].lower():
                    client_type = "cli"
                elif "browser" in client_info["user_agent"].lower():
                    client_type = "web"
                else:
                    client_type = "api"

        self.metrics.record_rate_limit_hit(endpoint, client_type)

    def update_health_status(self, component_statuses: Dict[str, bool]):
        """Update health status for all components."""
        for component, healthy in component_statuses.items():
            self.metrics.set_health_status(component, healthy)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        uptime = time.time() - self.start_time

        return {
            "uptime_seconds": uptime,
            "monitoring": {
                "prometheus_enabled": self.metrics.enabled,
                "tracing_enabled": self.tracing.enabled,
            },
            "features": [
                "store_context",
                "retrieve_context",
                "query_graph",
                "rate_limiting",
                "ssl_support",
                "embedding_generation",
            ],
        }

    def get_metrics_endpoint(self) -> str:
        """Get Prometheus metrics for /metrics endpoint."""
        return self.metrics.get_metrics()

    def cleanup(self):
        """Clean up monitoring resources."""
        try:
            if self.tracing.enabled:
                self.tracing.cleanup()
            logger.info("Monitoring cleanup completed")
        except Exception as e:
            logger.warning(f"Error during monitoring cleanup: {e}")


# Global monitor instance
_monitor = None


def get_monitor() -> MCPMonitor:
    """Get global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = MCPMonitor()
    return _monitor


def monitor_mcp_request(endpoint: str):
    """Convenience decorator for monitoring MCP requests."""
    return get_monitor().monitor_request(endpoint)


def monitor_storage(backend: str, operation: str):
    """Convenience decorator for monitoring storage operations."""
    return get_monitor().monitor_storage_operation(backend, operation)


def monitor_embedding(provider: str):
    """Convenience decorator for monitoring embedding operations."""
    return get_monitor().monitor_embedding_operation(provider)
