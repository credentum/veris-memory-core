#!/usr/bin/env python3
"""
Comprehensive tests for core monitoring module to achieve 50% coverage.
"""
# Conditional OpenTelemetry import
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace.status import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.monitoring import (
    MCPMetrics,
    MCPMonitor,
    MCPTracing,
    get_monitor,
    monitor_embedding,
    monitor_mcp_request,
    monitor_storage,
)


class TestMCPMetrics:
    """Test MCPMetrics Prometheus metrics collection."""

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", True)
    @patch("core.monitoring.Counter")
    @patch("core.monitoring.Histogram")
    @patch("core.monitoring.Gauge")
    @patch("core.monitoring.Info")
    def test_mcp_metrics_init_with_prometheus(
        self, mock_info, mock_gauge, mock_histogram, mock_counter
    ):
        """Test MCPMetrics initialization with Prometheus available."""
        # Mock prometheus metrics
        mock_counter_instance = Mock()
        mock_histogram_instance = Mock()
        mock_gauge_instance = Mock()
        mock_info_instance = Mock()

        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = mock_gauge_instance
        mock_info.return_value = mock_info_instance

        metrics = MCPMetrics()

        assert metrics.enabled is True
        assert metrics.request_total == mock_counter_instance
        assert metrics.request_duration == mock_histogram_instance
        assert metrics.storage_operations == mock_counter_instance

        # Verify metrics were created with correct parameters
        assert mock_counter.call_count >= 3  # Multiple counter metrics
        assert mock_histogram.call_count >= 3  # Multiple histogram metrics

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", False)
    def test_mcp_metrics_init_without_prometheus(self):
        """Test MCPMetrics initialization without Prometheus."""
        metrics = MCPMetrics()

        assert metrics.enabled is False

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", True)
    @patch("core.monitoring.Counter")
    @patch("core.monitoring.Histogram")
    @patch("core.monitoring.Gauge")
    @patch("core.monitoring.Info")
    def test_mcp_metrics_record_request(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test recording MCP request metrics."""
        # Setup mocks
        mock_counter_instance = Mock()
        mock_histogram_instance = Mock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = Mock()
        mock_info.return_value = Mock()

        metrics = MCPMetrics()

        # Test request recording
        metrics.record_request("store_context", "success", 0.5)

        mock_counter_instance.labels.assert_called_with(endpoint="store_context", status="success")
        mock_counter_instance.labels.return_value.inc.assert_called_once()

        mock_histogram_instance.labels.assert_called_with(endpoint="store_context")
        mock_histogram_instance.labels.return_value.observe.assert_called_with(0.5)

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", True)
    @patch("core.monitoring.Counter")
    @patch("core.monitoring.Histogram")
    @patch("core.monitoring.Gauge")
    @patch("core.monitoring.Info")
    def test_mcp_metrics_record_storage_operation(
        self, mock_info, mock_gauge, mock_histogram, mock_counter
    ):
        """Test recording storage operation metrics."""
        # Setup mocks
        mock_counter_instance = Mock()
        mock_histogram_instance = Mock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = Mock()
        mock_info.return_value = Mock()

        metrics = MCPMetrics()

        # Test storage operation recording
        metrics.record_storage_operation("qdrant", "upsert", "success", 0.1)

        mock_counter_instance.labels.assert_called_with(
            backend="qdrant", operation="upsert", status="success"
        )
        mock_counter_instance.labels.return_value.inc.assert_called()

        mock_histogram_instance.labels.assert_called_with(backend="qdrant", operation="upsert")
        mock_histogram_instance.labels.return_value.observe.assert_called_with(0.1)

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", True)
    @patch("core.monitoring.Counter")
    @patch("core.monitoring.Histogram")
    @patch("core.monitoring.Gauge")
    @patch("core.monitoring.Info")
    def test_mcp_metrics_record_embedding_operation(
        self, mock_info, mock_gauge, mock_histogram, mock_counter
    ):
        """Test recording embedding operation metrics."""
        # Setup mocks
        mock_counter_instance = Mock()
        mock_histogram_instance = Mock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = Mock()
        mock_info.return_value = Mock()

        metrics = MCPMetrics()

        # Test embedding recording
        metrics.record_embedding_operation("openai", "success", 1.2)

        mock_counter_instance.labels.assert_called_with(provider="openai", status="success")
        mock_counter_instance.labels.return_value.inc.assert_called()

        mock_histogram_instance.labels.assert_called_with(provider="openai")
        mock_histogram_instance.labels.return_value.observe.assert_called_with(1.2)

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", True)
    @patch("core.monitoring.Counter")
    @patch("core.monitoring.Histogram")
    @patch("core.monitoring.Gauge")
    @patch("core.monitoring.Info")
    def test_mcp_metrics_set_server_info(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test setting server info metrics."""
        # Setup mocks
        mock_info_instance = Mock()
        mock_info.return_value = mock_info_instance
        mock_counter.return_value = Mock()
        mock_histogram.return_value = Mock()
        mock_gauge.return_value = Mock()

        metrics = MCPMetrics()

        # Test server info setting
        metrics.set_server_info(version="1.0.0", features="store,retrieve")

        mock_info_instance.info.assert_called_with(
            {"version": "1.0.0", "features": "store,retrieve", "prometheus_enabled": "true"}
        )

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", False)
    def test_mcp_metrics_disabled_operations(self):
        """Test metric operations when Prometheus is disabled."""
        metrics = MCPMetrics()

        # Should not raise errors when disabled
        metrics.record_request("test", "success", 1.0)
        metrics.record_storage_operation("test", "test", "success", 1.0)
        metrics.record_embedding_operation("test", "success", 1.0)
        metrics.set_server_info(version="1.0.0", features="test")

        # Should pass without errors since metrics are disabled


class TestMCPTracing:
    """Test MCPTracing OpenTelemetry tracing functionality."""

    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", True)
    @patch("opentelemetry.trace")
    @patch("opentelemetry.sdk.trace.TracerProvider")
    def test_mcp_tracing_init_with_opentelemetry(self, mock_tracer_provider, mock_trace):
        """Test MCPTracing initialization with OpenTelemetry available."""
        mock_tracer_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_tracer_provider_instance
        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        tracing = MCPTracing("test-service")

        assert tracing.enabled is True
        assert tracing.service_name == "test-service"
        assert tracing.tracer == mock_tracer

        mock_trace.set_tracer_provider.assert_called_once_with(mock_tracer_provider_instance)

    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False)
    def test_mcp_tracing_init_without_opentelemetry(self):
        """Test MCPTracing initialization without OpenTelemetry."""
        tracing = MCPTracing()

        assert tracing.enabled is False

    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", True)
    @patch("core.monitoring.trace")
    @patch("core.monitoring.TracerProvider")
    @patch("core.monitoring.JaegerExporter")
    @patch("core.monitoring.BatchSpanProcessor")
    def test_mcp_tracing_init_with_jaeger(
        self, mock_span_processor, mock_jaeger_exporter, mock_tracer_provider, mock_trace
    ):
        """Test MCPTracing initialization with Jaeger configured."""
        mock_tracer_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_tracer_provider_instance
        mock_tracer_provider.return_value.add_span_processor = Mock()
        mock_trace.get_tracer_provider.return_value = mock_tracer_provider_instance

        mock_jaeger_instance = Mock()
        mock_jaeger_exporter.return_value = mock_jaeger_instance
        mock_processor_instance = Mock()
        mock_span_processor.return_value = mock_processor_instance

        with patch.dict(os.environ, {"JAEGER_ENDPOINT": "http://jaeger:14268"}):
            tracing = MCPTracing()

        assert tracing.enabled is True
        mock_jaeger_exporter.assert_called_once()
        mock_span_processor.assert_called_once_with(mock_jaeger_instance)
        mock_tracer_provider_instance.add_span_processor.assert_called_once()

    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", True)
    @patch("core.monitoring.trace")
    @patch("core.monitoring.TracerProvider")
    def test_mcp_tracing_cleanup(self, mock_tracer_provider, mock_trace):
        """Test MCPTracing cleanup."""
        mock_span_processor = Mock()
        mock_jaeger_exporter = Mock()
        mock_jaeger_exporter.shutdown = Mock()

        tracing = MCPTracing()
        tracing._span_processor = mock_span_processor
        tracing._jaeger_exporter = mock_jaeger_exporter

        tracing.cleanup()

        mock_span_processor.shutdown.assert_called_once()
        mock_jaeger_exporter.shutdown.assert_called_once()

    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", True)
    @patch("core.monitoring.trace")
    @patch("core.monitoring.TracerProvider")
    def test_mcp_tracing_cleanup_with_errors(self, mock_tracer_provider, mock_trace):
        """Test MCPTracing cleanup with errors."""
        mock_span_processor = Mock()
        mock_span_processor.shutdown.side_effect = Exception("Shutdown error")

        tracing = MCPTracing()
        tracing._span_processor = mock_span_processor

        # Should not raise exception
        tracing.cleanup()

    @pytest.mark.asyncio
    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", True)
    @patch("core.monitoring.trace")
    @patch("core.monitoring.TracerProvider")
    async def test_mcp_tracing_trace_operation_context_manager(
        self, mock_tracer_provider, mock_trace
    ):
        """Test trace_operation context manager."""
        mock_tracer = Mock()
        mock_span = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        mock_tracer.start_span.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer

        tracing = MCPTracing()

        async with tracing.trace_operation("test_operation", {"key": "value"}) as span:
            assert span == mock_span

        mock_tracer.start_span.assert_called_once_with("test_operation")
        mock_span.set_attribute.assert_called_with("key", "value")

    @pytest.mark.asyncio
    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False)
    async def test_mcp_tracing_trace_operation_disabled(self):
        """Test trace_operation when tracing is disabled."""
        tracing = MCPTracing()

        async with tracing.trace_operation("test_operation") as span:
            assert span is None


class TestMCPMonitor:
    """Test MCPMonitor main monitoring coordinator."""

    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    def test_mcp_monitor_init(self, mock_tracing_class, mock_metrics_class):
        """Test MCPMonitor initialization."""
        mock_metrics = Mock()
        mock_metrics.enabled = True
        mock_metrics_class.return_value = mock_metrics
        mock_tracing = Mock()
        mock_tracing_class.return_value = mock_tracing

        with patch("core.monitoring.time.time", return_value=1234567890.0):
            monitor = MCPMonitor()

        assert monitor.metrics == mock_metrics
        assert monitor.tracing == mock_tracing
        assert monitor.start_time == 1234567890.0

        # Should set server info when metrics enabled
        mock_metrics.set_server_info.assert_called_once()

    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    def test_mcp_monitor_init_metrics_disabled(self, mock_tracing_class, mock_metrics_class):
        """Test MCPMonitor initialization with metrics disabled."""
        mock_metrics = Mock()
        mock_metrics.enabled = False
        mock_metrics_class.return_value = mock_metrics
        mock_tracing = Mock()
        mock_tracing_class.return_value = mock_tracing

        monitor = MCPMonitor()

        # Should not set server info when metrics disabled
        mock_metrics.set_server_info.assert_not_called()

    @pytest.mark.asyncio
    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    async def test_mcp_monitor_monitor_request_success(
        self, mock_tracing_class, mock_metrics_class
    ):
        """Test monitor_request decorator with successful request."""
        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics
        mock_tracing = Mock()
        mock_tracing.trace_operation = AsyncMock()
        mock_tracing_class.return_value = mock_tracing

        # Mock the async context manager
        mock_span = Mock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_span)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_tracing.trace_operation.return_value = mock_context_manager

        monitor = MCPMonitor()

        # Create a test function to decorate
        @monitor.monitor_request("test_endpoint")
        async def test_func(arguments):
            return {"success": True, "data": "test"}

        with patch("core.monitoring.time.time", side_effect=[1000.0, 1001.5]):
            result = await test_func({"test": "args"})

        assert result == {"success": True, "data": "test"}
        mock_metrics.record_request.assert_called_once_with("test_endpoint", "success", 1.5)

    @pytest.mark.asyncio
    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    async def test_mcp_monitor_monitor_request_failure(
        self, mock_tracing_class, mock_metrics_class
    ):
        """Test monitor_request decorator with failed request."""
        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics
        mock_tracing = Mock()
        mock_tracing.trace_operation = AsyncMock()
        mock_tracing_class.return_value = mock_tracing

        # Mock the async context manager
        mock_span = Mock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_span)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_tracing.trace_operation.return_value = mock_context_manager

        monitor = MCPMonitor()

        # Create a test function that returns failure
        @monitor.monitor_request("test_endpoint")
        async def test_func(arguments):
            return {"success": False, "error_type": "validation", "message": "Invalid input"}

        with patch("core.monitoring.time.time", side_effect=[1000.0, 1001.2]):
            result = await test_func({"test": "args"})

        assert result["success"] is False
        mock_metrics.record_request.assert_called_once_with("test_endpoint", "error", 1.2)
        mock_span.set_attribute.assert_any_call("error.type", "validation")
        mock_span.set_attribute.assert_any_call("error.message", "Invalid input")

    @pytest.mark.asyncio
    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    async def test_mcp_monitor_monitor_request_exception(
        self, mock_tracing_class, mock_metrics_class
    ):
        """Test monitor_request decorator with exception."""
        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics
        mock_tracing = Mock()
        mock_tracing.trace_operation = AsyncMock()
        mock_tracing_class.return_value = mock_tracing

        # Mock the async context manager
        mock_span = Mock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_span)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_tracing.trace_operation.return_value = mock_context_manager

        monitor = MCPMonitor()

        # Create a test function that raises exception
        @monitor.monitor_request("test_endpoint")
        async def test_func(arguments):
            raise ValueError("Test error")

        with patch("core.monitoring.time.time", side_effect=[1000.0, 1001.0]):
            with pytest.raises(ValueError, match="Test error"):
                await test_func({"test": "args"})

        mock_metrics.record_request.assert_called_once_with("test_endpoint", "error", 1.0)

    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    def test_mcp_monitor_record_context_stored(self, mock_tracing_class, mock_metrics_class):
        """Test recording context stored metrics."""
        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = Mock()

        monitor = MCPMonitor()
        monitor.record_context_stored("design")

        mock_metrics.record_context_stored.assert_called_once_with("design")

    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    def test_mcp_monitor_get_uptime(self, mock_tracing_class, mock_metrics_class):
        """Test getting monitor uptime."""
        mock_metrics_class.return_value = Mock()
        mock_tracing_class.return_value = Mock()

        with patch("core.monitoring.time.time", return_value=1000.0):
            monitor = MCPMonitor()

        with patch("core.monitoring.time.time", return_value=1030.5):
            uptime = monitor.get_uptime()

        assert uptime == 30.5

    @patch("core.monitoring.MCPMetrics")
    @patch("core.monitoring.MCPTracing")
    def test_mcp_monitor_cleanup(self, mock_tracing_class, mock_metrics_class):
        """Test monitor cleanup."""
        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics
        mock_tracing = Mock()
        mock_tracing_class.return_value = mock_tracing

        monitor = MCPMonitor()
        monitor.cleanup()

        mock_tracing.cleanup.assert_called_once()


class TestMonitoringHelpers:
    """Test monitoring helper functions and decorators."""

    @patch("core.monitoring.MCPMonitor")
    def test_get_monitor_singleton(self, mock_monitor_class):
        """Test get_monitor returns singleton instance."""
        mock_monitor = Mock()
        mock_monitor_class.return_value = mock_monitor

        # First call should create instance
        monitor1 = get_monitor()
        assert monitor1 == mock_monitor
        mock_monitor_class.assert_called_once()

        # Second call should return same instance
        monitor2 = get_monitor()
        assert monitor2 == mock_monitor
        # Should not create new instance
        mock_monitor_class.assert_called_once()

    @patch("core.monitoring.get_monitor")
    def test_monitor_mcp_request_decorator(self, mock_get_monitor):
        """Test monitor_mcp_request decorator function."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        mock_monitor.monitor_request.return_value = lambda func: func

        @monitor_mcp_request("test_endpoint")
        def test_func():
            return "test"

        result = test_func()
        assert result == "test"
        mock_monitor.monitor_request.assert_called_once_with("test_endpoint")

    @patch("core.monitoring.get_monitor")
    def test_monitor_storage_decorator(self, mock_get_monitor):
        """Test monitor_storage decorator function."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        mock_monitor.monitor_storage.return_value = lambda func: func

        @monitor_storage("qdrant", "upsert")
        def test_func():
            return "test"

        result = test_func()
        assert result == "test"
        mock_monitor.monitor_storage.assert_called_once_with("qdrant", "upsert")

    @patch("core.monitoring.get_monitor")
    def test_monitor_embedding_decorator(self, mock_get_monitor):
        """Test monitor_embedding decorator function."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        mock_monitor.monitor_embedding.return_value = lambda func: func

        @monitor_embedding("openai")
        def test_func():
            return "test"

        result = test_func()
        assert result == "test"
        mock_monitor.monitor_embedding.assert_called_once_with("openai")


class TestMonitoringIntegration:
    """Test monitoring integration scenarios."""

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", True)
    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", True)
    @patch("core.monitoring.Counter")
    @patch("core.monitoring.Histogram")
    @patch("core.monitoring.Gauge")
    @patch("core.monitoring.Info")
    @patch("core.monitoring.trace")
    @patch("core.monitoring.TracerProvider")
    def test_full_monitoring_stack_enabled(
        self, mock_tracer_provider, mock_trace, mock_info, mock_gauge, mock_histogram, mock_counter
    ):
        """Test monitoring with both Prometheus and OpenTelemetry enabled."""
        # Setup mocks
        mock_counter.return_value = Mock()
        mock_histogram.return_value = Mock()
        mock_gauge.return_value = Mock()
        mock_info.return_value = Mock()
        mock_tracer_provider.return_value = Mock()
        mock_trace.get_tracer.return_value = Mock()

        monitor = MCPMonitor()

        assert monitor.metrics.enabled is True
        assert monitor.tracing.enabled is True

        # Should initialize both components
        mock_counter.assert_called()
        mock_tracer_provider.assert_called_once()

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", False)
    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False)
    def test_monitoring_stack_disabled(self):
        """Test monitoring with both Prometheus and OpenTelemetry disabled."""
        monitor = MCPMonitor()

        assert monitor.metrics.enabled is False
        assert monitor.tracing.enabled is False

        # Should not raise errors when recording metrics
        monitor.record_context_stored("test")
        monitor.cleanup()

    @patch("core.monitoring.PROMETHEUS_AVAILABLE", True)
    @patch("core.monitoring.OPENTELEMETRY_AVAILABLE", False)
    @patch("core.monitoring.Counter")
    @patch("core.monitoring.Histogram")
    @patch("core.monitoring.Gauge")
    @patch("core.monitoring.Info")
    def test_partial_monitoring_prometheus_only(
        self, mock_info, mock_gauge, mock_histogram, mock_counter
    ):
        """Test monitoring with only Prometheus enabled."""
        # Setup mocks
        mock_counter.return_value = Mock()
        mock_histogram.return_value = Mock()
        mock_gauge.return_value = Mock()
        mock_info.return_value = Mock()

        monitor = MCPMonitor()

        assert monitor.metrics.enabled is True
        assert monitor.tracing.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
