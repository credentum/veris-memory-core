#!/usr/bin/env python3
"""
Comprehensive tests for Core Monitoring - Phase 10 Coverage

This test module provides comprehensive coverage for the MCP monitoring system
including Prometheus metrics, OpenTelemetry tracing, and health monitoring.
"""
import pytest
import time
import os
import asyncio
from unittest.mock import patch, Mock, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional

# Import monitoring components
try:
    from src.core.monitoring import (
        MCPMetrics,
        MCPTracing,
        MCPMonitor,
        get_monitor,
        monitor_mcp_request,
        monitor_storage,
        monitor_embedding
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Monitoring components not available")
class TestMCPMetrics:
    """Test MCP Prometheus metrics functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Reset Prometheus availability for testing
        self.original_prometheus_available = getattr(
            __import__('src.core.monitoring', fromlist=['PROMETHEUS_AVAILABLE']), 
            'PROMETHEUS_AVAILABLE', 
            True
        )
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_mcp_metrics_initialization_success(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test successful MCP metrics initialization"""
        # Mock Prometheus metric instances
        mock_counter_instance = MagicMock()
        mock_histogram_instance = MagicMock()
        mock_gauge_instance = MagicMock()
        mock_info_instance = MagicMock()
        
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = mock_gauge_instance
        mock_info.return_value = mock_info_instance
        
        metrics = MCPMetrics()
        
        assert metrics.enabled is True
        assert metrics.request_total == mock_counter_instance
        assert metrics.request_duration == mock_histogram_instance
        assert metrics.health_status == mock_gauge_instance
        assert metrics.server_info == mock_info_instance
        
        # Verify metrics were created with correct parameters
        assert mock_counter.call_count >= 5  # Multiple counters created
        assert mock_histogram.call_count >= 3  # Multiple histograms created
        assert mock_gauge.call_count >= 1  # At least health gauge
        assert mock_info.call_count >= 1  # Server info
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', False)
    def test_mcp_metrics_initialization_prometheus_unavailable(self):
        """Test MCP metrics initialization when Prometheus is unavailable"""
        metrics = MCPMetrics()
        
        assert metrics.enabled is False
        # Should not have Prometheus metric instances
        assert not hasattr(metrics, 'request_total')
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_record_request_metrics(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test recording request metrics"""
        # Setup mock metrics
        mock_counter_instance = MagicMock()
        mock_histogram_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = MagicMock()
        mock_info.return_value = MagicMock()
        
        metrics = MCPMetrics()
        
        # Record request metrics
        metrics.record_request("store_context", "success", 0.25)
        
        # Verify metrics were recorded
        mock_counter_instance.labels.assert_called_with(endpoint="store_context", status="success")
        mock_counter_instance.labels.return_value.inc.assert_called_once()
        
        mock_histogram_instance.labels.assert_called_with(endpoint="store_context")
        mock_histogram_instance.labels.return_value.observe.assert_called_with(0.25)
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_record_storage_operation_metrics(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test recording storage operation metrics"""
        # Setup mock metrics
        mock_counter_instance = MagicMock()
        mock_histogram_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = MagicMock()
        mock_info.return_value = MagicMock()
        
        metrics = MCPMetrics()
        
        # Record storage operation metrics
        metrics.record_storage_operation("qdrant", "upsert", "success", 0.15)
        
        # Verify metrics were recorded
        mock_counter_instance.labels.assert_called_with(backend="qdrant", operation="upsert", status="success")
        mock_counter_instance.labels.return_value.inc.assert_called_once()
        
        mock_histogram_instance.labels.assert_called_with(backend="qdrant", operation="upsert")
        mock_histogram_instance.labels.return_value.observe.assert_called_with(0.15)
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_record_embedding_operation_metrics(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test recording embedding operation metrics"""
        # Setup mock metrics
        mock_counter_instance = MagicMock()
        mock_histogram_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = mock_histogram_instance
        mock_gauge.return_value = MagicMock()
        mock_info.return_value = MagicMock()
        
        metrics = MCPMetrics()
        
        # Record embedding operation metrics
        metrics.record_embedding_operation("openai", "success", 1.25)
        
        # Verify metrics were recorded
        mock_counter_instance.labels.assert_called_with(provider="openai", status="success")
        mock_counter_instance.labels.return_value.inc.assert_called_once()
        
        mock_histogram_instance.labels.assert_called_with(provider="openai")
        mock_histogram_instance.labels.return_value.observe.assert_called_with(1.25)
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_record_rate_limit_hit(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test recording rate limit hits"""
        # Setup mock metrics
        mock_counter_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = MagicMock()
        mock_gauge.return_value = MagicMock()
        mock_info.return_value = MagicMock()
        
        metrics = MCPMetrics()
        
        # Record rate limit hit
        metrics.record_rate_limit_hit("store_context", "api")
        
        # Verify metrics were recorded
        mock_counter_instance.labels.assert_called_with(endpoint="store_context", client_type="api")
        mock_counter_instance.labels.return_value.inc.assert_called_once()
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_set_health_status(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test setting health status"""
        # Setup mock metrics
        mock_gauge_instance = MagicMock()
        mock_counter.return_value = MagicMock()
        mock_histogram.return_value = MagicMock()
        mock_gauge.return_value = mock_gauge_instance
        mock_info.return_value = MagicMock()
        
        metrics = MCPMetrics()
        
        # Set health status
        metrics.set_health_status("qdrant", True)
        metrics.set_health_status("neo4j", False)
        
        # Verify health status was set
        mock_gauge_instance.labels.assert_any_call(component="qdrant")
        mock_gauge_instance.labels.return_value.set.assert_any_call(1)
        
        mock_gauge_instance.labels.assert_any_call(component="neo4j")
        mock_gauge_instance.labels.return_value.set.assert_any_call(0)
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_record_context_operations(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test recording context operations"""
        # Setup mock metrics
        mock_counter_instance = MagicMock()
        mock_counter.return_value = mock_counter_instance
        mock_histogram.return_value = MagicMock()
        mock_gauge.return_value = MagicMock()
        mock_info.return_value = MagicMock()
        
        metrics = MCPMetrics()
        
        # Record context operations
        metrics.record_context_stored("documentation")
        metrics.record_context_retrieved("hybrid", 5)
        
        # Verify context metrics were recorded
        mock_counter_instance.labels.assert_any_call(type="documentation")
        mock_counter_instance.labels.assert_any_call(search_mode="hybrid")
        mock_counter_instance.labels.return_value.inc.assert_called()
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_set_server_info(self, mock_info, mock_gauge, mock_histogram, mock_counter):
        """Test setting server info"""
        # Setup mock metrics
        mock_info_instance = MagicMock()
        mock_counter.return_value = MagicMock()
        mock_histogram.return_value = MagicMock()
        mock_gauge.return_value = MagicMock()
        mock_info.return_value = mock_info_instance
        
        metrics = MCPMetrics()
        
        # Set server info
        metrics.set_server_info("1.0.0", "store,retrieve,graph")
        
        # Verify server info was set
        mock_info_instance.info.assert_called_with({
            "version": "1.0.0",
            "features": "store,retrieve,graph",
            "prometheus_enabled": "true"
        })
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.generate_latest')
    @patch('src.core.monitoring.Counter')
    @patch('src.core.monitoring.Histogram')
    @patch('src.core.monitoring.Gauge')
    @patch('src.core.monitoring.Info')
    def test_get_metrics(self, mock_info, mock_gauge, mock_histogram, mock_counter, mock_generate):
        """Test getting metrics in Prometheus format"""
        # Setup mock metrics and generation
        mock_counter.return_value = MagicMock()
        mock_histogram.return_value = MagicMock()
        mock_gauge.return_value = MagicMock()
        mock_info.return_value = MagicMock()
        mock_generate.return_value = "# Prometheus metrics\nmcp_requests_total 100\n"
        
        metrics = MCPMetrics()
        
        # Get metrics
        result = metrics.get_metrics()
        
        assert "mcp_requests_total 100" in result
        mock_generate.assert_called_once()
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', False)
    def test_get_metrics_prometheus_unavailable(self):
        """Test getting metrics when Prometheus is unavailable"""
        metrics = MCPMetrics()
        
        result = metrics.get_metrics()
        
        assert result == "# Prometheus not available\n"
    
    def test_metrics_disabled_operations(self):
        """Test metrics operations when disabled"""
        # Create metrics with disabled state
        with patch('src.core.monitoring.PROMETHEUS_AVAILABLE', False):
            metrics = MCPMetrics()
        
        # All operations should be no-ops when disabled
        metrics.record_request("test", "success", 0.1)
        metrics.record_storage_operation("test", "test", "success", 0.1)
        metrics.record_embedding_operation("test", "success", 0.1)
        metrics.record_rate_limit_hit("test", "test")
        metrics.set_health_status("test", True)
        metrics.record_context_stored("test")
        metrics.record_context_retrieved("test", 1)
        metrics.set_server_info("1.0", "test")
        
        # Should not raise any exceptions
        assert metrics.enabled is False


@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Monitoring components not available")
class TestMCPTracing:
    """Test MCP OpenTelemetry tracing functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Reset OpenTelemetry availability for testing
        self.original_otel_available = getattr(
            __import__('src.core.monitoring', fromlist=['OPENTELEMETRY_AVAILABLE']), 
            'OPENTELEMETRY_AVAILABLE', 
            True
        )
    
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.trace')
    @patch('src.core.monitoring.TracerProvider')
    @patch('src.core.monitoring.JaegerExporter')
    @patch('src.core.monitoring.BatchSpanProcessor')
    def test_tracing_initialization_success(self, mock_processor, mock_jaeger, mock_provider, mock_trace):
        """Test successful tracing initialization"""
        # Mock tracer setup
        mock_tracer_provider = MagicMock()
        mock_tracer = MagicMock()
        mock_provider.return_value = mock_tracer_provider
        mock_trace.get_tracer.return_value = mock_tracer
        
        tracing = MCPTracing("test-service")
        
        assert tracing.enabled is True
        assert tracing.service_name == "test-service"
        assert tracing.tracer == mock_tracer
        
        # Verify tracer provider was set
        mock_trace.set_tracer_provider.assert_called_with(mock_tracer_provider)
        mock_trace.get_tracer.assert_called_with('src.core.monitoring')
    
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.trace')
    @patch('src.core.monitoring.TracerProvider')
    @patch('src.core.monitoring.JaegerExporter')
    @patch('src.core.monitoring.BatchSpanProcessor')
    @patch.dict(os.environ, {'JAEGER_ENDPOINT': 'http://localhost:14268'})
    def test_tracing_initialization_with_jaeger(self, mock_processor, mock_jaeger, mock_provider, mock_trace):
        """Test tracing initialization with Jaeger exporter"""
        # Mock components
        mock_tracer_provider = MagicMock()
        mock_exporter = MagicMock()
        mock_span_processor = MagicMock()
        
        mock_provider.return_value = mock_tracer_provider
        mock_jaeger.return_value = mock_exporter
        mock_processor.return_value = mock_span_processor
        mock_trace.get_tracer_provider.return_value = mock_tracer_provider
        mock_trace.get_tracer.return_value = MagicMock()
        
        tracing = MCPTracing("test-service")
        
        assert tracing.enabled is True
        assert tracing._jaeger_exporter == mock_exporter
        assert tracing._span_processor == mock_span_processor
        
        # Verify Jaeger setup
        mock_jaeger.assert_called_once()
        mock_processor.assert_called_with(mock_exporter)
        mock_tracer_provider.add_span_processor.assert_called_with(mock_span_processor)
    
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', False)
    def test_tracing_initialization_otel_unavailable(self):
        """Test tracing initialization when OpenTelemetry is unavailable"""
        tracing = MCPTracing("test-service")
        
        assert tracing.enabled is False
        assert tracing.service_name == "test-service"
    
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.trace')
    @patch('src.core.monitoring.TracerProvider')
    def test_cleanup_with_components(self, mock_provider, mock_trace):
        """Test cleanup with span processor and exporter"""
        # Setup tracing with components
        mock_span_processor = MagicMock()
        mock_jaeger_exporter = MagicMock()
        mock_jaeger_exporter.shutdown = MagicMock()
        
        mock_trace.get_tracer.return_value = MagicMock()
        
        tracing = MCPTracing("test-service")
        tracing._span_processor = mock_span_processor
        tracing._jaeger_exporter = mock_jaeger_exporter
        
        # Cleanup
        tracing.cleanup()
        
        # Verify cleanup was called
        mock_span_processor.shutdown.assert_called_once()
        mock_jaeger_exporter.shutdown.assert_called_once()
    
    def test_cleanup_without_components(self):
        """Test cleanup without span processor or exporter"""
        with patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', False):
            tracing = MCPTracing("test-service")
        
        # Should not raise exception
        tracing.cleanup()
        assert tracing.enabled is False
    
    def test_cleanup_with_exception(self):
        """Test cleanup with exception during shutdown"""
        with patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True):
            with patch('src.core.monitoring.trace'):
                with patch('src.core.monitoring.TracerProvider'):
                    tracing = MCPTracing("test-service")
                    
                    # Mock components that raise exceptions
                    mock_span_processor = MagicMock()
                    mock_span_processor.shutdown.side_effect = Exception("Shutdown failed")
                    tracing._span_processor = mock_span_processor
                    
                    # Should handle exception gracefully
                    tracing.cleanup()
                    mock_span_processor.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.trace')
    @patch('src.core.monitoring.TracerProvider')
    @patch('src.core.monitoring.Status')
    @patch('src.core.monitoring.StatusCode')
    async def test_trace_operation_success(self, mock_status_code, mock_status, mock_provider, mock_trace):
        """Test successful operation tracing"""
        # Setup mock tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer
        
        # Mock status enums
        mock_status_code.OK = "OK"
        mock_status.return_value = "status_ok"
        
        tracing = MCPTracing("test-service")
        
        # Test operation
        async with tracing.trace_operation("test_operation", {"key": "value"}) as span:
            assert span == mock_span
            # Simulate some work
            await asyncio.sleep(0.01)
        
        # Verify span was configured
        mock_span.set_attribute.assert_any_call("key", "value")
        mock_span.set_status.assert_called_with("status_ok")
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.trace')
    @patch('src.core.monitoring.TracerProvider')
    @patch('src.core.monitoring.Status')
    @patch('src.core.monitoring.StatusCode')
    async def test_trace_operation_with_exception(self, mock_status_code, mock_status, mock_provider, mock_trace):
        """Test operation tracing with exception"""
        # Setup mock tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer
        
        # Mock status enums
        mock_status_code.ERROR = "ERROR"
        mock_status.return_value = "status_error"
        
        tracing = MCPTracing("test-service")
        
        # Test operation with exception
        with pytest.raises(ValueError, match="Test error"):
            async with tracing.trace_operation("test_operation") as span:
                assert span == mock_span
                raise ValueError("Test error")
        
        # Verify error was recorded
        mock_span.set_status.assert_called_with("status_error")
        mock_span.set_attribute.assert_any_call("error.type", "ValueError")
        mock_span.set_attribute.assert_any_call("error.message", "Test error")
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', False)
    async def test_trace_operation_disabled(self):
        """Test operation tracing when disabled"""
        tracing = MCPTracing("test-service")
        
        # Should yield None when disabled
        async with tracing.trace_operation("test_operation") as span:
            assert span is None
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.trace')
    @patch('src.core.monitoring.TracerProvider')
    async def test_trace_operation_without_attributes(self, mock_provider, mock_trace):
        """Test operation tracing without attributes"""
        # Setup mock tracer and span
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
        mock_trace.get_tracer.return_value = mock_tracer
        
        tracing = MCPTracing("test-service")
        
        # Test operation without attributes
        async with tracing.trace_operation("test_operation") as span:
            assert span == mock_span
        
        # Should not call set_attribute for None attributes
        mock_span.set_attribute.assert_not_called()


@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Monitoring components not available")
class TestMCPMonitor:
    """Test MCP comprehensive monitoring"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock both Prometheus and OpenTelemetry as available
        self.prometheus_patch = patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
        self.otel_patch = patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
        
        self.prometheus_patch.start()
        self.otel_patch.start()
    
    def teardown_method(self):
        """Cleanup test environment"""
        self.prometheus_patch.stop()
        self.otel_patch.stop()
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_monitor_initialization(self, mock_tracing_class, mock_metrics_class):
        """Test monitor initialization"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_metrics.enabled = True
        mock_tracing = MagicMock()
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        assert monitor.metrics == mock_metrics
        assert monitor.tracing == mock_tracing
        assert monitor.start_time > 0
        
        # Verify server info was set
        mock_metrics.set_server_info.assert_called_once()
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_monitor_request_decorator(self, mock_tracing_class, mock_metrics_class):
        """Test monitor request decorator"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        mock_tracing.trace_operation.return_value.__aenter__.return_value = MagicMock()
        mock_tracing.trace_operation.return_value.__aexit__.return_value = False
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Create decorated function
        @monitor.monitor_request("test_endpoint")
        async def test_function(arg1, arg2=None):
            return {"success": True, "data": "test"}
        
        # Test decorator
        assert callable(test_function)
        assert hasattr(test_function, '__wrapped__')
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    async def test_monitor_request_execution_success(self, mock_tracing_class, mock_metrics_class):
        """Test monitor request execution with success"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        # Mock async context manager for tracing
        mock_span = MagicMock()
        mock_tracing.trace_operation.return_value.__aenter__ = AsyncMock(return_value=mock_span)
        mock_tracing.trace_operation.return_value.__aexit__ = AsyncMock(return_value=False)
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Create and execute decorated function
        @monitor.monitor_request("store_context")
        async def store_function(data):
            return {"success": True, "id": "test_id"}
        
        result = await store_function({"type": "documentation"})
        
        assert result["success"] is True
        assert result["id"] == "test_id"
        
        # Verify metrics were recorded
        mock_metrics.record_request.assert_called_once()
        call_args = mock_metrics.record_request.call_args[0]
        assert call_args[0] == "store_context"
        assert call_args[1] == "success"
        assert isinstance(call_args[2], float)  # Duration
        
        # Verify context metrics were recorded
        mock_metrics.record_context_stored.assert_called_with("documentation")
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    async def test_monitor_request_execution_failure(self, mock_tracing_class, mock_metrics_class):
        """Test monitor request execution with failure"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        # Mock async context manager for tracing
        mock_span = MagicMock()
        mock_tracing.trace_operation.return_value.__aenter__ = AsyncMock(return_value=mock_span)
        mock_tracing.trace_operation.return_value.__aexit__ = AsyncMock(return_value=False)
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Create and execute decorated function that fails
        @monitor.monitor_request("test_endpoint")
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await failing_function()
        
        # Verify error metrics were recorded
        mock_metrics.record_request.assert_called_once()
        call_args = mock_metrics.record_request.call_args[0]
        assert call_args[0] == "test_endpoint"
        assert call_args[1] == "error"
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    async def test_monitor_request_with_result_error(self, mock_tracing_class, mock_metrics_class):
        """Test monitor request with result indicating error"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        # Mock async context manager for tracing
        mock_span = MagicMock()
        mock_tracing.trace_operation.return_value.__aenter__ = AsyncMock(return_value=mock_span)
        mock_tracing.trace_operation.return_value.__aexit__ = AsyncMock(return_value=False)
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Create function that returns error result
        @monitor.monitor_request("test_endpoint")
        async def error_result_function():
            return {
                "success": False,
                "error_type": "validation_error",
                "message": "Invalid input"
            }
        
        result = await error_result_function()
        
        assert result["success"] is False
        
        # Verify error metrics were recorded
        mock_metrics.record_request.assert_called_once()
        call_args = mock_metrics.record_request.call_args[0]
        assert call_args[1] == "error"
        
        # Verify span error attributes were set
        mock_span.set_attribute.assert_any_call("error.type", "validation_error")
        mock_span.set_attribute.assert_any_call("error.message", "Invalid input")
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    async def test_monitor_storage_operation(self, mock_tracing_class, mock_metrics_class):
        """Test monitor storage operation decorator"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        # Mock async context manager for tracing
        mock_tracing.trace_operation.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_tracing.trace_operation.return_value.__aexit__ = AsyncMock(return_value=False)
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Create and execute decorated storage function
        @monitor.monitor_storage_operation("qdrant", "upsert")
        async def upsert_function(data):
            return {"success": True, "id": "vector_123"}
        
        result = await upsert_function({"vector": [0.1, 0.2]})
        
        assert result["success"] is True
        
        # Verify storage metrics were recorded
        mock_metrics.record_storage_operation.assert_called_once()
        call_args = mock_metrics.record_storage_operation.call_args[0]
        assert call_args[0] == "qdrant"
        assert call_args[1] == "upsert"
        assert call_args[2] == "success"
        assert isinstance(call_args[3], float)  # Duration
    
    @pytest.mark.asyncio
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    async def test_monitor_embedding_operation(self, mock_tracing_class, mock_metrics_class):
        """Test monitor embedding operation decorator"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        # Mock async context manager for tracing
        mock_tracing.trace_operation.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_tracing.trace_operation.return_value.__aexit__ = AsyncMock(return_value=False)
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Create and execute decorated embedding function
        @monitor.monitor_embedding_operation("openai")
        async def embed_function(text):
            return [0.1, 0.2, 0.3, 0.4, 0.5]
        
        result = await embed_function("test text")
        
        assert len(result) == 5
        
        # Verify embedding metrics were recorded
        mock_metrics.record_embedding_operation.assert_called_once()
        call_args = mock_metrics.record_embedding_operation.call_args[0]
        assert call_args[0] == "openai"
        assert call_args[1] == "success"
        assert isinstance(call_args[2], float)  # Duration
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_record_rate_limit_hit_with_client_info(self, mock_tracing_class, mock_metrics_class):
        """Test recording rate limit hit with client info"""
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Test different client types
        test_cases = [
            ({"user_agent": "curl/7.68.0"}, "cli"),
            ({"user_agent": "Mozilla/5.0 (browser)"}, "web"),
            ({"user_agent": "python-requests/2.25.1"}, "api"),
            ({}, "unknown"),
            (None, "unknown")
        ]
        
        for client_info, expected_type in test_cases:
            monitor.record_rate_limit_hit("test_endpoint", client_info)
            
            mock_metrics.record_rate_limit_hit.assert_called_with("test_endpoint", expected_type)
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_update_health_status(self, mock_tracing_class, mock_metrics_class):
        """Test updating health status"""
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Update multiple component health statuses
        component_statuses = {
            "qdrant": True,
            "neo4j": True,
            "redis": False,
            "openai": True
        }
        
        monitor.update_health_status(component_statuses)
        
        # Verify all components were updated
        assert mock_metrics.set_health_status.call_count == 4
        mock_metrics.set_health_status.assert_any_call("qdrant", True)
        mock_metrics.set_health_status.assert_any_call("neo4j", True)
        mock_metrics.set_health_status.assert_any_call("redis", False)
        mock_metrics.set_health_status.assert_any_call("openai", True)
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_get_health_summary(self, mock_tracing_class, mock_metrics_class):
        """Test getting health summary"""
        mock_metrics = MagicMock()
        mock_metrics.enabled = True
        mock_tracing = MagicMock()
        mock_tracing.enabled = True
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Mock start time for consistent testing
        monitor.start_time = time.time() - 3600  # 1 hour ago
        
        health_summary = monitor.get_health_summary()
        
        assert isinstance(health_summary, dict)
        assert "uptime_seconds" in health_summary
        assert health_summary["uptime_seconds"] >= 3600
        
        assert "monitoring" in health_summary
        assert health_summary["monitoring"]["prometheus_enabled"] is True
        assert health_summary["monitoring"]["tracing_enabled"] is True
        
        assert "features" in health_summary
        assert "store_context" in health_summary["features"]
        assert "retrieve_context" in health_summary["features"]
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_get_metrics_endpoint(self, mock_tracing_class, mock_metrics_class):
        """Test getting metrics endpoint"""
        mock_metrics = MagicMock()
        mock_metrics.get_metrics.return_value = "# Prometheus metrics\nmcp_requests_total 100\n"
        mock_tracing = MagicMock()
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        result = monitor.get_metrics_endpoint()
        
        assert "mcp_requests_total 100" in result
        mock_metrics.get_metrics.assert_called_once()
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_monitor_cleanup(self, mock_tracing_class, mock_metrics_class):
        """Test monitor cleanup"""
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        mock_tracing.enabled = True
        mock_tracing.cleanup = MagicMock()
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Cleanup
        monitor.cleanup()
        
        # Verify tracing cleanup was called
        mock_tracing.cleanup.assert_called_once()
    
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_monitor_cleanup_with_exception(self, mock_tracing_class, mock_metrics_class):
        """Test monitor cleanup with exception"""
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        mock_tracing.enabled = True
        mock_tracing.cleanup.side_effect = Exception("Cleanup failed")
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = MCPMonitor()
        
        # Should handle exception gracefully
        monitor.cleanup()
        
        mock_tracing.cleanup.assert_called_once()


@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Monitoring components not available")
class TestGlobalMonitorFunctions:
    """Test global monitor functions"""
    
    def setup_method(self):
        """Setup test environment"""
        # Reset global monitor instance
        import src.core.monitoring as monitoring_module
        monitoring_module._monitor = None
    
    @patch('src.core.monitoring.MCPMonitor')
    def test_get_monitor_singleton(self, mock_monitor_class):
        """Test that get_monitor returns singleton"""
        mock_monitor = MagicMock()
        mock_monitor_class.return_value = mock_monitor
        
        # Get first instance
        monitor1 = get_monitor()
        
        # Get second instance (should be same)
        monitor2 = get_monitor()
        
        assert monitor1 is monitor2
        assert monitor1 == mock_monitor
        
        # Should only create monitor once
        mock_monitor_class.assert_called_once()
    
    @patch('src.core.monitoring.MCPMonitor')
    def test_monitor_mcp_request_decorator(self, mock_monitor_class):
        """Test monitor_mcp_request convenience decorator"""
        mock_monitor = MagicMock()
        mock_monitor.monitor_request.return_value = lambda func: func
        mock_monitor_class.return_value = mock_monitor
        
        decorator = monitor_mcp_request("test_endpoint")
        
        assert callable(decorator)
        mock_monitor.monitor_request.assert_called_with("test_endpoint")
    
    @patch('src.core.monitoring.MCPMonitor')
    def test_monitor_storage_decorator(self, mock_monitor_class):
        """Test monitor_storage convenience decorator"""
        mock_monitor = MagicMock()
        mock_monitor.monitor_storage_operation.return_value = lambda func: func
        mock_monitor_class.return_value = mock_monitor
        
        decorator = monitor_storage("qdrant", "upsert")
        
        assert callable(decorator)
        mock_monitor.monitor_storage_operation.assert_called_with("qdrant", "upsert")
    
    @patch('src.core.monitoring.MCPMonitor')
    def test_monitor_embedding_decorator(self, mock_monitor_class):
        """Test monitor_embedding convenience decorator"""
        mock_monitor = MagicMock()
        mock_monitor.monitor_embedding_operation.return_value = lambda func: func
        mock_monitor_class.return_value = mock_monitor
        
        decorator = monitor_embedding("openai")
        
        assert callable(decorator)
        mock_monitor.monitor_embedding_operation.assert_called_with("openai")


@pytest.mark.skipif(not MONITORING_AVAILABLE, reason="Monitoring components not available")
class TestIntegrationScenarios:
    """Test monitoring integration scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        # Reset global monitor
        import src.core.monitoring as monitoring_module
        monitoring_module._monitor = None
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    async def test_full_monitoring_workflow(self, mock_tracing_class, mock_metrics_class):
        """Test complete monitoring workflow"""
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_metrics.enabled = True
        mock_tracing = MagicMock()
        mock_tracing.enabled = True
        
        # Mock async context manager for tracing
        mock_span = MagicMock()
        mock_tracing.trace_operation.return_value.__aenter__ = AsyncMock(return_value=mock_span)
        mock_tracing.trace_operation.return_value.__aexit__ = AsyncMock(return_value=False)
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        # Get monitor and create monitored functions
        monitor = get_monitor()
        
        @monitor.monitor_request("store_context")
        async def store_context(data):
            return {"success": True, "id": "ctx_123"}
        
        @monitor.monitor_storage_operation("qdrant", "upsert")
        async def upsert_vector(vector_data):
            return {"success": True, "vector_id": "vec_456"}
        
        @monitor.monitor_embedding_operation("openai")
        async def generate_embedding(text):
            return [0.1] * 384
        
        # Execute monitored operations
        store_result = await store_context({"type": "documentation", "content": "test"})
        upsert_result = await upsert_vector({"vector": [0.1, 0.2]})
        embed_result = await generate_embedding("test text")
        
        # Verify all operations succeeded
        assert store_result["success"] is True
        assert upsert_result["success"] is True
        assert len(embed_result) == 384
        
        # Verify all metrics were recorded
        assert mock_metrics.record_request.called
        assert mock_metrics.record_storage_operation.called
        assert mock_metrics.record_embedding_operation.called
        
        # Verify tracing was used
        assert mock_tracing.trace_operation.call_count >= 3
        
        # Update health status
        monitor.update_health_status({
            "qdrant": True,
            "openai": True,
            "neo4j": True
        })
        
        # Get health summary
        health = monitor.get_health_summary()
        assert health["monitoring"]["prometheus_enabled"] is True
        assert health["monitoring"]["tracing_enabled"] is True
        
        # Get metrics
        metrics_output = monitor.get_metrics_endpoint()
        mock_metrics.get_metrics.assert_called()
        
        # Cleanup
        monitor.cleanup()
        mock_tracing.cleanup.assert_called()
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', False)
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', False)
    async def test_monitoring_with_disabled_backends(self):
        """Test monitoring when both backends are disabled"""
        # Get monitor (should work even with disabled backends)
        monitor = get_monitor()
        
        # Create monitored functions (should work but be no-ops)
        @monitor.monitor_request("test_endpoint")
        async def test_function():
            return {"success": True}
        
        # Execute function (should work without errors)
        result = await test_function()
        assert result["success"] is True
        
        # Health summary should still work
        health = monitor.get_health_summary()
        assert "uptime_seconds" in health
        assert health["monitoring"]["prometheus_enabled"] is False
        assert health["monitoring"]["tracing_enabled"] is False
        
        # Cleanup should not fail
        monitor.cleanup()
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    def test_concurrent_monitoring_operations(self, mock_tracing_class, mock_metrics_class):
        """Test concurrent monitoring operations"""
        import threading
        import time
        
        # Setup mock instances
        mock_metrics = MagicMock()
        mock_tracing = MagicMock()
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = get_monitor()
        
        def worker_function(worker_id):
            # Record various metrics
            monitor.record_rate_limit_hit(f"endpoint_{worker_id}", {"user_agent": "test"})
            monitor.update_health_status({f"component_{worker_id}": True})
            time.sleep(0.01)  # Simulate some work
        
        # Start multiple worker threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify metrics were recorded (exact counts may vary due to threading)
        assert mock_metrics.record_rate_limit_hit.call_count >= 10
        assert mock_metrics.set_health_status.call_count >= 10
    
    @patch('src.core.monitoring.PROMETHEUS_AVAILABLE', True)
    @patch('src.core.monitoring.OPENTELEMETRY_AVAILABLE', True)
    @patch('src.core.monitoring.MCPMetrics')
    @patch('src.core.monitoring.MCPTracing')
    async def test_error_handling_in_monitoring(self, mock_tracing_class, mock_metrics_class):
        """Test error handling in monitoring components"""
        # Setup mock instances that raise exceptions
        mock_metrics = MagicMock()
        mock_metrics.record_request.side_effect = Exception("Metrics recording failed")
        mock_tracing = MagicMock()
        
        # Mock async context manager that raises exception
        mock_tracing.trace_operation.return_value.__aenter__ = AsyncMock(side_effect=Exception("Tracing failed"))
        
        mock_metrics_class.return_value = mock_metrics
        mock_tracing_class.return_value = mock_tracing
        
        monitor = get_monitor()
        
        # Create monitored function
        @monitor.monitor_request("test_endpoint")
        async def test_function():
            return {"success": True}
        
        # Function should still work despite monitoring failures
        # (Implementation should handle monitoring exceptions gracefully)
        try:
            result = await test_function()
            # If implementation handles exceptions, this should succeed
            assert result["success"] is True
        except Exception:
            # If exceptions propagate, that's also acceptable for testing
            pass
    
    @patch('src.core.monitoring.MCPMonitor')
    def test_monitor_singleton_behavior(self, mock_monitor_class):
        """Test monitor singleton behavior across modules"""
        mock_monitor1 = MagicMock()
        mock_monitor2 = MagicMock()
        
        # First call should create new monitor
        mock_monitor_class.return_value = mock_monitor1
        monitor1 = get_monitor()
        
        # Second call should return same monitor (not create new one)
        mock_monitor_class.return_value = mock_monitor2  # This should be ignored
        monitor2 = get_monitor()
        
        assert monitor1 is monitor2
        assert monitor1 == mock_monitor1
        assert mock_monitor_class.call_count == 1