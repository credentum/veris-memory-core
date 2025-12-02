#!/usr/bin/env python3
"""
Tests for structured logging middleware.
"""

import json
import time
import pytest
import asyncio
from unittest.mock import patch
from contextlib import redirect_stdout

from src.utils.logging_middleware import (
    StructuredLogger,
    LogLevel,
    LogEntry,
    log_backend_timing,
    trace_context,
    get_current_trace_id,
    get_request_duration_ms,
    TimingCollector,
    setup_logging
)


class TestLogEntry:
    """Test LogEntry data class."""
    
    def test_basic_log_entry(self):
        """Test creating a basic log entry."""
        entry = LogEntry(
            timestamp=1692542400.0,
            level="INFO",
            message="Test message",
            logger_name="test_logger",
            trace_id="test_123"
        )
        
        assert entry.timestamp == 1692542400.0
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.logger_name == "test_logger"
        assert entry.trace_id == "test_123"
    
    def test_log_entry_to_dict(self):
        """Test converting log entry to dictionary."""
        entry = LogEntry(
            timestamp=1692542400.0,
            level="ERROR",
            message="Error occurred",
            logger_name="error_logger",
            trace_id="error_123",
            duration_ms=125.5,
            backend="vector",
            metadata={"retry_count": 3}
        )
        
        result = entry.to_dict()
        
        assert result["timestamp"] == 1692542400.0
        assert result["level"] == "ERROR"
        assert result["message"] == "Error occurred"
        assert result["logger_name"] == "error_logger"
        assert result["trace_id"] == "error_123"
        assert result["duration_ms"] == 125.5
        assert result["backend"] == "vector"
        assert result["metadata"] == {"retry_count": 3}
    
    def test_log_entry_none_filtering(self):
        """Test that None values are filtered out."""
        entry = LogEntry(
            timestamp=1692542400.0,
            level="INFO",
            message="Test message",
            logger_name="test_logger",
            trace_id="test_123",
            module=None,
            function=None,
            duration_ms=None
        )
        
        result = entry.to_dict()
        
        assert "module" not in result
        assert "function" not in result
        assert "duration_ms" not in result


class TestStructuredLogger:
    """Test StructuredLogger functionality."""
    
    @pytest.fixture
    def logger(self):
        return StructuredLogger("test_logger", enable_console=False)  # Disable console for testing
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_name", enable_console=True, pii_redaction=False)
        
        assert logger.name == "test_name"
        assert logger.enable_console is True
        assert logger.pii_redaction is False
        assert logger.stdlib_logger.name == "test_name"
    
    def test_pii_redaction(self, logger):
        """Test PII redaction functionality."""
        # Enable PII redaction
        logger.pii_redaction = True
        
        test_data = {
            "user_id": "12345678",
            "password": "secret123",
            "email": "test@example.com",
            "normal_field": "normal_value",
            "nested": {
                "api_key": "abcdefghijkl",
                "name": "John Doe"
            }
        }
        
        redacted = logger._redact_pii(test_data)
        
        # PII fields should be redacted
        assert redacted["user_id"] == "12****78"
        assert redacted["password"] == "***REDACTED***"  # Too short for partial redaction
        assert redacted["email"] == "te***@example.com"
        
        # Normal fields should remain unchanged
        assert redacted["normal_field"] == "normal_value"
        
        # Nested PII should be redacted
        assert redacted["nested"]["api_key"] == "ab*******kl"
        assert redacted["nested"]["name"] == "John Doe"
    
    @patch('builtins.print')
    def test_log_output(self, mock_print, logger):
        """Test structured log output."""
        logger.enable_console = True
        
        with patch('src.utils.logging_middleware.trace_id_var.get', return_value='test_trace'):
            logger.info("Test message", backend="test_backend", duration_ms=42.5)
        
        # Should have called print with JSON
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        
        # Parse the JSON output
        log_data = json.loads(call_args)
        
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["logger_name"] == "test_logger"
        assert log_data["trace_id"] == "test_trace"
        assert log_data["backend"] == "test_backend"
        assert log_data["duration_ms"] == 42.5
    
    def test_log_levels(self, logger):
        """Test all log levels."""
        with patch('builtins.print') as mock_print:
            logger.enable_console = True
            
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        
        # Should have 5 calls to print
        assert mock_print.call_count == 5
        
        # Check log levels in output
        calls = [json.loads(call[0][0]) for call in mock_print.call_args_list]
        levels = [call["level"] for call in calls]
        
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def test_log_backend_timing(self, logger):
        """Test backend timing logging."""
        with patch('builtins.print') as mock_print:
            logger.enable_console = True
            
            logger.log_backend_timing(
                backend="vector",
                operation="search",
                duration_ms=125.5,
                result_count=10,
                top_score=0.95,
                extra_metadata="test"
            )
        
        mock_print.assert_called_once()
        log_data = json.loads(mock_print.call_args[0][0])
        
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Backend operation completed"
        assert log_data["backend"] == "vector"
        assert log_data["module"] == "search"
        assert log_data["duration_ms"] == 125.5
        assert log_data["metadata"]["result_count"] == 10
        assert log_data["metadata"]["top_score"] == 0.95
        assert log_data["metadata"]["extra_metadata"] == "test"


class TestContextManagers:
    """Test context managers for tracing and timing."""
    
    def test_trace_context(self):
        """Test trace context manager."""
        # Test with provided trace ID
        with trace_context("custom_trace_123") as trace_id:
            assert trace_id == "custom_trace_123"
            assert get_current_trace_id() == "custom_trace_123"
        
        # Test with generated trace ID
        with trace_context() as trace_id:
            assert trace_id is not None
            assert len(trace_id) > 0
            assert get_current_trace_id() == trace_id
    
    def test_request_duration_tracking(self):
        """Test request duration tracking."""
        with trace_context():
            initial_duration = get_request_duration_ms()
            assert initial_duration >= 0
            
            # Small delay to measure duration
            time.sleep(0.01)
            
            final_duration = get_request_duration_ms()
            assert final_duration > initial_duration
            assert final_duration >= 10  # At least 10ms
    
    @pytest.mark.asyncio
    async def test_log_backend_timing_context(self):
        """Test async backend timing context manager."""
        logger = StructuredLogger("test", enable_console=False)
        
        with patch.object(logger, 'log_backend_timing') as mock_log:
            async with log_backend_timing("test_backend", "search", logger) as metadata:
                metadata["result_count"] = 5
                metadata["top_score"] = 0.8
                
                # Simulate some work
                await asyncio.sleep(0.01)
            
            # Should have logged timing
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            call_kwargs = mock_log.call_args[1]
            
            assert call_args[0] == "test_backend"  # backend name
            assert call_args[1] == "search"       # operation
            assert call_args[2] > 0               # duration_ms
            assert call_kwargs["result_count"] == 5
            assert call_kwargs["top_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_log_backend_timing_error_handling(self):
        """Test error handling in backend timing context."""
        logger = StructuredLogger("test", enable_console=False)
        
        with patch.object(logger, 'error') as mock_error:
            with pytest.raises(ValueError):
                async with log_backend_timing("test_backend", "search", logger) as metadata:
                    metadata["attempted"] = True
                    raise ValueError("Test error")
            
            # Should have logged error
            mock_error.assert_called_once()
            call_kwargs = mock_error.call_args[1]
            
            assert call_kwargs["backend"] == "test_backend"
            assert call_kwargs["module"] == "search"
            assert call_kwargs["duration_ms"] > 0
            assert call_kwargs["metadata"]["error"] == "Test error"
            assert call_kwargs["metadata"]["attempted"] is True


class TestTimingCollector:
    """Test TimingCollector utility."""
    
    def test_timing_collection(self):
        """Test collecting multiple timing measurements."""
        collector = TimingCollector()
        
        # Record some timings
        collector.record_timing("search", 45.2, result_count=10)
        collector.record_timing("search", 52.1, result_count=5)
        collector.record_timing("health_check", 3.1, status="healthy")
        
        summary = collector.get_summary()
        
        # Check search operation summary
        search_summary = summary["search"]
        assert search_summary["count"] == 2
        assert search_summary["total_ms"] == 97.3
        assert search_summary["avg_ms"] == 48.65
        assert search_summary["min_ms"] == 45.2
        assert search_summary["max_ms"] == 52.1
        
        # Check health_check operation summary
        health_summary = summary["health_check"]
        assert health_summary["count"] == 1
        assert health_summary["total_ms"] == 3.1
        assert health_summary["avg_ms"] == 3.1
        assert health_summary["metadata"]["status"] == "healthy"
    
    def test_empty_collector(self):
        """Test empty timing collector."""
        collector = TimingCollector()
        summary = collector.get_summary()
        assert summary == {}


class TestSetupLogging:
    """Test global logging setup."""
    
    def test_setup_logging_configuration(self):
        """Test setup_logging function."""
        # Test basic setup (should not raise errors)
        setup_logging(level="DEBUG", enable_json=True)
        setup_logging(level="INFO", enable_json=False)
        
        # Test invalid level handling
        setup_logging(level="INVALID", enable_json=True)  # Should default to INFO


# Integration test
@pytest.mark.asyncio
async def test_full_logging_integration():
    """Test full logging integration with trace context and backend timing."""
    logger = StructuredLogger("integration_test", enable_console=False)
    
    with patch('builtins.print') as mock_print:
        logger.enable_console = True
        
        with trace_context("integration_trace"):
            async with log_backend_timing("vector", "search", logger) as metadata:
                metadata["query"] = "test query"
                metadata["result_count"] = 3
                
                # Simulate work
                await asyncio.sleep(0.01)
                
                # Log additional info
                logger.info("Search completed", query_type="semantic")
    
    # Should have logged both the info message and the timing
    assert mock_print.call_count == 2
    
    # Parse log outputs
    logs = [json.loads(call[0][0]) for call in mock_print.call_args_list]
    
    # Check that both logs have the same trace_id
    trace_ids = [log["trace_id"] for log in logs]
    assert all(tid == "integration_trace" for tid in trace_ids)
    
    # Check log contents
    info_log = next(log for log in logs if log["message"] == "Search completed")
    timing_log = next(log for log in logs if log["message"] == "Backend operation completed")
    
    assert info_log["query_type"] == "semantic"
    assert timing_log["backend"] == "vector"
    assert timing_log["module"] == "search"
    assert timing_log["duration_ms"] > 0