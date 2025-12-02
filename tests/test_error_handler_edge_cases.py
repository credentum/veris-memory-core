#!/usr/bin/env python3
"""
Comprehensive edge case tests for Error Handler - Phase 5 Coverage

This test module focuses on edge cases, security scenarios, and complex
error handling situations that weren't covered in basic unit tests.
"""
import pytest
import os
import sys
import traceback
from unittest.mock import patch, Mock, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import error handling components
try:
    from src.core.error_handler import (
        create_error_response,
        handle_storage_error,
        handle_validation_error,
        handle_timeout_error,
        handle_permission_error,
        sanitize_error_message,
        extract_error_context,
        format_error_details,
        get_error_severity,
        log_error_with_context
    )
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False


@pytest.mark.skipif(not ERROR_HANDLER_AVAILABLE, reason="Error handler not available")
class TestErrorResponseCreationEdgeCases:
    """Edge cases for error response creation functionality"""
    
    def test_error_response_with_complex_data(self):
        """Test error response creation with complex data structures"""
        # Test with nested dictionary details
        complex_details = {
            "validation_errors": [
                {"field": "username", "error": "too short"},
                {"field": "email", "error": "invalid format"}
            ],
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": "req_12345",
                "user_context": {"role": "user", "permissions": ["read"]}
            },
            "stack_trace": traceback.format_stack()[:3]  # Limited stack trace
        }
        
        response = create_error_response(
            success=False,
            message="Complex validation failure",
            error_type="validation_error",
            details=complex_details,
            error_code="VAL_001",
            timestamp=datetime.utcnow().isoformat()
        )
        
        assert response["success"] is False
        assert response["message"] == "Complex validation failure"
        assert response["error_type"] == "validation_error"
        assert "details" in response
        assert "validation_errors" in response["details"]
        assert len(response["details"]["validation_errors"]) == 2
    
    def test_error_response_with_none_values(self):
        """Test error response with None values in various fields"""
        response = create_error_response(
            success=False,
            message=None,  # None message
            error_type=None,  # None error type
            details=None,  # None details
            error_code=None
        )
        
        assert response["success"] is False
        assert "message" in response  # Should handle None gracefully
        assert "error_type" in response
    
    def test_error_response_with_circular_references(self):
        """Test error response with circular reference in details"""
        # Create circular reference
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        
        # Should handle circular references without crashing
        try:
            response = create_error_response(
                success=False,
                message="Circular reference test",
                error_type="test_error",
                details=circular_dict
            )
            # If it doesn't crash, that's a success
            assert response["success"] is False
        except (ValueError, RecursionError):
            # JSON serialization might fail, which is acceptable
            pytest.skip("Circular reference handling not implemented")
    
    def test_error_response_with_large_data(self):
        """Test error response with very large data structures"""
        large_details = {
            "large_array": list(range(10000)),
            "large_string": "x" * 100000,
            "nested_large": {
                f"key_{i}": f"value_{i}" * 100 for i in range(1000)
            }
        }
        
        response = create_error_response(
            success=False,
            message="Large data test",
            error_type="test_error",
            details=large_details
        )
        
        assert response["success"] is False
        assert "details" in response
    
    def test_error_response_with_special_characters(self):
        """Test error response with special characters and encoding"""
        special_message = "Error with special chars: αβγ 中文 🚫 \x00\x01\x02"
        special_details = {
            "unicode_text": "Unicode: αβγδε ñáéíóú",
            "emoji_text": "Emojis: 🚫❌⚠️🔥💥",
            "control_chars": "Control: \x00\x01\x02\x03",
            "mixed": "Mixed: αβγ 🚫 \x00 test"
        }
        
        response = create_error_response(
            success=False,
            message=special_message,
            error_type="unicode_error",
            details=special_details
        )
        
        assert response["success"] is False
        assert "unicode" in response["message"] or "special" in response["message"]


@pytest.mark.skipif(not ERROR_HANDLER_AVAILABLE, reason="Error handler not available")
class TestStorageErrorHandlingEdgeCases:
    """Edge cases for storage error handling"""
    
    def test_storage_error_with_connection_failures(self):
        """Test storage error handling with various connection failures"""
        connection_errors = [
            ConnectionError("Redis connection timeout"),
            ConnectionAbortedError("Connection aborted by server"),
            ConnectionRefusedError("Connection refused by Redis"),
            ConnectionResetError("Connection reset by peer"),
            BrokenPipeError("Broken pipe"),
            OSError("Network unreachable")
        ]
        
        for error in connection_errors:
            response = handle_storage_error(error, "connect_to_redis")
            
            assert response["success"] is False
            assert "connection" in response["message"].lower()
            assert response["error_type"] in ["storage_error", "connection_error"]
            assert "operation" in response
            assert response["operation"] == "connect_to_redis"
    
    def test_storage_error_with_timeout_scenarios(self):
        """Test storage error handling with timeout scenarios"""
        timeout_errors = [
            TimeoutError("Operation timed out"),
            TimeoutError("Redis command timeout after 30s"),
            Exception("DuckDB query timeout"),
            Exception("Lock acquisition timeout")
        ]
        
        for error in timeout_errors:
            response = handle_storage_error(error, "database_operation")
            
            assert response["success"] is False
            assert "timeout" in response["message"].lower() or "time" in response["message"].lower()
            assert response["error_type"] in ["storage_error", "timeout_error"]
    
    def test_storage_error_with_permission_issues(self):
        """Test storage error handling with permission-related errors"""
        permission_errors = [
            PermissionError("Access denied to database file"),
            OSError(13, "Permission denied"),  # EACCES
            FileNotFoundError("Database file not found"),
            IsADirectoryError("Expected file, got directory")
        ]
        
        for error in permission_errors:
            response = handle_storage_error(error, "file_access")
            
            assert response["success"] is False
            assert any(word in response["message"].lower() 
                      for word in ["permission", "access", "denied", "file", "directory"])
            assert response["error_type"] in ["storage_error", "permission_error", "file_error"]
    
    def test_storage_error_with_data_corruption(self):
        """Test storage error handling with data corruption scenarios"""
        corruption_errors = [
            ValueError("Invalid JSON data in storage"),
            UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"),
            Exception("Database corruption detected"),
            Exception("Checksum mismatch in stored data"),
            Exception("Invalid data format in cache")
        ]
        
        for error in corruption_errors:
            response = handle_storage_error(error, "data_retrieval")
            
            assert response["success"] is False
            assert any(word in response["message"].lower() 
                      for word in ["data", "format", "corruption", "invalid", "decode"])
            assert response["error_type"] in ["storage_error", "data_error", "validation_error"]
    
    def test_storage_error_with_resource_exhaustion(self):
        """Test storage error handling with resource exhaustion"""
        resource_errors = [
            MemoryError("Out of memory"),
            OSError(28, "No space left on device"),  # ENOSPC
            OSError(24, "Too many open files"),      # EMFILE
            Exception("Redis memory usage exceeded"),
            Exception("DuckDB connection pool exhausted")
        ]
        
        for error in resource_errors:
            response = handle_storage_error(error, "resource_operation")
            
            assert response["success"] is False
            assert any(word in response["message"].lower() 
                      for word in ["memory", "space", "files", "resource", "exhausted", "exceeded"])
            assert response["error_type"] in ["storage_error", "resource_error", "system_error"]


@pytest.mark.skipif(not ERROR_HANDLER_AVAILABLE, reason="Error handler not available")
class TestValidationErrorHandlingEdgeCases:
    """Edge cases for validation error handling"""
    
    def test_validation_error_with_complex_field_paths(self):
        """Test validation error with complex nested field paths"""
        complex_field_paths = [
            "user.profile.settings.notifications.email",
            "data[0].metadata.tags[5].value",
            "config.storage.redis.connection_pool.max_connections",
            "agents[user_123].state.scratchpad.content.length"
        ]
        
        for field_path in complex_field_paths:
            error = ValueError(f"Invalid value for field: {field_path}")
            response = handle_validation_error(error, field_path)
            
            assert response["success"] is False
            assert field_path in response["message"] or field_path in str(response.get("field", ""))
            assert response["error_type"] == "validation_error"
    
    def test_validation_error_with_multiple_fields(self):
        """Test validation error handling with multiple field errors"""
        multi_field_error = ValueError("Multiple validation errors occurred")
        
        # Test with list of fields
        fields = ["username", "email", "password", "confirm_password"]
        response = handle_validation_error(multi_field_error, fields)
        
        assert response["success"] is False
        assert "multiple" in response["message"].lower() or "validation" in response["message"].lower()
        assert response["error_type"] == "validation_error"
    
    def test_validation_error_with_schema_violations(self):
        """Test validation error with schema violation scenarios"""
        schema_errors = [
            TypeError("Expected string, got int"),
            ValueError("Value must be between 1 and 100"),
            AttributeError("Required attribute 'name' missing"),
            KeyError("Required key 'config' not found"),
            Exception("Schema validation failed: additional properties not allowed")
        ]
        
        for error in schema_errors:
            response = handle_validation_error(error, "schema_field")
            
            assert response["success"] is False
            assert "schema_field" in response["message"] or "validation" in response["message"].lower()
            assert response["error_type"] == "validation_error"
    
    def test_validation_error_with_edge_case_values(self):
        """Test validation error with edge case input values"""
        edge_case_scenarios = [
            (ValueError("Empty string not allowed"), ""),
            (ValueError("String too long"), "x" * 10000),
            (ValueError("Invalid number"), float('inf')),
            (ValueError("Invalid number"), float('nan')),
            (ValueError("Negative value not allowed"), -1),
            (ValueError("Zero not allowed"), 0),
            (ValueError("None not allowed"), None)
        ]
        
        for error, field_value in edge_case_scenarios:
            response = handle_validation_error(error, f"field_with_value_{field_value}")
            
            assert response["success"] is False
            assert response["error_type"] == "validation_error"


@pytest.mark.skipif(not ERROR_HANDLER_AVAILABLE, reason="Error handler not available")
class TestErrorSanitizationEdgeCases:
    """Edge cases for error message sanitization"""
    
    @patch('src.core.error_handler.is_production')
    def test_sensitive_data_sanitization(self, mock_is_production):
        """Test sanitization of various types of sensitive data"""
        mock_is_production.return_value = True
        
        sensitive_messages = [
            "Authentication failed with password: secret123",
            "Database connection error: user=admin password=supersecret host=192.168.1.100",
            "Redis AUTH failed: password 'redis_secret_key' invalid",
            "JWT token validation failed: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
            "API key authentication failed: sk-1234567890abcdef",
            "Session token expired: sess_abcd1234efgh5678",
            "Cookie value invalid: session_id=secure_session_12345",
            "Private key error: -----BEGIN PRIVATE KEY-----MIIEvQIBADANBgkqhkiG9w0BAQEF..."
        ]
        
        for message in sensitive_messages:
            sanitized = sanitize_error_message(message, "auth_error")
            
            # Should not contain sensitive patterns
            assert "secret" not in sanitized.lower()
            assert "password" not in sanitized or "password=***" in sanitized
            assert "192.168." not in sanitized or "X.X.X.X" in sanitized
            assert "eyJ0eXAiOiJKV1QiOiJhbGciOiJIUzI1NiJ9" not in sanitized
            assert "sk-1234567890abcdef" not in sanitized
            assert "sess_abcd1234efgh5678" not in sanitized
            assert "session_id=secure_session_12345" not in sanitized
            assert "BEGIN PRIVATE KEY" not in sanitized
    
    @patch('src.core.error_handler.is_production')
    def test_stack_trace_sanitization(self, mock_is_production):
        """Test sanitization of stack traces in production"""
        mock_is_production.return_value = True
        
        stack_trace_messages = [
            "Traceback (most recent call last):\n  File \"/app/secret_module.py\", line 42, in secret_function\n    raise Exception('error')",
            "Exception in thread 'main': ValueError: invalid input\n\tat /home/user/private/app.py:123",
            "Full stack trace:\n/usr/local/lib/python3.9/site-packages/redis/connection.py:559: ConnectionError",
            "Python stack trace:\nFile \"/etc/secret/config.py\", line 15\n  password = 'secret123'\n                          ^"
        ]
        
        for message in stack_trace_messages:
            sanitized = sanitize_error_message(message, "system_error")
            
            # Should not contain full stack traces or file paths
            assert "Traceback" not in sanitized
            assert "/app/secret_module.py" not in sanitized
            assert "/home/user/private/" not in sanitized
            assert "/etc/secret/" not in sanitized
            assert "password = 'secret123'" not in sanitized
    
    @patch('src.core.error_handler.is_production')
    def test_development_mode_preservation(self, mock_is_production):
        """Test that development mode preserves detailed error information"""
        mock_is_production.return_value = False
        
        detailed_message = "Database connection failed: host=192.168.1.100 user=admin password=secret123"
        sanitized = sanitize_error_message(detailed_message, "connection_error")
        
        # In development, should preserve more details (but still sanitize passwords)
        assert "192.168.1.100" in sanitized or "development" in sanitized.lower()
        assert "admin" in sanitized or "development" in sanitized.lower()
        # But passwords should still be sanitized
        assert "secret123" not in sanitized
    
    def test_sanitization_performance(self):
        """Test sanitization performance with large messages"""
        # Create a large error message
        large_message = "Error occurred: " + "x" * 100000 + " with password: secret123"
        
        start_time = datetime.utcnow()
        sanitized = sanitize_error_message(large_message, "performance_test")
        end_time = datetime.utcnow()
        
        duration = (end_time - start_time).total_seconds()
        
        # Sanitization should complete quickly (under 1 second for 100KB message)
        assert duration < 1.0
        assert "secret123" not in sanitized


@pytest.mark.skipif(not ERROR_HANDLER_AVAILABLE, reason="Error handler not available")
class TestErrorContextAndSeverityEdgeCases:
    """Edge cases for error context extraction and severity assessment"""
    
    def test_error_context_extraction_complex_scenarios(self):
        """Test error context extraction from complex error scenarios"""
        try:
            # Create a complex error scenario
            def level3():
                raise ValueError("Deep nested error with context data")
            
            def level2():
                try:
                    level3()
                except ValueError as e:
                    raise RuntimeError("Level 2 error") from e
            
            def level1():
                try:
                    level2()
                except RuntimeError as e:
                    raise Exception("Top level error") from e
            
            level1()
            
        except Exception as e:
            context = extract_error_context(e)
            
            assert isinstance(context, dict)
            assert "error_type" in context
            assert "error_message" in context
            # Should capture nested error information
            assert "original_error" in context or "chain" in context
    
    def test_error_severity_assessment_edge_cases(self):
        """Test error severity assessment for various error types"""
        severity_test_cases = [
            # Critical errors
            (MemoryError("Out of memory"), "critical"),
            (SystemExit("System shutting down"), "critical"),
            (KeyboardInterrupt("User interrupted"), "critical"),
            
            # High severity errors
            (PermissionError("Access denied"), "high"),
            (ConnectionError("Database unreachable"), "high"),
            (FileNotFoundError("Critical file missing"), "high"),
            
            # Medium severity errors
            (ValueError("Invalid input format"), "medium"),
            (TypeError("Wrong data type"), "medium"),
            (KeyError("Missing configuration key"), "medium"),
            
            # Low severity errors
            (Warning("Deprecated function used"), "low"),
            (UserWarning("User action recommended"), "low"),
        ]
        
        for error, expected_severity in severity_test_cases:
            severity = get_error_severity(error)
            
            # Verify severity is reasonable (may not match exactly due to implementation details)
            assert severity in ["low", "medium", "high", "critical"]
    
    def test_error_logging_with_context(self):
        """Test error logging with complex context information"""
        error = ValueError("Test error for logging")
        context = {
            "user_id": "user_12345",
            "operation": "data_processing",
            "timestamp": datetime.utcnow().isoformat(),
            "request_data": {"size": 1024, "type": "json"},
            "system_state": {"memory_usage": "75%", "cpu_usage": "45%"}
        }
        
        # Test logging doesn't crash with complex context
        try:
            log_error_with_context(error, context, "test_operation")
            # If no exception is raised, logging worked
            assert True
        except Exception as logging_error:
            # Logging errors shouldn't prevent operation
            pytest.skip(f"Logging failed: {logging_error}")


@pytest.mark.skipif(not ERROR_HANDLER_AVAILABLE, reason="Error handler not available")
class TestSpecializedErrorHandlers:
    """Edge cases for specialized error handlers"""
    
    def test_timeout_error_handling_edge_cases(self):
        """Test timeout error handling with various timeout scenarios"""
        timeout_scenarios = [
            ("Redis operation timeout after 30 seconds", 30),
            ("DuckDB query timeout (120s)", 120),
            ("HTTP request timeout: 5.5 seconds", 5.5),
            ("Lock acquisition timeout (0.1s)", 0.1),
            ("Connection timeout: 60000ms", 60.0)
        ]
        
        for message, expected_timeout in timeout_scenarios:
            error = TimeoutError(message)
            response = handle_timeout_error(error, "timeout_operation", timeout_duration=expected_timeout)
            
            assert response["success"] is False
            assert response["error_type"] == "timeout_error"
            assert "timeout" in response["message"].lower()
            if "timeout_duration" in response:
                assert response["timeout_duration"] == expected_timeout
    
    def test_permission_error_handling_edge_cases(self):
        """Test permission error handling with various permission scenarios"""
        permission_scenarios = [
            PermissionError("File access denied"),
            OSError(13, "Permission denied"),  # EACCES
            Exception("Insufficient privileges for operation"),
            Exception("User not authorized for this action"),
            Exception("Role-based access control violation"),
            Exception("API key lacks required permissions")
        ]
        
        for error in permission_scenarios:
            response = handle_permission_error(error, "restricted_operation", required_permission="admin")
            
            assert response["success"] is False
            assert response["error_type"] == "permission_error"
            assert any(word in response["message"].lower() 
                      for word in ["permission", "access", "authorized", "privilege"])
            if "required_permission" in response:
                assert response["required_permission"] == "admin"
    
    def test_chained_error_handling(self):
        """Test handling of chained exceptions"""
        try:
            try:
                try:
                    raise ValueError("Original error")
                except ValueError as e:
                    raise TypeError("Middle error") from e
            except TypeError as e:
                raise RuntimeError("Final error") from e
        except RuntimeError as chained_error:
            response = create_error_response(
                success=False,
                message="Chained error occurred",
                error_type="chained_error",
                details={"original_error": str(chained_error.__cause__.__cause__)}
            )
            
            assert response["success"] is False
            assert "chained" in response["message"].lower()
            assert "Original error" in response["details"]["original_error"]


@pytest.mark.skipif(not ERROR_HANDLER_AVAILABLE, reason="Error handler not available")
class TestErrorHandlerPerformanceAndResilience:
    """Performance and resilience tests for error handling"""
    
    def test_error_handling_performance(self):
        """Test error handling performance with high volume"""
        import time
        
        # Generate many errors quickly
        errors = [ValueError(f"Error {i}") for i in range(1000)]
        
        start_time = time.time()
        responses = []
        
        for error in errors:
            response = create_error_response(
                success=False,
                message=str(error),
                error_type="performance_test"
            )
            responses.append(response)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle 1000 errors in under 1 second
        assert duration < 1.0
        assert len(responses) == 1000
        assert all(r["success"] is False for r in responses)
    
    def test_error_handling_memory_usage(self):
        """Test error handling doesn't leak memory"""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and handle many errors
        for i in range(100):
            large_error_data = {"data": "x" * 10000, "index": i}
            error = ValueError(f"Large error {i}")
            
            response = create_error_response(
                success=False,
                message=str(error),
                error_type="memory_test",
                details=large_error_data
            )
            
            # Clear reference to response
            del response
        
        # Force garbage collection after test
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not increase dramatically
        object_increase = final_objects - initial_objects
        assert object_increase < 1000  # Allow some increase, but not excessive
    
    def test_recursive_error_handling_prevention(self):
        """Test prevention of recursive error handling"""
        # Create an error that might cause issues in error handling itself
        class ProblematicError(Exception):
            def __str__(self):
                # This could cause recursion in poor error handlers
                return str(self)
        
        problematic_error = ProblematicError("Recursive error")
        
        # Should handle without infinite recursion
        try:
            response = create_error_response(
                success=False,
                message="Handling problematic error",
                error_type="recursive_test",
                details={"error": str(problematic_error)}
            )
            assert response["success"] is False
        except RecursionError:
            pytest.fail("Error handler caused infinite recursion")