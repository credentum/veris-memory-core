#!/usr/bin/env python3
"""
Simple test suite for error_handler.py - Phase 2 Coverage Improvement
"""
import pytest
import logging
import os
from unittest.mock import patch, Mock
from typing import Dict, Any

from src.core.error_handler import (
    is_production,
    sanitize_error_message,
    create_error_response,
    handle_storage_error,
    handle_validation_error,
    handle_generic_error
)


class TestIsProduction:
    """Test suite for is_production function"""
    
    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_is_production_true_with_production(self):
        """Test is_production returns True for 'production' environment"""
        assert is_production() is True
    
    @patch.dict(os.environ, {"ENVIRONMENT": "prod"})
    def test_is_production_true_with_prod(self):
        """Test is_production returns True for 'prod' environment"""
        assert is_production() is True
    
    @patch.dict(os.environ, {"ENVIRONMENT": "PRODUCTION"})
    def test_is_production_case_insensitive(self):
        """Test is_production is case insensitive"""
        assert is_production() is True
    
    @patch.dict(os.environ, {"ENVIRONMENT": "development"})
    def test_is_production_false_with_development(self):
        """Test is_production returns False for 'development' environment"""
        assert is_production() is False
    
    @patch.dict(os.environ, {"ENVIRONMENT": "test"})
    def test_is_production_false_with_test(self):
        """Test is_production returns False for 'test' environment"""
        assert is_production() is False
    
    @patch.dict(os.environ, {}, clear=True)
    def test_is_production_default_development(self):
        """Test is_production defaults to False when ENVIRONMENT not set"""
        assert is_production() is False


class TestSanitizeErrorMessage:
    """Test suite for sanitize_error_message function"""
    
    @patch('src.core.error_handler.is_production')
    def test_sanitize_error_message_development_passthrough(self, mock_is_prod):
        """Test error message passes through unchanged in development"""
        mock_is_prod.return_value = False
        
        error_msg = "Database connection failed with password secret123"
        result = sanitize_error_message(error_msg, "storage_error")
        
        assert result == error_msg
        assert "secret123" in result
    
    @patch('src.core.error_handler.is_production')
    def test_sanitize_error_message_production_password(self, mock_is_prod):
        """Test password sanitization in production"""
        mock_is_prod.return_value = True
        
        error_msg = "Authentication failed with password=secret123"
        result = sanitize_error_message(error_msg, "storage_error")
        
        assert "secret123" not in result
        assert "password=***" in result
    
    @patch('src.core.error_handler.is_production')
    def test_sanitize_error_message_production_token(self, mock_is_prod):
        """Test token sanitization in production"""
        mock_is_prod.return_value = True
        
        error_msg = "API call failed with token: abc123xyz"
        result = sanitize_error_message(error_msg, "storage_error")
        
        assert "abc123xyz" not in result
        assert "token=***" in result
    
    @patch('src.core.error_handler.is_production')
    def test_sanitize_error_message_production_ip_address(self, mock_is_prod):
        """Test IP address sanitization in production"""
        mock_is_prod.return_value = True
        
        error_msg = "Connection failed to 192.168.1.100"
        result = sanitize_error_message(error_msg, "storage_error")
        
        assert "192.168.1.100" not in result
        assert "X.X.X.X" in result
    
    @patch('src.core.error_handler.is_production')
    def test_sanitize_error_message_generic_fallback(self, mock_is_prod):
        """Test generic message fallback for sensitive content"""
        mock_is_prod.return_value = True
        
        error_msg = "Traceback (most recent call last): Exception occurred"
        result = sanitize_error_message(error_msg, "unexpected_error")
        
        assert "Traceback" not in result
        assert result == "Internal server error"
    
    @patch('src.core.error_handler.is_production')
    def test_sanitize_error_message_storage_generic(self, mock_is_prod):
        """Test storage-specific generic messages"""
        mock_is_prod.return_value = True
        
        error_msg = "Exception: Database connection stack trace error"
        result = sanitize_error_message(error_msg, "storage_unavailable")
        
        assert result == "Storage service temporarily unavailable"


class TestCreateErrorResponse:
    """Test suite for create_error_response function"""
    
    @patch('src.core.error_handler.logger')
    def test_create_error_response_basic(self, mock_logger):
        """Test basic error response creation"""
        response = create_error_response(
            success=False,
            message="Test error",
            error_type="test_error"
        )
        
        assert response["success"] is False
        assert response["message"] == "Test error"
        assert response["error_type"] == "test_error"
        mock_logger.error.assert_called_once()
    
    @patch('src.core.error_handler.logger')
    @patch('src.core.error_handler.is_production')
    def test_create_error_response_with_details_development(self, mock_is_prod, mock_logger):
        """Test error response with details in development"""
        mock_is_prod.return_value = False
        
        response = create_error_response(
            success=False,
            message="Test error",
            error_type="test_error",
            error_details="Detailed error information"
        )
        
        assert "error_details" in response
        assert response["error_details"] == "Detailed error information"
        mock_logger.error.assert_called_once()
    
    @patch('src.core.error_handler.logger')
    @patch('src.core.error_handler.is_production')
    def test_create_error_response_with_details_production(self, mock_is_prod, mock_logger):
        """Test error response without details in production"""
        mock_is_prod.return_value = True
        
        response = create_error_response(
            success=False,
            message="Test error",
            error_type="test_error",
            error_details="Detailed error information"
        )
        
        assert "error_details" not in response
        mock_logger.error.assert_called_once()
    
    def test_create_error_response_with_kwargs(self):
        """Test error response with additional keyword arguments"""
        response = create_error_response(
            success=False,
            message="Test error",
            error_type="test_error",
            custom_field="custom_value",
            status_code=500
        )
        
        assert response["custom_field"] == "custom_value"
        assert response["status_code"] == 500


class TestHandleStorageError:
    """Test suite for handle_storage_error function"""
    
    @patch('src.core.error_handler.create_error_response')
    def test_handle_storage_error_with_exception(self, mock_create_response):
        """Test handle_storage_error with Exception object"""
        mock_create_response.return_value = {"success": False, "message": "handled"}
        
        exception = ConnectionError("Database connection failed")
        result = handle_storage_error(exception, "connect")
        
        mock_create_response.assert_called_once()
        assert result["success"] is False
    
    def test_handle_storage_error_response_structure(self):
        """Test handle_storage_error returns proper response structure"""
        exception = ValueError("Storage validation error")
        result = handle_storage_error(exception, "validate")
        
        assert "success" in result
        assert "message" in result
        assert "error_type" in result
        assert result["success"] is False
    
    def test_handle_storage_error_different_operations(self):
        """Test handle_storage_error with different operation types"""
        operations = ["read", "write", "delete", "connect", "query"]
        
        for operation in operations:
            exception = RuntimeError(f"Error in {operation}")
            result = handle_storage_error(exception, operation)
            
            assert result["success"] is False
            assert operation in result["message"] or "storage" in result["message"].lower()


class TestHandleValidationError:
    """Test suite for handle_validation_error function"""
    
    def test_handle_validation_error_basic(self):
        """Test basic validation error handling"""
        exception = ValueError("Invalid input format")
        result = handle_validation_error(exception, "user_input")
        
        assert result["success"] is False
        assert "message" in result
        assert "error_type" in result
        assert result["error_type"] == "validation_error"
    
    def test_handle_validation_error_field_specific(self):
        """Test validation error with specific field names"""
        fields = ["email", "password", "age", "username"]
        
        for field in fields:
            exception = ValueError(f"Invalid {field}")
            result = handle_validation_error(exception, field)
            
            assert result["success"] is False
            assert field in result["message"] or "validation" in result["message"].lower()
    
    @patch('src.core.error_handler.logger')
    def test_handle_validation_error_logging(self, mock_logger):
        """Test that validation errors are logged properly"""
        exception = ValueError("Validation failed")
        result = handle_validation_error(exception, "test_field")
        
        mock_logger.error.assert_called_once()


class TestHandleGenericError:
    """Test suite for handle_generic_error function"""
    
    def test_handle_generic_error_basic(self):
        """Test basic generic error handling"""
        exception = RuntimeError("Something went wrong")
        result = handle_generic_error(exception, "test_operation")
        
        assert result["success"] is False
        assert "message" in result
        assert "error_type" in result
    
    def test_handle_generic_error_different_exceptions(self):
        """Test generic error handling with different exception types"""
        exceptions = [
            ValueError("Value error"),
            TypeError("Type error"),
            RuntimeError("Runtime error"),
            ConnectionError("Connection error")
        ]
        
        for exception in exceptions:
            result = handle_generic_error(exception, "test_op")
            
            assert result["success"] is False
            assert str(exception) in result["message"] or "error" in result["message"].lower()
    
    def test_handle_generic_error_operation_context(self):
        """Test generic error handling preserves operation context"""
        operations = ["data_processing", "file_upload", "user_authentication"]
        
        for operation in operations:
            exception = Exception(f"Error in {operation}")
            result = handle_generic_error(exception, operation)
            
            assert result["success"] is False
            # Operation should be referenced in the response somehow
            assert operation in str(result) or "operation" in result


class TestErrorHandlerIntegration:
    """Integration tests for error handler components"""
    
    @patch('src.core.error_handler.is_production')
    def test_end_to_end_error_handling_development(self, mock_is_prod):
        """Test complete error handling flow in development"""
        mock_is_prod.return_value = False
        
        # Simulate an error occurring
        try:
            raise ConnectionError("Database connection failed to 192.168.1.1:5432")
        except Exception as e:
            response = handle_storage_error(e, "connection")
        
        assert response["success"] is False
        assert "192.168.1.1" in response["message"]  # Not sanitized in dev
    
    @patch('src.core.error_handler.is_production')
    def test_end_to_end_error_handling_production(self, mock_is_prod):
        """Test complete error handling flow in production"""
        mock_is_prod.return_value = True
        
        # Simulate an error occurring
        try:
            raise ConnectionError("Database connection failed to 192.168.1.1:5432")
        except Exception as e:
            response = handle_storage_error(e, "connection")
        
        assert response["success"] is False
        # IP should be sanitized in production
        assert "192.168.1.1" not in response["message"] or "X.X.X.X" in response["message"]
    
    def test_different_error_handlers_consistency(self):
        """Test that different error handlers return consistent response structure"""
        exception = ValueError("Test error")
        
        storage_response = handle_storage_error(exception, "test")
        validation_response = handle_validation_error(exception, "test")
        generic_response = handle_generic_error(exception, "test")
        
        # All should have the same basic structure
        for response in [storage_response, validation_response, generic_response]:
            assert "success" in response
            assert "message" in response
            assert "error_type" in response
            assert response["success"] is False


class TestErrorHandlerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_sanitize_empty_message(self):
        """Test sanitizing empty messages"""
        assert sanitize_error_message("", "test") == ""
    
    def test_create_error_response_no_message(self):
        """Test creating error response without message"""
        response = create_error_response(error_type="test_error")
        
        assert response["success"] is False
        assert response["message"] == ""
        assert response["error_type"] == "test_error"
    
    def test_handle_storage_error_none_exception(self):
        """Test handle_storage_error with minimal input"""
        # Test with basic exception
        exception = Exception()
        result = handle_storage_error(exception)
        
        assert result["success"] is False
        assert "message" in result
    
    @patch('src.core.error_handler.is_production')
    def test_sanitize_with_regex_special_characters(self, mock_is_prod):
        """Test sanitization handles regex special characters properly"""
        mock_is_prod.return_value = True
        
        # Test with special regex characters in the error message
        error_msg = "Password is: test[123]+()*?"
        result = sanitize_error_message(error_msg, "auth_error")
        
        # Should not crash due to regex special characters
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_all_error_handlers_with_empty_operation(self):
        """Test all error handlers work with empty operation strings"""
        exception = ValueError("Test error")
        
        # All should handle empty operation gracefully
        storage_result = handle_storage_error(exception, "")
        validation_result = handle_validation_error(exception, "")
        generic_result = handle_generic_error(exception, "")
        
        for result in [storage_result, validation_result, generic_result]:
            assert result["success"] is False
            assert "message" in result