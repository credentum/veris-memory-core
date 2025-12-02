#!/usr/bin/env python3
"""
Test suite for config_error.py - Configuration error handling tests
"""
import pytest

# Import the module under test
from src.core.config_error import (
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError
)


class TestConfigurationError:
    """Test suite for ConfigurationError base class"""

    def test_configuration_error_minimal_init(self):
        """Test ConfigurationError initialization with minimal parameters"""
        error = ConfigurationError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.config_path is None
        assert error.details == {}
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationError)

    def test_configuration_error_with_config_path(self):
        """Test ConfigurationError initialization with config path"""
        config_path = "/path/to/config.yaml"
        error = ConfigurationError("Test error message", config_path=config_path)
        
        assert str(error) == "Test error message"
        assert error.config_path == config_path
        assert error.details == {}

    def test_configuration_error_with_details(self):
        """Test ConfigurationError initialization with details"""
        details = {"field": "value", "error_code": 123}
        error = ConfigurationError("Test error message", details=details)
        
        assert str(error) == "Test error message"
        assert error.config_path is None
        assert error.details == details

    def test_configuration_error_complete_init(self):
        """Test ConfigurationError initialization with all parameters"""
        config_path = "/path/to/config.yaml"
        details = {"validation_errors": ["field1", "field2"], "line_number": 42}
        error = ConfigurationError(
            "Complete error message",
            config_path=config_path,
            details=details
        )
        
        assert str(error) == "Complete error message"
        assert error.config_path == config_path
        assert error.details == details

    def test_configuration_error_none_details(self):
        """Test ConfigurationError with None details (should default to empty dict)"""
        error = ConfigurationError("Test error", details=None)
        
        assert error.details == {}

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance hierarchy"""
        error = ConfigurationError("Test error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationError)
        assert issubclass(ConfigurationError, Exception)

    def test_configuration_error_message_types(self):
        """Test ConfigurationError with different message types"""
        # String message
        error1 = ConfigurationError("String message")
        assert str(error1) == "String message"
        
        # Empty string message
        error2 = ConfigurationError("")
        assert str(error2) == ""
        
        # Multiline message
        multiline_msg = "Line 1\nLine 2\nLine 3"
        error3 = ConfigurationError(multiline_msg)
        assert str(error3) == multiline_msg


class TestConfigFileNotFoundError:
    """Test suite for ConfigFileNotFoundError"""

    def test_config_file_not_found_error_init(self):
        """Test ConfigFileNotFoundError initialization"""
        config_path = "/path/to/missing/config.yaml"
        error = ConfigFileNotFoundError(config_path)
        
        expected_message = f"Configuration file not found: {config_path}"
        assert str(error) == expected_message
        assert error.config_path == config_path
        assert error.details == {}

    def test_config_file_not_found_error_inheritance(self):
        """Test ConfigFileNotFoundError inheritance"""
        error = ConfigFileNotFoundError("/path/to/config.yaml")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, ConfigFileNotFoundError)
        assert issubclass(ConfigFileNotFoundError, ConfigurationError)

    def test_config_file_not_found_error_various_paths(self):
        """Test ConfigFileNotFoundError with various file paths"""
        test_paths = [
            "/absolute/path/config.yaml",
            "relative/path/config.yaml",
            "config.yaml",
            "/path/with spaces/config.yaml",
            "/path/with-dashes/config.yaml",
            "/path/with_underscores/config.yaml",
            "~/home/config.yaml"
        ]
        
        for path in test_paths:
            error = ConfigFileNotFoundError(path)
            expected_message = f"Configuration file not found: {path}"
            assert str(error) == expected_message
            assert error.config_path == path

    def test_config_file_not_found_error_empty_path(self):
        """Test ConfigFileNotFoundError with empty path"""
        error = ConfigFileNotFoundError("")
        
        expected_message = "Configuration file not found: "
        assert str(error) == expected_message
        assert error.config_path == ""


class TestConfigParseError:
    """Test suite for ConfigParseError"""

    def test_config_parse_error_init(self):
        """Test ConfigParseError initialization"""
        config_path = "/path/to/config.yaml"
        parse_error = "Invalid YAML syntax at line 5"
        error = ConfigParseError(config_path, parse_error)
        
        expected_message = f"Failed to parse configuration file {config_path}: {parse_error}"
        assert str(error) == expected_message
        assert error.config_path == config_path
        assert error.details == {"parse_error": parse_error}

    def test_config_parse_error_inheritance(self):
        """Test ConfigParseError inheritance"""
        error = ConfigParseError("/path/config.yaml", "Parse error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, ConfigParseError)
        assert issubclass(ConfigParseError, ConfigurationError)

    def test_config_parse_error_various_parse_errors(self):
        """Test ConfigParseError with various parse error messages"""
        config_path = "/test/config.yaml"
        parse_errors = [
            "Invalid YAML syntax",
            "Unexpected character at line 10, column 5",
            "Missing closing bracket",
            "Duplicate key 'database' found",
            "Invalid indentation",
            "Malformed list structure",
            ""  # Empty parse error
        ]
        
        for parse_error in parse_errors:
            error = ConfigParseError(config_path, parse_error)
            expected_message = f"Failed to parse configuration file {config_path}: {parse_error}"
            assert str(error) == expected_message
            assert error.config_path == config_path
            assert error.details == {"parse_error": parse_error}

    def test_config_parse_error_special_characters(self):
        """Test ConfigParseError with special characters in paths and messages"""
        config_path = "/path/with/special chars!/config.yaml"
        parse_error = "Error with symbols: @#$%^&*()"
        error = ConfigParseError(config_path, parse_error)
        
        expected_message = f"Failed to parse configuration file {config_path}: {parse_error}"
        assert str(error) == expected_message
        assert error.config_path == config_path
        assert error.details == {"parse_error": parse_error}

    def test_config_parse_error_multiline_error(self):
        """Test ConfigParseError with multiline parse error"""
        config_path = "/config.yaml"
        parse_error = "Multiple errors found:\n  - Line 1: Invalid key\n  - Line 5: Missing value"
        error = ConfigParseError(config_path, parse_error)
        
        expected_message = f"Failed to parse configuration file {config_path}: {parse_error}"
        assert str(error) == expected_message
        assert error.details == {"parse_error": parse_error}


class TestConfigValidationError:
    """Test suite for ConfigValidationError"""

    def test_config_validation_error_minimal_init(self):
        """Test ConfigValidationError initialization with minimal parameters"""
        message = "Validation failed"
        error = ConfigValidationError(message)
        
        assert str(error) == message
        assert error.config_path is None
        assert error.details == {}

    def test_config_validation_error_with_invalid_fields(self):
        """Test ConfigValidationError initialization with invalid fields"""
        message = "Multiple validation errors"
        invalid_fields = ["database.host", "redis.port", "security.ssl"]
        error = ConfigValidationError(message, invalid_fields=invalid_fields)
        
        assert str(error) == message
        assert error.config_path is None
        assert error.details == {"invalid_fields": invalid_fields}

    def test_config_validation_error_inheritance(self):
        """Test ConfigValidationError inheritance"""
        error = ConfigValidationError("Validation error")
        
        assert isinstance(error, Exception)
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, ConfigValidationError)
        assert issubclass(ConfigValidationError, ConfigurationError)

    def test_config_validation_error_none_invalid_fields(self):
        """Test ConfigValidationError with None invalid_fields"""
        error = ConfigValidationError("Validation error", invalid_fields=None)
        
        assert error.details == {}

    def test_config_validation_error_empty_invalid_fields(self):
        """Test ConfigValidationError with empty invalid_fields list"""
        error = ConfigValidationError("Validation error", invalid_fields=[])
        
        assert error.details == {}

    def test_config_validation_error_various_validation_scenarios(self):
        """Test ConfigValidationError with various validation scenarios"""
        test_cases = [
            ("Missing required field: database.host", ["database.host"]),
            ("Invalid port numbers", ["neo4j.port", "redis.port", "api.port"]),
            ("SSL configuration errors", ["neo4j.ssl", "redis.ssl"]),
            ("Multiple validation failures", ["field1", "field2", "field3", "field4"]),
            ("Single field validation", ["single_field"]),
            ("Complex nested field paths", ["config.database.neo4j.auth.username"])
        ]
        
        for message, invalid_fields in test_cases:
            error = ConfigValidationError(message, invalid_fields=invalid_fields)
            assert str(error) == message
            assert error.details == {"invalid_fields": invalid_fields}

    def test_config_validation_error_field_names_with_special_chars(self):
        """Test ConfigValidationError with special characters in field names"""
        invalid_fields = [
            "field-with-dashes",
            "field_with_underscores",
            "field.with.dots",
            "field/with/slashes",
            "field@with@symbols"
        ]
        error = ConfigValidationError("Special field names", invalid_fields=invalid_fields)
        
        assert error.details == {"invalid_fields": invalid_fields}


class TestConfigErrorIntegration:
    """Integration tests for configuration error classes"""

    def test_exception_catching_hierarchy(self):
        """Test that all config errors can be caught by base ConfigurationError"""
        errors = [
            ConfigurationError("Base error"),
            ConfigFileNotFoundError("/missing/config.yaml"),
            ConfigParseError("/config.yaml", "Parse error"),
            ConfigValidationError("Validation error", ["field1"])
        ]
        
        for error in errors:
            # All should be catchable as ConfigurationError
            with pytest.raises(ConfigurationError):
                raise error
            
            # All should be catchable as Exception
            with pytest.raises(Exception):
                raise error

    def test_error_message_formatting_consistency(self):
        """Test consistent error message formatting across error types"""
        # Test that all errors produce meaningful string representations
        errors = [
            ConfigurationError("Generic config error"),
            ConfigFileNotFoundError("/path/to/config.yaml"),
            ConfigParseError("/path/to/config.yaml", "YAML syntax error"),
            ConfigValidationError("Field validation failed", ["invalid_field"])
        ]
        
        for error in errors:
            error_str = str(error)
            assert len(error_str) > 0
            assert isinstance(error_str, str)
            # Should not contain raw object representations
            assert not error_str.startswith("<")
            assert not error_str.endswith(">")

    def test_error_attributes_preservation(self):
        """Test that error attributes are preserved correctly"""
        config_path = "/test/config.yaml"
        details = {"custom": "data", "numbers": [1, 2, 3]}
        
        # Create error with all attributes
        error = ConfigurationError(
            "Test message",
            config_path=config_path,
            details=details
        )
        
        # Verify attributes are preserved
        assert error.config_path == config_path
        assert error.details == details
        assert str(error) == "Test message"
        
        # Test with subclass
        parse_error = ConfigParseError(config_path, "Parse issue")
        assert parse_error.config_path == config_path
        assert parse_error.details["parse_error"] == "Parse issue"

    def test_real_world_error_scenarios(self):
        """Test realistic error scenarios"""
        # File not found scenario
        try:
            raise ConfigFileNotFoundError("/app/.ctxrc.yaml")
        except ConfigurationError as e:
            assert "/app/.ctxrc.yaml" in str(e)
            assert e.config_path == "/app/.ctxrc.yaml"
        
        # Parse error scenario
        try:
            raise ConfigParseError("/app/config.yaml", "line 15: invalid character '!' at column 8")
        except ConfigurationError as e:
            assert "line 15" in str(e)
            assert "invalid character" in str(e)
            assert e.config_path == "/app/config.yaml"
        
        # Validation error scenario
        try:
            invalid_fields = ["database.neo4j.uri", "redis.password"]
            raise ConfigValidationError("Configuration validation failed", invalid_fields)
        except ConfigurationError as e:
            assert "validation failed" in str(e).lower()
            assert e.details["invalid_fields"] == invalid_fields

    def test_error_serialization_compatibility(self):
        """Test that errors work well with serialization/logging"""
        errors = [
            ConfigFileNotFoundError("/config.yaml"),
            ConfigParseError("/config.yaml", "YAML error"),
            ConfigValidationError("Validation error", ["field1", "field2"])
        ]
        
        for error in errors:
            # Should be representable
            repr_str = repr(error)
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0
            
            # Should be convertible to string
            str_repr = str(error)
            assert isinstance(str_repr, str)
            assert len(str_repr) > 0
            
            # Should have accessible attributes
            assert hasattr(error, 'config_path')
            assert hasattr(error, 'details')