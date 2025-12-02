#!/usr/bin/env python3
"""
config_error.py: Configuration error handling for context storage system

This module provides custom exceptions for configuration-related errors
to replace sys.exit calls and enable proper test isolation.
"""


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""

    def __init__(self, message: str, config_path: str = None, details: dict = None):
        """
        Initialize configuration error.

        Args:
            message: Error message
            config_path: Path to the configuration file that caused the error
            details: Additional error details
        """
        self.config_path = config_path
        self.details = details or {}
        super().__init__(message)


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a required configuration file is not found."""

    def __init__(self, config_path: str):
        message = f"Configuration file not found: {config_path}"
        super().__init__(message, config_path=config_path)


class ConfigParseError(ConfigurationError):
    """Raised when configuration file cannot be parsed."""

    def __init__(self, config_path: str, parse_error: str):
        message = f"Failed to parse configuration file {config_path}: {parse_error}"
        super().__init__(message, config_path=config_path, details={"parse_error": parse_error})


class ConfigValidationError(ConfigurationError):
    """Raised when configuration validation fails."""

    def __init__(self, message: str, invalid_fields: list = None):
        details = {"invalid_fields": invalid_fields} if invalid_fields else {}
        super().__init__(message, details=details)
