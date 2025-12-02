#!/usr/bin/env python3
"""
base_component.py: Base class for consistent error handling and logging
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import click

try:
    from core.utils import get_environment, sanitize_error_message
except ImportError:
    from .utils import get_environment, sanitize_error_message


class BaseComponent(ABC):
    """Base class for all components with consistent error handling"""

    def __init__(self, config_path: str = ".ctxrc.yaml", verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.config = self._load_config()
        self.environment = get_environment()

        # Warn if in production without proper config
        if self.environment == "production" and not self._validate_production_config():
            self.logger.warning("Running in production without proper security configuration")

    def _setup_logger(self) -> logging.Logger:
        """Setup logger with appropriate level"""
        logger = logging.getLogger(self.__class__.__name__)

        # Set level based on verbosity
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            import yaml

            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                return dict(config) if config is not None else {}
        except FileNotFoundError:
            self.log_error(f"Configuration file {self.config_path} not found")
            return {}
        except Exception as e:
            self.log_error(f"Error loading config: {e}")
            return {}

    def _validate_production_config(self) -> bool:
        """Validate configuration for production use"""
        # Override in subclasses for specific validation
        return True

    def log_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        sensitive_values: Optional[List[str]] = None,
    ) -> None:
        """Log error with sanitization"""
        sanitized_msg = sanitize_error_message(message, sensitive_values)

        if exception:
            exc_msg = sanitize_error_message(str(exception), sensitive_values)
            sanitized_msg = f"{sanitized_msg}: {exc_msg}"

        self.logger.error(sanitized_msg)

        if self.verbose:
            click.echo(f"❌ {sanitized_msg}", err=True)

    def log_warning(self, message: str) -> None:
        """Log warning"""
        self.logger.warning(message)

        if self.verbose:
            click.echo(f"⚠️  {message}", err=True)

    def log_info(self, message: str) -> None:
        """Log info"""
        self.logger.info(message)

        if self.verbose:
            click.echo(f"ℹ️  {message}")

    def log_success(self, message: str) -> None:
        """Log success"""
        self.logger.info(f"Success: {message}")

        if self.verbose:
            click.echo(f"✅ {message}")

    @abstractmethod
    def connect(self, **kwargs: Any) -> bool:
        """Connect to the service - must be implemented by subclasses"""
        pass

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper error handling"""
        try:
            self.close()
        except Exception as e:
            self.log_error("Error during cleanup", e)

        # Return False to propagate any exception
        return False

    def close(self):
        """Close connections - override in subclasses"""
        pass


class DatabaseComponent(BaseComponent):
    """Base class for database components with connection management"""

    def __init__(self, config_path: str = ".ctxrc.yaml", verbose: bool = False):
        super().__init__(config_path, verbose)
        self.connection = None
        self.is_connected = False

    def _validate_production_config(self) -> bool:
        """Validate database configuration for production"""
        # Check if SSL is enabled for production
        service_name = self._get_service_name()
        if service_name:
            service_config = self.config.get(service_name, {})
            if not service_config.get("ssl", False):
                self.log_warning(f"SSL is disabled for {service_name} in production")
                return False

        return True

    @abstractmethod
    def _get_service_name(self) -> str:
        """Get the service name for configuration - must be implemented"""
        pass

    def ensure_connected(self) -> bool:
        """Ensure database is connected"""
        if not self.is_connected:
            self.log_error("Not connected to database")
            return False
        return True
