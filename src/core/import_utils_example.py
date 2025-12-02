#!/usr/bin/env python3
"""
Example demonstrating how to use the new import_utils for common patterns.

This file shows how existing import patterns can be refactored to use
the centralized import utilities for better consistency and maintainability.
"""

import logging
from typing import Any, Dict

# Import the new utilities
from .import_utils import (
    safe_import, 
    bulk_import, 
    try_relative_then_absolute,
    create_fallback_class,
    create_mock_redis,
    create_mock_qdrant
)

logger = logging.getLogger(__name__)


# Example 1: Simple safe import with fallback factory
def example_simple_import():
    """Example of simple import with fallback."""
    
    # Old way (scattered throughout codebase):
    # try:
    #     import redis
    # except ImportError:
    #     class MockRedis:
    #         def get(self, key): return None
    #         def set(self, key, value): return True
    #     redis = MockRedis()
    
    # New way (using import_utils):
    result = safe_import(
        module_path='redis',
        fallback_factory=create_mock_redis
    )
    
    if result.success:
        redis_client = result.module
        logger.info(f"Redis imported via {result.method}: {result.import_path}")
        return redis_client
    else:
        logger.error(f"Failed to import redis: {result.error}")
        return None


# Example 2: Bulk import pattern (like dashboard_api.py)
def example_bulk_import():
    """Example of bulk importing multiple related modules."""
    
    # Define import specifications
    import_specs = {
        'dashboard': {
            'module_path': '.dashboard',
            'fallback_paths': ['src.monitoring.dashboard'],
            'required_attrs': ['UnifiedDashboard']
        },
        'streaming': {
            'module_path': '.streaming', 
            'fallback_paths': ['src.monitoring.streaming'],
            'required_attrs': ['MetricsStreamer']
        },
        'rate_limiter': {
            'module_path': '..core.rate_limiter',
            'fallback_paths': ['src.core.rate_limiter'],
            'required_attrs': ['get_rate_limiter', 'MCPRateLimiter']
        }
    }
    
    # Import all modules
    results = bulk_import(import_specs, continue_on_failure=True)
    
    # Extract successfully imported modules
    imported_modules = {}
    for name, result in results.items():
        if result.success:
            imported_modules[name] = result.module
            logger.info(f"{name} imported via {result.method}")
        else:
            logger.warning(f"Failed to import {name}: {result.error}")
            # Create fallback if needed
            imported_modules[name] = create_fallback_for_module(name)
    
    return imported_modules


# Example 3: Relative then absolute pattern
def example_relative_absolute():
    """Example of the common relative-then-absolute pattern."""
    
    # Old way (repeated in many files):
    # try:
    #     from .dashboard import UnifiedDashboard
    # except ImportError:
    #     import sys, os
    #     sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    #     from src.monitoring.dashboard import UnifiedDashboard
    
    # New way:
    result = try_relative_then_absolute(
        relative_import='.dashboard',
        absolute_import='src.monitoring.dashboard',
        required_attrs=['UnifiedDashboard']
    )
    
    if result.success:
        dashboard_module = result.module
        UnifiedDashboard = getattr(dashboard_module, 'UnifiedDashboard')
        return UnifiedDashboard
    else:
        logger.error(f"Could not import dashboard: {result.error}")
        return create_fallback_dashboard_class()


# Example 4: Optional dependency handling
def example_optional_dependency():
    """Example of handling optional dependencies."""
    from .import_utils import get_optional_dependency
    
    # Check for optional ML libraries
    torch_available, torch, torch_install = get_optional_dependency(
        'torch', 
        pip_install_name='torch',
        conda_install_name='pytorch'
    )
    
    sklearn_available, sklearn, sklearn_install = get_optional_dependency(
        'sklearn',
        pip_install_name='scikit-learn'
    )
    
    if not torch_available:
        logger.warning(f"PyTorch not available. {torch_install}")
    
    if not sklearn_available:
        logger.warning(f"Scikit-learn not available. {sklearn_install}")
    
    return {
        'torch': torch if torch_available else None,
        'sklearn': sklearn if sklearn_available else None
    }


# Helper functions for creating fallbacks
def create_fallback_for_module(module_name: str) -> Any:
    """Create appropriate fallback objects for different modules."""
    
    fallback_factories = {
        'dashboard': create_fallback_dashboard_class,
        'streaming': create_fallback_streaming_class,
        'rate_limiter': create_fallback_rate_limiter_class
    }
    
    factory = fallback_factories.get(module_name)
    if factory:
        return factory()
    
    # Generic fallback
    return create_fallback_class(
        f'Fallback{module_name.title()}',
        methods={'__init__': lambda self, *args, **kwargs: None}
    )()


def create_fallback_dashboard_class():
    """Create fallback dashboard class."""
    return create_fallback_class(
        'FallbackUnifiedDashboard',
        methods={
            'collect_all_metrics': lambda self, *args, **kwargs: {'error': 'Dashboard module not available'},
            'generate_ascii_dashboard': lambda self, *args, **kwargs: 'Dashboard module not available',
            'shutdown': lambda self: None
        },
        attributes={'last_update': None}
    )()


def create_fallback_streaming_class():
    """Create fallback streaming class."""
    return create_fallback_class(
        'FallbackMetricsStreamer',
        methods={
            'start': lambda self: None,
            'stop': lambda self: None,
            'send_update': lambda self, data: None
        }
    )()


def create_fallback_rate_limiter_class():
    """Create fallback rate limiter class."""
    
    # Create mock rate limiter
    MockRateLimiter = create_fallback_class(
        'MockRateLimiter',
        methods={
            'get_client_id': lambda self, *args: 'unknown',
            'check_rate_limit': lambda self, *args: (True, None),
            'check_burst_protection': lambda self, *args: (True, None),
        },
        attributes={'endpoint_limits': {}}
    )
    
    # Create module-like object with the functions
    class MockRateLimiterModule:
        def __init__(self):
            self.MCPRateLimiter = MockRateLimiter
            self._instance = None
        
        def get_rate_limiter(self):
            if self._instance is None:
                self._instance = MockRateLimiter()
            return self._instance
    
    return MockRateLimiterModule()


# Example 5: Configuration-driven imports
def example_config_driven_imports():
    """Example of using configuration to control imports."""
    
    # This could be loaded from a config file
    import_config = {
        'optional_modules': {
            'torch': {
                'enabled': True,
                'fallback_factory': lambda: None
            },
            'sentence_transformers': {
                'enabled': True,
                'pip_install_name': 'sentence-transformers'
            }
        },
        'required_modules': {
            'fastapi': {
                'required_attrs': ['FastAPI', 'Request']
            }
        }
    }
    
    imported_modules = {}
    
    # Import optional modules
    for module_name, config in import_config['optional_modules'].items():
        if config.get('enabled', True):
            result = safe_import(
                module_path=module_name,
                fallback_factory=config.get('fallback_factory'),
                log_level='info'
            )
            imported_modules[module_name] = result.module if result.success else None
    
    # Import required modules
    for module_name, config in import_config['required_modules'].items():
        result = safe_import(
            module_path=module_name,
            required_attrs=config.get('required_attrs', []),
            log_level='error'
        )
        
        if not result.success:
            raise ImportError(f"Required module '{module_name}' could not be imported: {result.error}")
        
        imported_modules[module_name] = result.module
    
    return imported_modules


if __name__ == "__main__":
    # Demo the examples
    logging.basicConfig(level=logging.INFO)
    
    print("=== Import Utils Examples ===\n")
    
    print("1. Simple import with fallback:")
    redis_client = example_simple_import()
    print(f"Redis client: {type(redis_client).__name__}\n")
    
    print("2. Bulk import pattern:")
    modules = example_bulk_import()
    print(f"Imported modules: {list(modules.keys())}\n")
    
    print("3. Relative then absolute pattern:")
    dashboard_class = example_relative_absolute()
    print(f"Dashboard class: {type(dashboard_class).__name__}\n")
    
    print("4. Optional dependency handling:")
    deps = example_optional_dependency()
    print(f"Dependencies: {[k for k, v in deps.items() if v is not None]}\n")
    
    print("5. Configuration-driven imports:")
    try:
        config_modules = example_config_driven_imports()
        print(f"Config-driven modules: {list(config_modules.keys())}")
    except ImportError as e:
        print(f"Required module import failed: {e}")