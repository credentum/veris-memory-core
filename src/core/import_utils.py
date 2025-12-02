#!/usr/bin/env python3
"""
Import utilities for the Veris Memory system.

This module provides common patterns for safe importing with fallbacks,
error handling, and logging. It centralizes import logic to reduce
code duplication and ensure consistent behavior across the codebase.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable


logger = logging.getLogger(__name__)


class ImportResult:
    """Result of an import attempt with metadata."""
    
    def __init__(self, success: bool, module: Any = None, error: Optional[Exception] = None, 
                 import_path: str = "", method: str = ""):
        self.success = success
        self.module = module
        self.error = error
        self.import_path = import_path
        self.method = method
    
    def __bool__(self) -> bool:
        return self.success
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"ImportResult({status}, path='{self.import_path}', method='{self.method}')"


def safe_import(
    module_path: str,
    fallback_paths: Optional[List[str]] = None,
    required_attrs: Optional[List[str]] = None,
    fallback_factory: Optional[Callable] = None,
    log_level: str = "warning"
) -> ImportResult:
    """
    Safely import a module with fallback options and comprehensive error handling.
    
    Args:
        module_path: Primary module path to import (e.g., 'src.core.config')
        fallback_paths: List of alternative import paths to try
        required_attrs: List of attributes that must exist in the imported module
        fallback_factory: Function to create a fallback object if all imports fail
        log_level: Logging level for import failures ('debug', 'info', 'warning', 'error')
        
    Returns:
        ImportResult containing the imported module or fallback object
        
    Examples:
        # Simple import with fallback
        result = safe_import('src.monitoring.dashboard', 
                           fallback_paths=['monitoring.dashboard'])
        
        # Import with required attributes check
        result = safe_import('qdrant_client', 
                           required_attrs=['QdrantClient'])
        
        # Import with fallback factory
        def create_mock_client():
            class MockClient:
                def connect(self): return True
            return MockClient()
            
        result = safe_import('redis', fallback_factory=create_mock_client)
    """
    import_paths = [module_path] + (fallback_paths or [])
    log_func = getattr(logger, log_level.lower(), logger.warning)
    
    for i, path in enumerate(import_paths):
        try:
            # Handle both absolute and relative imports
            if path.startswith('.'):
                # Relative import - need to specify package
                module = __import__(path, fromlist=[''])
            else:
                # Absolute import
                module = __import__(path, fromlist=[''])
                
                # For nested modules, get the leaf module
                for component in path.split('.')[1:]:
                    module = getattr(module, component)
            
            # Check required attributes if specified
            if required_attrs:
                missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
                if missing_attrs:
                    raise AttributeError(f"Module '{path}' missing required attributes: {missing_attrs}")
            
            method = "primary" if i == 0 else f"fallback_{i}"
            logger.debug(f"Successfully imported '{path}' via {method} method")
            
            return ImportResult(
                success=True,
                module=module,
                import_path=path,
                method=method
            )
            
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            method = "primary" if i == 0 else f"fallback_{i}"
            log_func(f"Failed to import '{path}' via {method} method: {e}")
            continue
    
    # All imports failed - try fallback factory
    if fallback_factory:
        try:
            fallback_obj = fallback_factory()
            logger.info(f"Created fallback object for '{module_path}' using factory function")
            return ImportResult(
                success=True,
                module=fallback_obj,
                import_path=module_path,
                method="fallback_factory"
            )
        except Exception as e:
            logger.error(f"Fallback factory failed for '{module_path}': {e}")
    
    # Everything failed
    logger.error(f"All import attempts failed for '{module_path}'")
    return ImportResult(
        success=False,
        error=ImportError(f"Could not import '{module_path}' or any fallbacks"),
        import_path=module_path,
        method="none"
    )


def bulk_import(
    import_specs: Dict[str, Dict[str, Any]],
    continue_on_failure: bool = True
) -> Dict[str, ImportResult]:
    """
    Import multiple modules in bulk with individual configurations.
    
    Args:
        import_specs: Dictionary mapping names to import specifications
                     Each spec can contain: module_path, fallback_paths, 
                     required_attrs, fallback_factory, log_level
        continue_on_failure: Whether to continue importing other modules if one fails
        
    Returns:
        Dictionary mapping names to ImportResult objects
        
    Example:
        specs = {
            'dashboard': {
                'module_path': 'src.monitoring.dashboard',
                'fallback_paths': ['monitoring.dashboard'],
                'required_attrs': ['UnifiedDashboard']
            },
            'redis': {
                'module_path': 'redis',
                'fallback_factory': lambda: MockRedis()
            }
        }
        results = bulk_import(specs)
    """
    results = {}
    
    for name, spec in import_specs.items():
        try:
            result = safe_import(**spec)
            results[name] = result
            
            if not result.success and not continue_on_failure:
                logger.error(f"Bulk import stopped due to failure importing '{name}'")
                break
                
        except Exception as e:
            logger.error(f"Unexpected error importing '{name}': {e}")
            results[name] = ImportResult(
                success=False,
                error=e,
                import_path=spec.get('module_path', ''),
                method="error"
            )
            
            if not continue_on_failure:
                break
    
    return results


def setup_import_path(additional_paths: Optional[List[str]] = None) -> None:
    """
    Setup sys.path for imports, adding project root and additional paths.
    
    Args:
        additional_paths: Additional paths to add to sys.path
    """
    # Add project root to path
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Go up 3 levels from src/core/import_utils.py
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.debug(f"Added project root to sys.path: {project_root}")
    
    # Add additional paths
    if additional_paths:
        for path in additional_paths:
            abs_path = str(Path(path).resolve())
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
                logger.debug(f"Added additional path to sys.path: {abs_path}")


def try_relative_then_absolute(
    relative_import: str,
    absolute_import: str,
    required_attrs: Optional[List[str]] = None,
    package: Optional[str] = None
) -> ImportResult:
    """
    Try relative import first, then absolute import with sys.path setup.
    
    This is a common pattern in the codebase for handling both
    development and production import scenarios.
    
    Args:
        relative_import: Relative import path (e.g., '.dashboard')
        absolute_import: Absolute import path (e.g., 'src.monitoring.dashboard')
        required_attrs: List of required attributes to check
        package: Package name for relative imports
        
    Returns:
        ImportResult with the successful import or failure details
        
    Example:
        result = try_relative_then_absolute(
            relative_import='.dashboard',
            absolute_import='src.monitoring.dashboard',
            required_attrs=['UnifiedDashboard']
        )
    """
    # Try relative import first
    try:
        if package:
            module = __import__(relative_import, fromlist=[''], package=package)
        else:
            module = __import__(relative_import, fromlist=[''])
        
        # Check required attributes
        if required_attrs:
            missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
            if missing_attrs:
                raise AttributeError(f"Module missing required attributes: {missing_attrs}")
        
        logger.debug(f"Successfully imported '{relative_import}' via relative import")
        return ImportResult(
            success=True,
            module=module,
            import_path=relative_import,
            method="relative"
        )
        
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        logger.debug(f"Relative import failed for '{relative_import}': {e}")
    
    # Setup import path and try absolute import
    setup_import_path()
    
    try:
        module = __import__(absolute_import, fromlist=[''])
        
        # For nested modules, get the leaf module
        for component in absolute_import.split('.')[1:]:
            module = getattr(module, component)
        
        # Check required attributes
        if required_attrs:
            missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
            if missing_attrs:
                raise AttributeError(f"Module missing required attributes: {missing_attrs}")
        
        logger.info(f"Successfully imported '{absolute_import}' via absolute import fallback")
        return ImportResult(
            success=True,
            module=module,
            import_path=absolute_import,
            method="absolute"
        )
        
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        logger.error(f"Both relative and absolute imports failed. Relative: '{relative_import}', Absolute: '{absolute_import}', Error: {e}")
        return ImportResult(
            success=False,
            error=e,
            import_path=f"{relative_import} | {absolute_import}",
            method="none"
        )


def create_fallback_class(
    class_name: str,
    methods: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    base_classes: Optional[List[type]] = None
) -> type:
    """
    Dynamically create a fallback class for failed imports.
    
    Args:
        class_name: Name of the fallback class
        methods: Dictionary of method names to implementations
        attributes: Dictionary of attribute names to default values
        base_classes: List of base classes to inherit from
        
    Returns:
        Dynamically created class type
        
    Example:
        MockRedis = create_fallback_class(
            'MockRedis',
            methods={
                'get': lambda self, key: None,
                'set': lambda self, key, value: True,
                'ping': lambda self: True
            },
            attributes={'connected': False}
        )
    """
    # Default methods and attributes
    default_methods = {
        '__init__': lambda self, *args, **kwargs: None,
        '__repr__': lambda self: f"<Fallback {class_name}>"
    }
    
    # Merge with provided methods
    all_methods = {**default_methods, **(methods or {})}
    
    # Add attributes as properties
    if attributes:
        for attr_name, default_value in attributes.items():
            all_methods[attr_name] = default_value
    
    # Create the class
    fallback_class = type(
        class_name,
        tuple(base_classes or []),
        all_methods
    )
    
    logger.debug(f"Created fallback class '{class_name}' with {len(all_methods)} methods/attributes")
    return fallback_class


def get_optional_dependency(
    module_name: str,
    pip_install_name: Optional[str] = None,
    conda_install_name: Optional[str] = None
) -> Tuple[bool, Optional[Any], str]:
    """
    Try to import an optional dependency with helpful error messages.
    
    Args:
        module_name: Name of the module to import
        pip_install_name: Package name for pip install (if different from module_name)
        conda_install_name: Package name for conda install (if different from module_name)
        
    Returns:
        Tuple of (success, module_or_none, install_suggestion)
        
    Example:
        available, torch, install_msg = get_optional_dependency(
            'torch', 
            pip_install_name='torch',
            conda_install_name='pytorch'
        )
        if not available:
            logger.warning(f"PyTorch not available: {install_msg}")
    """
    try:
        module = __import__(module_name)
        return True, module, ""
    except ImportError:
        pip_name = pip_install_name or module_name
        conda_name = conda_install_name or module_name
        
        install_suggestion = (
            f"To install {module_name}, try:\n"
            f"  pip install {pip_name}\n"
            f"  or\n"
            f"  conda install {conda_name}"
        )
        
        return False, None, install_suggestion


# Commonly used fallback factories for the Veris Memory system
def create_mock_redis():
    """Create a mock Redis client for testing/fallback."""
    return create_fallback_class(
        'MockRedis',
        methods={
            'get': lambda self, key: None,
            'set': lambda self, key, value: True,
            'delete': lambda self, *keys: len(keys),
            'exists': lambda self, *keys: 0,
            'ping': lambda self: True,
            'flushdb': lambda self: True,
        },
        attributes={'connected': False}
    )()


def create_mock_qdrant():
    """Create a mock Qdrant client for testing/fallback."""
    return create_fallback_class(
        'MockQdrantClient',
        methods={
            'search': lambda self, *args, **kwargs: [],
            'upsert': lambda self, *args, **kwargs: True,
            'delete': lambda self, *args, **kwargs: True,
            'get_collections': lambda self: [],
            'create_collection': lambda self, *args, **kwargs: True,
        }
    )()


def create_mock_neo4j():
    """Create a mock Neo4j driver for testing/fallback."""
    MockSession = create_fallback_class(
        'MockNeo4jSession',
        methods={
            'run': lambda self, query, parameters=None: [],
            'close': lambda self: None,
            'begin_transaction': lambda self: self,
            'commit': lambda self: None,
            'rollback': lambda self: None,
        }
    )
    
    return create_fallback_class(
        'MockNeo4jDriver',
        methods={
            'session': lambda self: MockSession(),
            'close': lambda self: None,
            'verify_connectivity': lambda self: None,
        }
    )()