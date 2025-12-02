#!/usr/bin/env python3
"""
Test suite for base_component.py - Foundation infrastructure tests
"""
import logging
import os
import tempfile
import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from typing import Any

# Import the module under test
from src.core.base_component import BaseComponent, DatabaseComponent


class _TestableBaseComponent(BaseComponent):
    """Concrete implementation for testing abstract BaseComponent"""
    
    def connect(self, **kwargs: Any) -> bool:
        """Test implementation of abstract connect method"""
        return True


class _TestableDatabaseComponent(DatabaseComponent):
    """Concrete implementation for testing abstract DatabaseComponent"""
    
    def connect(self, **kwargs: Any) -> bool:
        """Test implementation of abstract connect method"""
        self.is_connected = True
        return True
    
    def _get_service_name(self) -> str:
        """Test implementation of abstract _get_service_name method"""
        return "test_service"


class TestBaseComponent:
    """Test suite for BaseComponent class"""
    
    @pytest.fixture
    def mock_config_file(self):
        """Create a temporary config file for testing"""
        config_content = """test_key: test_value
nested:
  key1: value1
  key2: value2
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def component(self, mock_config_file):
        """Create a testable component instance"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            with patch('src.core.base_component.sanitize_error_message', side_effect=lambda x, y=None: x):
                return _TestableBaseComponent(config_path=mock_config_file, verbose=False)
    
    def test_init_basic(self, mock_config_file):
        """Test basic initialization of BaseComponent"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path=mock_config_file, verbose=True)
            
            assert component.config_path == mock_config_file
            assert component.verbose is True
            assert component.logger is not None
            assert isinstance(component.config, dict)
            assert component.environment == 'development'
    
    def test_init_missing_config_file(self):
        """Test initialization with missing config file"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path="nonexistent.yaml", verbose=False)
            
            assert component.config == {}
            # Should not raise exception, just log error and continue
    
    def test_init_production_warning(self, mock_config_file):
        """Test production environment warning"""
        with patch('src.core.base_component.get_environment', return_value='production'):
            with patch.object(_TestableBaseComponent, '_validate_production_config', return_value=False):
                component = _TestableBaseComponent(config_path=mock_config_file, verbose=False)
                
                # Verify logger was set up and warning would be logged
                assert component.environment == 'production'
                assert component.logger is not None
    
    def test_setup_logger_verbose(self):
        """Test logger setup with verbose mode"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path=".ctxrc.yaml", verbose=True)
            
            assert component.logger.level == logging.DEBUG
            assert component.logger.name == '_TestableBaseComponent'
    
    def test_setup_logger_non_verbose(self):
        """Test logger setup without verbose mode"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path=".ctxrc.yaml", verbose=False)
            
            assert component.logger.level == logging.INFO
    
    def test_load_config_success(self, mock_config_file):
        """Test successful config loading"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            with patch('src.core.base_component.sanitize_error_message', side_effect=lambda x, y=None: x):
                component = _TestableBaseComponent(config_path=mock_config_file, verbose=False)
                
                assert 'test_key' in component.config
                assert component.config['test_key'] == 'test_value'
                assert 'nested' in component.config
                assert component.config['nested']['key1'] == 'value1'
    
    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML"""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with patch('src.core.base_component.get_environment', return_value='development'):
                component = _TestableBaseComponent(config_path=temp_path, verbose=False)
                
                # Should return empty dict on parse error
                assert component.config == {}
        finally:
            os.unlink(temp_path)
    
    def test_load_config_empty_file(self):
        """Test config loading with empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with patch('src.core.base_component.get_environment', return_value='development'):
                component = _TestableBaseComponent(config_path=temp_path, verbose=False)
                
                # Should return empty dict for empty file
                assert component.config == {}
        finally:
            os.unlink(temp_path)
    
    def test_validate_production_config_default(self, component):
        """Test default production config validation"""
        result = component._validate_production_config()
        assert result is True  # Default implementation returns True
    
    def test_log_error_basic(self, component):
        """Test basic error logging"""
        with patch.object(component.logger, 'error') as mock_logger:
            with patch('src.core.base_component.sanitize_error_message', return_value='sanitized message'):
                component.log_error("test error message")
                
                mock_logger.assert_called_once_with('sanitized message')
    
    def test_log_error_with_exception(self, component):
        """Test error logging with exception"""
        test_exception = ValueError("test exception")
        
        with patch.object(component.logger, 'error') as mock_logger:
            with patch('src.core.base_component.sanitize_error_message', side_effect=['sanitized msg', 'sanitized exc']):
                component.log_error("test error", exception=test_exception)
                
                mock_logger.assert_called_once_with('sanitized msg: sanitized exc')
    
    def test_log_error_verbose_mode(self, mock_config_file):
        """Test error logging in verbose mode"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path=mock_config_file, verbose=True)
            
            with patch.object(component.logger, 'error'):
                with patch('src.core.base_component.sanitize_error_message', return_value='sanitized message'):
                    with patch('click.echo') as mock_click:
                        component.log_error("test error message")
                        
                        mock_click.assert_called_once_with("❌ sanitized message", err=True)
    
    def test_log_error_with_sensitive_values(self, component):
        """Test error logging with sensitive value sanitization"""
        sensitive_values = ["password123", "secret_key"]
        
        with patch.object(component.logger, 'error') as mock_logger:
            with patch('src.core.base_component.sanitize_error_message', return_value='sanitized message') as mock_sanitize:
                component.log_error("test error", sensitive_values=sensitive_values)
                
                mock_sanitize.assert_called_once_with("test error", sensitive_values)
                mock_logger.assert_called_once_with('sanitized message')
    
    def test_log_warning(self, component):
        """Test warning logging"""
        with patch.object(component.logger, 'warning') as mock_logger:
            component.log_warning("test warning")
            
            mock_logger.assert_called_once_with("test warning")
    
    def test_log_warning_verbose_mode(self, mock_config_file):
        """Test warning logging in verbose mode"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path=mock_config_file, verbose=True)
            
            with patch.object(component.logger, 'warning'):
                with patch('click.echo') as mock_click:
                    component.log_warning("test warning")
                    
                    mock_click.assert_called_once_with("⚠️  test warning", err=True)
    
    def test_log_info(self, component):
        """Test info logging"""
        with patch.object(component.logger, 'info') as mock_logger:
            component.log_info("test info")
            
            mock_logger.assert_called_once_with("test info")
    
    def test_log_info_verbose_mode(self, mock_config_file):
        """Test info logging in verbose mode"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path=mock_config_file, verbose=True)
            
            with patch.object(component.logger, 'info'):
                with patch('click.echo') as mock_click:
                    component.log_info("test info")
                    
                    mock_click.assert_called_once_with("ℹ️  test info")
    
    def test_log_success(self, component):
        """Test success logging"""
        with patch.object(component.logger, 'info') as mock_logger:
            component.log_success("test success")
            
            mock_logger.assert_called_once_with("Success: test success")
    
    def test_log_success_verbose_mode(self, mock_config_file):
        """Test success logging in verbose mode"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            component = _TestableBaseComponent(config_path=mock_config_file, verbose=True)
            
            with patch.object(component.logger, 'info'):
                with patch('click.echo') as mock_click:
                    component.log_success("test success")
                    
                    mock_click.assert_called_once_with("✅ test success")
    
    def test_connect_abstract_method(self, component):
        """Test that connect method is properly implemented"""
        result = component.connect()
        assert result is True
    
    def test_context_manager_entry(self, component):
        """Test context manager __enter__ method"""
        with component as ctx:
            assert ctx is component
    
    def test_context_manager_exit_normal(self, component):
        """Test context manager __exit__ with normal execution"""
        with patch.object(component, 'close') as mock_close:
            result = component.__exit__(None, None, None)
            
            mock_close.assert_called_once()
            assert result is False  # Should not suppress exceptions
    
    def test_context_manager_exit_with_close_error(self, component):
        """Test context manager __exit__ when close() raises exception"""
        close_exception = RuntimeError("Close failed")
        
        with patch.object(component, 'close', side_effect=close_exception):
            with patch.object(component, 'log_error') as mock_log_error:
                result = component.__exit__(None, None, None)
                
                mock_log_error.assert_called_once_with("Error during cleanup", close_exception)
                assert result is False
    
    def test_close_default_implementation(self, component):
        """Test default close method implementation"""
        # Should not raise exception
        component.close()


class TestDatabaseComponent:
    """Test suite for DatabaseComponent class"""
    
    @pytest.fixture
    def mock_config_file(self):
        """Create a temporary config file for testing"""
        config_content = """test_service:
  ssl: true
  host: localhost
  port: 5432
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def db_component(self, mock_config_file):
        """Create a testable database component instance"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            return _TestableDatabaseComponent(config_path=mock_config_file, verbose=False)
    
    def test_init_database_component(self, mock_config_file):
        """Test DatabaseComponent initialization"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            db_comp = _TestableDatabaseComponent(config_path=mock_config_file, verbose=False)
            
            assert db_comp.connection is None
            assert db_comp.is_connected is False
            assert hasattr(db_comp, 'config')
            assert hasattr(db_comp, 'logger')
    
    def test_validate_production_config_ssl_enabled(self, mock_config_file):
        """Test production config validation with SSL enabled"""
        with patch('src.core.base_component.get_environment', return_value='production'):
            db_comp = _TestableDatabaseComponent(config_path=mock_config_file, verbose=False)
            
            result = db_comp._validate_production_config()
            assert result is True
    
    def test_validate_production_config_ssl_disabled(self):
        """Test production config validation with SSL disabled"""
        config_content = """test_service:
  ssl: false
  host: localhost
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            with patch('src.core.base_component.get_environment', return_value='production'):
                db_comp = _TestableDatabaseComponent(config_path=temp_path, verbose=False)
                
                with patch.object(db_comp, 'log_warning') as mock_warning:
                    result = db_comp._validate_production_config()
                    
                    assert result is False
                    mock_warning.assert_called_once_with("SSL is disabled for test_service in production")
        finally:
            os.unlink(temp_path)
    
    def test_validate_production_config_no_service_config(self, mock_config_file):
        """Test production config validation with missing service config"""
        with patch('src.core.base_component.get_environment', return_value='production'):
            # Create component that returns different service name
            class TestComponentNoConfig(DatabaseComponent):
                def connect(self, **kwargs): return True
                def _get_service_name(self): return "nonexistent_service"
            
            db_comp = TestComponentNoConfig(config_path=mock_config_file, verbose=False)
            result = db_comp._validate_production_config()
            assert result is True  # Should pass if service config doesn't exist
    
    def test_get_service_name_abstract_method(self, db_component):
        """Test that _get_service_name method is properly implemented"""
        service_name = db_component._get_service_name()
        assert service_name == "test_service"
    
    def test_ensure_connected_when_connected(self, db_component):
        """Test ensure_connected when database is connected"""
        db_component.is_connected = True
        
        result = db_component.ensure_connected()
        assert result is True
    
    def test_ensure_connected_when_not_connected(self, db_component):
        """Test ensure_connected when database is not connected"""
        db_component.is_connected = False
        
        with patch.object(db_component, 'log_error') as mock_log_error:
            result = db_component.ensure_connected()
            
            assert result is False
            mock_log_error.assert_called_once_with("Not connected to database")
    
    def test_connect_implementation(self, db_component):
        """Test connect method implementation in testable component"""
        result = db_component.connect()
        
        assert result is True
        assert db_component.is_connected is True


class TestBaseComponentIntegration:
    """Integration tests for BaseComponent functionality"""
    
    def test_full_lifecycle_with_real_config(self):
        """Test complete component lifecycle with actual config file"""
        config_content = """database:
  host: localhost
  port: 5432
  ssl: true
logging:
  level: info
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            with patch('src.core.base_component.get_environment', return_value='development'):
                # Test complete lifecycle
                with _TestableBaseComponent(config_path=config_path, verbose=True) as component:
                    # Component should be initialized
                    assert component.config['database']['host'] == 'localhost'
                    assert component.config['database']['ssl'] is True
                    
                    # Test all logging methods
                    component.log_info("Integration test info")
                    component.log_warning("Integration test warning")
                    component.log_error("Integration test error")
                    component.log_success("Integration test success")
                    
                    # Test connection
                    assert component.connect() is True
                
                # Context manager should have called close
                # No exception should be raised
                
        finally:
            os.unlink(config_path)
    
    def test_error_handling_resilience(self):
        """Test that component handles various error conditions gracefully"""
        with patch('src.core.base_component.get_environment', return_value='development'):
            # Test with completely missing config file
            component = _TestableBaseComponent(config_path="definitely_does_not_exist.yaml", verbose=False)
            
            # Should still be functional despite missing config
            assert component.config == {}
            assert component.logger is not None
            
            # Should handle logging without errors
            component.log_error("Test error handling")
            component.log_info("Test info handling")
            
            # Should handle connection attempts
            assert component.connect() is True
            
            # Should handle context manager operations
            with component:
                pass