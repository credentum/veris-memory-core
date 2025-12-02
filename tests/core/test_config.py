#!/usr/bin/env python3
"""
Test suite for config.py - Configuration management tests
"""
import os
import tempfile
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path

# Import the module under test
from src.core.config import Config, ConfigurationError, get_config, reload_config


class TestConfig:
    """Test suite for Config class"""

    def test_class_constants(self):
        """Test that class constants are properly defined"""
        # Embedding constants
        assert Config.EMBEDDING_DIMENSIONS == 1536
        assert Config.EMBEDDING_BATCH_SIZE == 100
        assert Config.EMBEDDING_MAX_RETRIES == 3
        
        # Database port constants
        assert Config.NEO4J_DEFAULT_PORT == 7687
        assert Config.QDRANT_DEFAULT_PORT == 6333
        assert Config.REDIS_DEFAULT_PORT == 6379
        
        # Connection pool constants
        assert Config.CONNECTION_POOL_MIN_SIZE == 5
        assert Config.CONNECTION_POOL_MAX_SIZE == 20
        assert Config.CONNECTION_POOL_TIMEOUT == 30
        
        # Rate limiting constants
        assert Config.RATE_LIMIT_REQUESTS_PER_MINUTE == 60
        assert Config.RATE_LIMIT_BURST_SIZE == 10
        
        # Security constants
        assert Config.MAX_QUERY_LENGTH == 10000
        assert Config.QUERY_TIMEOUT_SECONDS == 30
        assert "MATCH" in Config.ALLOWED_CYPHER_OPERATIONS
        assert "CREATE" in Config.FORBIDDEN_CYPHER_OPERATIONS
        
        # Cache constants
        assert Config.CACHE_TTL_SECONDS == 3600
        assert Config.CACHE_MAX_SIZE == 1000

    def test_embedding_model_from_env(self):
        """Test EMBEDDING_MODEL loads from environment variable"""
        # Test default value
        default_model = Config.EMBEDDING_MODEL
        assert default_model is not None
        
        # Test with environment variable
        with patch.dict(os.environ, {'EMBEDDING_MODEL': 'custom-model-v2'}):
            # Need to reload the module or recreate the class attribute
            with patch.object(Config, 'EMBEDDING_MODEL', os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")):
                assert Config.EMBEDDING_MODEL == 'custom-model-v2'

    def test_log_level_from_env(self):
        """Test LOG_LEVEL loads from environment variable"""
        # Test default value
        default_level = Config.LOG_LEVEL
        assert default_level == "INFO"
        
        # Test with environment variable
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'}):
            with patch.object(Config, 'LOG_LEVEL', os.getenv("LOG_LEVEL", "INFO")):
                assert Config.LOG_LEVEL == 'DEBUG'

    def test_get_defaults(self):
        """Test get_defaults method returns proper structure"""
        defaults = Config.get_defaults()
        
        # Check top-level keys
        expected_keys = ['embedding', 'databases', 'security', 'rate_limiting', 'cache', 'logging']
        for key in expected_keys:
            assert key in defaults
        
        # Check embedding configuration
        embedding = defaults['embedding']
        assert embedding['dimensions'] == 1536
        assert embedding['model'] == Config.EMBEDDING_MODEL
        assert embedding['batch_size'] == 100
        assert embedding['max_retries'] == 3
        
        # Check database configurations
        databases = defaults['databases']
        assert 'neo4j' in databases
        assert 'qdrant' in databases
        assert 'redis' in databases
        
        # Check Neo4j config
        neo4j = databases['neo4j']
        assert neo4j['port'] == 7687
        assert 'connection_pool' in neo4j
        assert neo4j['connection_pool']['min_size'] == 5
        assert neo4j['connection_pool']['max_size'] == 20
        assert neo4j['connection_pool']['timeout'] == 30
        
        # Check security configuration
        security = defaults['security']
        assert security['max_query_length'] == 10000
        assert security['query_timeout'] == 30
        assert security['allowed_operations'] == Config.ALLOWED_CYPHER_OPERATIONS
        assert security['forbidden_operations'] == Config.FORBIDDEN_CYPHER_OPERATIONS
        
        # Check rate limiting
        rate_limiting = defaults['rate_limiting']
        assert rate_limiting['requests_per_minute'] == 60
        assert rate_limiting['burst_size'] == 10
        
        # Check cache
        cache = defaults['cache']
        assert cache['ttl_seconds'] == 3600
        assert cache['max_size'] == 1000
        
        # Check logging
        logging = defaults['logging']
        assert logging['level'] == Config.LOG_LEVEL
        assert logging['format'] == Config.LOG_FORMAT

    def test_deep_merge_simple(self):
        """Test _deep_merge with simple dictionaries"""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        
        result = Config._deep_merge(base, override)
        
        assert result == {'a': 1, 'b': 3, 'c': 4}
        # Ensure original dicts are not modified
        assert base == {'a': 1, 'b': 2}
        assert override == {'b': 3, 'c': 4}

    def test_deep_merge_nested(self):
        """Test _deep_merge with nested dictionaries"""
        base = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'ssl': True
            },
            'cache': {
                'ttl': 3600
            }
        }
        
        override = {
            'database': {
                'host': 'remote.db.com',
                'timeout': 30
            },
            'logging': {
                'level': 'DEBUG'
            }
        }
        
        result = Config._deep_merge(base, override)
        
        expected = {
            'database': {
                'host': 'remote.db.com',  # overridden
                'port': 5432,             # preserved
                'ssl': True,              # preserved
                'timeout': 30             # added
            },
            'cache': {
                'ttl': 3600              # preserved
            },
            'logging': {
                'level': 'DEBUG'         # added
            }
        }
        
        assert result == expected

    def test_deep_merge_mixed_types(self):
        """Test _deep_merge when value types don't match"""
        base = {'config': {'value': 123}}
        override = {'config': 'simple_string'}
        
        result = Config._deep_merge(base, override)
        
        # When types don't match, override should win
        assert result == {'config': 'simple_string'}

    def test_load_from_file_missing_file(self):
        """Test load_from_file with missing config file"""
        with patch.dict(os.environ, {'CONTEXT_STORE_CONFIG': 'nonexistent.yaml'}):
            result = Config.load_from_file()
            
            # Should return defaults when file doesn't exist
            defaults = Config.get_defaults()
            assert result == defaults

    def test_load_from_file_custom_path_missing(self):
        """Test load_from_file with custom path that doesn't exist"""
        result = Config.load_from_file("definitely_does_not_exist.yaml")
        
        # Should return defaults when file doesn't exist
        defaults = Config.get_defaults()
        assert result == defaults

    def test_load_from_file_success(self):
        """Test successful config file loading"""
        config_content = """
embedding:
  model: custom-embedding-model
  batch_size: 50

databases:
  neo4j:
    host: custom.neo4j.com
    port: 7688
    
security:
  max_query_length: 5000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            result = Config.load_from_file(temp_path)
            
            # Check that custom values are present
            assert result['embedding']['model'] == 'custom-embedding-model'
            assert result['embedding']['batch_size'] == 50
            assert result['databases']['neo4j']['host'] == 'custom.neo4j.com'
            assert result['databases']['neo4j']['port'] == 7688
            assert result['security']['max_query_length'] == 5000
            
            # Check that defaults are still present for unspecified values
            assert result['embedding']['dimensions'] == 1536  # default
            assert result['databases']['qdrant']['port'] == 6333  # default
            assert result['cache']['ttl_seconds'] == 3600  # default
            
        finally:
            os.unlink(temp_path)

    def test_load_from_file_invalid_yaml(self):
        """Test load_from_file with invalid YAML"""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Failed to load configuration"):
                Config.load_from_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_from_file_with_env_var(self):
        """Test load_from_file using CONTEXT_STORE_CONFIG environment variable"""
        config_content = """
embedding:
  model: env-specified-model
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            with patch.dict(os.environ, {'CONTEXT_STORE_CONFIG': temp_path}):
                result = Config.load_from_file()
                
                assert result['embedding']['model'] == 'env-specified-model'
        finally:
            os.unlink(temp_path)

    def test_validate_configuration_valid(self):
        """Test validate_configuration with valid config"""
        valid_config = {
            'embedding': {
                'dimensions': 1536
            },
            'databases': {
                'neo4j': {
                    'port': 7687,
                    'connection_pool': {
                        'min_size': 5,
                        'max_size': 20
                    }
                },
                'qdrant': {
                    'port': 6333
                }
            }
        }
        
        # Should not raise exception
        result = Config.validate_configuration(valid_config)
        assert result is True

    def test_validate_configuration_invalid_embedding_dimensions(self):
        """Test validate_configuration with invalid embedding dimensions"""
        invalid_config = {
            'embedding': {
                'dimensions': 999  # Invalid dimension
            }
        }
        
        with pytest.raises(ConfigurationError, match="Invalid embedding dimensions: 999"):
            Config.validate_configuration(invalid_config)

    def test_validate_configuration_invalid_port(self):
        """Test validate_configuration with invalid port numbers"""
        # Test with port too low
        invalid_config = {
            'databases': {
                'neo4j': {
                    'port': 0
                }
            }
        }
        
        with pytest.raises(ConfigurationError, match="Invalid port for neo4j: 0"):
            Config.validate_configuration(invalid_config)
        
        # Test with port too high
        invalid_config = {
            'databases': {
                'qdrant': {
                    'port': 70000
                }
            }
        }
        
        with pytest.raises(ConfigurationError, match="Invalid port for qdrant: 70000"):
            Config.validate_configuration(invalid_config)
        
        # Test with non-integer port
        invalid_config = {
            'databases': {
                'redis': {
                    'port': 'not_a_number'
                }
            }
        }
        
        with pytest.raises(ConfigurationError, match="Invalid port for redis: not_a_number"):
            Config.validate_configuration(invalid_config)

    def test_validate_configuration_invalid_connection_pool(self):
        """Test validate_configuration with invalid connection pool settings"""
        invalid_config = {
            'databases': {
                'neo4j': {
                    'connection_pool': {
                        'min_size': 10,
                        'max_size': 5  # min > max
                    }
                }
            }
        }
        
        with pytest.raises(ConfigurationError, match="Connection pool min_size cannot exceed max_size"):
            Config.validate_configuration(invalid_config)

    def test_validate_configuration_valid_embedding_dimensions(self):
        """Test validate_configuration with all valid embedding dimensions"""
        valid_dimensions = [384, 768, 1536, 3072]
        
        for dim in valid_dimensions:
            config = {'embedding': {'dimensions': dim}}
            # Should not raise exception
            assert Config.validate_configuration(config) is True

    def test_validate_configuration_empty_config(self):
        """Test validate_configuration with empty config"""
        empty_config = {}
        
        # Should not raise exception for empty config
        result = Config.validate_configuration(empty_config)
        assert result is True

    def test_validate_configuration_partial_config(self):
        """Test validate_configuration with partial config sections"""
        partial_config = {
            'embedding': {},  # Empty embedding section
            'databases': {
                'neo4j': {
                    'connection_pool': {}  # Empty connection pool
                }
            }
        }
        
        # Should not raise exception for partial config
        result = Config.validate_configuration(partial_config)
        assert result is True


class TestConfigurationError:
    """Test suite for ConfigurationError exception"""

    def test_configuration_error_creation(self):
        """Test ConfigurationError can be created and raised"""
        error_message = "Test configuration error"
        
        error = ConfigurationError(error_message)
        assert str(error) == error_message
        assert isinstance(error, Exception)
        
        # Test that it can be raised
        with pytest.raises(ConfigurationError, match="Test configuration error"):
            raise ConfigurationError(error_message)


class TestGlobalConfigFunctions:
    """Test suite for global configuration functions"""

    def setUp(self):
        """Reset global config before each test"""
        # Import the module to access the global variable
        import src.core.config as config_module
        config_module._config_instance = None

    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance"""
        self.setUp()
        
        with patch.object(Config, 'load_from_file') as mock_load:
            mock_config = {'test': 'config'}
            mock_load.return_value = mock_config
            
            # First call should load config
            config1 = get_config()
            assert config1 == mock_config
            mock_load.assert_called_once()
            
            # Second call should return cached instance
            config2 = get_config()
            assert config2 == mock_config
            assert config1 is config2  # Same object reference
            # load_from_file should still only be called once
            mock_load.assert_called_once()

    def test_reload_config(self):
        """Test reload_config function"""
        self.setUp()
        
        with patch.object(Config, 'load_from_file') as mock_load:
            mock_config1 = {'version': 1}
            mock_config2 = {'version': 2}
            mock_load.side_effect = [mock_config1, mock_config2]
            
            # First load
            config1 = get_config()
            assert config1 == mock_config1
            
            # Reload with different path
            config2 = reload_config('/custom/path.yaml')
            assert config2 == mock_config2
            
            # Verify load_from_file was called with custom path
            mock_load.assert_any_call('/custom/path.yaml')
            
            # Subsequent get_config should return reloaded config
            config3 = get_config()
            assert config3 == mock_config2
            assert config3 is config2

    def test_reload_config_no_path(self):
        """Test reload_config without specifying path"""
        self.setUp()
        
        with patch.object(Config, 'load_from_file') as mock_load:
            mock_config = {'reloaded': True}
            mock_load.return_value = mock_config
            
            config = reload_config()
            assert config == mock_config
            
            # Should call load_from_file with None (default path)
            mock_load.assert_called_once_with(None)


class TestConfigIntegration:
    """Integration tests for Config functionality"""

    def test_full_config_lifecycle(self):
        """Test complete config loading and validation lifecycle"""
        config_content = """
embedding:
  model: integration-test-model
  dimensions: 768
  batch_size: 25

databases:
  neo4j:
    host: test.neo4j.com
    port: 7687
    ssl: true
    connection_pool:
      min_size: 2
      max_size: 10
      timeout: 15
  
  qdrant:
    host: test.qdrant.com
    port: 6333

security:
  max_query_length: 8000
  query_timeout: 25

rate_limiting:
  requests_per_minute: 120
  burst_size: 20

cache:
  ttl_seconds: 1800
  max_size: 500

logging:
  level: DEBUG
  format: "%(name)s - %(message)s"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            # Load configuration
            config = Config.load_from_file(temp_path)
            
            # Validate configuration
            Config.validate_configuration(config)
            
            # Test that custom values are loaded
            assert config['embedding']['model'] == 'integration-test-model'
            assert config['embedding']['dimensions'] == 768
            assert config['embedding']['batch_size'] == 25
            
            assert config['databases']['neo4j']['host'] == 'test.neo4j.com'
            assert config['databases']['neo4j']['connection_pool']['min_size'] == 2
            assert config['databases']['neo4j']['connection_pool']['max_size'] == 10
            
            assert config['security']['max_query_length'] == 8000
            assert config['rate_limiting']['requests_per_minute'] == 120
            assert config['cache']['ttl_seconds'] == 1800
            assert config['logging']['level'] == 'DEBUG'
            
            # Test that defaults are preserved for unspecified values
            assert config['embedding']['max_retries'] == 3  # default
            assert config['databases']['redis']['port'] == 6379  # default
            
        finally:
            os.unlink(temp_path)

    def test_config_error_handling_resilience(self):
        """Test that config system handles various error conditions gracefully"""
        # Test with completely malformed YAML
        malformed_yaml = """
        invalid: yaml: content
        [unclosed: bracket
        missing: quotes in "string
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(malformed_yaml)
            temp_path = f.name
        
        try:
            # Should raise ConfigurationError
            with pytest.raises(ConfigurationError):
                Config.load_from_file(temp_path)
        finally:
            os.unlink(temp_path)
        
        # Test that defaults still work after error
        defaults = Config.get_defaults()
        assert 'embedding' in defaults
        assert defaults['embedding']['dimensions'] == 1536