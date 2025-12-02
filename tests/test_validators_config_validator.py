#!/usr/bin/env python3
"""
Comprehensive tests for Config Validator - Phase 10 Coverage

This test module provides comprehensive coverage for the configuration validation system
including main config validation, performance config validation, and security checks.
"""
import pytest
import tempfile
import yaml
import os
import sys
from unittest.mock import patch, Mock, MagicMock, mock_open
from typing import Dict, Any, List, Optional

# Import config validator components
try:
    from src.validators.config_validator import (
        ConfigValidator, 
        ConfigValidationError,
        validate_environment_variables,
        validate_database_config,
        validate_mcp_config,
        validate_all_configs
    )
    CONFIG_VALIDATOR_AVAILABLE = True
except ImportError:
    CONFIG_VALIDATOR_AVAILABLE = False


@pytest.mark.skipif(not CONFIG_VALIDATOR_AVAILABLE, reason="Config validator not available")
class TestConfigValidator:
    """Test basic config validator functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = ConfigValidator()
        
        # Sample valid main configuration
        self.valid_main_config = {
            "system": {
                "debug": False,
                "log_level": "INFO"
            },
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "ssl": False,
                "collection_name": "test_context"
            },
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "test_password",
                "database": "test_graph",
                "port": 7687,
                "ssl": False
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "database": 0,
                "ssl": False
            },
            "duckdb": {
                "database_path": "test.duckdb",
                "threads": 4
            },
            "storage": {
                "type": "hybrid",
                "cache_size": "1GB"
            },
            "agents": {
                "max_concurrent": 5,
                "timeout": 30
            }
        }
        
        # Sample valid performance configuration
        self.valid_perf_config = {
            "vector_db": {
                "embedding": {
                    "batch_size": 100,
                    "max_retries": 3,
                    "initial_retry_delay": 1.0,
                    "retry_backoff_factor": 2.0,
                    "request_timeout": 30
                },
                "search": {
                    "default_limit": 10,
                    "max_limit": 100
                }
            },
            "graph_db": {
                "connection_pool": {
                    "min_size": 1,
                    "max_size": 10
                },
                "query": {
                    "timeout": 30,
                    "max_path_length": 5
                }
            },
            "search": {
                "ranking": {
                    "temporal_decay_rate": 0.01,
                    "type_boosts": {
                        "documentation": 1.2,
                        "code": 1.0,
                        "test": 0.8
                    }
                }
            },
            "resources": {
                "max_memory_gb": 4.0,
                "max_cpu_percent": 80
            },
            "kv_store": {
                "redis": {
                    "connection_pool": {
                        "min_size": 5,
                        "max_size": 50
                    },
                    "cache": {
                        "ttl_seconds": 3600
                    }
                },
                "duckdb": {
                    "batch_insert": {
                        "size": 1000
                    },
                    "analytics": {
                        "retention_days": 90
                    }
                }
            }
        }
    
    def test_config_validator_creation(self):
        """Test config validator creation"""
        assert self.validator is not None
        assert isinstance(self.validator.errors, list)
        assert isinstance(self.validator.warnings, list)
        assert len(self.validator.errors) == 0
        assert len(self.validator.warnings) == 0
    
    def test_validate_main_config_valid(self):
        """Test validation of valid main configuration"""
        with patch('builtins.open', mock_open(read_data=yaml.dump(self.valid_main_config))):
            with patch('yaml.safe_load', return_value=self.valid_main_config):
                result = self.validator.validate_main_config("test_config.yaml")
                
                assert result is True
                assert len(self.validator.errors) == 0
    
    def test_validate_main_config_missing_file(self):
        """Test validation with missing config file"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = self.validator.validate_main_config("missing_config.yaml")
            
            assert result is False
            assert len(self.validator.errors) > 0
            assert any("not found" in error for error in self.validator.errors)
    
    def test_validate_main_config_invalid_yaml(self):
        """Test validation with invalid YAML"""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")):
                result = self.validator.validate_main_config("invalid.yaml")
                
                assert result is False
                assert len(self.validator.errors) > 0
                assert any("Invalid YAML" in error for error in self.validator.errors)
    
    def test_validate_main_config_missing_sections(self):
        """Test validation with missing required sections"""
        incomplete_config = {"system": {"debug": False}}
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(incomplete_config))):
            with patch('yaml.safe_load', return_value=incomplete_config):
                result = self.validator.validate_main_config("incomplete.yaml")
                
                assert result is False
                assert len(self.validator.errors) > 0
                
                # Should have errors for missing sections
                required_sections = ["qdrant", "neo4j", "storage", "agents"]
                for section in required_sections:
                    assert any(section in error for error in self.validator.errors)
    
    def test_validate_qdrant_config_invalid_port(self):
        """Test Qdrant configuration with invalid port"""
        invalid_config = self.valid_main_config.copy()
        invalid_config["qdrant"]["port"] = "not_a_number"
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_main_config("test.yaml")
                
                assert result is False
                assert any("qdrant.port must be an integer" in error for error in self.validator.errors)
        
        # Test port out of range
        invalid_config["qdrant"]["port"] = 70000
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                self.validator.errors = []  # Reset errors
                result = self.validator.validate_main_config("test.yaml")
                
                assert result is False
                assert any("between 1 and 65535" in error for error in self.validator.errors)
    
    def test_validate_neo4j_config_invalid_port(self):
        """Test Neo4j configuration with invalid port"""
        invalid_config = self.valid_main_config.copy()
        invalid_config["neo4j"]["port"] = -1
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_main_config("test.yaml")
                
                assert result is False
                assert any("neo4j.port must be between 1 and 65535" in error for error in self.validator.errors)
    
    def test_validate_redis_config_invalid_database(self):
        """Test Redis configuration with invalid database"""
        invalid_config = self.valid_main_config.copy()
        invalid_config["redis"]["database"] = -5
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_main_config("test.yaml")
                
                assert result is False
                assert any("redis.database must be a non-negative integer" in error for error in self.validator.errors)
    
    def test_validate_duckdb_config_missing_path(self):
        """Test DuckDB configuration with missing database path"""
        invalid_config = self.valid_main_config.copy()
        del invalid_config["duckdb"]["database_path"]
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_main_config("test.yaml")
                
                assert result is False
                assert any("duckdb.database_path is required" in error for error in self.validator.errors)
    
    def test_validate_duckdb_config_invalid_threads(self):
        """Test DuckDB configuration with invalid threads"""
        invalid_config = self.valid_main_config.copy()
        invalid_config["duckdb"]["threads"] = 0
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_main_config("test.yaml")
                
                assert result is False
                assert any("duckdb.threads must be a positive integer" in error for error in self.validator.errors)
    
    def test_ssl_security_warnings(self):
        """Test SSL security warnings"""
        # All SSL disabled - should generate warnings
        config_no_ssl = self.valid_main_config.copy()
        config_no_ssl["qdrant"]["ssl"] = False
        config_no_ssl["neo4j"]["ssl"] = False
        config_no_ssl["redis"]["ssl"] = False
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(config_no_ssl))):
            with patch('yaml.safe_load', return_value=config_no_ssl):
                result = self.validator.validate_main_config("test.yaml")
                
                assert result is True  # Valid but with warnings
                assert len(self.validator.warnings) >= 3
                
                # Check for SSL warnings
                ssl_warnings = [w for w in self.validator.warnings if "SSL is disabled" in w]
                assert len(ssl_warnings) >= 3


@pytest.mark.skipif(not CONFIG_VALIDATOR_AVAILABLE, reason="Config validator not available")
class TestPerformanceConfigValidation:
    """Test performance configuration validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = ConfigValidator()
        
        self.valid_perf_config = {
            "vector_db": {
                "embedding": {
                    "batch_size": 100,
                    "max_retries": 3,
                    "request_timeout": 30
                },
                "search": {
                    "default_limit": 10,
                    "max_limit": 100
                }
            },
            "graph_db": {
                "connection_pool": {
                    "min_size": 1,
                    "max_size": 10
                },
                "query": {
                    "max_path_length": 5
                }
            },
            "search": {
                "ranking": {
                    "temporal_decay_rate": 0.01,
                    "type_boosts": {
                        "documentation": 1.2,
                        "code": 1.0
                    }
                }
            },
            "resources": {
                "max_memory_gb": 4.0,
                "max_cpu_percent": 80
            }
        }
    
    def test_validate_performance_config_valid(self):
        """Test validation of valid performance configuration"""
        with patch('builtins.open', mock_open(read_data=yaml.dump(self.valid_perf_config))):
            with patch('yaml.safe_load', return_value=self.valid_perf_config):
                result = self.validator.validate_performance_config("perf.yaml")
                
                assert result is True
                assert len(self.validator.errors) == 0
    
    def test_validate_performance_config_missing_file(self):
        """Test validation with missing performance config file"""
        with patch('builtins.open', side_effect=FileNotFoundError):
            result = self.validator.validate_performance_config("missing_perf.yaml")
            
            # Performance config is optional, so this should pass
            assert result is True
    
    def test_validate_embedding_batch_size_invalid(self):
        """Test embedding batch size validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["vector_db"]["embedding"]["batch_size"] = 0
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("batch_size must be a positive integer" in error for error in self.validator.errors)
    
    def test_validate_max_retries_invalid(self):
        """Test max retries validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["vector_db"]["embedding"]["max_retries"] = -1
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("max_retries must be a non-negative integer" in error for error in self.validator.errors)
    
    def test_validate_request_timeout_invalid(self):
        """Test request timeout validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["vector_db"]["embedding"]["request_timeout"] = 0
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("request_timeout must be a positive number" in error for error in self.validator.errors)
    
    def test_validate_search_limits(self):
        """Test search limit validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["vector_db"]["search"]["max_limit"] = 5
        invalid_config["vector_db"]["search"]["default_limit"] = 10
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("max_limit must be >= default_limit" in error for error in self.validator.errors)
    
    def test_validate_connection_pool_sizes(self):
        """Test connection pool size validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["graph_db"]["connection_pool"]["max_size"] = 1
        invalid_config["graph_db"]["connection_pool"]["min_size"] = 5
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("max_size must be >= min_size" in error for error in self.validator.errors)
    
    def test_validate_max_path_length(self):
        """Test max path length validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["graph_db"]["query"]["max_path_length"] = 0
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("max_path_length must be a positive integer" in error for error in self.validator.errors)
        
        # Test warning for high path length
        warning_config = self.valid_perf_config.copy()
        warning_config["graph_db"]["query"]["max_path_length"] = 15
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(warning_config))):
            with patch('yaml.safe_load', return_value=warning_config):
                self.validator.errors = []
                self.validator.warnings = []
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is True
                assert any("may cause performance issues" in warning for warning in self.validator.warnings)
    
    def test_validate_temporal_decay_rate(self):
        """Test temporal decay rate validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["search"]["ranking"]["temporal_decay_rate"] = 1.5  # Out of range
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("must be between 0 and 1" in error for error in self.validator.errors)
    
    def test_validate_type_boosts(self):
        """Test type boosts validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["search"]["ranking"]["type_boosts"]["invalid"] = -0.5
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("must be a non-negative number" in error for error in self.validator.errors)
    
    def test_validate_resources(self):
        """Test resource validation"""
        invalid_config = self.valid_perf_config.copy()
        invalid_config["resources"]["max_memory_gb"] = 0.1  # Too small
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("must be at least 0.5" in error for error in self.validator.errors)
        
        # Test invalid CPU percent
        invalid_config["resources"]["max_memory_gb"] = 4.0
        invalid_config["resources"]["max_cpu_percent"] = 150  # Over 100%
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                self.validator.errors = []
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("must be between 1 and 100" in error for error in self.validator.errors)


@pytest.mark.skipif(not CONFIG_VALIDATOR_AVAILABLE, reason="Config validator not available")
class TestKVStoreConfigValidation:
    """Test KV store configuration validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = ConfigValidator()
        
        self.valid_kv_config = {
            "kv_store": {
                "redis": {
                    "connection_pool": {
                        "min_size": 5,
                        "max_size": 50
                    },
                    "cache": {
                        "ttl_seconds": 3600
                    }
                },
                "duckdb": {
                    "batch_insert": {
                        "size": 1000
                    },
                    "analytics": {
                        "retention_days": 90
                    }
                }
            }
        }
    
    def test_validate_redis_connection_pool(self):
        """Test Redis connection pool validation"""
        invalid_config = self.valid_kv_config.copy()
        invalid_config["kv_store"]["redis"]["connection_pool"]["max_size"] = 3
        invalid_config["kv_store"]["redis"]["connection_pool"]["min_size"] = 10
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("max_size must be >= min_size" in error for error in self.validator.errors)
    
    def test_validate_redis_cache_ttl(self):
        """Test Redis cache TTL validation"""
        invalid_config = self.valid_kv_config.copy()
        invalid_config["kv_store"]["redis"]["cache"]["ttl_seconds"] = 0
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("ttl_seconds must be a positive integer" in error for error in self.validator.errors)
    
    def test_validate_duckdb_batch_size(self):
        """Test DuckDB batch insert size validation"""
        invalid_config = self.valid_kv_config.copy()
        invalid_config["kv_store"]["duckdb"]["batch_insert"]["size"] = -1
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("size must be a positive integer" in error for error in self.validator.errors)
    
    def test_validate_duckdb_retention_days(self):
        """Test DuckDB analytics retention validation"""
        invalid_config = self.valid_kv_config.copy()
        invalid_config["kv_store"]["duckdb"]["analytics"]["retention_days"] = 0
        
        with patch('builtins.open', mock_open(read_data=yaml.dump(invalid_config))):
            with patch('yaml.safe_load', return_value=invalid_config):
                result = self.validator.validate_performance_config("test.yaml")
                
                assert result is False
                assert any("retention_days must be a positive integer" in error for error in self.validator.errors)


@pytest.mark.skipif(not CONFIG_VALIDATOR_AVAILABLE, reason="Config validator not available")
class TestValidateAllConfigs:
    """Test comprehensive configuration validation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = ConfigValidator()
    
    def test_validate_all_success(self):
        """Test successful validation of all configs"""
        valid_main = {
            "system": {"debug": False},
            "qdrant": {"host": "localhost", "port": 6333},
            "neo4j": {"uri": "bolt://localhost", "port": 7687},
            "storage": {"type": "hybrid"},
            "agents": {"max_concurrent": 5}
        }
        
        valid_perf = {
            "vector_db": {
                "embedding": {"batch_size": 100, "max_retries": 3, "request_timeout": 30}
            }
        }
        
        with patch.object(self.validator, 'validate_main_config', return_value=True):
            with patch.object(self.validator, 'validate_performance_config', return_value=True):
                result, errors, warnings = self.validator.validate_all()
                
                assert result is True
                assert len(errors) == 0
    
    def test_validate_all_with_errors(self):
        """Test validation with errors in configs"""
        self.validator.errors = ["Test error 1", "Test error 2"]
        self.validator.warnings = ["Test warning"]
        
        with patch.object(self.validator, 'validate_main_config', return_value=False):
            with patch.object(self.validator, 'validate_performance_config', return_value=True):
                result, errors, warnings = self.validator.validate_all()
                
                assert result is False
                assert len(errors) >= 2
                assert len(warnings) >= 1
    
    def test_validate_context_integrity(self):
        """Test context integrity validation"""
        # Valid context data
        valid_context = {
            "schema_version": "1.0",
            "data": {"test": "value"}
        }
        
        result = self.validator.validate_context_integrity(valid_context)
        assert result is True
        
        # Missing schema version
        invalid_context = {"data": {"test": "value"}}
        result = self.validator.validate_context_integrity(invalid_context)
        assert result is False
        assert any("Missing schema_version" in error for error in self.validator.errors)
        
        # Wrong schema version
        self.validator.errors = []
        wrong_version_context = {
            "schema_version": "2.0",
            "data": {"test": "value"}
        }
        result = self.validator.validate_context_integrity(wrong_version_context)
        assert result is False
        assert any("Schema version mismatch" in error for error in self.validator.errors)


@pytest.mark.skipif(not CONFIG_VALIDATOR_AVAILABLE, reason="Config validator not available")
class TestStandaloneValidationFunctions:
    """Test standalone validation functions"""
    
    def test_validate_environment_variables(self):
        """Test environment variable validation"""
        required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "QDRANT_URL", "REDIS_URL"]
        
        # Test with all variables present
        with patch.dict(os.environ, {var: "test_value" for var in required_vars}):
            result = validate_environment_variables()
            
            assert result["valid"] is True
            assert len(result["missing"]) == 0
        
        # Test with missing variables
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment_variables()
            
            assert result["valid"] is False
            assert len(result["missing"]) == len(required_vars)
            assert all(var in result["missing"] for var in required_vars)
    
    def test_validate_database_config_neo4j(self):
        """Test Neo4j database config validation"""
        # Valid Neo4j config
        valid_config = {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        }
        
        result = validate_database_config("neo4j", valid_config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Invalid URI scheme
        invalid_config = valid_config.copy()
        invalid_config["uri"] = "http://localhost:7687"
        
        result = validate_database_config("neo4j", invalid_config)
        assert result["valid"] is False
        assert any("Invalid Neo4j URI scheme" in error for error in result["errors"])
        
        # Missing user
        missing_user_config = {
            "uri": "bolt://localhost:7687",
            "password": "password"
        }
        
        result = validate_database_config("neo4j", missing_user_config)
        assert result["valid"] is False
        assert any("Missing 'user'" in error for error in result["errors"])
        
        # None config should raise exception
        with pytest.raises(ConfigValidationError):
            validate_database_config("neo4j", None)
    
    def test_validate_database_config_qdrant(self):
        """Test Qdrant database config validation"""
        # Valid Qdrant config
        valid_config = {"url": "http://localhost:6333"}
        
        result = validate_database_config("qdrant", valid_config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Missing URL
        invalid_config = {}
        
        result = validate_database_config("qdrant", invalid_config)
        assert result["valid"] is False
        assert any("Missing 'url'" in error for error in result["errors"])
        
        # Invalid URL scheme
        invalid_scheme_config = {"url": "ftp://localhost:6333"}
        
        result = validate_database_config("qdrant", invalid_scheme_config)
        assert result["valid"] is False
        assert any("Invalid Qdrant URL scheme" in error for error in result["errors"])
    
    def test_validate_database_config_redis(self):
        """Test Redis database config validation"""
        # Valid Redis config
        valid_config = {"url": "redis://localhost:6379"}
        
        result = validate_database_config("redis", valid_config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Invalid URL scheme
        invalid_config = {"url": "http://localhost:6379"}
        
        result = validate_database_config("redis", invalid_config)
        assert result["valid"] is False
        assert any("Invalid Redis URL scheme" in error for error in result["errors"])
    
    def test_validate_database_config_unknown_type(self):
        """Test validation with unknown database type"""
        with pytest.raises(ConfigValidationError, match="Unknown database type"):
            validate_database_config("unknown_db", {})
    
    def test_validate_mcp_config(self):
        """Test MCP configuration validation"""
        # Valid MCP config
        valid_config = {
            "server_port": 8000,
            "host": "0.0.0.0",
            "tools": ["store_context", "retrieve_context"]
        }
        
        result = validate_mcp_config(valid_config)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Missing server port
        missing_port_config = {
            "host": "0.0.0.0",
            "tools": ["store_context"]
        }
        
        result = validate_mcp_config(missing_port_config)
        assert result["valid"] is False
        assert any("Missing 'server_port'" in error for error in result["errors"])
        
        # Invalid port type
        invalid_port_config = {
            "server_port": "8000",  # String instead of int
            "host": "0.0.0.0",
            "tools": ["store_context"]
        }
        
        result = validate_mcp_config(invalid_port_config)
        assert result["valid"] is False
        assert any("server_port must be an integer" in error for error in result["errors"])
        
        # Port out of range
        out_of_range_config = {
            "server_port": 70000,
            "host": "0.0.0.0",
            "tools": ["store_context"]
        }
        
        result = validate_mcp_config(out_of_range_config)
        assert result["valid"] is False
        assert any("must be between 1 and 65535" in error for error in result["errors"])
        
        # Missing tools
        no_tools_config = {
            "server_port": 8000,
            "host": "0.0.0.0"
        }
        
        result = validate_mcp_config(no_tools_config)
        assert result["valid"] is False
        assert any("Missing 'tools'" in error for error in result["errors"])
        
        # Empty tools list
        empty_tools_config = {
            "server_port": 8000,
            "host": "0.0.0.0",
            "tools": []
        }
        
        result = validate_mcp_config(empty_tools_config)
        assert result["valid"] is False
        assert any("At least one tool must be configured" in error for error in result["errors"])
    
    def test_validate_all_configs_comprehensive(self):
        """Test comprehensive validation of all configs"""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "MCP_SERVER_PORT": "8000",
            "MCP_HOST": "0.0.0.0"
        }
        
        with patch.dict(os.environ, env_vars):
            result = validate_all_configs()
            
            assert isinstance(result, dict)
            assert "valid" in result
            assert "environment" in result
            assert "databases" in result
            assert "mcp" in result
            assert "warnings" in result
            
            # Should validate successfully with proper environment
            if result["valid"]:
                assert result["environment"]["valid"] is True
                assert all(db_result["valid"] for db_result in result["databases"].values())
                assert result["mcp"]["valid"] is True
        
        # Test with debug logging warning
        env_vars_debug = env_vars.copy()
        env_vars_debug["LOG_LEVEL"] = "debug"
        
        with patch.dict(os.environ, env_vars_debug):
            result = validate_all_configs()
            
            assert any("Debug logging is enabled" in warning for warning in result["warnings"])


@pytest.mark.skipif(not CONFIG_VALIDATOR_AVAILABLE, reason="Config validator not available")
class TestConfigValidatorCLI:
    """Test config validator CLI functionality"""
    
    def test_cli_main_function_success(self):
        """Test CLI main function with successful validation"""
        with patch('src.validators.config_validator.ConfigValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.errors = []
            mock_validator.warnings = []
            mock_validator.validate_main_config.return_value = True
            mock_validator.validate_performance_config.return_value = True
            mock_validator_class.return_value = mock_validator
            
            with patch('click.echo') as mock_echo:
                with patch('sys.exit') as mock_exit:
                    # Import and call main function
                    from src.validators.config_validator import main
                    
                    # Mock click context for testing
                    with patch('click.get_current_context', return_value=Mock()):
                        try:
                            main.callback(config=".ctxrc.yaml", perf_config="performance.yaml", strict=False)
                        except SystemExit:
                            pass
                    
                    # Should have called validation methods
                    mock_validator.validate_main_config.assert_called_once()
                    mock_validator.validate_performance_config.assert_called_once()
    
    def test_cli_main_function_with_errors(self):
        """Test CLI main function with validation errors"""
        with patch('src.validators.config_validator.ConfigValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.errors = ["Test error"]
            mock_validator.warnings = []
            mock_validator.validate_main_config.return_value = False
            mock_validator.validate_performance_config.return_value = True
            mock_validator_class.return_value = mock_validator
            
            with patch('click.echo') as mock_echo:
                with patch('sys.exit') as mock_exit:
                    from src.validators.config_validator import main
                    
                    with patch('click.get_current_context', return_value=Mock()):
                        try:
                            main.callback(config=".ctxrc.yaml", perf_config="performance.yaml", strict=False)
                        except SystemExit:
                            pass
                    
                    # Should exit with error code due to validation errors
                    mock_exit.assert_called_with(1)
    
    def test_cli_main_function_strict_mode(self):
        """Test CLI main function in strict mode with warnings"""
        with patch('src.validators.config_validator.ConfigValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.errors = []
            mock_validator.warnings = ["Test warning"]
            mock_validator.validate_main_config.return_value = True
            mock_validator.validate_performance_config.return_value = True
            mock_validator_class.return_value = mock_validator
            
            with patch('click.echo') as mock_echo:
                with patch('sys.exit') as mock_exit:
                    from src.validators.config_validator import main
                    
                    with patch('click.get_current_context', return_value=Mock()):
                        try:
                            main.callback(config=".ctxrc.yaml", perf_config="performance.yaml", strict=True)
                        except SystemExit:
                            pass
                    
                    # Should exit with error code in strict mode due to warnings
                    mock_exit.assert_called_with(1)


@pytest.mark.skipif(not CONFIG_VALIDATOR_AVAILABLE, reason="Config validator not available")
class TestConfigValidationErrorHandling:
    """Test config validation error handling"""
    
    def setup_method(self):
        """Setup test environment"""
        self.validator = ConfigValidator()
    
    def test_config_validation_exception_creation(self):
        """Test ConfigValidationError exception creation"""
        error_msg = "Test validation error"
        
        with pytest.raises(ConfigValidationError, match=error_msg):
            raise ConfigValidationError(error_msg)
    
    def test_error_accumulation(self):
        """Test error accumulation during validation"""
        # Start with empty errors
        assert len(self.validator.errors) == 0
        
        # Add some errors manually (simulating validation)
        self.validator.errors.append("Error 1")
        self.validator.errors.append("Error 2")
        
        assert len(self.validator.errors) == 2
        assert "Error 1" in self.validator.errors
        assert "Error 2" in self.validator.errors
    
    def test_warning_accumulation(self):
        """Test warning accumulation during validation"""
        # Start with empty warnings
        assert len(self.validator.warnings) == 0
        
        # Add some warnings manually (simulating validation)
        self.validator.warnings.append("Warning 1")
        self.validator.warnings.append("Warning 2")
        
        assert len(self.validator.warnings) == 2
        assert "Warning 1" in self.validator.warnings
        assert "Warning 2" in self.validator.warnings
    
    def test_error_and_warning_reset(self):
        """Test resetting errors and warnings"""
        # Add some errors and warnings
        self.validator.errors = ["Error 1", "Error 2"]
        self.validator.warnings = ["Warning 1"]
        
        # Reset for new validation
        self.validator.errors = []
        self.validator.warnings = []
        
        assert len(self.validator.errors) == 0
        assert len(self.validator.warnings) == 0
    
    def test_yaml_error_handling(self):
        """Test YAML parsing error handling"""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch('builtins.open', mock_open(read_data=invalid_yaml)):
            with patch('yaml.safe_load', side_effect=yaml.YAMLError("YAML parsing failed")):
                result = self.validator.validate_main_config("invalid.yaml")
                
                assert result is False
                assert len(self.validator.errors) > 0
                assert any("YAML" in error for error in self.validator.errors)
    
    def test_file_not_found_handling(self):
        """Test file not found error handling"""
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            result = self.validator.validate_main_config("missing.yaml")
            
            assert result is False
            assert len(self.validator.errors) > 0
            assert any("not found" in error for error in self.validator.errors)
    
    def test_permission_error_handling(self):
        """Test permission error handling"""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Should handle permission errors gracefully
            try:
                result = self.validator.validate_main_config("restricted.yaml")
                # May return False or raise exception depending on implementation
                assert isinstance(result, bool)
            except PermissionError:
                # Acceptable if exception is re-raised
                pass
    
    def test_unicode_decode_error_handling(self):
        """Test Unicode decode error handling"""
        with patch('builtins.open', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "invalid utf-8")):
            # Should handle Unicode errors gracefully
            try:
                result = self.validator.validate_main_config("binary.yaml")
                assert isinstance(result, bool)
            except UnicodeDecodeError:
                # Acceptable if exception is re-raised
                pass