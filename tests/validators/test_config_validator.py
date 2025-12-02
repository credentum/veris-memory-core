#!/usr/bin/env python3
"""
Comprehensive tests for src/validators/config_validator.py

Tests cover:
- ConfigValidator class: main config, performance config, and context validation
- ConfigValidationError exception handling  
- Standalone validation functions: environment, database, MCP configs
- CLI interface functionality
- Error accumulation and warning generation
- YAML parsing and validation logic
- Port range and type validation
- SSL security warnings
- Edge cases and malformed configurations
"""

import os
import tempfile
import yaml
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from src.validators.config_validator import (
    ConfigValidator,
    ConfigValidationError,
    validate_environment_variables,
    validate_database_config,
    validate_mcp_config,
    validate_all_configs,
    main
)


class TestConfigValidationError:
    """Test ConfigValidationError exception class."""

    def test_config_validation_error_creation(self):
        """Test creating ConfigValidationError with message."""
        error = ConfigValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_config_validation_error_inheritance(self):
        """Test ConfigValidationError inherits from Exception."""
        error = ConfigValidationError("Test")
        assert isinstance(error, Exception)


class TestConfigValidator:
    """Test ConfigValidator class for configuration validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ConfigValidator()

    def test_init(self):
        """Test ConfigValidator initialization."""
        validator = ConfigValidator()
        assert validator.errors == []
        assert validator.warnings == []

    def test_validate_main_config_file_not_found(self):
        """Test main config validation with missing file."""
        result = self.validator.validate_main_config("nonexistent.yaml")
        
        assert result is False
        assert len(self.validator.errors) == 1
        assert "not found" in self.validator.errors[0]

    def test_validate_main_config_invalid_yaml(self):
        """Test main config validation with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is False
            assert len(self.validator.errors) == 1
            assert "Invalid YAML" in self.validator.errors[0]
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_missing_required_sections(self):
        """Test main config validation with missing required sections."""
        config = {"storage": {}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is False
            assert len(self.validator.errors) == 4  # Missing system, qdrant, neo4j, agents
            missing_sections = ["system", "qdrant", "neo4j", "agents"]
            for section in missing_sections:
                assert any(section in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_valid_complete(self):
        """Test main config validation with valid complete configuration."""
        config = {
            "system": {"log_level": "info"},
            "qdrant": {"host": "localhost", "port": 6333, "ssl": True},
            "neo4j": {"host": "localhost", "port": 7687, "ssl": True},
            "redis": {"host": "localhost", "port": 6379, "database": 0, "ssl": True},
            "duckdb": {"database_path": "/tmp/test.db", "threads": 4},
            "storage": {"base_path": "/tmp"},
            "agents": {"enabled": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is True
            assert len(self.validator.errors) == 0
            assert len(self.validator.warnings) == 0
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_qdrant_invalid_port_type(self):
        """Test qdrant port validation with invalid type."""
        config = {
            "system": {}, "qdrant": {"port": "invalid"}, "neo4j": {},
            "storage": {}, "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is False
            assert any("qdrant.port must be an integer" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_qdrant_invalid_port_range(self):
        """Test qdrant port validation with invalid range."""
        config = {
            "system": {}, "qdrant": {"port": 70000}, "neo4j": {},
            "storage": {}, "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is False
            assert any("qdrant.port must be between 1 and 65535" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_neo4j_invalid_port(self):
        """Test neo4j port validation."""
        config = {
            "system": {}, "qdrant": {}, "neo4j": {"port": -1},
            "storage": {}, "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is False
            assert any("neo4j.port must be between 1 and 65535" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_redis_invalid_port_and_database(self):
        """Test redis port and database validation."""
        config = {
            "system": {}, "qdrant": {}, "neo4j": {},
            "redis": {"port": "invalid", "database": -1},
            "storage": {}, "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is False
            assert any("redis.port must be an integer" in error for error in self.validator.errors)
            assert any("redis.database must be a non-negative integer" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_duckdb_missing_path(self):
        """Test duckdb validation with missing database_path."""
        config = {
            "system": {}, "qdrant": {}, "neo4j": {},
            "duckdb": {"threads": "invalid"},
            "storage": {}, "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is False
            assert any("duckdb.database_path is required" in error for error in self.validator.errors)
            assert any("duckdb.threads must be a positive integer" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_main_config_ssl_warnings(self):
        """Test SSL security warnings generation."""
        config = {
            "system": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
            "redis": {"ssl": False},
            "storage": {},
            "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_main_config(temp_path)
            
            assert result is True
            assert len(self.validator.warnings) == 3
            assert any("SSL is disabled for Qdrant" in warning for warning in self.validator.warnings)
            assert any("SSL is disabled for Neo4j" in warning for warning in self.validator.warnings)
            assert any("SSL is disabled for Redis" in warning for warning in self.validator.warnings)
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_file_not_found(self):
        """Test performance config validation with missing file (should pass)."""
        result = self.validator.validate_performance_config("nonexistent.yaml")
        
        assert result is True
        assert len(self.validator.errors) == 0

    def test_validate_performance_config_invalid_yaml(self):
        """Test performance config validation with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: [[[")
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is False
            assert len(self.validator.errors) == 1
            assert "Invalid YAML" in self.validator.errors[0]
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_vector_db_embedding(self):
        """Test performance config vector_db embedding validation."""
        config = {
            "vector_db": {
                "embedding": {
                    "batch_size": -1,
                    "max_retries": -5,
                    "request_timeout": -1
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is False
            assert any("batch_size must be a positive integer" in error for error in self.validator.errors)
            assert any("max_retries must be a non-negative integer" in error for error in self.validator.errors)
            assert any("request_timeout must be a positive number" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_vector_db_search(self):
        """Test performance config vector_db search validation."""
        config = {
            "vector_db": {
                "search": {
                    "max_limit": 5,
                    "default_limit": 10
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is False
            assert any("max_limit must be >= default_limit" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_graph_db(self):
        """Test performance config graph_db validation."""
        config = {
            "graph_db": {
                "connection_pool": {
                    "max_size": 5,
                    "min_size": 10
                },
                "query": {
                    "max_path_length": -1
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is False
            assert any("max_size must be >= min_size" in error for error in self.validator.errors)
            assert any("max_path_length must be a positive integer" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_graph_db_warning(self):
        """Test performance config graph_db warning for high path length."""
        config = {
            "graph_db": {
                "query": {
                    "max_path_length": 15
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is True
            assert len(self.validator.warnings) == 1
            assert "may cause performance issues" in self.validator.warnings[0]
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_search_ranking(self):
        """Test performance config search ranking validation."""
        config = {
            "search": {
                "ranking": {
                    "temporal_decay_rate": 1.5,
                    "type_boosts": {
                        "doc": -1,
                        "code": "invalid"
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is False
            assert any("temporal_decay_rate must be between 0 and 1" in error for error in self.validator.errors)
            assert any("type_boosts.doc must be a non-negative number" in error for error in self.validator.errors)
            assert any("type_boosts.code must be a non-negative number" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_resources(self):
        """Test performance config resources validation."""
        config = {
            "resources": {
                "max_memory_gb": 0.1,
                "max_cpu_percent": 150
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is False
            assert any("max_memory_gb must be at least 0.5" in error for error in self.validator.errors)
            assert any("max_cpu_percent must be between 1 and 100" in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_performance_config_kv_store(self):
        """Test performance config KV store validation."""
        config = {
            "kv_store": {
                "redis": {
                    "connection_pool": {
                        "max_size": 5,
                        "min_size": 10
                    },
                    "cache": {
                        "ttl_seconds": -1
                    }
                },
                "duckdb": {
                    "batch_insert": {
                        "size": 0
                    },
                    "analytics": {
                        "retention_days": -1
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            result = self.validator.validate_performance_config(temp_path)
            
            assert result is False
            error_messages = [
                "connection_pool.max_size must be >= min_size",
                "ttl_seconds must be a positive integer",
                "batch_insert.size must be a positive integer",
                "retention_days must be a positive integer"
            ]
            for msg in error_messages:
                assert any(msg in error for error in self.validator.errors)
        finally:
            os.unlink(temp_path)

    def test_validate_all_success(self):
        """Test validate_all with successful validation."""
        # Create valid main config
        main_config = {
            "system": {},
            "qdrant": {"port": 6333},
            "neo4j": {"port": 7687},
            "storage": {},
            "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(main_config, f)
            main_path = f.name

        # Create valid performance config
        perf_config = {
            "resources": {
                "max_memory_gb": 4.0,
                "max_cpu_percent": 80
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(perf_config, f)
            perf_path = f.name

        try:
            validator = ConfigValidator()
            
            # Mock the file paths
            with patch.object(validator, 'validate_main_config') as mock_main:
                with patch.object(validator, 'validate_performance_config') as mock_perf:
                    mock_main.return_value = True
                    mock_perf.return_value = True
                    
                    valid, errors, warnings = validator.validate_all()
                    
                    assert valid is True
                    assert errors == []
                    assert warnings == []
        finally:
            os.unlink(main_path)
            os.unlink(perf_path)

    def test_validate_all_with_errors(self):
        """Test validate_all with validation errors."""
        validator = ConfigValidator()
        
        with patch.object(validator, 'validate_main_config') as mock_main:
            with patch.object(validator, 'validate_performance_config') as mock_perf:
                mock_main.return_value = False
                mock_perf.return_value = True
                
                # Mock the methods to add errors when called
                def add_main_errors():
                    validator.errors.append("Main config error")
                    return False
                    
                def add_perf_warnings():
                    validator.warnings.append("Warning message")
                    return True
                
                mock_main.side_effect = add_main_errors
                mock_perf.side_effect = add_perf_warnings
                
                valid, errors, warnings = validator.validate_all()
                
                assert valid is False
                assert len(errors) == 1
                assert len(warnings) == 1

    def test_validate_context_integrity_success(self):
        """Test context integrity validation with valid data."""
        context_data = {"schema_version": "1.0", "data": "test"}
        
        result = self.validator.validate_context_integrity(context_data)
        
        assert result is True
        assert len(self.validator.errors) == 0

    def test_validate_context_integrity_missing_schema(self):
        """Test context integrity validation with missing schema_version."""
        context_data = {"data": "test"}
        
        result = self.validator.validate_context_integrity(context_data)
        
        assert result is False
        assert len(self.validator.errors) == 1
        assert "Missing schema_version field" in self.validator.errors[0]

    def test_validate_context_integrity_wrong_version(self):
        """Test context integrity validation with wrong schema version."""
        context_data = {"schema_version": "2.0", "data": "test"}
        
        result = self.validator.validate_context_integrity(context_data)
        
        assert result is False
        assert len(self.validator.errors) == 1
        assert "Schema version mismatch" in self.validator.errors[0]


class TestValidateEnvironmentVariables:
    """Test validate_environment_variables function."""

    def test_all_variables_present(self):
        """Test validation when all required environment variables are present."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379"
        }
        
        with patch.dict(os.environ, env_vars):
            result = validate_environment_variables()
            
            assert result["valid"] is True
            assert result["missing"] == []

    def test_missing_variables(self):
        """Test validation when some environment variables are missing."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j"
            # Missing NEO4J_PASSWORD, QDRANT_URL, REDIS_URL
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            result = validate_environment_variables()
            
            assert result["valid"] is False
            assert len(result["missing"]) == 3
            assert "NEO4J_PASSWORD" in result["missing"]
            assert "QDRANT_URL" in result["missing"]
            assert "REDIS_URL" in result["missing"]

    def test_no_variables_present(self):
        """Test validation when no environment variables are present."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment_variables()
            
            assert result["valid"] is False
            assert len(result["missing"]) == 5


class TestValidateDatabaseConfig:
    """Test validate_database_config function."""

    def test_none_config_raises_exception(self):
        """Test that None config raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_database_config("neo4j", None)
        
        assert "cannot be None" in str(exc_info.value)

    def test_neo4j_valid_config(self):
        """Test Neo4j validation with valid configuration."""
        config = {
            "uri": "bolt://localhost:7687",
            "user": "neo4j",
            "password": "password"
        }
        
        result = validate_database_config("neo4j", config)
        
        assert result["valid"] is True
        assert result["errors"] == []

    def test_neo4j_missing_fields(self):
        """Test Neo4j validation with missing fields."""
        config = {"uri": "bolt://localhost:7687"}
        
        result = validate_database_config("neo4j", config)
        
        assert result["valid"] is False
        assert len(result["errors"]) == 2
        assert any("Missing 'user'" in error for error in result["errors"])
        assert any("Missing 'password'" in error for error in result["errors"])

    def test_neo4j_invalid_uri_scheme(self):
        """Test Neo4j validation with invalid URI scheme."""
        config = {
            "uri": "http://localhost:7687",
            "user": "neo4j",
            "password": "password"
        }
        
        result = validate_database_config("neo4j", config)
        
        assert result["valid"] is False
        assert any("Invalid Neo4j URI scheme" in error for error in result["errors"])

    def test_neo4j_valid_schemes(self):
        """Test Neo4j validation with all valid URI schemes."""
        valid_schemes = ["bolt://", "neo4j://", "bolt+s://", "neo4j+s://"]
        
        for scheme in valid_schemes:
            config = {
                "uri": f"{scheme}localhost:7687",
                "user": "neo4j",
                "password": "password"
            }
            
            result = validate_database_config("neo4j", config)
            assert result["valid"] is True

    def test_qdrant_valid_config(self):
        """Test Qdrant validation with valid configuration."""
        config = {"url": "http://localhost:6333"}
        
        result = validate_database_config("qdrant", config)
        
        assert result["valid"] is True
        assert result["errors"] == []

    def test_qdrant_missing_url(self):
        """Test Qdrant validation with missing URL."""
        config = {"host": "localhost"}
        
        result = validate_database_config("qdrant", config)
        
        assert result["valid"] is False
        assert any("Missing 'url'" in error for error in result["errors"])

    def test_qdrant_invalid_url_scheme(self):
        """Test Qdrant validation with invalid URL scheme."""
        config = {"url": "ftp://localhost:6333"}
        
        result = validate_database_config("qdrant", config)
        
        assert result["valid"] is False
        assert any("Invalid Qdrant URL scheme" in error for error in result["errors"])

    def test_redis_valid_config(self):
        """Test Redis validation with valid configuration."""
        config = {"url": "redis://localhost:6379"}
        
        result = validate_database_config("redis", config)
        
        assert result["valid"] is True
        assert result["errors"] == []

    def test_redis_missing_url(self):
        """Test Redis validation with missing URL."""
        config = {"host": "localhost"}
        
        result = validate_database_config("redis", config)
        
        assert result["valid"] is False
        assert any("Missing 'url'" in error for error in result["errors"])

    def test_redis_invalid_url_scheme(self):
        """Test Redis validation with invalid URL scheme."""
        config = {"url": "http://localhost:6379"}
        
        result = validate_database_config("redis", config)
        
        assert result["valid"] is False
        assert any("Invalid Redis URL scheme" in error for error in result["errors"])

    def test_unknown_database_type(self):
        """Test validation with unknown database type."""
        config = {"url": "test://localhost"}
        
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_database_config("unknown", config)
        
        assert "Unknown database type: unknown" in str(exc_info.value)


class TestValidateMcpConfig:
    """Test validate_mcp_config function."""

    def test_valid_mcp_config(self):
        """Test MCP validation with valid configuration."""
        config = {
            "server_port": 8000,
            "host": "localhost",
            "tools": ["store_context", "retrieve_context"]
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_server_port(self):
        """Test MCP validation with missing server_port."""
        config = {
            "host": "localhost",
            "tools": ["store_context"]
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is False
        assert any("Missing 'server_port'" in error for error in result["errors"])

    def test_invalid_port_type(self):
        """Test MCP validation with invalid port type."""
        config = {
            "server_port": "8000",
            "host": "localhost",
            "tools": ["store_context"]
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is False
        assert any("server_port must be an integer" in error for error in result["errors"])

    def test_invalid_port_range(self):
        """Test MCP validation with invalid port range."""
        config = {
            "server_port": 70000,
            "host": "localhost",
            "tools": ["store_context"]
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is False
        assert any("server_port must be between 1 and 65535" in error for error in result["errors"])

    def test_missing_host(self):
        """Test MCP validation with missing host."""
        config = {
            "server_port": 8000,
            "tools": ["store_context"]
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is False
        assert any("Missing 'host'" in error for error in result["errors"])

    def test_missing_tools(self):
        """Test MCP validation with missing tools."""
        config = {
            "server_port": 8000,
            "host": "localhost"
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is False
        assert any("Missing 'tools'" in error for error in result["errors"])

    def test_invalid_tools_type(self):
        """Test MCP validation with invalid tools type."""
        config = {
            "server_port": 8000,
            "host": "localhost",
            "tools": "store_context"
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is False
        assert any("tools must be a list" in error for error in result["errors"])

    def test_empty_tools_list(self):
        """Test MCP validation with empty tools list."""
        config = {
            "server_port": 8000,
            "host": "localhost",
            "tools": []
        }
        
        result = validate_mcp_config(config)
        
        assert result["valid"] is False
        assert any("At least one tool must be configured" in error for error in result["errors"])


class TestValidateAllConfigs:
    """Test validate_all_configs function."""

    def test_all_valid_configs(self):
        """Test comprehensive validation with all valid configurations."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "MCP_SERVER_PORT": "8000",
            "MCP_HOST": "localhost"
        }
        
        with patch.dict(os.environ, env_vars):
            result = validate_all_configs()
            
            assert result["valid"] is True
            assert result["environment"]["valid"] is True
            assert result["databases"]["neo4j"]["valid"] is True
            assert result["databases"]["qdrant"]["valid"] is True
            assert result["databases"]["redis"]["valid"] is True
            assert result["mcp"]["valid"] is True

    def test_missing_environment_variables(self):
        """Test comprehensive validation with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_all_configs()
            
            assert result["valid"] is False
            assert result["environment"]["valid"] is False
            assert len(result["environment"]["missing"]) == 5

    def test_invalid_database_configs(self):
        """Test comprehensive validation with invalid database configurations."""
        env_vars = {
            "NEO4J_URI": "http://invalid:7687",  # Invalid scheme
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "ftp://invalid:6333",  # Invalid scheme
            "REDIS_URL": "http://invalid:6379",  # Invalid scheme
            "MCP_SERVER_PORT": "8000",
            "MCP_HOST": "localhost"
        }
        
        with patch.dict(os.environ, env_vars):
            result = validate_all_configs()
            
            assert result["valid"] is False
            assert result["databases"]["neo4j"]["valid"] is False
            assert result["databases"]["qdrant"]["valid"] is False
            assert result["databases"]["redis"]["valid"] is False

    def test_debug_logging_warning(self):
        """Test warning generation for debug logging."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "QDRANT_URL": "http://localhost:6333",
            "REDIS_URL": "redis://localhost:6379",
            "LOG_LEVEL": "debug"
        }
        
        with patch.dict(os.environ, env_vars):
            result = validate_all_configs()
            
            assert len(result["warnings"]) == 1
            assert "Debug logging is enabled" in result["warnings"][0]

    def test_partial_environment_variables(self):
        """Test with only some environment variables set."""
        env_vars = {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password"
            # Missing QDRANT_URL and REDIS_URL
        }
        
        with patch.dict(os.environ, env_vars):
            result = validate_all_configs()
            
            assert result["valid"] is False
            assert result["environment"]["valid"] is False
            assert "neo4j" in result["databases"]
            assert "qdrant" not in result["databases"]
            assert "redis" not in result["databases"]


class TestCLIInterface:
    """Test CLI interface functionality."""

    def test_cli_success(self):
        """Test CLI with successful validation."""
        # Create valid config files with SSL enabled to avoid warnings
        main_config = {
            "system": {},
            "qdrant": {"port": 6333, "ssl": True},
            "neo4j": {"port": 7687, "ssl": True},
            "redis": {"ssl": True},
            "storage": {},
            "agents": {}
        }
        
        perf_config = {
            "resources": {
                "max_memory_gb": 4.0,
                "max_cpu_percent": 80
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(main_config, f)
            main_path = f.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(perf_config, f)
            perf_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(main, ['--config', main_path, '--perf-config', perf_path])
            
            assert result.exit_code == 0
            assert "All configurations are valid!" in result.output
        finally:
            os.unlink(main_path)
            os.unlink(perf_path)

    def test_cli_with_errors(self):
        """Test CLI with validation errors."""
        # Create invalid config file
        invalid_config = {"invalid": "config"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(main, ['--config', config_path])
            
            assert result.exit_code == 1
            assert "Errors:" in result.output
        finally:
            os.unlink(config_path)

    def test_cli_with_warnings_strict_mode(self):
        """Test CLI with warnings in strict mode."""
        # Create config with SSL warnings
        config_with_warnings = {
            "system": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
            "redis": {"ssl": False},
            "storage": {},
            "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_warnings, f)
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(main, ['--config', config_path, '--strict'])
            
            assert result.exit_code == 1
            assert "Warnings:" in result.output
        finally:
            os.unlink(config_path)

    def test_cli_with_warnings_non_strict(self):
        """Test CLI with warnings in non-strict mode."""
        # Create config with SSL warnings
        config_with_warnings = {
            "system": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
            "redis": {"ssl": False},
            "storage": {},
            "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_warnings, f)
            config_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(main, ['--config', config_path])
            
            assert result.exit_code == 0
            assert "Warnings:" in result.output
        finally:
            os.unlink(config_path)

    def test_cli_default_arguments(self):
        """Test CLI with default arguments."""
        runner = CliRunner()
        
        # This will fail because default files don't exist, but we test argument parsing
        result = runner.invoke(main, [])
        
        assert result.exit_code == 1
        assert "Configuration Validation" in result.output


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_error_accumulation(self):
        """Test that errors accumulate across multiple validations."""
        validator = ConfigValidator()
        
        # Add initial errors
        validator.errors = ["Initial error"]
        validator.warnings = ["Initial warning"]
        
        # Validate invalid config (should add more errors)
        config = {"invalid": "config"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            validator.validate_main_config(temp_path)
            
            # Should have accumulated more errors
            assert len(validator.errors) > 1
        finally:
            os.unlink(temp_path)

    def test_warning_accumulation(self):
        """Test that warnings accumulate properly."""
        validator = ConfigValidator()
        
        # Create config that generates multiple warnings
        config = {
            "system": {},
            "qdrant": {"ssl": False},
            "neo4j": {"ssl": False},
            "redis": {"ssl": False},
            "storage": {},
            "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            validator.validate_main_config(temp_path)
            
            assert len(validator.warnings) == 3
        finally:
            os.unlink(temp_path)

    def test_validate_all_resets_errors_and_warnings(self):
        """Test that validate_all resets errors and warnings."""
        validator = ConfigValidator()
        
        # Add initial errors and warnings
        validator.errors = ["Old error"]
        validator.warnings = ["Old warning"]
        
        with patch.object(validator, 'validate_main_config', return_value=True):
            with patch.object(validator, 'validate_performance_config', return_value=True):
                valid, errors, warnings = validator.validate_all()
                
                # Should have reset the lists
                assert validator.errors == []
                assert validator.warnings == []

    def test_empty_config_file(self):
        """Test validation with empty config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            validator = ConfigValidator()
            # Empty YAML files load as None, which causes TypeError when checking 'if section not in config'
            with pytest.raises(TypeError):
                validator.validate_main_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_null_config_file(self):
        """Test validation with null config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("null")  # Explicit null
            temp_path = f.name

        try:
            validator = ConfigValidator()
            # Null config should also cause TypeError
            with pytest.raises(TypeError):
                validator.validate_main_config(temp_path)
        finally:
            os.unlink(temp_path)

    def test_redis_config_edge_cases(self):
        """Test Redis configuration edge cases."""
        config = {
            "system": {}, "qdrant": {}, "neo4j": {},
            "redis": {
                "port": 65535,  # Max valid port
                "database": 0   # Min valid database
            },
            "storage": {}, "agents": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            validator = ConfigValidator()
            result = validator.validate_main_config(temp_path)
            
            # Should pass with valid edge values
            assert result is True
        finally:
            os.unlink(temp_path)