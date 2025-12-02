#!/usr/bin/env python3
"""
Test suite for storage/neo4j_client.py - Neo4j database client tests
"""
import pytest
import sys
import tempfile
import yaml
from unittest.mock import patch, MagicMock, mock_open, call
from neo4j.exceptions import AuthError, ServiceUnavailable

# Mock dependencies to avoid import issues
import sys
from unittest.mock import MagicMock

# Mock the types module imports
types_mock = MagicMock()
types_mock.JSON = dict
types_mock.NodeID = str  
types_mock.QueryResult = list
sys.modules['types'] = types_mock

# Mock config error imports
config_error_mock = MagicMock()
config_error_mock.ConfigFileNotFoundError = Exception
config_error_mock.ConfigParseError = Exception
sys.modules['core.config_error'] = config_error_mock

# Mock test config
test_config_mock = MagicMock()
test_config_mock.get_test_config.return_value = {
    "neo4j": {"database": "test_context", "host": "localhost", "port": 7687}
}
sys.modules['core.test_config'] = test_config_mock

# Now import the module
import importlib.util
import os

module_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'storage', 'neo4j_client.py')
spec = importlib.util.spec_from_file_location("neo4j_client", module_path)
neo4j_client_module = importlib.util.module_from_spec(spec)
sys.modules['neo4j_client'] = neo4j_client_module
spec.loader.exec_module(neo4j_client_module)

Neo4jInitializer = neo4j_client_module.Neo4jInitializer


class TestNeo4jInitializerInit:
    """Test suite for Neo4jInitializer initialization"""

    def test_init_with_config_dict(self):
        """Test initialization with provided config dictionary"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        assert initializer.config == config
        assert initializer.database == "test_db"
        assert initializer.driver is None
        assert initializer.test_mode is False

    def test_init_with_test_mode(self):
        """Test initialization with test mode enabled"""
        initializer = Neo4jInitializer(test_mode=True)
        
        assert initializer.test_mode is True
        assert "neo4j" in initializer.config
        assert initializer.database == "test_context"  # From test config

    def test_init_with_config_path(self):
        """Test initialization with config file path"""
        config_data = {"neo4j": {"database": "custom_db"}}
        
        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            with patch("yaml.safe_load", return_value=config_data):
                initializer = Neo4jInitializer(config_path="test.yaml")
        
        assert initializer.config == config_data
        assert initializer.database == "custom_db"

    def test_init_default_database_name(self):
        """Test initialization with default database name"""
        config = {"neo4j": {}}
        initializer = Neo4jInitializer(config=config)
        
        assert initializer.database == "context_graph"


class TestLoadConfig:
    """Test suite for configuration loading"""

    def test_load_config_success(self):
        """Test successful config loading"""
        config_data = {"neo4j": {"host": "localhost"}}
        yaml_content = yaml.dump(config_data)
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            initializer = Neo4jInitializer(test_mode=False)
            result = initializer._load_config("test.yaml")
        
        assert result == config_data

    def test_load_config_file_not_found_test_mode(self):
        """Test config loading when file not found in test mode"""
        with patch("builtins.open", side_effect=FileNotFoundError):
            initializer = Neo4jInitializer(test_mode=True)
            result = initializer._load_config("missing.yaml")
        
        # Should return test config
        assert "neo4j" in result
        assert result["neo4j"]["database"] == "test_context"

    def test_load_config_file_not_found_production(self):
        """Test config loading when file not found in production mode"""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    initializer = Neo4jInitializer(test_mode=False)
                    initializer._load_config("missing.yaml")
        
        mock_echo.assert_called_with("Error: missing.yaml not found", err=True)
        mock_exit.assert_called_with(1)

    def test_load_config_yaml_error_test_mode(self):
        """Test config loading with YAML error in test mode"""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("Parse error")):
                initializer = Neo4jInitializer(test_mode=True)
                result = initializer._load_config("test.yaml")
        
        # Should return test config
        assert "neo4j" in result

    def test_load_config_yaml_error_production(self):
        """Test config loading with YAML error in production mode"""
        from src.core.config_error import ConfigParseError
        
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError("Parse error")):
                initializer = Neo4jInitializer(test_mode=False)
                
                with pytest.raises(ConfigParseError) as exc_info:
                    initializer._load_config("test.yaml")
                
                assert "Parse error" in str(exc_info.value)

    def test_load_config_invalid_format_test_mode(self):
        """Test config loading with invalid format in test mode"""
        with patch("builtins.open", mock_open(read_data="not_a_dict")):
            with patch("yaml.safe_load", return_value="not_a_dict"):
                initializer = Neo4jInitializer(test_mode=True)
                result = initializer._load_config("test.yaml")
        
        # Should return test config
        assert "neo4j" in result

    def test_load_config_invalid_format_production(self):
        """Test config loading with invalid format in production mode"""
        from src.core.config_error import ConfigParseError
        
        with patch("builtins.open", mock_open(read_data="not_a_dict")):
            with patch("yaml.safe_load", return_value="not_a_dict"):
                initializer = Neo4jInitializer(test_mode=False)
                
                with pytest.raises(ConfigParseError) as exc_info:
                    initializer._load_config("test.yaml")
                
                assert "Configuration must be a dictionary" in str(exc_info.value)


class TestConnect:
    """Test suite for Neo4j connection functionality"""

    def test_connect_success(self):
        """Test successful connection to Neo4j"""
        config = {"neo4j": {"host": "localhost", "port": 7687, "ssl": False}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", return_value=mock_driver):
            with patch("click.echo") as mock_echo:
                result = initializer.connect(username="neo4j", password="password")
        
        assert result is True
        assert initializer.driver == mock_driver
        mock_echo.assert_called_with("✓ Connected to Neo4j at bolt://localhost:7687")

    def test_connect_with_ssl(self):
        """Test connection with SSL enabled"""
        config = {"neo4j": {"host": "localhost", "port": 7687, "ssl": True}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", return_value=mock_driver) as mock_gdb:
            with patch("click.echo"):
                result = initializer.connect(username="neo4j", password="password")
        
        assert result is True
        mock_gdb.assert_called_with("bolt+s://localhost:7687", auth=("neo4j", "password"))

    def test_connect_no_password_prompt(self):
        """Test connection when no password provided (prompts for password)"""
        config = {"neo4j": {"host": "localhost"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", return_value=mock_driver):
            with patch("click.echo"):
                with patch("getpass.getpass", return_value="prompted_password"):
                    result = initializer.connect(username="neo4j")
        
        assert result is True

    def test_connect_service_unavailable(self):
        """Test connection when Neo4j service is unavailable"""
        config = {"neo4j": {"host": "localhost"}}
        initializer = Neo4jInitializer(config=config)
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", side_effect=ServiceUnavailable("Service unavailable")):
            with patch("click.echo") as mock_echo:
                result = initializer.connect(username="neo4j", password="password")
        
        assert result is False
        # Check that error messages were displayed
        assert any("Neo4j is not available" in str(call) for call in mock_echo.call_args_list)

    def test_connect_auth_error(self):
        """Test connection with authentication error"""
        config = {"neo4j": {"host": "localhost"}}
        initializer = Neo4jInitializer(config=config)
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", side_effect=AuthError("Auth failed")):
            with patch("click.echo") as mock_echo:
                result = initializer.connect(username="neo4j", password="wrong_password")
        
        assert result is False
        mock_echo.assert_any_call("✗ Authentication failed", err=True)

    def test_connect_generic_exception(self):
        """Test connection with generic exception"""
        config = {"neo4j": {"host": "localhost"}}
        initializer = Neo4jInitializer(config=config)
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", side_effect=Exception("Generic error")):
            with patch("click.echo") as mock_echo:
                result = initializer.connect(username="neo4j", password="password")
        
        assert result is False
        # Should sanitize the error message
        assert any("Failed to connect" in str(call) for call in mock_echo.call_args_list)

    def test_connect_password_sanitization(self):
        """Test that passwords are sanitized from error messages"""
        config = {"neo4j": {"host": "localhost"}}
        initializer = Neo4jInitializer(config=config)
        
        password = "secret123"
        error_with_password = f"Connection failed with password {password}"
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", side_effect=Exception(error_with_password)):
            with patch("click.echo") as mock_echo:
                result = initializer.connect(username="neo4j", password=password)
        
        assert result is False
        # Check that password was sanitized from error message
        error_calls = [call for call in mock_echo.call_args_list if "Failed to connect" in str(call)]
        assert len(error_calls) > 0
        assert password not in str(error_calls[0])


class TestCreateConstraints:
    """Test suite for constraint creation"""

    def test_create_constraints_success(self):
        """Test successful constraint creation"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.create_constraints()
        
        assert result is True
        # Verify session was called with correct database
        mock_driver.session.assert_called_with(database="test_db")
        # Should have multiple constraint creation calls
        assert mock_session.run.call_count > 0

    def test_create_constraints_not_connected(self):
        """Test constraint creation when not connected"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        with patch("click.echo") as mock_echo:
            result = initializer.create_constraints()
        
        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_create_constraints_exception(self):
        """Test constraint creation with exception"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_session.run.side_effect = Exception("Constraint creation failed")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.create_constraints()
        
        assert result is False
        mock_echo.assert_any_call("✗ Failed to create constraints: Constraint creation failed", err=True)

    def test_create_constraints_specific_constraints(self):
        """Test that specific constraints are created"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo"):
            result = initializer.create_constraints()
        
        assert result is True
        
        # Check that specific constraint queries were executed
        call_args = [call[0][0] for call in mock_session.run.call_args_list]
        
        # Should have constraints for Document.id, Agent.name, etc.
        document_constraint = any("Document" in query and "id" in query for query in call_args)
        agent_constraint = any("Agent" in query and "name" in query for query in call_args)
        
        assert document_constraint
        assert agent_constraint


class TestCreateIndexes:
    """Test suite for index creation"""

    def test_create_indexes_success(self):
        """Test successful index creation"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.create_indexes()
        
        assert result is True
        # Should have multiple index creation calls
        assert mock_session.run.call_count > 0

    def test_create_indexes_not_connected(self):
        """Test index creation when not connected"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        with patch("click.echo") as mock_echo:
            result = initializer.create_indexes()
        
        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_create_indexes_fulltext_indexes(self):
        """Test creation of fulltext indexes"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo"):
            result = initializer.create_indexes()
        
        assert result is True
        
        # Check that fulltext index queries were executed
        call_args = [call[0][0] for call in mock_session.run.call_args_list]
        fulltext_calls = [query for query in call_args if "fulltext" in query.lower()]
        
        assert len(fulltext_calls) > 0

    def test_create_indexes_existing_index_warning(self):
        """Test handling of existing index warnings"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        # First call succeeds, second call fails with "already exists"
        mock_session.run.side_effect = [None, Exception("Index already exists"), None]
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.create_indexes()
        
        assert result is True
        # Should not show warning for "already exists" errors


class TestSetupGraphSchema:
    """Test suite for graph schema setup"""

    def test_setup_graph_schema_success(self):
        """Test successful graph schema setup"""
        config = {
            "neo4j": {"database": "test_db"},
            "system": {"schema_version": "2.0.0", "created_date": "2023-01-01"}
        }
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.setup_graph_schema()
        
        assert result is True
        mock_echo.assert_any_call("✓ Graph schema initialized")
        # Should have multiple queries for system, agents, and document types
        assert mock_session.run.call_count >= 10

    def test_setup_graph_schema_not_connected(self):
        """Test schema setup when not connected"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        with patch("click.echo") as mock_echo:
            result = initializer.setup_graph_schema()
        
        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_setup_graph_schema_exception(self):
        """Test schema setup with exception"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_session.run.side_effect = Exception("Schema setup failed")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.setup_graph_schema()
        
        assert result is False
        mock_echo.assert_any_call("✗ Failed to setup schema: Schema setup failed", err=True)

    def test_setup_graph_schema_default_values(self):
        """Test schema setup with default system values"""
        config = {"neo4j": {"database": "test_db"}}  # No system config
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo"):
            result = initializer.setup_graph_schema()
        
        assert result is True
        
        # Check that default values were used in system node creation
        system_calls = [call for call in mock_session.run.call_args_list 
                       if len(call[0]) > 0 and "System" in call[0][0]]
        assert len(system_calls) > 0
        
        # Should use default version and date
        system_call = system_calls[0]
        if len(system_call) > 1 and isinstance(system_call[1], dict):
            params = system_call[1]
            assert params.get("version", "1.0.0") == "1.0.0"
            assert params.get("date", "2025-07-11") == "2025-07-11"


class TestVerifySetup:
    """Test suite for setup verification"""

    def test_verify_setup_success_with_apoc(self):
        """Test successful setup verification with APOC available"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        # Mock APOC query results
        mock_result = MagicMock()
        mock_result.__iter__.return_value = [
            {"label": "Document", "count": 5},
            {"label": "Agent", "count": 3}
        ]
        
        mock_session.run.side_effect = [mock_result, mock_result]  # Labels and relationships
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()
        
        assert result is True
        # Should display node and relationship counts
        mock_echo.assert_any_call("\nNode counts by label:")
        mock_echo.assert_any_call("\nRelationship counts by type:")

    def test_verify_setup_fallback_without_apoc(self):
        """Test setup verification fallback when APOC is not available"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        # First call (APOC) fails, second and third calls (fallback) succeed
        node_result = MagicMock()
        node_result.single.return_value = {"total": 10}
        
        rel_result = MagicMock()
        rel_result.single.return_value = {"total": 5}
        
        mock_session.run.side_effect = [
            Exception("APOC not available"),  # First call fails
            node_result,  # Node count
            rel_result    # Relationship count
        ]
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()
        
        assert result is True
        mock_echo.assert_any_call("\nTotal nodes: 10")
        mock_echo.assert_any_call("Total relationships: 5")

    def test_verify_setup_not_connected(self):
        """Test setup verification when not connected"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()
        
        assert result is False
        mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_verify_setup_complete_failure(self):
        """Test setup verification when all queries fail"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        initializer.driver = mock_driver
        
        mock_session.run.side_effect = Exception("Query failed")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()
        
        assert result is False
        mock_echo.assert_any_call("✗ Failed to verify setup: Query failed", err=True)


class TestClose:
    """Test suite for connection closure"""

    def test_close_with_driver(self):
        """Test closing connection when driver exists"""
        initializer = Neo4jInitializer(config={})
        mock_driver = MagicMock()
        initializer.driver = mock_driver
        
        initializer.close()
        
        mock_driver.close.assert_called_once()

    def test_close_without_driver(self):
        """Test closing connection when no driver exists"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        # Should not raise exception
        initializer.close()


class TestCreateNode:
    """Test suite for node creation"""

    def test_create_node_success(self):
        """Test successful node creation"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_record.__getitem__.return_value = "123"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        labels = ["Document", "Note"]
        properties = {"title": "Test", "content": "Content"}
        
        result = initializer.create_node(labels, properties)
        
        assert result == "123"
        mock_session.run.assert_called_once()

    def test_create_node_not_connected(self):
        """Test node creation when not connected"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            initializer.create_node(["Document"], {"title": "Test"})

    def test_create_node_invalid_label_type(self):
        """Test node creation with invalid label type"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        initializer.driver = MagicMock()
        
        with pytest.raises(ValueError, match="Label must be string"):
            initializer.create_node([123], {"title": "Test"})

    def test_create_node_invalid_label_length(self):
        """Test node creation with invalid label length"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        initializer.driver = MagicMock()
        
        # Too short
        with pytest.raises(ValueError, match="Label length must be 1-50 characters"):
            initializer.create_node([""], {"title": "Test"})
        
        # Too long
        with pytest.raises(ValueError, match="Label length must be 1-50 characters"):
            initializer.create_node(["a" * 51], {"title": "Test"})

    def test_create_node_invalid_label_format(self):
        """Test node creation with invalid label format"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        initializer.driver = MagicMock()
        
        invalid_labels = [
            "123invalid",  # Starts with number
            "invalid-@#$",  # Invalid characters
            "invalid space",  # Contains space
            "invalid--dash",  # Consecutive dashes
            "invalid__underscore"  # Consecutive underscores
        ]
        
        for label in invalid_labels:
            with pytest.raises(ValueError, match="Invalid label format"):
                initializer.create_node([label], {"title": "Test"})

    def test_create_node_valid_label_formats(self):
        """Test node creation with valid label formats"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_record.__getitem__.return_value = "123"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        valid_labels = [
            "Document",
            "Document_Type",
            "Document-Type",
            "a",
            "A123",
            "test_label-name"
        ]
        
        for label in valid_labels:
            result = initializer.create_node([label], {"title": "Test"})
            assert result == "123"

    def test_create_node_empty_labels(self):
        """Test node creation with empty labels list"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_record.__getitem__.return_value = "123"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        result = initializer.create_node([], {"title": "Test"})
        
        assert result == "123"
        # Should use default "Node" label
        call_args = mock_session.run.call_args[0][0]
        assert "CREATE (n:Node)" in call_args

    def test_create_node_no_result(self):
        """Test node creation when no result is returned"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        with pytest.raises(RuntimeError, match="Failed to create node - no result returned"):
            initializer.create_node(["Document"], {"title": "Test"})

    def test_create_node_exception(self):
        """Test node creation with exception"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_session.run.side_effect = Exception("Database error")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        with pytest.raises(RuntimeError, match="Failed to create node: Database error"):
            initializer.create_node(["Document"], {"title": "Test"})


class TestQuery:
    """Test suite for Cypher query execution"""

    def test_query_success(self):
        """Test successful query execution"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        # Mock result records
        mock_record1 = MagicMock()
        mock_record1.keys.return_value = ["id", "name"]
        mock_record1.__getitem__.side_effect = lambda key: {"id": 1, "name": "Test"}[key]
        
        mock_record2 = MagicMock()
        mock_record2.keys.return_value = ["id", "name"]
        mock_record2.__getitem__.side_effect = lambda key: {"id": 2, "name": "Test2"}[key]
        
        mock_result.__iter__.return_value = [mock_record1, mock_record2]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        cypher = "MATCH (n:Document) RETURN n.id as id, n.name as name"
        parameters = {"limit": 10}
        
        result = initializer.query(cypher, parameters)
        
        assert len(result) == 2
        assert result[0] == {"id": 1, "name": "Test"}
        assert result[1] == {"id": 2, "name": "Test2"}
        
        mock_session.run.assert_called_once_with(cypher, parameters)

    def test_query_not_connected(self):
        """Test query execution when not connected"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            initializer.query("MATCH (n) RETURN n")

    def test_query_no_parameters(self):
        """Test query execution without parameters"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_result.__iter__.return_value = []
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        result = initializer.query("MATCH (n) RETURN count(n)")
        
        assert result == []
        mock_session.run.assert_called_once_with("MATCH (n) RETURN count(n)", {})

    def test_query_with_neo4j_objects(self):
        """Test query execution with Neo4j object conversion"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        # Mock Neo4j object with __dict__
        mock_neo4j_object = MagicMock()
        mock_neo4j_object.__dict__ = {"property1": "value1", "property2": "value2"}
        
        mock_record = MagicMock()
        mock_record.keys.return_value = ["node"]
        mock_record.__getitem__.return_value = mock_neo4j_object
        
        mock_result.__iter__.return_value = [mock_record]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        result = initializer.query("MATCH (n) RETURN n as node")
        
        assert len(result) == 1
        assert result[0]["node"] == {"property1": "value1", "property2": "value2"}

    @patch('src.storage.neo4j_client.logger')
    def test_query_retry_on_session_closed(self, mock_logger):
        """Test query retry logic when Neo4j session is closed

        Verifies that:
        1. Query retries with fresh session when "Session is closed" error occurs
        2. Logger warning is called on retry attempt
        3. Query succeeds on second attempt
        """
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)

        mock_driver = MagicMock()

        # First attempt: session raises "Session is closed" error
        mock_session_closed = MagicMock()
        mock_session_closed.run.side_effect = Exception("Session is closed")

        # Second attempt: session succeeds
        mock_session_success = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.keys.return_value = ["id"]
        mock_record.__getitem__.return_value = 123
        mock_result.__iter__.return_value = [mock_record]
        mock_session_success.run.return_value = mock_result

        # Configure driver to return different sessions on each call
        mock_driver.session.return_value.__enter__.side_effect = [
            mock_session_closed,  # First attempt fails
            mock_session_success  # Second attempt succeeds
        ]

        initializer.driver = mock_driver

        # Execute query - should succeed after retry
        cypher = "MATCH (c:Context {id: $id}) RETURN c.id as id"
        parameters = {"id": "test-id"}
        result = initializer.query(cypher, parameters)

        # Verify query succeeded
        assert len(result) == 1
        assert result[0] == {"id": 123}

        # Verify logger.warning was called for retry attempt
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "session closed" in warning_call.lower()
        assert "retrying" in warning_call.lower()
        assert "attempt 1/2" in warning_call.lower()

        # Verify driver.session was called twice (initial + retry)
        assert mock_driver.session.call_count == 2

    def test_query_exception(self):
        """Test query execution with exception"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_session.run.side_effect = Exception("Query execution failed")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        with pytest.raises(RuntimeError, match="Failed to execute query: Query execution failed"):
            initializer.query("INVALID CYPHER")


class TestCreateRelationship:
    """Test suite for relationship creation"""

    def test_create_relationship_success(self):
        """Test successful relationship creation"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_record.__getitem__.return_value = "456"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        result = initializer.create_relationship(
            start_node="123",
            end_node="789",
            relationship_type="RELATES_TO",
            properties={"weight": 0.8}
        )
        
        assert result == "456"
        mock_session.run.assert_called_once()

    def test_create_relationship_not_connected(self):
        """Test relationship creation when not connected"""
        initializer = Neo4jInitializer(config={})
        initializer.driver = None
        
        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            initializer.create_relationship("123", "456", "RELATES_TO")

    def test_create_relationship_invalid_node_ids(self):
        """Test relationship creation with invalid node IDs"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        initializer.driver = MagicMock()
        
        # Non-numeric node IDs
        with pytest.raises(ValueError, match="Invalid node ID format"):
            initializer.create_relationship("abc", "456", "RELATES_TO")
        
        with pytest.raises(ValueError, match="Invalid node ID format"):
            initializer.create_relationship("123", "def", "RELATES_TO")

    def test_create_relationship_invalid_type(self):
        """Test relationship creation with invalid relationship type"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        initializer.driver = MagicMock()
        
        # Empty relationship type
        with pytest.raises(ValueError, match="Relationship type must be a non-empty string"):
            initializer.create_relationship("123", "456", "")
        
        # Non-string relationship type
        with pytest.raises(ValueError, match="Relationship type must be a non-empty string"):
            initializer.create_relationship("123", "456", None)

    def test_create_relationship_no_properties(self):
        """Test relationship creation without properties"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        mock_record.__getitem__.return_value = "456"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        result = initializer.create_relationship("123", "789", "RELATES_TO")
        
        assert result == "456"
        # Should pass empty dict for properties
        call_args = mock_session.run.call_args[1]
        assert call_args["properties"] == {}

    def test_create_relationship_nodes_not_exist(self):
        """Test relationship creation when nodes don't exist"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_result.single.return_value = None  # No result means nodes don't exist
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        with pytest.raises(RuntimeError, match="Failed to create relationship - nodes may not exist"):
            initializer.create_relationship("999", "888", "RELATES_TO")

    def test_create_relationship_exception(self):
        """Test relationship creation with exception"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        mock_driver = MagicMock()
        mock_session = MagicMock()
        
        mock_session.run.side_effect = Exception("Relationship creation failed")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        initializer.driver = mock_driver
        
        with pytest.raises(RuntimeError, match="Failed to create relationship: Relationship creation failed"):
            initializer.create_relationship("123", "456", "RELATES_TO")


class TestMainFunction:
    """Test suite for main CLI function"""

    def test_main_success_all_steps(self):
        """Test successful execution of main function with all steps"""
        mock_initializer = MagicMock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_constraints.return_value = True
        mock_initializer.create_indexes.return_value = True
        mock_initializer.setup_graph_schema.return_value = True
        mock_initializer.driver = MagicMock()
        mock_initializer.database = "test_db"
        
        # Mock database check (Community Edition)
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = None  # Database doesn't exist
        mock_session.run.side_effect = [
            mock_result,  # SHOW DATABASES
            Exception("UnsupportedAdministrationCommand")  # CREATE DATABASE fails
        ]
        mock_initializer.driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("src.storage.neo4j_client.Neo4jInitializer", return_value=mock_initializer):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    from src.storage.neo4j_client import main
                    
                    # Mock the click command to call main directly
                    main.callback(
                        username="neo4j",
                        password="password",
                        skip_constraints=False,
                        skip_indexes=False,
                        skip_schema=False
                    )
        
        # Verify all steps were called
        mock_initializer.connect.assert_called_once_with(username="neo4j", password="password")
        mock_initializer.create_constraints.assert_called_once()
        mock_initializer.create_indexes.assert_called_once()
        mock_initializer.setup_graph_schema.assert_called_once()
        mock_initializer.verify_setup.assert_called_once()
        mock_initializer.close.assert_called_once()
        
        # Should display success message
        mock_echo.assert_any_call("\n✓ Neo4j initialization complete!")

    def test_main_connection_failure(self):
        """Test main function when connection fails"""
        mock_initializer = MagicMock()
        mock_initializer.connect.return_value = False
        
        with patch("src.storage.neo4j_client.Neo4jInitializer", return_value=mock_initializer):
            with patch("sys.exit") as mock_exit:
                from src.storage.neo4j_client import main
                
                main.callback(
                    username="neo4j",
                    password="password",
                    skip_constraints=False,
                    skip_indexes=False,
                    skip_schema=False
                )
        
        mock_exit.assert_called_with(1)
        mock_initializer.close.assert_called_once()

    def test_main_skip_flags(self):
        """Test main function with skip flags"""
        mock_initializer = MagicMock()
        mock_initializer.connect.return_value = True
        mock_initializer.driver = MagicMock()
        mock_initializer.database = "test_db"
        
        # Mock successful database check
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"name": "test_db"}  # Database exists
        mock_session.run.return_value = mock_result
        mock_initializer.driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("src.storage.neo4j_client.Neo4jInitializer", return_value=mock_initializer):
            with patch("click.echo"):
                from src.storage.neo4j_client import main
                
                main.callback(
                    username="neo4j",
                    password="password",
                    skip_constraints=True,
                    skip_indexes=True,
                    skip_schema=True
                )
        
        # Verify skipped steps were not called
        mock_initializer.create_constraints.assert_not_called()
        mock_initializer.create_indexes.assert_not_called()
        mock_initializer.setup_graph_schema.assert_not_called()
        
        # But verification should still be called
        mock_initializer.verify_setup.assert_called_once()


class TestNeo4jIntegration:
    """Integration tests for Neo4j client functionality"""

    def test_full_workflow_simulation(self):
        """Test complete workflow simulation"""
        config = {
            "neo4j": {"database": "test_db", "host": "localhost", "port": 7687, "ssl": False},
            "system": {"schema_version": "1.0.0", "created_date": "2023-01-01"}
        }
        
        initializer = Neo4jInitializer(config=config)
        
        # Mock successful workflow
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = MagicMock()
        
        # Connection test
        mock_record_connection = MagicMock()
        mock_result_connection = MagicMock()
        mock_result_connection.single.return_value = mock_record_connection
        
        # Node creation
        mock_record.__getitem__.return_value = "123"
        mock_result.single.return_value = mock_record
        
        # Query result
        mock_query_record = MagicMock()
        mock_query_record.keys.return_value = ["id", "title"]
        mock_query_record.__getitem__.side_effect = lambda key: {"id": "123", "title": "Test"}[key]
        mock_query_result = MagicMock()
        mock_query_result.__iter__.return_value = [mock_query_record]
        
        mock_session.run.side_effect = [
            mock_result_connection,  # Connection test
            mock_result,  # Node creation
            mock_query_result  # Query
        ]
        mock_driver.session.return_value.__enter__.return_value = mock_session
        
        with patch("src.storage.neo4j_client.GraphDatabase.driver", return_value=mock_driver):
            with patch("click.echo"):
                # Connect
                connected = initializer.connect(username="neo4j", password="password")
                assert connected is True
                
                # Create node
                node_id = initializer.create_node(["Document"], {"title": "Test Document"})
                assert node_id == "123"
                
                # Query
                results = initializer.query("MATCH (n:Document) RETURN n.id as id, n.title as title")
                assert len(results) == 1
                assert results[0]["id"] == "123"
                assert results[0]["title"] == "Test"
                
                # Close
                initializer.close()

    def test_error_handling_edge_cases(self):
        """Test various error handling edge cases"""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config)
        
        # Test with no driver
        assert initializer.driver is None
        
        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            initializer.create_node(["Test"], {})
        
        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            initializer.query("MATCH (n) RETURN n")
        
        with pytest.raises(RuntimeError, match="Not connected to Neo4j"):
            initializer.create_relationship("1", "2", "RELATES_TO")

    def test_configuration_variations(self):
        """Test various configuration scenarios"""
        # Minimal config
        config1 = {"neo4j": {}}
        initializer1 = Neo4jInitializer(config=config1)
        assert initializer1.database == "context_graph"
        
        # Custom database
        config2 = {"neo4j": {"database": "custom_db"}}
        initializer2 = Neo4jInitializer(config=config2)
        assert initializer2.database == "custom_db"
        
        # Test mode
        initializer3 = Neo4jInitializer(test_mode=True)
        assert "neo4j" in initializer3.config
        assert initializer3.config["neo4j"]["database"] == "test_context"