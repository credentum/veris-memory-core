#!/usr/bin/env python3
"""
Comprehensive tests for Neo4j client to achieve high coverage.

This test suite covers:
- Connection management with various configurations
- Constraint and index creation
- Graph schema setup and verification
- Error handling and edge cases
- Configuration loading and validation
- SSL/security settings
"""

import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.storage.neo4j_client import Neo4jInitializer


class TestNeo4jInitializer:
    """Test cases for Neo4jInitializer class."""

    def test_init_with_config_dict(self):
        """Test initialization with provided config dictionary."""
        config = {"neo4j": {"host": "test-host", "port": 7687, "database": "test_db"}}

        initializer = Neo4jInitializer(config=config, test_mode=True)

        assert initializer.config == config
        assert initializer.test_mode is True
        assert initializer.database == "test_db"
        assert initializer.driver is None

    def test_init_with_config_file(self):
        """Test initialization with config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {"neo4j": {"host": "localhost", "port": 7687, "database": "context_graph"}}
            yaml.dump(config, f)
            f.flush()

            initializer = Neo4jInitializer(config_path=f.name, test_mode=True)

            assert initializer.config == config
            assert initializer.database == "context_graph"

    def test_init_with_missing_config_test_mode(self):
        """Test initialization with missing config in test mode."""
        with patch("storage.neo4j_client.get_test_config") as mock_get_test_config:
            mock_config = {"neo4j": {"database": "test"}}
            mock_get_test_config.return_value = mock_config

            initializer = Neo4jInitializer(config_path="nonexistent.yaml", test_mode=True)

            assert initializer.config == mock_config
            mock_get_test_config.assert_called_once()

    def test_init_with_missing_config_production_mode(self):
        """Test initialization with missing config in production mode."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("click.echo") as mock_echo:
                with patch("sys.exit") as mock_exit:
                    Neo4jInitializer(config_path="nonexistent.yaml", test_mode=False)

                    mock_echo.assert_called()
                    mock_exit.assert_called_with(1)

    def test_load_config_invalid_yaml(self):
        """Test loading invalid YAML configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with patch("storage.neo4j_client.get_test_config") as mock_get_test_config:
                mock_config = {"neo4j": {"database": "test"}}
                mock_get_test_config.return_value = mock_config

                initializer = Neo4jInitializer(config_path=f.name, test_mode=True)

                assert initializer.config == mock_config
                mock_get_test_config.assert_called_once()

    def test_load_config_non_dict_content(self):
        """Test loading config with non-dictionary content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump("not a dictionary", f)
            f.flush()

            with patch("storage.neo4j_client.get_test_config") as mock_get_test_config:
                mock_config = {"neo4j": {"database": "test"}}
                mock_get_test_config.return_value = mock_config

                initializer = Neo4jInitializer(config_path=f.name, test_mode=True)

                assert initializer.config == mock_config

    @patch("storage.neo4j_client.GraphDatabase")
    @patch("storage.neo4j_client.get_secure_connection_config")
    def test_connect_success_without_ssl(self, mock_get_config, mock_graph_db):
        """Test successful connection without SSL."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_record = Mock()

        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = mock_record

        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.connect(username="neo4j", password="password")

            assert result is True
            assert initializer.driver == mock_driver
            mock_graph_db.driver.assert_called_with(
                "bolt://localhost:7687", auth=("neo4j", "password")
            )
            mock_echo.assert_called_with("✓ Connected to Neo4j at bolt://localhost:7687")

    @patch("storage.neo4j_client.GraphDatabase")
    @patch("storage.neo4j_client.get_secure_connection_config")
    def test_connect_success_with_ssl(self, mock_get_config, mock_graph_db):
        """Test successful connection with SSL."""
        mock_get_config.return_value = {"host": "secure-host", "port": 7687, "ssl": True}

        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()

        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = Mock()

        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo"):
            result = initializer.connect(username="neo4j", password="password")

            assert result is True
            mock_graph_db.driver.assert_called_with(
                "bolt+s://secure-host:7687", auth=("neo4j", "password")
            )

    @patch("storage.neo4j_client.GraphDatabase")
    @patch("storage.neo4j_client.get_secure_connection_config")
    def test_connect_no_password_prompt(self, mock_get_config, mock_graph_db):
        """Test connection without password (should prompt)."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()

        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.return_value = mock_result
        mock_result.single.return_value = Mock()

        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo"):
            with patch("getpass.getpass", return_value="prompted_password"):
                result = initializer.connect(username="neo4j")

                assert result is True
                mock_graph_db.driver.assert_called_with(
                    "bolt://localhost:7687", auth=("neo4j", "prompted_password")
                )

    @patch("storage.neo4j_client.GraphDatabase")
    @patch("storage.neo4j_client.get_secure_connection_config")
    def test_connect_service_unavailable(self, mock_get_config, mock_graph_db):
        """Test connection failure when service is unavailable."""
        mock_get_config.return_value = {"host": "unavailable-host", "port": 7687, "ssl": False}

        mock_graph_db.driver.side_effect = ServiceUnavailable("Service unavailable")

        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.connect(username="neo4j", password="password")

            assert result is False
            assert initializer.driver is None
            mock_echo.assert_any_call(
                "✗ Neo4j is not available at bolt://unavailable-host:7687", err=True
            )

    @patch("storage.neo4j_client.GraphDatabase")
    @patch("storage.neo4j_client.get_secure_connection_config")
    def test_connect_auth_error(self, mock_get_config, mock_graph_db):
        """Test connection failure with authentication error."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        mock_graph_db.driver.side_effect = AuthError("Authentication failed")

        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.connect(username="neo4j", password="wrong_password")

            assert result is False
            mock_echo.assert_any_call("✗ Authentication failed", err=True)

    @patch("storage.neo4j_client.GraphDatabase")
    @patch("storage.neo4j_client.get_secure_connection_config")
    @patch("storage.neo4j_client.sanitize_error_message")
    def test_connect_generic_error(self, mock_sanitize, mock_get_config, mock_graph_db):
        """Test connection failure with generic error."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        mock_graph_db.driver.side_effect = Exception("Generic connection error")
        mock_sanitize.return_value = "Sanitized error message"

        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.connect(username="neo4j", password="password")

            assert result is False
            mock_sanitize.assert_called_with("Generic connection error", ["password", "neo4j"])
            mock_echo.assert_any_call("✗ Failed to connect: Sanitized error message", err=True)

    def test_create_constraints_not_connected(self):
        """Test creating constraints when not connected."""
        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.create_constraints()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_create_constraints_success(self):
        """Test successful constraint creation."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session

        with patch("click.echo") as mock_echo:
            result = initializer.create_constraints()

            assert result is True
            # Should create constraints for all defined node types
            assert mock_session.run.call_count >= 8  # At least 8 constraints
            mock_echo.assert_any_call("  Created constraint: Document.id")

    def test_create_constraints_error(self):
        """Test constraint creation with database error."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Constraint creation failed")

        with patch("click.echo") as mock_echo:
            result = initializer.create_constraints()

            assert result is False
            mock_echo.assert_any_call(
                "✗ Failed to create constraints: Constraint creation failed", err=True
            )

    def test_create_indexes_not_connected(self):
        """Test creating indexes when not connected."""
        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.create_indexes()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_create_indexes_success(self):
        """Test successful index creation."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session

        with patch("click.echo") as mock_echo:
            result = initializer.create_indexes()

            assert result is True
            # Should create multiple indexes
            assert mock_session.run.call_count >= 7
            mock_echo.assert_any_call("  Created index: idx_Document_document_type_created_date")

    def test_create_indexes_with_existing_index_warning(self):
        """Test index creation with existing index warning."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = [
            None,  # First index succeeds
            Exception("already exists"),  # Second index already exists
            None,  # Third index succeeds
        ]

        with patch("click.echo"):
            result = initializer.create_indexes()

            assert result is True
            # Should not show warning for "already exists" error

    def test_create_indexes_error(self):
        """Test index creation with database error."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Index creation failed")

        with patch("click.echo") as mock_echo:
            result = initializer.create_indexes()

            assert result is False
            mock_echo.assert_any_call("✗ Failed to create indexes: Index creation failed", err=True)

    def test_setup_graph_schema_not_connected(self):
        """Test setting up graph schema when not connected."""
        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.setup_graph_schema()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_setup_graph_schema_success(self):
        """Test successful graph schema setup."""
        config = {
            "neo4j": {"database": "test_db"},
            "system": {"schema_version": "2.0.0", "created_date": "2025-01-01"},
        }
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session

        with patch("click.echo") as mock_echo:
            result = initializer.setup_graph_schema()

            assert result is True
            # Should execute multiple queries for system, agents, and document types
            assert mock_session.run.call_count >= 10
            mock_echo.assert_called_with("✓ Graph schema initialized")

    def test_setup_graph_schema_error(self):
        """Test graph schema setup with database error."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Schema setup failed")

        with patch("click.echo") as mock_echo:
            result = initializer.setup_graph_schema()

            assert result is False
            mock_echo.assert_any_call("✗ Failed to setup schema: Schema setup failed", err=True)

    def test_verify_setup_not_connected(self):
        """Test verifying setup when not connected."""
        initializer = Neo4jInitializer(test_mode=True)

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is False
            mock_echo.assert_called_with("✗ Not connected to Neo4j", err=True)

    def test_verify_setup_with_apoc(self):
        """Test verify setup with APOC procedures available."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Mock APOC query results
        mock_labels_result = Mock()
        mock_labels_result.__iter__ = Mock(
            return_value=iter([{"label": "Document", "count": 5}, {"label": "Agent", "count": 4}])
        )

        mock_rels_result = Mock()
        mock_rels_result.__iter__ = Mock(
            return_value=iter(
                [{"type": "HAS_AGENT", "count": 4}, {"type": "HAS_DOCUMENT_TYPE", "count": 3}]
            )
        )

        mock_session.run.side_effect = [mock_labels_result, mock_rels_result]

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("\nNode counts by label:")
            mock_echo.assert_any_call("\nRelationship counts by type:")

    def test_verify_setup_without_apoc(self):
        """Test verify setup when APOC is not available."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session

        # First query (APOC) fails, fallback queries succeed
        mock_node_record = Mock()
        mock_node_record.__getitem__ = Mock(return_value=10)
        mock_node_result = Mock()
        mock_node_result.single.return_value = mock_node_record

        mock_rel_record = Mock()
        mock_rel_record.__getitem__ = Mock(return_value=7)
        mock_rel_result = Mock()
        mock_rel_result.single.return_value = mock_rel_record

        mock_session.run.side_effect = [
            Exception("APOC not available"),  # First query fails
            mock_node_result,  # Fallback node count
            mock_rel_result,  # Fallback relationship count
        ]

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is True
            mock_echo.assert_any_call("\nTotal nodes: 10")
            mock_echo.assert_any_call("Total relationships: 7")

    def test_verify_setup_complete_failure(self):
        """Test verify setup when all queries fail."""
        config = {"neo4j": {"database": "test_db"}}
        initializer = Neo4jInitializer(config=config, test_mode=True)

        mock_driver = Mock()
        mock_session = Mock()
        initializer.driver = mock_driver

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Database error")

        with patch("click.echo") as mock_echo:
            result = initializer.verify_setup()

            assert result is False
            mock_echo.assert_any_call("✗ Failed to verify setup: Database error", err=True)

    def test_close_connection(self):
        """Test closing connection."""
        initializer = Neo4jInitializer(test_mode=True)
        mock_driver = Mock()
        initializer.driver = mock_driver

        initializer.close()

        mock_driver.close.assert_called_once()

    def test_close_no_connection(self):
        """Test closing when no connection exists."""
        initializer = Neo4jInitializer(test_mode=True)

        # Should not raise exception
        initializer.close()
        assert initializer.driver is None


class TestNeo4jCliCommands:
    """Test CLI command functionality."""

    @patch("storage.neo4j_client.Neo4jInitializer")
    def test_main_success_all_steps(self, mock_initializer_class):
        """Test successful execution of all setup steps."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_constraints.return_value = True
        mock_initializer.create_indexes.return_value = True
        mock_initializer.setup_graph_schema.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.driver = Mock()
        mock_initializer.database = "context_graph"

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.neo4j_client import main

        runner = CliRunner()
        result = runner.invoke(main, ["--username", "neo4j", "--password", "password"])

        assert result.exit_code == 0
        mock_initializer.connect.assert_called_once_with(username="neo4j", password="password")
        mock_initializer.create_constraints.assert_called_once()
        mock_initializer.create_indexes.assert_called_once()
        mock_initializer.setup_graph_schema.assert_called_once()
        mock_initializer.verify_setup.assert_called_once()
        mock_initializer.close.assert_called_once()

    @patch("storage.neo4j_client.Neo4jInitializer")
    def test_main_connection_failure(self, mock_initializer_class):
        """Test main command with connection failure."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = False

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.neo4j_client import main

        runner = CliRunner()
        result = runner.invoke(main, ["--username", "neo4j", "--password", "password"])

        assert result.exit_code == 1
        mock_initializer.connect.assert_called_once()
        mock_initializer.create_constraints.assert_not_called()

    @patch("storage.neo4j_client.Neo4jInitializer")
    def test_main_with_skip_flags(self, mock_initializer_class):
        """Test main command with skip flags."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.driver = Mock()

        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.neo4j_client import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--username",
                "neo4j",
                "--password",
                "password",
                "--skip-constraints",
                "--skip-indexes",
                "--skip-schema",
            ],
        )

        assert result.exit_code == 0
        mock_initializer.connect.assert_called_once()
        mock_initializer.create_constraints.assert_not_called()
        mock_initializer.create_indexes.assert_not_called()
        mock_initializer.setup_graph_schema.assert_not_called()
        mock_initializer.verify_setup.assert_called_once()

    @patch("storage.neo4j_client.Neo4jInitializer")
    def test_main_enterprise_database_creation(self, mock_initializer_class):
        """Test database creation for Enterprise Edition."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_constraints.return_value = True
        mock_initializer.create_indexes.return_value = True
        mock_initializer.setup_graph_schema.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.database = "custom_db"

        mock_driver = Mock()
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = None  # Database doesn't exist

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = [mock_result, None]  # SHOW DATABASES, CREATE DATABASE

        mock_initializer.driver = mock_driver
        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.neo4j_client import main

        runner = CliRunner()
        with patch("click.echo") as mock_echo:
            result = runner.invoke(main, ["--username", "neo4j", "--password", "password"])

            assert result.exit_code == 0
            mock_echo.assert_any_call("Creating database 'custom_db'...")

    @patch("storage.neo4j_client.Neo4jInitializer")
    def test_main_community_edition_fallback(self, mock_initializer_class):
        """Test fallback to default database for Community Edition."""
        mock_initializer = Mock()
        mock_initializer.connect.return_value = True
        mock_initializer.create_constraints.return_value = True
        mock_initializer.create_indexes.return_value = True
        mock_initializer.setup_graph_schema.return_value = True
        mock_initializer.verify_setup.return_value = True
        mock_initializer.database = "custom_db"

        mock_driver = Mock()
        mock_session = Mock()

        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("UnsupportedAdministrationCommand")

        mock_initializer.driver = mock_driver
        mock_initializer_class.return_value = mock_initializer

        from click.testing import CliRunner

        from src.storage.neo4j_client import main

        runner = CliRunner()
        with patch("click.echo") as mock_echo:
            result = runner.invoke(main, ["--username", "neo4j", "--password", "password"])

            assert result.exit_code == 0
            assert mock_initializer.database == "neo4j"  # Should fallback to default
            mock_echo.assert_any_call(
                "Note: Using Neo4j Community Edition - using default database"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
