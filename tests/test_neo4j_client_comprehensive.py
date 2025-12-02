"""
Comprehensive Neo4j client tests for maximum coverage improvement.

Tests Neo4jInitializer with comprehensive coverage of connection management,
schema setup, constraints, indexes, and error handling.
"""

import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.storage.neo4j_client import Neo4jInitializer


class TestNeo4jInitializer:
    """Comprehensive tests for Neo4jInitializer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_config_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_config_dir, ".ctxrc.yaml")

        # Create test config
        self.test_config = {
            "neo4j": {
                "host": "localhost",
                "port": 7687,
                "database": "test_context_graph",
                "ssl": False,
                "username": "neo4j",
                "password": "test_password",
            }
        }

        with open(self.config_path, "w") as f:
            yaml.dump(self.test_config, f)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.test_config_dir, ignore_errors=True)

    def test_neo4j_initializer_creation(self):
        """Test Neo4jInitializer initialization."""
        initializer = Neo4jInitializer(self.config_path)

        assert initializer.config == self.test_config
        assert initializer.driver is None
        assert initializer.database == "test_context_graph"

    def test_neo4j_initializer_default_database(self):
        """Test Neo4jInitializer with default database name."""
        # Create config without database field
        config_without_db = {"neo4j": {"host": "localhost", "port": 7687}}

        config_path = os.path.join(self.test_config_dir, "no_db.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_without_db, f)

        initializer = Neo4jInitializer(config_path)
        assert initializer.database == "context_graph"  # Default value

    def test_load_config_success(self):
        """Test successful config loading."""
        initializer = Neo4jInitializer(self.config_path)
        assert initializer.config == self.test_config

    def test_load_config_file_not_found(self):
        """Test config loading when file not found."""
        with pytest.raises(SystemExit):
            Neo4jInitializer("nonexistent.yaml")

    def test_load_config_invalid_format(self):
        """Test config loading with invalid YAML format."""
        invalid_config_path = os.path.join(self.test_config_dir, "invalid.yaml")
        with open(invalid_config_path, "w") as f:
            f.write("not a dict\n- invalid\n- yaml")

        with pytest.raises(SystemExit):
            Neo4jInitializer(invalid_config_path)

    def test_load_config_not_dict(self):
        """Test config loading when YAML is not a dictionary."""
        not_dict_config_path = os.path.join(self.test_config_dir, "not_dict.yaml")
        with open(not_dict_config_path, "w") as f:
            yaml.dump(["list", "instead", "of", "dict"], f)

        with pytest.raises(SystemExit):
            Neo4jInitializer(not_dict_config_path)

    @patch("storage.neo4j_client.get_secure_connection_config")
    @patch("neo4j.GraphDatabase.driver")
    def test_connect_success(self, mock_driver, mock_get_config):
        """Test successful Neo4j connection."""
        # Mock the secure connection config
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        # Mock the driver
        mock_driver_instance = AsyncMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None

        initializer = Neo4jInitializer(self.config_path)
        result = initializer.connect("neo4j", "password")

        assert result is True
        assert initializer.driver == mock_driver_instance
        mock_driver_instance.verify_connectivity.assert_called_once_with()

    @patch("storage.neo4j_client.get_secure_connection_config")
    @patch("neo4j.GraphDatabase.driver")
    def test_connect_ssl_enabled(self, mock_driver, mock_get_config):
        """Test Neo4j connection with SSL enabled."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": True}

        mock_driver_instance = AsyncMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None

        initializer = Neo4jInitializer(self.config_path)
        result = initializer.connect("neo4j", "password")

        assert result is True
        # Verify SSL URI was used
        mock_driver.assert_called_once_with()
        args, kwargs = mock_driver.call_args
        assert args[0].startswith("neo4j+s://")

    @patch("storage.neo4j_client.get_secure_connection_config")
    @patch("neo4j.GraphDatabase.driver")
    def test_connect_auth_error(self, mock_driver, mock_get_config):
        """Test Neo4j connection with authentication error."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        mock_driver_instance = AsyncMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.side_effect = Exception(
            "AuthError: Invalid credentials"
        )

        initializer = Neo4jInitializer(self.config_path)
        result = initializer.connect("neo4j", "wrong_password")

        assert result is False
        assert initializer.driver is None

    @patch("storage.neo4j_client.get_secure_connection_config")
    @patch("neo4j.GraphDatabase.driver")
    def test_connect_service_unavailable(self, mock_driver, mock_get_config):
        """Test Neo4j connection when service is unavailable."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        mock_driver.side_effect = Exception("ServiceUnavailable: Cannot connect")

        initializer = Neo4jInitializer(self.config_path)
        result = initializer.connect("neo4j", "password")

        assert result is False
        assert initializer.driver is None

    @patch("storage.neo4j_client.get_secure_connection_config")
    @patch("neo4j.GraphDatabase.driver")
    def test_connect_no_password(self, mock_driver, mock_get_config):
        """Test Neo4j connection without password."""
        mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

        mock_driver_instance = AsyncMock()
        mock_driver.return_value = mock_driver_instance
        mock_driver_instance.verify_connectivity.return_value = None

        initializer = Neo4jInitializer(self.config_path)
        result = initializer.connect("neo4j", None)

        assert result is True

    def test_create_constraints_success(self):
        """Test successful constraint creation."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.create_constraints()

        assert result is True
        # Verify session was used and run was called multiple times
        assert mock_session.run.call_count >= 3  # Multiple constraints

    def test_create_constraints_no_driver(self):
        """Test constraint creation without driver."""
        initializer = Neo4jInitializer(self.config_path)
        # Don't set driver

        result = initializer.create_constraints()

        assert result is False

    def test_create_constraints_exception(self):
        """Test constraint creation with exception."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Constraint creation failed")

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.create_constraints()

        assert result is False

    def test_create_indexes_success(self):
        """Test successful index creation."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.create_indexes()

        assert result is True
        # Verify session was used and run was called multiple times
        assert mock_session.run.call_count >= 3  # Multiple indexes

    def test_create_indexes_no_driver(self):
        """Test index creation without driver."""
        initializer = Neo4jInitializer(self.config_path)
        # Don't set driver

        result = initializer.create_indexes()

        assert result is False

    def test_create_indexes_exception(self):
        """Test index creation with exception."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Index creation failed")

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.create_indexes()

        assert result is False

    def test_setup_graph_schema_success(self):
        """Test successful graph schema setup."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.setup_graph_schema()

        assert result is True
        # Should have called run multiple times for schema setup
        assert mock_session.run.call_count >= 5

    def test_setup_graph_schema_no_driver(self):
        """Test graph schema setup without driver."""
        initializer = Neo4jInitializer(self.config_path)
        # Don't set driver

        result = initializer.setup_graph_schema()

        assert result is False

    def test_setup_graph_schema_exception(self):
        """Test graph schema setup with exception."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Schema setup failed")

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.setup_graph_schema()

        assert result is False

    def test_verify_setup_success(self):
        """Test successful setup verification."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Mock successful verification queries
        mock_result = AsyncMock()
        mock_result.single.return_value = {"count": 5}  # Some constraints/indexes exist
        mock_session.run.return_value = mock_result

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.verify_setup()

        assert result is True

    def test_verify_setup_no_driver(self):
        """Test setup verification without driver."""
        initializer = Neo4jInitializer(self.config_path)
        # Don't set driver

        result = initializer.verify_setup()

        assert result is False

    def test_verify_setup_no_constraints(self):
        """Test setup verification when no constraints exist."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session

        # Mock verification queries returning zero counts
        mock_result = AsyncMock()
        mock_result.single.return_value = {"count": 0}
        mock_session.run.return_value = mock_result

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.verify_setup()

        assert result is False

    def test_verify_setup_exception(self):
        """Test setup verification with exception."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_session.run.side_effect = Exception("Verification failed")

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        result = initializer.verify_setup()

        assert result is False

    def test_close_with_driver(self):
        """Test closing connection when driver exists."""
        mock_driver = AsyncMock()

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        initializer.close()

        mock_driver.close.assert_called_once_with()
        assert initializer.driver is None

    def test_close_without_driver(self):
        """Test closing connection when no driver exists."""
        initializer = Neo4jInitializer(self.config_path)
        # Don't set driver

        # Should not raise exception
        initializer.close()

        assert initializer.driver is None

    def test_close_with_exception(self):
        """Test closing connection with exception."""
        mock_driver = AsyncMock()
        mock_driver.close.side_effect = Exception("Close failed")

        initializer = Neo4jInitializer(self.config_path)
        initializer.driver = mock_driver

        # Should handle exception gracefully
        initializer.close()

        assert initializer.driver is None

    def test_full_workflow_success(self):
        """Test complete successful workflow."""
        mock_driver = AsyncMock()
        mock_session = AsyncMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.verify_connectivity.return_value = None

        # Mock verification returning successful results
        mock_result = AsyncMock()
        mock_result.single.return_value = {"count": 5}
        mock_session.run.return_value = mock_result

        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            with patch("neo4j.GraphDatabase.driver") as mock_driver_class:
                mock_driver_class.return_value = mock_driver

                initializer = Neo4jInitializer(self.config_path)

                # Test complete workflow
                assert initializer.connect("neo4j", "password") is True
                assert initializer.create_constraints() is True
                assert initializer.create_indexes() is True
                assert initializer.setup_graph_schema() is True
                assert initializer.verify_setup() is True

                initializer.close()

    def test_workflow_with_failures(self):
        """Test workflow with various failure points."""
        # Test connection failure
        with patch("storage.neo4j_client.get_secure_connection_config") as mock_get_config:
            mock_get_config.return_value = {"host": "localhost", "port": 7687, "ssl": False}

            with patch("neo4j.GraphDatabase.driver") as mock_driver_class:
                mock_driver_class.side_effect = Exception("Connection failed")

                initializer = Neo4jInitializer(self.config_path)
                assert initializer.connect("neo4j", "password") is False

                # All subsequent operations should fail gracefully
                assert initializer.create_constraints() is False
                assert initializer.create_indexes() is False
                assert initializer.setup_graph_schema() is False
                assert initializer.verify_setup() is False

    @patch("storage.neo4j_client.get_secure_connection_config")
    def test_config_variations(self, mock_get_config):
        """Test different configuration variations."""
        # Test with different port
        mock_get_config.return_value = {"host": "neo4j.example.com", "port": 7688, "ssl": True}

        with patch("neo4j.GraphDatabase.driver") as mock_driver_class:
            mock_driver_instance = AsyncMock()
            mock_driver_class.return_value = mock_driver_instance
            mock_driver_instance.verify_connectivity.return_value = None

            initializer = Neo4jInitializer(self.config_path)
            result = initializer.connect("admin", "admin_pass")

            assert result is True
            # Verify correct URI was constructed
            mock_driver_class.assert_called_once_with()
            args, kwargs = mock_driver_class.call_args
            assert "neo4j+s://neo4j.example.com:7688" in args[0]

    def test_error_handling_robustness(self):
        """Test error handling in various scenarios."""
        initializer = Neo4jInitializer(self.config_path)

        # Test operations without connection
        assert initializer.create_constraints() is False
        assert initializer.create_indexes() is False
        assert initializer.setup_graph_schema() is False
        assert initializer.verify_setup() is False

        # Close should not fail
        initializer.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
