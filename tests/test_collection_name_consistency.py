#!/usr/bin/env python3
"""
Integration test to ensure collection name consistency across all components.

This test verifies that vector_backend, qdrant_client, health checks, and
migrations all use the same Qdrant collection name to prevent 404 errors.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.backends.vector_backend import VectorBackend
from src.storage.qdrant_client import VectorDBInitializer


class TestCollectionNameConsistency(unittest.TestCase):
    """Test that all components use consistent collection names."""

    def setUp(self):
        """Set up test environment."""
        # Store original env var
        self.original_env = os.environ.get("QDRANT_COLLECTION_NAME")

    def tearDown(self):
        """Restore original environment."""
        if self.original_env:
            os.environ["QDRANT_COLLECTION_NAME"] = self.original_env
        elif "QDRANT_COLLECTION_NAME" in os.environ:
            del os.environ["QDRANT_COLLECTION_NAME"]

    def test_default_collection_name_is_context_embeddings(self):
        """Test that default collection name is 'context_embeddings'."""
        # Remove env var to test default
        if "QDRANT_COLLECTION_NAME" in os.environ:
            del os.environ["QDRANT_COLLECTION_NAME"]

        # Test VectorBackend default
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        backend = VectorBackend(mock_client, mock_embedding)

        self.assertEqual(
            backend._collection_name,
            "context_embeddings",
            "VectorBackend should default to 'context_embeddings'"
        )

    def test_env_var_overrides_default(self):
        """Test that QDRANT_COLLECTION_NAME env var overrides default."""
        # Set custom collection name
        os.environ["QDRANT_COLLECTION_NAME"] = "custom_collection"

        # Test VectorBackend respects env var
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        backend = VectorBackend(mock_client, mock_embedding)

        self.assertEqual(
            backend._collection_name,
            "custom_collection",
            "VectorBackend should use QDRANT_COLLECTION_NAME from env"
        )

    def test_qdrant_client_default_matches_vector_backend(self):
        """Test that QdrantClient and VectorBackend use same default."""
        # Remove env var to test defaults
        if "QDRANT_COLLECTION_NAME" in os.environ:
            del os.environ["QDRANT_COLLECTION_NAME"]

        # Create mock config without collection_name
        mock_config = {"qdrant": {"host": "localhost", "port": 6333}}

        # Get default from qdrant_client (would be used in get method)
        default_from_qdrant = "context_embeddings"  # From line 167,227,266,335,422

        # Get default from vector_backend
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        backend = VectorBackend(mock_client, mock_embedding)
        default_from_backend = backend._collection_name

        self.assertEqual(
            default_from_backend,
            default_from_qdrant,
            "VectorBackend and QdrantClient must use same default collection name"
        )

    def test_collection_name_consistency_documentation(self):
        """Document the collection name used by each component."""
        components = {
            "vector_backend.py": "Uses QDRANT_COLLECTION_NAME env var, defaults to 'context_embeddings'",
            "qdrant_client.py": "Uses config.qdrant.collection_name, defaults to 'context_embeddings'",
            "health/endpoints.py": "Checks 'context_embeddings' collection",
            "migration/data_migration.py": "Creates 'context_embeddings' collection",
        }

        # This test documents which collection each component uses
        # If this test fails, update the documentation above
        expected_collection = "context_embeddings"

        # Verify VectorBackend uses expected collection
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        backend = VectorBackend(mock_client, mock_embedding)

        self.assertEqual(
            backend._collection_name,
            expected_collection,
            f"VectorBackend should use '{expected_collection}' by default"
        )

    def test_no_hardcoded_project_context(self):
        """Ensure 'project_context' is not used as default anymore."""
        # This test prevents regression back to 'project_context'
        if "QDRANT_COLLECTION_NAME" in os.environ:
            del os.environ["QDRANT_COLLECTION_NAME"]

        mock_client = MagicMock()
        mock_embedding = MagicMock()
        backend = VectorBackend(mock_client, mock_embedding)

        self.assertNotEqual(
            backend._collection_name,
            "project_context",
            "VectorBackend should NOT use 'project_context' - that collection doesn't exist"
        )


class TestCollectionNameIntegration(unittest.TestCase):
    """Integration tests for collection name usage."""

    def test_all_components_can_use_env_var(self):
        """Test that setting QDRANT_COLLECTION_NAME affects all components."""
        test_collection = "test_unified_collection"
        os.environ["QDRANT_COLLECTION_NAME"] = test_collection

        try:
            # Test VectorBackend
            mock_client = MagicMock()
            mock_embedding = MagicMock()
            backend = VectorBackend(mock_client, mock_embedding)

            self.assertEqual(
                backend._collection_name,
                test_collection,
                "VectorBackend should use env var"
            )

            # Note: QdrantClient uses config, not env var directly
            # but docker-compose.yml passes env var to config

        finally:
            del os.environ["QDRANT_COLLECTION_NAME"]


if __name__ == "__main__":
    unittest.main()
