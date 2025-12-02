#!/usr/bin/env python3
"""
Test to verify the extracted components can be imported
"""
import pytest


def test_storage_imports():
    """Test that all storage components can be imported successfully."""
    # Test storage imports
    from src.storage.hash_diff_embedder import (  # noqa: F401
        DocumentHash,
        EmbeddingTask,
        HashDiffEmbedder,
    )

    from src.storage.neo4j_client import Neo4jInitializer  # noqa: F401
    from src.storage.qdrant_client import VectorDBInitializer  # noqa: F401
    from src.storage.kv_store import ContextKV  # noqa: F401


def test_core_imports():
    """Test that all core components can be imported successfully."""
    from src.core.base_component import DatabaseComponent  # noqa: F401
    from src.core.utils import get_secure_connection_config  # noqa: F401


def test_validator_imports():
    """Test that all validator components can be imported successfully."""
    from src.validators.kv_validators import validate_redis_key  # noqa: F401
