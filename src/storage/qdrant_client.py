#!/usr/bin/env python3
"""
qdrant_client.py: Qdrant client for context storage system

This module:
1. Manages Qdrant vector database connections
2. Creates and configures collections
3. Handles vector operations
4. Provides search and indexing capabilities
"""

import logging
import sys
import time
from typing import Any, Dict, Optional

import click
import yaml
from qdrant_client import QdrantClient

# Configure module logger
logger = logging.getLogger(__name__)

# Import configuration error handling
try:
    from ..core.config_error import ConfigParseError
    from ..core.test_config import get_test_config
except ImportError:
    from core.config_error import ConfigParseError
    from core.test_config import get_test_config

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

# SPRINT 11: Import HNSW parameter manager
try:
    from .qdrant_index_config import index_config_manager, SearchIntent
except ImportError:
    from qdrant_index_config import index_config_manager, SearchIntent

# Import Config for standardized settings
try:
    from ..core.config import Config
except ImportError:
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.config import Config


class VectorDBInitializer:
    """Initialize and configure Qdrant vector database"""

    def __init__(
        self,
        config_path: str = ".ctxrc.yaml",
        config: Optional[Dict[str, Any]] = None,
        test_mode: bool = False,
    ):
        """Initialize Qdrant client with optional config injection for testing.

        Args:
            config_path: Path to configuration file
            config: Optional configuration dictionary (overrides file loading)
            test_mode: If True, use test defaults when config is missing
        """
        self.test_mode = test_mode

        if config is not None:
            self.config = config
        else:
            self.config = self._load_config(config_path)

        self.client: Optional[QdrantClient] = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from .ctxrc.yaml"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    if self.test_mode:
                        return get_test_config()
                    raise ConfigParseError(config_path, "Configuration must be a dictionary")
                return config
        except FileNotFoundError:
            if self.test_mode:
                # Return test configuration when in test mode
                return get_test_config()
            # In production, still use sys.exit for backward compatibility
            click.echo(f"Error: {config_path} not found", err=True)
            sys.exit(1)
        except yaml.YAMLError as e:
            error_msg = f"Error parsing {config_path}: {e}"
            click.echo(error_msg, err=True)
            raise ConfigParseError(config_path, error_msg)

    def connect(self) -> bool:
        """Connect to Qdrant instance"""
        # Import locally
        from ..core.utils import get_secure_connection_config

        qdrant_config = get_secure_connection_config(self.config, "qdrant")
        host = qdrant_config["host"]
        port = qdrant_config.get("port", 6333)
        use_ssl = qdrant_config.get("ssl", False)
        timeout = qdrant_config.get("timeout", 5)

        # PHASE 0: Check Qdrant client and server compatibility
        try:
            import qdrant_client
            client_version = getattr(qdrant_client, '__version__', 'unknown')
            logger.info(f"🔗 Qdrant client version: {client_version}")
        except Exception as e:
            logger.warning(f"⚠️  Could not determine Qdrant client version: {e}")

        try:
            # Use appropriate protocol based on SSL setting
            if use_ssl:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    https=True,
                    verify=qdrant_config.get("verify_ssl", True),
                    timeout=timeout,
                )
            else:
                self.client = QdrantClient(host=host, port=port, timeout=timeout)
            # Test connection
            if self.client:
                collections = self.client.get_collections()
                
                # PHASE 0: Try to get server version for compatibility check
                try:
                    # Attempt to get server info/version if available
                    server_info = getattr(self.client, '_client', None)
                    if hasattr(server_info, 'get') and callable(getattr(server_info, 'get', None)):
                        try:
                            # This is a best-effort attempt - different Qdrant versions expose this differently
                            version_response = server_info.get("/")
                            if hasattr(version_response, 'json'):
                                server_info_data = version_response.json()
                                server_version = server_info_data.get('version', 'unknown')
                                logger.info(f"🖥️  Qdrant server version: {server_version}")
                        except:
                            logger.info("🖥️  Qdrant server version: unable to determine (connection works)")
                    else:
                        logger.info("🖥️  Qdrant server version: unable to determine (connection works)")
                except Exception as version_error:
                    logger.debug(f"🖥️  Qdrant server version check failed: {version_error}")
                
                logger.info(f"✓ Connected to Qdrant at {host}:{port}")
                return True
            return False
        except Exception as e:
            logger.error(f"✗ Failed to connect to Qdrant at {host}:{port}: {e}")
            return False

    def create_collection(self, force: bool = False) -> bool:
        """Create the Qdrant collection for vector storage (configured via collection_name)"""
        collection_name = self.config.get("qdrant", {}).get("collection_name", "context_embeddings")

        if not self.client:
            logger.error("✗ Not connected to Qdrant")
            return False

        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if exists and not force:
                logger.info(
                    f"Collection '{collection_name}' already exists. Use --force to recreate."
                )
                return True

            if exists and force:
                logger.info(f"Deleting existing collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
                time.sleep(1)  # Give Qdrant time to process

            # Create collection with optimal settings for embeddings
            logger.info(f"Creating collection '{collection_name}'...")

            # SPRINT 11: Enforce v1.0 dimension requirement (384)
            if Config.EMBEDDING_DIMENSIONS != 384:
                from ..core.error_handler import handle_v1_dimension_mismatch
                error_response = handle_v1_dimension_mismatch(384, Config.EMBEDDING_DIMENSIONS)
                raise ValueError(f"Configuration error: {error_response['message']}")
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # SPRINT 11: Hard-coded to 384 for v1.0 compliance
                    distance=Distance.COSINE,
                ),
                optimizers_config=OptimizersConfigDiff(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=2,
                    flush_interval_sec=5,
                ),
                # SPRINT 11: Use standardized HNSW parameters per v1.0 spec
                hnsw_config=HnswConfigDiff(
                    m=index_config_manager.REQUIRED_M,                    # 32 per Sprint 11
                    ef_construct=index_config_manager.REQUIRED_EF_CONSTRUCT,  # 128 per Sprint 11
                    full_scan_threshold=10000,
                ),
            )

            logger.info(f"✓ Collection '{collection_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to create collection: {e}")
            return False

    def verify_setup(self) -> bool:
        """Verify the Qdrant setup is correct"""
        collection_name = self.config.get("qdrant", {}).get("collection_name", "context_embeddings")

        if not self.client:
            click.echo("✗ Not connected to Qdrant", err=True)
            return False

        try:
            # Get collection info
            info = self.client.get_collection(collection_name)

            click.echo("\nCollection Info:")
            click.echo(f"  Name: {collection_name}")

            # Handle different vector config formats
            if info.config and info.config.params and info.config.params.vectors:
                vectors_config = info.config.params.vectors
                if isinstance(vectors_config, VectorParams):
                    click.echo(f"  Vector size: {vectors_config.size}")
                    click.echo(f"  Distance metric: {vectors_config.distance}")
                elif isinstance(vectors_config, dict):
                    # Handle named vectors
                    for name, params in vectors_config.items():
                        click.echo(f"  Vector '{name}' size: {params.size}")
                        click.echo(f"  Vector '{name}' distance: {params.distance}")

            click.echo(f"  Points count: {info.points_count}")

            # Check Qdrant version
            qdrant_version = self.config.get("qdrant", {}).get("version", "1.14.x")
            click.echo(f"\nExpected Qdrant version: {qdrant_version}")

            return True

        except Exception as e:
            click.echo(f"✗ Failed to verify setup: {e}", err=True)
            return False

    def insert_test_point(self) -> bool:
        """Insert a test point to verify everything works"""
        collection_name = self.config.get("qdrant", {}).get("collection_name", "context_embeddings")

        if not self.client:
            click.echo("✗ Not connected to Qdrant", err=True)
            return False

        try:
            # Create a test embedding (random for now)
            import random

            test_vector = [random.random() for _ in range(Config.EMBEDDING_DIMENSIONS)]

            # Insert test point
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id="test-point-001",
                        vector=test_vector,
                        payload={
                            "document_type": "test",
                            "content": "This is a test point for verification",
                            "created_date": "2025-07-11",
                        },
                    )
                ],
                wait=True,  # Ensure immediate availability for retrieval
            )

            # Search for it (using query_points for qdrant-client v1.7+)
            response = self.client.query_points(
                collection_name=collection_name, query=test_vector, limit=1
            )
            results = response.points

            if results and results[0].id == "test-point-001":
                click.echo("✓ Test point inserted and retrieved successfully")

                # Clean up
                self.client.delete(
                    collection_name=collection_name, points_selector=["test-point-001"]
                )
                return True
            else:
                click.echo("✗ Test point verification failed", err=True)
                return False

        except Exception as e:
            click.echo(f"✗ Failed to test point operations: {e}", err=True)
            return False

    def store_vector(
        self, vector_id: str, embedding: list, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a vector in the Qdrant collection.

        Args:
            vector_id: Unique identifier for the vector
            embedding: The vector embedding
            metadata: Optional metadata to store with the vector

        Returns:
            str: The vector ID that was stored

        Raises:
            RuntimeError: If not connected or storage fails
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant")

        collection_name = self.config.get("qdrant", {}).get("collection_name", "context_embeddings")

        # Validate inputs
        if not vector_id or not isinstance(vector_id, str):
            raise ValueError("vector_id must be a non-empty string")
        if not embedding or not isinstance(embedding, list):
            raise ValueError("embedding must be a non-empty list")
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise ValueError("embedding must contain only numeric values")

        try:
            from qdrant_client.models import PointStruct

            # PHASE 0: Verbose logging for storage pipeline
            logger.info(f"📦 Storing vector: ID={vector_id}, embedding_dims={len(embedding)}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📊 Embedding checksum: first_6_values={embedding[:6]}, last_value={embedding[-1] if embedding else 'None'}")
                logger.debug(f"📋 Metadata keys: {list((metadata or {}).keys())}")

            upsert_result = self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload=metadata or {},
                    )
                ],
                wait=True,  # Ensure immediate availability for retrieval
            )
            
            # PHASE 0: Log upsert response details  
            logger.info(f"✅ Qdrant upsert response: operation_id={getattr(upsert_result, 'operation_id', 'N/A')}, status={getattr(upsert_result, 'status', 'unknown')}")
            
            # PHASE 0 FIX: Write-after-read verification
            # Verify the vector was actually stored by attempting to retrieve it
            try:
                retrieved_points = self.client.retrieve(
                    collection_name=collection_name,
                    ids=[vector_id],
                    with_vectors=True  # Required to get vector data for verification
                )
                if not retrieved_points or len(retrieved_points) == 0:
                    raise RuntimeError(f"Storage verification failed: Vector {vector_id} not found after upsert")
                
                # Additional verification: check if vector exists and has expected properties
                stored_point = retrieved_points[0]
                if not stored_point.vector or len(stored_point.vector) != len(embedding):
                    raise RuntimeError(f"Storage verification failed: Vector {vector_id} corrupted or incomplete")
                    
                logger.info(f"✓ Vector storage verified: {vector_id} exists with {len(stored_point.vector)} dimensions")
                
            except Exception as verification_error:
                # This is a critical failure - the upsert claimed success but verification failed
                raise RuntimeError(f"Storage verification failed for vector {vector_id}: {verification_error}")
            
            return vector_id
        except ConnectionError as e:
            raise RuntimeError(f"Qdrant connection error: {e}")
        except TimeoutError as e:
            raise RuntimeError(f"Qdrant timeout error: {e}")
        except ValueError as e:
            # Re-raise validation errors
            raise
        except ImportError as e:
            raise RuntimeError(f"Missing Qdrant dependencies: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to store vector: {e}")

    def search(
        self, query_vector: list, limit: int = 10, filter_dict: Optional[Dict[str, Any]] = None
    ) -> list:
        """Search for similar vectors in the collection.

        Args:
            query_vector: The query vector to search for
            limit: Maximum number of results to return
            filter_dict: Optional filter conditions

        Returns:
            list: Search results with scores and metadata

        Raises:
            RuntimeError: If not connected or search fails
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant")

        collection_name = self.config.get("qdrant", {}).get("collection_name", "context_embeddings")

        # Validate inputs
        if not query_vector or not isinstance(query_vector, list):
            raise ValueError("query_vector must be a non-empty list")
        if not all(isinstance(x, (int, float)) for x in query_vector):
            raise ValueError("query_vector must contain only numeric values")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if limit > 1000:  # Reasonable upper bound
            raise ValueError("limit cannot exceed 1000")
        if filter_dict is not None and not isinstance(filter_dict, dict):
            raise ValueError("filter_dict must be a dictionary or None")

        try:
            # Use query_points for qdrant-client v1.7+ (search() was deprecated)
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=filter_dict,
            )
            results = response.points

            # Convert results to a more usable format
            search_results = []
            for result in results:
                search_results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload or {},
                    }
                )

            return search_results
        except ConnectionError as e:
            raise RuntimeError(f"Qdrant connection error during search: {e}")
        except TimeoutError as e:
            raise RuntimeError(f"Qdrant timeout error during search: {e}")
        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to search vectors: {e}")

    def get_collections(self):
        """Get information about available collections.

        Returns:
            Qdrant collections object

        Raises:
            RuntimeError: If not connected or operation fails
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant")

        try:
            return self.client.get_collections()
        except Exception as e:
            raise RuntimeError(f"Failed to get collections: {e}")

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector from the Qdrant collection.

        Args:
            vector_id: Unique identifier of the vector to delete

        Returns:
            bool: True if deletion was successful

        Raises:
            RuntimeError: If not connected or deletion fails
        """
        if not self.client:
            raise RuntimeError("Not connected to Qdrant")

        collection_name = self.config.get("qdrant", {}).get("collection_name", "context_embeddings")

        if not vector_id or not isinstance(vector_id, str):
            raise ValueError("vector_id must be a non-empty string")

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=[vector_id]
            )
            logger.info(f"✓ Vector deleted from Qdrant: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            raise RuntimeError(f"Failed to delete vector: {e}")

    def close(self) -> None:
        """Close the client connection (no-op for Qdrant client)."""
        # Qdrant client doesn't require explicit closing
        pass


@click.command()
@click.option("--force", is_flag=True, help="Force recreation of collection if exists")
@click.option("--skip-test", is_flag=True, help="Skip test point insertion")
def main(force: bool, skip_test: bool):
    """Initialize Qdrant vector database for the Agent-First Context System"""
    click.echo("=== Qdrant Vector Database Initialization ===\n")

    initializer = VectorDBInitializer()

    # Connect to Qdrant
    if not initializer.connect():
        click.echo("\nPlease ensure Qdrant is running:")
        click.echo("  docker run -p 6333:6333 qdrant/qdrant:v1.14.0")
        sys.exit(1)

    # Create collection
    if not initializer.create_collection(force=force):
        sys.exit(1)

    # Verify setup
    if not initializer.verify_setup():
        sys.exit(1)

    # Test operations
    if not skip_test:
        if not initializer.insert_test_point():
            sys.exit(1)

    click.echo("\n✓ Qdrant initialization complete!")


if __name__ == "__main__":
    main()
