#!/usr/bin/env python3
"""
Tests for vector storage verification implementation.

Covers verification failures, network errors, corrupted storage scenarios,
and edge cases to ensure the write-after-read verification works correctly.
"""

import pytest
from unittest.mock import MagicMock, patch, call
from qdrant_client.models import PointStruct
import logging

from src.storage.qdrant_client import VectorDBInitializer


class TestVectorStorageVerification:
    """Test vector storage verification functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection"
            }
        }

    @pytest.fixture
    def vector_initializer(self, mock_config):
        """Create VectorDBInitializer with mock config."""
        return VectorDBInitializer(config=mock_config, test_mode=True)

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        client = MagicMock()
        client.upsert.return_value = MagicMock(operation_id="test-op-123", status="completed")
        return client

    def test_successful_vector_storage_with_verification(self, vector_initializer, mock_qdrant_client):
        """Test successful vector storage with verification."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup successful retrieval for verification
        mock_point = MagicMock()
        mock_point.vector = [0.1, 0.2, 0.3, 0.4]  # 4 dimensions
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        # Test data
        vector_id = "test-vector-123"
        embedding = [0.1, 0.2, 0.3, 0.4]
        metadata = {"type": "test", "content": "test content"}
        
        with patch('logging.getLogger') as mock_logger_factory:
            mock_logger = MagicMock()
            mock_logger_factory.return_value = mock_logger
            
            # Execute storage
            result = vector_initializer.store_vector(vector_id, embedding, metadata)
            
            # Verify result
            assert result == vector_id
            
            # Verify upsert was called correctly
            mock_qdrant_client.upsert.assert_called_once()
            upsert_call = mock_qdrant_client.upsert.call_args
            assert upsert_call[1]["collection_name"] == "test_collection"
            assert upsert_call[1]["wait"] == True
            
            # Verify verification retrieve was called
            mock_qdrant_client.retrieve.assert_called_once_with(
                collection_name="test_collection",
                ids=[vector_id]
            )
            
            # Verify logging
            mock_logger.info.assert_any_call(
                f"📦 Storing vector: ID={vector_id}, embedding_dims=4, metadata_keys=['type', 'content']"
            )
            mock_logger.info.assert_any_call(
                f"✓ Vector storage verified: {vector_id} exists with 4 dimensions"
            )

    def test_verification_failure_vector_not_found(self, vector_initializer, mock_qdrant_client):
        """Test verification failure when vector is not found after upsert."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup: upsert succeeds but retrieve returns empty
        mock_qdrant_client.retrieve.return_value = []
        
        vector_id = "missing-vector-123"
        embedding = [0.1, 0.2, 0.3]
        
        # Should raise RuntimeError due to verification failure
        with pytest.raises(RuntimeError, match="Storage verification failed: Vector missing-vector-123 not found after upsert"):
            vector_initializer.store_vector(vector_id, embedding)
        
        # Verify both upsert and retrieve were called
        mock_qdrant_client.upsert.assert_called_once()
        mock_qdrant_client.retrieve.assert_called_once()

    def test_verification_failure_corrupted_vector(self, vector_initializer, mock_qdrant_client):
        """Test verification failure when retrieved vector is corrupted."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup: retrieve returns point with wrong dimensions
        mock_point = MagicMock()
        mock_point.vector = [0.1, 0.2]  # 2 dimensions instead of 4
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        vector_id = "corrupted-vector-123"
        embedding = [0.1, 0.2, 0.3, 0.4]  # 4 dimensions
        
        # Should raise RuntimeError due to dimension mismatch
        with pytest.raises(RuntimeError, match="Storage verification failed: Vector corrupted-vector-123 corrupted or incomplete"):
            vector_initializer.store_vector(vector_id, embedding)

    def test_verification_failure_no_vector_in_point(self, vector_initializer, mock_qdrant_client):
        """Test verification failure when retrieved point has no vector."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup: retrieve returns point with None vector
        mock_point = MagicMock()
        mock_point.vector = None
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        vector_id = "no-vector-123"
        embedding = [0.1, 0.2, 0.3]
        
        # Should raise RuntimeError due to missing vector
        with pytest.raises(RuntimeError, match="Storage verification failed: Vector no-vector-123 corrupted or incomplete"):
            vector_initializer.store_vector(vector_id, embedding)

    def test_upsert_failure_propagated(self, vector_initializer, mock_qdrant_client):
        """Test that upsert failures are properly propagated."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup: upsert raises exception
        mock_qdrant_client.upsert.side_effect = ConnectionError("Network failure")
        
        vector_id = "network-fail-123"
        embedding = [0.1, 0.2, 0.3]
        
        # Should propagate the ConnectionError as RuntimeError
        with pytest.raises(RuntimeError, match="Qdrant connection error: Network failure"):
            vector_initializer.store_vector(vector_id, embedding)
        
        # Verification should not be called if upsert fails
        mock_qdrant_client.retrieve.assert_not_called()

    def test_verification_retrieve_failure(self, vector_initializer, mock_qdrant_client):
        """Test verification failure when retrieve operation fails."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup: upsert succeeds but retrieve fails
        mock_qdrant_client.retrieve.side_effect = TimeoutError("Retrieve timeout")
        
        vector_id = "retrieve-fail-123"
        embedding = [0.1, 0.2, 0.3]
        
        # Should raise RuntimeError with verification failure message
        with pytest.raises(RuntimeError, match="Storage verification failed for vector retrieve-fail-123"):
            vector_initializer.store_vector(vector_id, embedding)
        
        # Both operations should have been attempted
        mock_qdrant_client.upsert.assert_called_once()
        mock_qdrant_client.retrieve.assert_called_once()

    def test_input_validation_edge_cases(self, vector_initializer, mock_qdrant_client):
        """Test input validation for edge cases."""
        vector_initializer.client = mock_qdrant_client
        
        # Test empty vector_id
        with pytest.raises(ValueError, match="vector_id must be a non-empty string"):
            vector_initializer.store_vector("", [0.1, 0.2])
        
        # Test None vector_id
        with pytest.raises(ValueError, match="vector_id must be a non-empty string"):
            vector_initializer.store_vector(None, [0.1, 0.2])
        
        # Test non-string vector_id
        with pytest.raises(ValueError, match="vector_id must be a non-empty string"):
            vector_initializer.store_vector(123, [0.1, 0.2])
        
        # Test empty embedding
        with pytest.raises(ValueError, match="embedding must be a non-empty list"):
            vector_initializer.store_vector("test-id", [])
        
        # Test None embedding
        with pytest.raises(ValueError, match="embedding must be a non-empty list"):
            vector_initializer.store_vector("test-id", None)
        
        # Test non-list embedding
        with pytest.raises(ValueError, match="embedding must be a non-empty list"):
            vector_initializer.store_vector("test-id", "not-a-list")
        
        # Test embedding with non-numeric values
        with pytest.raises(ValueError, match="embedding must contain only numeric values"):
            vector_initializer.store_vector("test-id", [0.1, "invalid", 0.3])

    def test_no_client_connection(self, vector_initializer):
        """Test behavior when client is not connected."""
        # Don't set client (simulates not connected)
        vector_initializer.client = None
        
        with pytest.raises(RuntimeError, match="Not connected to Qdrant"):
            vector_initializer.store_vector("test-id", [0.1, 0.2])

    def test_large_vector_storage(self, vector_initializer, mock_qdrant_client):
        """Test storage of large vectors."""
        vector_initializer.client = mock_qdrant_client
        
        # Create large vector (1536 dimensions like OpenAI embeddings)
        large_embedding = [0.001 * i for i in range(1536)]
        
        # Setup successful verification
        mock_point = MagicMock()
        mock_point.vector = large_embedding
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        vector_id = "large-vector-123"
        
        with patch('logging.getLogger'):
            result = vector_initializer.store_vector(vector_id, large_embedding)
        
        assert result == vector_id
        mock_qdrant_client.upsert.assert_called_once()
        mock_qdrant_client.retrieve.assert_called_once()

    def test_metadata_handling_edge_cases(self, vector_initializer, mock_qdrant_client):
        """Test metadata handling with various data types."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup successful verification
        mock_point = MagicMock()
        mock_point.vector = [0.1, 0.2]
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        # Test with complex metadata
        complex_metadata = {
            "text": "sample text",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
            "none_value": None
        }
        
        with patch('logging.getLogger'):
            result = vector_initializer.store_vector("complex-meta-123", [0.1, 0.2], complex_metadata)
        
        # Verify the complex metadata was passed to upsert
        upsert_call = mock_qdrant_client.upsert.call_args
        point = upsert_call[1]["points"][0]
        assert point.payload == complex_metadata

    def test_concurrent_storage_simulation(self, vector_initializer, mock_qdrant_client):
        """Test storage behavior under simulated concurrent conditions."""
        vector_initializer.client = mock_qdrant_client
        
        # Simulate race condition where vector exists during upsert but gone during retrieve
        call_count = 0
        def mock_retrieve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return []  # First call: not found (race condition)
            else:
                # Subsequent calls would find it
                mock_point = MagicMock()
                mock_point.vector = [0.1, 0.2]
                return [mock_point]
        
        mock_qdrant_client.retrieve.side_effect = mock_retrieve
        
        # First storage should fail due to race condition
        with pytest.raises(RuntimeError, match="Storage verification failed"):
            vector_initializer.store_vector("race-condition-123", [0.1, 0.2])

    def test_logging_output_format(self, vector_initializer, mock_qdrant_client):
        """Test that logging output follows expected format."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup successful scenario
        mock_point = MagicMock()
        mock_point.vector = [0.1, 0.2, 0.3]
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        vector_id = "log-test-123"
        embedding = [0.1, 0.2, 0.3]
        metadata = {"key1": "value1", "key2": "value2"}
        
        with patch('logging.getLogger') as mock_logger_factory:
            mock_logger = MagicMock()
            mock_logger_factory.return_value = mock_logger
            
            vector_initializer.store_vector(vector_id, embedding, metadata)
            
            # Verify specific log messages
            expected_calls = [
                call(f"📦 Storing vector: ID={vector_id}, embedding_dims=3, metadata_keys=['key1', 'key2']"),
                call("✅ Qdrant upsert response: operation_id=test-op-123, status=completed"),
                call(f"✓ Vector storage verified: {vector_id} exists with 3 dimensions")
            ]
            
            mock_logger.info.assert_has_calls(expected_calls, any_order=False)

    def test_debug_logging_embedding_checksum(self, vector_initializer, mock_qdrant_client):
        """Test debug logging includes embedding checksum."""
        vector_initializer.client = mock_qdrant_client
        
        # Setup successful scenario
        mock_point = MagicMock()
        mock_point.vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        with patch('logging.getLogger') as mock_logger_factory:
            mock_logger = MagicMock()
            mock_logger_factory.return_value = mock_logger
            
            vector_initializer.store_vector("checksum-test", embedding)
            
            # Verify debug message includes first 6 values and last value
            mock_logger.debug.assert_called_with(
                f"📊 Embedding checksum: first_6_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], last_value=0.7"
            )


class TestVectorStoragePerformance:
    """Test performance-related aspects of vector storage."""
    
    @pytest.fixture
    def vector_initializer(self):
        """Create VectorDBInitializer with performance test config."""
        config = {
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "performance_test"
            }
        }
        return VectorDBInitializer(config=config, test_mode=True)

    def test_timeout_handling(self, vector_initializer):
        """Test timeout handling in storage operations."""
        mock_client = MagicMock()
        vector_initializer.client = mock_client
        
        # Simulate timeout on upsert
        mock_client.upsert.side_effect = TimeoutError("Operation timed out")
        
        with pytest.raises(RuntimeError, match="Qdrant timeout error: Operation timed out"):
            vector_initializer.store_vector("timeout-test", [0.1, 0.2])

    def test_batch_storage_verification(self, vector_initializer):
        """Test verification works correctly with batch-like operations."""
        mock_client = MagicMock()
        vector_initializer.client = mock_client
        
        # Simulate storing multiple vectors
        vectors = [
            ("vec1", [0.1, 0.2]),
            ("vec2", [0.3, 0.4]),
            ("vec3", [0.5, 0.6])
        ]
        
        for i, (vector_id, embedding) in enumerate(vectors):
            # Each retrieval returns the correct vector
            mock_point = MagicMock()
            mock_point.vector = embedding
            mock_client.retrieve.return_value = [mock_point]
            
            with patch('logging.getLogger'):
                result = vector_initializer.store_vector(vector_id, embedding)
                assert result == vector_id
        
        # Verify all operations completed
        assert mock_client.upsert.call_count == 3
        assert mock_client.retrieve.call_count == 3


class TestErrorRecoveryScenarios:
    """Test error recovery and edge case scenarios."""
    
    def test_storage_with_special_characters(self):
        """Test storage with special characters in vector_id and metadata."""
        config = {"qdrant": {"collection_name": "test"}}
        initializer = VectorDBInitializer(config=config, test_mode=True)
        
        mock_client = MagicMock()
        initializer.client = mock_client
        
        # Setup successful verification
        mock_point = MagicMock()
        mock_point.vector = [0.1, 0.2]
        mock_client.retrieve.return_value = [mock_point]
        
        # Test with special characters
        special_vector_id = "vec-123_test@domain.com"
        special_metadata = {
            "text": "Special chars: áéíóú, 中文, emoji: 🚀",
            "path": "/path/with spaces/file.txt",
            "url": "https://example.com/path?param=value&other=123"
        }
        
        with patch('logging.getLogger'):
            result = initializer.store_vector(special_vector_id, [0.1, 0.2], special_metadata)
        
        assert result == special_vector_id
        
        # Verify special characters were preserved
        upsert_call = mock_client.upsert.call_args
        point = upsert_call[1]["points"][0]
        assert point.id == special_vector_id
        assert point.payload == special_metadata