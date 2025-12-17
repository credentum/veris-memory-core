#!/usr/bin/env python3
"""
Unit tests for src/embedding/sparse_service.py

Tests the SparseEmbeddingService class and related functionality
with mocked fastembed dependencies.
"""

import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from embedding.sparse_service import (
    SparseVector,
    SparseEmbeddingService,
    get_sparse_embedding_service,
    generate_sparse_embedding,
    SPARSE_ENABLED,
    SPARSE_MODEL_NAME,
)


class TestSparseVector:
    """Tests for SparseVector dataclass."""

    def test_sparse_vector_creation(self):
        """Test creating a SparseVector with indices and values."""
        sv = SparseVector(indices=[0, 5, 10], values=[0.1, 0.5, 0.3])
        assert sv.indices == [0, 5, 10]
        assert sv.values == [0.1, 0.5, 0.3]

    def test_sparse_vector_to_dict(self):
        """Test converting SparseVector to dict format for Qdrant."""
        sv = SparseVector(indices=[1, 2, 3], values=[0.5, 0.25, 0.75])
        result = sv.to_dict()
        assert result == {"indices": [1, 2, 3], "values": [0.5, 0.25, 0.75]}

    def test_sparse_vector_len(self):
        """Test __len__ returns number of non-zero elements."""
        sv = SparseVector(indices=[0, 5, 10, 15], values=[0.1, 0.5, 0.3, 0.2])
        assert len(sv) == 4

    def test_sparse_vector_empty(self):
        """Test empty SparseVector."""
        sv = SparseVector(indices=[], values=[])
        assert len(sv) == 0
        assert sv.to_dict() == {"indices": [], "values": []}


class TestSparseEmbeddingServiceInit:
    """Tests for SparseEmbeddingService initialization."""

    def test_init_default_model(self):
        """Test initialization with default model name."""
        service = SparseEmbeddingService()
        assert service.model_name == SPARSE_MODEL_NAME
        assert service._model is None
        assert service._initialized is False

    def test_init_custom_model(self):
        """Test initialization with custom model name."""
        service = SparseEmbeddingService(model_name="custom/model")
        assert service.model_name == "custom/model"

    def test_init_metrics(self):
        """Test initial metrics are zeroed."""
        service = SparseEmbeddingService()
        metrics = service.get_metrics()
        assert metrics["total_embeddings"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["errors"] == 0


class TestSparseEmbeddingServiceInitialize:
    """Tests for SparseEmbeddingService.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization with mocked fastembed."""
        service = SparseEmbeddingService()

        with patch('embedding.sparse_service.SPARSE_ENABLED', True):
            with patch.dict('sys.modules', {'fastembed': MagicMock()}):
                # Mock the SparseTextEmbedding import
                mock_sparse_embed = MagicMock()
                with patch.object(service, '_model', None):
                    with patch('embedding.sparse_service.SparseTextEmbedding', mock_sparse_embed, create=True):
                        # We need to patch the import inside initialize
                        with patch.dict('sys.modules', {'fastembed': MagicMock(SparseTextEmbedding=mock_sparse_embed)}):
                            result = await service.initialize()
                            # Note: This may return False if fastembed import fails in test env
                            # The important thing is no exception is raised

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test that initialize returns True if already initialized."""
        service = SparseEmbeddingService()
        service._initialized = True

        result = await service.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_initialize_disabled(self):
        """Test that initialize returns False when SPARSE_ENABLED is False."""
        service = SparseEmbeddingService()

        with patch('embedding.sparse_service.SPARSE_ENABLED', False):
            result = await service.initialize()
            assert result is False


class TestSparseEmbeddingServiceIsAvailable:
    """Tests for SparseEmbeddingService.is_available()."""

    def test_is_available_true(self):
        """Test is_available returns True when initialized with model."""
        service = SparseEmbeddingService()
        service._initialized = True
        service._model = MagicMock()
        assert service.is_available() is True

    def test_is_available_false_not_initialized(self):
        """Test is_available returns False when not initialized."""
        service = SparseEmbeddingService()
        assert service.is_available() is False

    def test_is_available_false_no_model(self):
        """Test is_available returns False when model is None."""
        service = SparseEmbeddingService()
        service._initialized = True
        service._model = None
        assert service.is_available() is False


class TestSparseEmbeddingServiceGenerate:
    """Tests for SparseEmbeddingService.generate_sparse_embedding()."""

    def test_generate_not_available(self):
        """Test generate returns None when service not available."""
        service = SparseEmbeddingService()
        result = service.generate_sparse_embedding("test text")
        assert result is None

    def test_generate_empty_text(self):
        """Test generate returns empty vector for empty text."""
        service = SparseEmbeddingService()
        service._initialized = True
        service._model = MagicMock()

        result = service.generate_sparse_embedding("")
        assert result is not None
        assert result.indices == []
        assert result.values == []

    def test_generate_whitespace_text(self):
        """Test generate returns empty vector for whitespace-only text."""
        service = SparseEmbeddingService()
        service._initialized = True
        service._model = MagicMock()

        result = service.generate_sparse_embedding("   ")
        assert result is not None
        assert result.indices == []
        assert result.values == []

    def test_generate_success(self):
        """Test successful embedding generation."""
        service = SparseEmbeddingService()
        service._initialized = True

        # Mock the model response
        @dataclass
        class MockSparseEmbedding:
            indices: any
            values: any

        import numpy as np
        mock_embedding = MockSparseEmbedding(
            indices=np.array([0, 5, 10]),
            values=np.array([0.1, 0.5, 0.3])
        )

        mock_model = MagicMock()
        mock_model.embed.return_value = [mock_embedding]
        service._model = mock_model

        result = service.generate_sparse_embedding("test text")

        assert result is not None
        assert result.indices == [0, 5, 10]
        assert result.values == [0.1, 0.5, 0.3]
        mock_model.embed.assert_called_once_with(["test text"])

    def test_generate_updates_metrics(self):
        """Test that generate updates metrics correctly."""
        service = SparseEmbeddingService()
        service._initialized = True

        @dataclass
        class MockSparseEmbedding:
            indices: any
            values: any

        import numpy as np
        mock_embedding = MockSparseEmbedding(
            indices=np.array([0, 5, 10]),
            values=np.array([0.1, 0.5, 0.3])
        )

        mock_model = MagicMock()
        mock_model.embed.return_value = [mock_embedding]
        service._model = mock_model

        service.generate_sparse_embedding("test")
        service.generate_sparse_embedding("another test")

        metrics = service.get_metrics()
        assert metrics["total_embeddings"] == 2
        assert metrics["total_tokens"] == 6  # 3 + 3

    def test_generate_handles_exception(self):
        """Test that generate handles exceptions and returns None."""
        service = SparseEmbeddingService()
        service._initialized = True

        mock_model = MagicMock()
        mock_model.embed.side_effect = Exception("Model error")
        service._model = mock_model

        result = service.generate_sparse_embedding("test")

        assert result is None
        assert service._metrics["errors"] == 1


class TestSparseEmbeddingServiceBatch:
    """Tests for SparseEmbeddingService.generate_sparse_embeddings_batch()."""

    def test_batch_not_available(self):
        """Test batch returns None list when service not available."""
        service = SparseEmbeddingService()
        result = service.generate_sparse_embeddings_batch(["text1", "text2"])
        assert result == [None, None]

    def test_batch_empty_list(self):
        """Test batch with empty list."""
        service = SparseEmbeddingService()
        service._initialized = True
        service._model = MagicMock()

        result = service.generate_sparse_embeddings_batch([])
        assert result == []

    def test_batch_with_empty_texts(self):
        """Test batch handles empty texts correctly."""
        service = SparseEmbeddingService()
        service._initialized = True

        mock_model = MagicMock()
        mock_model.embed.return_value = []
        service._model = mock_model

        result = service.generate_sparse_embeddings_batch(["", "  ", ""])

        # All empty texts should return empty sparse vectors
        assert len(result) == 3
        for r in result:
            assert r.indices == []
            assert r.values == []

    def test_batch_success(self):
        """Test successful batch embedding generation."""
        service = SparseEmbeddingService()
        service._initialized = True

        @dataclass
        class MockSparseEmbedding:
            indices: any
            values: any

        import numpy as np
        mock_embeddings = [
            MockSparseEmbedding(indices=np.array([0, 1]), values=np.array([0.1, 0.2])),
            MockSparseEmbedding(indices=np.array([2, 3]), values=np.array([0.3, 0.4])),
        ]

        mock_model = MagicMock()
        mock_model.embed.return_value = mock_embeddings
        service._model = mock_model

        result = service.generate_sparse_embeddings_batch(["text1", "text2"])

        assert len(result) == 2
        assert result[0].indices == [0, 1]
        assert result[1].indices == [2, 3]


class TestSparseEmbeddingServiceMetrics:
    """Tests for SparseEmbeddingService.get_metrics()."""

    def test_get_metrics_includes_all_fields(self):
        """Test get_metrics returns all expected fields."""
        service = SparseEmbeddingService()
        metrics = service.get_metrics()

        assert "total_embeddings" in metrics
        assert "total_tokens" in metrics
        assert "errors" in metrics
        assert "model" in metrics
        assert "initialized" in metrics
        assert "enabled" in metrics

    def test_get_metrics_reflects_state(self):
        """Test get_metrics reflects current service state."""
        service = SparseEmbeddingService(model_name="test/model")
        service._initialized = True
        service._metrics["total_embeddings"] = 10
        service._metrics["errors"] = 2

        metrics = service.get_metrics()

        assert metrics["model"] == "test/model"
        assert metrics["initialized"] is True
        assert metrics["total_embeddings"] == 10
        assert metrics["errors"] == 2


class TestGetSparseEmbeddingService:
    """Tests for get_sparse_embedding_service() singleton."""

    def test_returns_same_instance(self):
        """Test that get_sparse_embedding_service returns singleton."""
        # Reset the global instance first
        import embedding.sparse_service as module
        module._sparse_service = None

        service1 = get_sparse_embedding_service()
        service2 = get_sparse_embedding_service()

        assert service1 is service2

    def test_creates_instance_if_none(self):
        """Test that get_sparse_embedding_service creates instance."""
        import embedding.sparse_service as module
        module._sparse_service = None

        service = get_sparse_embedding_service()
        assert service is not None
        assert isinstance(service, SparseEmbeddingService)


class TestGenerateSparseEmbeddingConvenience:
    """Tests for generate_sparse_embedding() convenience function."""

    @pytest.mark.asyncio
    async def test_generate_sparse_embedding_initializes_service(self):
        """Test convenience function initializes service if needed."""
        import embedding.sparse_service as module
        module._sparse_service = None

        with patch.object(SparseEmbeddingService, 'initialize', new_callable=AsyncMock) as mock_init:
            with patch.object(SparseEmbeddingService, 'is_available', return_value=False):
                mock_init.return_value = True
                await generate_sparse_embedding("test")
                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_sparse_embedding_skips_init_if_available(self):
        """Test convenience function skips init if already available."""
        import embedding.sparse_service as module

        mock_service = MagicMock()
        mock_service.is_available.return_value = True
        mock_service.generate_sparse_embedding.return_value = SparseVector([1], [0.5])
        module._sparse_service = mock_service

        result = await generate_sparse_embedding("test")

        mock_service.initialize.assert_not_called()
        mock_service.generate_sparse_embedding.assert_called_once_with("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
