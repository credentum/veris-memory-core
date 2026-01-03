"""
Tests for OpenRouter Embedding Provider with Matryoshka truncation.

Tests the OpenRouterEmbeddingProvider class which uses OpenRouter API
to generate embeddings from text-embedding-3-large and truncates them
to 384 dimensions using Matryoshka Representation Learning (MRL).
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List

# Import the provider class
from src.embedding.service import (
    OpenRouterEmbeddingProvider,
    get_openrouter_provider,
    generate_embedding,
    EMBEDDING_PROVIDER,
)


class TestMatryoshkaTruncation:
    """Test Matryoshka truncation functionality."""

    def test_truncation_from_3072_to_384(self):
        """Test truncating 3072D embedding to 384D."""
        provider = OpenRouterEmbeddingProvider(target_dimensions=384)

        # Full 3072D embedding (0.0, 0.001, 0.002, ..., 3.071)
        full_embedding = [i / 1000.0 for i in range(3072)]
        truncated = provider._truncate_embedding(full_embedding)

        assert len(truncated) == 384
        # Should be first 384 values
        assert truncated == [i / 1000.0 for i in range(384)]

    def test_no_truncation_needed(self):
        """Test when embedding is already at target dimensions."""
        provider = OpenRouterEmbeddingProvider(target_dimensions=384)

        embedding = [0.1] * 384
        result = provider._truncate_embedding(embedding)

        assert len(result) == 384
        assert result == embedding

    def test_padding_when_smaller(self):
        """Test padding when embedding is smaller than target."""
        provider = OpenRouterEmbeddingProvider(target_dimensions=384)

        # Smaller embedding
        embedding = [0.5] * 256
        result = provider._truncate_embedding(embedding)

        assert len(result) == 384
        # First 256 should be original values
        assert result[:256] == [0.5] * 256
        # Rest should be zeros (padding)
        assert result[256:] == [0.0] * 128

    def test_custom_target_dimensions(self):
        """Test with custom target dimensions."""
        provider = OpenRouterEmbeddingProvider(target_dimensions=768)

        full_embedding = [0.1] * 3072
        result = provider._truncate_embedding(full_embedding)

        assert len(result) == 768


class TestOpenRouterClient:
    """Test OpenRouter API client initialization."""

    @pytest.mark.asyncio
    async def test_client_initialization_with_api_key(self):
        """Test client initialization when API key is set."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-123"}):
            # Patch at the openai module level since it's imported dynamically
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                provider = OpenRouterEmbeddingProvider()
                client = await provider._get_client()

                # Verify AsyncOpenAI was called with correct params
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args[1]
                assert call_kwargs["api_key"] == "test-key-123"
                assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"

    @pytest.mark.asyncio
    async def test_client_initialization_without_api_key(self):
        """Test client raises error when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            # Also clear the specific key
            import os

            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]

            provider = OpenRouterEmbeddingProvider()

            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                await provider._get_client()

    @pytest.mark.asyncio
    async def test_client_caching(self):
        """Test that client is cached after first initialization."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            # Patch at the openai module level since it's imported dynamically
            with patch("openai.AsyncOpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_openai.return_value = mock_client

                provider = OpenRouterEmbeddingProvider()

                # Get client twice
                client1 = await provider._get_client()
                client2 = await provider._get_client()

                # Should be same instance, OpenAI called only once
                assert client1 is client2
                assert mock_openai.call_count == 1


class TestEmbeddingGeneration:
    """Test embedding generation via OpenRouter."""

    @pytest.mark.asyncio
    async def test_generate_single_embedding(self):
        """Test generating a single embedding."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterEmbeddingProvider(target_dimensions=384)

            # Mock the client and response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 3072)]

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)

            provider._client = mock_client
            provider._initialized = True

            result = await provider.generate_embedding("Test text")

            # Should be truncated to 384D
            assert len(result) == 384
            assert all(v == 0.1 for v in result)

            # Verify API call
            mock_client.embeddings.create.assert_called_once_with(
                model="openai/text-embedding-3-large", input="Test text"
            )

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self):
        """Test batch embedding generation."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterEmbeddingProvider(target_dimensions=384)

            # Mock response with 3 embeddings
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1] * 3072),
                MagicMock(embedding=[0.2] * 3072),
                MagicMock(embedding=[0.3] * 3072),
            ]

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)

            provider._client = mock_client
            provider._initialized = True

            texts = ["Text 1", "Text 2", "Text 3"]
            results = await provider.generate_batch_embeddings(texts)

            assert len(results) == 3
            for result in results:
                assert len(result) == 384

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test batch with empty list."""
        provider = OpenRouterEmbeddingProvider()
        results = await provider.generate_batch_embeddings([])
        assert results == []


class TestMetrics:
    """Test metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_on_success(self):
        """Test metrics are updated on successful generation."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterEmbeddingProvider()

            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 3072)]

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)

            provider._client = mock_client
            provider._initialized = True

            await provider.generate_embedding("Test")

            metrics = provider.get_metrics()
            assert metrics["total_requests"] == 1
            assert metrics["successful_requests"] == 1
            assert metrics["failed_requests"] == 0
            assert metrics["provider"] == "openrouter"

    @pytest.mark.asyncio
    async def test_metrics_on_failure(self):
        """Test metrics are updated on failed generation."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            provider = OpenRouterEmbeddingProvider()

            mock_client = AsyncMock()
            mock_client.embeddings.create = AsyncMock(
                side_effect=Exception("API Error")
            )

            provider._client = mock_client
            provider._initialized = True

            with pytest.raises(Exception):
                await provider.generate_embedding("Test")

            metrics = provider.get_metrics()
            assert metrics["total_requests"] == 1
            assert metrics["successful_requests"] == 0
            assert metrics["failed_requests"] == 1


class TestProviderRouting:
    """Test that generate_embedding routes to correct provider."""

    @pytest.mark.asyncio
    async def test_routes_to_openrouter_when_configured(self):
        """Test that generate_embedding uses OpenRouter when configured."""
        with patch.dict(
            "os.environ",
            {"EMBEDDING_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "test-key"},
        ):
            # Need to reload the module to pick up new env var
            with patch(
                "src.embedding.service.EMBEDDING_PROVIDER", "openrouter"
            ), patch(
                "src.embedding.service.get_openrouter_provider"
            ) as mock_get_provider:
                mock_provider = AsyncMock()
                mock_provider.generate_embedding = AsyncMock(
                    return_value=[0.1] * 384
                )
                mock_get_provider.return_value = mock_provider

                result = await generate_embedding("Test text")

                assert len(result) == 384
                mock_get_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_local_by_default(self):
        """Test that generate_embedding uses local service by default."""
        with patch("src.embedding.service.EMBEDDING_PROVIDER", "local"), patch(
            "src.embedding.service.get_embedding_service"
        ) as mock_get_service:
            mock_service = AsyncMock()
            mock_service.generate_embedding = AsyncMock(return_value=[0.1] * 384)
            mock_get_service.return_value = mock_service

            result = await generate_embedding("Test text")

            assert len(result) == 384
            mock_get_service.assert_called_once()


class TestIntegration:
    """Integration tests (require actual API key - skipped in CI)."""

    @pytest.mark.skip(reason="Requires real API key")
    @pytest.mark.asyncio
    async def test_real_openrouter_api(self):
        """Test with real OpenRouter API (manual testing only)."""
        provider = OpenRouterEmbeddingProvider(target_dimensions=384)

        result = await provider.generate_embedding(
            "This is a test of the OpenRouter embedding API."
        )

        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)
        # Values should be normalized
        assert all(-2.0 <= v <= 2.0 for v in result)
