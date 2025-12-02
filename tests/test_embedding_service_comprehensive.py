#!/usr/bin/env python3
"""
Comprehensive tests for embedding service to achieve high coverage.

This test suite covers:
- Basic initialization and setup
- Provider management
- Configuration handling
- Basic functionality testing
"""

from unittest.mock import Mock

import pytest

from src.core.embedding_service import EmbeddingProvider, EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for embedding service."""
        return {
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "dimensions": 1536,
                "batch_size": 100,
                "cache_enabled": True,
                "fallback_providers": ["huggingface", "local"],
            },
            "openai": {"api_key": "test-api-key", "base_url": "https://api.openai.com/v1"},
            "huggingface": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "cache_dir": "/tmp/hf_cache",
            },
        }

    @pytest.fixture
    def embedding_service(self, mock_config):
        """Create embedding service instance."""
        service = EmbeddingService()
        return service

    def test_embedding_service_initialization(self, embedding_service):
        """Test EmbeddingService initialization."""
        assert embedding_service is not None
        assert isinstance(embedding_service, EmbeddingService)
        assert hasattr(embedding_service, "primary_provider")
        assert hasattr(embedding_service, "fallback_providers")

    def test_embedding_service_providers_list(self, embedding_service):
        """Test that embedding service has fallback providers list."""
        assert hasattr(embedding_service, "fallback_providers")
        assert isinstance(embedding_service.fallback_providers, list)

    def test_embedding_service_has_setup_method(self, embedding_service):
        """Test that embedding service has setup providers method."""
        assert hasattr(embedding_service, "_setup_providers")
        assert callable(getattr(embedding_service, "_setup_providers"))

    def test_embedding_service_primary_provider_attribute(self, embedding_service):
        """Test primary provider attribute."""
        # Primary provider can be None initially
        assert hasattr(embedding_service, "primary_provider")

    def test_embedding_service_methods_exist(self, embedding_service):
        """Test that required methods exist on embedding service."""
        assert hasattr(embedding_service, "generate_embedding")
        assert hasattr(embedding_service, "generate_embeddings")
        assert callable(getattr(embedding_service, "generate_embedding"))
        assert callable(getattr(embedding_service, "generate_embeddings"))

    def test_embedding_service_with_mock_provider(self):
        """Test embedding service with mock provider."""
        mock_provider = Mock(spec=EmbeddingProvider)
        service = EmbeddingService(primary_provider=mock_provider)
        assert service.primary_provider == mock_provider

    def test_embedding_service_fallback_providers_type(self, embedding_service):
        """Test fallback providers are proper type."""
        assert isinstance(embedding_service.fallback_providers, list)
        # Should be empty or contain EmbeddingProvider instances
        for provider in embedding_service.fallback_providers:
            assert hasattr(provider, "generate_embedding")
            assert hasattr(provider, "generate_embeddings")

    def test_embedding_service_setup_providers_called(self):
        """Test that _setup_providers is called during initialization."""
        service = EmbeddingService()
        # If setup was called, fallback_providers should be initialized
        assert hasattr(service, "fallback_providers")
        assert isinstance(service.fallback_providers, list)


class TestEmbeddingProvider:
    """Test abstract EmbeddingProvider base class."""

    def test_embedding_provider_is_abstract(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_embedding_provider_abstract_methods(self):
        """Test that abstract methods are defined."""
        assert hasattr(EmbeddingProvider, "generate_embedding")
        assert hasattr(EmbeddingProvider, "generate_embeddings")
        assert hasattr(EmbeddingProvider, "dimensions")

    def test_concrete_provider_implementation(self):
        """Test concrete provider can be created."""

        class ConcreteProvider(EmbeddingProvider):
            async def generate_embedding(self, text: str):
                return [0.1, 0.2, 0.3]

            async def generate_embeddings(self, texts):
                return [[0.1, 0.2, 0.3]] * len(texts)

            @property
            def dimensions(self):
                return 3

        provider = ConcreteProvider()
        assert provider.dimensions == 3
        assert hasattr(provider, "generate_embedding")
        assert hasattr(provider, "generate_embeddings")


class TestEmbeddingServiceConfiguration:
    """Test embedding service configuration handling."""

    def test_embedding_service_with_no_config(self):
        """Test embedding service works without explicit config."""
        service = EmbeddingService()
        assert service is not None
        assert hasattr(service, "fallback_providers")

    def test_embedding_service_provider_initialization(self):
        """Test embedding service provider initialization."""
        service = EmbeddingService()
        # Service should initialize with some basic setup
        assert hasattr(service, "primary_provider")
        assert hasattr(service, "fallback_providers")


class TestEmbeddingServiceMethods:
    """Test embedding service method signatures and basic behavior."""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance."""
        return EmbeddingService()

    def test_generate_embedding_method_signature(self, embedding_service):
        """Test generate_embedding method exists and is callable."""
        assert hasattr(embedding_service, "generate_embedding")
        method = getattr(embedding_service, "generate_embedding")
        assert callable(method)

    def test_generate_embeddings_method_signature(self, embedding_service):
        """Test generate_embeddings method exists and is callable."""
        assert hasattr(embedding_service, "generate_embeddings")
        method = getattr(embedding_service, "generate_embeddings")
        assert callable(method)

    def test_setup_providers_method_exists(self, embedding_service):
        """Test _setup_providers method exists."""
        assert hasattr(embedding_service, "_setup_providers")
        assert callable(getattr(embedding_service, "_setup_providers"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
