#!/usr/bin/env python3
"""
Comprehensive test coverage for embedding_config.py module.

This test suite provides thorough coverage of:
- EmbeddingConfig class with all provider types
- EmbeddingGenerator class with initialization, embedding generation, and batch operations
- create_embedding_generator factory function
- Error handling, fallback scenarios, and edge cases
- Environment variable configurations
- Mock external dependencies (OpenAI and HuggingFace)
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.core.embedding_config import (  # noqa: E402
    EmbeddingConfig,
    EmbeddingGenerator,
    create_embedding_generator,
)


class TestEmbeddingConfig:
    """Test suite for EmbeddingConfig."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = {"embeddings": {"provider": "openai"}}
        embedding_config = EmbeddingConfig(config)
        assert embedding_config.provider == "openai"

    def test_init_with_environment(self):
        """Test initialization with environment variables."""
        config = {}
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "huggingface"}):
            embedding_config = EmbeddingConfig(config)
            assert embedding_config.provider == "huggingface"

    def test_get_provider_unknown(self):
        """Test fallback for unknown provider."""
        config = {"embeddings": {"provider": "unknown_provider"}}
        embedding_config = EmbeddingConfig(config)
        assert embedding_config.provider == "development"

    def test_get_model_from_env(self):
        """Test getting model from environment."""
        config = {"embeddings": {"provider": "openai"}}
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "text-embedding-3-large"}):
            embedding_config = EmbeddingConfig(config)
            assert embedding_config.model == "text-embedding-3-large"

    def test_get_model_invalid(self):
        """Test fallback for invalid model."""
        config = {"embeddings": {"provider": "openai", "model": "invalid_model"}}
        embedding_config = EmbeddingConfig(config)
        assert embedding_config.model == "text-embedding-3-small"  # Default

    def test_get_api_key_openai(self):
        """Test getting OpenAI API key."""
        config = {"embeddings": {"provider": "openai"}}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            embedding_config = EmbeddingConfig(config)
            assert embedding_config.get_api_key() == "test_key"

    def test_get_api_key_huggingface(self):
        """Test getting HuggingFace API key."""
        config = {"embeddings": {"provider": "huggingface"}}
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "test_key"}):
            embedding_config = EmbeddingConfig(config)
            assert embedding_config.get_api_key() == "test_key"

    def test_get_api_key_development(self):
        """Test getting API key for development provider."""
        config = {"embeddings": {"provider": "development"}}
        embedding_config = EmbeddingConfig(config)
        assert embedding_config.get_api_key() is None

    def test_validate_configuration_dev_in_prod(self):
        """Test validation warning for development in production."""
        config = {"embeddings": {"provider": "development"}}
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            embedding_config = EmbeddingConfig(config)
            result = embedding_config.validate_configuration()
            assert not result["valid"] is False  # Should have warnings but still be valid
            assert len(result["warnings"]) > 0

    def test_validate_configuration_missing_api_key(self):
        """Test validation error for missing API key."""
        config = {"embeddings": {"provider": "openai"}}
        with patch.dict(os.environ, {}, clear=True):
            embedding_config = EmbeddingConfig(config)
            result = embedding_config.validate_configuration()
            assert result["valid"] is False
            assert len(result["errors"]) > 0

    def test_validate_configuration_unusual_dimensions(self):
        """Test validation warning for unusual dimensions."""
        config = {"embeddings": {"provider": "development"}}
        embedding_config = EmbeddingConfig(config)
        # Mock unusual dimensions
        embedding_config.dimensions = 999
        result = embedding_config.validate_configuration()
        assert len(result["warnings"]) > 0


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""

    def test_init(self):
        """Test initialization."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)
        assert generator.config == config
        assert generator._client is None
        assert generator._model is None

    @pytest.mark.asyncio
    async def test_initialize_development(self):
        """Test initialization for development provider."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)
        result = await generator.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_initialize_unsupported(self):
        """Test initialization for unsupported provider."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        config.provider = "unsupported"
        generator = EmbeddingGenerator(config)
        result = await generator.initialize()
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_openai_success(self):
        """Test successful OpenAI initialization."""
        with patch("builtins.__import__") as mock_import:
            mock_openai = MagicMock()
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            mock_import.return_value = mock_openai

            config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                generator = EmbeddingGenerator(config)
                result = await generator.initialize()
                assert result is True
                assert generator._client == mock_client
                mock_openai.OpenAI.assert_called_once_with(api_key="test_key")

    @pytest.mark.asyncio
    async def test_initialize_openai_no_key(self):
        """Test OpenAI initialization without API key."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        with patch.dict(os.environ, {}, clear=True):
            generator = EmbeddingGenerator(config)
            result = await generator.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_initialize_openai_import_error(self):
        """Test OpenAI initialization with import error."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with patch("builtins.__import__", side_effect=ImportError("openai not installed")):
                generator = EmbeddingGenerator(config)
                result = await generator.initialize()
                assert result is False

    @pytest.mark.asyncio
    async def test_initialize_huggingface_success(self):
        """Test successful HuggingFace initialization."""
        with patch("builtins.__import__") as mock_import:
            mock_sentence_transformers = MagicMock()
            mock_model = Mock()
            mock_sentence_transformers.SentenceTransformer.return_value = mock_model
            mock_import.return_value = mock_sentence_transformers

            config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
            generator = EmbeddingGenerator(config)
            result = await generator.initialize()
            assert result is True
            assert generator._model == mock_model
            mock_sentence_transformers.SentenceTransformer.assert_called_once_with(config.model)

    @pytest.mark.asyncio
    async def test_initialize_huggingface_import_error(self):
        """Test HuggingFace initialization with import error."""
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        with patch(
            "builtins.__import__", side_effect=ImportError("sentence-transformers not installed")
        ):
            generator = EmbeddingGenerator(config)
            result = await generator.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_generate_embedding_development(self):
        """Test hash-based embedding generation."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)
        embedding = await generator.generate_embedding("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # Default dimensions
        assert all(isinstance(x, float) for x in embedding)
        assert all(-1.0 <= x <= 1.0 for x in embedding)  # Values should be in range [-1, 1]

    @pytest.mark.asyncio
    async def test_generate_embedding_openai_success(self):
        """Test successful OpenAI embedding generation."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        generator = EmbeddingGenerator(config)

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response
        generator._client = mock_client

        embedding = await generator.generate_embedding("test text")
        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once_with(input="test text", model=config.model)

    @pytest.mark.asyncio
    async def test_generate_embedding_openai_fallback(self):
        """Test OpenAI embedding with fallback."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        generator = EmbeddingGenerator(config)

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        generator._client = mock_client

        with patch.dict(os.environ, {"EMBEDDING_ALLOW_FALLBACK": "true"}):
            embedding = await generator.generate_embedding("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == config.dimensions

    @pytest.mark.asyncio
    async def test_generate_embedding_openai_no_fallback(self):
        """Test OpenAI embedding without fallback."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        generator = EmbeddingGenerator(config)

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        generator._client = mock_client

        with patch.dict(os.environ, {"EMBEDDING_ALLOW_FALLBACK": "false"}):
            with pytest.raises(
                RuntimeError, match="OpenAI embedding failed and fallback is disabled"
            ):
                await generator.generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_generate_embedding_huggingface_success(self):
        """Test successful HuggingFace embedding generation."""
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        generator = EmbeddingGenerator(config)

        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding
        generator._model = mock_model

        embedding = await generator.generate_embedding("test text")
        assert embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with("test text")
        mock_embedding.tolist.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embedding_huggingface_fallback(self):
        """Test HuggingFace embedding with fallback."""
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        generator = EmbeddingGenerator(config)

        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Model Error")
        generator._model = mock_model

        with patch.dict(os.environ, {"EMBEDDING_ALLOW_FALLBACK": "true"}):
            embedding = await generator.generate_embedding("test text")
            assert isinstance(embedding, list)
            assert len(embedding) == config.dimensions

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_openai(self):
        """Test batch embedding generation with OpenAI."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        generator = EmbeddingGenerator(config)

        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(), Mock()]
        mock_response.data[0].embedding = [0.1, 0.2]
        mock_response.data[1].embedding = [0.3, 0.4]
        mock_client.embeddings.create.return_value = mock_response
        generator._client = mock_client

        embeddings = await generator.generate_batch_embeddings(["text1", "text2"])
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]
        mock_client.embeddings.create.assert_called_once_with(
            input=["text1", "text2"], model=config.model
        )

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_huggingface(self):
        """Test batch embedding generation with HuggingFace."""
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        generator = EmbeddingGenerator(config)

        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.return_value = mock_embeddings
        generator._model = mock_model

        embeddings = await generator.generate_batch_embeddings(["text1", "text2"])
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]
        mock_model.encode.assert_called_once_with(["text1", "text2"])
        mock_embeddings.tolist.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_fallback(self):
        """Test batch embedding generation fallback."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)

        embeddings = await generator.generate_batch_embeddings(["text1", "text2"])
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == config.dimensions for emb in embeddings)
        # Test that different texts produce different embeddings
        assert embeddings[0] != embeddings[1]


@pytest.mark.asyncio
async def test_create_embedding_generator_success():
    """Test successful embedding generator creation."""
    config = {"embeddings": {"provider": "development"}}
    generator = await create_embedding_generator(config)
    assert isinstance(generator, EmbeddingGenerator)
    assert generator.config.provider == "development"


@pytest.mark.asyncio
async def test_create_embedding_generator_fallback():
    """Test embedding generator creation with fallback."""
    config = {"embeddings": {"provider": "openai"}}
    with patch.dict(os.environ, {}, clear=True):  # No API key
        generator = await create_embedding_generator(config)
        assert isinstance(generator, EmbeddingGenerator)
        assert generator.config.provider == "development"


class TestEmbeddingConfigExtensive:
    """Additional comprehensive tests for EmbeddingConfig."""

    def test_all_provider_defaults(self):
        """Test all provider configurations and their defaults."""
        # Test all providers
        for provider_name in EmbeddingConfig.PROVIDERS.keys():
            config_dict = {"embeddings": {"provider": provider_name}}
            embedding_config = EmbeddingConfig(config_dict)

            assert embedding_config.provider == provider_name
            expected_model = EmbeddingConfig.PROVIDERS[provider_name]["default_model"]
            assert embedding_config.model == expected_model

            expected_dimensions = EmbeddingConfig.PROVIDERS[provider_name]["dimensions"][
                expected_model
            ]
            assert embedding_config.dimensions == expected_dimensions

    def test_get_dimensions_edge_cases(self):
        """Test dimension retrieval for various scenarios."""
        # Test with custom dimensions
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        config.model = "unknown_model"
        dimensions = config._get_dimensions()
        assert dimensions == 1536  # Default fallback

    def test_provider_validation_cases(self):
        """Test various provider validation scenarios."""
        # Test empty config
        config = EmbeddingConfig({})
        assert config.provider == "development"  # Default fallback

        # Test None provider
        config = EmbeddingConfig({"embeddings": {"provider": None}})
        with patch.dict(os.environ, {}, clear=True):
            # Will use environment default
            new_config = EmbeddingConfig({"embeddings": {"provider": None}})
            assert new_config.provider == "development"

    def test_environment_variable_precedence(self):
        """Test that environment variables act as fallback when config is not set."""
        # Test provider env var fallback (when provider not in config)
        config_dict = {"embeddings": {}}  # No provider specified
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"}, clear=True):
            config = EmbeddingConfig(config_dict)
            assert config.provider == "openai"
            # Model should be OpenAI default
            assert config.model == "text-embedding-3-small"

        # Test model env var fallback (when model not in config)
        config_dict = {"embeddings": {"provider": "huggingface"}}  # No model specified
        with patch.dict(
            os.environ, {"EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"}, clear=True
        ):
            config = EmbeddingConfig(config_dict)
            assert config.provider == "huggingface"
            assert config.model == "sentence-transformers/all-MiniLM-L6-v2"

        # Test that config values take precedence over environment variables
        config_dict = {"embeddings": {"provider": "development", "model": "hash-based"}}
        with patch.dict(
            os.environ,
            {"EMBEDDING_PROVIDER": "openai", "EMBEDDING_MODEL": "text-embedding-3-large"},
            clear=True,
        ):
            config = EmbeddingConfig(config_dict)
            # Config values should take precedence
            assert config.provider == "development"
            assert config.model == "hash-based"

    def test_validate_configuration_comprehensive(self):
        """Test comprehensive validation scenarios."""
        # Test valid configuration
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        result = config.validate_configuration()
        assert result["valid"] is True
        assert result["provider"] == "development"
        assert result["model"] == "hash-based"
        assert result["dimensions"] == 1536
        assert isinstance(result["warnings"], list)
        assert isinstance(result["errors"], list)

        # Test development in production warning
        with patch.dict(os.environ, {"ENVIRONMENT": "PRODUCTION"}):
            result = config.validate_configuration()
            assert len(result["warnings"]) > 0
            assert "development embeddings in production" in result["warnings"][0].lower()

        # Test OpenAI without API key error
        openai_config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        with patch.dict(os.environ, {}, clear=True):
            result = openai_config.validate_configuration()
            assert result["valid"] is False
            assert len(result["errors"]) > 0
            assert "OPENAI_API_KEY is required" in result["errors"][0]

        # Test unusual dimensions warning
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        config.dimensions = 999  # Unusual dimension
        result = config.validate_configuration()
        assert len(result["warnings"]) > 0
        assert "may not be optimal" in result["warnings"][-1]

        # Test optimal dimensions (no warning)
        for optimal_dim in [384, 768, 1536, 3072]:
            config.dimensions = optimal_dim
            result = config.validate_configuration()
            # Should not have dimension warning
            dimension_warnings = [w for w in result["warnings"] if "may not be optimal" in w]
            assert len(dimension_warnings) == 0

    def test_get_api_key_scenarios(self):
        """Test API key retrieval for all scenarios."""
        # Test OpenAI with key
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            assert config.get_api_key() == "sk-test123"

        # Test OpenAI without key
        with patch.dict(os.environ, {}, clear=True):
            assert config.get_api_key() is None

        # Test HuggingFace with key
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        with patch.dict(os.environ, {"HUGGINGFACE_API_KEY": "hf_test123"}):
            assert config.get_api_key() == "hf_test123"

        # Test HuggingFace without key (optional)
        with patch.dict(os.environ, {}, clear=True):
            assert config.get_api_key() is None

        # Test development (no key needed)
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        assert config.get_api_key() is None


class TestEmbeddingGeneratorExtensive:
    """Additional comprehensive tests for EmbeddingGenerator."""

    @pytest.mark.asyncio
    async def test_initialize_exception_handling(self):
        """Test exception handling during initialization."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        generator = EmbeddingGenerator(config)

        # Test general exception handling
        with patch.object(
            generator, "_initialize_openai", side_effect=RuntimeError("Unexpected error")
        ):
            result = await generator.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_initialize_openai_api_exception(self):
        """Test OpenAI initialization with API-level exceptions."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        generator = EmbeddingGenerator(config)

        with patch("builtins.__import__") as mock_import:
            mock_openai = MagicMock()
            mock_openai.OpenAI.side_effect = Exception("Connection failed")
            mock_import.return_value = mock_openai
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                result = await generator.initialize()
                assert result is False

    @pytest.mark.asyncio
    async def test_initialize_huggingface_model_exception(self):
        """Test HuggingFace initialization with model loading exceptions."""
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        generator = EmbeddingGenerator(config)

        with patch("builtins.__import__") as mock_import:
            mock_sentence_transformers = MagicMock()
            mock_sentence_transformers.SentenceTransformer.side_effect = Exception(
                "Model loading failed"
            )
            mock_import.return_value = mock_sentence_transformers
            result = await generator.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_generate_embedding_unsupported_provider(self):
        """Test embedding generation with unsupported provider."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        config.provider = "unsupported_provider"  # Force unsupported
        generator = EmbeddingGenerator(config)

        with pytest.raises(ValueError, match="Unsupported provider: unsupported_provider"):
            await generator.generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_generate_embedding_huggingface_no_fallback(self):
        """Test HuggingFace embedding without fallback enabled."""
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        generator = EmbeddingGenerator(config)

        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Model Error")
        generator._model = mock_model

        with patch.dict(os.environ, {"EMBEDDING_ALLOW_FALLBACK": "false"}):
            with pytest.raises(
                RuntimeError, match="HuggingFace embedding failed and fallback is disabled"
            ):
                await generator.generate_embedding("test text")

    def test_hash_embedding_deterministic(self):
        """Test that hash embeddings are deterministic."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)

        # Same input should produce same embedding
        embedding1 = generator._generate_hash_embedding("test text")
        embedding2 = generator._generate_hash_embedding("test text")
        assert embedding1 == embedding2

        # Different inputs should produce different embeddings
        embedding3 = generator._generate_hash_embedding("different text")
        assert embedding1 != embedding3

        # Test with various dimension sizes
        for dimensions in [128, 384, 768, 1536, 3072]:
            config.dimensions = dimensions
            embedding = generator._generate_hash_embedding("test")
            assert len(embedding) == dimensions
            assert all(isinstance(x, float) for x in embedding)
            assert all(-1.0 <= x <= 1.0 for x in embedding)

    def test_hash_embedding_edge_cases(self):
        """Test hash embedding with edge cases."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)

        # Test empty string
        embedding = generator._generate_hash_embedding("")
        assert len(embedding) == config.dimensions

        # Test unicode characters
        embedding = generator._generate_hash_embedding("こんにちは🚀")
        assert len(embedding) == config.dimensions

        # Test very long string
        long_text = "a" * 10000
        embedding = generator._generate_hash_embedding(long_text)
        assert len(embedding) == config.dimensions

    @pytest.mark.asyncio
    async def test_batch_embeddings_openai_error_fallback(self):
        """Test OpenAI batch embedding error handling with fallback."""
        config = EmbeddingConfig({"embeddings": {"provider": "openai"}})
        generator = EmbeddingGenerator(config)

        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Batch API Error")
        generator._client = mock_client

        embeddings = await generator._generate_openai_batch(["text1", "text2"])
        assert len(embeddings) == 2
        assert all(len(emb) == config.dimensions for emb in embeddings)
        # Should fallback to hash-based embeddings
        assert embeddings[0] != embeddings[1]  # Different texts, different hashes

    @pytest.mark.asyncio
    async def test_batch_embeddings_huggingface_error_fallback(self):
        """Test HuggingFace batch embedding error handling with fallback."""
        config = EmbeddingConfig({"embeddings": {"provider": "huggingface"}})
        generator = EmbeddingGenerator(config)

        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Batch Model Error")
        generator._model = mock_model

        embeddings = await generator._generate_huggingface_batch(["text1", "text2"])
        assert len(embeddings) == 2
        assert all(len(emb) == config.dimensions for emb in embeddings)
        # Should fallback to hash-based embeddings
        assert embeddings[0] != embeddings[1]  # Different texts, different hashes

    @pytest.mark.asyncio
    async def test_batch_embeddings_individual_fallback(self):
        """Test batch embeddings with individual embedding fallback for unsupported providers."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)

        # For development provider, should use individual generation
        embeddings = await generator.generate_batch_embeddings(["text1", "text2", "text3"])
        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == config.dimensions for emb in embeddings)

        # Each should be different
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]
        assert embeddings[0] != embeddings[2]


class TestCreateEmbeddingGeneratorExtensive:
    """Additional comprehensive tests for create_embedding_generator function."""

    @pytest.mark.asyncio
    async def test_create_generator_openai_success(self):
        """Test successful OpenAI generator creation."""
        config = {"embeddings": {"provider": "openai", "model": "text-embedding-3-large"}}

        with patch("builtins.__import__") as mock_import:
            mock_openai = MagicMock()
            mock_client = Mock()
            mock_openai.OpenAI.return_value = mock_client
            mock_import.return_value = mock_openai

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                generator = await create_embedding_generator(config)

                assert isinstance(generator, EmbeddingGenerator)
                assert generator.config.provider == "openai"
                assert generator.config.model == "text-embedding-3-large"
                assert generator._client == mock_client

    @pytest.mark.asyncio
    async def test_create_generator_huggingface_success(self):
        """Test successful HuggingFace generator creation."""
        config = {
            "embeddings": {
                "provider": "huggingface",
                "model": "sentence-transformers/all-mpnet-base-v2",
            }
        }

        with patch("builtins.__import__") as mock_import:
            mock_sentence_transformers = MagicMock()
            mock_model = Mock()
            mock_sentence_transformers.SentenceTransformer.return_value = mock_model
            mock_import.return_value = mock_sentence_transformers

            generator = await create_embedding_generator(config)

            assert isinstance(generator, EmbeddingGenerator)
            assert generator.config.provider == "huggingface"
            assert generator.config.model == "sentence-transformers/all-mpnet-base-v2"
            assert generator._model == mock_model

    @pytest.mark.asyncio
    async def test_create_generator_initialization_failure_fallback(self):
        """Test generator creation with initialization failure and fallback."""
        config = {"embeddings": {"provider": "openai"}}

        # Mock failed initialization
        with patch("builtins.__import__", side_effect=ImportError("openai not installed")):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
                generator = await create_embedding_generator(config)

                # Should fall back to development mode
                assert isinstance(generator, EmbeddingGenerator)
                assert generator.config.provider == "development"
                assert generator.config.model == "hash-based"
                assert generator.config.dimensions == 1536

    @pytest.mark.asyncio
    async def test_create_generator_with_empty_config(self):
        """Test generator creation with empty/minimal config."""
        # Empty config should use defaults
        generator = await create_embedding_generator({})

        assert isinstance(generator, EmbeddingGenerator)
        assert generator.config.provider == "development"  # Default
        assert generator.config.model == "hash-based"
        assert generator.config.dimensions == 1536

    @pytest.mark.asyncio
    async def test_create_generator_various_configs(self):
        """Test generator creation with various configuration combinations."""
        test_configs = [
            {"embeddings": {"provider": "development"}},
            {"embeddings": {"provider": "openai", "model": "text-embedding-ada-002"}},
            {
                "embeddings": {
                    "provider": "huggingface",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                }
            },
        ]

        for config in test_configs:
            with (
                patch("builtins.__import__") as mock_import,
                patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True),
            ):

                def mock_import_side_effect(name, *args, **kwargs):
                    if name == "openai":
                        mock_openai = MagicMock()
                        mock_openai.OpenAI.return_value = Mock()
                        return mock_openai
                    elif name == "sentence_transformers":
                        mock_st = MagicMock()
                        mock_st.SentenceTransformer.return_value = Mock()
                        return mock_st
                    else:
                        return MagicMock()

                mock_import.side_effect = mock_import_side_effect

                generator = await create_embedding_generator(config)

                assert isinstance(generator, EmbeddingGenerator)
                expected_provider = config["embeddings"]["provider"]

                # Check that the generator was created with expected provider
                # (or development as fallback if initialization failed)
                assert generator.config.provider in [expected_provider, "development"]


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_providers_constant_structure(self):
        """Test that PROVIDERS constant has expected structure."""
        providers = EmbeddingConfig.PROVIDERS

        assert isinstance(providers, dict)
        assert "openai" in providers
        assert "huggingface" in providers
        assert "development" in providers

        for provider_name, provider_info in providers.items():
            assert "models" in provider_info
            assert "default_model" in provider_info
            assert "dimensions" in provider_info

            assert isinstance(provider_info["models"], list)
            assert len(provider_info["models"]) > 0
            assert provider_info["default_model"] in provider_info["models"]

            for model in provider_info["models"]:
                assert model in provider_info["dimensions"]
                assert isinstance(provider_info["dimensions"][model], int)
                assert provider_info["dimensions"][model] > 0

    def test_config_with_none_values(self):
        """Test configuration with None values."""
        config = {"embeddings": {"provider": None, "model": None}}

        with patch.dict(os.environ, {}, clear=True):
            embedding_config = EmbeddingConfig(config)
            # Should fallback to defaults
            assert embedding_config.provider == "development"
            assert embedding_config.model in EmbeddingConfig.PROVIDERS["development"]["models"]

    def test_config_with_missing_embeddings_section(self):
        """Test configuration without embeddings section."""
        config = {"other_section": {"key": "value"}}

        embedding_config = EmbeddingConfig(config)
        # Should use all defaults
        assert embedding_config.provider == "development"
        assert embedding_config.model == "hash-based"
        assert embedding_config.dimensions == 1536

    @pytest.mark.asyncio
    async def test_embedding_generation_with_special_characters(self):
        """Test embedding generation with various special characters and encodings."""
        config = EmbeddingConfig({"embeddings": {"provider": "development"}})
        generator = EmbeddingGenerator(config)

        test_texts = [
            "Hello, World!",
            "¡Hola! ¿Cómo estás?",
            "こんにちは世界",
            "🚀💻🌟",
            "\n\t\r",
            """Multi
line
text
with
tabs	and
linebreaks""",
            "Numbers: 123456789 and symbols: !@#$%^&*()_+-=[]{}|;':,.<>?",
        ]

        for text in test_texts:
            embedding = await generator.generate_embedding(text)
            assert isinstance(embedding, list)
            assert len(embedding) == config.dimensions
            assert all(isinstance(x, float) for x in embedding)
            assert all(-1.0 <= x <= 1.0 for x in embedding)


# Additional integration tests
class TestIntegrationScenarios:
    """Integration tests that combine multiple components."""

    @pytest.mark.asyncio
    async def test_full_workflow_development(self):
        """Test complete workflow with development provider."""
        config = {"embeddings": {"provider": "development"}}

        # Create generator
        generator = await create_embedding_generator(config)
        assert generator.config.provider == "development"

        # Generate single embedding
        embedding = await generator.generate_embedding("test document")
        assert len(embedding) == 1536

        # Generate batch embeddings
        texts = ["doc1", "doc2", "doc3"]
        embeddings = await generator.generate_batch_embeddings(texts)
        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)

        # Verify embeddings are different for different texts
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]

    @pytest.mark.asyncio
    async def test_provider_switching_workflow(self):
        """Test workflow with provider switching scenarios."""
        # Start with OpenAI config but no API key (should fallback)
        config = {"embeddings": {"provider": "openai", "model": "text-embedding-3-small"}}

        with patch.dict(os.environ, {}, clear=True):  # No API key
            generator = await create_embedding_generator(config)
            # Should fallback to development
            assert generator.config.provider == "development"

            # Should still work for embedding generation
            embedding = await generator.generate_embedding("test")
            assert len(embedding) == 1536  # Development default

    def test_configuration_validation_comprehensive_workflow(self):
        """Test comprehensive configuration validation workflow."""
        # Test all combinations of providers and validation scenarios
        test_scenarios = [
            {
                "config": {"embeddings": {"provider": "development"}},
                "env": {},
                "expected_valid": True,
                "expected_warnings": 0,
                "expected_errors": 0,
            },
            {
                "config": {"embeddings": {"provider": "development"}},
                "env": {"ENVIRONMENT": "production"},
                "expected_valid": True,  # Valid but with warnings
                "expected_warnings": 1,  # Production warning
                "expected_errors": 0,
            },
            {
                "config": {"embeddings": {"provider": "openai"}},
                "env": {"OPENAI_API_KEY": "sk-test123"},
                "expected_valid": True,
                "expected_warnings": 0,
                "expected_errors": 0,
            },
            {
                "config": {"embeddings": {"provider": "openai"}},
                "env": {},  # No API key
                "expected_valid": False,
                "expected_warnings": 0,
                "expected_errors": 1,
            },
            {
                "config": {"embeddings": {"provider": "huggingface"}},
                "env": {},  # HuggingFace API key is optional
                "expected_valid": True,
                "expected_warnings": 0,
                "expected_errors": 0,
            },
        ]

        for scenario in test_scenarios:
            with patch.dict(os.environ, scenario["env"], clear=True):
                config = EmbeddingConfig(scenario["config"])
                result = config.validate_configuration()

                assert (
                    result["valid"] == scenario["expected_valid"]
                ), f"Failed for scenario: {scenario}"
                assert (
                    len(result["warnings"]) == scenario["expected_warnings"]
                ), f"Wrong warning count for: {scenario}"
                assert (
                    len(result["errors"]) == scenario["expected_errors"]
                ), f"Wrong error count for: {scenario}"
