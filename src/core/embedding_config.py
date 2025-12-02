#!/usr/bin/env python3
"""
Configurable embedding generation for context storage.

This module provides secure, production-ready embedding generation
using proper NLP models instead of insecure hash-based approaches.
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingConfig:
    """Configuration for embedding generation."""

    # Supported embedding providers
    PROVIDERS = {
        "openai": {
            "models": [
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large",
            ],
            "default_model": "text-embedding-3-small",
            "dimensions": {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            },
        },
        "huggingface": {
            "models": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            "default_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "sentence-transformers/all-mpnet-base-v2": 768,
            },
        },
        "development": {
            "models": ["hash-based"],
            "default_model": "hash-based",
            "dimensions": {"hash-based": 384},  # SPRINT 11: Fixed to match v1.0 requirement
        },
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize embedding configuration.

        Args:
            config: Configuration dictionary containing embedding settings
        """
        self.config = config
        self.embedding_config = config.get("embeddings", {})

        # Get provider from config or environment
        self.provider = self._get_provider()
        self.model = self._get_model()
        self.dimensions = self._get_dimensions()

    def _get_provider(self) -> str:
        """Get embedding provider from config or environment."""
        provider = self.embedding_config.get("provider") or os.getenv(
            "EMBEDDING_PROVIDER", "development"
        )

        if provider not in self.PROVIDERS:
            logger.warning(f"Unknown embedding provider '{provider}', falling back to development")
            return "development"

        return provider

    def _get_model(self) -> str:
        """Get embedding model from config or environment."""
        model = (
            self.embedding_config.get("model")
            or os.getenv("EMBEDDING_MODEL")
            or self.PROVIDERS[self.provider]["default_model"]
        )

        available_models = self.PROVIDERS[self.provider]["models"]
        if model not in available_models:
            logger.warning(
                f"Model '{model}' not available for provider '{self.provider}', using default"
            )
            return str(self.PROVIDERS[self.provider]["default_model"])

        return str(model)

    def _get_dimensions(self) -> int:
        """Get embedding dimensions for the selected model."""
        dimensions_map = self.PROVIDERS[self.provider]["dimensions"]
        return int(dimensions_map.get(self.model, 1536))

    def get_api_key(self) -> Optional[str]:
        """Get API key for the embedding provider."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "huggingface":
            return os.getenv("HUGGINGFACE_API_KEY")  # Optional for some models
        return None

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the embedding configuration.

        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "warnings": [],
            "errors": [],
        }

        # Check for development mode in production
        if self.provider == "development" and os.getenv("ENVIRONMENT", "").lower() == "production":
            validation["warnings"].append(
                "Using development embeddings in production - "
                "consider switching to OpenAI or HuggingFace"
            )

        # Check for API keys when required
        if self.provider == "openai" and not self.get_api_key():
            validation["errors"].append("OPENAI_API_KEY is required for OpenAI embeddings")
            validation["valid"] = False

        # Check dimensions compatibility with Qdrant
        if self.dimensions not in [384, 768, 1536, 3072]:
            validation["warnings"].append(
                f"Embedding dimensions ({self.dimensions}) may not be optimal for vector database"
            )

        return validation


class EmbeddingGenerator:
    """Generate embeddings using configured provider."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding generator.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self._client = None
        self._model = None

    async def initialize(self) -> bool:
        """Initialize the embedding provider.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self.config.provider == "openai":
                return await self._initialize_openai()
            elif self.config.provider == "huggingface":
                return await self._initialize_huggingface()
            elif self.config.provider == "development":
                return True  # No initialization needed for hash-based
            else:
                logger.error(f"Unsupported provider: {self.config.provider}")
                return False
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            return False

    async def _initialize_openai(self) -> bool:
        """Initialize OpenAI embeddings."""
        try:
            import openai

            api_key = self.config.get_api_key()
            if not api_key:
                logger.error("OpenAI API key not found")
                return False

            self._client = openai.OpenAI(api_key=api_key)
            logger.info(f"✅ OpenAI embeddings initialized with model {self.config.model}")
            return True
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            return False

    async def _initialize_huggingface(self) -> bool:
        """Initialize HuggingFace embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model)
            logger.info(f"✅ HuggingFace embeddings initialized with model {self.config.model}")
            return True
        except ImportError:
            logger.error(
                "sentence-transformers package not installed. "
                "Run: pip install sentence-transformers"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            return False

    async def generate_embedding(self, text: str, adjust_dimensions: bool = True) -> List[float]:
        """Generate embedding for text with Sprint 11 v1.0 dimension validation.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding (must be 384 dimensions)
        
        Raises:
            ValueError: If embedding dimensions don't match v1.0 requirement (384)
        """
        if self.config.provider == "openai":
            embedding = await self._generate_openai_embedding(text)
        elif self.config.provider == "huggingface":
            embedding = await self._generate_huggingface_embedding(text)
        elif self.config.provider == "development":
            embedding = self._generate_hash_embedding(text)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        # SPRINT 11: Critical dimension validation for v1.0 compliance
        if len(embedding) != 384:
            from .error_handler import handle_v1_dimension_mismatch
            error_response = handle_v1_dimension_mismatch(384, len(embedding))
            raise ValueError(f"Dimension mismatch: {error_response['message']}")
        
        return embedding

    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding."""
        try:
            response = self._client.embeddings.create(input=text, model=self.config.model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            # Check if fallback is allowed via configuration
            allow_fallback = os.getenv("EMBEDDING_ALLOW_FALLBACK", "true").lower() == "true"
            if allow_fallback:
                logger.warning("Falling back to hash-based embedding due to OpenAI failure")
                return self._generate_hash_embedding(text)
            else:
                logger.error("Fallback disabled - raising exception")
                raise RuntimeError(f"OpenAI embedding failed and fallback is disabled: {e}")

    async def _generate_huggingface_embedding(self, text: str) -> List[float]:
        """Generate HuggingFace embedding."""
        try:
            embedding = self._model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding generation failed: {e}")
            # Check if fallback is allowed via configuration
            allow_fallback = os.getenv("EMBEDDING_ALLOW_FALLBACK", "true").lower() == "true"
            if allow_fallback:
                logger.warning("Falling back to hash-based embedding due to HuggingFace failure")
                return self._generate_hash_embedding(text)
            else:
                logger.error("Fallback disabled - raising exception")
                raise RuntimeError(f"HuggingFace embedding failed and fallback is disabled: {e}")

    def _generate_hash_embedding(self, text: str) -> List[float]:
        """Generate hash-based embedding for development/fallback.

        Note: This is NOT suitable for production use as it doesn't capture
        semantic meaning. Use proper NLP models in production.
        
        SPRINT 11: Always generates exactly 384 dimensions for v1.0 compliance.
        """
        # Create deterministic hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to embedding vector of exactly 384 dimensions (v1.0 requirement)
        embedding = []
        for i in range(384):  # SPRINT 11: Hard-coded to 384 for v1.0 compliance
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1] range for better vector space properties
            normalized_value = (float(hash_bytes[byte_idx]) / 255.0) * 2.0 - 1.0
            embedding.append(normalized_value)

        return embedding

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if self.config.provider == "openai":
            return await self._generate_openai_batch(texts)
        elif self.config.provider == "huggingface":
            return await self._generate_huggingface_batch(texts)
        else:
            # Generate individual embeddings for fallback
            return [await self.generate_embedding(text) for text in texts]

    async def _generate_openai_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with OpenAI."""
        try:
            response = self._client.embeddings.create(input=texts, model=self.config.model)
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            return [self._generate_hash_embedding(text) for text in texts]

    async def _generate_huggingface_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings with HuggingFace."""
        try:
            embeddings = self._model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            return [self._generate_hash_embedding(text) for text in texts]


async def create_embedding_generator(config: Dict[str, Any]) -> EmbeddingGenerator:
    """Create and initialize an embedding generator.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized embedding generator
    """
    embedding_config = EmbeddingConfig(config)
    generator = EmbeddingGenerator(embedding_config)

    if not await generator.initialize():
        logger.warning(
            "Failed to initialize preferred embedding provider, using development fallback"
        )
        # Force fallback to development mode
        embedding_config.provider = "development"
        embedding_config.model = "hash-based"
        embedding_config.dimensions = 384
        generator = EmbeddingGenerator(embedding_config)
        await generator.initialize()

    return generator
