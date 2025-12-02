#!/usr/bin/env python3
"""
Embedding service with proper fallback strategies.
Provides consistent embedding generation across all components.
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model to use for embeddings
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._dimensions = Config.EMBEDDING_DIMENSIONS

        # Import OpenAI client if available
        try:
            import openai

            self.client = openai.Client(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai")
            self.client = None

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        if not self.client:
            raise RuntimeError("OpenAI client not configured")

        try:
            response = await self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.client:
            raise RuntimeError("OpenAI client not configured")

        try:
            response = await self.client.embeddings.create(model=self.model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI batch embedding generation failed: {e}")
            raise

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformer embedding provider for local embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Sentence Transformer provider.

        Args:
            model_name: Model name to use
        """
        self.model_name = model_name
        self.model = None
        self._dimensions = None

        # Try to load the model
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self._dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded SentenceTransformer model: {model_name} with {self._dimensions} dimensions"
            )
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformer."""
        if not self.model:
            raise RuntimeError("SentenceTransformer model not loaded")

        try:
            # SentenceTransformer is synchronous, so we run it directly
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer embedding generation failed: {e}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.model:
            raise RuntimeError("SentenceTransformer model not loaded")

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"SentenceTransformer batch embedding generation failed: {e}")
            raise

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions or 384  # Default for MiniLM


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Deterministic fallback embedding provider using advanced hashing."""

    def __init__(self, dimensions: int = None):
        """Initialize deterministic provider.

        Args:
            dimensions: Embedding dimensions
        """
        self._dimensions = dimensions or Config.EMBEDDING_DIMENSIONS

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding using advanced hashing.

        This uses multiple hash functions and combines them to create
        a more distributed embedding than simple hash-based approach.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Use multiple hash functions for better distribution
        hash_funcs = [
            hashlib.sha256,
            hashlib.sha512,
            hashlib.blake2b,
            hashlib.md5,
        ]

        # Generate multiple hashes
        hashes = []
        for func in hash_funcs:
            hash_obj = func(text.encode("utf-8"))
            hash_bytes = hash_obj.digest()
            hashes.append(hash_bytes)

        # Combine hashes
        combined = b"".join(hashes)

        # Generate embedding with better distribution
        embedding = []
        np.random.seed(int.from_bytes(combined[:4], "big"))

        # Use normal distribution for more realistic embeddings
        for i in range(self._dimensions):
            # Use different parts of the hash for each dimension
            idx = i % len(combined)
            seed_value = int.from_bytes(combined[idx : idx + 4], "big", signed=False)
            np.random.seed(seed_value)

            # Generate value from normal distribution
            value = np.random.normal(0, 0.5)
            # Clip to reasonable range
            value = np.clip(value, -1, 1)
            embedding.append(float(value))

        # Add some text-based features
        text_features = self._extract_text_features(text)
        for i, feature in enumerate(text_features[: min(10, self._dimensions)]):
            embedding[i] = (embedding[i] + feature) / 2

        return embedding

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def _extract_text_features(self, text: str) -> List[float]:
        """Extract simple text features for embedding enhancement.

        Args:
            text: Text to analyze

        Returns:
            Feature vector
        """
        features = []

        # Text length (normalized)
        features.append(min(len(text) / 1000.0, 1.0))

        # Word count (normalized)
        words = text.split()
        features.append(min(len(words) / 100.0, 1.0))

        # Average word length (normalized)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            features.append(min(avg_word_len / 10.0, 1.0))
        else:
            features.append(0.0)

        # Uppercase ratio
        if text:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
            features.append(uppercase_ratio)
        else:
            features.append(0.0)

        # Digit ratio
        if text:
            digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
            features.append(digit_ratio)
        else:
            features.append(0.0)

        # Special character ratio
        if text:
            special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
            features.append(special_ratio)
        else:
            features.append(0.0)

        # Line count (normalized)
        features.append(min(text.count("\n") / 100.0, 1.0))

        # Sentence count estimate (normalized)
        features.append(min((text.count(".") + text.count("!") + text.count("?")) / 50.0, 1.0))

        # Code-like features
        features.append(1.0 if "def " in text or "function " in text else 0.0)
        features.append(1.0 if "import " in text or "require(" in text else 0.0)

        # Convert to range [-1, 1]
        return [(f * 2 - 1) for f in features]

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions


class EmbeddingService:
    """Main embedding service with fallback support."""

    def __init__(self, primary_provider: Optional[EmbeddingProvider] = None):
        """Initialize embedding service.

        Args:
            primary_provider: Primary embedding provider to use
        """
        self.primary_provider = primary_provider
        self.fallback_providers: List[EmbeddingProvider] = []

        # Setup default providers
        self._setup_providers()

    def _setup_providers(self):
        """Setup embedding providers with fallbacks."""
        # If no primary provider, try to setup OpenAI
        if not self.primary_provider:
            if os.getenv("OPENAI_API_KEY"):
                try:
                    self.primary_provider = OpenAIEmbeddingProvider()
                    logger.info("Using OpenAI as primary embedding provider")
                except Exception as e:
                    logger.warning(f"Failed to setup OpenAI provider: {e}")

        # Add SentenceTransformer as first fallback
        try:
            st_provider = SentenceTransformerProvider()
            if st_provider.model:
                self.fallback_providers.append(st_provider)
                logger.info("Added SentenceTransformer as fallback provider")
        except Exception as e:
            logger.warning(f"Failed to setup SentenceTransformer: {e}")

        # Always add deterministic provider as last fallback
        self.fallback_providers.append(DeterministicEmbeddingProvider())
        logger.info("Added deterministic provider as final fallback")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with fallback support.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Try primary provider
        if self.primary_provider:
            try:
                return await self.primary_provider.generate_embedding(text)
            except Exception as e:
                logger.warning(f"Primary provider failed: {e}")

        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                return await provider.generate_embedding(text)
            except Exception as e:
                logger.warning(f"Fallback provider {provider.__class__.__name__} failed: {e}")

        # This should never happen as deterministic provider should always work
        raise RuntimeError("All embedding providers failed")

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Try primary provider
        if self.primary_provider:
            try:
                return await self.primary_provider.generate_embeddings(texts)
            except Exception as e:
                logger.warning(f"Primary provider batch failed: {e}")

        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                return await provider.generate_embeddings(texts)
            except Exception as e:
                logger.warning(f"Fallback provider {provider.__class__.__name__} batch failed: {e}")

        # This should never happen
        raise RuntimeError("All embedding providers failed")

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        if self.primary_provider:
            return self.primary_provider.dimensions
        elif self.fallback_providers:
            return self.fallback_providers[0].dimensions
        else:
            return Config.EMBEDDING_DIMENSIONS


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance.

    Returns:
        Embedding service
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
