#!/usr/bin/env python3
"""
Sparse Embedding Service for Hybrid Search.

Generates sparse (BM25-style) embeddings using fastembed for keyword matching
alongside dense semantic embeddings. This enables hybrid search where:
- Dense vectors capture semantic similarity
- Sparse vectors capture keyword/lexical matching

Uses Qdrant's BM25 model via fastembed for efficient sparse vector generation.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Configuration
SPARSE_MODEL_NAME = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
SPARSE_ENABLED = os.getenv("SPARSE_EMBEDDINGS_ENABLED", "true").lower() == "true"


@dataclass
class SparseVector:
    """
    Sparse vector representation for Qdrant.

    Attributes:
        indices: List of non-zero dimension indices
        values: List of values at those indices
    """
    indices: List[int]
    values: List[float]

    def to_dict(self) -> Dict[str, List]:
        """Convert to dict format for Qdrant."""
        return {
            "indices": self.indices,
            "values": self.values
        }

    def __len__(self) -> int:
        """Return number of non-zero elements."""
        return len(self.indices)


class SparseEmbeddingService:
    """
    Service for generating sparse embeddings using fastembed.

    Uses the Qdrant/bm25 model which provides BM25-style sparse vectors
    optimized for keyword matching in hybrid search scenarios.
    """

    def __init__(self, model_name: str = SPARSE_MODEL_NAME):
        """
        Initialize the sparse embedding service.

        Args:
            model_name: Name of the sparse embedding model to use.
                       Default: "Qdrant/bm25"
        """
        self.model_name = model_name
        self._model = None
        self._initialized = False
        self._metrics = {
            "total_embeddings": 0,
            "total_tokens": 0,
            "errors": 0,
        }

        logger.info(f"SparseEmbeddingService created with model: {model_name}")

    async def initialize(self) -> bool:
        """
        Initialize the sparse embedding model.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        if not SPARSE_ENABLED:
            logger.info("Sparse embeddings disabled via SPARSE_EMBEDDINGS_ENABLED=false")
            return False

        try:
            from fastembed import SparseTextEmbedding

            logger.info(f"Loading sparse embedding model: {self.model_name}")
            self._model = SparseTextEmbedding(model_name=self.model_name)
            self._initialized = True
            logger.info(f"✅ Sparse embedding model loaded: {self.model_name}")
            return True

        except ImportError as e:
            logger.error(f"❌ fastembed not installed: {e}")
            logger.error("   Install with: pip install fastembed")
            return False

        except Exception as e:
            logger.error(f"❌ Failed to load sparse embedding model: {e}")
            self._metrics["errors"] += 1
            return False

    def is_available(self) -> bool:
        """Check if sparse embeddings are available."""
        return self._initialized and self._model is not None

    def generate_sparse_embedding(self, text: str) -> Optional[SparseVector]:
        """
        Generate sparse embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            SparseVector with indices and values, or None if unavailable
        """
        if not self.is_available():
            logger.warning("Sparse embedding service not available")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for sparse embedding")
            return SparseVector(indices=[], values=[])

        try:
            # fastembed returns generator, get first result
            embeddings = list(self._model.embed([text]))

            if not embeddings:
                logger.warning("No sparse embedding generated")
                return SparseVector(indices=[], values=[])

            sparse_embedding = embeddings[0]

            # fastembed returns SparseEmbedding with indices and values attributes
            indices = sparse_embedding.indices.tolist()
            values = sparse_embedding.values.tolist()

            self._metrics["total_embeddings"] += 1
            self._metrics["total_tokens"] += len(indices)

            logger.debug(f"Generated sparse embedding: {len(indices)} non-zero elements")

            return SparseVector(indices=indices, values=values)

        except Exception as e:
            logger.error(f"Error generating sparse embedding: {e}")
            self._metrics["errors"] += 1
            return None

    def generate_sparse_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[Optional[SparseVector]]:
        """
        Generate sparse embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of SparseVectors (None for any that failed)
        """
        if not self.is_available():
            logger.warning("Sparse embedding service not available")
            return [None] * len(texts)

        try:
            # Filter empty texts
            valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
            valid_texts = [texts[i] for i in valid_indices]

            if not valid_texts:
                return [SparseVector(indices=[], values=[]) for _ in texts]

            # Generate embeddings for valid texts
            embeddings = list(self._model.embed(valid_texts))

            # Map back to original positions
            results = [SparseVector(indices=[], values=[]) for _ in texts]
            for i, embedding in zip(valid_indices, embeddings):
                results[i] = SparseVector(
                    indices=embedding.indices.tolist(),
                    values=embedding.values.tolist()
                )
                self._metrics["total_embeddings"] += 1
                self._metrics["total_tokens"] += len(embedding.indices)

            return results

        except Exception as e:
            logger.error(f"Error generating batch sparse embeddings: {e}")
            self._metrics["errors"] += 1
            return [None] * len(texts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            **self._metrics,
            "model": self.model_name,
            "initialized": self._initialized,
            "enabled": SPARSE_ENABLED,
        }


# Global instance
_sparse_service: Optional[SparseEmbeddingService] = None


def get_sparse_embedding_service() -> SparseEmbeddingService:
    """Get or create the global sparse embedding service."""
    global _sparse_service
    if _sparse_service is None:
        _sparse_service = SparseEmbeddingService()
    return _sparse_service


async def generate_sparse_embedding(text: str) -> Optional[SparseVector]:
    """
    Convenience function to generate a sparse embedding.

    Args:
        text: Text to embed

    Returns:
        SparseVector or None if unavailable
    """
    service = get_sparse_embedding_service()
    if not service.is_available():
        await service.initialize()
    return service.generate_sparse_embedding(text)
