"""
Semantic cache key generation for query caching.

Generates cache keys from quantized embeddings instead of raw text,
ensuring that semantically similar queries share cache entries.

This addresses the S3-Paraphrase-Robustness issue where different
phrasings of the same query (e.g., "configure Neo4j" vs "set up Neo4j")
would get different cache keys despite being semantically equivalent.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic cache key generation."""

    enabled: bool = True
    quantization_precision: int = 1  # Decimal places for rounding (1 = 0.1 precision)
    embedding_prefix_length: int = 32  # Number of embedding dimensions to use
    cache_key_prefix: str = "semantic"
    metrics_enabled: bool = True


@dataclass
class CacheKeyResult:
    """Result of cache key generation with metadata."""

    cache_key: str
    is_semantic: bool
    generation_time_ms: float
    embedding_hash: Optional[str] = None
    fallback_reason: Optional[str] = None


class SemanticCacheKeyGenerator:
    """
    Generate cache keys from semantic embeddings.

    Instead of using raw query text for cache keys, this generator
    creates keys from quantized embeddings. This ensures that
    semantically similar queries (paraphrases) will often share
    the same cache key, improving cache hit rates and result consistency.

    Example:
        "How do I configure Neo4j?" and "What are the steps to set up Neo4j?"
        may produce the same or similar embeddings, leading to cache hits.
    """

    def __init__(self, config: Optional[SemanticCacheConfig] = None):
        """
        Initialize the semantic cache key generator.

        Args:
            config: Configuration options. If None, uses defaults from environment.
        """
        if config is None:
            config = SemanticCacheConfig(
                enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true",
                quantization_precision=int(os.getenv("SEMANTIC_CACHE_PRECISION", "1")),
                embedding_prefix_length=int(os.getenv("SEMANTIC_CACHE_PREFIX_LENGTH", "32")),
            )

        self.config = config
        self._metrics = {
            "total_generations": 0,
            "semantic_keys_generated": 0,
            "fallback_keys_generated": 0,
            "average_generation_time_ms": 0.0,
            "cache_key_collisions": 0,  # Same key from different queries
        }
        self._recent_keys: Dict[str, str] = {}  # query -> key mapping for collision detection

        logger.info(
            f"SemanticCacheKeyGenerator initialized: enabled={config.enabled}, "
            f"precision={config.quantization_precision}, prefix_length={config.embedding_prefix_length}"
        )

    def quantize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Quantize embedding to reduce sensitivity to small differences.

        By rounding embedding values and using only a prefix of dimensions,
        we create a coarser representation that groups similar embeddings together.

        Args:
            embedding: Full embedding vector (e.g., 384 dimensions)

        Returns:
            Quantized prefix of the embedding
        """
        # Take only the first N dimensions
        prefix = embedding[: self.config.embedding_prefix_length]

        # Round to specified precision
        quantized = [round(x, self.config.quantization_precision) for x in prefix]

        return quantized

    def generate_cache_key(
        self,
        embedding: List[float],
        limit: int,
        search_mode: str,
        context_type: Optional[str] = None,
        sort_by: str = "relevance",
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> CacheKeyResult:
        """
        Generate a semantic cache key from an embedding.

        Args:
            embedding: Query embedding vector
            limit: Number of results requested
            search_mode: Search mode (e.g., "hybrid", "vector", "keyword")
            context_type: Optional filter by context type
            sort_by: Sort order for results
            additional_params: Any additional parameters that affect results

        Returns:
            CacheKeyResult with the generated key and metadata
        """
        start_time = time.time()
        self._metrics["total_generations"] += 1

        if not self.config.enabled:
            return CacheKeyResult(
                cache_key="",
                is_semantic=False,
                generation_time_ms=0.0,
                fallback_reason="semantic_cache_disabled",
            )

        try:
            # Quantize the embedding
            quantized = self.quantize_embedding(embedding)

            # Build cache parameters
            cache_params = {
                "embedding_prefix": quantized,
                "limit": limit,
                "search_mode": search_mode,
                "context_type": context_type,
                "sort_by": sort_by,
            }

            # Add any additional parameters
            if additional_params:
                cache_params["additional"] = additional_params

            # Generate hash
            params_json = json.dumps(cache_params, sort_keys=True)
            cache_hash = hashlib.sha256(params_json.encode()).hexdigest()

            # Create prefixed key
            cache_key = f"{self.config.cache_key_prefix}:{cache_hash[:16]}"

            generation_time_ms = (time.time() - start_time) * 1000

            # Update metrics
            self._metrics["semantic_keys_generated"] += 1
            self._update_average_time(generation_time_ms)

            # Track for collision detection (optional, for metrics)
            embedding_hash = hashlib.md5(str(embedding[:8]).encode()).hexdigest()[:8]

            logger.debug(
                f"Generated semantic cache key: {cache_key} "
                f"(quantized {len(quantized)} dims, {generation_time_ms:.2f}ms)"
            )

            return CacheKeyResult(
                cache_key=cache_key,
                is_semantic=True,
                generation_time_ms=generation_time_ms,
                embedding_hash=embedding_hash,
            )

        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            self._metrics["fallback_keys_generated"] += 1

            logger.warning(f"Semantic cache key generation failed: {e}")

            return CacheKeyResult(
                cache_key="",
                is_semantic=False,
                generation_time_ms=generation_time_ms,
                fallback_reason=str(e),
            )

    def generate_text_fallback_key(
        self,
        query: str,
        limit: int,
        search_mode: str,
        context_type: Optional[str] = None,
        sort_by: str = "relevance",
    ) -> str:
        """
        Generate a traditional text-based cache key as fallback.

        This is used when semantic key generation fails or is disabled.

        Args:
            query: Raw query text
            limit: Number of results requested
            search_mode: Search mode
            context_type: Optional context type filter
            sort_by: Sort order

        Returns:
            Text-based cache key
        """
        cache_params = {
            "query": query,
            "limit": limit,
            "search_mode": search_mode,
            "context_type": context_type,
            "sort_by": sort_by,
        }

        cache_hash = hashlib.sha256(json.dumps(cache_params, sort_keys=True).encode()).hexdigest()

        return f"text:{cache_hash[:16]}"

    def _update_average_time(self, new_time_ms: float) -> None:
        """Update running average of generation time."""
        total = self._metrics["semantic_keys_generated"]
        if total > 0:
            current_avg = self._metrics["average_generation_time_ms"]
            self._metrics["average_generation_time_ms"] = (current_avg * (total - 1) + new_time_ms) / total

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache key generation metrics."""
        total = self._metrics["total_generations"]
        return {
            **self._metrics,
            "semantic_key_rate": (
                self._metrics["semantic_keys_generated"] / total if total > 0 else 0.0
            ),
            "fallback_rate": (
                self._metrics["fallback_keys_generated"] / total if total > 0 else 0.0
            ),
            "config": {
                "enabled": self.config.enabled,
                "quantization_precision": self.config.quantization_precision,
                "embedding_prefix_length": self.config.embedding_prefix_length,
            },
        }

    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self._metrics = {
            "total_generations": 0,
            "semantic_keys_generated": 0,
            "fallback_keys_generated": 0,
            "average_generation_time_ms": 0.0,
            "cache_key_collisions": 0,
        }


# Global instance
_semantic_cache_generator: Optional[SemanticCacheKeyGenerator] = None


def get_semantic_cache_generator() -> SemanticCacheKeyGenerator:
    """
    Get or create the global semantic cache key generator.

    Returns:
        Global SemanticCacheKeyGenerator instance
    """
    global _semantic_cache_generator
    if _semantic_cache_generator is None:
        _semantic_cache_generator = SemanticCacheKeyGenerator()
    return _semantic_cache_generator


def reset_semantic_cache_generator() -> None:
    """Reset the global semantic cache generator (useful for testing)."""
    global _semantic_cache_generator
    _semantic_cache_generator = None
