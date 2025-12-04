#!/usr/bin/env python3
"""
HyDE (Hypothetical Document Embeddings) Generator for improved retrieval.

This module generates hypothetical answer documents using an LLM, then embeds
those documents for searching in document embedding space. This improves
paraphrase robustness by ensuring that semantically equivalent queries
generate similar hypothetical documents.

Reference: https://arxiv.org/abs/2212.10496
"""

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HyDEConfig:
    """Configuration for HyDE generator."""

    enabled: bool = True
    # Default to Mistral Small via OpenRouter (excellent for RAG/document generation)
    model: str = "mistralai/mistral-small-3.1-24b-instruct-2503"
    # API provider: "openrouter" (default, free) or "openai"
    api_provider: str = "openrouter"
    # Base URL for the API (OpenRouter by default)
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 150
    temperature: float = 0.7
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    fallback_to_mq: bool = True  # Fall back to template MQE on failure
    system_prompt: str = "You are a technical documentation assistant for Veris Memory, a context storage system."


@dataclass
class HyDEResult:
    """Result of HyDE generation."""

    hypothetical_doc: str
    embedding: List[float]
    cache_hit: bool
    generation_time_ms: float
    error: Optional[str] = None


class HyDEGenerator:
    """
    Generate hypothetical documents for improved retrieval.

    Instead of searching with the raw query embedding, HyDE:
    1. Uses an LLM to generate a hypothetical answer document
    2. Embeds that hypothetical document
    3. Searches using the hypothetical doc embedding

    This works because the hypothetical document is in the same
    semantic space as the actual documents, leading to better
    retrieval for paraphrased queries.
    """

    PROMPT_TEMPLATE = """Write a short, informative paragraph that directly answers this question.
Be specific and technical. Focus on configuration, setup, or implementation details.
Do not include any preamble like "Here's the answer" - just provide the answer directly.

Question: {query}

Answer:"""

    def __init__(self, config: Optional[HyDEConfig] = None):
        """
        Initialize the HyDE generator.

        Args:
            config: Configuration options. If None, uses defaults from environment.
        """
        if config is None:
            # Determine API provider - OpenRouter is default (free Grok model)
            api_provider = os.getenv("HYDE_API_PROVIDER", "openrouter").lower()

            # Set defaults based on provider
            if api_provider == "openrouter":
                default_model = "mistralai/mistral-small-3.1-24b-instruct-2503"
                default_base_url = "https://openrouter.ai/api/v1"
            else:  # openai
                default_model = "gpt-4o-mini"
                default_base_url = "https://api.openai.com/v1"

            config = HyDEConfig(
                enabled=os.getenv("HYDE_ENABLED", "true").lower() == "true",
                model=os.getenv("HYDE_MODEL", default_model),
                api_provider=api_provider,
                base_url=os.getenv("HYDE_BASE_URL", default_base_url),
                max_tokens=int(os.getenv("HYDE_MAX_TOKENS", "150")),
                temperature=float(os.getenv("HYDE_TEMPERATURE", "0.7")),
                cache_enabled=os.getenv("HYDE_CACHE_ENABLED", "true").lower() == "true",
            )

        self.config = config
        self._client = None
        # NOTE: In-memory cache - does not persist across restarts and won't scale
        # horizontally across multiple instances. For production at scale, consider
        # replacing with Redis cache. The current implementation is suitable for
        # single-instance deployments and development/testing.
        self._cache: Dict[str, Dict[str, Any]] = {}  # Simple in-memory cache
        self._cache_timestamps: Dict[str, float] = {}  # Track cache entry times
        self._embedding_service = None
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "llm_errors": 0,
            "average_generation_time_ms": 0.0,
        }

        logger.info(
            f"HyDEGenerator initialized: enabled={config.enabled}, "
            f"model={config.model}, cache={config.cache_enabled}"
        )

    async def _get_openai_client(self):
        """Get or create the OpenAI-compatible client (supports OpenRouter and OpenAI)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                # Get API key based on provider
                if self.config.api_provider == "openrouter":
                    api_key = os.getenv("OPENROUTER_API_KEY")
                    if not api_key:
                        raise ValueError(
                            "OPENROUTER_API_KEY environment variable not set. "
                            "Get a free key at https://openrouter.ai/keys"
                        )
                    # OpenRouter uses OpenAI-compatible API
                    self._client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=self.config.base_url,
                        default_headers={
                            "HTTP-Referer": "https://github.com/credentum/veris-memory",
                            "X-Title": "Veris Memory HyDE",
                        }
                    )
                else:  # openai
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY environment variable not set")
                    self._client = AsyncOpenAI(
                        api_key=api_key,
                        base_url=self.config.base_url if self.config.base_url != "https://openrouter.ai/api/v1" else None
                    )

                logger.info(
                    f"HyDE LLM client initialized: provider={self.config.api_provider}, "
                    f"model={self.config.model}"
                )

            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        return self._client

    async def _get_embedding_service(self):
        """Get or create the embedding service."""
        if self._embedding_service is None:
            try:
                from ..embedding.service import EmbeddingService

                self._embedding_service = EmbeddingService()
                await self._embedding_service.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize embedding service: {e}")
                raise
        return self._embedding_service

    async def generate_hypothetical_doc(self, query: str) -> str:
        """
        Generate a hypothetical answer document using LLM.

        Args:
            query: The user's search query

        Returns:
            A hypothetical document that answers the query
        """
        client = await self._get_openai_client()
        prompt = self.PROMPT_TEMPLATE.format(query=query)

        self._metrics["llm_calls"] += 1

        try:
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            content = response.choices[0].message.content
            if content:
                return content.strip()
            return ""

        except Exception as e:
            self._metrics["llm_errors"] += 1
            logger.error(f"LLM generation failed: {e}")
            raise

    async def generate_hyde_embedding(self, query: str) -> HyDEResult:
        """
        Generate embedding from hypothetical document.

        Args:
            query: The user's search query

        Returns:
            HyDEResult with hypothetical doc, embedding, and metadata
        """
        start_time = time.time()
        self._metrics["total_requests"] += 1

        # Check cache first
        cache_key = self._get_cache_key(query)
        if self.config.cache_enabled:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._metrics["cache_hits"] += 1
                return HyDEResult(
                    hypothetical_doc=cached["doc"],
                    embedding=cached["embedding"],
                    cache_hit=True,
                    generation_time_ms=(time.time() - start_time) * 1000,
                )

        self._metrics["cache_misses"] += 1

        try:
            # Generate hypothetical document
            hyde_doc = await self.generate_hypothetical_doc(query)

            if not hyde_doc:
                return HyDEResult(
                    hypothetical_doc="",
                    embedding=[],
                    cache_hit=False,
                    generation_time_ms=(time.time() - start_time) * 1000,
                    error="Empty response from LLM",
                )

            # Get embedding of hypothetical doc
            embedding_service = await self._get_embedding_service()
            embedding = await embedding_service.generate_embedding(hyde_doc)

            # Cache result
            if self.config.cache_enabled:
                self._store_in_cache(cache_key, hyde_doc, embedding)

            generation_time_ms = (time.time() - start_time) * 1000
            self._update_average_time(generation_time_ms)

            logger.debug(
                f"HyDE generated: query='{query[:50]}...', "
                f"doc_length={len(hyde_doc)}, time={generation_time_ms:.2f}ms"
            )

            return HyDEResult(
                hypothetical_doc=hyde_doc,
                embedding=embedding,
                cache_hit=False,
                generation_time_ms=generation_time_ms,
            )

        except Exception as e:
            generation_time_ms = (time.time() - start_time) * 1000
            logger.error(f"HyDE generation failed: {e}")
            return HyDEResult(
                hypothetical_doc="",
                embedding=[],
                cache_hit=False,
                generation_time_ms=generation_time_ms,
                error=str(e),
            )

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get entry from cache if not expired."""
        if cache_key not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > self.config.cache_ttl_seconds:
            # Expired, remove from cache
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _store_in_cache(self, cache_key: str, doc: str, embedding: List[float]) -> None:
        """Store entry in cache."""
        self._cache[cache_key] = {"doc": doc, "embedding": embedding}
        self._cache_timestamps[cache_key] = time.time()

        # Simple cache size limit (keep last 1000 entries)
        if len(self._cache) > 1000:
            oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]

    def _update_average_time(self, new_time_ms: float) -> None:
        """Update running average generation time."""
        total = self._metrics["total_requests"]
        if total > 0:
            current_avg = self._metrics["average_generation_time_ms"]
            self._metrics["average_generation_time_ms"] = (
                current_avg * (total - 1) + new_time_ms
            ) / total

    def get_metrics(self) -> Dict[str, Any]:
        """Get HyDE metrics."""
        total = self._metrics["total_requests"]
        return {
            **self._metrics,
            "cache_hit_rate": self._metrics["cache_hits"] / total if total > 0 else 0.0,
            "llm_error_rate": self._metrics["llm_errors"] / self._metrics["llm_calls"]
            if self._metrics["llm_calls"] > 0
            else 0.0,
            "cache_size": len(self._cache),
            "config": {
                "enabled": self.config.enabled,
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "cache_enabled": self.config.cache_enabled,
            },
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "llm_errors": 0,
            "average_generation_time_ms": 0.0,
        }

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_timestamps.clear()


# Global instance
_hyde_generator: Optional[HyDEGenerator] = None


def get_hyde_generator() -> HyDEGenerator:
    """
    Get or create the global HyDE generator.

    Returns:
        Global HyDEGenerator instance
    """
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator()
    return _hyde_generator


def reset_hyde_generator() -> None:
    """Reset the global HyDE generator (useful for testing)."""
    global _hyde_generator
    _hyde_generator = None
