#!/usr/bin/env python3
"""
Unit tests for HyDE (Hypothetical Document Embeddings) generator.

Tests the HyDEGenerator which generates hypothetical documents using an LLM
and embeds them for improved semantic search.
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from core.hyde_generator import (
    HyDEConfig,
    HyDEGenerator,
    HyDEResult,
    get_hyde_generator,
    reset_hyde_generator,
)


class TestHyDEConfig(unittest.TestCase):
    """Test HyDE configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HyDEConfig()

        self.assertTrue(config.enabled)
        # Default to free Grok model via OpenRouter
        self.assertEqual(config.model, "x-ai/grok-4.1-fast:free")
        self.assertEqual(config.api_provider, "openrouter")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(config.max_tokens, 150)
        self.assertEqual(config.temperature, 0.7)
        self.assertTrue(config.cache_enabled)
        self.assertEqual(config.cache_ttl_seconds, 3600)
        self.assertTrue(config.fallback_to_mq)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HyDEConfig(
            enabled=False,
            model="gpt-3.5-turbo",
            max_tokens=100,
            temperature=0.5,
            cache_enabled=False,
        )

        self.assertFalse(config.enabled)
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.max_tokens, 100)
        self.assertEqual(config.temperature, 0.5)
        self.assertFalse(config.cache_enabled)


class TestHyDEResult(unittest.TestCase):
    """Test HyDE result dataclass."""

    def test_result_creation(self):
        """Test creating a HyDEResult."""
        result = HyDEResult(
            hypothetical_doc="Test document",
            embedding=[0.1, 0.2, 0.3],
            cache_hit=False,
            generation_time_ms=100.5,
        )

        self.assertEqual(result.hypothetical_doc, "Test document")
        self.assertEqual(result.embedding, [0.1, 0.2, 0.3])
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.generation_time_ms, 100.5)
        self.assertIsNone(result.error)

    def test_result_with_error(self):
        """Test creating a HyDEResult with error."""
        result = HyDEResult(
            hypothetical_doc="",
            embedding=[],
            cache_hit=False,
            generation_time_ms=50.0,
            error="API error",
        )

        self.assertEqual(result.error, "API error")
        self.assertEqual(result.hypothetical_doc, "")


class TestHyDEGenerator(unittest.TestCase):
    """Test HyDE generator functionality."""

    def setUp(self):
        """Reset global state before each test."""
        reset_hyde_generator()

    def tearDown(self):
        """Clean up after tests."""
        reset_hyde_generator()

    def test_generator_initialization(self):
        """Test generator initialization with default config."""
        generator = HyDEGenerator()

        self.assertIsNotNone(generator.config)
        self.assertTrue(generator.config.enabled)
        # Default to free Grok model via OpenRouter
        self.assertEqual(generator.config.model, "x-ai/grok-4.1-fast:free")
        self.assertEqual(generator.config.api_provider, "openrouter")

    def test_generator_initialization_from_env(self):
        """Test generator initialization from environment variables."""
        with patch.dict(
            os.environ,
            {
                "HYDE_ENABLED": "false",
                "HYDE_API_PROVIDER": "openai",
                "HYDE_MODEL": "gpt-3.5-turbo",
                "HYDE_MAX_TOKENS": "200",
                "HYDE_TEMPERATURE": "0.5",
            },
        ):
            reset_hyde_generator()
            generator = HyDEGenerator()

            self.assertFalse(generator.config.enabled)
            self.assertEqual(generator.config.api_provider, "openai")
            self.assertEqual(generator.config.model, "gpt-3.5-turbo")
            self.assertEqual(generator.config.max_tokens, 200)
            self.assertEqual(generator.config.temperature, 0.5)

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        generator = HyDEGenerator()

        key1 = generator._get_cache_key("How do I configure Neo4j?")
        key2 = generator._get_cache_key("How do I configure Neo4j?")
        key3 = generator._get_cache_key("How do I configure Redis?")

        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_cache_key_case_insensitive(self):
        """Test that cache keys are case-insensitive."""
        generator = HyDEGenerator()

        key1 = generator._get_cache_key("How do I configure Neo4j?")
        key2 = generator._get_cache_key("how do i configure neo4j?")
        key3 = generator._get_cache_key("HOW DO I CONFIGURE NEO4J?")

        self.assertEqual(key1, key2)
        self.assertEqual(key2, key3)

    def test_cache_storage_and_retrieval(self):
        """Test cache storage and retrieval."""
        generator = HyDEGenerator(HyDEConfig(cache_enabled=True))

        cache_key = "test_key"
        test_doc = "Test hypothetical document"
        test_embedding = [0.1, 0.2, 0.3]

        generator._store_in_cache(cache_key, test_doc, test_embedding)
        cached = generator._get_from_cache(cache_key)

        self.assertIsNotNone(cached)
        self.assertEqual(cached["doc"], test_doc)
        self.assertEqual(cached["embedding"], test_embedding)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        generator = HyDEGenerator(HyDEConfig(cache_enabled=True))

        cached = generator._get_from_cache("nonexistent_key")

        self.assertIsNone(cached)

    def test_prompt_template(self):
        """Test that prompt template is properly formatted."""
        generator = HyDEGenerator()

        prompt = generator.PROMPT_TEMPLATE.format(query="How do I configure Neo4j?")

        self.assertIn("How do I configure Neo4j?", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("Answer:", prompt)

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        generator = HyDEGenerator()

        metrics = generator.get_metrics()

        self.assertEqual(metrics["total_requests"], 0)
        self.assertEqual(metrics["cache_hits"], 0)
        self.assertEqual(metrics["cache_misses"], 0)
        self.assertEqual(metrics["llm_calls"], 0)
        self.assertEqual(metrics["llm_errors"], 0)

    def test_metrics_reset(self):
        """Test that metrics can be reset."""
        generator = HyDEGenerator()

        # Simulate some activity
        generator._metrics["total_requests"] = 10
        generator._metrics["cache_hits"] = 5

        generator.reset_metrics()

        metrics = generator.get_metrics()
        self.assertEqual(metrics["total_requests"], 0)
        self.assertEqual(metrics["cache_hits"], 0)

    def test_cache_clear(self):
        """Test that cache can be cleared."""
        generator = HyDEGenerator(HyDEConfig(cache_enabled=True))

        generator._store_in_cache("key1", "doc1", [0.1])
        generator._store_in_cache("key2", "doc2", [0.2])

        self.assertEqual(len(generator._cache), 2)

        generator.clear_cache()

        self.assertEqual(len(generator._cache), 0)


class TestHyDEGeneratorAsync(unittest.IsolatedAsyncioTestCase):
    """Async tests for HyDE generator."""

    def setUp(self):
        """Reset global state before each test."""
        reset_hyde_generator()

    def tearDown(self):
        """Clean up after tests."""
        reset_hyde_generator()

    async def test_generate_hypothetical_doc(self):
        """Test generating hypothetical document with mocked LLM."""
        generator = HyDEGenerator()

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "To configure Neo4j, set the NEO4J_URI environment variable..."

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        generator._client = mock_client

        doc = await generator.generate_hypothetical_doc("How do I configure Neo4j?")

        self.assertIn("configure", doc.lower())
        self.assertEqual(generator._metrics["llm_calls"], 1)

    async def test_generate_hyde_embedding(self):
        """Test generating HyDE embedding with mocked services."""
        generator = HyDEGenerator()

        # Mock the OpenAI client
        mock_llm_response = MagicMock()
        mock_llm_response.choices = [MagicMock()]
        mock_llm_response.choices[0].message.content = "To configure Neo4j..."

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_llm_response)
        generator._client = mock_client

        # Mock the embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        generator._embedding_service = mock_embedding_service

        result = await generator.generate_hyde_embedding("How do I configure Neo4j?")

        self.assertIsInstance(result, HyDEResult)
        self.assertEqual(result.hypothetical_doc, "To configure Neo4j...")
        self.assertEqual(result.embedding, [0.1, 0.2, 0.3])
        self.assertFalse(result.cache_hit)
        self.assertIsNone(result.error)

    async def test_generate_hyde_embedding_cache_hit(self):
        """Test that cache hits return cached results."""
        generator = HyDEGenerator(HyDEConfig(cache_enabled=True))

        # Pre-populate cache
        query = "How do I configure Neo4j?"
        cache_key = generator._get_cache_key(query)
        generator._store_in_cache(cache_key, "Cached doc", [0.4, 0.5, 0.6])

        result = await generator.generate_hyde_embedding(query)

        self.assertIsInstance(result, HyDEResult)
        self.assertEqual(result.hypothetical_doc, "Cached doc")
        self.assertEqual(result.embedding, [0.4, 0.5, 0.6])
        self.assertTrue(result.cache_hit)
        self.assertEqual(generator._metrics["cache_hits"], 1)

    async def test_generate_hyde_embedding_llm_error(self):
        """Test handling of LLM errors."""
        generator = HyDEGenerator()

        # Mock the OpenAI client to raise an error
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        generator._client = mock_client

        result = await generator.generate_hyde_embedding("How do I configure Neo4j?")

        self.assertIsInstance(result, HyDEResult)
        self.assertEqual(result.hypothetical_doc, "")
        self.assertEqual(result.embedding, [])
        self.assertFalse(result.cache_hit)
        self.assertIsNotNone(result.error)
        self.assertIn("API error", result.error)


class TestGlobalInstance(unittest.TestCase):
    """Test global instance management."""

    def setUp(self):
        reset_hyde_generator()

    def tearDown(self):
        reset_hyde_generator()

    def test_get_hyde_generator_singleton(self):
        """Test that get_hyde_generator returns singleton."""
        generator1 = get_hyde_generator()
        generator2 = get_hyde_generator()

        self.assertIs(generator1, generator2)

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        generator1 = get_hyde_generator()
        reset_hyde_generator()
        generator2 = get_hyde_generator()

        self.assertIsNot(generator1, generator2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
