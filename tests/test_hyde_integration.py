#!/usr/bin/env python3
"""
Integration tests for HyDE (Hypothetical Document Embeddings) with RetrievalCore.

Tests the complete flow from HyDE generation through search execution.
"""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.hyde_generator import HyDEConfig, HyDEGenerator, HyDEResult, reset_hyde_generator
from src.core.query_dispatcher import QueryDispatcher, SearchMode
from src.interfaces.backend_interface import SearchOptions, BackendSearchInterface, BackendHealthStatus
from src.interfaces.memory_result import MemoryResult, ResultSource, ContentType, SearchResultResponse


class MockVectorBackend(BackendSearchInterface):
    """Mock vector backend for integration testing."""

    def __init__(self, results=None):
        self._results = results or []

    @property
    def backend_name(self) -> str:
        return "vector"

    async def search(self, query: str, options: SearchOptions):
        return self._results

    async def search_by_embedding(self, embedding, options: SearchOptions):
        # Return results based on embedding
        return self._results

    async def health_check(self) -> BackendHealthStatus:
        return BackendHealthStatus(status="healthy", response_time_ms=5.0)


class TestHyDERetrievalCoreIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for HyDE with RetrievalCore."""

    def setUp(self):
        """Reset global state before each test."""
        reset_hyde_generator()

    def tearDown(self):
        """Clean up after tests."""
        reset_hyde_generator()

    async def test_hyde_search_flow_with_mocked_llm(self):
        """Test complete HyDE search flow with mocked LLM."""
        # Setup mock results
        mock_results = [
            MemoryResult(
                id="doc_1",
                text="Neo4j configuration requires setting NEO4J_URI",
                score=0.92,
                source=ResultSource.VECTOR,
                type=ContentType.DOCUMENTATION
            ),
            MemoryResult(
                id="doc_2",
                text="Configure Neo4j by setting environment variables",
                score=0.85,
                source=ResultSource.VECTOR,
                type=ContentType.DOCUMENTATION
            )
        ]

        # Create dispatcher with mock vector backend
        dispatcher = QueryDispatcher()
        vector_backend = MockVectorBackend(mock_results)
        dispatcher.register_backend("vector", vector_backend)

        # Create HyDE generator with mocked LLM
        config = HyDEConfig(enabled=True, cache_enabled=False)
        generator = HyDEGenerator(config)

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "To configure Neo4j, set the NEO4J_URI environment variable "
            "to your database connection string."
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        generator._client = mock_client

        # Mock the embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(
            return_value=[0.1, 0.2, 0.3] * 512
        )
        generator._embedding_service = mock_embedding_service

        # Generate HyDE embedding
        hyde_result = await generator.generate_hyde_embedding("How do I configure Neo4j?")

        # Verify HyDE generation
        self.assertIsInstance(hyde_result, HyDEResult)
        self.assertFalse(hyde_result.cache_hit)
        self.assertIsNone(hyde_result.error)
        self.assertIn("configure", hyde_result.hypothetical_doc.lower())
        self.assertEqual(len(hyde_result.embedding), 1536)

        # Search using the HyDE embedding
        response = await dispatcher.search_by_embedding(
            embedding=hyde_result.embedding,
            options=SearchOptions(limit=5),
            search_mode=SearchMode.VECTOR
        )

        # Verify search results
        self.assertTrue(response.success)
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.search_mode_used, "hyde")
        self.assertIn("vector", response.backends_used)

    async def test_hyde_cache_improves_paraphrase_consistency(self):
        """Test that HyDE cache helps with paraphrased queries."""
        config = HyDEConfig(enabled=True, cache_enabled=True)
        generator = HyDEGenerator(config)

        # Pre-populate cache with a hypothetical doc
        query = "How do I configure Neo4j?"
        cache_key = generator._get_cache_key(query)
        cached_doc = "To configure Neo4j, set the NEO4J_URI environment variable."
        cached_embedding = [0.1, 0.2, 0.3] * 512
        generator._store_in_cache(cache_key, cached_doc, cached_embedding)

        # Request with exact same query - should hit cache
        result1 = await generator.generate_hyde_embedding(query)
        self.assertTrue(result1.cache_hit)
        self.assertEqual(result1.hypothetical_doc, cached_doc)
        self.assertEqual(result1.embedding, cached_embedding)

        # Request with different case - should also hit cache (case insensitive)
        result2 = await generator.generate_hyde_embedding("HOW DO I CONFIGURE NEO4J?")
        self.assertTrue(result2.cache_hit)
        self.assertEqual(result2.hypothetical_doc, cached_doc)

    async def test_hyde_fallback_on_llm_error(self):
        """Test that HyDE gracefully handles LLM errors."""
        config = HyDEConfig(enabled=True, cache_enabled=False)
        generator = HyDEGenerator(config)

        # Mock the OpenAI client to raise an error
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API rate limit exceeded")
        )
        generator._client = mock_client

        # Generate HyDE embedding - should return error result
        result = await generator.generate_hyde_embedding("Test query")

        self.assertIsInstance(result, HyDEResult)
        self.assertEqual(result.hypothetical_doc, "")
        self.assertEqual(result.embedding, [])
        self.assertFalse(result.cache_hit)
        self.assertIsNotNone(result.error)
        self.assertIn("API rate limit", result.error)

    async def test_hyde_metrics_tracking(self):
        """Test that HyDE properly tracks metrics."""
        config = HyDEConfig(enabled=True, cache_enabled=True)
        generator = HyDEGenerator(config)

        # Pre-populate cache
        cache_key = generator._get_cache_key("cached query")
        generator._store_in_cache(cache_key, "cached doc", [0.1, 0.2])

        # Mock for non-cached query
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated doc"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        generator._client = mock_client

        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.3, 0.4])
        generator._embedding_service = mock_embedding_service

        # Request 1: Cache hit
        await generator.generate_hyde_embedding("cached query")

        # Request 2: Cache miss, LLM call
        await generator.generate_hyde_embedding("new query")

        # Check metrics
        metrics = generator.get_metrics()
        self.assertEqual(metrics["total_requests"], 2)
        self.assertEqual(metrics["cache_hits"], 1)
        self.assertEqual(metrics["cache_misses"], 1)
        self.assertEqual(metrics["llm_calls"], 1)
        self.assertEqual(metrics["cache_hit_rate"], 0.5)

    async def test_hyde_disabled_config(self):
        """Test that HyDE can be disabled via config."""
        config = HyDEConfig(enabled=False)
        generator = HyDEGenerator(config)

        self.assertFalse(generator.config.enabled)

        # When disabled, the calling code should skip HyDE
        # The generator itself doesn't block calls, but RetrievalCore checks config.enabled


class TestHyDEEnvironmentConfig(unittest.TestCase):
    """Test HyDE configuration from environment variables."""

    def setUp(self):
        reset_hyde_generator()

    def tearDown(self):
        reset_hyde_generator()

    def test_config_from_environment(self):
        """Test that config reads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "HYDE_ENABLED": "false",
                "HYDE_MODEL": "gpt-3.5-turbo",
                "HYDE_MAX_TOKENS": "100",
                "HYDE_TEMPERATURE": "0.5",
                "HYDE_CACHE_ENABLED": "false",
            },
        ):
            generator = HyDEGenerator()

            self.assertFalse(generator.config.enabled)
            self.assertEqual(generator.config.model, "gpt-3.5-turbo")
            self.assertEqual(generator.config.max_tokens, 100)
            self.assertEqual(generator.config.temperature, 0.5)
            self.assertFalse(generator.config.cache_enabled)


class TestHyDEWithRetrievalCoreFlow(unittest.IsolatedAsyncioTestCase):
    """Test the integration pattern used in RetrievalCore."""

    def setUp(self):
        reset_hyde_generator()

    def tearDown(self):
        reset_hyde_generator()

    async def test_retrieval_core_hyde_integration_pattern(self):
        """Test the pattern RetrievalCore uses for HyDE integration."""
        # This tests the pattern used in retrieval_core.py:
        # 1. Check if HyDE is enabled and available
        # 2. Generate HyDE embedding
        # 3. If successful, search by embedding
        # 4. If failed, fall back to MQE/standard search

        mock_results = [
            MemoryResult(
                id="result_1",
                text="Test result",
                score=0.9,
                source=ResultSource.VECTOR,
                type=ContentType.GENERAL
            )
        ]

        # Setup dispatcher
        dispatcher = QueryDispatcher()
        vector_backend = MockVectorBackend(mock_results)
        dispatcher.register_backend("vector", vector_backend)

        # Setup HyDE generator
        config = HyDEConfig(enabled=True, cache_enabled=False)
        generator = HyDEGenerator(config)

        # Mock LLM and embedding service
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hypothetical answer document"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        generator._client = mock_client

        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
        generator._embedding_service = mock_embedding_service

        # Simulate RetrievalCore logic
        hyde_used = False
        search_response = None

        if generator.config.enabled:
            hyde_result = await generator.generate_hyde_embedding("test query")

            if hyde_result.embedding and not hyde_result.error:
                search_response = await dispatcher.search_by_embedding(
                    embedding=hyde_result.embedding,
                    options=SearchOptions(limit=10),
                    search_mode=SearchMode.VECTOR
                )
                hyde_used = True

        # Verify the integration worked
        self.assertTrue(hyde_used)
        self.assertIsNotNone(search_response)
        self.assertTrue(search_response.success)
        self.assertEqual(len(search_response.results), 1)
        self.assertEqual(search_response.search_mode_used, "hyde")


if __name__ == "__main__":
    unittest.main(verbosity=2)
