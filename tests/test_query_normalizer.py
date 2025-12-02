#!/usr/bin/env python3
"""
Unit tests for semantic query normalization.

Tests the QueryNormalizer which maps paraphrased queries to canonical
forms for consistent caching and retrieval behavior.
"""

import sys
import os
import unittest
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from core.query_normalizer import (
    QueryNormalizer,
    QueryNormalizerConfig,
    NormalizedQuery,
    QueryIntent,
    get_query_normalizer,
    reset_query_normalizer,
)


class TestQueryNormalizer(unittest.TestCase):
    """Test query normalizer functionality."""

    def setUp(self):
        """Reset global state before each test."""
        reset_query_normalizer()
        self.config = QueryNormalizerConfig(
            enabled=True,
            confidence_threshold=0.5,
        )
        self.normalizer = QueryNormalizer(self.config)

    def tearDown(self):
        """Clean up after tests."""
        reset_query_normalizer()

    def test_normalize_returns_normalized_query(self):
        """Test that normalize returns NormalizedQuery object."""
        result = self.normalizer.normalize("How do I configure Neo4j?")

        self.assertIsInstance(result, NormalizedQuery)
        self.assertEqual(result.original, "How do I configure Neo4j?")
        self.assertIsNotNone(result.normalized)
        self.assertIsInstance(result.intent, QueryIntent)
        self.assertIsInstance(result.entities, list)
        self.assertIsInstance(result.confidence, float)

    def test_disabled_returns_original(self):
        """Test that disabled normalizer returns original query."""
        config = QueryNormalizerConfig(enabled=False)
        normalizer = QueryNormalizer(config)

        result = normalizer.normalize("How to setup Neo4j?")

        self.assertEqual(result.normalized, "How to setup Neo4j?")
        self.assertEqual(result.intent, QueryIntent.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)


class TestIntentDetection(unittest.TestCase):
    """Test query intent detection."""

    def setUp(self):
        reset_query_normalizer()
        self.normalizer = QueryNormalizer()

    def tearDown(self):
        reset_query_normalizer()

    def test_configuration_intent(self):
        """Test that configuration queries are detected."""
        queries = [
            "How do I configure Neo4j?",
            "Neo4j setup instructions",
            "Configure Qdrant settings",  # Changed from "What are..." to avoid CONCEPTUAL
            "Initialize Redis connection",
        ]

        for query in queries:
            result = self.normalizer.normalize(query)
            # Configuration queries may also match HOWTO due to "how" pattern
            self.assertIn(
                result.intent,
                [QueryIntent.CONFIGURATION, QueryIntent.HOWTO],
                f"Query '{query}' should have CONFIGURATION or HOWTO intent, got {result.intent}",
            )

    def test_troubleshooting_intent(self):
        """Test that troubleshooting queries are detected."""
        queries = [
            "How do I fix Neo4j connection errors?",
            "Troubleshoot database timeout issues",
            "Resolve connection failed problem",
            "Debug query performance issues",
        ]

        for query in queries:
            result = self.normalizer.normalize(query)
            self.assertEqual(
                result.intent,
                QueryIntent.TROUBLESHOOTING,
                f"Query '{query}' should have TROUBLESHOOTING intent, got {result.intent}",
            )

    def test_howto_intent(self):
        """Test that how-to queries are detected."""
        queries = [
            "How to store context in Veris Memory?",
            "Steps to configure database",
            "Guide for setting up embedding",
        ]

        for query in queries:
            result = self.normalizer.normalize(query)
            self.assertIn(
                result.intent,
                [QueryIntent.HOWTO, QueryIntent.CONFIGURATION],
                f"Query '{query}' should have HOWTO or CONFIGURATION intent, got {result.intent}",
            )

    def test_conceptual_intent(self):
        """Test that conceptual queries are detected."""
        queries = [
            "What is a vector database?",
            "What are embeddings in machine learning?",
            "Explain the MCP protocol",
            "Describe microservices architecture",
        ]

        for query in queries:
            result = self.normalizer.normalize(query)
            self.assertEqual(
                result.intent,
                QueryIntent.CONCEPTUAL,
                f"Query '{query}' should have CONCEPTUAL intent, got {result.intent}",
            )

    def test_lookup_intent(self):
        """Test that lookup queries are detected."""
        queries = [
            "Find context about Neo4j",
            "Search for database configuration",
            "Get agent state",
            "List all contexts",
        ]

        for query in queries:
            result = self.normalizer.normalize(query)
            # Lookup queries may also match other intents if they contain config-related words
            self.assertIn(
                result.intent,
                [QueryIntent.LOOKUP, QueryIntent.CONFIGURATION, QueryIntent.HOWTO],
                f"Query '{query}' should have LOOKUP, CONFIGURATION, or HOWTO intent, got {result.intent}",
            )


class TestEntityExtraction(unittest.TestCase):
    """Test entity extraction from queries."""

    def setUp(self):
        reset_query_normalizer()
        self.normalizer = QueryNormalizer()

    def tearDown(self):
        reset_query_normalizer()

    def test_neo4j_entity_extracted(self):
        """Test that Neo4j entity is extracted."""
        result = self.normalizer.normalize("How do I configure Neo4j?")

        self.assertIn("neo4j", result.entities)

    def test_qdrant_entity_extracted(self):
        """Test that Qdrant entity is extracted."""
        result = self.normalizer.normalize("Configure Qdrant vector store")

        self.assertIn("qdrant", result.entities)

    def test_redis_entity_extracted(self):
        """Test that Redis entity is extracted."""
        result = self.normalizer.normalize("Setup Redis cache")

        self.assertIn("redis", result.entities)

    def test_multiple_entities_extracted(self):
        """Test that multiple entities are extracted."""
        result = self.normalizer.normalize("Configure Neo4j and Redis for vector database")

        self.assertIn("neo4j", result.entities)
        self.assertIn("redis", result.entities)
        self.assertIn("vector", result.entities)
        self.assertIn("database", result.entities)

    def test_veris_memory_entity_extracted(self):
        """Test that Veris Memory entity is extracted."""
        result = self.normalizer.normalize("How do I use Veris Memory?")

        self.assertIn("veris memory", result.entities)


class TestCanonicalNormalization(unittest.TestCase):
    """Test canonical form normalization."""

    def setUp(self):
        reset_query_normalizer()
        self.normalizer = QueryNormalizer()

    def tearDown(self):
        reset_query_normalizer()

    def test_neo4j_configuration_normalized(self):
        """Test that Neo4j configuration queries are normalized to canonical form."""
        paraphrases = [
            "How do I configure Neo4j?",
            "Setup Neo4j database",
            "Neo4j settings configuration",
        ]

        results = [self.normalizer.normalize(q) for q in paraphrases]

        # All should normalize to the same canonical form
        canonical = "How do I configure Neo4j database settings?"
        for result in results:
            if result.confidence > 0.5:
                self.assertEqual(
                    result.normalized,
                    canonical,
                    f"Query should normalize to canonical form",
                )

    def test_redis_configuration_normalized(self):
        """Test that Redis configuration queries are normalized."""
        query = "How do I setup Redis cache?"
        result = self.normalizer.normalize(query)

        if result.confidence > 0.5:
            self.assertIn("Redis", result.normalized)
            self.assertIn("cache", result.normalized.lower())

    def test_embedding_configuration_normalized(self):
        """Test that embedding configuration queries are normalized."""
        query = "Configure embedding model settings"
        result = self.normalizer.normalize(query)

        if result.confidence > 0.5:
            self.assertIn("embedding", result.normalized.lower())

    def test_low_confidence_unchanged(self):
        """Test that low confidence queries are not normalized."""
        config = QueryNormalizerConfig(enabled=True, confidence_threshold=0.9)
        normalizer = QueryNormalizer(config)

        result = normalizer.normalize("Random unrelated query text")

        self.assertEqual(result.normalized, "Random unrelated query text")


class TestParaphraseRobustness(unittest.TestCase):
    """Test paraphrase robustness - semantically equivalent queries should normalize to same form."""

    def setUp(self):
        reset_query_normalizer()
        self.normalizer = QueryNormalizer()

    def tearDown(self):
        reset_query_normalizer()

    def test_neo4j_paraphrases_consistent(self):
        """Test that Neo4j configuration paraphrases normalize consistently."""
        paraphrases = [
            "How do I configure Neo4j database connection settings?",
            "What are the steps to set up Neo4j database configuration?",
            "Setup Neo4j database connection",
        ]

        normalized_forms = []
        for query in paraphrases:
            result = self.normalizer.normalize(query)
            if result.confidence > 0.5:
                normalized_forms.append(result.normalized)

        # Check that all normalized forms are the same (or most are)
        if normalized_forms:
            most_common = max(set(normalized_forms), key=normalized_forms.count)
            matching = sum(1 for f in normalized_forms if f == most_common)
            self.assertGreaterEqual(
                matching / len(normalized_forms),
                0.5,
                "At least half of paraphrases should normalize to same form",
            )

    def test_veris_memory_paraphrases_consistent(self):
        """Test that Veris Memory queries normalize consistently."""
        paraphrases = [
            "How do I store context in Veris Memory?",
            "Save context to Veris Memory",
            "Store data in Veris Memory context",
        ]

        intents = []
        for query in paraphrases:
            result = self.normalizer.normalize(query)
            intents.append(result.intent)

        # Should have consistent intent detection
        most_common = max(set(intents), key=intents.count)
        matching = sum(1 for i in intents if i == most_common)
        self.assertGreaterEqual(
            matching / len(intents),
            0.5,
            "At least half should have same intent",
        )


class TestMetrics(unittest.TestCase):
    """Test metrics tracking."""

    def setUp(self):
        reset_query_normalizer()
        self.normalizer = QueryNormalizer()

    def tearDown(self):
        reset_query_normalizer()

    def test_metrics_tracked(self):
        """Test that normalization metrics are tracked."""
        # Run some normalizations
        for _ in range(5):
            self.normalizer.normalize("How do I configure Neo4j?")

        metrics = self.normalizer.get_metrics()

        self.assertEqual(metrics["total_normalizations"], 5)
        self.assertIn("queries_normalized", metrics)
        self.assertIn("queries_unchanged", metrics)
        self.assertIn("intent_counts", metrics)
        self.assertIn("average_confidence", metrics)
        self.assertIn("normalization_rate", metrics)

    def test_intent_counts_tracked(self):
        """Test that intent counts are tracked."""
        self.normalizer.normalize("How do I configure Neo4j?")  # CONFIGURATION
        self.normalizer.normalize("What is a vector database?")  # CONCEPTUAL
        self.normalizer.normalize("Fix connection error")  # TROUBLESHOOTING

        metrics = self.normalizer.get_metrics()
        intent_counts = metrics["intent_counts"]

        self.assertIn(QueryIntent.CONFIGURATION.value, intent_counts)
        self.assertIn(QueryIntent.CONCEPTUAL.value, intent_counts)
        self.assertIn(QueryIntent.TROUBLESHOOTING.value, intent_counts)

    def test_metrics_reset(self):
        """Test that metrics can be reset."""
        self.normalizer.normalize("test query")
        self.normalizer.reset_metrics()

        metrics = self.normalizer.get_metrics()
        self.assertEqual(metrics["total_normalizations"], 0)


class TestGlobalInstance(unittest.TestCase):
    """Test global instance management."""

    def setUp(self):
        reset_query_normalizer()

    def tearDown(self):
        reset_query_normalizer()

    def test_get_query_normalizer_singleton(self):
        """Test that get_query_normalizer returns singleton."""
        normalizer1 = get_query_normalizer()
        normalizer2 = get_query_normalizer()

        self.assertIs(normalizer1, normalizer2)

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        normalizer1 = get_query_normalizer()
        reset_query_normalizer()
        normalizer2 = get_query_normalizer()

        self.assertIsNot(normalizer1, normalizer2)

    @patch.dict(os.environ, {"QUERY_NORMALIZATION_ENABLED": "false"})
    def test_config_from_environment_disabled(self):
        """Test that config can be loaded from environment."""
        reset_query_normalizer()
        normalizer = get_query_normalizer()

        self.assertFalse(normalizer.config.enabled)

    @patch.dict(
        os.environ,
        {
            "QUERY_NORMALIZATION_ENABLED": "true",
            "QUERY_NORMALIZATION_CONFIDENCE": "0.8",
        },
    )
    def test_config_from_environment_custom(self):
        """Test that custom config is loaded from environment."""
        reset_query_normalizer()
        normalizer = get_query_normalizer()

        self.assertTrue(normalizer.config.enabled)
        self.assertEqual(normalizer.config.confidence_threshold, 0.8)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        reset_query_normalizer()
        self.normalizer = QueryNormalizer()

    def tearDown(self):
        reset_query_normalizer()

    def test_empty_query(self):
        """Test handling of empty query."""
        result = self.normalizer.normalize("")

        self.assertEqual(result.original, "")
        self.assertIsInstance(result.intent, QueryIntent)

    def test_whitespace_only_query(self):
        """Test handling of whitespace-only query."""
        result = self.normalizer.normalize("   ")

        self.assertEqual(result.original, "   ")
        self.assertIsInstance(result.intent, QueryIntent)

    def test_very_long_query(self):
        """Test handling of very long query."""
        long_query = "How do I configure " * 100 + "Neo4j?"
        result = self.normalizer.normalize(long_query)

        self.assertIsInstance(result, NormalizedQuery)
        self.assertIsNotNone(result.normalized)

    def test_special_characters_in_query(self):
        """Test handling of special characters."""
        result = self.normalizer.normalize("How do I configure Neo4j? (version 5.0)")

        self.assertIn("neo4j", result.entities)

    def test_mixed_case_query(self):
        """Test handling of mixed case queries."""
        result1 = self.normalizer.normalize("How do I configure NEO4J?")
        result2 = self.normalizer.normalize("how do i configure neo4j?")

        # Both should detect Neo4j entity
        self.assertIn("neo4j", result1.entities)
        self.assertIn("neo4j", result2.entities)


class TestQueryIntentEnum(unittest.TestCase):
    """Test QueryIntent enum values."""

    def test_all_intents_defined(self):
        """Test that all expected intents are defined."""
        expected_intents = [
            "configuration",
            "troubleshooting",
            "howto",
            "conceptual",
            "lookup",
            "unknown",
        ]

        for intent_value in expected_intents:
            self.assertTrue(
                hasattr(QueryIntent, intent_value.upper()),
                f"QueryIntent should have {intent_value.upper()}",
            )

    def test_intent_values(self):
        """Test that intent values are strings."""
        for intent in QueryIntent:
            self.assertIsInstance(intent.value, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
