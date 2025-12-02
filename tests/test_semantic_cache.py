#!/usr/bin/env python3
"""
Unit tests for semantic cache key generation.

Tests the SemanticCacheKeyGenerator which creates cache keys from
quantized embeddings rather than raw query text, enabling semantically
similar queries to share cache entries.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from core.semantic_cache import (
    SemanticCacheKeyGenerator,
    SemanticCacheConfig,
    CacheKeyResult,
    get_semantic_cache_generator,
    reset_semantic_cache_generator,
)


class TestSemanticCacheKeyGenerator(unittest.TestCase):
    """Test semantic cache key generation functionality."""

    def setUp(self):
        """Reset global state before each test."""
        reset_semantic_cache_generator()
        self.config = SemanticCacheConfig(
            enabled=True,
            quantization_precision=1,
            embedding_prefix_length=32,
        )
        self.generator = SemanticCacheKeyGenerator(self.config)

    def tearDown(self):
        """Clean up after tests."""
        reset_semantic_cache_generator()

    def test_quantize_embedding_basic(self):
        """Test basic embedding quantization."""
        embedding = [0.123456, 0.789012, -0.345678, 0.901234] * 100  # 400 dims

        quantized = self.generator.quantize_embedding(embedding)

        # Should truncate to prefix length
        self.assertEqual(len(quantized), 32)

        # Should round to precision 1 (1 decimal place)
        self.assertEqual(quantized[0], 0.1)
        self.assertEqual(quantized[1], 0.8)
        self.assertEqual(quantized[2], -0.3)
        self.assertEqual(quantized[3], 0.9)

    def test_quantize_embedding_preserves_sign(self):
        """Test that quantization preserves sign of values."""
        embedding = [-0.5, 0.5, -0.05, 0.05] * 10

        quantized = self.generator.quantize_embedding(embedding)

        self.assertEqual(quantized[0], -0.5)
        self.assertEqual(quantized[1], 0.5)
        # -0.05 rounds to -0.1 (away from zero) or 0.0 depending on rounding mode
        # Python uses banker's rounding, so -0.05 rounds to 0.0
        self.assertIn(quantized[2], [-0.1, 0.0])

    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        embedding = [0.1] * 384  # Standard embedding size

        result = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
        )

        self.assertIsInstance(result, CacheKeyResult)
        self.assertTrue(result.is_semantic)
        self.assertTrue(result.cache_key.startswith("semantic:"))
        self.assertGreater(result.generation_time_ms, 0)

    def test_generate_cache_key_deterministic(self):
        """Test that same inputs produce same cache key."""
        embedding = [0.1, 0.2, 0.3] * 128

        result1 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
            context_type="code",
        )

        result2 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
            context_type="code",
        )

        self.assertEqual(result1.cache_key, result2.cache_key)

    def test_similar_embeddings_same_key(self):
        """Test that similar embeddings produce the same cache key (paraphrase robustness)."""
        # Two slightly different embeddings that should quantize to the same values
        # With precision=1, 0.11 and 0.13 both round to 0.1, 0.24 and 0.26 round to 0.2
        embedding1 = [0.11, 0.24, 0.31] * 128  # Rounds to [0.1, 0.2, 0.3]
        embedding2 = [0.13, 0.16, 0.34] * 128  # Rounds to [0.1, 0.2, 0.3]

        result1 = self.generator.generate_cache_key(
            embedding=embedding1,
            limit=10,
            search_mode="hybrid",
        )

        result2 = self.generator.generate_cache_key(
            embedding=embedding2,
            limit=10,
            search_mode="hybrid",
        )

        # These should be the same because they quantize to the same values
        self.assertEqual(result1.cache_key, result2.cache_key)

    def test_different_embeddings_different_key(self):
        """Test that significantly different embeddings produce different cache keys."""
        embedding1 = [0.1] * 384
        embedding2 = [0.9] * 384

        result1 = self.generator.generate_cache_key(
            embedding=embedding1,
            limit=10,
            search_mode="hybrid",
        )

        result2 = self.generator.generate_cache_key(
            embedding=embedding2,
            limit=10,
            search_mode="hybrid",
        )

        self.assertNotEqual(result1.cache_key, result2.cache_key)

    def test_different_params_different_key(self):
        """Test that different request parameters produce different cache keys."""
        embedding = [0.1] * 384

        result1 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
        )

        result2 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=20,  # Different limit
            search_mode="hybrid",
        )

        result3 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="vector",  # Different search mode
        )

        self.assertNotEqual(result1.cache_key, result2.cache_key)
        self.assertNotEqual(result1.cache_key, result3.cache_key)

    def test_disabled_returns_empty_key(self):
        """Test that disabled generator returns empty key with fallback reason."""
        config = SemanticCacheConfig(enabled=False)
        generator = SemanticCacheKeyGenerator(config)

        embedding = [0.1] * 384

        result = generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
        )

        self.assertFalse(result.is_semantic)
        self.assertEqual(result.cache_key, "")
        self.assertEqual(result.fallback_reason, "semantic_cache_disabled")

    def test_text_fallback_key(self):
        """Test text-based fallback key generation."""
        key = self.generator.generate_text_fallback_key(
            query="How do I configure Neo4j?",
            limit=10,
            search_mode="hybrid",
        )

        self.assertTrue(key.startswith("text:"))
        self.assertEqual(len(key), 21)  # "text:" + 16 chars

    def test_text_fallback_key_deterministic(self):
        """Test that same query produces same fallback key."""
        key1 = self.generator.generate_text_fallback_key(
            query="How do I configure Neo4j?",
            limit=10,
            search_mode="hybrid",
        )

        key2 = self.generator.generate_text_fallback_key(
            query="How do I configure Neo4j?",
            limit=10,
            search_mode="hybrid",
        )

        self.assertEqual(key1, key2)

    def test_text_fallback_different_queries_different_keys(self):
        """Test that different queries produce different fallback keys."""
        key1 = self.generator.generate_text_fallback_key(
            query="How do I configure Neo4j?",
            limit=10,
            search_mode="hybrid",
        )

        key2 = self.generator.generate_text_fallback_key(
            query="What are the steps to set up Neo4j?",  # Paraphrase
            limit=10,
            search_mode="hybrid",
        )

        # Text-based keys WILL be different for paraphrases (that's the problem we're solving)
        self.assertNotEqual(key1, key2)

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked."""
        embedding = [0.1] * 384

        # Generate a few keys
        for _ in range(5):
            self.generator.generate_cache_key(
                embedding=embedding,
                limit=10,
                search_mode="hybrid",
            )

        metrics = self.generator.get_metrics()

        self.assertEqual(metrics["total_generations"], 5)
        self.assertEqual(metrics["semantic_keys_generated"], 5)
        self.assertEqual(metrics["fallback_keys_generated"], 0)
        self.assertGreater(metrics["average_generation_time_ms"], 0)
        self.assertEqual(metrics["semantic_key_rate"], 1.0)

    def test_metrics_reset(self):
        """Test that metrics can be reset."""
        embedding = [0.1] * 384

        self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
        )

        self.generator.reset_metrics()
        metrics = self.generator.get_metrics()

        self.assertEqual(metrics["total_generations"], 0)
        self.assertEqual(metrics["semantic_keys_generated"], 0)

    def test_additional_params_affect_key(self):
        """Test that additional parameters affect the cache key."""
        embedding = [0.1] * 384

        result1 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
            additional_params={"exclude_sources": ["source1"]},
        )

        result2 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
            additional_params={"exclude_sources": ["source2"]},
        )

        self.assertNotEqual(result1.cache_key, result2.cache_key)

    def test_context_type_affects_key(self):
        """Test that context_type affects the cache key."""
        embedding = [0.1] * 384

        result1 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
            context_type="code",
        )

        result2 = self.generator.generate_cache_key(
            embedding=embedding,
            limit=10,
            search_mode="hybrid",
            context_type="documentation",
        )

        self.assertNotEqual(result1.cache_key, result2.cache_key)


class TestGlobalInstance(unittest.TestCase):
    """Test global instance management."""

    def setUp(self):
        reset_semantic_cache_generator()

    def tearDown(self):
        reset_semantic_cache_generator()

    def test_get_semantic_cache_generator_singleton(self):
        """Test that get_semantic_cache_generator returns singleton."""
        generator1 = get_semantic_cache_generator()
        generator2 = get_semantic_cache_generator()

        self.assertIs(generator1, generator2)

    def test_reset_clears_singleton(self):
        """Test that reset clears the singleton."""
        generator1 = get_semantic_cache_generator()
        reset_semantic_cache_generator()
        generator2 = get_semantic_cache_generator()

        self.assertIsNot(generator1, generator2)

    @patch.dict(os.environ, {"SEMANTIC_CACHE_ENABLED": "false"})
    def test_config_from_environment_disabled(self):
        """Test that config can be loaded from environment variables."""
        reset_semantic_cache_generator()
        generator = get_semantic_cache_generator()

        self.assertFalse(generator.config.enabled)

    @patch.dict(
        os.environ,
        {
            "SEMANTIC_CACHE_ENABLED": "true",
            "SEMANTIC_CACHE_PRECISION": "2",
            "SEMANTIC_CACHE_PREFIX_LENGTH": "64",
        },
    )
    def test_config_from_environment_custom(self):
        """Test that custom config values are loaded from environment."""
        reset_semantic_cache_generator()
        generator = get_semantic_cache_generator()

        self.assertTrue(generator.config.enabled)
        self.assertEqual(generator.config.quantization_precision, 2)
        self.assertEqual(generator.config.embedding_prefix_length, 64)


class TestQuantizationPrecision(unittest.TestCase):
    """Test different quantization precision levels."""

    def test_precision_0(self):
        """Test quantization with precision 0 (integer rounding)."""
        config = SemanticCacheConfig(
            enabled=True,
            quantization_precision=0,
            embedding_prefix_length=4,
        )
        generator = SemanticCacheKeyGenerator(config)

        embedding = [0.4, 0.6, 1.4, 1.6]
        quantized = generator.quantize_embedding(embedding)

        self.assertEqual(quantized, [0.0, 1.0, 1.0, 2.0])

    def test_precision_2(self):
        """Test quantization with precision 2 (two decimal places)."""
        config = SemanticCacheConfig(
            enabled=True,
            quantization_precision=2,
            embedding_prefix_length=4,
        )
        generator = SemanticCacheKeyGenerator(config)

        embedding = [0.123, 0.456, 0.789, 0.999]
        quantized = generator.quantize_embedding(embedding)

        self.assertEqual(quantized, [0.12, 0.46, 0.79, 1.0])

    def test_higher_precision_less_collision(self):
        """Test that higher precision reduces cache key collisions."""
        # These embeddings should collide at precision 1 but not at precision 2
        # At precision 1: 0.111 -> 0.1, 0.114 -> 0.1 (same)
        # At precision 2: 0.111 -> 0.11, 0.114 -> 0.11 (same, still collides)
        # Use values that actually differ more:
        # At precision 1: 0.11 -> 0.1, 0.11 -> 0.1 (same)
        # At precision 2: 0.111 -> 0.11, 0.119 -> 0.12 (different)
        embedding1 = [0.111, 0.211, 0.311, 0.411] * 100
        embedding2 = [0.119, 0.219, 0.319, 0.419] * 100

        # Low precision - should collide (both round to 0.1, 0.2, 0.3, 0.4)
        config_low = SemanticCacheConfig(
            enabled=True,
            quantization_precision=1,
            embedding_prefix_length=32,
        )
        generator_low = SemanticCacheKeyGenerator(config_low)

        key1_low = generator_low.generate_cache_key(
            embedding=embedding1, limit=10, search_mode="hybrid"
        )
        key2_low = generator_low.generate_cache_key(
            embedding=embedding2, limit=10, search_mode="hybrid"
        )

        # High precision - should not collide (0.11 vs 0.12, 0.21 vs 0.22, etc)
        config_high = SemanticCacheConfig(
            enabled=True,
            quantization_precision=2,
            embedding_prefix_length=32,
        )
        generator_high = SemanticCacheKeyGenerator(config_high)

        key1_high = generator_high.generate_cache_key(
            embedding=embedding1, limit=10, search_mode="hybrid"
        )
        key2_high = generator_high.generate_cache_key(
            embedding=embedding2, limit=10, search_mode="hybrid"
        )

        # Low precision keys should be same (collision - good for paraphrases)
        self.assertEqual(key1_low.cache_key, key2_low.cache_key)

        # High precision keys should be different (no collision)
        self.assertNotEqual(key1_high.cache_key, key2_high.cache_key)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.config = SemanticCacheConfig(
            enabled=True,
            quantization_precision=1,
            embedding_prefix_length=32,
        )
        self.generator = SemanticCacheKeyGenerator(self.config)

    def test_empty_embedding(self):
        """Test handling of empty embedding."""
        result = self.generator.generate_cache_key(
            embedding=[],
            limit=10,
            search_mode="hybrid",
        )

        # Should still work but produce a key based on empty prefix
        self.assertTrue(result.is_semantic)

    def test_short_embedding(self):
        """Test handling of embedding shorter than prefix length."""
        result = self.generator.generate_cache_key(
            embedding=[0.1, 0.2, 0.3],  # Only 3 dimensions
            limit=10,
            search_mode="hybrid",
        )

        # Should still work with truncated prefix
        self.assertTrue(result.is_semantic)

    def test_none_context_type(self):
        """Test handling of None context_type."""
        result = self.generator.generate_cache_key(
            embedding=[0.1] * 384,
            limit=10,
            search_mode="hybrid",
            context_type=None,
        )

        self.assertTrue(result.is_semantic)

    def test_empty_additional_params(self):
        """Test handling of empty additional_params."""
        result = self.generator.generate_cache_key(
            embedding=[0.1] * 384,
            limit=10,
            search_mode="hybrid",
            additional_params={},
        )

        self.assertTrue(result.is_semantic)


if __name__ == "__main__":
    unittest.main(verbosity=2)
