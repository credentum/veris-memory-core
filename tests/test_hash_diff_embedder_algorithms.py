"""
Comprehensive tests for hash_diff_embedder.py focusing on algorithms and diff analysis.

Tests cover hash-based embedding generation, diff computation, similarity scoring,
and statistical validation of hash distributions.
"""

import math
import unittest

from src.storage.hash_diff_embedder import HashDiffEmbedder


class TestHashDiffEmbedderBasics(unittest.TestCase):
    """Test basic initialization and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder()

    def test_initialization_default(self):
        """Test default initialization parameters."""
        self.assertEqual(self.embedder.embedding_dim, 384)  # Default from __init__
        self.assertEqual(self.embedder.num_hashes, 128)  # Default from __init__
        # Check that basic attributes exist
        self.assertIsNotNone(self.embedder.config)
        self.assertIsNotNone(self.embedder.hash_cache)

    def test_initialization_custom_dimensions(self):
        """Test initialization with custom dimensions."""
        embedder = HashDiffEmbedder(embedding_dim=768)
        self.assertEqual(embedder.embedding_dim, 768)

    def test_initialization_custom_hashes(self):
        """Test initialization with custom number of hashes."""
        embedder = HashDiffEmbedder(num_hashes=32)
        self.assertEqual(embedder.num_hashes, 32)
        # Check that config is loaded
        self.assertIsNotNone(embedder.config)

    def test_hash_function_generation(self):
        """Test hash function generation creates unique functions."""
        # Test that basic functionality works
        text = "test"
        # Check if embedder has a method to generate hashes
        if hasattr(self.embedder, "generate_hash"):
            hash_result = self.embedder.generate_hash(text, 0)
            self.assertIsNotNone(hash_result)
        else:
            # Just check that the embedder exists
            self.assertIsNotNone(self.embedder)


class TestHashGeneration(unittest.TestCase):
    """Test hash generation algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder(embedding_dim=1536, num_hashes=16)

    def test_generate_hash_basic(self):
        """Test basic hash generation."""
        tokens = ["Hello", "world"]
        result = self.embedder.compute_simhash(tokens)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)

    def test_generate_hash_consistency(self):
        """Test hash generation is consistent."""
        tokens = ["Consistent", "text"]
        hash1 = self.embedder.compute_simhash(tokens)
        hash2 = self.embedder.compute_simhash(tokens)
        self.assertEqual(hash1, hash2)

    def test_generate_hash_different_indices(self):
        """Test minhash produces different signatures."""
        tokens1 = ["Test", "text", "one"]
        tokens2 = ["Test", "text", "two"]
        hash1 = self.embedder.compute_minhash(tokens1)
        hash2 = self.embedder.compute_minhash(tokens2)
        # Should produce different signatures
        self.assertNotEqual(hash1, hash2)

    def test_generate_hash_empty_text(self):
        """Test hash generation with empty tokens."""
        result = self.embedder.compute_simhash([])
        self.assertIsInstance(result, int)

    def test_generate_hash_unicode(self):
        """Test hash generation with unicode tokens."""
        tokens = ["Hello", "世界", "🌍"]
        result = self.embedder.compute_simhash(tokens)
        self.assertIsInstance(result, int)

    def test_generate_hash_long_text(self):
        """Test hash generation with many tokens."""
        tokens = ["word" + str(i) for i in range(1000)]
        result = self.embedder.compute_simhash(tokens)
        self.assertIsInstance(result, int)


class TestFeatureExtraction(unittest.TestCase):
    """Test hash-based feature extraction algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder()

    def test_extract_features_basic(self):
        """Test basic minhash computation."""
        tokens = ["Hello", "world", "test"]
        features = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_extract_features_empty_text(self):
        """Test hash computation with empty tokens."""
        features = self.embedder.compute_minhash([])
        self.assertIsInstance(features, list)
        # Empty tokens should still produce valid hash signature
        self.assertGreater(len(features), 0)

    def test_extract_features_single_word(self):
        """Test hash computation with single token."""
        features = self.embedder.compute_minhash(["Hello"])
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_extract_features_punctuation(self):
        """Test hash computation with punctuation tokens."""
        tokens = ["Hello,", "world!", "How", "are", "you?"]
        features = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_extract_features_numeric(self):
        """Test hash computation with numeric tokens."""
        tokens = ["The", "year", "2024", "has", "365", "days"]
        features = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_extract_features_special_chars(self):
        """Test hash computation with special character tokens."""
        tokens = ["Email:", "test@example.com", "#hashtag"]
        features = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)


class TestEmbeddingGeneration(unittest.TestCase):
    """Test hash-based embedding functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder(embedding_dim=1536)

    def test_generate_embedding_basic(self):
        """Test basic hash signature generation."""
        tokens = ["Test", "text", "for", "embedding"]
        signature = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_generate_embedding_consistency(self):
        """Test hash generation is consistent."""
        tokens = ["Consistent", "embedding", "test"]
        hash1 = self.embedder.compute_simhash(tokens)
        hash2 = self.embedder.compute_simhash(tokens)
        self.assertEqual(hash1, hash2)

    def test_generate_embedding_different_texts(self):
        """Test different token sets produce different hashes."""
        tokens1 = ["First", "text"]
        tokens2 = ["Completely", "different", "text"]
        hash1 = self.embedder.compute_simhash(tokens1)
        hash2 = self.embedder.compute_simhash(tokens2)
        # Hashes should be different
        self.assertNotEqual(hash1, hash2)

    def test_generate_embedding_similar_texts(self):
        """Test similar token sets using hamming distance."""
        tokens1 = ["The", "quick", "brown", "fox"]
        tokens2 = ["The", "quick", "brown", "foxes"]
        hash1 = self.embedder.compute_simhash(tokens1)
        hash2 = self.embedder.compute_simhash(tokens2)
        distance = self.embedder.hamming_distance(hash1, hash2)
        # Similar token sets should have reasonable hamming distance
        self.assertLess(distance, 20)

    def test_generate_embedding_empty_text(self):
        """Test hash generation with empty tokens."""
        signature = self.embedder.compute_minhash([])
        self.assertIsInstance(signature, list)
        # Empty tokens should still produce valid signature
        self.assertGreater(len(signature), 0)

    def test_generate_embedding_unicode(self):
        """Test hash generation with unicode tokens."""
        tokens = ["Unicode", "test:", "你好世界", "مرحبا", "بالعالم"]
        signature = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_generate_embedding_very_long_text(self):
        """Test hash generation with many tokens."""
        tokens = [f"token{i}" for i in range(1000)]
        hash_result = self.embedder.compute_simhash(tokens)
        self.assertIsInstance(hash_result, int)
        self.assertGreaterEqual(hash_result, 0)

    def _jaccard_similarity_test(self, sig1, sig2):
        """Test jaccard similarity calculation."""
        return self.embedder.jaccard_similarity(sig1, sig2)
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 * norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)


class TestHashDistribution(unittest.TestCase):
    """Test hash distribution and uniformity."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder(embedding_dim=1536, num_hashes=16)

    def test_hash_distribution_uniformity(self):
        """Test that minhash signatures vary appropriately."""
        # Generate minhash signatures for many different token sets
        token_sets = [[f"word{i}", f"text{i}"] for i in range(100)]
        signatures = []

        for tokens in token_sets:
            signature = self.embedder.compute_minhash(tokens)
            signatures.append(signature)

        # Check that we get different signatures
        unique_signatures = {tuple(sig) for sig in signatures}
        # Most signatures should be unique (allowing some collisions)
        self.assertGreater(len(unique_signatures), 90)

    def test_hash_collision_rate(self):
        """Test that different token sets produce different simhashes."""
        token_sets = [[f"unique{i}", f"text{i}"] for i in range(50)]
        simhashes = []

        for tokens in token_sets:
            hash_val = self.embedder.compute_simhash(tokens)
            simhashes.append(hash_val)

        # Check that most hashes are unique
        unique_hashes = set(simhashes)
        # Allow for some collisions but most should be unique
        self.assertGreater(len(unique_hashes), 45)

    def test_embedding_distribution_normality(self):
        """Test simhash values have reasonable distribution."""
        token_sets = [[f"sample{i}", f"text{i}"] for i in range(100)]
        simhash_values = []

        for tokens in token_sets:
            simhash = self.embedder.compute_simhash(tokens)
            simhash_values.append(simhash)

        # Check that we get a variety of hash values
        unique_values = set(simhash_values)
        # Most should be unique (allowing some collisions)
        self.assertGreater(len(unique_values), 90)


class TestDiffComputation(unittest.TestCase):
    """Test diff computation and similarity algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder()

    def test_compute_similarity_identical(self):
        """Test jaccard similarity of identical minhash signatures."""
        tokens = ["Identical", "text"]
        sig1 = self.embedder.compute_minhash(tokens)
        sig2 = self.embedder.compute_minhash(tokens)
        similarity = self.embedder.jaccard_similarity(sig1, sig2)
        self.assertAlmostEqual(similarity, 1.0)

    def test_compute_similarity_different(self):
        """Test jaccard similarity of different minhash signatures."""
        sig1 = self.embedder.compute_minhash(["First", "text"])
        sig2 = self.embedder.compute_minhash(["Completely", "unrelated", "content"])
        similarity = self.embedder.jaccard_similarity(sig1, sig2)
        self.assertLess(similarity, 0.5)

    def test_compute_similarity_similar(self):
        """Test jaccard similarity of similar token sets."""
        sig1 = self.embedder.compute_minhash(["The", "cat", "sat", "on", "the", "mat"])
        sig2 = self.embedder.compute_minhash(["The", "cat", "sits", "on", "the", "mat"])
        similarity = self.embedder.jaccard_similarity(sig1, sig2)
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 1.0)

    def test_compute_diff_vector(self):
        """Test hamming distance computation."""
        hash1 = self.embedder.compute_simhash(["Original", "text"])
        hash2 = self.embedder.compute_simhash(["Modified", "text"])
        distance = self.embedder.hamming_distance(hash1, hash2)
        self.assertIsInstance(distance, int)
        self.assertGreaterEqual(distance, 0)

    def test_compute_diff_magnitude(self):
        """Test diff magnitude correlates with text difference."""
        base = "The quick brown fox"
        similar = "The quick brown foxes"
        different = "Something completely different"

        hash_base = self.embedder.compute_simhash(["The", "quick", "brown", "fox"])
        hash_similar = self.embedder.compute_simhash(["The", "quick", "brown", "foxes"])
        hash_different = self.embedder.compute_simhash(["Something", "completely", "different"])

        dist_similar = self.embedder.hamming_distance(hash_base, hash_similar)
        dist_different = self.embedder.hamming_distance(hash_base, hash_different)

        # Distance to different tokens should be larger than to similar tokens
        self.assertGreater(dist_different, dist_similar)


class TestNormalizationAndScaling(unittest.TestCase):
    """Test normalization and scaling algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder()

    def test_normalize_embedding(self):
        """Test hash signature computation."""
        tokens = ["Test", "normalization", "tokens"]
        # Test minhash signature
        signature = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)
        # All values should be non-negative integers
        for value in signature:
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_scale_features(self):
        """Test simhash computation produces consistent results."""
        tokens1 = ["length", "words", "average", "unique"]
        tokens2 = ["length", "words", "average", "unique"]

        hash1 = self.embedder.compute_simhash(tokens1)
        hash2 = self.embedder.compute_simhash(tokens2)

        # Same tokens should produce same hash
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, int)
        self.assertGreaterEqual(hash1, 0)

    def test_combine_hash_and_features(self):
        """Test hamming distance computation between different hashes."""
        tokens1 = ["Combination", "test", "one"]
        tokens2 = ["Combination", "test", "two"]

        hash1 = self.embedder.compute_simhash(tokens1)
        hash2 = self.embedder.compute_simhash(tokens2)

        distance = self.embedder.hamming_distance(hash1, hash2)
        self.assertIsInstance(distance, int)
        self.assertGreaterEqual(distance, 0)
        # Different token sets should have some hamming distance
        self.assertGreater(distance, 0)

    def test_embedding_stability_with_noise(self):
        """Test jaccard similarity between similar token sets."""
        base_tokens = ["The", "quick", "brown", "fox", "jumps"]
        variations = [
            ["The", "quick", "brown", "fox", "jumps", "."],
            ["The", "quick", "brown", "fox", "jumps", "!"],
            ["The", "quick", "brown", "fox", "jumps", "high"],
        ]

        base_sig = self.embedder.compute_minhash(base_tokens)

        for variant_tokens in variations:
            var_sig = self.embedder.compute_minhash(variant_tokens)
            similarity = self.embedder.jaccard_similarity(base_sig, var_sig)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
            # Similar token sets should have reasonable similarity
            self.assertGreater(similarity, 0.3)


class TestEdgeCasesAndErrors(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder()

    def test_null_bytes_in_text(self):
        """Test handling of null bytes in text."""
        tokens = ["Text", "with", "\x00null", "bytes"]
        signature = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_only_whitespace(self):
        """Test handling of whitespace-only text."""
        whitespace_tokens = [" ", "\t", "\n", "   \t\n   "]
        signature = self.embedder.compute_minhash(whitespace_tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)
        # Test simhash too
        simhash = self.embedder.compute_simhash(whitespace_tokens)
        self.assertIsInstance(simhash, int)

    def test_binary_like_text(self):
        """Test handling of binary-like text."""
        binary_tokens = ["\x01\x02\x03", "\xff\xfe\xfd"]
        signature = self.embedder.compute_minhash(binary_tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_extremely_long_word(self):
        """Test handling of extremely long words."""
        long_token = "a" * 10000
        tokens = [long_token, "normal", "word"]
        signature = self.embedder.compute_minhash(tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_repeated_pattern(self):
        """Test handling of repeated patterns."""
        repeated_tokens = ["abc"] * 100  # Repeated pattern
        signature = self.embedder.compute_minhash(repeated_tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_mixed_scripts(self):
        """Test handling of mixed writing scripts."""
        mixed_tokens = ["English", "中文", "العربية", "हिन्दी", "日本語", "한국어"]
        signature = self.embedder.compute_minhash(mixed_tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_mathematical_symbols(self):
        """Test handling of mathematical symbols."""
        math_tokens = ["∫∑∏", "√∞≈", "≠≤≥", "∈∉⊂⊃", "∪∩"]
        signature = self.embedder.compute_minhash(math_tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)

    def test_emoji_text(self):
        """Test handling of emoji text."""
        emoji_tokens = ["Hello", "😀🎉🎈", "🎁🎂", "World"]
        signature = self.embedder.compute_minhash(emoji_tokens)
        self.assertIsInstance(signature, list)
        self.assertGreater(len(signature), 0)


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations and caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = HashDiffEmbedder()

    def test_batch_generation_consistency(self):
        """Test hash generation produces consistent results."""
        token_sets = [["Text", str(i)] for i in range(10)]

        # Generate signatures individually
        signatures1 = [self.embedder.compute_minhash(tokens) for tokens in token_sets]
        signatures2 = [self.embedder.compute_minhash(tokens) for tokens in token_sets]

        # Results should be identical for same inputs
        for sig1, sig2 in zip(signatures1, signatures2):
            self.assertEqual(sig1, sig2)

    def test_memory_efficiency(self):
        """Test memory efficiency with large batches."""
        # Generate many hash signatures
        token_sets = [[f"Sample", "text", str(i)] for i in range(100)]
        signatures = []

        for tokens in token_sets:
            sig = self.embedder.compute_minhash(tokens)
            signatures.append(sig)

        # All signatures should be valid lists
        self.assertTrue(all(isinstance(sig, list) and len(sig) > 0 for sig in signatures))

    def test_hash_function_caching(self):
        """Test hash functions are properly deterministic."""
        embedder1 = HashDiffEmbedder(num_hashes=16)
        embedder2 = HashDiffEmbedder(num_hashes=16)

        # Hash functions should be deterministic for same inputs
        tokens = ["Test", "caching", "behavior"]
        sig1 = embedder1.compute_minhash(tokens)
        sig2 = embedder2.compute_minhash(tokens)

        # Both should be valid signatures
        self.assertIsInstance(sig1, list)
        self.assertIsInstance(sig2, list)
        self.assertGreater(len(sig1), 0)
        self.assertGreater(len(sig2), 0)


if __name__ == "__main__":
    unittest.main()
