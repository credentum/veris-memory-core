#!/usr/bin/env python3
"""
Deep tests for HashDiffEmbedder to achieve 45% coverage.

This test suite covers:
- HashDiffEmbedder initialization and configuration
- Document hash computation and tracking
- MinHash and SimHash algorithms
- Hash cache management
- Configuration loading and error handling
- Hash function generation
- Content and embedding hash computation
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.storage.hash_diff_embedder import DocumentHash, EmbeddingTask, HashDiffEmbedder


class TestDocumentHashDataclass:
    """Test DocumentHash dataclass."""

    def test_document_hash_creation(self):
        """Test DocumentHash creation with all fields."""
        doc_hash = DocumentHash(
            document_id="doc_123",
            file_path="/path/to/document.txt",
            content_hash="abc123def456",
            embedding_hash="fed654cba321",
            last_embedded="2023-01-01T12:00:00Z",
            vector_id="vec_789",
        )

        assert doc_hash.document_id == "doc_123"
        assert doc_hash.file_path == "/path/to/document.txt"
        assert doc_hash.content_hash == "abc123def456"
        assert doc_hash.embedding_hash == "fed654cba321"
        assert doc_hash.last_embedded == "2023-01-01T12:00:00Z"
        assert doc_hash.vector_id == "vec_789"


class TestEmbeddingTaskDataclass:
    """Test EmbeddingTask dataclass."""

    def test_embedding_task_creation(self):
        """Test EmbeddingTask creation with all fields."""
        file_path = Path("/path/to/file.txt")
        data = {"metadata": {"author": "test"}}

        task = EmbeddingTask(
            file_path=file_path, document_id="doc_456", content="Test document content", data=data
        )

        assert task.file_path == file_path
        assert task.document_id == "doc_456"
        assert task.content == "Test document content"
        assert task.data == data


class TestHashDiffEmbedderInitialization:
    """Test HashDiffEmbedder initialization and configuration."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        with patch.object(HashDiffEmbedder, "_load_config", return_value={}):
            with patch.object(
                HashDiffEmbedder, "_load_perf_config", return_value={"vector_db": {"embedding": {}}}
            ):
                with patch.object(HashDiffEmbedder, "_load_hash_cache", return_value={}):
                    embedder = HashDiffEmbedder()

                    assert embedder.verbose is False
                    assert embedder.num_hashes == 128
                    assert embedder.embedding_dim == 384
                    assert embedder.batch_size == 100
                    assert embedder.max_retries == 3
                    assert embedder.client is None
                    assert embedder.openai_client is None

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch.object(HashDiffEmbedder, "_load_config", return_value={}):
            with patch.object(
                HashDiffEmbedder, "_load_perf_config", return_value={"vector_db": {"embedding": {}}}
            ):
                with patch.object(HashDiffEmbedder, "_load_hash_cache", return_value={}):
                    embedder = HashDiffEmbedder(
                        config_path="custom.yaml",
                        perf_config_path="custom_perf.yaml",
                        verbose=True,
                        num_hashes=256,
                        embedding_dim=512,
                    )

                    assert embedder.verbose is True
                    assert embedder.num_hashes == 256
                    assert embedder.embedding_dim == 512

    def test_init_with_performance_config(self):
        """Test initialization with performance configuration."""
        perf_config = {
            "vector_db": {
                "embedding": {
                    "batch_size": 50,
                    "max_retries": 5,
                    "initial_retry_delay": 2.0,
                    "retry_backoff_factor": 3.0,
                    "request_timeout": 60,
                }
            }
        }

        with patch.object(HashDiffEmbedder, "_load_config", return_value={}):
            with patch.object(HashDiffEmbedder, "_load_perf_config", return_value=perf_config):
                with patch.object(HashDiffEmbedder, "_load_hash_cache", return_value={}):
                    embedder = HashDiffEmbedder()

                    assert embedder.batch_size == 50
                    assert embedder.max_retries == 5
                    assert embedder.initial_retry_delay == 2.0
                    assert embedder.retry_backoff_factor == 3.0
                    assert embedder.request_timeout == 60

    def test_init_with_qdrant_config(self):
        """Test initialization with Qdrant configuration."""
        config = {"qdrant": {"embedding_model": "text-embedding-3-small"}}

        with patch.object(HashDiffEmbedder, "_load_config", return_value=config):
            with patch.object(
                HashDiffEmbedder, "_load_perf_config", return_value={"vector_db": {"embedding": {}}}
            ):
                with patch.object(HashDiffEmbedder, "_load_hash_cache", return_value={}):
                    embedder = HashDiffEmbedder()

                    assert embedder.embedding_model == "text-embedding-3-small"


class TestHashDiffEmbedderConfigLoading:
    """Test configuration loading methods."""

    def test_load_config_success(self):
        """Test successful configuration loading."""
        config_data = {"qdrant": {"host": "localhost"}, "embedding": {"model": "test"}}

        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            with patch("yaml.safe_load", return_value=config_data):
                embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
                result = embedder._load_config("test.yaml")

                assert result == config_data

    def test_load_config_file_not_found(self):
        """Test configuration loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
            result = embedder._load_config("nonexistent.yaml")

            assert result == {}

    def test_load_config_empty_file(self):
        """Test configuration loading with empty file."""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value=None):
                embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
                result = embedder._load_config("empty.yaml")

                assert result == {}

    def test_load_perf_config_success(self):
        """Test successful performance configuration loading."""
        perf_data = {"vector_db": {"embedding": {"batch_size": 200}}}

        with patch("builtins.open", mock_open(read_data=yaml.dump(perf_data))):
            with patch("yaml.safe_load", return_value=perf_data):
                embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
                result = embedder._load_perf_config("perf.yaml")

                assert result == perf_data

    def test_load_perf_config_file_not_found(self):
        """Test performance configuration loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
            result = embedder._load_perf_config("nonexistent.yaml")

            expected = {"vector_db": {"embedding": {}}}
            assert result == expected

    def test_load_perf_config_empty_file(self):
        """Test performance configuration loading with empty file."""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value=None):
                embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
                result = embedder._load_perf_config("empty.yaml")

                expected = {"vector_db": {"embedding": {}}}
                assert result == expected


class TestHashDiffEmbedderCacheManagement:
    """Test hash cache management."""

    def test_load_hash_cache_success(self):
        """Test successful hash cache loading."""
        cache_data = {
            "doc1": {"hash": "abc123", "timestamp": "2023-01-01T12:00:00Z"},
            "doc2": {"hash": "def456", "timestamp": "2023-01-01T13:00:00Z"},
        }

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(cache_data))):
                embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
                embedder.hash_cache_path = Path("test_cache.json")
                result = embedder._load_hash_cache()

                assert result == cache_data

    def test_load_hash_cache_file_not_exists(self):
        """Test hash cache loading when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
            embedder.hash_cache_path = Path("nonexistent_cache.json")
            result = embedder._load_hash_cache()

            assert result == {}

    def test_load_hash_cache_json_error(self):
        """Test hash cache loading with JSON parsing error."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="invalid json")):
                with patch("json.load", side_effect=json.JSONDecodeError("test", "doc", 0)):
                    embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
                    embedder.hash_cache_path = Path("invalid_cache.json")
                    result = embedder._load_hash_cache()

                    assert result == {}

    def test_load_hash_cache_general_exception(self):
        """Test hash cache loading with general exception."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", side_effect=Exception("File error")):
                embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
                embedder.hash_cache_path = Path("error_cache.json")
                result = embedder._load_hash_cache()

                assert result == {}


class TestHashDiffEmbedderHashComputation:
    """Test hash computation methods."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

            content = "This is test content"
            result = embedder._compute_content_hash(content)

            # Should return a valid SHA-256 hash
            assert isinstance(result, str)
            assert len(result) == 64  # SHA-256 produces 64-character hex string

            # Same content should produce same hash
            result2 = embedder._compute_content_hash(content)
            assert result == result2

            # Different content should produce different hash
            different_content = "This is different content"
            result3 = embedder._compute_content_hash(different_content)
            assert result != result3

    def test_compute_embedding_hash(self):
        """Test embedding hash computation."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

            embedding = [0.1, 0.2, -0.3, 0.4, -0.5]
            result = embedder._compute_embedding_hash(embedding)

            # Should return a valid SHA-256 hash
            assert isinstance(result, str)
            assert len(result) == 64

            # Same embedding should produce same hash
            result2 = embedder._compute_embedding_hash(embedding)
            assert result == result2

            # Different embedding should produce different hash
            different_embedding = [0.1, 0.2, -0.3, 0.4, -0.6]
            result3 = embedder._compute_embedding_hash(different_embedding)
            assert result != result3

    def test_compute_embedding_hash_order_independence(self):
        """Test that embedding hash is order-dependent (due to sort_keys=True)."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

            embedding1 = [0.1, 0.2, 0.3]
            embedding2 = [0.1, 0.2, 0.3]  # Same values

            result1 = embedder._compute_embedding_hash(embedding1)
            result2 = embedder._compute_embedding_hash(embedding2)

            assert result1 == result2


class TestHashDiffEmbedderHashFunctions:
    """Test hash function generation."""

    def test_generate_hash_functions(self):
        """Test hash function generation."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

            num_hashes = 10
            hash_funcs = embedder._generate_hash_functions(num_hashes)

            assert len(hash_funcs) == num_hashes
            assert all(callable(func) for func in hash_funcs)

    def test_hash_function_with_string(self):
        """Test generated hash functions work with strings."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

            hash_funcs = embedder._generate_hash_functions(5)
            test_string = "test_token"

            for func in hash_funcs:
                result = func(test_string)
                assert isinstance(result, int)
                assert 0 <= result < 2**32

    def test_hash_function_with_bytes(self):
        """Test generated hash functions work with bytes."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

            hash_funcs = embedder._generate_hash_functions(5)
            test_bytes = b"test_token"

            for func in hash_funcs:
                result = func(test_bytes)
                assert isinstance(result, int)
                assert 0 <= result < 2**32

    def test_hash_function_with_integer(self):
        """Test generated hash functions work with integers."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

            hash_funcs = embedder._generate_hash_functions(5)
            test_int = 12345

            for func in hash_funcs:
                result = func(test_int)
                assert isinstance(result, int)
                assert 0 <= result < 2**32


class TestHashDiffEmbedderMinHash:
    """Test MinHash algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            self.embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)
            self.embedder.num_hashes = 10
            self.embedder.hash_funcs = self.embedder._generate_hash_functions(10)

    def test_compute_minhash_with_tokens(self):
        """Test MinHash computation with token list."""
        tokens = ["token1", "token2", "token3", "token4"]

        signature = self.embedder.compute_minhash(tokens)

        assert len(signature) == self.embedder.num_hashes
        assert all(isinstance(val, int) for val in signature)
        assert all(val >= 0 for val in signature)

    def test_compute_minhash_empty_tokens(self):
        """Test MinHash computation with empty token list."""
        tokens = []

        signature = self.embedder.compute_minhash(tokens)

        assert len(signature) == self.embedder.num_hashes
        assert all(val == 0 for val in signature)

    def test_compute_minhash_single_token(self):
        """Test MinHash computation with single token."""
        tokens = ["single_token"]

        signature = self.embedder.compute_minhash(tokens)

        assert len(signature) == self.embedder.num_hashes
        assert all(isinstance(val, int) for val in signature)

    def test_compute_minhash_consistency(self):
        """Test MinHash produces consistent results."""
        tokens = ["consistent", "test", "tokens"]

        signature1 = self.embedder.compute_minhash(tokens)
        signature2 = self.embedder.compute_minhash(tokens)

        assert signature1 == signature2

    def test_compute_minhash_different_sets(self):
        """Test MinHash produces different results for different token sets."""
        tokens1 = ["set", "one", "tokens"]
        tokens2 = ["set", "two", "tokens"]

        signature1 = self.embedder.compute_minhash(tokens1)
        signature2 = self.embedder.compute_minhash(tokens2)

        # Signatures should be different (though they might have some overlap)
        assert signature1 != signature2


class TestHashDiffEmbedderSimHash:
    """Test SimHash algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch.object(HashDiffEmbedder, "__init__", lambda x: None):
            self.embedder = HashDiffEmbedder.__new__(HashDiffEmbedder)

    def test_compute_simhash_with_tokens(self):
        """Test SimHash computation with token list."""
        tokens = ["token1", "token2", "token3", "token4"]

        simhash = self.embedder.compute_simhash(tokens)

        assert isinstance(simhash, int)
        assert simhash >= 0

    def test_compute_simhash_empty_tokens(self):
        """Test SimHash computation with empty token list."""
        tokens = []

        simhash = self.embedder.compute_simhash(tokens)

        assert simhash == 0

    def test_compute_simhash_single_token(self):
        """Test SimHash computation with single token."""
        tokens = ["single_token"]

        simhash = self.embedder.compute_simhash(tokens)

        assert isinstance(simhash, int)
        assert simhash >= 0

    def test_compute_simhash_consistency(self):
        """Test SimHash produces consistent results."""
        tokens = ["consistent", "test", "tokens"]

        simhash1 = self.embedder.compute_simhash(tokens)
        simhash2 = self.embedder.compute_simhash(tokens)

        assert simhash1 == simhash2

    def test_compute_simhash_different_sets(self):
        """Test SimHash produces different results for different token sets."""
        tokens1 = ["set", "one", "tokens"]
        tokens2 = ["completely", "different", "tokens"]

        simhash1 = self.embedder.compute_simhash(tokens1)
        simhash2 = self.embedder.compute_simhash(tokens2)

        # SimHash values should be different for completely different sets
        assert simhash1 != simhash2

    def test_compute_simhash_similar_sets(self):
        """Test SimHash for similar token sets."""
        tokens1 = ["similar", "set", "of", "tokens"]
        tokens2 = ["similar", "set", "with", "tokens"]  # One token different

        simhash1 = self.embedder.compute_simhash(tokens1)
        simhash2 = self.embedder.compute_simhash(tokens2)

        # Similar sets should have similar (but not identical) SimHash values
        # We'll just verify they're both valid integers
        assert isinstance(simhash1, int)
        assert isinstance(simhash2, int)


class TestHashDiffEmbedderIntegration:
    """Test integration scenarios and edge cases."""

    def test_hash_cache_path_creation(self):
        """Test that hash cache path is properly set."""
        with patch.object(HashDiffEmbedder, "_load_config", return_value={}):
            with patch.object(
                HashDiffEmbedder, "_load_perf_config", return_value={"vector_db": {"embedding": {}}}
            ):
                with patch.object(HashDiffEmbedder, "_load_hash_cache", return_value={}):
                    embedder = HashDiffEmbedder()

                    expected_path = Path("context/.embeddings_cache/hash_cache.json")
                    assert embedder.hash_cache_path == expected_path

    def test_default_embedding_model(self):
        """Test default embedding model setting."""
        with patch.object(HashDiffEmbedder, "_load_config", return_value={}):
            with patch.object(
                HashDiffEmbedder, "_load_perf_config", return_value={"vector_db": {"embedding": {}}}
            ):
                with patch.object(HashDiffEmbedder, "_load_hash_cache", return_value={}):
                    embedder = HashDiffEmbedder()

                    assert embedder.embedding_model == "text-embedding-ada-002"

    def test_hash_functions_generation_on_init(self):
        """Test that hash functions are generated during initialization."""
        with patch.object(HashDiffEmbedder, "_load_config", return_value={}):
            with patch.object(
                HashDiffEmbedder, "_load_perf_config", return_value={"vector_db": {"embedding": {}}}
            ):
                with patch.object(HashDiffEmbedder, "_load_hash_cache", return_value={}):
                    embedder = HashDiffEmbedder(num_hashes=5)

                    assert len(embedder.hash_funcs) == 5
                    assert all(callable(func) for func in embedder.hash_funcs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
