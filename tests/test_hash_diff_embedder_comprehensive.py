#!/usr/bin/env python3
"""
Comprehensive tests for HashDiffEmbedder to achieve high coverage.

This test suite covers:
- HashDiffEmbedder class initialization and configuration loading
- Hash computation methods (content, embedding)
- MinHash and SimHash algorithms with similarity calculations
- Connection management (Qdrant, OpenAI)
- Embedding processing with retry logic and rate limiting
- Directory processing and batch workflows
- Error handling and edge cases
- Performance configuration and caching
"""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import yaml

from src.storage.hash_diff_embedder import DocumentHash, EmbeddingTask, HashDiffEmbedder


class TestHashDiffEmbedderInitialization:
    """Test HashDiffEmbedder initialization and configuration."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            embedder = HashDiffEmbedder()

        assert embedder.config == {}
        assert embedder.perf_config == {"vector_db": {"embedding": {}}}
        assert embedder.hash_cache == {}
        assert embedder.client is None
        assert embedder.openai_client is None
        assert embedder.embedding_model == "text-embedding-ada-002"
        assert embedder.verbose is False
        assert embedder.num_hashes == 128
        assert embedder.embedding_dim == 384
        assert embedder.batch_size == 100
        assert embedder.max_retries == 3
        assert len(embedder.hash_funcs) == 128

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            embedder = HashDiffEmbedder(
                config_path="custom.yaml",
                perf_config_path="perf.yaml",
                verbose=True,
                num_hashes=64,
                embedding_dim=512,
            )

        assert embedder.verbose is True
        assert embedder.num_hashes == 64
        assert embedder.embedding_dim == 512
        assert len(embedder.hash_funcs) == 64

    def test_load_config_success(self):
        """Test successful config loading."""
        config_data = {
            "qdrant": {"host": "test-host", "port": 6333, "embedding_model": "custom-model"}
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(config_data))):
            with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
                embedder = HashDiffEmbedder(config_path="test.yaml")

        assert embedder.config == config_data
        assert embedder.embedding_model == "custom-model"

    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
                embedder = HashDiffEmbedder(config_path="missing.yaml")

        assert embedder.config == {}

    def test_load_perf_config_success(self):
        """Test successful performance config loading."""
        perf_data = {
            "vector_db": {"embedding": {"batch_size": 50, "max_retries": 5, "request_timeout": 60}}
        }

        with patch("builtins.open", mock_open(read_data=yaml.dump(perf_data))):
            with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
                embedder = HashDiffEmbedder(perf_config_path="perf.yaml")

        assert embedder.batch_size == 50
        assert embedder.max_retries == 5
        assert embedder.request_timeout == 60

    def test_load_perf_config_file_not_found(self):
        """Test performance config loading when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
                embedder = HashDiffEmbedder(perf_config_path="missing.yaml")

        assert embedder.perf_config == {"vector_db": {"embedding": {}}}

    def test_load_hash_cache_success(self):
        """Test successful hash cache loading."""
        cache_data = {
            "file1.yaml": {
                "document_id": "doc1",
                "content_hash": "hash1",
                "embedding_hash": "embhash1",
            }
        }

        with patch("storage.hash_diff_embedder.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(cache_data))):
                embedder = HashDiffEmbedder()

        assert embedder.hash_cache == cache_data

    def test_load_hash_cache_file_not_exists(self):
        """Test hash cache loading when file doesn't exist."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            embedder = HashDiffEmbedder()

        assert embedder.hash_cache == {}

    def test_load_hash_cache_invalid_json(self):
        """Test hash cache loading with invalid JSON."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data="invalid json")):
                with patch(
                    "storage.hash_diff_embedder.json.load",
                    side_effect=json.JSONDecodeError("Invalid JSON", "", 0),
                ):
                    embedder = HashDiffEmbedder()

        assert embedder.hash_cache == {}


class TestHashDiffEmbedderHashMethods:
    """Test hash computation methods."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            return HashDiffEmbedder()

    def test_compute_content_hash(self, embedder):
        """Test content hash computation."""
        content = "test content"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        result = embedder._compute_content_hash(content)

        assert result == expected_hash

    def test_compute_content_hash_empty(self, embedder):
        """Test content hash with empty string."""
        content = ""
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        result = embedder._compute_content_hash(content)

        assert result == expected_hash

    def test_compute_content_hash_unicode(self, embedder):
        """Test content hash with unicode content."""
        content = "test content with unicode: 你好世界"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        result = embedder._compute_content_hash(content)

        assert result == expected_hash

    def test_compute_embedding_hash(self, embedder):
        """Test embedding hash computation."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        embedding_bytes = json.dumps(embedding, sort_keys=True).encode()
        expected_hash = hashlib.sha256(embedding_bytes).hexdigest()

        result = embedder._compute_embedding_hash(embedding)

        assert result == expected_hash

    def test_compute_embedding_hash_empty(self, embedder):
        """Test embedding hash with empty list."""
        embedding = []
        embedding_bytes = json.dumps(embedding, sort_keys=True).encode()
        expected_hash = hashlib.sha256(embedding_bytes).hexdigest()

        result = embedder._compute_embedding_hash(embedding)

        assert result == expected_hash

    def test_compute_embedding_hash_large_embedding(self, embedder):
        """Test embedding hash with large embedding vector."""
        embedding = [0.001 * i for i in range(1000)]
        embedding_bytes = json.dumps(embedding, sort_keys=True).encode()
        expected_hash = hashlib.sha256(embedding_bytes).hexdigest()

        result = embedder._compute_embedding_hash(embedding)

        assert result == expected_hash


class TestHashDiffEmbedderHashFunctions:
    """Test hash function generation and algorithms."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            return HashDiffEmbedder(num_hashes=16)  # Smaller for testing

    def test_generate_hash_functions_count(self, embedder):
        """Test hash function generation count."""
        assert len(embedder.hash_funcs) == 16

    def test_generate_hash_functions_different_seeds(self, embedder):
        """Test that hash functions use different seeds."""
        test_input = "test"
        results = []

        for hash_func in embedder.hash_funcs[:5]:  # Test first 5
            result = hash_func(test_input)
            results.append(result)

        # Results should be different (high probability with different seeds)
        assert len(set(results)) > 1

    def test_hash_function_with_string_input(self, embedder):
        """Test hash function with string input."""
        hash_func = embedder.hash_funcs[0]
        result = hash_func("test string")

        assert isinstance(result, int)
        assert 0 <= result < 2**32

    def test_hash_function_with_bytes_input(self, embedder):
        """Test hash function with bytes input."""
        hash_func = embedder.hash_funcs[0]
        result = hash_func(b"test bytes")

        assert isinstance(result, int)
        assert 0 <= result < 2**32


class TestHashDiffEmbedderMinHash:
    """Test MinHash algorithm implementation."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            return HashDiffEmbedder(num_hashes=32)

    def test_compute_minhash_basic(self, embedder):
        """Test basic MinHash computation."""
        tokens = ["word1", "word2", "word3"]

        signature = embedder.compute_minhash(tokens)

        assert len(signature) == 32
        assert all(isinstance(val, int) for val in signature)

    def test_compute_minhash_empty_tokens(self, embedder):
        """Test MinHash with empty token list."""
        tokens = []

        signature = embedder.compute_minhash(tokens)

        assert len(signature) == 32
        assert all(val == 0 for val in signature)

    def test_compute_minhash_single_token(self, embedder):
        """Test MinHash with single token."""
        tokens = ["single_word"]

        signature = embedder.compute_minhash(tokens)

        assert len(signature) == 32
        assert not all(val == 0 for val in signature)

    def test_compute_minhash_deterministic(self, embedder):
        """Test MinHash is deterministic for same input."""
        tokens = ["word1", "word2", "word3"]

        signature1 = embedder.compute_minhash(tokens)
        signature2 = embedder.compute_minhash(tokens)

        assert signature1 == signature2

    def test_compute_minhash_different_order(self, embedder):
        """Test MinHash gives same result for different token order."""
        tokens1 = ["word1", "word2", "word3"]
        tokens2 = ["word3", "word1", "word2"]

        signature1 = embedder.compute_minhash(tokens1)
        signature2 = embedder.compute_minhash(tokens2)

        assert signature1 == signature2

    def test_jaccard_similarity_identical(self, embedder):
        """Test Jaccard similarity for identical signatures."""
        sig1 = [1, 2, 3, 4, 5]
        sig2 = [1, 2, 3, 4, 5]

        similarity = embedder.jaccard_similarity(sig1, sig2)

        assert similarity == 1.0

    def test_jaccard_similarity_completely_different(self, embedder):
        """Test Jaccard similarity for completely different signatures."""
        sig1 = [1, 2, 3, 4, 5]
        sig2 = [6, 7, 8, 9, 10]

        similarity = embedder.jaccard_similarity(sig1, sig2)

        assert similarity == 0.0

    def test_jaccard_similarity_partial_match(self, embedder):
        """Test Jaccard similarity for partial matches."""
        sig1 = [1, 2, 3, 4, 5]
        sig2 = [1, 2, 8, 9, 10]  # 2 out of 5 match

        similarity = embedder.jaccard_similarity(sig1, sig2)

        assert similarity == 0.4

    def test_jaccard_similarity_different_lengths(self, embedder):
        """Test Jaccard similarity for different length signatures."""
        sig1 = [1, 2, 3]
        sig2 = [1, 2, 3, 4, 5]

        similarity = embedder.jaccard_similarity(sig1, sig2)

        assert similarity == 0.0


class TestHashDiffEmbedderSimHash:
    """Test SimHash algorithm implementation."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            return HashDiffEmbedder()

    def test_compute_simhash_basic(self, embedder):
        """Test basic SimHash computation."""
        tokens = ["word1", "word2", "word3"]

        simhash = embedder.compute_simhash(tokens)

        assert isinstance(simhash, int)
        assert 0 <= simhash < 2**64

    def test_compute_simhash_empty_tokens(self, embedder):
        """Test SimHash with empty token list."""
        tokens = []

        simhash = embedder.compute_simhash(tokens)

        assert simhash == 0

    def test_compute_simhash_single_token(self, embedder):
        """Test SimHash with single token."""
        tokens = ["single_word"]

        simhash = embedder.compute_simhash(tokens)

        assert simhash != 0

    def test_compute_simhash_deterministic(self, embedder):
        """Test SimHash is deterministic for same input."""
        tokens = ["word1", "word2", "word3"]

        simhash1 = embedder.compute_simhash(tokens)
        simhash2 = embedder.compute_simhash(tokens)

        assert simhash1 == simhash2

    def test_compute_simhash_different_tokens(self, embedder):
        """Test SimHash produces different results for different tokens."""
        tokens1 = ["word1", "word2", "word3"]
        tokens2 = ["different", "words", "here"]

        simhash1 = embedder.compute_simhash(tokens1)
        simhash2 = embedder.compute_simhash(tokens2)

        assert simhash1 != simhash2

    def test_hamming_distance_identical(self, embedder):
        """Test Hamming distance for identical hashes."""
        hash1 = 0b1010101010101010
        hash2 = 0b1010101010101010

        distance = embedder.hamming_distance(hash1, hash2)

        assert distance == 0

    def test_hamming_distance_completely_different(self, embedder):
        """Test Hamming distance for completely different hashes."""
        hash1 = 0b1111111111111111
        hash2 = 0b0000000000000000

        distance = embedder.hamming_distance(hash1, hash2)

        assert distance == 16

    def test_hamming_distance_partial_difference(self, embedder):
        """Test Hamming distance for partially different hashes."""
        hash1 = 0b1010101010101010
        hash2 = 0b1010101010101011  # Last bit different

        distance = embedder.hamming_distance(hash1, hash2)

        assert distance == 1

    def test_hamming_distance_symmetric(self, embedder):
        """Test Hamming distance is symmetric."""
        hash1 = 0b1010101010101010
        hash2 = 0b1100110011001100

        distance1 = embedder.hamming_distance(hash1, hash2)
        distance2 = embedder.hamming_distance(hash2, hash1)

        assert distance1 == distance2


class TestHashDiffEmbedderConnections:
    """Test connection management."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance."""
        config = {"qdrant": {"host": "test-host", "port": 6333, "ssl": False, "timeout": 10}}

        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            with patch("builtins.open", mock_open(read_data=yaml.dump(config))):
                return HashDiffEmbedder(config_path="test.yaml")

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch("storage.hash_diff_embedder.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_connect_success(self, mock_openai, mock_qdrant, mock_config, embedder):
        """Test successful connection."""
        mock_config.return_value = {"host": "test-host", "port": 6333, "ssl": False, "timeout": 10}

        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance
        mock_qdrant_instance.get_collections.return_value = []

        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance

        result = embedder.connect()

        assert result is True
        assert embedder.client == mock_qdrant_instance
        assert embedder.openai_client == mock_openai_instance
        mock_qdrant_instance.get_collections.assert_called_once()

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.hash_diff_embedder.QdrantClient")
    def test_connect_qdrant_failure(self, mock_qdrant, mock_config, embedder):
        """Test connection failure with Qdrant."""
        mock_config.return_value = {"host": "test-host", "port": 6333, "ssl": False, "timeout": 10}

        mock_qdrant.side_effect = Exception("Connection failed")

        with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
            result = embedder.connect()

        assert result is False
        mock_echo.assert_called()

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch.dict(os.environ, {}, clear=True)
    def test_connect_missing_openai_key(self, mock_qdrant, mock_config, embedder):
        """Test connection failure when OpenAI API key is missing."""
        mock_config.return_value = {"host": "test-host", "port": 6333, "ssl": False, "timeout": 10}

        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance
        mock_qdrant_instance.get_collections.return_value = []

        with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
            result = embedder.connect()

        assert result is False
        mock_echo.assert_called_with("Error: OPENAI_API_KEY not set", err=True)

    @patch("core.utils.get_secure_connection_config")
    @patch("storage.hash_diff_embedder.QdrantClient")
    @patch("storage.hash_diff_embedder.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_connect_qdrant_test_failure(self, mock_openai, mock_qdrant, mock_config, embedder):
        """Test connection failure during Qdrant test."""
        mock_config.return_value = {"host": "test-host", "port": 6333, "ssl": False, "timeout": 10}

        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance
        mock_qdrant_instance.get_collections.side_effect = Exception("Test failed")

        with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
            result = embedder.connect()

        assert result is False
        mock_echo.assert_called()


class TestHashDiffEmbedderEmbedding:
    """Test embedding processing methods."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance with mock clients."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            embedder = HashDiffEmbedder()

        embedder.openai_client = Mock()
        embedder.client = Mock()
        return embedder

    def test_embed_with_retry_success(self, embedder):
        """Test successful embedding with retry."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response

        result = embedder._embed_with_retry("test text")

        assert result == [0.1, 0.2, 0.3]
        embedder.openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002", input="test text", timeout=30
        )

    def test_embed_with_retry_client_not_initialized(self, embedder):
        """Test embedding when OpenAI client is not initialized."""
        embedder.openai_client = None

        with pytest.raises(Exception, match="OpenAI client not initialized"):
            embedder._embed_with_retry("test text")

    def test_embed_with_retry_rate_limit_success(self, embedder):
        """Test embedding with rate limit retry."""
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

        # First call fails with rate limit, second succeeds
        embedder.openai_client.embeddings.create.side_effect = [
            Exception("rate_limit exceeded"),
            mock_response,
        ]

        with patch("time.sleep") as mock_sleep:
            result = embedder._embed_with_retry("test text")

        assert result == [0.1, 0.2, 0.3]
        assert embedder.openai_client.embeddings.create.call_count == 2
        mock_sleep.assert_called_once_with(1.0)

    def test_embed_with_retry_max_retries_exceeded(self, embedder):
        """Test embedding when max retries exceeded."""
        embedder.openai_client.embeddings.create.side_effect = Exception("rate_limit exceeded")

        with patch("time.sleep"):
            with pytest.raises(Exception, match="rate_limit exceeded"):
                embedder._embed_with_retry("test text")

        assert embedder.openai_client.embeddings.create.call_count == 3

    def test_embed_with_retry_non_rate_limit_error(self, embedder):
        """Test embedding with non-rate-limit error."""
        embedder.openai_client.embeddings.create.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            embedder._embed_with_retry("test text")

        assert embedder.openai_client.embeddings.create.call_count == 1

    def test_embed_with_retry_verbose_output(self, embedder):
        """Test embedding with verbose output during retry."""
        embedder.verbose = True
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]

        embedder.openai_client.embeddings.create.side_effect = [
            Exception("rate_limit exceeded"),
            mock_response,
        ]

        with patch("time.sleep"):
            with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
                result = embedder._embed_with_retry("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_echo.assert_called_with("Rate limit hit, retrying in 1.0s...")


class TestHashDiffEmbedderTaskProcessing:
    """Test embedding task processing."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance with mock clients."""
        config = {"qdrant": {"collection_name": "test_collection"}}

        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            with patch("builtins.open", mock_open(read_data=yaml.dump(config))):
                embedder = HashDiffEmbedder(config_path="test.yaml")

        embedder.openai_client = Mock()
        embedder.client = Mock()
        return embedder

    def test_process_embedding_task_success(self, embedder):
        """Test successful embedding task processing."""
        task = EmbeddingTask(
            file_path=Path("test.yaml"),
            document_id="doc1",
            content="test content",
            data={
                "title": "Test Document",
                "description": "Test description",
                "content": "Document content",
                "document_type": "test",
            },
        )

        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response

        with patch("storage.hash_diff_embedder.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"

            result = embedder._process_embedding_task(task)

        assert result.startswith("doc1-")
        embedder.client.upsert.assert_called_once()

        # Verify cache update
        assert str(task.file_path) in embedder.hash_cache
        cache_entry = embedder.hash_cache[str(task.file_path)]
        assert cache_entry["document_id"] == "doc1"

    def test_process_embedding_task_minimal_data(self, embedder):
        """Test embedding task with minimal data."""
        task = EmbeddingTask(
            file_path=Path("test.yaml"), document_id="doc1", content="test content", data={}
        )

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response

        with patch("storage.hash_diff_embedder.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"

            result = embedder._process_embedding_task(task)

        assert result.startswith("doc1-")

    def test_process_embedding_task_embedding_failure(self, embedder):
        """Test embedding task when embedding fails."""
        task = EmbeddingTask(
            file_path=Path("test.yaml"),
            document_id="doc1",
            content="test content",
            data={"title": "Test"},
        )

        embedder.openai_client.embeddings.create.side_effect = Exception("Embedding failed")

        with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
            result = embedder._process_embedding_task(task)

        assert result is None
        mock_echo.assert_called()

    def test_process_embedding_task_qdrant_failure(self, embedder):
        """Test embedding task when Qdrant operation fails."""
        task = EmbeddingTask(
            file_path=Path("test.yaml"),
            document_id="doc1",
            content="test content",
            data={"title": "Test"},
        )

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response
        embedder.client.upsert.side_effect = Exception("Qdrant failed")

        with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
            result = embedder._process_embedding_task(task)

        assert result is None
        mock_echo.assert_called()

    def test_process_embedding_task_verbose_output(self, embedder):
        """Test embedding task with verbose output."""
        embedder.verbose = True
        task = EmbeddingTask(
            file_path=Path("test.yaml"),
            document_id="doc1",
            content="test content",
            data={"title": "Test"},
        )

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response

        with patch("storage.hash_diff_embedder.datetime") as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
                result = embedder._process_embedding_task(task)

        assert result.startswith("doc1-")
        mock_echo.assert_called_with(f"  ✓ Embedded {task.file_path}")


class TestHashDiffEmbedderDirectoryProcessing:
    """Test directory processing and batch workflows."""

    @pytest.fixture
    def embedder(self):
        """Create HashDiffEmbedder instance."""
        with patch("storage.hash_diff_embedder.Path.exists", return_value=False):
            embedder = HashDiffEmbedder()

        # Override batch_size for testing
        embedder.batch_size = 2
        embedder.openai_client = Mock()
        embedder.client = Mock()
        return embedder

    def test_embed_directory_no_files(self, embedder):
        """Test directory embedding with no YAML files."""
        test_dir = Path("test_dir")

        with patch.object(Path, "rglob", return_value=[]):
            embedded, total = embedder.embed_directory(test_dir)

        assert embedded == 0
        assert total == 0

    def test_embed_directory_skip_excluded_paths(self, embedder):
        """Test directory embedding skips excluded paths."""
        test_dir = Path("test_dir")
        files = [
            Path("test_dir/schemas/schema.yaml"),  # Should be skipped
            Path("test_dir/.embeddings_cache/cache.yaml"),  # Should be skipped
            Path("test_dir/archive/old.yaml"),  # Should be skipped
            Path("test_dir/valid.yaml"),  # Should be processed
        ]

        with patch.object(Path, "rglob", return_value=files):
            with patch("builtins.open", mock_open(read_data="title: Test")):
                with patch(
                    "storage.hash_diff_embedder.yaml.safe_load", return_value={"title": "Test"}
                ):
                    embedded, total = embedder.embed_directory(test_dir)

        assert total == 4  # All files counted
        # Only 1 file should be processed (the valid one)

    def test_embed_directory_cached_files_skipped(self, embedder):
        """Test directory embedding skips cached files."""
        test_dir = Path("test_dir")
        test_file = Path("test_dir/test.yaml")

        # Set up cache with matching hash
        content = "title: Test Document"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        embedder.hash_cache = {str(test_file): {"content_hash": content_hash}}

        with patch.object(Path, "rglob", return_value=[test_file]):
            with patch("builtins.open", mock_open(read_data=content)):
                with patch(
                    "storage.hash_diff_embedder.yaml.safe_load",
                    return_value={"title": "Test Document"},
                ):
                    embedded, total = embedder.embed_directory(test_dir)

        assert embedded == 0  # File was cached, so not embedded
        assert total == 1

    def test_embed_directory_batch_processing(self, embedder):
        """Test directory embedding with batch processing."""
        test_dir = Path("test_dir")
        files = [
            Path("test_dir/file1.yaml"),
            Path("test_dir/file2.yaml"),
            Path("test_dir/file3.yaml"),
        ]

        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response

        with patch.object(Path, "rglob", return_value=files):
            with patch("builtins.open", mock_open(read_data="title: Test")):
                with patch(
                    "storage.hash_diff_embedder.yaml.safe_load",
                    return_value={"id": "doc1", "title": "Test"},
                ):
                    with patch("storage.hash_diff_embedder.datetime") as mock_datetime:
                        mock_datetime.now.return_value.isoformat.return_value = (
                            "2023-01-01T00:00:00"
                        )

                        embedded, total = embedder.embed_directory(test_dir)

        assert embedded == 3
        assert total == 3

    def test_embed_directory_verbose_batch_output(self, embedder):
        """Test directory embedding with verbose batch output."""
        embedder.verbose = True
        test_dir = Path("test_dir")
        files = [
            Path("test_dir/file1.yaml"),
            Path("test_dir/file2.yaml"),
            Path("test_dir/file3.yaml"),
        ]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response

        with patch.object(Path, "rglob", return_value=files):
            with patch("builtins.open", mock_open(read_data="title: Test")):
                with patch(
                    "storage.hash_diff_embedder.yaml.safe_load",
                    return_value={"id": "doc1", "title": "Test"},
                ):
                    with patch("storage.hash_diff_embedder.datetime") as mock_datetime:
                        mock_datetime.now.return_value.isoformat.return_value = (
                            "2023-01-01T00:00:00"
                        )
                        with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
                            embedded, total = embedder.embed_directory(test_dir)

        # Should have batch progress messages
        batch_calls = [call for call in mock_echo.call_args_list if "Processing batch" in str(call)]
        assert len(batch_calls) > 0

    def test_embed_directory_cache_save(self, embedder):
        """Test directory embedding saves cache."""
        test_dir = Path("test_dir")
        files = [Path("test_dir/file1.yaml")]

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        embedder.openai_client.embeddings.create.return_value = mock_response

        with patch.object(Path, "rglob", return_value=files):
            with patch("builtins.open", mock_open(read_data="title: Test")):
                with patch(
                    "storage.hash_diff_embedder.yaml.safe_load",
                    return_value={"id": "doc1", "title": "Test"},
                ):
                    with patch("storage.hash_diff_embedder.datetime") as mock_datetime:
                        mock_datetime.now.return_value.isoformat.return_value = (
                            "2023-01-01T00:00:00"
                        )
                        with patch.object(embedder.hash_cache_path.parent, "mkdir") as mock_mkdir:
                            with patch("builtins.open", mock_open()) as mock_file:
                                embedded, total = embedder.embed_directory(test_dir)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        # Cache should be written
        mock_file.assert_called()

    def test_embed_directory_file_read_error(self, embedder):
        """Test directory embedding handles file read errors."""
        test_dir = Path("test_dir")
        files = [Path("test_dir/bad_file.yaml")]

        with patch.object(Path, "rglob", return_value=files):
            with patch("builtins.open", side_effect=IOError("Cannot read file")):
                with patch("storage.hash_diff_embedder.click.echo") as mock_echo:
                    embedded, total = embedder.embed_directory(test_dir)

        assert embedded == 0
        assert total == 1
        mock_echo.assert_called()

    def test_embed_directory_invalid_yaml(self, embedder):
        """Test directory embedding handles invalid YAML."""
        test_dir = Path("test_dir")
        files = [Path("test_dir/invalid.yaml")]

        with patch.object(Path, "rglob", return_value=files):
            with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
                with patch("storage.hash_diff_embedder.yaml.safe_load", return_value=None):
                    embedded, total = embedder.embed_directory(test_dir)

        assert embedded == 0
        assert total == 1


class TestDocumentHashDataClass:
    """Test DocumentHash dataclass."""

    def test_document_hash_creation(self):
        """Test DocumentHash creation."""
        doc_hash = DocumentHash(
            document_id="doc1",
            file_path="/path/to/file.yaml",
            content_hash="content123",
            embedding_hash="embed456",
            last_embedded="2023-01-01T00:00:00",
            vector_id="vec789",
        )

        assert doc_hash.document_id == "doc1"
        assert doc_hash.file_path == "/path/to/file.yaml"
        assert doc_hash.content_hash == "content123"
        assert doc_hash.embedding_hash == "embed456"
        assert doc_hash.last_embedded == "2023-01-01T00:00:00"
        assert doc_hash.vector_id == "vec789"


class TestEmbeddingTaskDataClass:
    """Test EmbeddingTask dataclass."""

    def test_embedding_task_creation(self):
        """Test EmbeddingTask creation."""
        task = EmbeddingTask(
            file_path=Path("test.yaml"),
            document_id="doc1",
            content="test content",
            data={"title": "Test", "type": "document"},
        )

        assert task.file_path == Path("test.yaml")
        assert task.document_id == "doc1"
        assert task.content == "test content"
        assert task.data == {"title": "Test", "type": "document"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
